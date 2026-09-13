"""
forecast.py — Recurrence detection and 90-day cash-flow projection.

Strategy:
- Include all confirmed future events (scheduled/pending debits) directly
- Detect recurrence from historical settled events
- Project salary forward using the scheduled salary as an anchor (monthly)
- Use P75 for variable categories (conservative estimate)
- Apply message-derived amendments (salary changes, new expenses, etc.)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
from collections import defaultdict
from .reconcile import LedgerEntry
from .evidence import MessageFact


# ── Recurrence category classification ────────────────────────────────────────
# These categories are ALWAYS considered recurring if seen >= 2 times
ALWAYS_RECURRING = {
    "rent", "housing", "utilities", "salary", "debt_repayment", "insurance",
    "education", "music_subscription", "streaming", "cloud_storage",
    "delivery_membership", "gym", "healthcare", "family_support"
}

VARIABLE_RECURRING = {
    "groceries", "transport", "shopping", "entertainment"
}

ONE_OFF_CATEGORIES = {
    "investment", "windfall", "travel", "dining", "work_expense", "other"
}

MIN_OCCURRENCES_RECURRING = 2

# Typical cadences in days — used as sanity check
MONTHLY_CADENCE = 30
WEEKLY_CADENCE = 7


@dataclass
class ForecastEntry:
    date: pd.Timestamp
    direction: str          # debit / credit
    amount: float
    category: str
    flexibility: str
    minimum_allowed_amount: Optional[float]
    event_id: Optional[str]    # None for projected entries
    is_projected: bool
    event_type: str


def _detect_cadence(dates: list[pd.Timestamp]) -> int:
    """Infer recurrence cadence in days from a sorted list of dates."""
    if len(dates) < 2:
        return MONTHLY_CADENCE

    intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
    med = float(np.median(intervals))
    rounded = int(round(med))
    for target in [5, 7, 10, 14, 21, 28, 30]:
        if abs(rounded - target) <= 1:
            return target
    if 25 <= med <= 35:
        return MONTHLY_CADENCE
    return max(1, rounded)


def build_forecast(
    entries: list[LedgerEntry],
    message_facts: list[MessageFact],
    profile: pd.Series,
    request_date: pd.Timestamp,
    horizon_days: int = 90,
) -> list[ForecastEntry]:
    """
    Build a list of ForecastEntry for the window (request_date, request_date + horizon_days].
    Includes:
      - Confirmed future scheduled/pending-debit events on their stated dates
      - Projected recurring income and expenses from historical patterns
      - Salary projected forward monthly from the confirmed next salary
    """
    end_date = request_date + pd.Timedelta(days=horizon_days)
    forecast: list[ForecastEntry] = []

    # ── Part 1: confirmed future events ──────────────────────────────────────
    confirmed_event_ids: set[str] = set()

    for e in entries:
        # Include confirmed events FROM request_date onward (>= not strictly >)
        # Scheduled events on today's date (e.g., education payment today) must
        # be counted as outflows in the balance simulation
        if e.date >= request_date and e.status == "scheduled":
            forecast.append(ForecastEntry(
                date=e.date,
                direction=e.direction,
                amount=e.amount,
                category=e.category,
                flexibility=e.flexibility,
                minimum_allowed_amount=e.minimum_allowed_amount,
                event_id=e.event_id,
                is_projected=False,
                event_type=e.event_type,
            ))
            confirmed_event_ids.add(e.event_id)
        elif e.date >= request_date and e.status == "pending" and e.direction == "debit":
            # Confirmed pending outflows
            forecast.append(ForecastEntry(
                date=e.date,
                direction=e.direction,
                amount=e.amount,
                category=e.category,
                flexibility=e.flexibility,
                minimum_allowed_amount=e.minimum_allowed_amount,
                event_id=e.event_id,
                is_projected=False,
                event_type=e.event_type,
            ))
            confirmed_event_ids.add(e.event_id)

    # ── Part 2: detect recurrence from historical settled events ──────────────
    historical = [e for e in entries if e.date <= request_date and e.status == "settled"]

    # Group by (category, direction, flexibility)
    by_cat: dict[tuple, list[LedgerEntry]] = defaultdict(list)
    for e in historical:
        by_cat[(e.category, e.direction, e.flexibility)].append(e)

    # Find scheduled salary to use as anchor for salary projection
    scheduled_salaries = [e for e in entries if e.category == "salary"
                          and e.direction == "credit" and e.status == "scheduled"]

    for (cat, direction, flex), group in by_cat.items():
        if cat in ONE_OFF_CATEGORIES:
            continue
        if len(group) < MIN_OCCURRENCES_RECURRING:
            continue

        sorted_group = sorted(group, key=lambda e: e.date)
        dates = [e.date for e in sorted_group]
        cadence = _detect_cadence(dates)

        # Amount estimation
        amounts = [e.amount for e in sorted_group]
        if cat in VARIABLE_RECURRING:
            recent = amounts[-6:] if len(amounts) >= 6 else amounts
            proj_amount = float(np.median(recent))
        else:
            # Fixed recurring: use most recent amount
            proj_amount = sorted_group[-1].amount

        last_flex = sorted_group[-1].flexibility
        last_min_allowed = sorted_group[-1].minimum_allowed_amount
        last_event_type = sorted_group[-1].event_type

        # For salary, use the scheduled next salary amount if available
        if cat == "salary" and direction == "credit":
            if scheduled_salaries:
                # Use the confirmed next salary as anchor
                next_sal = sorted(scheduled_salaries, key=lambda x: x.date)[0]
                proj_amount = next_sal.amount
                # Project from the scheduled salary forward
                last_date = next_sal.date
                step = 1
                while True:
                    proj_date = last_date + pd.DateOffset(months=step)
                    if proj_date > end_date:
                        break
                    forecast.append(ForecastEntry(
                        date=proj_date,
                        direction="credit",
                        amount=proj_amount,
                        category="salary",
                        flexibility="fixed",
                        minimum_allowed_amount=None,
                        event_id=None,
                        is_projected=True,
                        event_type="income",
                    ))
                    step += 1
                continue  # Skip default projection loop below
            else:
                # No confirmed future salary; check if last historical was a final payroll
                last_desc = (getattr(sorted_group[-1], "description", "") or "").lower()
                if any(w in last_desc for w in ["final", "severance", "terminated", "ended"]):
                    continue  # Employment ended, no future salary to project
                proj_amount = sorted_group[-1].amount

        # Default: project from last occurrence
        last_date = sorted_group[-1].date
        source_event_id = sorted_group[-1].event_id
        step = 1
        is_monthly = cadence in (28, 29, 30, 31)

        while True:
            if is_monthly:
                proj_date = last_date + pd.DateOffset(months=step)
            else:
                proj_date = last_date + pd.Timedelta(days=cadence * step)
            if proj_date > end_date:
                break
            if proj_date >= request_date:
                already_settled_today = any(
                    e.category == cat and e.date == request_date and e.status == "settled"
                    for e in entries
                )
                if proj_date > request_date or not already_settled_today:
                    forecast.append(ForecastEntry(
                        date=proj_date,
                        direction=direction,
                        amount=proj_amount,
                        category=cat,
                        flexibility=last_flex,
                        minimum_allowed_amount=last_min_allowed,
                        event_id=source_event_id,
                        is_projected=True,
                        event_type=last_event_type,
                    ))
            step += 1

    # ── Part 3: handle users with ONE salary in history + one scheduled ────────
    # (new employees). If salary has only 1 historical occurrence (or none), we
    # still project from the scheduled next salary.
    sal_key = ("salary", "credit", "fixed")
    sal_hist_count = len(by_cat.get(sal_key, []))
    if sal_hist_count < MIN_OCCURRENCES_RECURRING and scheduled_salaries:
        # No historical salary to detect recurrence from; project from scheduled
        next_sal = sorted(scheduled_salaries, key=lambda x: x.date)[0]
        proj_amount = next_sal.amount
        step = 1
        while True:
            proj_date = next_sal.date + pd.DateOffset(months=step)
            if proj_date > end_date:
                break
            forecast.append(ForecastEntry(
                date=proj_date,
                direction="credit",
                amount=proj_amount,
                category="salary",
                flexibility="fixed",
                minimum_allowed_amount=None,
                event_id=None,
                is_projected=True,
                event_type="income",
            ))
            step += 1

    # ── Part 4: apply message-derived amendments ───────────────────────────────
    salary_seasonal_end = False
    salary_change_amount: Optional[float] = None
    salary_change_date: Optional[pd.Timestamp] = None
    new_salary_date_override: Optional[pd.Timestamp] = None

    for mf in message_facts:
        if mf.fact_type == "salary_seasonal_end" and mf.confidence >= 0.7:
            salary_seasonal_end = True
        elif mf.fact_type == "salary_change" and mf.confidence >= 0.7:
            if mf.new_amount is not None:
                new_date = pd.Timestamp(mf.effective_date) if mf.effective_date else request_date
                if salary_change_amount is None or (
                    salary_change_date and new_date > salary_change_date
                ):
                    salary_change_amount = mf.new_amount
                    salary_change_date = new_date
        elif mf.fact_type == "salary_date_change" and mf.confidence >= 0.7:
            if mf.effective_date:
                new_salary_date_override = pd.Timestamp(mf.effective_date)

    if salary_seasonal_end:
        forecast = [fe for fe in forecast
                    if not (fe.is_projected and fe.category == "salary")]

    if salary_change_amount is not None and salary_change_date is not None:
        for fe in forecast:
            if fe.category == "salary" and fe.direction == "credit":
                if fe.date >= salary_change_date:
                    fe.amount = salary_change_amount

    if new_salary_date_override is not None:
        for fe in forecast:
            if fe.category == "salary" and not fe.is_projected:
                fe.date = new_salary_date_override

    # ── Part 5: filter to horizon and sort ─────────────────────────────────────
    forecast = [fe for fe in forecast
                if fe.date >= request_date and fe.date <= end_date]
    forecast.sort(key=lambda fe: fe.date)

    return forecast


def compute_running_balance(
    start_balance: float,
    request_date: pd.Timestamp,
    forecast: list[ForecastEntry],
    extra_payments: list[tuple[pd.Timestamp, float]] = None,
    exclude_event_ids: set[str] = None,
    reduce_amounts: dict[str, float] = None,
) -> list[tuple[pd.Timestamp, float]]:
    """
    Simulate daily running balance from request_date through the forecast horizon.
    Returns list of (date, balance) tuples.
    """
    if extra_payments is None:
        extra_payments = []
    if exclude_event_ids is None:
        exclude_event_ids = set()
    if reduce_amounts is None:
        reduce_amounts = {}

    balance = start_balance

    # Collect all events as (date, signed_amount)
    all_events: list[tuple[pd.Timestamp, float]] = []

    for fe in forecast:
        if fe.event_id and fe.event_id in exclude_event_ids:
            continue
        amt = fe.amount
        if fe.event_id and fe.event_id in reduce_amounts:
            amt = reduce_amounts[fe.event_id]
        signed = amt if fe.direction == "credit" else -amt
        all_events.append((fe.date, signed))

    for pay_date, pay_amount in extra_payments:
        all_events.append((pay_date, -pay_amount))  # payment = outflow

    if not all_events and not extra_payments:
        return [(request_date, balance)]

    # Sort and simulate
    all_events.sort(key=lambda x: x[0])

    # Determine date range
    if all_events:
        max_date = max(ev[0] for ev in all_events)
    else:
        max_date = request_date

    timeline = []
    current_date = request_date
    event_idx = 0
    n = len(all_events)

    while current_date <= max_date:
        while event_idx < n and all_events[event_idx][0] == current_date:
            balance += all_events[event_idx][1]
            event_idx += 1
        timeline.append((current_date, round(balance, 2)))
        current_date += pd.Timedelta(days=1)

    return timeline
