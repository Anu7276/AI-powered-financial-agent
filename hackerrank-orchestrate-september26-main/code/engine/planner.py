"""
planner.py — Candidate plan generation, eligibility, ranking, and selection.

Generates all eligible payment plans (full_payment, partial_payment, installments,
wait) and ranks them by the spec's exact tie-break ladder:
  1. Completes by desired_completion_date
  2. Requires no spending changes
  3. Minimizes total amount paid
  4. Starts payment earliest
  5. Fewest payments
  6. Lowest payment_option_id (tie-breaker)

Returns the best safe plan or not_recommended.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np

from .forecast import ForecastEntry, build_forecast
from .reconcile import LedgerEntry
from .solver import (
    is_safe, compute_amount_safe_to_pay,
    find_earliest_full_payment_date, compute_installment_plan_safety
)
from .ingest import DataStore


@dataclass
class SpendingChange:
    change_type: str       # stop | reduce_to
    event_id: str
    new_amount: Optional[float] = None  # only for reduce_to


@dataclass
class PaymentPlan:
    affordability_status: str
    recommended_payment_method: str
    payment_plan_str: str              # "YYYY-MM-DD:amount|..."  or "none"
    earliest_date_for_full_payment: Optional[pd.Timestamp]
    spending_changes: list[SpendingChange]
    amount_safe_to_pay: float
    total_paid: float
    num_payments: int
    first_payment_date: Optional[pd.Timestamp]
    payment_option_id: Optional[str]   # for installments tie-break
    completes_by_deadline: bool


def _format_amount(amount: float) -> str:
    """Format amount — integer if whole number, else 2 decimal places."""
    if amount == int(amount):
        return str(int(amount))
    return f"{amount:.2f}"


def _build_payment_plan_str(schedule: list[tuple[pd.Timestamp, float]]) -> str:
    """Format payment plan as 'YYYY-MM-DD:amount|...'"""
    parts = [f"{d.strftime('%Y-%m-%d')}:{_format_amount(a)}" for d, a in schedule]
    return "|".join(parts)


def _parse_pipe_field(val) -> set[str]:
    if pd.isna(val) or val is None:
        return set()
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return set()
    return set(s.split("|")) - {"", "nan"}


def _get_flexible_events(
    entries: list[LedgerEntry],
    profile: pd.Series,
    request_date: pd.Timestamp,
) -> list[LedgerEntry]:
    """
    Return recurring flexible events that the user is willing to stop or reduce.
    Only considers events within recent history (proxy for recurring).
    """
    willing_to_reduce = _parse_pipe_field(profile.get("expense_categories_user_is_willing_to_reduce"))
    willing_to_stop = _parse_pipe_field(profile.get("expense_categories_user_is_willing_to_stop"))

    result = []
    # Find settled recurring events close to request_date as proxies
    seen_categories = {}
    for e in sorted(entries, key=lambda x: x.date, reverse=True):
        if e.status != "settled":
            continue
        key = (e.category, e.direction)
        if key in seen_categories:
            continue
        seen_categories[key] = e
        can_change = False
        if e.flexibility in ("stoppable", "reducible_or_stoppable") and e.category in willing_to_stop:
            can_change = True
        if e.flexibility in ("reducible", "reducible_or_stoppable") and e.category in willing_to_reduce:
            can_change = True
        if can_change:
            result.append(e)

    return result


def _try_spending_changes(
    entries: list[LedgerEntry],
    profile: pd.Series,
    request_date: pd.Timestamp,
    forecast: list[ForecastEntry],
    requested_amount: float,
    desired_completion_date: pd.Timestamp,
    start_balance: float,
    minimum_balance: float,
    message_facts: list,
    max_changes: int = 3,
) -> tuple[list[SpendingChange], list[ForecastEntry], float]:
    """
    Try applying flexible spending changes (stop/reduce) to make more room.
    Returns (changes, modified_forecast, new_safe_amount).
    Tries up to 3 changes, only modifies forecast copy.
    """
    flexible = _get_flexible_events(entries, profile, request_date)
    if not flexible:
        return [], forecast, compute_amount_safe_to_pay(
            start_balance, minimum_balance, requested_amount, request_date, forecast
        )

    willing_to_stop = _parse_pipe_field(profile.get("expense_categories_user_is_willing_to_stop"))
    willing_to_reduce = _parse_pipe_field(profile.get("expense_categories_user_is_willing_to_reduce"))

    # Build working exclude/reduce sets
    exclude_ids: set[str] = set()
    reduce_map: dict[str, float] = {}
    applied_changes: list[SpendingChange] = []

    # Greedy: try stops first (larger impact), then reductions
    for e in flexible:
        if len(applied_changes) >= max_changes:
            break

        already_stopped = e.event_id in exclude_ids
        already_reduced = e.event_id in reduce_map

        if not already_stopped and not already_reduced:
            if e.flexibility in ("stoppable", "reducible_or_stoppable") and e.category in willing_to_stop:
                # Try stopping
                trial_exclude = exclude_ids | {e.event_id}
                safe_with = compute_amount_safe_to_pay(
                    start_balance, minimum_balance, requested_amount,
                    request_date, forecast, trial_exclude, reduce_map
                )
                if safe_with > compute_amount_safe_to_pay(
                    start_balance, minimum_balance, requested_amount,
                    request_date, forecast, exclude_ids, reduce_map
                ):
                    exclude_ids.add(e.event_id)
                    applied_changes.append(SpendingChange("stop", e.event_id))
                    continue

            if e.flexibility in ("reducible", "reducible_or_stoppable") and e.category in willing_to_reduce:
                # Try reducing to minimum_allowed_amount (or 50% if no min specified)
                min_allowed = e.minimum_allowed_amount
                if min_allowed is None:
                    min_allowed = round(e.amount * 0.5, 2)
                if min_allowed < e.amount:
                    trial_reduce = {**reduce_map, e.event_id: min_allowed}
                    safe_with = compute_amount_safe_to_pay(
                        start_balance, minimum_balance, requested_amount,
                        request_date, forecast, exclude_ids, trial_reduce
                    )
                    if safe_with > compute_amount_safe_to_pay(
                        start_balance, minimum_balance, requested_amount,
                        request_date, forecast, exclude_ids, reduce_map
                    ):
                        reduce_map[e.event_id] = min_allowed
                        applied_changes.append(SpendingChange("reduce_to", e.event_id, min_allowed))

    final_safe = compute_amount_safe_to_pay(
        start_balance, minimum_balance, requested_amount,
        request_date, forecast, exclude_ids, reduce_map
    )
    return applied_changes, forecast, final_safe


def select_best_plan(
    request: pd.Series,
    profile: pd.Series,
    entries: list[LedgerEntry],
    forecast: list[ForecastEntry],
    message_facts: list,
    ds: DataStore,
) -> PaymentPlan:
    """
    Generate all candidate plans and return the best one according to spec ranking.
    """
    user_id = request["user_id"]
    request_id = request["request_id"]
    request_date = request["request_date"]
    requested_amount = request["requested_amount"]
    desired_completion_date = request["desired_completion_date"]
    allows_partial = request["allows_partial_payment"]
    # NaN or non-True values are treated as False (partial payment not allowed)
    if not isinstance(allows_partial, bool):
        allows_partial = False
    else:
        allows_partial = bool(allows_partial)

    start_balance = float(profile["current_available_balance"])
    minimum_balance = float(profile["minimum_balance_to_keep"])
    max_installment_months = profile.get("max_installment_months")
    if pd.isna(max_installment_months):
        max_installment_months = None
    else:
        max_installment_months = int(max_installment_months)

    payment_methods = _parse_pipe_field(profile.get("payment_methods_user_will_consider"))

    # ── Compute base safe amount and earliest date ─────────────────────────────
    amount_safe = compute_amount_safe_to_pay(
        start_balance, minimum_balance, requested_amount, request_date, forecast
    )
    earliest_full_date = find_earliest_full_payment_date(
        start_balance, minimum_balance, requested_amount, request_date, forecast,
        90, desired_completion_date
    )

    candidates: list[PaymentPlan] = []

    # ── Candidate 1: full_payment on request_date ──────────────────────────────
    if "full_payment" in payment_methods:
        if amount_safe >= requested_amount:
            # Check it passes the safety check
            if is_safe(start_balance, minimum_balance, request_date, forecast,
                       request_date, requested_amount):
                plan_str = _build_payment_plan_str([(request_date, requested_amount)])
                candidates.append(PaymentPlan(
                    affordability_status="affordable_now",
                    recommended_payment_method="full_payment",
                    payment_plan_str=plan_str,
                    earliest_date_for_full_payment=request_date,
                    spending_changes=[],
                    amount_safe_to_pay=requested_amount,
                    total_paid=requested_amount,
                    num_payments=1,
                    first_payment_date=request_date,
                    payment_option_id=None,
                    completes_by_deadline=True,
                ))

    # ── Candidate 2: wait for full_payment ────────────────────────────────────
    if "full_payment" in payment_methods and earliest_full_date is not None:
        if earliest_full_date > request_date:
            completes_by = earliest_full_date <= desired_completion_date
            plan_str = _build_payment_plan_str([(earliest_full_date, requested_amount)])
            candidates.append(PaymentPlan(
                affordability_status="affordable_later",
                recommended_payment_method="wait",
                payment_plan_str=plan_str,
                earliest_date_for_full_payment=earliest_full_date,
                spending_changes=[],
                amount_safe_to_pay=amount_safe,
                total_paid=requested_amount,
                num_payments=1,
                first_payment_date=earliest_full_date,
                payment_option_id=None,
                completes_by_deadline=completes_by,
            ))

    # ── Candidate 3: installments ──────────────────────────────────────────────
    if "installments" in payment_methods:
        options = ds.get_request_options(request_id)
        installment_opts = options[options["payment_method"] == "installments"]

        for _, opt in installment_opts.iterrows():
            n = int(opt["number_of_payments"])
            first_date = opt["first_payment_date"]
            freq_days = int(opt["payment_frequency_days"]) if not pd.isna(opt["payment_frequency_days"]) else 30
            payment_amount = float(opt["payment_amount"])
            total_payable = float(opt["total_payable_amount"])
            opt_id = opt["payment_option_id"]

            # Check max installment months constraint
            if max_installment_months is not None:
                months_duration = (n * freq_days) / 30
                if months_duration > max_installment_months:
                    continue  # exceeds user's max installment period

            # Build schedule
            schedule = []
            pay_date = first_date
            for i in range(n):
                schedule.append((pay_date, payment_amount))
                pay_date = pay_date + pd.Timedelta(days=freq_days)

            # Check safety
            safe = compute_installment_plan_safety(
                start_balance, minimum_balance, request_date, forecast,
                schedule, desired_completion_date
            )
            if safe:
                plan_str = _build_payment_plan_str(schedule)
                last_pay = max(d for d, _ in schedule)
                completes_by = last_pay <= desired_completion_date
                candidates.append(PaymentPlan(
                    affordability_status="affordable_with_plan",
                    recommended_payment_method="installments",
                    payment_plan_str=plan_str,
                    earliest_date_for_full_payment=earliest_full_date,
                    spending_changes=[],
                    amount_safe_to_pay=amount_safe,
                    total_paid=total_payable,
                    num_payments=n,
                    first_payment_date=first_date,
                    payment_option_id=opt_id,
                    completes_by_deadline=completes_by,
                ))

    # ── Candidate 4: partial_payment ──────────────────────────────────────────
    if "partial_payment" in payment_methods and allows_partial:
        if 0 < amount_safe < requested_amount and earliest_full_date is not None:
            if earliest_full_date <= desired_completion_date:
                remainder = requested_amount - amount_safe
                schedule = [
                    (request_date, amount_safe),
                    (earliest_full_date, remainder),
                ]
                # Verify the full schedule is safe
                safe = compute_installment_plan_safety(
                    start_balance, minimum_balance, request_date, forecast,
                    schedule, desired_completion_date
                )
                if safe:
                    plan_str = _build_payment_plan_str(schedule)
                    candidates.append(PaymentPlan(
                        affordability_status="affordable_with_plan",
                        recommended_payment_method="partial_payment",
                        payment_plan_str=plan_str,
                        earliest_date_for_full_payment=earliest_full_date,
                        spending_changes=[],
                        amount_safe_to_pay=amount_safe,
                        total_paid=requested_amount,
                        num_payments=2,
                        first_payment_date=request_date,
                        payment_option_id=None,
                        completes_by_deadline=True,
                    ))

    # ── Try spending changes if no safe plan found yet ─────────────────────────
    has_safe = any(c.completes_by_deadline for c in candidates)
    if not has_safe:
        changes, mod_forecast, new_safe = _try_spending_changes(
            entries, profile, request_date, forecast, requested_amount,
            desired_completion_date, start_balance, minimum_balance, message_facts
        )
        if changes:
            # Recompute with spending changes applied
            exclude_ids = {c.event_id for c in changes if c.change_type == "stop"}
            reduce_map = {c.event_id: c.new_amount for c in changes if c.change_type == "reduce_to"}

            new_earliest = find_earliest_full_payment_date(
                start_balance, minimum_balance, requested_amount, request_date,
                forecast, 90, desired_completion_date
            )

            # full_payment with spending changes
            if "full_payment" in payment_methods:
                full_safe = is_safe(
                    start_balance, minimum_balance, request_date, forecast,
                    request_date, requested_amount, None, exclude_ids, reduce_map
                )
                if full_safe:
                    plan_str = _build_payment_plan_str([(request_date, requested_amount)])
                    candidates.append(PaymentPlan(
                        affordability_status="affordable_with_plan",
                        recommended_payment_method="full_payment",
                        payment_plan_str=plan_str,
                        earliest_date_for_full_payment=earliest_full_date,
                        spending_changes=changes,
                        amount_safe_to_pay=amount_safe,
                        total_paid=requested_amount,
                        num_payments=1,
                        first_payment_date=request_date,
                        payment_option_id=None,
                        completes_by_deadline=True,
                    ))

            # installments with spending changes
            if "installments" in payment_methods:
                options = ds.get_request_options(request_id)
                for _, opt in options[options["payment_method"] == "installments"].iterrows():
                    n = int(opt["number_of_payments"])
                    freq_days = int(opt["payment_frequency_days"]) if not pd.isna(opt["payment_frequency_days"]) else 30
                    if max_installment_months is not None:
                        if (n * freq_days) / 30 > max_installment_months:
                            continue
                    first_date = opt["first_payment_date"]
                    payment_amount = float(opt["payment_amount"])
                    total_payable = float(opt["total_payable_amount"])
                    opt_id = opt["payment_option_id"]
                    schedule = []
                    pay_date = first_date
                    for i in range(n):
                        schedule.append((pay_date, payment_amount))
                        pay_date = pay_date + pd.Timedelta(days=freq_days)
                    
                    # Safety check with spending changes
                    from .forecast import compute_running_balance
                    from .solver import _min_future_balance
                    min_bal = _min_future_balance(
                        start_balance, request_date, forecast,
                        extra_payments=schedule,
                        exclude_event_ids=exclude_ids,
                        reduce_amounts=reduce_map,
                    )
                    last_pay = max(d for d, _ in schedule)
                    if min_bal >= minimum_balance and last_pay <= desired_completion_date:
                        plan_str = _build_payment_plan_str(schedule)
                        candidates.append(PaymentPlan(
                            affordability_status="affordable_with_plan",
                            recommended_payment_method="installments",
                            payment_plan_str=plan_str,
                            earliest_date_for_full_payment=earliest_full_date,
                            spending_changes=changes,
                            amount_safe_to_pay=amount_safe,
                            total_paid=total_payable,
                            num_payments=n,
                            first_payment_date=first_date,
                            payment_option_id=opt_id,
                            completes_by_deadline=True,
                        ))

    # ── Select best candidate ─────────────────────────────────────────────────
    if not candidates:
        # not_affordable / not_recommended
        return PaymentPlan(
            affordability_status="not_affordable",
            recommended_payment_method="not_recommended",
            payment_plan_str="none",
            earliest_date_for_full_payment=earliest_full_date,
            spending_changes=[],
            amount_safe_to_pay=amount_safe,
            total_paid=0.0,
            num_payments=0,
            first_payment_date=None,
            payment_option_id=None,
            completes_by_deadline=False,
        )

    # Rank by spec's tie-break ladder
    def _rank_key(p: PaymentPlan):
        # Lower score = better
        completes = 0 if p.completes_by_deadline else 1
        has_changes = 0 if not p.spending_changes else 1
        total = p.total_paid
        first = p.first_payment_date or pd.Timestamp("2099-01-01")
        n_payments = p.num_payments
        opt_id = p.payment_option_id or "zzz"
        # Extract numeric part for sort
        opt_num = int("".join(filter(str.isdigit, opt_id))) if opt_id != "zzz" else 9999
        return (completes, has_changes, total, first, n_payments, opt_num)

    candidates.sort(key=_rank_key)
    best = candidates[0]
    return best
