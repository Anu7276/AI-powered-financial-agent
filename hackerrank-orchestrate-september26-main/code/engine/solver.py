"""
solver.py — Safety-check engine.

Computes:
  - amount_safe_to_pay: max amount payable today that keeps balance >= minimum
    throughout the 90-day forecast (capped at requested_amount).
  - earliest_date_for_full_payment: first date in [request_date, request_date+90]
    when paying the full requested_amount keeps the forecast safe.

"Safe" = balance never drops below minimum_balance_to_keep on any forecasted day.
"""
from __future__ import annotations
from typing import Optional
import pandas as pd
import numpy as np
from .forecast import ForecastEntry, compute_running_balance


def _min_future_balance(
    start_balance: float,
    request_date: pd.Timestamp,
    forecast: list[ForecastEntry],
    extra_payments: list[tuple[pd.Timestamp, float]] = None,
    exclude_event_ids: set[str] = None,
    reduce_amounts: dict[str, float] = None,
) -> float:
    """Return the minimum balance over the 90-day simulation."""
    timeline = compute_running_balance(
        start_balance=start_balance,
        request_date=request_date,
        forecast=forecast,
        extra_payments=extra_payments or [],
        exclude_event_ids=exclude_event_ids or set(),
        reduce_amounts=reduce_amounts or {},
    )
    if not timeline:
        return start_balance
    return min(bal for _, bal in timeline)


def is_safe(
    start_balance: float,
    minimum_balance: float,
    request_date: pd.Timestamp,
    forecast: list[ForecastEntry],
    payment_date: pd.Timestamp,
    payment_amount: float,
    extra_payments: list[tuple[pd.Timestamp, float]] = None,
    exclude_event_ids: set[str] = None,
    reduce_amounts: dict[str, float] = None,
) -> bool:
    """Return True if adding payment_amount on payment_date keeps the forecast safe."""
    payments = [(payment_date, payment_amount)]
    if extra_payments:
        payments.extend(extra_payments)
    min_bal = _min_future_balance(
        start_balance=start_balance,
        request_date=request_date,
        forecast=forecast,
        extra_payments=payments,
        exclude_event_ids=exclude_event_ids,
        reduce_amounts=reduce_amounts,
    )
    return min_bal >= minimum_balance


def compute_amount_safe_to_pay(
    start_balance: float,
    minimum_balance: float,
    requested_amount: float,
    request_date: pd.Timestamp,
    forecast: list[ForecastEntry],
    exclude_event_ids: set[str] = None,
    reduce_amounts: dict[str, float] = None,
) -> float:
    """
    Binary-search for the maximum amount payable TODAY (request_date) that
    keeps the 90-day balance >= minimum_balance at all times.
    Capped at requested_amount.
    Returns 0.0 if even 1 unit is unsafe.
    """
    # Quick full check first
    if is_safe(start_balance, minimum_balance, request_date, forecast,
               request_date, requested_amount, None, exclude_event_ids, reduce_amounts):
        return requested_amount

    # Check if paying 0 is safe (baseline)
    min_bal_no_payment = _min_future_balance(
        start_balance, request_date, forecast,
        extra_payments=None,
        exclude_event_ids=exclude_event_ids,
        reduce_amounts=reduce_amounts,
    )
    if min_bal_no_payment < minimum_balance:
        # Even without payment, balance goes below minimum — safe amount is 0
        return 0.0

    # Available headroom above minimum on the tightest day
    # The headroom is the maximum we can pay today without making any future day unsafe
    headroom = min_bal_no_payment - minimum_balance
    safe_upper = min(headroom, requested_amount)

    if safe_upper <= 0:
        return 0.0

    # Binary search for exact amount
    lo, hi = 0.0, safe_upper
    for _ in range(40):
        mid = (lo + hi) / 2
        if is_safe(start_balance, minimum_balance, request_date, forecast,
                   request_date, mid, None, exclude_event_ids, reduce_amounts):
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.01:
            break

    return round(lo, 2)


def find_earliest_full_payment_date(
    start_balance: float,
    minimum_balance: float,
    requested_amount: float,
    request_date: pd.Timestamp,
    forecast: list[ForecastEntry],
    horizon_days: int = 90,
    desired_completion_date: Optional[pd.Timestamp] = None,
) -> Optional[pd.Timestamp]:
    """
    Scan [request_date, request_date + horizon_days] to find the earliest date
    on which paying requested_amount as a single payment keeps the forecast safe.
    Returns None if no such date exists within the horizon.
    """
    for offset in range(horizon_days + 1):
        candidate_date = request_date + pd.Timedelta(days=offset)
        if is_safe(start_balance, minimum_balance, request_date, forecast,
                   candidate_date, requested_amount):
            return candidate_date
        # If candidate_date completes by desired_completion_date, also check if safe up to desired_completion_date
        if desired_completion_date is not None and candidate_date <= desired_completion_date:
            timeline = compute_running_balance(
                start_balance, request_date, forecast,
                extra_payments=[(candidate_date, requested_amount)]
            )
            sub = [b for d, b in timeline if d <= desired_completion_date]
            if sub and min(sub) >= minimum_balance:
                return candidate_date
    return None


def compute_installment_plan_safety(
    start_balance: float,
    minimum_balance: float,
    request_date: pd.Timestamp,
    forecast: list[ForecastEntry],
    payment_schedule: list[tuple[pd.Timestamp, float]],  # [(date, amount), ...]
    desired_completion_date: pd.Timestamp,
) -> bool:
    """
    Check if the given installment payment schedule is safe AND completes
    on or before desired_completion_date.
    """
    if not payment_schedule:
        return False
    last_payment_date = max(d for d, _ in payment_schedule)
    if last_payment_date > desired_completion_date:
        return False

    # Simulate all installments up to the completion horizon
    timeline = compute_running_balance(
        start_balance=start_balance,
        request_date=request_date,
        forecast=forecast,
        extra_payments=payment_schedule,
    )
    horizon_cutoff = max(last_payment_date, desired_completion_date)
    sub = [b for d, b in timeline if d <= horizon_cutoff]
    return min(sub) >= minimum_balance if sub else True
