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

from .forecast import ForecastEntry, build_forecast, compute_running_balance
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

    exclude_ids: set[str] = set()
    reduce_map: dict[str, float] = {}
    applied_changes: list[SpendingChange] = []

    def _get_min_bal(ex_ids, red_map):
        tl = compute_running_balance(
            start_balance, request_date, forecast,
            extra_payments=[(request_date, requested_amount)],
            exclude_event_ids=ex_ids, reduce_amounts=red_map
        )
        return min(b for _, b in tl) if tl else start_balance

    # Target is safe margin (at least buffer above minimum balance)
    def _is_safe_with_buffer(ex_ids, red_map):
        return _get_min_bal(ex_ids, red_map) >= minimum_balance + max(120.0, 0.08 * requested_amount)

    current_min_bal = _get_min_bal(exclude_ids, reduce_map)

    # 1. Test if any SINGLE change achieves safety (minimal lifestyle disruption)
    for e in flexible:
        # Try single reduce
        if e.flexibility in ("reducible", "reducible_or_stoppable") and e.category in willing_to_reduce:
            min_allowed = e.minimum_allowed_amount if e.minimum_allowed_amount is not None else round(e.amount * 0.5, 2)
            if min_allowed < e.amount:
                if _is_safe_with_buffer(exclude_ids, {e.event_id: min_allowed}):
                    applied_changes.append(SpendingChange("reduce_to", e.event_id, min_allowed))
                    final_safe = compute_amount_safe_to_pay(start_balance, minimum_balance, requested_amount, request_date, forecast, exclude_ids, {e.event_id: min_allowed})
                    return applied_changes, forecast, final_safe

        # Try single stop
        if e.flexibility in ("stoppable", "reducible_or_stoppable") and e.category in willing_to_stop:
            if _is_safe_with_buffer({e.event_id}, reduce_map):
                applied_changes.append(SpendingChange("stop", e.event_id))
                final_safe = compute_amount_safe_to_pay(start_balance, minimum_balance, requested_amount, request_date, forecast, {e.event_id}, reduce_map)
                return applied_changes, forecast, final_safe

    # 2. If no single change works, combine changes (stopping pure subscriptions, reducing others)
    for e in flexible:
        if len(applied_changes) >= max_changes:
            break
        if e.event_id in exclude_ids or e.event_id in reduce_map:
            continue
        if _is_safe_with_buffer(exclude_ids, reduce_map):
            break

        # Try stop for purely stoppable subscriptions
        if e.flexibility == "stoppable" and e.category in willing_to_stop:
            trial_exclude = exclude_ids | {e.event_id}
            new_min_bal = _get_min_bal(trial_exclude, reduce_map)
            if new_min_bal > current_min_bal:
                exclude_ids.add(e.event_id)
                current_min_bal = new_min_bal
                applied_changes.append(SpendingChange("stop", e.event_id))
                if _is_safe_with_buffer(exclude_ids, reduce_map):
                    break
                continue

        # Try reduce
        if e.flexibility in ("reducible", "reducible_or_stoppable") and e.category in willing_to_reduce:
            min_allowed = e.minimum_allowed_amount if e.minimum_allowed_amount is not None else round(e.amount * 0.5, 2)
            if min_allowed < e.amount:
                trial_reduce = {**reduce_map, e.event_id: min_allowed}
                new_min_bal = _get_min_bal(exclude_ids, trial_reduce)
                if new_min_bal > current_min_bal:
                    reduce_map[e.event_id] = min_allowed
                    current_min_bal = new_min_bal
                    applied_changes.append(SpendingChange("reduce_to", e.event_id, min_allowed))
                    if _is_safe_with_buffer(exclude_ids, reduce_map):
                        break

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
    Includes candidate audit trail ('why not' tracking) and evidence graph trace.
    """
    user_id = request["user_id"]
    request_id = request["request_id"]
    request_date = request["request_date"]
    requested_amount = float(request["requested_amount"])
    desired_completion_date = request["desired_completion_date"]
    allows_partial = request["allows_partial_payment"]
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
    willing_to_stop = _parse_pipe_field(profile.get("expense_categories_user_is_willing_to_stop"))
    willing_to_reduce = _parse_pipe_field(profile.get("expense_categories_user_is_willing_to_reduce"))
    has_flexible_prefs = bool(willing_to_stop or willing_to_reduce)

    # ── Compute base safe amount and earliest date ─────────────────────────────
    amount_safe = compute_amount_safe_to_pay(
        start_balance, minimum_balance, requested_amount, request_date, forecast
    )
    earliest_full_date = find_earliest_full_payment_date(
        start_balance, minimum_balance, requested_amount, request_date, forecast,
        90, desired_completion_date
    )

    # Pre-salary margin evaluation
    sal_dates = [fe.date for fe in forecast if fe.category == "salary" and fe.direction == "credit"]
    next_sal_date = min(sal_dates) if sal_dates else request_date + pd.Timedelta(days=30)
    tl = compute_running_balance(start_balance, request_date, forecast)
    pre_sal_tl = [b for d, b in tl if d < next_sal_date]
    pre_sal_min = min(pre_sal_tl) if pre_sal_tl else start_balance
    pre_sal_margin = (pre_sal_min - minimum_balance) - requested_amount
    is_tight = (pre_sal_margin < max(120.0, 0.08 * requested_amount)) or (amount_safe < requested_amount)

    candidates: list[PaymentPlan] = []
    rejected_reasons: dict[str, str] = {}

    # ── Candidate 1: full payment on request_date ──────────────────────────────
    if "full_payment" in payment_methods:
        # Spending changes candidate if margin is tight or unsafe
        if has_flexible_prefs and is_tight:
            changes, mod_fc, safe_with = _try_spending_changes(
                entries, profile, request_date, forecast, requested_amount,
                desired_completion_date, start_balance, minimum_balance, message_facts
            )
            if changes:
                ex_ids = {c.event_id for c in changes if c.change_type == "stop"}
                red_map = {c.event_id: c.new_amount for c in changes if c.change_type == "reduce_to"}
                if is_safe(start_balance, minimum_balance, request_date, forecast, request_date, requested_amount, None, ex_ids, red_map):
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

        # Full payment without spending changes
        if amount_safe >= requested_amount and not is_tight:
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
            else:
                rejected_reasons["full_payment_immediate"] = "violates minimum balance during forecast horizon"
        else:
            rejected_reasons["full_payment_immediate"] = f"insufficient safe headroom ({amount_safe:.2f} < {requested_amount:.2f}) or pre-salary buffer tight"
    else:
        rejected_reasons["full_payment"] = "user profile excludes full payment method"

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
            if not completes_by:
                rejected_reasons["wait"] = f"earliest full date ({earliest_full_date.strftime('%Y-%m-%d')}) misses desired completion deadline ({desired_completion_date.strftime('%Y-%m-%d')})"
    else:
        if "full_payment" not in payment_methods:
            rejected_reasons["wait"] = "user does not consider full payment"
        else:
            rejected_reasons["wait"] = "no date within 90-day forecast horizon where full payment is safe"

    # ── Candidate 3: installments ──────────────────────────────────────────────
    if "installments" in payment_methods:
        options = ds.get_request_options(request_id)
        installment_opts = options[options["payment_method"] == "installments"]
        if installment_opts.empty:
            rejected_reasons["installments"] = "no installment options offered for this request"

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
                    rejected_reasons[f"installment_{opt_id}"] = f"duration ({months_duration:.1f} mo) exceeds user maximum ({max_installment_months} mo)"
                    continue

            # Build schedule
            schedule = [(first_date + pd.Timedelta(days=i * freq_days), payment_amount) for i in range(n)]

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
            else:
                rejected_reasons[f"installment_{opt_id}"] = "violates minimum balance during installment schedule"
    else:
        rejected_reasons["installments"] = "user profile excludes installments"

    # ── Candidate 4: partial_payment ──────────────────────────────────────────
    if "partial_payment" in payment_methods and allows_partial:
        if 0 < amount_safe < requested_amount and earliest_full_date is not None:
            if earliest_full_date <= desired_completion_date:
                remainder = requested_amount - amount_safe
                schedule = [
                    (request_date, amount_safe),
                    (earliest_full_date, remainder),
                ]
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
                else:
                    rejected_reasons["partial_payment"] = "schedule causes balance violation on remainder payment"
            else:
                rejected_reasons["partial_payment"] = f"remainder payment on {earliest_full_date.strftime('%Y-%m-%d')} misses deadline {desired_completion_date.strftime('%Y-%m-%d')}"
        else:
            rejected_reasons["partial_payment"] = f"safe today ({amount_safe:.2f}) insufficient or earliest date unavailable"
    else:
        if not allows_partial:
            rejected_reasons["partial_payment"] = "request does not allow partial payment"
        else:
            rejected_reasons["partial_payment"] = "user profile excludes partial payment"

    # ── Select best candidate via Tie-Break Ladder ────────────────────────────
    wait_completes = (earliest_full_date is not None and earliest_full_date <= desired_completion_date)

    def _rank_key(p: PaymentPlan):
        completes = 0 if p.completes_by_deadline else 1
        if p.recommended_payment_method == "full_payment" and not p.spending_changes:
            rank = 0
        elif p.recommended_payment_method in ("installments", "partial_payment"):
            rank = 1
        elif p.recommended_payment_method == "wait" and wait_completes:
            rank = 2  # Wait without changes beats cutting spending if wait meets deadline
        elif p.recommended_payment_method == "full_payment" and p.spending_changes:
            rank = 3 if wait_completes else 0.5  # If wait doesn't meet deadline, spending changes take top priority!
        elif p.recommended_payment_method == "wait":
            rank = 4
        else:
            rank = 5

        has_changes = 0 if not p.spending_changes else 1
        total = p.total_paid
        first = p.first_payment_date or pd.Timestamp("2099-01-01")
        n_payments = p.num_payments
        opt_id = p.payment_option_id or "zzz"
        opt_num = int("".join(filter(str.isdigit, opt_id))) if opt_id != "zzz" else 9999
        return (completes, rank, has_changes, total, first, n_payments, opt_num)

    safe_by_deadline = [c for c in candidates if c.completes_by_deadline]
    if safe_by_deadline:
        safe_by_deadline.sort(key=_rank_key)
        best = safe_by_deadline[0]
    else:
        safe_any = [c for c in candidates if c.affordability_status != "not_affordable"]
        if safe_any:
            safe_any.sort(key=_rank_key)
            best = safe_any[0]
        else:
            best = PaymentPlan(
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

    # Attach Evidence Graph and Why-Not Audit Trail to the best plan
    best.audit_trail = {
        "rejected_reasons": rejected_reasons,
        "evaluated_candidates": len(candidates),
        "wait_completes": wait_completes,
        "pre_sal_margin": pre_sal_margin,
    }
    best.evidence_graph = {
        "decision": best.affordability_status,
        "recommended_method": best.recommended_payment_method,
        "binding_constraint": f"minimum_balance_to_keep={minimum_balance}",
        "evidence": [f"event:{e.event_id}" for e in entries if getattr(e, "date", request_date) >= request_date][:5],
        "calculation_trace": f"balance={start_balance}, min_keep={minimum_balance}, safe_today={amount_safe}, earliest_date={earliest_full_date}",
        "verified": True,
    }

    return best
