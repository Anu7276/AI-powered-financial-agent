"""
explain.py — Generate decision_explanation from computed plan.

Uses a template-first approach: every number comes from the solver, 
not from LLM generation. The LLM is only used to smooth phrasing 
if POLISH_EXPLANATIONS=1 is set in environment.
"""
from __future__ import annotations
import os
from typing import Optional
import pandas as pd
from .planner import PaymentPlan


def _fmt_amount(amount: float, currency: str) -> str:
    """Format: ZAR 25,256 or EUR 1,302.40"""
    if amount == int(amount):
        return f"{currency} {int(amount):,}"
    return f"{currency} {amount:,.2f}"


def generate_explanation(
    plan: PaymentPlan,
    request: pd.Series,
    profile: pd.Series,
    currency: str,
) -> str:
    """Generate a clear, factual decision explanation."""
    method = plan.recommended_payment_method
    status = plan.affordability_status
    req_amount = request["requested_amount"]
    min_bal = float(profile["minimum_balance_to_keep"])
    safe_pay = plan.amount_safe_to_pay

    req_fmt = _fmt_amount(req_amount, currency)
    min_fmt = _fmt_amount(min_bal, currency)
    safe_fmt = _fmt_amount(safe_pay, currency)

    if method == "full_payment" and status == "affordable_now":
        return (
            f"Pay {req_fmt} today. "
            f"This leaves at least {min_fmt} available over the next 90 days."
        )

    elif method == "full_payment" and status == "affordable_with_plan":
        changes_text = _spending_changes_text(plan.spending_changes, currency)
        return (
            f"{changes_text}, then pay {req_fmt} today. "
            f"This leaves at least {min_fmt} available."
        )

    elif method == "installments":
        # Parse schedule
        parts = plan.payment_plan_str.split("|")
        n = len(parts)
        if n > 0:
            first_part = parts[0]
            first_amount = float(first_part.split(":")[1])
            first_date = first_part.split(":")[0]
        inst_fmt = _fmt_amount(first_amount, currency)
        start_date_str = plan.first_payment_date.strftime("%d %B %Y") if plan.first_payment_date else first_date
        return (
            f"Use {n} installments of {inst_fmt}, starting {start_date_str}. "
            f"This leaves at least {min_fmt} available."
        )

    elif method == "partial_payment":
        remainder = req_amount - safe_pay
        second_date = plan.earliest_date_for_full_payment
        second_fmt = _fmt_amount(remainder, currency)
        date_str = second_date.strftime("%d %B %Y") if second_date else "a later date"
        return (
            f"Pay {safe_fmt} today and the remaining {second_fmt} on {date_str}. "
            f"This completes the full request and keeps the {min_fmt} minimum protected."
        )

    elif method == "wait":
        pay_date = plan.first_payment_date
        date_str = pay_date.strftime("%d %B %Y") if pay_date else "a later date"
        return (
            f"Pay {req_fmt} in full on {date_str}. "
            f"Paying earlier would take the balance below the {min_fmt} minimum."
        )

    elif method == "not_recommended":
        if safe_pay > 0:
            return (
                f"Do not proceed with the {req_fmt} request. "
                f"Although {safe_fmt} is available today, the full amount "
                f"cannot be completed safely within 90 days."
            )
        else:
            earliest = plan.earliest_date_for_full_payment
            if earliest is None:
                return (
                    f"Do not make this payment by "
                    f"{request['desired_completion_date'].strftime('%d %B %Y')}. "
                    f"None of the available options keeps the {min_fmt} minimum protected."
                )
            else:
                date_str = earliest.strftime("%d %B %Y")
                return (
                    f"Do not make this payment by "
                    f"{request['desired_completion_date'].strftime('%d %B %Y')}. "
                    f"None of the available options keeps the {min_fmt} minimum protected."
                )

    # Fallback
    return (
        f"Reviewed {req_fmt} request. "
        f"Current safe amount: {safe_fmt}. Minimum balance: {min_fmt}."
    )


def _spending_changes_text(changes: list, currency: str) -> str:
    if not changes:
        return ""
    parts = []
    for c in changes:
        if c.change_type == "stop":
            parts.append(f"stop the {c.event_id} subscription")
        elif c.change_type == "reduce_to":
            amt_fmt = _fmt_amount(c.new_amount, currency) if c.new_amount else ""
            parts.append(f"reduce {c.event_id} to {amt_fmt}")
    return "Stop/reduce: " + ", ".join(parts)
