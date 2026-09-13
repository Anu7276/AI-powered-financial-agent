"""
verify.py — Deterministic post-hoc validation of output rows.

Checks every hard rule from the problem spec before writing output.csv.
Logs any violations and applies safe fallbacks.
"""
from __future__ import annotations
from typing import Optional
import pandas as pd
import numpy as np


VALID_STATUSES = {
    "affordable_now", "affordable_with_plan",
    "affordable_later", "not_affordable"
}
VALID_METHODS = {
    "full_payment", "partial_payment", "installments",
    "wait", "not_recommended"
}


def verify_row(row: dict, request: pd.Series, profile: pd.Series) -> tuple[dict, list[str]]:
    """
    Validate a single output row against all spec rules.
    Returns (corrected_row, list_of_violations).
    """
    violations = []
    r = dict(row)  # work on a copy

    req_amount = float(request["requested_amount"])
    request_date = request["request_date"]
    completion_date = request["desired_completion_date"]

    # ── Rule: amount_safe_to_pay ∈ [0, requested_amount] ─────────────────────
    safe_pay = r.get("amount_safe_to_pay", 0.0)
    if pd.isna(safe_pay) or safe_pay < 0:
        violations.append(f"amount_safe_to_pay={safe_pay} < 0, clamping to 0")
        r["amount_safe_to_pay"] = 0.0
    elif safe_pay > req_amount:
        violations.append(f"amount_safe_to_pay={safe_pay} > requested_amount={req_amount}, clamping")
        r["amount_safe_to_pay"] = req_amount

    # ── Rule: valid affordability_status ──────────────────────────────────────
    if r.get("affordability_status") not in VALID_STATUSES:
        violations.append(f"invalid affordability_status: {r.get('affordability_status')}")
        r["affordability_status"] = "not_affordable"

    # ── Rule: valid recommended_payment_method ────────────────────────────────
    if r.get("recommended_payment_method") not in VALID_METHODS:
        violations.append(f"invalid recommended_payment_method: {r.get('recommended_payment_method')}")
        r["recommended_payment_method"] = "not_recommended"

    # ── Rule: affordable_now ⇒ earliest_date_for_full_payment == request_date ─
    if r.get("affordability_status") == "affordable_now":
        efp = r.get("earliest_date_for_full_payment")
        if efp is None or pd.isna(efp) or str(efp) == "nan":
            violations.append("affordable_now but earliest_date_for_full_payment missing")
            r["earliest_date_for_full_payment"] = request_date.strftime("%Y-%m-%d")
        else:
            efp_dt = pd.Timestamp(str(efp)) if not isinstance(efp, pd.Timestamp) else efp
            if efp_dt != request_date:
                violations.append(f"affordable_now but earliest={efp} != request_date={request_date}")
                r["earliest_date_for_full_payment"] = request_date.strftime("%Y-%m-%d")

    # ── Rule: partial_payment requirements ────────────────────────────────────
    if r.get("recommended_payment_method") == "partial_payment":
        plan_str = r.get("payment_plan", "none")
        if plan_str and plan_str != "none":
            parts = plan_str.split("|")
            if len(parts) != 2:
                violations.append(f"partial_payment must have exactly 2 payments, got {len(parts)}")
            else:
                try:
                    amt1 = float(parts[0].split(":")[1])
                    amt2 = float(parts[1].split(":")[1])
                    total = round(amt1 + amt2, 2)
                    if abs(total - req_amount) > 0.02:
                        violations.append(
                            f"partial_payment amounts {amt1}+{amt2}={total} != requested {req_amount}"
                        )
                except (IndexError, ValueError):
                    violations.append("Could not parse partial_payment plan amounts")

    # ── Rule: payment_plan format ─────────────────────────────────────────────
    plan_str = r.get("payment_plan", "none")
    if plan_str and plan_str != "none":
        # Verify chronological order
        try:
            parts = plan_str.split("|")
            dates = [pd.Timestamp(p.split(":")[0]) for p in parts]
            for i in range(len(dates) - 1):
                if dates[i] > dates[i+1]:
                    violations.append(f"payment_plan not in chronological order: {plan_str[:100]}")
                    break
        except Exception:
            violations.append(f"payment_plan parse error: {plan_str[:100]}")

    # ── Rule: not_recommended ⇒ payment_plan = "none" ─────────────────────────
    if r.get("recommended_payment_method") == "not_recommended":
        if r.get("payment_plan") not in (None, "none", ""):
            violations.append("not_recommended should have payment_plan=none")
            r["payment_plan"] = "none"

    # ── Rule: spending_changes format ─────────────────────────────────────────
    sc = r.get("spending_changes_needed", "none")
    if sc and sc != "none":
        parts = sc.split("|")
        if len(parts) > 3:
            violations.append(f"spending_changes_needed has {len(parts)} entries (max 3)")
        # Check for stop+reduce on same event
        stopped = set()
        reduced = set()
        for p in parts:
            if p.startswith("stop:"):
                eid = p[5:].strip()
                stopped.add(eid)
            elif p.startswith("reduce_to:"):
                bits = p.split(":")
                eid = bits[1].strip() if len(bits) > 1 else ""
                reduced.add(eid)
        overlap = stopped & reduced
        if overlap:
            violations.append(f"same events appear in both stop and reduce_to: {overlap}")

    # ── Ensure required fields are present ───────────────────────────────────
    required = [
        "request_id", "amount_safe_to_pay", "affordability_status",
        "recommended_payment_method", "payment_plan",
        "earliest_date_for_full_payment", "spending_changes_needed",
        "decision_explanation"
    ]
    for field in required:
        if field not in r or r[field] is None:
            violations.append(f"missing required field: {field}")
            r[field] = "" if field != "amount_safe_to_pay" else 0.0

    return r, violations
