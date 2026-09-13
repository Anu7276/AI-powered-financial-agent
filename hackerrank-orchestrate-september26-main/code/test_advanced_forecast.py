import sys
sys.path.insert(0, 'code')
from engine.ingest import DataStore
import pandas as pd
from collections import defaultdict
import numpy as np
from engine.forecast import _detect_cadence, ForecastEntry, compute_running_balance
from engine.solver import compute_amount_safe_to_pay, find_earliest_full_payment_date
from engine.planner import _try_spending_changes, select_best_plan
from engine.reconcile import reconcile_user_events
from main import load_image_facts, load_message_facts

ds = DataStore()
samples = ds.sample_requests

def advanced_build_forecast(entries, message_facts, profile, request_date, horizon_days=90):
    end_date = request_date + pd.Timedelta(days=horizon_days)
    forecast = []
    
    # Confirmed future scheduled / pending
    for e in entries:
        if e.date >= request_date and e.status in ("scheduled", "pending"):
            if e.status == "scheduled" or (e.status == "pending" and e.direction == "debit"):
                forecast.append(ForecastEntry(
                    date=e.date, direction=e.direction, amount=e.amount,
                    category=e.category, flexibility=e.flexibility,
                    minimum_allowed_amount=e.minimum_allowed_amount,
                    event_id=e.event_id, is_projected=False, event_type=e.event_type
                ))

    # Recurrence detection from historical settled
    historical = [e for e in entries if e.date <= request_date and e.status == "settled"]
    
    # Distinguish salary streams by description / day of month
    by_cat = defaultdict(list)
    for e in historical:
        desc = (e.description or "").lower()
        stream = ""
        if e.category == "salary":
            if "comm" in desc or "bonus" in desc:
                stream = "commission"
            else:
                stream = "base"
        by_cat[(e.category, e.direction, e.flexibility, stream)].append(e)

    # Check unconfirmed commission messages
    user_id = profile["user_id"]
    user_msgs = ds.messages[ds.messages["user_id"] == user_id]
    commission_cancelled = any(
        any(w in (m.message_text or "").lower() for w in ["komisi belum", "commission unconfirmed", "commission not approved", "komisi ditunda", "belum disetujui"])
        for m in user_msgs.itertuples()
    )

    for (cat, direction, flex, stream), group in by_cat.items():
        # Only skip genuine one-offs like windfall or travel or one-off investments
        if cat in {"windfall", "travel", "work_expense"} and flex == "fixed":
            continue
        if len(group) < 2:
            continue
        # Skip commission stream if message cancelled/unconfirmed
        if cat == "salary" and stream == "commission" and commission_cancelled:
            continue
            
        sorted_group = sorted(group, key=lambda e: e.date)
        dates = [e.date for e in sorted_group]
        cadence = _detect_cadence(dates)
        
        # Don't project irregular dining or one-off events if cadence > 45 days and not monthly
        if cadence > 35 and len(group) < 3:
            continue
            
        amounts = [e.amount for e in sorted_group]
        if cat in {"groceries", "transport", "shopping", "entertainment", "dining"}:
            recent = amounts[-6:] if len(amounts) >= 6 else amounts
            proj_amount = float(np.median(recent))
        else:
            proj_amount = sorted_group[-1].amount
            
        last_date = sorted_group[-1].date
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
                forecast.append(ForecastEntry(
                    date=proj_date, direction=direction, amount=proj_amount,
                    category=cat, flexibility=flex, minimum_allowed_amount=sorted_group[-1].minimum_allowed_amount,
                    event_id=sorted_group[-1].event_id, is_projected=True, event_type=sorted_group[-1].event_type
                ))
            step += 1

    # Salary change message amendments
    for mf in message_facts:
        if mf.fact_type == "salary_change" and mf.new_amount is not None:
            for fe in forecast:
                if fe.category == "salary" and fe.direction == "credit":
                    fe.amount = mf.new_amount

    forecast.sort(key=lambda fe: fe.date)
    return forecast

print("Testing all 25 sample requests with advanced forecast...")
correct_status = 0
correct_method = 0
results = []

for idx, req in samples.iterrows():
    req_id = req["request_id"]
    user_id = req["user_id"]
    profile = ds.get_profile(user_id)
    img_facts = load_image_facts(ds, req_id, user_id)
    msg_facts = load_message_facts(ds, user_id, req_id)
    entries, _ = reconcile_user_events(user_id, ds, img_facts, msg_facts, req["request_date"])
    
    fc = advanced_build_forecast(entries, msg_facts, profile, req["request_date"])
    
    # Let's inspect plan
    start_bal = float(profile["current_available_balance"])
    min_bal = float(profile["minimum_balance_to_keep"])
    req_amt = float(req["requested_amount"])
    
    plan = select_best_plan(req, profile, entries, fc, msg_facts, ds)
    
    status_match = (plan.affordability_status == req["affordability_status"])
    method_match = (plan.recommended_payment_method == req["recommended_payment_method"])
    if status_match: correct_status += 1
    if method_match: correct_method += 1
    
    print(f"{req_id}: Status {plan.affordability_status:20s} (GT: {req['affordability_status']:20s}) [{'OK' if status_match else 'FAIL'}] | Method {plan.recommended_payment_method:15s} (GT: {req['recommended_payment_method']:15s}) [{'OK' if method_match else 'FAIL'}]")
    if not status_match or not method_match:
        changes = [c.change_type + ':' + c.event_id for c in plan.spending_changes]
        print(f"   Safe: {plan.amount_safe_to_pay} vs GT {req['amount_safe_to_pay']} | Changes: {changes} vs GT {req['spending_changes_needed']}")

print(f"\nTotal: Status {correct_status}/25 ({correct_status/25*100:.1f}%), Method {correct_method}/25 ({correct_method/25*100:.1f}%)")
