import sys
sys.path.insert(0, 'code')
from engine.ingest import DataStore
from engine.forecast import build_forecast, compute_running_balance
from main import load_image_facts, load_message_facts
from engine.reconcile import reconcile_user_events
import pandas as pd

ds = DataStore()
samples = ds.sample_requests

for req_id in ['request_06', 'request_11', 'request_21']:
    req = samples[samples['request_id'] == req_id].iloc[0]
    user_id = req['user_id']
    profile = ds.get_profile(user_id)
    img_facts = load_image_facts(ds, req_id, user_id)
    msg_facts = load_message_facts(ds, user_id, req_id)
    entries, _ = reconcile_user_events(user_id, ds, img_facts, msg_facts, req['request_date'])
    fc = build_forecast(entries, msg_facts, profile, req['request_date'])
    
    start_bal = float(profile['current_available_balance'])
    min_bal = float(profile['minimum_balance_to_keep'])
    req_amt = float(req['requested_amount'])
    
    print(f"\n==================== {req_id} ({user_id}) ====================")
    print(f"Request Date: {req['request_date']}, Deadline: {req['desired_completion_date']}")
    print(f"Start Bal: {start_bal}, Min Bal: {min_bal}, Req Amt: {req_amt}")
    print(f"GT Safe: {req['amount_safe_to_pay']}, GT Status: {req['affordability_status']}, GT Method: {req['recommended_payment_method']}")
    print(f"GT Changes: {req['spending_changes_needed']}")
    
    # Check running balance timeline
    tl = compute_running_balance(start_bal, req['request_date'], fc)
    min_bal_seen = min(b for _, b in tl)
    min_bal_date = [d for d, b in tl if b == min_bal_seen][0]
    print(f"No payment: Min balance in 90d = {min_bal_seen:.2f} on {min_bal_date.strftime('%Y-%m-%d')}")
    print(f"Headroom (Min bal - min_keep) = {min_bal_seen - min_bal:.2f}")
    
    # Check running balance with requested_amount payment on request_date
    tl_pay = compute_running_balance(start_bal, req['request_date'], fc, extra_payments=[(req['request_date'], req_amt)])
    min_bal_pay = min(b for _, b in tl_pay)
    min_bal_pay_date = [d for d, b in tl_pay if b == min_bal_pay][0]
    print(f"With req_amt payment: Min balance = {min_bal_pay:.2f} on {min_bal_pay_date.strftime('%Y-%m-%d')} (deficit: {min_bal - min_bal_pay:.2f})")
    
    # Check running balance with GT safe payment
    gt_safe = float(req['amount_safe_to_pay'])
    tl_gt = compute_running_balance(start_bal, req['request_date'], fc, extra_payments=[(req['request_date'], gt_safe)])
    min_bal_gt = min(b for _, b in tl_gt)
    min_bal_gt_date = [d for d, b in tl_gt if b == min_bal_gt][0]
    print(f"With GT safe payment: Min balance = {min_bal_gt:.2f} on {min_bal_gt_date.strftime('%Y-%m-%d')} (diff from min_keep: {min_bal_gt - min_bal:.2f})")

    # List events between request_date and min_bal_date
    print(f"\nEvents between {req['request_date'].strftime('%Y-%m-%d')} and {min_bal_date.strftime('%Y-%m-%d')}:")
    for fe in fc:
        if fe.date <= min_bal_date + pd.Timedelta(days=5):
            print(f"  {fe.date.strftime('%Y-%m-%d')}: {fe.direction:6s} {fe.amount:10.2f} [{fe.category:12s}] flex={fe.flexibility:10s} proj={str(fe.is_projected):5s} id={fe.event_id}")
