import sys
sys.path.insert(0, 'code')
from engine.ingest import DataStore
import pandas as pd
from engine.forecast import build_forecast, compute_running_balance
from engine.solver import compute_amount_safe_to_pay, find_earliest_full_payment_date, is_safe
from engine.planner import _try_spending_changes, _build_payment_plan_str, PaymentPlan, _parse_pipe_field
from engine.reconcile import reconcile_user_events
from main import load_image_facts, load_message_facts

ds = DataStore()
samples = ds.sample_requests

# Let's inspect each of the 25 sample requests
print("Inspecting all 25 requests...")
all_matches = 0
for idx, req in samples.iterrows():
    req_id = req["request_id"]
    user_id = req["user_id"]
    profile = ds.get_profile(user_id)
    img_facts = load_image_facts(ds, req_id, user_id)
    msg_facts = load_message_facts(ds, user_id, req_id)
    entries, _ = reconcile_user_events(user_id, ds, img_facts, msg_facts, req["request_date"])
    fc = build_forecast(entries, msg_facts, profile, req["request_date"])
    
    start_bal = float(profile["current_available_balance"])
    min_bal = float(profile["minimum_balance_to_keep"])
    req_amt = float(req["requested_amount"])
    
    # Calculate base headroom
    tl = compute_running_balance(start_bal, req["request_date"], fc)
    min_b = min(b for _, b in tl) if tl else start_bal
    headroom = min_b - min_bal
    
    # Check pre-salary margin before next salary
    sal_dates = [fe.date for fe in fc if fe.category == "salary" and fe.direction == "credit"]
    next_sal_date = min(sal_dates) if sal_dates else req["request_date"] + pd.Timedelta(days=30)
    pre_sal_tl = [b for d, b in tl if d < next_sal_date]
    pre_sal_min = min(pre_sal_tl) if pre_sal_tl else start_bal
    pre_sal_margin = (pre_sal_min - min_bal) - req_amt
    
    print(f"{req_id} ({user_id}): req_amt={req_amt:10.2f}, headroom={headroom:10.2f}, pre_sal_margin={pre_sal_margin:10.2f}, GT_status={req['affordability_status']:20s}, GT_method={req['recommended_payment_method']:15s}, GT_changes={req['spending_changes_needed']}")
