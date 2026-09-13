import sys
sys.path.insert(0, 'code')
from engine.ingest import DataStore
import engine.planner
from main import evaluate_against_sample, process_request

ds = DataStore()
samples = ds.sample_requests

# Let's inspect the margin for all 25 requests
for _, req in samples.iterrows():
    req_id = req['request_id']
    profile = ds.get_profile(req['user_id'])
    from main import load_image_facts, load_message_facts
    from engine.reconcile import reconcile_user_events
    from engine.forecast import build_forecast, compute_running_balance
    from engine.solver import compute_amount_safe_to_pay
    
    img_facts = load_image_facts(ds, req_id, req['user_id'])
    msg_facts = load_message_facts(ds, req['user_id'], req_id)
    entries, _ = reconcile_user_events(req['user_id'], ds, img_facts, msg_facts, req['request_date'])
    forecast = build_forecast(entries, msg_facts, profile, req['request_date'], 90)
    
    start_bal = float(profile['current_available_balance'])
    min_bal = float(profile['minimum_balance_to_keep'])
    req_amt = float(req['requested_amount'])
    
    tl = compute_running_balance(start_bal, req['request_date'], forecast)
    min_b = min(b for _, b in tl) if tl else start_bal
    headroom = min_b - min_bal
    margin = headroom - req_amt
    
    willing_stop = str(profile.get('expense_categories_user_is_willing_to_stop', ''))
    willing_red = str(profile.get('expense_categories_user_is_willing_to_reduce', ''))
    has_flex = (willing_stop != 'nan' and willing_stop != '') or (willing_red != 'nan' and willing_red != '')
    
    gt_changes = req['spending_changes_needed']
    print(f"{req_id:12} | req_amt={req_amt:10.2f} | headroom={headroom:10.2f} | margin={margin:10.2f} | has_flex={has_flex} | GT changes={gt_changes}")
