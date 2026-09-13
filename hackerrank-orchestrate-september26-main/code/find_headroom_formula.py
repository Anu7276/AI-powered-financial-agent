import sys
sys.path.insert(0, 'code')
from engine.ingest import DataStore
import pandas as pd
from engine.reconcile import reconcile_user_events
from main import load_image_facts, load_message_facts

ds = DataStore()
samples = ds.sample_requests

for req_id, expected_safe, req_amt in [
    ('request_06', 603.30, 620.40),
    ('request_11', 12510645.00, 13110000.00),
    ('request_21', 1543.35, 1574.40),
]:
    req = samples[samples['request_id'] == req_id].iloc[0]
    user_id = req['user_id']
    profile = ds.get_profile(user_id)
    start_bal = float(profile['current_available_balance'])
    min_bal = float(profile['minimum_balance_to_keep'])
    img_facts = load_image_facts(ds, req_id, user_id)
    msg_facts = load_message_facts(ds, user_id, req_id)
    entries, _ = reconcile_user_events(user_id, ds, img_facts, msg_facts, req['request_date'])
    
    print(f"\n*** {req_id} ({user_id}) ***")
    print(f"Start bal: {start_bal}, Min bal: {min_bal}")
    print(f"Target Safe: {expected_safe}, Req amt: {req_amt}")
    print(f"Start bal - min_bal: {start_bal - min_bal}")
    print(f"Required Outflow deduction to reach target safe: {(start_bal - min_bal) - expected_safe}")
    
    # Check all events for this user
    user_events = ds.events[ds.events['user_id'] == user_id].copy()
    user_events['event_date'] = pd.to_datetime(user_events['event_date'])
    # Historical recurring events
    hist = user_events[user_events['event_date'] <= req['request_date']]
    print(f"Hist events count: {len(hist)}")
    
    # Group historical by category
    print("Categories and last event:")
    for cat, grp in hist.groupby('category'):
        last_e = grp.sort_values('event_date').iloc[-1]
        print(f"  {cat:15s}: last={last_e['event_date'].strftime('%Y-%m-%d')}, amt={last_e['amount']}, flex={last_e['flexibility']}, min_amt={last_e['minimum_allowed_amount']}")
