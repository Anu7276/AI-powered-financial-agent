import sys
sys.path.insert(0, 'code')
from engine.ingest import DataStore
import pandas as pd
from engine.forecast import ForecastEntry, _detect_cadence, ONE_OFF_CATEGORIES, VARIABLE_RECURRING, MIN_OCCURRENCES_RECURRING, compute_running_balance
from engine.planner import _try_spending_changes, _build_payment_plan_str, PaymentPlan
from collections import defaultdict
import numpy as np

ds = DataStore()
samples = ds.sample_requests

# Test on request_06, request_11, request_21
for req_id in ['request_06', 'request_11', 'request_21']:
    req = samples[samples['request_id'] == req_id].iloc[0]
    user_id = req['user_id']
    profile = ds.get_profile(user_id)
    from main import load_image_facts, load_message_facts
    from engine.reconcile import reconcile_user_events
    
    img_facts = load_image_facts(ds, req_id, user_id)
    msg_facts = load_message_facts(ds, user_id, req_id)
    entries, _ = reconcile_user_events(user_id, ds, img_facts, msg_facts, req['request_date'])
    
    # Enhanced forecast with stream separation
    end_date = req['request_date'] + pd.Timedelta(days=90)
    forecast = []
    for e in entries:
        if e.date >= req['request_date'] and e.status in ('scheduled', 'pending'):
            forecast.append(ForecastEntry(
                date=e.date, direction=e.direction, amount=e.amount,
                category=e.category, flexibility=e.flexibility,
                minimum_allowed_amount=e.minimum_allowed_amount,
                event_id=e.event_id, is_projected=False, event_type=e.event_type
            ))
    historical = [e for e in entries if e.date <= req['request_date'] and e.status == 'settled']
    by_cat = defaultdict(list)
    for e in historical:
        stream = 'base' if ('base' in (e.description or '').lower() or e.date.day in (14, 15, 16)) else ('comm' if 'comm' in (e.description or '').lower() else '')
        by_cat[(e.category, e.direction, e.flexibility, stream)].append(e)

    for (cat, direction, flex, stream), group in by_cat.items():
        if cat in ONE_OFF_CATEGORIES and flex not in ('reducible', 'stoppable', 'reducible_or_stoppable'):
            continue
        if len(group) < MIN_OCCURRENCES_RECURRING:
            continue
        if cat == 'salary' and stream == 'comm' and any('komisi' in (m.message_text or '').lower() for m in ds.messages[ds.messages['user_id'] == user_id].itertuples()):
            continue
        sorted_group = sorted(group, key=lambda e: e.date)
        dates = [e.date for e in sorted_group]
        cadence = _detect_cadence(dates)
        amounts = [e.amount for e in sorted_group]
        if cat in VARIABLE_RECURRING:
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
            if proj_date >= req['request_date']:
                forecast.append(ForecastEntry(
                    date=proj_date, direction=direction, amount=proj_amount,
                    category=cat, flexibility=flex, minimum_allowed_amount=sorted_group[-1].minimum_allowed_amount,
                    event_id=sorted_group[-1].event_id, is_projected=True, event_type=sorted_group[-1].event_type
                ))
            step += 1

    # Salary change amendment
    for mf in msg_facts:
        if mf.fact_type == 'salary_change' and mf.new_amount:
            for fe in forecast:
                if fe.category == 'salary' and fe.direction == 'credit':
                    fe.amount = mf.new_amount

    forecast.sort(key=lambda fe: fe.date)
    start_bal = float(profile['current_available_balance'])
    min_bal = float(profile['minimum_balance_to_keep'])
    req_amt = float(req['requested_amount'])
    
    tl = compute_running_balance(start_bal, req['request_date'], forecast)
    min_b = min(b for _, b in tl) if tl else start_bal
    headroom = min_b - min_bal
    margin = headroom - req_amt
    print(f"\n{req_id}: req_amt={req_amt}, headroom={headroom:.2f}, margin={margin:.2f}")
    
    changes, _, safe_with = _try_spending_changes(
        entries, profile, req['request_date'], forecast, req_amt,
        req['desired_completion_date'], start_bal, min_bal, msg_facts
    )
    print(f"Changes found: {[c.change_type + ':' + c.event_id for c in changes]}")
    print(f"GT changes:    {req['spending_changes_needed']}")
