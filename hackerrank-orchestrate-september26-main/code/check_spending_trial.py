import sys
sys.path.insert(0, 'code')
from engine.ingest import DataStore
from test_v2_engine import build_forecast_v2
from engine.planner import _get_flexible_events, _parse_pipe_field, SpendingChange
from engine.forecast import compute_running_balance
from engine.solver import compute_amount_safe_to_pay, is_safe
from engine.reconcile import reconcile_user_events
from main import load_image_facts, load_message_facts
import pandas as pd

ds = DataStore()

def try_spending_changes_v3(entries, profile, request_date, forecast, requested_amount, desired_completion_date, start_balance, minimum_balance, message_facts, max_changes=3):
    flexible = _get_flexible_events(entries, profile, request_date)
    if not flexible:
        return [], forecast, compute_amount_safe_to_pay(start_balance, minimum_balance, requested_amount, request_date, forecast)

    willing_to_stop = _parse_pipe_field(profile.get("expense_categories_user_is_willing_to_stop"))
    willing_to_reduce = _parse_pipe_field(profile.get("expense_categories_user_is_willing_to_reduce"))

    exclude_ids = set()
    reduce_map = {}
    applied_changes = []

    # Prioritize subscription / fixed recurring commitments (streaming, cloud_storage, memberships)
    # over general shopping / dining
    def _flex_priority(e):
        cat = e.category
        if cat in ("streaming", "cloud_storage", "music_subscription", "delivery_membership"):
            return 0
        return 1

    flexible_sorted = sorted(flexible, key=_flex_priority)

    def _is_target_met(ex_ids, red_map):
        return is_safe(start_balance, minimum_balance, request_date, forecast, request_date, requested_amount, None, ex_ids, red_map)

    def _get_min_bal(ex_ids, red_map):
        tl = compute_running_balance(start_balance, request_date, forecast, exclude_event_ids=ex_ids, reduce_amounts=red_map)
        return min(b for _, b in tl) if tl else start_balance

    current_min_bal = _get_min_bal(exclude_ids, reduce_map)

    for e in flexible_sorted:
        if len(applied_changes) >= max_changes:
            break
        if e.event_id in exclude_ids or e.event_id in reduce_map:
            continue

        # If already safe for requested_amount with a safe buffer, stop!
        if _is_target_met(exclude_ids, reduce_map) and len(applied_changes) > 0:
            break

        # Try stop
        if e.flexibility in ("stoppable", "reducible_or_stoppable") and e.category in willing_to_stop:
            # If reducible_or_stoppable and category is ALSO in willing_to_reduce, check if reduction is sufficient!
            trial_exclude = exclude_ids | {e.event_id}
            new_min_bal = _get_min_bal(trial_exclude, reduce_map)
            
            # If user allows reduce and reduction alone is enough, prefer reduction
            if e.flexibility in ("reducible", "reducible_or_stoppable") and e.category in willing_to_reduce and e.minimum_allowed_amount is not None:
                trial_reduce = {**reduce_map, e.event_id: e.minimum_allowed_amount}
                if _is_target_met(exclude_ids, trial_reduce):
                    reduce_map[e.event_id] = e.minimum_allowed_amount
                    applied_changes.append(SpendingChange("reduce_to", e.event_id, e.minimum_allowed_amount))
                    if _is_target_met(exclude_ids, reduce_map):
                        break
                    continue
            
            if new_min_bal > current_min_bal:
                exclude_ids.add(e.event_id)
                current_min_bal = new_min_bal
                applied_changes.append(SpendingChange("stop", e.event_id))
                if _is_target_met(exclude_ids, reduce_map):
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
                    if _is_target_met(exclude_ids, reduce_map):
                        break

    final_safe = compute_amount_safe_to_pay(start_balance, minimum_balance, requested_amount, request_date, forecast, exclude_ids, reduce_map)
    return applied_changes, forecast, final_safe

for req_id in ['request_06', 'request_11', 'request_21']:
    req = ds.sample_requests[ds.sample_requests['request_id'] == req_id].iloc[0]
    user_id = req['user_id']
    profile = ds.get_profile(user_id)
    img_facts = load_image_facts(ds, req_id, user_id)
    msg_facts = load_message_facts(ds, user_id, req_id)
    entries, _ = reconcile_user_events(user_id, ds, img_facts, msg_facts, req['request_date'])
    fc = build_forecast_v2(entries, msg_facts, profile, req['request_date'])
    
    start_bal = float(profile['current_available_balance'])
    min_bal = float(profile['minimum_balance_to_keep'])
    req_amt = float(req['requested_amount'])
    
    changes, _, _ = try_spending_changes_v3(
        entries, profile, req['request_date'], fc, req_amt,
        req['desired_completion_date'], start_bal, min_bal, msg_facts
    )
    print(f"\n=== {req_id} ===")
    changes_str = [c.change_type + ':' + c.event_id + (f':{c.new_amount:.2f}' if c.change_type == 'reduce_to' else '') for c in changes]
    print(f"v3 changes: {'|'.join(changes_str)}")
    print(f"GT changes: {req['spending_changes_needed']}")
