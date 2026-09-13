import sys
sys.path.insert(0, 'code')
from engine.ingest import DataStore
import pandas as pd
ds = DataStore()
for idx, r in ds.sample_requests.iterrows():
    u = r['user_id']
    p = ds.get_profile(u)
    stop = p.get('expense_categories_user_is_willing_to_stop')
    red = p.get('expense_categories_user_is_willing_to_reduce')
    if pd.notna(stop) or pd.notna(red):
        print(f"{r['request_id']} ({u}): stop={stop}, reduce={red}, GT_status={r['affordability_status']}, GT_changes={r['spending_changes_needed']}")
