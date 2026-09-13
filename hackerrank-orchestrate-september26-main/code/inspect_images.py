import pandas as pd

img = pd.read_csv(r'c:\Users\anura\Downloads\hackerrank-orchestrate-september26-main\hackerrank-orchestrate-september26-main\dataset\images.csv')
ev = pd.read_csv(r'c:\Users\anura\Downloads\hackerrank-orchestrate-september26-main\hackerrank-orchestrate-september26-main\dataset\financial_events.csv')
prof = pd.read_csv(r'c:\Users\anura\Downloads\hackerrank-orchestrate-september26-main\hackerrank-orchestrate-september26-main\dataset\financial_profiles.csv')

m = img.merge(ev, left_on='related_event_id', right_on='event_id', how='left')
m = m.merge(prof[['user_id', 'home_currency']], left_on='user_id_x', right_on='user_id', how='left')
print(m[['image_id', 'user_id', 'request_id', 'related_event_id', 'description', 'category', 'amount', 'currency', 'home_currency']].to_string())
