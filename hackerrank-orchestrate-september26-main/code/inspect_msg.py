import pandas as pd

msg = pd.read_csv(r'c:\Users\anura\Downloads\hackerrank-orchestrate-september26-main\hackerrank-orchestrate-september26-main\dataset\messages.csv')
print("Total messages:", len(msg))
print("Message columns:", msg.columns.tolist())
print("\nSource types:")
print(msg['source_type'].value_counts())

print("\nSample messages:")
for i in range(15):
    r = msg.iloc[i]
    print(f"[{r['message_id']}] user={r['user_id']} req={r.get('request_id')} ev={r.get('related_event_id')}")
    print(f"  type: {r['source_type']}")
    print(f"  text: {str(r['message_text'])[:120]}...")
