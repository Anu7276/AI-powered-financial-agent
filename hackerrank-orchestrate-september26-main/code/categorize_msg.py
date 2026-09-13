import pandas as pd
import re

msg = pd.read_csv(r'c:\Users\anura\Downloads\hackerrank-orchestrate-september26-main\hackerrank-orchestrate-september26-main\dataset\messages.csv')

categories = {
    'salary_amount': [],
    'salary_date': [],
    'seasonal_end': [],
    'bonus': [],
    'rent_change': [],
    'refund_pending': [],
    'unrealized_investment': [],
    'other': []
}

for idx, r in msg.iterrows():
    text = str(r['message_text'])
    lower = text.lower()
    
    if any(w in lower for w in ['seasonal contract has ended', 'kontrak musiman', 'no off-season income', 'akhir kontrak']):
        categories['seasonal_end'].append((r['message_id'], r['user_id'], text))
    elif any(w in lower for w in ['bonus']):
        categories['bonus'].append((r['message_id'], r['user_id'], text))
    elif any(w in lower for w in ['refund has been initiated', 'dana masih tertunda', 'refund is pending', 'pengembalian dana']):
        categories['refund_pending'].append((r['message_id'], r['user_id'], text))
    elif any(w in lower for w in ['portfolio', 'market value', 'nilai pasar', 'unrealized', 'belum terealisasi']):
        categories['unrealized_investment'].append((r['message_id'], r['user_id'], text))
    elif any(w in lower for w in ['rent', 'sewa', 'lease']):
        categories['rent_change'].append((r['message_id'], r['user_id'], text))
    elif any(w in lower for w in ['gaji', 'salary', 'pay', 'penggajian']):
        if any(w in lower for w in ['expected on', 'dijadwalkan', 'credit date', 'tanggal']):
            categories['salary_date'].append((r['message_id'], r['user_id'], text))
        else:
            categories['salary_amount'].append((r['message_id'], r['user_id'], text))
    else:
        categories['other'].append((r['message_id'], r['user_id'], text))

for k, v in categories.items():
    print(f"{k}: {len(v)} messages")

print("\n--- Samples from 'other' ---")
for mid, uid, t in categories['other'][:10]:
    print(f"[{mid}] ({uid}): {t[:100]}...")
