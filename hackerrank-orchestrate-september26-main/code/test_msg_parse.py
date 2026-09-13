import pandas as pd
import re

msg = pd.read_csv(r'c:\Users\anura\Downloads\hackerrank-orchestrate-september26-main\hackerrank-orchestrate-september26-main\dataset\messages.csv')

def parse_message_deterministic(row) -> dict:
    text = str(row['message_text'])
    lower = text.lower()
    mid = row['message_id']
    uid = row['user_id']
    ev_ref = row.get('related_event_id') if not pd.isna(row.get('related_event_id')) else None
    
    # Defaults
    fact_type = "unknown"
    category = None
    new_amount = None
    effective_date = None
    confidence = 0.95
    
    # 1. Seasonal end
    if 'seasonal contract has ended' in lower or 'kontrak musiman' in lower or 'no off-season income' in lower:
        fact_type = "salary_seasonal_end"
        category = "salary"
        return dict(message_id=mid, fact_type=fact_type, category=category, new_amount=None, effective_date=None, confidence=confidence)
    
    # 2. Bonus pending
    if 'bonus' in lower and any(w in lower for w in ['pending', 'menunggu', 'unconfirmed', 'belum']):
        fact_type = "bonus_pending"
        category = "salary"
        return dict(message_id=mid, fact_type=fact_type, category=category, new_amount=None, effective_date=None, confidence=confidence)
        
    # 3. Pending gig / payouts (QuickCrew, RideGrid, TaskSprint, InvoiceFlow)
    if any(w in lower for w in ['payout is still pending', 'masih tertunda', 'faktur sebesar', 'tertunda']):
        fact_type = "payment_delayed"
        return dict(message_id=mid, fact_type=fact_type, category=None, new_amount=None, effective_date=None, confidence=confidence)
        
    # 4. Pending refund
    if 'refund' in lower or 'pengembalian dana' in lower:
        fact_type = "payment_delayed"
        category = "refund"
        return dict(message_id=mid, fact_type=fact_type, category=category, new_amount=None, effective_date=None, confidence=confidence)
        
    # 5. Unrealized investment
    if any(w in lower for w in ['unrealized', 'belum terealisasi', 'market value', 'nilai pasar']):
        fact_type = "unknown"
        category = "investment"
        return dict(message_id=mid, fact_type=fact_type, category=category, new_amount=None, effective_date=None, confidence=confidence)

    # 6. Internal account transfer
    if 'transfer between your accounts' in lower or 'transfer antar-rekening' in lower or 'transfer antar rekening' in lower:
        fact_type = "internal_transfer"
        return dict(message_id=mid, fact_type=fact_type, category=None, new_amount=None, effective_date=None, confidence=confidence)
        
    # 7. Rent change
    if any(w in lower for w in ['rent', 'sewa', 'lease']):
        category = "rent"
        fact_type = "rent_change"
        # look for amount or percentage
        m_amt = re.search(r'(?:inr|usd|eur|zar|idr|rs\.?|₹|\$)\s*([\d,]+(?:\.\d+)?)', text, re.I)
        if m_amt:
            new_amount = float(m_amt.group(1).replace(',', ''))
        m_date = re.search(r'\b(202\d-\d{2}-\d{2})\b', text)
        if m_date:
            effective_date = m_date.group(1)
        return dict(message_id=mid, fact_type=fact_type, category=category, new_amount=new_amount, effective_date=effective_date, confidence=confidence)
        
    # 8. Salary change / salary date change
    if any(w in lower for w in ['salary', 'gaji', 'pay', 'penggajian']):
        category = "salary"
        # Find date
        m_date = re.search(r'\b(202\d-\d{2}-\d{2})\b', text)
        if m_date:
            effective_date = m_date.group(1)
        # Find amount: e.g. "IDR 42750000", "EUR 1037.52", "EUR 1422.85", "EUR 1661"
        m_amt = re.search(r'(?:inr|usd|eur|zar|idr|rs\.?|₹|\$)\s*([\d,]+(?:\.\d+)?)', text, re.I)
        if m_amt:
            new_amount = float(m_amt.group(1).replace(',', ''))
            
        if any(w in lower for w in ['naik menjadi', 'reduced to', 'temporary monthly pay is', 'first salary will be', 'resumes on', 'berubah', 'penyesuaian', 'gaji pokok']):
            fact_type = "salary_change"
        elif any(w in lower for w in ['expected on', 'replaces the previous date', 'dijadwalkan', 'credit date']):
            fact_type = "salary_date_change"
        else:
            fact_type = "salary_confirmed"
            
        return dict(message_id=mid, fact_type=fact_type, category=category, new_amount=new_amount, effective_date=effective_date, confidence=confidence)
        
    return dict(message_id=mid, fact_type=fact_type, category=category, new_amount=new_amount, effective_date=effective_date, confidence=confidence)

parsed = [parse_message_deterministic(r) for _, r in msg.iterrows()]
df_p = pd.DataFrame(parsed)
print(df_p['fact_type'].value_counts())
print("\nSample parsed salary changes:")
print(df_p[df_p['fact_type']=='salary_change'][['message_id', 'new_amount', 'effective_date']].head(10))
print("\nSample parsed salary date changes:")
print(df_p[df_p['fact_type']=='salary_date_change'][['message_id', 'new_amount', 'effective_date']].head(10))
