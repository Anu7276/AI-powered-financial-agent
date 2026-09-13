import os
for k in ['ANTHROPIC_API_KEY', 'GEMINI_API_KEY', 'OPENAI_API_KEY', 'GOOGLE_API_KEY']:
    val = os.environ.get(k)
    status = f"SET (len {len(val)})" if val else "NOT SET"
    print(f"{k}: {status}")
