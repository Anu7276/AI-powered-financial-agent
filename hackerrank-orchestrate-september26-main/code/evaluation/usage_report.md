# Token Usage Report — Financial Affordability Agent

## Final Full-Dataset Run

| Metric | Value |
|--------|-------|
| Total requests processed | 250 |
| Total elapsed time | 25.5s |
| Avg time per request | 0.10s |

## Model Usage

| Model | Provider | Usage |
|-------|----------|-------|
| claude-sonnet-4-5 | Anthropic | Image extraction + Message extraction |

## LLM Call Breakdown

| Call Type | Calls | Input Tokens | Output Tokens | Est. Cost (USD) |
|-----------|-------|-------------|---------------|-----------------|
| Image extraction (VLM) | 16 | 24,000 | 800 | $0.0840 |
| Message extraction (LLM) | 215 | 172,000 | 21,500 | $0.8385 |
| **Total** | **17** | **196,000** | **22,300** | **$0.9225** |

## Per-Request Averages

| Metric | Value |
|--------|-------|
| Avg input tokens / request | 784.0 |
| Avg output tokens / request | 89.2 |
| Avg total tokens / request | 873.2 |
| Avg cost / request (USD) | $0.003690 |

## Notes

- All LLM calls are cached on disk (`code/cache/`). Re-runs use the cache
  and incur zero additional API cost.
- The deterministic financial engine (solver, forecaster, planner, verifier)
  does not use LLM calls — all affordability decisions are computed in Python.
- LLM is used only for: (1) extracting amounts from financial images, and 
  (2) extracting structured facts from free-text messages.
- Image extraction: 16 total images with blank amounts in financial_events.csv.
- Message extraction: batched per user (one call per user that has messages).
- Model pricing: Claude Sonnet 4.5 at $3.00/M input, $15.00/M output tokens.
