# AI-Assisted, Evidence-Grounded Deterministic Financial Decision Engine

[![Benchmark](https://img.shields.io/badge/Affordability%20Status%20Accuracy-100%25%20(25%2F25)-brightgreen.svg)]()
[![Method Accuracy](https://img.shields.io/badge/Payment%20Method%20Accuracy-100%25%20(25%2F25)-brightgreen.svg)]()
[![Pipeline](https://img.shields.io/badge/Full%20Dataset%20Processed-250%2F250-blue.svg)]()
[![Schema Violations](https://img.shields.io/badge/Schema%20Violations-0-success.svg)]()

---

## 1. Problem

When a user asks **"Can I afford this laptop?"**, a simplistic check against their bank balance is dangerous. Real financial health depends on:
1. **Unsettled & Pending Transactions**: Pending card authorizations, outgoing checks, and delayed transfers that will reduce liquid capital.
2. **Fixed Living Commitments**: Scheduled rent, utility bills, loan amortizations, and recurring subscriptions that must never be missed.
3. **Safety Headroom**: The user's personalized `minimum_balance_to_keep` buffer, which must be protected on every single day over a 90-day forward horizon.
4. **Multimodal Evidence**: Important financial signals buried in unstructured media—such as salary slip images with missing amounts, employer Slack/SMS messages stating unconfirmed commissions or seasonal contract expirations, and delayed gig payouts.
5. **Personalized Preference Constraints**: Whether the user accepts installment options, partial payments, or is willing to stop/reduce specific lifestyle spending (e.g., dining, subscriptions).

The objective is to produce mathematically sound, personalized affordability decisions across 250 diverse requests spanning multiple global currencies (USD, EUR, INR, ZAR, IDR).

---

## 2. Solution

We implement an **AI-assisted, evidence-grounded deterministic financial decision engine**. 

Our core architectural principle is:
> **AI extracts and understands messy multimodal evidence; deterministic code executes financial calculations and makes decisions.**

```
Messy Information (Images, Text, Ledger)
       ↓
AI / VLM Extraction
       ↓
Multi-Channel Evidence Routing & Normalization
       ↓
Deterministic Cash-Flow Forecasting (90-Day Simulation)
       ↓
Constrained Optimization & Decision Ladder
       ↓
Deterministic Verification & Multi-Layer Audit Trace
       ↓
Evidence-Grounded Recommendation
```

The system guarantees that:
- No LLM ever guesses a balance or performs arithmetic.
- Every payment plan keeps running balances strictly above `minimum_balance_to_keep` for all 90 days.
- Every candidate is evaluated through an exhaustive audit trail ("why not" testing), recording exactly why alternatives were eliminated.
- Every recommendation is backed by a verifiable **Evidence Graph**.

---

## 3. Key Results

Evaluated against the official 25 golden sample requests (`dataset/sample_requests.csv`):

| Evaluation Metric | Score | Note |
|---|---|---|
| **Affordability Status Accuracy** | **100.0% (25/25)** | Perfect classification across all 4 statuses |
| **Payment Method Accuracy** | **100.0% (25/25)** | Perfect match across all 5 payment methods |
| **Median Absolute Headroom Error** | **427.78** | High median precision across global scales |
| **Median Relative Error %** | **4.67%** | Near-exact headroom preservation |
| **Non-IDR Headroom MAE (USD/EUR/INR)** | **4,332.46** | Extremely tight error margin on standard scales |
| **Non-IDR Median Absolute Error** | **85.89** | Sub-100 unit median error on USD, EUR, INR |
| **Full Dataset Processing** | **250/250 (100%)** | Processed in 123.7s (0.49s/request) |
| **Schema / Constraint Violations** | **0** | Verified via deterministic post-validator |

### Full Statistical Headroom Distribution
To ensure rigorous evaluation beyond aggregate medians:
- **Mean Absolute Error (MAE)**: 238,004.71 *(dominated by Indonesian Rupiah currency scale where 1 USD ≈ 15,000 IDR)*
- **Root Mean Square Error (RMSE)**: 800,213.25
- **95th Percentile Absolute Error (P95)**: 945,539.97
- **Max Absolute Error**: 3,804,865.54 *(single IDR outlier involving speculative commission variance)*
- **Exact Match % (Error ≤ 1.0)**: 12.0%

---

## 4. Architecture

The engine is structured into five distinct, specialized layers:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer 1 — Understanding: Claude 3.5 Sonnet VLM, Vision Parser, Regex   │
│ - Extracts missing amounts from invoice/receipt images                  │
│ - Identifies payroll dates, bonus delays, and seasonal contract notices │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────┐
│ Layer 2 — Retrieval: Multi-Channel Evidence Router (SQLite / DuckDB)    │
│ - Transaction RAG: In-memory relational queries on events               │
│ - Messages RAG: BM25/keyword retrieval for financial notifications     │
│ - Profile RAG: Extracts user boundaries, limits, and flexible flags    │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────┐
│ Layer 3 — Financial Intelligence: Multi-Feature Recurrence & Forecast  │
│ - 9-feature recurrence detection (merchant, stream, cadence, median)   │
│ - 90-day daily balance timeline simulation with running minimums        │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────┐
│ Layer 4 — Decision: Constrained Optimization & Decision Ladder          │
│ - Evaluates candidate actions: Full payment, Wait, Installments, Partial│
│ - Lifestyle change minimizer (tests 1 change before trying 2 or 3)     │
│ - Eliminates unsafe options via deterministic tie-break ranking         │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────┐
│ Layer 5 — Trust: Evidence Graph, Verifier & Audit Trail                 │
│ - "Why not" rejection audit trail for all eliminated payment candidates │
│ - Evidence Graph linking decisions to source events and messages        │
│ - Math and schema verifier enforcing strict challenge specifications    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Data Pipeline

1. **Ingestion (`engine/ingest.py`)**:
   - Reads `requests.csv`, `financial_profiles.csv`, `financial_events.csv`, `request_payment_options.csv`, `messages.csv`, `images.csv`, and `exchange_rates.csv`.
   - Normalizes currency codes, strips whitespace, and parses dates to `pd.Timestamp`.
2. **Reconciliation (`engine/reconcile.py`)**:
   - Reconstructs accurate balances by linking blank-amount events to OCR/VLM image facts.
   - De-duplicates double-reported transactions (e.g. transfer credit vs debit pairs).
   - Replaces unconfirmed or outdated scheduled records with verified message facts (e.g. rent increases or salary adjustments).

---

## 6. AI / RAG Layer

The AI/RAG layer handles unstructured data without touching numerical solvers:
- **Vision-Language Extraction (`engine/evidence.py`)**:
  Extracts missing transaction amounts and settlement dates from receipts and pay stubs using multimodal prompts with high confidence filters.
- **Evidence Router (`engine/router.py`)**:
  - **Transaction RAG**: In-memory SQLite engine enabling exact relational queries (`SELECT * FROM events WHERE user_id = ? AND category = 'salary'`).
  - **Messages RAG**: Keyword and semantic indexing targeting 5 financial signal domains: `salary`, `bonus`, `rent`, `delay`, and `termination`.
  - **Profile RAG**: Extracts risk thresholds, max installment horizons, and allowed payment methods.

---

## 7. Financial Reasoning Engine

The engine computes two fundamental numbers before generating payment plans:
1. **Safe Headroom (`amount_safe_to_pay`)**:
   $$H = \min_{t \in [t_{\text{req}}, t_{\text{req}} + 90]} \left( B(t) - M \right)$$
   Where $B(t)$ is the projected running available balance at day $t$, and $M$ is `minimum_balance_to_keep`. Clamped strictly to $[0, \text{requested\_amount}]$.
2. **Pre-Salary Buffer Check**:
   Evaluates liquidity prior to the next incoming paycheck. If paying today leaves a margin smaller than $\max(120, 0.08 \times \text{requested\_amount})$, the plan is flagged as tight, prioritizing conservative planning or spending adjustments.

---

## 8. Forecasting

Implemented in `engine/forecast.py`:
- **9-Feature Recurrence Engine**:
  Detects regular streams using merchant, category, direction, flexibility, stream type (base vs commission), interval cadence, day-of-month alignment, historical settled status, and description signals (filtering final severance or seasonal ends).
- **Multi-Feature Amount Estimation**:
  - Variable expenses (groceries, dining, utilities): uses rolling median of recent occurrences.
  - Step-up fixed commitments (rent increases): locks onto the latest scheduled amount.
  - Commission isolation: speculative unconfirmed commissions are excluded when messages indicate approvals are pending.
- **90-Day Daily Simulation**:
  Simulates account balances step-by-step for 90 days, accurately reflecting credit and debit settlement dates.

---

## 9. Payment Planning

Implemented in `engine/planner.py`:
Generates and ranks candidates across all 5 payment methods:
1. **Immediate Full Payment (`affordable_now`)**:
   Safe today with adequate pre-salary liquidity buffer.
2. **Installments (`affordable_with_plan`)**:
   Matches offered payment options in `request_payment_options.csv`. Checks installment count against user's `max_installment_months` and validates that every installment date maintains running balances above minimum keep.
3. **Partial Payment (`affordable_with_plan`)**:
   If allowed by request and accepted by user, pays `amount_safe_to_pay` on `request_date` and the remainder on `earliest_date_for_full_payment`, provided completion meets `desired_completion_date`.
4. **Wait (`affordable_later`)**:
   Scans the 90-day forecast to identify the earliest safe full-payment date. Chosen when waiting meets the deadline without lifestyle disruption.
5. **Spending Adjustments (`affordable_with_plan`)**:
   If waiting misses the deadline, the engine identifies flexible expenses (`stoppable` or `reducible`) matching user willingness. Applies a minimal-lifestyle-disruption search: testing if any single change achieves safety before attempting combinations.
6. **Not Affordable (`not_affordable` / `not_recommended`)**:
   Returned when no safe schedule exists within the 90-day horizon or constraints.

---

## 10. Verification

Implemented in `engine/verify.py`:
Every generated recommendation passes through a strict post-decision verifier before output:
- **Bound Enforcement**: Confirms $0 \le \text{amount\_safe\_to_pay} \le \text{requested\_amount}$.
- **Plan Schema Validity**: Validates `YYYY-MM-DD:amount` syntax and date monotonicity.
- **Option ID Consistency**: Verifies that installment plans correspond to existing offered payment options.
- **Flexibility Compliance**: Ensures modified expenses are marked reducible or stoppable and align with user profile preferences.
- **Status & Method Concordance**: Guarantees zero contradictions between status, method, plan, and date.

---

## 11. Evaluation

The system includes an automated evaluation harness (`code/main.py --sample --no-llm`):
- Compares predictions against all 25 ground-truth requests in `sample_requests.csv`.
- Computes comprehensive error distribution metrics (MAE, RMSE, Median AE, P95, Max AE, Relative Error %, Exact Match %).
- Produces **`evaluation/error_report.json`** tracking:
  - `current_balance`, `requested_amount`, `safe_amount`, `minimum_balance`
  - `future_income`, `future_expenses`, `recurring_events`, `payment_options`
  - `earliest_safe_date`, `chosen_method`, `expected_status`, `actual_status`
  - `status_match`, `method_match`
- Produces **`evaluation/decision_traces.json`** containing:
  - `"why not"` audit trail explaining why rejected alternatives were eliminated.
  - Detailed Evidence Graph traces linking decisions to binding constraints.

---

## 12. Results

Running the benchmark on the 25 sample requests yields:

```text
======================================================================
COMPREHENSIVE BENCHMARK EVALUATION RESULTS (25 samples)
======================================================================
  Affordability Status Accuracy: 25/25 = 100.0%
  Payment Method Accuracy:       25/25 = 100.0%
  Exact Match (Safe Amount):     3/25 = 12.0%
  -------------------------------------------------------------
  Headroom MAE:                  238,004.71
  Median Absolute Error:         427.78
  RMSE:                          800,213.25
  P95 Absolute Error:            945,539.97
  Max Absolute Error:            3,804,865.54
  Relative MAE %:                44.12%
  Median Relative Error %:       4.67%
  -------------------------------------------------------------
  Non-IDR Headroom MAE (USD/EUR/INR): 4,332.46
  Non-IDR Median AE:                  85.89
  IDR Headroom MAE (1 USD ~ 15k IDR): 1,172,693.72
======================================================================
```

Full production execution on all 250 requests:
- **Output Files Generated**: `output.csv` (root) and `dataset/output.csv`.
- **Row Count**: 250 data rows + 1 header row (exactly 8 required columns).
- **Status Distribution**:
  - `affordable_with_plan`: 82 requests (32.8%)
  - `not_affordable`: 68 requests (27.2%)
  - `affordable_later`: 57 requests (22.8%)
  - `affordable_now`: 43 requests (17.2%)
- **Method Distribution**:
  - `not_recommended`: 68 requests (27.2%)
  - `full_payment`: 67 requests (26.8%)
  - `wait`: 57 requests (22.8%)
  - `installments`: 48 requests (19.2%)
  - `partial_payment`: 10 requests (4.0%)

---

## 13. How to Run

### Prerequisites
- Python 3.10+
- Dependencies: `pandas`, `numpy`, `anthropic` (optional, for fresh LLM extraction)

```bash
pip install pandas numpy anthropic
```

### 1. Evaluate on 25 Sample Requests (Fast Deterministic Benchmark)
```bash
python code/main.py --sample --no-llm
```

### 2. Generate Full 250-Row Submission Predictions
```bash
python code/main.py --no-llm
```
This generates `output.csv` in the root repository and `dataset/output.csv`.

### 3. Debug a Single Request
```bash
python code/main.py --request request_11 --no-llm --verbose
```

---

## 14. Repository Structure

```text
.
├── output.csv                        # Final 250 predictions in root
├── README.md                         # Architecture, evaluation & documentation
├── problem_statement.md              # Official challenge statement
├── AGENTS.md                         # Guidelines for agentic pair-programming
│
├── dataset/                          # Input data
│   ├── requests.csv                  # 250 requests to evaluate
│   ├── sample_requests.csv           # 25 ground-truth benchmark samples
│   ├── financial_profiles.csv        # Balances, minimum keeps, preferences
│   ├── financial_events.csv          # Historical, pending, confirmed events
│   ├── request_payment_options.csv   # Installment offers per request
│   ├── messages.csv                  # Slack/SMS/email communications
│   ├── images.csv                    # OCR image references
│   ├── exchange_rates.csv            # Currency conversion rates
│   └── output.csv                    # Dataset output mirror
│
├── evaluation/                       # Audit & evaluation reports
│   ├── error_report.json             # Full per-case comparison & metrics
│   ├── decision_traces.json          # Evidence graph and why-not audit trail
│   └── usage_report.md               # Token usage & cost analysis
│
└── code/                             # Decision engine source code
    ├── main.py                       # Main CLI & batch execution pipeline
    ├── cache/                        # Extracted multimodal fact cache
    │   ├── image_extractions.json    # Cached VLM image extractions
    │   └── message_facts.json        # Cached NLP message extractions
    └── engine/
        ├── ingest.py                 # DataStore and CSV parser
        ├── router.py                 # Layer 2: Transaction RAG & Evidence Router
        ├── evidence.py               # Layer 1: VLM/LLM multimodal extraction
        ├── reconcile.py              # Event reconciliation & ledger reconstruction
        ├── forecast.py               # 9-feature recurrence & 90-day cash flow simulation
        ├── solver.py                 # Headroom formula & installment feasibility
        ├── planner.py                # Decision ladder & spending reduction optimizer
        ├── explain.py                # Human-readable rationale generation
        └── verify.py                 # Math, bound & schema validator
```

---

## 15. Limitations

1. **Static 90-Day Boundary**: The forecast considers events up to 90 days forward. Unscheduled financial events occurring beyond 90 days are not captured.
2. **Fixed Category Taxonomies**: Recurrence and flexibility rules rely on predefined category lists. Uncategorized one-off expenses with arbitrary descriptions are treated conservatively as non-recurring.
3. **Currency Conversion Scope**: Exchange rates are fixed to the provided dated rates in `exchange_rates.csv`; live forex market fluctuations are neither fetched nor modeled.
4. **Deterministic Preference Enforcement**: The engine respects explicit user profile willingness for stopping or reducing spending; it does not automatically impose aggressive budget austerity unless the user has opted into those specific categories.
****
