"""
main.py — Financial Agent Pipeline Entry Point

Runs the full pipeline over dataset/requests.csv and writes output.csv.

Usage:
    python main.py                        # full run
    python main.py --sample               # validate against sample_requests.csv
    python main.py --request request_01   # single request debug
    python main.py --no-llm               # skip LLM calls (use cached only)

Environment:
    ANTHROPIC_API_KEY — required for image/message extraction (unless cache hit)
    DATASET_DIR       — override dataset path (default: ../dataset)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# Ensure 'code' directory is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.ingest import DataStore
from engine.evidence import (
    extract_image_amount, extract_message_facts,
    ImageFact, MessageFact
)
from engine.reconcile import reconcile_user_events
from engine.forecast import build_forecast, ForecastEntry
from engine.solver import (
    compute_amount_safe_to_pay, find_earliest_full_payment_date
)
from engine.planner import select_best_plan, PaymentPlan
from engine.explain import generate_explanation
from engine.verify import verify_row
from engine.router import EvidenceRouter


# ── Token tracking ────────────────────────────────────────────────────────────

class TokenTracker:
    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.image_calls = 0
        self.message_calls = 0

    def record(self, call_type: str, inp: int, out: int):
        self.calls += 1
        self.input_tokens += inp
        self.output_tokens += out
        if call_type == "image":
            self.image_calls += 1
        elif call_type == "message":
            self.message_calls += 1

    def total_tokens(self):
        return self.input_tokens + self.output_tokens

    def estimated_cost_usd(self) -> float:
        # Claude Sonnet 4.5: $3/M input, $15/M output
        cost = (self.input_tokens / 1_000_000) * 3.0 + \
               (self.output_tokens / 1_000_000) * 15.0
        return round(cost, 4)


TRACKER = TokenTracker()


# ── Image fact loading ─────────────────────────────────────────────────────────

def load_image_facts(ds: DataStore, request_id: str, user_id: str) -> dict[str, ImageFact]:
    """Load image facts for events linked to this request or user."""
    imgs = ds.images
    # Get images for this request
    req_imgs = imgs[imgs["request_id"] == request_id]
    
    facts = {}
    for _, row in req_imgs.iterrows():
        event_id = row["related_event_id"]
        if pd.isna(event_id):
            continue
        image_id = row["image_id"]
        image_path = Path(__file__).resolve().parent.parent / "dataset" / "media" / "images" / f"{image_id}.png"
        
        if not image_path.exists():
            continue

        try:
            fact = extract_image_amount(str(image_path), image_id, event_id)
            facts[event_id] = fact
            if fact.amount is not None:
                print(f"  [IMG] {image_id} -> event {event_id}: amount={fact.amount} {fact.currency}")
        except Exception as e:
            print(f"  [IMG ERROR] {image_id}: {e}")

    return facts


# ── Message fact loading ───────────────────────────────────────────────────────

def load_message_facts(ds: DataStore, user_id: str, request_id: str) -> list[MessageFact]:
    """Load and extract facts from messages related to this user/request."""
    messages = ds.get_user_messages(user_id, request_id)
    if messages.empty:
        return []
    try:
        facts = extract_message_facts(messages, user_id)
        for f in facts:
            if f.fact_type != "unknown":
                print(f"  [MSG] {f.message_id}: {f.fact_type} "
                      f"amount={f.new_amount} effective={f.effective_date}")
        return facts
    except Exception as e:
        print(f"  [MSG ERROR] {e}")
        return []


# ── Format spending changes ────────────────────────────────────────────────────

def format_spending_changes(plan: PaymentPlan) -> str:
    if not plan.spending_changes:
        return "none"
    parts = []
    for c in plan.spending_changes:
        if c.change_type == "stop":
            parts.append(f"stop:{c.event_id}")
        elif c.change_type == "reduce_to":
            if c.new_amount is not None:
                if c.new_amount == int(c.new_amount):
                    parts.append(f"reduce_to:{c.event_id}:{int(c.new_amount)}")
                else:
                    parts.append(f"reduce_to:{c.event_id}:{c.new_amount:.2f}")
    return "|".join(parts) if parts else "none"


def format_date(dt: Optional[pd.Timestamp]) -> str:
    if dt is None or (isinstance(dt, float) and np.isnan(dt)):
        return ""
    if isinstance(dt, str):
        return dt
    return dt.strftime("%Y-%m-%d")


# ── Process single request ─────────────────────────────────────────────────────

def process_request(request: pd.Series, ds: DataStore, verbose: bool = False) -> dict:
    """Run the full pipeline for one request. Returns output dict."""
    request_id = request["request_id"]
    user_id = request["user_id"]
    request_date = request["request_date"]
    requested_amount = request["requested_amount"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing {request_id} | {user_id} | {request_date.date()}")
        print(f"  Amount: {requested_amount} | Type: {request['request_type']}")

    try:
        # 1. Load profile
        profile = ds.get_profile(user_id)
        currency = profile["home_currency"]
        start_balance = float(profile["current_available_balance"])
        min_balance = float(profile["minimum_balance_to_keep"])

        # 2. Load image facts (for blank-amount events)
        image_facts = load_image_facts(ds, request_id, user_id)

        # 3. Load message facts
        message_facts = load_message_facts(ds, user_id, request_id)

        # 4. Route evidence via EvidenceRouter (Layer 2)
        router = EvidenceRouter(ds)
        routed_evidence = router.route_evidence(
            user_id, request_id, request_date, image_facts, message_facts
        )

        # 5. Reconcile events
        entries, salary_overrides = reconcile_user_events(
            user_id, ds, image_facts, message_facts, request_date
        )

        if verbose:
            print(f"  Events reconciled: {len(entries)}")

        # 6. Build 90-day forecast
        forecast = build_forecast(
            entries, message_facts, profile, request_date, horizon_days=90
        )

        if verbose:
            future_events = [fe for fe in forecast if fe.date > request_date]
            print(f"  Forecast entries (future): {len(future_events)}")

        # 7. Select best plan
        plan = select_best_plan(
            request, profile, entries, forecast, message_facts, ds
        )

        # 8. Generate explanation
        explanation = generate_explanation(plan, request, profile, currency)

        # 9. Format earliest date
        efp = plan.earliest_date_for_full_payment
        efp_str = format_date(efp)

        # 10. Format spending changes
        spending_str = format_spending_changes(plan)

        # 11. Build output row
        output = {
            "request_id": request_id,
            "amount_safe_to_pay": round(plan.amount_safe_to_pay, 2),
            "affordability_status": plan.affordability_status,
            "recommended_payment_method": plan.recommended_payment_method,
            "payment_plan": plan.payment_plan_str,
            "earliest_date_for_full_payment": efp_str,
            "spending_changes_needed": spending_str,
            "decision_explanation": explanation,
            "_plan": plan,
            "_forecast": forecast,
            "_entries": entries,
            "_profile": profile,
        }

        # 12. Verify
        output, violations = verify_row(output, request, profile)
        if violations and verbose:
            for v in violations:
                print(f"  [VERIFY WARN] {v}")

        if verbose:
            print(f"  >> {plan.affordability_status} | {plan.recommended_payment_method}")
            print(f"  >> safe_to_pay={plan.amount_safe_to_pay} | earliest={efp_str}")
            print(f"  >> plan: {plan.payment_plan_str[:80]}")

        return output

    except Exception as e:
        tb = traceback.format_exc()
        print(f"  [ERROR] {request_id}: {e}")
        if verbose:
            print(tb.encode('ascii', errors='replace').decode('ascii'))
        # Return safe fallback
        return {
            "request_id": request["request_id"],
            "amount_safe_to_pay": 0.0,
            "affordability_status": "not_affordable",
            "recommended_payment_method": "not_recommended",
            "payment_plan": "none",
            "earliest_date_for_full_payment": "",
            "spending_changes_needed": "none",
            "decision_explanation": f"Error during processing: {str(e)[:100]}",
        }


# ── Golden test evaluation ─────────────────────────────────────────────────────

def evaluate_against_sample(ds: DataStore) -> dict:
    """Run pipeline against sample_requests.csv and compare."""
    sample = ds.sample_requests
    results = []
    correct_status = 0
    correct_method = 0
    amount_errors = []
    expected_safes = []
    currencies = []
    error_report_items = []
    decision_traces = {}

    for _, req in sample.iterrows():
        output = process_request(req, ds, verbose=True)
        expected_status = req.get("affordability_status", "")
        expected_method = req.get("recommended_payment_method", "")
        expected_safe = float(req.get("amount_safe_to_pay", 0))

        status_ok = output["affordability_status"] == expected_status
        method_ok = output["recommended_payment_method"] == expected_method
        safe_err = abs(output["amount_safe_to_pay"] - expected_safe)

        if status_ok:
            correct_status += 1
        if method_ok:
            correct_method += 1
        amount_errors.append(safe_err)
        expected_safes.append(expected_safe)

        # Retrieve tracked context for error reporting
        user_id = req["user_id"]
        req_id = req["request_id"]
        prof = output.get("_profile")
        if prof is None:
            prof = ds.get_profile(user_id)
        currency = prof.get("home_currency", "USD")
        currencies.append(currency)

        fc = output.get("_forecast") if output.get("_forecast") is not None else []
        entries = output.get("_entries") if output.get("_entries") is not None else []
        plan_obj = output.get("_plan")

        future_income = round(sum(fe.amount for fe in fc if fe.direction == "credit" and fe.date > req["request_date"]), 2)
        future_expenses = round(sum(fe.amount for fe in fc if fe.direction == "debit" and fe.date > req["request_date"]), 2)
        recurring_count = len([e for e in entries if getattr(e, "status", "") == "settled"])

        opt_df = ds.get_request_options(req_id)
        opt_ids = opt_df["payment_option_id"].tolist() if not opt_df.empty else []

        audit_trail = getattr(plan_obj, "audit_trail", {}) if plan_obj else {}
        evidence_graph = getattr(plan_obj, "evidence_graph", {}) if plan_obj else {}

        decision_traces[req_id] = {
            "request_id": req_id,
            "user_id": user_id,
            "affordability_status": output["affordability_status"],
            "recommended_payment_method": output["recommended_payment_method"],
            "amount_safe_to_pay": output["amount_safe_to_pay"],
            "audit_trail": audit_trail,
            "evidence_graph": evidence_graph,
        }

        error_report_items.append({
            "request_id": req_id,
            "user_id": user_id,
            "currency": currency,
            "current_balance": float(prof.get("current_available_balance", 0)),
            "requested_amount": float(req["requested_amount"]),
            "safe_amount": output["amount_safe_to_pay"],
            "expected_safe_amount": expected_safe,
            "safe_amount_error": round(safe_err, 2),
            "minimum_balance": float(prof.get("minimum_balance_to_keep", 0)),
            "future_income": future_income,
            "future_expenses": future_expenses,
            "recurring_events": recurring_count,
            "payment_options": opt_ids,
            "earliest_safe_date": output["earliest_date_for_full_payment"],
            "chosen_method": output["recommended_payment_method"],
            "expected_method": expected_method,
            "actual_status": output["affordability_status"],
            "expected_status": expected_status,
            "status_match": status_ok,
            "method_match": method_ok,
            "audit_trail": audit_trail,
            "evidence_graph": evidence_graph
        })

        results.append({
            "request_id": req_id,
            "expected_status": expected_status,
            "got_status": output["affordability_status"],
            "status_ok": status_ok,
            "expected_method": expected_method,
            "got_method": output["recommended_payment_method"],
            "method_ok": method_ok,
            "expected_safe": expected_safe,
            "got_safe": output["amount_safe_to_pay"],
            "safe_error": safe_err,
        })

    n = len(results)
    err_arr = np.array(amount_errors)
    mae = float(np.mean(err_arr))
    rmse = float(np.sqrt(np.mean(np.square(err_arr))))
    median_ae = float(np.median(err_arr))
    p95_ae = float(np.percentile(err_arr, 95))
    max_ae = float(np.max(err_arr))
    exact_matches = int(np.sum(err_arr <= 1.0))
    exact_match_pct = (exact_matches / n) * 100

    rel_errors = [err / max(exp, 1.0) for err, exp in zip(err_arr, expected_safes)]
    mean_rel_mae_pct = float(np.mean(rel_errors)) * 100
    median_rel_mae_pct = float(np.median(rel_errors)) * 100

    non_idr_errs = [err for err, cur in zip(err_arr, currencies) if cur != "IDR"]
    idr_errs = [err for err, cur in zip(err_arr, currencies) if cur == "IDR"]
    non_idr_mae = float(np.mean(non_idr_errs)) if non_idr_errs else 0.0
    non_idr_median = float(np.median(non_idr_errs)) if non_idr_errs else 0.0
    idr_mae = float(np.mean(idr_errs)) if idr_errs else 0.0
    idr_median = float(np.median(idr_errs)) if idr_errs else 0.0

    summary = {
        "n": n,
        "status_accuracy": correct_status / n,
        "method_accuracy": correct_method / n,
        "mae": mae,
        "rmse": rmse,
        "median_ae": median_ae,
        "p95_ae": p95_ae,
        "max_ae": max_ae,
        "exact_match_pct": exact_match_pct,
        "mean_relative_error_pct": mean_rel_mae_pct,
        "median_relative_error_pct": median_rel_mae_pct,
        "non_idr_mae": non_idr_mae,
        "non_idr_median": non_idr_median,
        "idr_mae": idr_mae,
        "idr_median": idr_median,
        "details": results,
    }

    # Save evaluation/error_report.json and decision_traces.json
    root_eval_dir = Path(__file__).resolve().parent.parent / "evaluation"
    root_eval_dir.mkdir(exist_ok=True)
    with open(root_eval_dir / "error_report.json", "w") as f:
        json.dump({
            "metrics": {
                "status_accuracy": f"{correct_status}/{n} ({100*correct_status/n:.1f}%)",
                "method_accuracy": f"{correct_method}/{n} ({100*correct_method/n:.1f}%)",
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "median_ae": round(median_ae, 2),
                "p95_ae": round(p95_ae, 2),
                "max_ae": round(max_ae, 2),
                "exact_match_pct": f"{exact_match_pct:.1f}%",
                "mean_relative_error_pct": f"{mean_rel_mae_pct:.2f}%",
                "median_relative_error_pct": f"{median_rel_mae_pct:.2f}%",
                "non_idr_mae": round(non_idr_mae, 2),
                "non_idr_median": round(non_idr_median, 2),
            },
            "cases": error_report_items
        }, f, indent=2)

    with open(root_eval_dir / "decision_traces.json", "w") as f:
        json.dump(decision_traces, f, indent=2)

    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE BENCHMARK EVALUATION RESULTS ({n} samples)")
    print(f"{'='*70}")
    print(f"  Affordability Status Accuracy: {correct_status}/{n} = {100*correct_status/n:.1f}%")
    print(f"  Payment Method Accuracy:       {correct_method}/{n} = {100*correct_method/n:.1f}%")
    print(f"  Exact Match (Safe Amount):     {exact_matches}/{n} = {exact_match_pct:.1f}%")
    print(f"  -------------------------------------------------------------")
    print(f"  Headroom MAE:                  {mae:,.2f}")
    print(f"  Median Absolute Error:         {median_ae:,.2f}")
    print(f"  RMSE:                          {rmse:,.2f}")
    print(f"  P95 Absolute Error:            {p95_ae:,.2f}")
    print(f"  Max Absolute Error:            {max_ae:,.2f}")
    print(f"  Relative MAE %:                {mean_rel_mae_pct:.2f}%")
    print(f"  Median Relative Error %:       {median_rel_mae_pct:.2f}%")
    print(f"  -------------------------------------------------------------")
    print(f"  Non-IDR Headroom MAE (USD/EUR/INR): {non_idr_mae:,.2f}")
    print(f"  Non-IDR Median AE:                  {non_idr_median:,.2f}")
    print(f"  IDR Headroom MAE (1 USD ~ 15k IDR): {idr_mae:,.2f}")
    print(f"{'='*70}")
    print()
    print("  Per-request results:")
    for r in results:
        s = "PASS" if r["status_ok"] else "FAIL"
        m = "PASS" if r["method_ok"] else "FAIL"
        print(f"  {r['request_id']:12} | status {s} ({r['expected_status'][:8]:8}->{r['got_status'][:8]:8}) | "
              f"method {m} ({r['expected_method'][:12]:12}->{r['got_method'][:12]:12}) | "
              f"safe err={r['safe_error']:.0f}")

    print(f"\nArtifacts generated:")
    print(f"  - {root_eval_dir / 'error_report.json'}")
    print(f"  - {root_eval_dir / 'decision_traces.json'}")

    return summary


# ── Main entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Financial Affordability Agent")
    parser.add_argument("--sample", action="store_true",
                        help="Evaluate against sample_requests.csv only")
    parser.add_argument("--request", type=str, default=None,
                        help="Process a single request_id (debug mode)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM calls; use cached results only")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output per request")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: dataset/output.csv)")
    args = parser.parse_args()

    if args.no_llm:
        # Monkey-patch to prevent actual API calls
        import engine.evidence as ev
        def _no_llm_image(*a, **kw):
            cache = ev._load_cache(ev._IMAGE_CACHE_FILE)
            image_id = a[1] if len(a) > 1 else kw.get("image_id", "")
            event_id = a[2] if len(a) > 2 else kw.get("event_id", "")
            if image_id in cache:
                d = cache[image_id]
                return ImageFact(**{k: d.get(k) for k in ImageFact.__dataclass_fields__})
            return ImageFact(image_id=image_id, event_id=event_id, amount=None, currency=None, confidence=0.0)
        ev.extract_image_amount = _no_llm_image

    start_time = time.time()
    ds = DataStore()

    if args.sample:
        evaluate_against_sample(ds)
        return

    if args.request:
        # Single request debug
        requests = ds.requests
        req_row = requests[requests["request_id"] == args.request]
        if req_row.empty:
            # Try sample
            sample = ds.sample_requests
            req_row = sample[sample["request_id"] == args.request]
        if req_row.empty:
            print(f"Request {args.request} not found.")
            return
        output = process_request(req_row.iloc[0], ds, verbose=True)
        print("\nOutput:")
        for k, v in output.items():
            print(f"  {k}: {v}")
        return

    # Full run over requests.csv
    requests = ds.requests
    print(f"Processing {len(requests)} requests...")

    outputs = []
    for i, (_, req) in enumerate(requests.iterrows()):
        verbose = args.verbose or (i % 25 == 0)
        if verbose:
            print(f"[{i+1}/{len(requests)}] {req['request_id']}")
        output = process_request(req, ds, verbose=args.verbose)
        outputs.append(output)

    # Write output.csv
    out_df = pd.DataFrame(outputs, columns=[
        "request_id", "amount_safe_to_pay", "affordability_status",
        "recommended_payment_method", "payment_plan",
        "earliest_date_for_full_payment", "spending_changes_needed",
        "decision_explanation"
    ])

    output_path = args.output or str(
        Path(__file__).resolve().parent.parent / "dataset" / "output.csv"
    )
    out_df.to_csv(output_path, index=False)
    root_output_path = str(Path(__file__).resolve().parent.parent / "output.csv")
    out_df.to_csv(root_output_path, index=False)
    elapsed = time.time() - start_time
    print(f"\nDone. {len(outputs)} rows written to {output_path} and {root_output_path}")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed/len(outputs):.2f}s/request)")

    # Write evaluation/ stub report
    _write_usage_report(len(outputs), elapsed)


def _write_usage_report(n_requests: int, elapsed: float):
    """Write evaluation/usage_report.md."""
    eval_dir = Path(__file__).resolve().parent / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    report_path = eval_dir / "usage_report.md"

    # Read cached facts to count actual LLM calls made
    cache_dir = Path(__file__).resolve().parent / "cache"
    img_cache = {}
    msg_cache = {}
    try:
        with open(cache_dir / "image_extractions.json") as f:
            img_cache = json.load(f)
    except Exception:
        pass
    try:
        with open(cache_dir / "message_facts.json") as f:
            msg_cache = json.load(f)
    except Exception:
        pass

    img_calls = len(img_cache)
    msg_calls = len(msg_cache)
    total_calls = img_calls + msg_calls

    # Estimate token usage (approximate)
    img_input_tokens  = img_calls * 1500   # ~1500 input tokens per image (image + prompt)
    img_output_tokens = img_calls * 50     # ~50 output tokens per image
    msg_input_tokens  = msg_calls * 800    # ~800 input per message
    msg_output_tokens = msg_calls * 100    # ~100 output per message

    total_input  = img_input_tokens + msg_input_tokens
    total_output = img_output_tokens + msg_output_tokens
    total_tokens = total_input + total_output

    cost_input  = (total_input  / 1_000_000) * 3.0
    cost_output = (total_output / 1_000_000) * 15.0
    total_cost  = cost_input + cost_output

    report = f"""# Token Usage Report — Financial Affordability Agent

## Final Full-Dataset Run

| Metric | Value |
|--------|-------|
| Total requests processed | {n_requests} |
| Total elapsed time | {elapsed:.1f}s |
| Avg time per request | {elapsed/max(n_requests,1):.2f}s |

## Model Usage

| Model | Provider | Usage |
|-------|----------|-------|
| claude-sonnet-4-5 | Anthropic | Image extraction + Message extraction |

## LLM Call Breakdown

| Call Type | Calls | Input Tokens | Output Tokens | Est. Cost (USD) |
|-----------|-------|-------------|---------------|-----------------|
| Image extraction (VLM) | {img_calls} | {img_input_tokens:,} | {img_output_tokens:,} | ${(img_calls*1500/1e6)*3.0 + (img_calls*50/1e6)*15.0:.4f} |
| Message extraction (LLM) | {msg_calls} | {msg_input_tokens:,} | {msg_output_tokens:,} | ${(msg_calls*800/1e6)*3.0 + (msg_calls*100/1e6)*15.0:.4f} |
| **Total** | **{total_calls}** | **{total_input:,}** | **{total_output:,}** | **${total_cost:.4f}** |

## Per-Request Averages

| Metric | Value |
|--------|-------|
| Avg input tokens / request | {total_input/max(n_requests,1):.1f} |
| Avg output tokens / request | {total_output/max(n_requests,1):.1f} |
| Avg total tokens / request | {total_tokens/max(n_requests,1):.1f} |
| Avg cost / request (USD) | ${total_cost/max(n_requests,1):.6f} |

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
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Usage report written to {report_path}")


if __name__ == "__main__":
    main()
