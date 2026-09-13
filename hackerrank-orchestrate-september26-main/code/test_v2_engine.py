import sys
sys.path.insert(0, 'code')
from engine.ingest import DataStore
import pandas as pd
from collections import defaultdict
import numpy as np
from engine.forecast import _detect_cadence, ForecastEntry, compute_running_balance, ONE_OFF_CATEGORIES, VARIABLE_RECURRING
from engine.solver import compute_amount_safe_to_pay, find_earliest_full_payment_date, is_safe, compute_installment_plan_safety
from engine.planner import _try_spending_changes, _build_payment_plan_str, PaymentPlan, _parse_pipe_field
from engine.reconcile import reconcile_user_events
from main import load_image_facts, load_message_facts

ds = DataStore()
samples = ds.sample_requests

def build_forecast_v2(entries, message_facts, profile, request_date, horizon_days=90):
    end_date = request_date + pd.Timedelta(days=horizon_days)
    forecast = []
    
    # 1. Confirmed future events
    for e in entries:
        if e.date >= request_date and e.status == "scheduled":
            forecast.append(ForecastEntry(
                date=e.date, direction=e.direction, amount=e.amount,
                category=e.category, flexibility=e.flexibility,
                minimum_allowed_amount=e.minimum_allowed_amount,
                event_id=e.event_id, is_projected=False, event_type=e.event_type
            ))
        elif e.date >= request_date and e.status == "pending" and e.direction == "debit":
            forecast.append(ForecastEntry(
                date=e.date, direction=e.direction, amount=e.amount,
                category=e.category, flexibility=e.flexibility,
                minimum_allowed_amount=e.minimum_allowed_amount,
                event_id=e.event_id, is_projected=False, event_type=e.event_type
            ))

    # 2. Historical settled
    historical = [e for e in entries if e.date <= request_date and e.status == "settled"]
    
    # Message checks for salary / commission
    user_id = profile["user_id"]
    user_msgs = ds.messages[ds.messages["user_id"] == user_id]
    commission_cancelled = any(
        any(w in (m.message_text or "").lower() for w in ["komisi belum", "commission unconfirmed", "commission not approved", "komisi ditunda", "belum disetujui"])
        for m in user_msgs.itertuples()
    )

    # Group historical entries: for salary, separate base from commission by description/day
    by_cat = defaultdict(list)
    for e in historical:
        cat = e.category
        desc = (getattr(e, "description", "") or "").lower()
        stream = ""
        if cat == "salary":
            if any(w in desc for w in ["commission", "komisi", "bonus"]):
                stream = "commission"
            else:
                stream = "base"
        by_cat[(cat, e.direction, e.flexibility, stream)].append(e)

    scheduled_salaries = [e for e in entries if e.category == "salary" and e.direction == "credit" and e.status == "scheduled"]

    for (cat, direction, flex, stream), group in by_cat.items():
        # Only skip genuine one-offs if NOT flexible/reducible
        if cat in ONE_OFF_CATEGORIES and flex not in ("reducible", "stoppable", "reducible_or_stoppable"):
            continue
        if len(group) < 2:
            continue
        # Skip commission stream if message cancelled/unconfirmed
        if cat == "salary" and stream == "commission" and commission_cancelled:
            continue

        sorted_group = sorted(group, key=lambda e: e.date)
        dates = [e.date for e in sorted_group]
        cadence = _detect_cadence(dates)
        
        amounts = [e.amount for e in sorted_group]
        if cat in VARIABLE_RECURRING or cat == "dining":
            recent = amounts[-6:] if len(amounts) >= 6 else amounts
            proj_amount = float(np.median(recent))
        else:
            proj_amount = sorted_group[-1].amount

        # Salary special handling
        if cat == "salary" and direction == "credit":
            if scheduled_salaries and stream != "commission":
                next_sal = sorted(scheduled_salaries, key=lambda x: x.date)[0]
                proj_amount = next_sal.amount
                last_date = next_sal.date
                step = 1
                while True:
                    proj_date = last_date + pd.DateOffset(months=step)
                    if proj_date > end_date:
                        break
                    forecast.append(ForecastEntry(
                        date=proj_date, direction="credit", amount=proj_amount,
                        category="salary", flexibility="fixed", minimum_allowed_amount=None,
                        event_id=None, is_projected=True, event_type="income"
                    ))
                    step += 1
                continue
            else:
                last_desc = (getattr(sorted_group[-1], "description", "") or "").lower()
                if any(w in last_desc for w in ["final", "severance", "terminated", "ended"]):
                    continue
                # For base salary without scheduled next, project monthly from last date
                cadence = 30

        last_date = sorted_group[-1].date
        source_event_id = sorted_group[-1].event_id
        step = 1
        is_monthly = cadence in (28, 29, 30, 31)

        while True:
            if is_monthly:
                proj_date = last_date + pd.DateOffset(months=step)
            else:
                proj_date = last_date + pd.Timedelta(days=cadence * step)
            if proj_date > end_date:
                break
            if proj_date >= request_date:
                forecast.append(ForecastEntry(
                    date=proj_date, direction=direction, amount=proj_amount,
                    category=cat, flexibility=flex, minimum_allowed_amount=sorted_group[-1].minimum_allowed_amount,
                    event_id=source_event_id, is_projected=True, event_type=sorted_group[-1].event_type
                ))
            step += 1

    # Single salary history + scheduled next
    sal_hist_count = len(by_cat.get(("salary", "credit", "fixed", "base"), []))
    if sal_hist_count < 2 and scheduled_salaries:
        next_sal = sorted(scheduled_salaries, key=lambda x: x.date)[0]
        step = 1
        while True:
            proj_date = next_sal.date + pd.DateOffset(months=step)
            if proj_date > end_date: break
            forecast.append(ForecastEntry(
                date=proj_date, direction="credit", amount=next_sal.amount,
                category="salary", flexibility="fixed", minimum_allowed_amount=None,
                event_id=None, is_projected=True, event_type="income"
            ))
            step += 1

    # Message facts
    for mf in message_facts:
        if mf.fact_type == "salary_seasonal_end" and mf.confidence >= 0.7:
            forecast = [fe for fe in forecast if not (fe.is_projected and fe.category == "salary")]
        elif mf.fact_type == "salary_change" and mf.confidence >= 0.7 and mf.new_amount:
            eff = pd.Timestamp(mf.effective_date) if mf.effective_date else request_date
            for fe in forecast:
                if fe.category == "salary" and fe.direction == "credit" and fe.date >= eff:
                    fe.amount = mf.new_amount

    forecast = [fe for fe in forecast if request_date <= fe.date <= end_date]
    forecast.sort(key=lambda fe: fe.date)
    return forecast

def select_best_plan_v2(request, profile, entries, forecast, message_facts, ds):
    user_id = request["user_id"]
    request_id = request["request_id"]
    request_date = request["request_date"]
    requested_amount = float(request["requested_amount"])
    desired_completion_date = request["desired_completion_date"]
    allows_partial = bool(request["allows_partial_payment"]) if isinstance(request["allows_partial_payment"], bool) else False

    start_balance = float(profile["current_available_balance"])
    minimum_balance = float(profile["minimum_balance_to_keep"])
    max_installment_months = int(profile["max_installment_months"]) if pd.notna(profile.get("max_installment_months")) else None
    payment_methods = _parse_pipe_field(profile.get("payment_methods_user_will_consider"))

    willing_to_stop = _parse_pipe_field(profile.get("expense_categories_user_is_willing_to_stop"))
    willing_to_reduce = _parse_pipe_field(profile.get("expense_categories_user_is_willing_to_reduce"))
    has_flexible_prefs = bool(willing_to_stop or willing_to_reduce)

    amount_safe = compute_amount_safe_to_pay(start_balance, minimum_balance, requested_amount, request_date, forecast)
    earliest_full_date = find_earliest_full_payment_date(start_balance, minimum_balance, requested_amount, request_date, forecast, 90, desired_completion_date)

    # Compute pre-salary margin
    sal_dates = [fe.date for fe in forecast if fe.category == "salary" and fe.direction == "credit"]
    next_sal_date = min(sal_dates) if sal_dates else request_date + pd.Timedelta(days=30)
    tl = compute_running_balance(start_balance, request_date, forecast)
    pre_sal_tl = [b for d, b in tl if d < next_sal_date]
    pre_sal_min = min(pre_sal_tl) if pre_sal_tl else start_balance
    pre_sal_margin = (pre_sal_min - minimum_balance) - requested_amount

    # Check if paying in full right now is tight (< 120 buffer or < 8% buffer) or unsafe
    is_tight = (pre_sal_margin < max(120.0, 0.08 * requested_amount)) or (amount_safe < requested_amount)

    candidates = []

    # Check full payment with spending changes if tight and user has flexible prefs
    spending_changes_plan = None
    if has_flexible_prefs and is_tight and "full_payment" in payment_methods:
        changes, mod_fc, safe_with = _try_spending_changes(
            entries, profile, request_date, forecast, requested_amount,
            desired_completion_date, start_balance, minimum_balance, message_facts
        )
        if changes:
            exclude_ids = {c.event_id for c in changes if c.change_type == "stop"}
            reduce_map = {c.event_id: c.new_amount for c in changes if c.change_type == "reduce_to"}
            if is_safe(start_balance, minimum_balance, request_date, forecast, request_date, requested_amount, None, exclude_ids, reduce_map):
                plan_str = _build_payment_plan_str([(request_date, requested_amount)])
                # amount_safe_to_pay reports base safe amount before changes
                spending_changes_plan = PaymentPlan(
                    affordability_status="affordable_with_plan",
                    recommended_payment_method="full_payment",
                    payment_plan_str=plan_str,
                    earliest_date_for_full_payment=earliest_full_date,
                    spending_changes=changes,
                    amount_safe_to_pay=amount_safe,
                    total_paid=requested_amount,
                    num_payments=1,
                    first_payment_date=request_date,
                    payment_option_id=None,
                    completes_by_deadline=True,
                )

    if spending_changes_plan is not None:
        candidates.append(spending_changes_plan)
    elif "full_payment" in payment_methods and amount_safe >= requested_amount:
        if is_safe(start_balance, minimum_balance, request_date, forecast, request_date, requested_amount):
            plan_str = _build_payment_plan_str([(request_date, requested_amount)])
            candidates.append(PaymentPlan(
                affordability_status="affordable_now",
                recommended_payment_method="full_payment",
                payment_plan_str=plan_str,
                earliest_date_for_full_payment=request_date,
                spending_changes=[],
                amount_safe_to_pay=requested_amount,
                total_paid=requested_amount,
                num_payments=1,
                first_payment_date=request_date,
                payment_option_id=None,
                completes_by_deadline=True,
            ))

    # Candidate 2: wait for full_payment
    if "full_payment" in payment_methods and earliest_full_date is not None:
        if earliest_full_date > request_date:
            completes_by = earliest_full_date <= desired_completion_date
            plan_str = _build_payment_plan_str([(earliest_full_date, requested_amount)])
            candidates.append(PaymentPlan(
                affordability_status="affordable_later",
                recommended_payment_method="wait",
                payment_plan_str=plan_str,
                earliest_date_for_full_payment=earliest_full_date,
                spending_changes=[],
                amount_safe_to_pay=amount_safe,
                total_paid=requested_amount,
                num_payments=1,
                first_payment_date=earliest_full_date,
                payment_option_id=None,
                completes_by_deadline=completes_by,
            ))

    # Candidate 3: installments
    if "installments" in payment_methods:
        options = ds.get_request_options(request_id)
        for _, opt in options[options["payment_method"] == "installments"].iterrows():
            n = int(opt["number_of_payments"])
            first_date = opt["first_payment_date"]
            freq_days = int(opt["payment_frequency_days"]) if pd.notna(opt["payment_frequency_days"]) else 30
            payment_amount = float(opt["payment_amount"])
            total_payable = float(opt["total_payable_amount"])
            opt_id = opt["payment_option_id"]
            if max_installment_months is not None and (n * freq_days) / 30 > max_installment_months:
                continue
            schedule = [(first_date + pd.Timedelta(days=i * freq_days), payment_amount) for i in range(n)]
            if compute_installment_plan_safety(start_balance, minimum_balance, request_date, forecast, schedule, desired_completion_date):
                plan_str = _build_payment_plan_str(schedule)
                candidates.append(PaymentPlan(
                    affordability_status="affordable_with_plan",
                    recommended_payment_method="installments",
                    payment_plan_str=plan_str,
                    earliest_date_for_full_payment=earliest_full_date,
                    spending_changes=[],
                    amount_safe_to_pay=amount_safe,
                    total_paid=total_payable,
                    num_payments=n,
                    first_payment_date=first_date,
                    payment_option_id=opt_id,
                    completes_by_deadline=(max(d for d, _ in schedule) <= desired_completion_date),
                ))

    # Candidate 4: partial payment
    if "partial_payment" in payment_methods and allows_partial:
        if 0 < amount_safe < requested_amount and earliest_full_date is not None:
            if earliest_full_date <= desired_completion_date:
                schedule = [(request_date, amount_safe), (earliest_full_date, requested_amount - amount_safe)]
                if compute_installment_plan_safety(start_balance, minimum_balance, request_date, forecast, schedule, desired_completion_date):
                    plan_str = _build_payment_plan_str(schedule)
                    candidates.append(PaymentPlan(
                        affordability_status="affordable_with_plan",
                        recommended_payment_method="partial_payment",
                        payment_plan_str=plan_str,
                        earliest_date_for_full_payment=earliest_full_date,
                        spending_changes=[],
                        amount_safe_to_pay=amount_safe,
                        total_paid=requested_amount,
                        num_payments=2,
                        first_payment_date=request_date,
                        payment_option_id=None,
                        completes_by_deadline=True,
                    ))

    # Candidate 5: fallback spending changes if no safe plan found yet
    if not any(c.completes_by_deadline for c in candidates):
        changes, mod_fc, new_safe = _try_spending_changes(
            entries, profile, request_date, forecast, requested_amount,
            desired_completion_date, start_balance, minimum_balance, message_facts
        )
        if changes and "full_payment" in payment_methods:
            exclude_ids = {c.event_id for c in changes if c.change_type == "stop"}
            reduce_map = {c.event_id: c.new_amount for c in changes if c.change_type == "reduce_to"}
            if is_safe(start_balance, minimum_balance, request_date, forecast, request_date, requested_amount, None, exclude_ids, reduce_map):
                plan_str = _build_payment_plan_str([(request_date, requested_amount)])
                candidates.append(PaymentPlan(
                    affordability_status="affordable_with_plan",
                    recommended_payment_method="full_payment",
                    payment_plan_str=plan_str,
                    earliest_date_for_full_payment=earliest_full_date,
                    spending_changes=changes,
                    amount_safe_to_pay=amount_safe,
                    total_paid=requested_amount,
                    num_payments=1,
                    first_payment_date=request_date,
                    payment_option_id=None,
                    completes_by_deadline=True,
                ))

    # Filter and rank
    def _rank_key(p: PaymentPlan):
        completes = 0 if p.completes_by_deadline else 1
        has_changes = 0 if not p.spending_changes else 1
        total = p.total_paid
        first = p.first_payment_date or pd.Timestamp("2099-01-01")
        n_payments = p.num_payments
        opt_id = p.payment_option_id or "zzz"
        opt_num = int("".join(filter(str.isdigit, opt_id))) if opt_id != "zzz" else 9999
        return (completes, has_changes, total, first, n_payments, opt_num)

    safe_by_deadline = [c for c in candidates if c.completes_by_deadline]
    if safe_by_deadline:
        safe_by_deadline.sort(key=_rank_key)
        return safe_by_deadline[0]
    safe_any = [c for c in candidates if c.affordability_status != "not_affordable"]
    if safe_any:
        safe_any.sort(key=_rank_key)
        return safe_any[0]

    # not_affordable
    return PaymentPlan(
        affordability_status="not_affordable",
        recommended_payment_method="not_recommended",
        payment_plan_str="none",
        earliest_date_for_full_payment=None,
        spending_changes=[],
        amount_safe_to_pay=amount_safe,
        total_paid=0.0,
        num_payments=0,
        first_payment_date=None,
        payment_option_id=None,
        completes_by_deadline=False,
    )

print("Running benchmark on all 25 sample requests with v2 engine...")
st_correct = 0
meth_correct = 0
for idx, req in samples.iterrows():
    req_id = req["request_id"]
    user_id = req["user_id"]
    profile = ds.get_profile(user_id)
    img_facts = load_image_facts(ds, req_id, user_id)
    msg_facts = load_message_facts(ds, user_id, req_id)
    entries, _ = reconcile_user_events(user_id, ds, img_facts, msg_facts, req["request_date"])
    fc = build_forecast_v2(entries, msg_facts, profile, req["request_date"])
    plan = select_best_plan_v2(req, profile, entries, fc, msg_facts, ds)
    
    st_ok = plan.affordability_status == req["affordability_status"]
    me_ok = plan.recommended_payment_method == req["recommended_payment_method"]
    if st_ok: st_correct += 1
    if me_ok: meth_correct += 1
    
    print(f"{req_id}: Status {plan.affordability_status:20s} (GT: {req['affordability_status']:20s}) [{'OK' if st_ok else 'FAIL'}] | Method {plan.recommended_payment_method:15s} (GT: {req['recommended_payment_method']:15s}) [{'OK' if me_ok else 'FAIL'}]")
    if not st_ok or not me_ok:
        changes = [c.change_type + ':' + c.event_id for c in plan.spending_changes]
        print(f"   Changes: {changes} vs GT {req['spending_changes_needed']}")

print(f"\nFINAL ACCURACY: Status {st_correct}/25 ({st_correct/25*100:.1f}%), Method {meth_correct}/25 ({meth_correct/25*100:.1f}%)")
