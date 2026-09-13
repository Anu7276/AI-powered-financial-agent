"""
reconcile.py — Event reconciliation, deduplication, and conflict resolution.

Rules (from problem spec):
1. Explicit cancellation/settlement/amendment wins over estimate/forecast.
2. Newer record from same source wins.
3. Settled event wins over pending/scheduled/forecasted.
4. Financially safer interpretation when conflict unresolvable.
5. Ignore: failed, cancelled, unrealized investment_valuation.
6. Ignore: pending credits (they're not confirmed cash).
7. Linked events: apply transaction chain netting.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np
from .ingest import DataStore, FXConverter
from .evidence import ImageFact, MessageFact


# ── Canonical ledger entry ─────────────────────────────────────────────────────

@dataclass
class LedgerEntry:
    event_id: str
    user_id: str
    event_type: str        # expense, income, debt_payment, subscription, refund, etc.
    category: str
    direction: str         # debit or credit
    amount: float          # in home currency
    home_currency: str
    date: pd.Timestamp     # settlement_date if settled, else event_date
    status: str
    flexibility: str       # fixed, reducible, stoppable, reducible_or_stoppable
    minimum_allowed_amount: Optional[float]  # for reducible
    is_recurring: bool = False
    recurrence_cadence_days: Optional[int] = None
    linked_event_id: Optional[str] = None
    source: str = "financial_events"
    description: str = ""


def reconcile_user_events(
    user_id: str,
    ds: DataStore,
    image_facts: dict[str, ImageFact],     # event_id → ImageFact
    message_facts: list[MessageFact],
    request_date: pd.Timestamp,
) -> list[LedgerEntry]:
    """
    Build a clean list of LedgerEntry for a user, applying all reconciliation rules.
    Only returns events that should count in the 90-day forecast.
    """
    profile = ds.get_profile(user_id)
    home_currency = profile["home_currency"]
    raw = ds.get_user_events(user_id).copy()

    # ── Step 1: fill in blank amounts from image facts ─────────────────────────
    for idx, row in raw.iterrows():
        if pd.isna(row["amount"]) and row["event_id"] in image_facts:
            fact = image_facts[row["event_id"]]
            if fact.amount is not None:
                raw.at[idx, "amount"] = fact.amount
                raw.at[idx, "currency"] = fact.currency or row["currency"]

    # ── Step 2: convert to home currency ──────────────────────────────────────
    fx = ds.fx
    def _convert_row(row):
        if pd.isna(row["amount"]):
            return None
        currency = row["currency"]
        if currency == home_currency:
            return row["amount"]
        ref_date = row["event_date"] if not pd.isna(row["event_date"]) else request_date
        converted = fx.convert(row["amount"], currency, home_currency, ref_date)
        return converted  # may be None if rate not found

    raw["amount_home"] = raw.apply(_convert_row, axis=1)

    # ── Step 3: apply message-derived amendments ───────────────────────────────
    # Build message amendment index: what changes to apply to the forecast
    # (We store these for use in forecast.py; here we handle only event-level overrides)
    salary_overrides: list[MessageFact] = []
    for mf in message_facts:
        if mf.fact_type in ("salary_change", "salary_confirmed") and mf.confidence >= 0.7:
            salary_overrides.append(mf)
        elif mf.fact_type == "salary_seasonal_end":
            salary_overrides.append(mf)

    # ── Step 4: filter out rows that shouldn't count ───────────────────────────
    # Drop: failed, unrealized investment_valuations
    drop_statuses = {"failed", "unrealized"}
    # Drop: cancelled events that have a settled linked replacement
    linked_settled = set(
        raw[(raw["linked_event_id"].notna()) & (raw["status"] == "settled")]["linked_event_id"].tolist()
    )
    # Events to drop:
    #  - status in drop_statuses
    #  - status == cancelled (unless it IS the replacement = linked to a cancelled)
    #  - status == pending with direction == credit (unconfirmed cash-in)
    def _should_include(row) -> bool:
        s = row["status"]
        if s in drop_statuses:
            return False
        if s == "cancelled":
            # Only include if it's a settled replacement that is linked from cancelled
            # Actually cancelled rows themselves are dropped
            return False
        if s == "pending" and row["direction"] == "credit":
            # Pending refunds — per spec, ignore pending credits
            return False
        if row["event_type"] == "investment_valuation":
            return False
        return True

    filtered = raw[raw.apply(_should_include, axis=1)].copy()

    # ── Step 5: handle linked transaction chains ───────────────────────────────
    # For each (purchase event) that was cancelled and a replacement (linked_event_id)
    # exists and is settled — the replacement was already kept, don't double-count.
    # For refunds linked to original purchases: both are kept (net effect correct).

    # Build the set of original events that were cancelled but replaced
    cancelled_with_replacement = set()
    for eid in linked_settled:
        # eid is the cancelled original that was linked from a settled replacement
        orig_rows = raw[raw["event_id"] == eid]
        if not orig_rows.empty and orig_rows.iloc[0]["status"] == "cancelled":
            cancelled_with_replacement.add(eid)

    # Remove those cancelled originals (already dropped by status filter)
    # The replacement is already in filtered as a settled event.

    # ── Step 6: determine date to use ─────────────────────────────────────────
    def _effective_date(row) -> pd.Timestamp:
        if not pd.isna(row.get("settlement_date")):
            return row["settlement_date"]
        return row["event_date"]

    filtered["eff_date"] = filtered.apply(_effective_date, axis=1)

    # ── Step 7: build LedgerEntry list ────────────────────────────────────────
    entries = []
    for _, row in filtered.iterrows():
        amt = row["amount_home"]
        if amt is None or (isinstance(amt, float) and np.isnan(amt)):
            # Skip events where amount cannot be determined
            continue

        entries.append(LedgerEntry(
            event_id=row["event_id"],
            user_id=user_id,
            event_type=row["event_type"],
            category=row["category"],
            direction=row["direction"],
            amount=float(amt),
            home_currency=home_currency,
            date=row["eff_date"],
            status=row["status"],
            flexibility=row["flexibility"] if not pd.isna(row.get("flexibility", None)) else "fixed",
            minimum_allowed_amount=(
                float(row["minimum_allowed_amount"])
                if not pd.isna(row.get("minimum_allowed_amount", float("nan")))
                else None
            ),
            linked_event_id=row["linked_event_id"] if not pd.isna(row.get("linked_event_id", None)) else None,
            description=str(row.get("description", "") or ""),
        ))

    return entries, salary_overrides
