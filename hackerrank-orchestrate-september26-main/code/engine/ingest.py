"""
ingest.py — Load and normalize all dataset CSVs.
Handles currency conversion (FX), dtype coercion, and exposes clean DataFrames.
"""
from __future__ import annotations
import os
import pandas as pd
import numpy as np
from functools import lru_cache
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
DATASET_DIR = _HERE.parent.parent / "dataset"
MEDIA_DIR   = DATASET_DIR / "media" / "images"


def _load(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(DATASET_DIR / name, dtype=str, **kw)


# ── raw loaders (string dtypes preserved) ─────────────────────────────────────

def load_requests() -> pd.DataFrame:
    df = _load("requests.csv")
    df["requested_amount"]  = df["requested_amount"].astype(float)
    df["allows_partial_payment"] = df["allows_partial_payment"].astype(str).str.lower().map(
        {"true": True, "false": False}
    ).fillna(False).astype(bool)
    df["request_date"]             = pd.to_datetime(df["request_date"])
    df["desired_completion_date"]  = pd.to_datetime(df["desired_completion_date"])
    return df


def load_sample_requests() -> pd.DataFrame:
    df = _load("sample_requests.csv")
    df["requested_amount"]  = df["requested_amount"].astype(float)
    df["allows_partial_payment"] = df["allows_partial_payment"].astype(str).str.lower().map(
        {"true": True, "false": False}
    ).fillna(False).astype(bool)
    df["request_date"]             = pd.to_datetime(df["request_date"])
    df["desired_completion_date"]  = pd.to_datetime(df["desired_completion_date"])
    df["amount_safe_to_pay"]       = pd.to_numeric(df["amount_safe_to_pay"], errors="coerce")
    return df


def load_profiles() -> pd.DataFrame:
    df = _load("financial_profiles.csv")
    df["current_available_balance"]  = df["current_available_balance"].astype(float)
    df["minimum_balance_to_keep"]    = df["minimum_balance_to_keep"].astype(float)
    df["max_installment_months"]     = pd.to_numeric(df["max_installment_months"], errors="coerce")
    return df


def load_events() -> pd.DataFrame:
    df = _load("financial_events.csv")
    df["amount"]                 = pd.to_numeric(df["amount"], errors="coerce")
    df["minimum_allowed_amount"] = pd.to_numeric(df["minimum_allowed_amount"], errors="coerce")
    df["event_date"]             = pd.to_datetime(df["event_date"], errors="coerce")
    df["settlement_date"]        = pd.to_datetime(df["settlement_date"], errors="coerce")
    return df


def load_payment_options() -> pd.DataFrame:
    df = _load("request_payment_options.csv")
    df["payment_amount"]         = df["payment_amount"].astype(float)
    df["number_of_payments"]     = df["number_of_payments"].astype(int)
    df["payment_frequency_days"] = pd.to_numeric(df["payment_frequency_days"], errors="coerce")
    df["financing_fee"]          = df["financing_fee"].astype(float)
    df["total_payable_amount"]   = df["total_payable_amount"].astype(float)
    df["first_payment_date"]     = pd.to_datetime(df["first_payment_date"], errors="coerce")
    return df


def load_messages() -> pd.DataFrame:
    df = _load("messages.csv")
    df["sent_at"] = pd.to_datetime(df["sent_at"], errors="coerce")
    return df


def load_images() -> pd.DataFrame:
    return _load("images.csv")


def load_exchange_rates() -> pd.DataFrame:
    df = _load("exchange_rates.csv")
    df["rate"]      = df["rate"].astype(float)
    df["rate_date"] = pd.to_datetime(df["rate_date"])
    return df


# ── FX converter ──────────────────────────────────────────────────────────────

class FXConverter:
    """
    Convert amounts between currencies using the provided exchange_rates.csv.
    Strategy:
      1. Exact (from, to, closest date on or before event date)
      2. Inverse (to, from) and invert the rate
      3. Two-hop via USD or EUR as bridge
    Returns None if conversion cannot be resolved.
    """

    def __init__(self, rates_df: pd.DataFrame):
        self._df = rates_df.copy()
        # index: (from_currency, to_currency) → list of (date, rate) sorted ascending
        self._index: dict[tuple, list] = {}
        for _, row in self._df.iterrows():
            key = (row["from_currency"], row["to_currency"])
            self._index.setdefault(key, []).append((row["rate_date"], row["rate"]))
        for k in self._index:
            self._index[k].sort(key=lambda x: x[0])

    def _lookup(self, frm: str, to: str, on_or_before: pd.Timestamp) -> float | None:
        pairs = self._index.get((frm, to))
        if not pairs:
            return None
        # pick the most recent rate on or before the date
        rate = None
        for dt, r in pairs:
            if dt <= on_or_before:
                rate = r
            else:
                break
        # if nothing found before, take the earliest available
        if rate is None and pairs:
            rate = pairs[0][1]
        return rate

    def convert(self, amount: float, frm: str, to: str, ref_date: pd.Timestamp) -> float | None:
        if frm == to:
            return amount
        if amount is None or np.isnan(amount):
            return None

        # 1. direct
        r = self._lookup(frm, to, ref_date)
        if r is not None:
            return round(amount * r, 4)

        # 2. inverse
        r = self._lookup(to, frm, ref_date)
        if r is not None and r != 0:
            return round(amount / r, 4)

        # 3. two-hop through USD
        for bridge in ["USD", "EUR"]:
            if bridge == frm or bridge == to:
                continue
            r1 = self._lookup(frm, bridge, ref_date)
            if r1 is None:
                r1_inv = self._lookup(bridge, frm, ref_date)
                if r1_inv and r1_inv != 0:
                    r1 = 1 / r1_inv
            r2 = self._lookup(bridge, to, ref_date)
            if r2 is None:
                r2_inv = self._lookup(to, bridge, ref_date)
                if r2_inv and r2_inv != 0:
                    r2 = 1 / r2_inv
            if r1 is not None and r2 is not None:
                return round(amount * r1 * r2, 4)

        return None  # unresolvable


# ── singleton loader ───────────────────────────────────────────────────────────

class DataStore:
    """Lazily loads and caches all datasets. Pass to engine modules."""

    def __init__(self):
        self._requests:       pd.DataFrame | None = None
        self._sample:         pd.DataFrame | None = None
        self._profiles:       pd.DataFrame | None = None
        self._events:         pd.DataFrame | None = None
        self._options:        pd.DataFrame | None = None
        self._messages:       pd.DataFrame | None = None
        self._images:         pd.DataFrame | None = None
        self._fx_converter:   FXConverter  | None = None

    @property
    def requests(self) -> pd.DataFrame:
        if self._requests is None:
            self._requests = load_requests()
        return self._requests

    @property
    def sample_requests(self) -> pd.DataFrame:
        if self._sample is None:
            self._sample = load_sample_requests()
        return self._sample

    @property
    def profiles(self) -> pd.DataFrame:
        if self._profiles is None:
            self._profiles = load_profiles()
        return self._profiles

    @property
    def events(self) -> pd.DataFrame:
        if self._events is None:
            self._events = load_events()
        return self._events

    @property
    def payment_options(self) -> pd.DataFrame:
        if self._options is None:
            self._options = load_payment_options()
        return self._options

    @property
    def messages(self) -> pd.DataFrame:
        if self._messages is None:
            self._messages = load_messages()
        return self._messages

    @property
    def images(self) -> pd.DataFrame:
        if self._images is None:
            self._images = load_images()
        return self._images

    @property
    def fx(self) -> FXConverter:
        if self._fx_converter is None:
            self._fx_converter = FXConverter(load_exchange_rates())
        return self._fx_converter

    def get_profile(self, user_id: str) -> pd.Series:
        return self.profiles[self.profiles["user_id"] == user_id].iloc[0]

    def get_user_events(self, user_id: str) -> pd.DataFrame:
        return self.events[self.events["user_id"] == user_id].copy()

    def get_request_options(self, request_id: str) -> pd.DataFrame:
        return self.payment_options[self.payment_options["request_id"] == request_id].copy()

    def get_user_messages(self, user_id: str, request_id: str | None = None) -> pd.DataFrame:
        m = self.messages
        mask = (m["user_id"] == user_id)
        if request_id:
            mask |= (m["request_id"] == request_id)
        return m[mask].copy()

    def get_event_image(self, event_id: str) -> str | None:
        imgs = self.images
        row = imgs[imgs["related_event_id"] == event_id]
        if row.empty:
            return None
        img_id = row.iloc[0]["image_id"]
        path = MEDIA_DIR / f"{img_id}.png"
        return str(path) if path.exists() else None
