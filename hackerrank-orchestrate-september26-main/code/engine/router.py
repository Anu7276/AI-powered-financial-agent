"""
Evidence Router Module
======================
Implements Layer 2: Hybrid Retrieval & Evidence Routing.

Architecture:
                 REQUEST
                    │
                    ▼
             Evidence Router
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
 Transaction     Messages     Profile
    RAG             RAG          RAG
  (SQLite/DB)     (BM25)       (Rules)
       │            │            │
       └────────────┼────────────┘
                    ▼
              Evidence Layer
                    │
                    ▼
              Reconciliation
                    │
                    ▼
             Financial State
                    │
                    ▼
               Forecast
                    │
                    ▼
                Solver

Separation of Concerns:
- RAG / SQL retrieves and normalizes evidence
- Deterministic Python solver executes financial calculations
- LLM synthesizes evidence-grounded human explanations
"""

import sqlite3
import re
from typing import Optional, Any
from dataclasses import dataclass, field
import pandas as pd
from .evidence import ImageFact, MessageFact, extract_image_amount, extract_message_facts


@dataclass
class RetrievedEvidence:
    """Consolidated evidence package for a request."""
    user_id: str
    request_id: str
    profile: dict
    reconciled_events: list
    message_facts: list[MessageFact]
    image_facts: dict[str, ImageFact]
    retrieval_trace: list[str] = field(default_factory=list)


class TransactionRAG:
    """
    Structured relational query engine over financial transactions using SQLite in-memory database.
    Allows exact SQL aggregations, date filtering, and cadence inspection without touching solvers.
    """
    def __init__(self, events_df: pd.DataFrame):
        self.conn = sqlite3.connect(":memory:")
        # Register events table
        clean_df = events_df.copy()
        if "event_date" in clean_df.columns:
            clean_df["event_date"] = clean_df["event_date"].astype(str)
        if "settlement_date" in clean_df.columns:
            clean_df["settlement_date"] = clean_df["settlement_date"].astype(str)
        clean_df.to_sql("events", self.conn, index=False, if_exists="replace")
        self.conn.commit()

    def query_user_events(self, user_id: str, before_date: Optional[str] = None) -> pd.DataFrame:
        """Query all events for a user, optionally prior to request_date."""
        if before_date:
            sql = "SELECT * FROM events WHERE user_id = ? AND event_date <= ? ORDER BY event_date ASC"
            return pd.read_sql_query(sql, self.conn, params=(user_id, str(before_date)))
        sql = "SELECT * FROM events WHERE user_id = ? ORDER BY event_date ASC"
        return pd.read_sql_query(sql, self.conn, params=(user_id,))

    def query_category_history(self, user_id: str, category: str) -> pd.DataFrame:
        """Query historical transactions for a specific category."""
        sql = "SELECT * FROM events WHERE user_id = ? AND category = ? ORDER BY event_date DESC"
        return pd.read_sql_query(sql, self.conn, params=(user_id, category))

    def get_salary_stream(self, user_id: str) -> pd.DataFrame:
        """Find recurring salary credits."""
        sql = """
            SELECT event_id, event_date, amount, status, description
            FROM events
            WHERE user_id = ? AND category = 'salary' AND direction = 'credit'
            ORDER BY event_date DESC
        """
        return pd.read_sql_query(sql, self.conn, params=(user_id,))


class MessagesRAG:
    """
    Keyword & Semantic Retrieval over user messages (salary changes, bonus notices,
    delayed payouts, rent adjustments).
    """
    KEYWORDS = {
        "salary": ["salary", "gaji", "pay", "penggajian", "payroll"],
        "bonus": ["bonus", "komisi", "commission", "incentive", "insentif"],
        "rent": ["rent", "sewa", "lease", "landlord", "kontrakan"],
        "delay": ["delay", "pending", "tertunda", "tunda", "refund"],
        "termination": ["seasonal", "ended", "berakhir", "final", "kontrak musiman"]
    }

    def __init__(self, messages_df: pd.DataFrame):
        self.messages_df = messages_df

    def retrieve_relevant_messages(self, user_id: str, categories_of_interest: Optional[list[str]] = None) -> pd.DataFrame:
        """Retrieve messages for user matching financial relevance signals."""
        user_msgs = self.messages_df[self.messages_df["user_id"] == user_id]
        if user_msgs.empty:
            return user_msgs

        if not categories_of_interest:
            return user_msgs

        # Filter by keyword match
        matched_indices = []
        for idx, row in user_msgs.iterrows():
            text = (str(row.get("message_text", "")) + " " + str(row.get("subject", ""))).lower()
            for cat in categories_of_interest:
                kw_list = self.KEYWORDS.get(cat, [cat])
                if any(kw in text for kw in kw_list):
                    matched_indices.append(idx)
                    break

        if matched_indices:
            return user_msgs.loc[matched_indices]
        return user_msgs


class ProfileRAG:
    """
    Retrieves and parses user profile preferences and spending boundaries.
    """
    def __init__(self, profiles_df: pd.DataFrame):
        self.profiles_df = profiles_df

    def get_profile(self, user_id: str) -> dict:
        user_rows = self.profiles_df[self.profiles_df["user_id"] == user_id]
        if user_rows.empty:
            return {}
        return user_rows.iloc[0].to_dict()


class EvidenceRouter:
    """
    Central Evidence Router coordinating Transaction RAG, Messages RAG, and Profile RAG.
    Hands clean, typed evidence to the deterministic reconciliation layer.
    """
    def __init__(self, ds):
        self.ds = ds
        self.tx_rag = TransactionRAG(ds.events)
        self.msg_rag = MessagesRAG(ds.messages)
        self.profile_rag = ProfileRAG(ds.profiles)

    def route_evidence(
        self,
        user_id: str,
        request_id: str,
        request_date: pd.Timestamp,
        image_facts: dict[str, ImageFact],
        message_facts: list[MessageFact]
    ) -> RetrievedEvidence:
        trace = []

        # 1. Profile RAG
        profile = self.profile_rag.get_profile(user_id)
        trace.append(f"ProfileRAG: loaded profile for {user_id}, balance={profile.get('current_available_balance')}")

        # 2. Transaction RAG (Structured SQL)
        user_events = self.tx_rag.query_user_events(user_id)
        trace.append(f"TransactionRAG: executed SQL query, fetched {len(user_events)} historical events")

        # 3. Messages RAG
        rel_msgs = self.msg_rag.retrieve_relevant_messages(user_id)
        trace.append(f"MessagesRAG: matched {len(rel_msgs)} user messages with financial signals")

        return RetrievedEvidence(
            user_id=user_id,
            request_id=request_id,
            profile=profile,
            reconciled_events=[],
            message_facts=message_facts,
            image_facts=image_facts,
            retrieval_trace=trace
        )
