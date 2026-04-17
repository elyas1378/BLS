"""
Persistent Cache (Google Sheets)
================================
Three-tab logging for the BLS matcher:
  - Sheet1 (flags): user-reported problem queries — never logged/promoted
  - log: every search ever — one row per search (with session_id)
  - review_queue: queries searched REVIEW_THRESHOLD+ times without being flagged
"""

from __future__ import annotations

from datetime import datetime

import gspread
from google.oauth2.service_account import Credentials
import streamlit as st


SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SPREADSHEET_ID = "1FvMb4GZi2T1g6EN4uLMew1RVS3ua0BUbPOPfRSiO8Hc"

REVIEW_THRESHOLD = 3

_LOG_COLUMNS = [
    "session_id", "query",
    "bls302_code", "bls302_name", "bls302_source", "bls302_conf",
    "bls40_code", "bls40_name", "bls40_source", "bls40_conf",
    "timestamp",
]

_REVIEW_COLUMNS = [
    "query",
    "bls302_code", "bls302_name",
    "bls40_code", "bls40_name",
    "hit_count", "first_seen", "last_seen",
]


@st.cache_resource
def _get_gspread_client():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=SCOPES
    )
    return gspread.authorize(creds)


class PersistentCache:
    def __init__(self):
        self._flagged_queries: set[str] = set()
        self._log_counts: dict[str, int] = {}
        self._review_queries: set[str] = set()
        self._log_sheet = None
        self._review_sheet = None
        self._flags_sheet = None
        self._loaded = False

    def _connect(self):
        if self._loaded:
            return
        try:
            client = _get_gspread_client()
            spreadsheet = client.open_by_key(SPREADSHEET_ID)
            self._flags_sheet = spreadsheet.worksheet("Sheet1")
            self._log_sheet = spreadsheet.worksheet("log")
            self._review_sheet = spreadsheet.worksheet("review_queue")
            self._load_all()
            self._loaded = True
        except Exception as e:
            print(f"PersistentCache connect error: {e}")
            try:
                st.warning(f"Cache connect error: {type(e).__name__}: {e}")
            except Exception:
                pass

    def _load_all(self):
        try:
            rows = self._flags_sheet.get_all_records()
            for row in rows:
                food = str(row.get("food", "")).strip().lower()
                if food:
                    self._flagged_queries.add(food)
        except Exception as e:
            print(f"PersistentCache load flags error: {e}")

        try:
            rows = self._log_sheet.get_all_records()
            for row in rows:
                q = str(row.get("query", "")).strip().lower()
                if q:
                    self._log_counts[q] = self._log_counts.get(q, 0) + 1
        except Exception as e:
            print(f"PersistentCache load log error: {e}")

        try:
            rows = self._review_sheet.get_all_records()
            for row in rows:
                q = str(row.get("query", "")).strip().lower()
                if q:
                    self._review_queries.add(q)
        except Exception as e:
            print(f"PersistentCache load review error: {e}")

    @staticmethod
    def _norm(q: str) -> str:
        return q.strip().lower()

    def is_flagged(self, query: str) -> bool:
        self._connect()
        return self._norm(query) in self._flagged_queries

    def log_search(
        self,
        session_id: str,
        query: str,
        bls302_code: str = "",
        bls302_name: str = "",
        bls302_source: str = "",
        bls302_conf: float = 0.0,
        bls40_code: str = "",
        bls40_name: str = "",
        bls40_source: str = "",
        bls40_conf: float = 0.0,
    ):
        """Log one search, then promote to review_queue at REVIEW_THRESHOLD hits."""
        self._connect()

        q_norm = self._norm(query)
        if not q_norm:
            return

        now = datetime.now().isoformat()

        log_row = {
            "session_id": session_id,
            "query": q_norm,
            "bls302_code": bls302_code,
            "bls302_name": bls302_name,
            "bls302_source": bls302_source,
            "bls302_conf": round(float(bls302_conf or 0), 3),
            "bls40_code": bls40_code,
            "bls40_name": bls40_name,
            "bls40_source": bls40_source,
            "bls40_conf": round(float(bls40_conf or 0), 3),
            "timestamp": now,
        }
        log_list = [log_row.get(c, "") for c in _LOG_COLUMNS]

        try:
            self._ensure_headers(self._log_sheet, _LOG_COLUMNS)
            self._log_sheet.append_row(log_list, value_input_option="RAW")
        except Exception as e:
            print(f"PersistentCache log error: {e}")
            try:
                st.warning(f"Cache log write error: {type(e).__name__}: {e}")
            except Exception:
                pass
            return

        count = self._log_counts.get(q_norm, 0) + 1
        self._log_counts[q_norm] = count

        # Promote to review_queue only if not flagged
        if (count >= REVIEW_THRESHOLD
                and q_norm not in self._review_queries
                and q_norm not in self._flagged_queries):
            review_row = {
                "query": q_norm,
                "bls302_code": bls302_code,
                "bls302_name": bls302_name,
                "bls40_code": bls40_code,
                "bls40_name": bls40_name,
                "hit_count": count,
                "first_seen": now,
                "last_seen": now,
            }
            review_list = [review_row.get(c, "") for c in _REVIEW_COLUMNS]
            try:
                self._ensure_headers(self._review_sheet, _REVIEW_COLUMNS)
                self._review_sheet.append_row(review_list, value_input_option="RAW")
                self._review_queries.add(q_norm)
            except Exception as e:
                print(f"PersistentCache promote error: {e}")

    def _ensure_headers(self, sheet, columns):
        try:
            existing = sheet.row_values(1)
            if not existing:
                sheet.append_row(columns, value_input_option="RAW")
        except Exception:
            try:
                sheet.append_row(columns, value_input_option="RAW")
            except Exception as e:
                print(f"PersistentCache header error: {e}")
