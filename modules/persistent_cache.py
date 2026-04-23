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
    # NOVA verification (Layer 3). Empty on old rows / rows that didn't
    # run verification. Used as a disk-free cache for Claude responses.
    "rule_nova", "llm_nova", "llm_agreed", "llm_method", "llm_reason",
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
        # NOVA cache: (bls_code_upper, normalized_query) -> latest verdict dict
        # Populated from log rows that have llm_nova filled; updated whenever
        # log_search() writes a new row with NOVA fields.
        self._nova_cache: dict[tuple[str, str], dict] = {}
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
            self._flags_sheet = spreadsheet.worksheet("flags")
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
                if not q:
                    continue
                self._log_counts[q] = self._log_counts.get(q, 0) + 1

                # Rebuild the NOVA cache from any log row that carries a
                # llm_nova value. Later rows overwrite earlier ones so the
                # cache reflects the most recent verdict per (code, query).
                llm_nova = row.get("llm_nova", "")
                if llm_nova not in ("", None):
                    try:
                        nova_int = int(llm_nova)
                    except (ValueError, TypeError):
                        continue
                    code = str(row.get("bls302_code", "")).strip().upper()
                    key = (code, q)
                    agreed = row.get("llm_agreed", "")
                    self._nova_cache[key] = {
                        "nova": nova_int,
                        "agree": str(agreed).lower() in ("true", "1", "yes"),
                        "reason": str(row.get("llm_reason", "")),
                        "method": str(row.get("llm_method", "")),
                    }
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

    def get_nova_cache(self, bls_code: str, query: str) -> dict | None:
        """Return the most-recent cached LLM NOVA verdict for (code, query).

        Returns None on miss. The lookup is in-memory (hydrated on load and
        updated by log_search), so this is cheap to call per search.
        """
        self._connect()
        key = ((bls_code or "").strip().upper(), self._norm(query))
        cached = self._nova_cache.get(key)
        return dict(cached) if cached else None

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
        # NOVA verification fields. All optional — empty when Layer 3 wasn't
        # run. Populated here to act as the cache for future runs.
        rule_nova: int | None = None,
        llm_nova: int | None = None,
        llm_agreed: bool | None = None,
        llm_method: str = "",
        llm_reason: str = "",
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
            "rule_nova": "" if rule_nova is None else int(rule_nova),
            "llm_nova": "" if llm_nova is None else int(llm_nova),
            "llm_agreed": "" if llm_agreed is None else bool(llm_agreed),
            "llm_method": llm_method or "",
            "llm_reason": (llm_reason or "")[:500],
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

        # Keep the in-memory NOVA cache in sync with what we just wrote.
        if llm_nova is not None:
            key = ((bls302_code or "").strip().upper(), q_norm)
            self._nova_cache[key] = {
                "nova": int(llm_nova),
                "agree": bool(llm_agreed) if llm_agreed is not None else False,
                "reason": llm_reason or "",
                "method": llm_method or "",
            }

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
