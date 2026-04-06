"""
Persistent Cache (Google Sheets)
================================
Three-tier caching for Sonnet re-ranking results:
  - verified_cache: high-confidence results (≥0.85), served without re-ranking
  - review_queue: medium-confidence results (0.60–0.84), served but flaggable
  - Sheet1 (flags): flagged results — never serve from cache

All data lives in a single Google Spreadsheet with 3 tabs.
On init, loads all rows into memory for fast lookup.
Writes go to Sheets immediately for durability.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime

import gspread
from google.oauth2.service_account import Credentials
import streamlit as st


SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SPREADSHEET_ID = "1FvMb4GZi2T1g6EN4uLMew1RVS3ua0BUbPOPfRSiO8Hc"

# Column order for cache sheets
_COLUMNS = [
    "query", "bls_version", "code", "name", "confidence",
    "full_result_json", "prompt_hash", "timestamp",
]


def _prompt_hash() -> str:
    """Compute short hash of current Sonnet system prompt."""
    try:
        from modules.reranker_v2 import SYSTEM_PROMPT
        return hashlib.md5(SYSTEM_PROMPT.encode()).hexdigest()[:8]
    except Exception:
        return "unknown"


@st.cache_resource
def _get_gspread_client():
    """Cached gspread client — shared across all PersistentCache instances."""
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=SCOPES
    )
    return gspread.authorize(creds)


class PersistentCache:
    def __init__(self):
        self._verified: dict[str, dict] = {}   # "query|version" → row dict
        self._review: dict[str, dict] = {}      # "query|version" → row dict
        self._flagged_queries: set[str] = set()
        self._verified_sheet = None
        self._review_sheet = None
        self._flags_sheet = None
        self._prompt_hash = _prompt_hash()
        self._loaded = False

    def _connect(self):
        """Connect to sheets and load all data into memory."""
        if self._loaded:
            return
        try:
            client = _get_gspread_client()
            spreadsheet = client.open_by_key(SPREADSHEET_ID)
            self._flags_sheet = spreadsheet.worksheet("Sheet1")
            self._verified_sheet = spreadsheet.worksheet("verified_cache")
            self._review_sheet = spreadsheet.worksheet("review_queue")
            self._load_all()
            self._loaded = True
        except Exception as e:
            print(f"PersistentCache connect error: {e}")

    def _load_all(self):
        """Batch-load all rows from both cache sheets + flags into memory."""
        # Load verified cache
        try:
            rows = self._verified_sheet.get_all_records()
            for row in rows:
                key = self._key(row.get("query", ""), row.get("bls_version", ""))
                self._verified[key] = row
        except Exception as e:
            print(f"PersistentCache load verified error: {e}")

        # Load review queue
        try:
            rows = self._review_sheet.get_all_records()
            for row in rows:
                key = self._key(row.get("query", ""), row.get("bls_version", ""))
                self._review[key] = row
        except Exception as e:
            print(f"PersistentCache load review error: {e}")

        # Load flagged queries
        try:
            rows = self._flags_sheet.get_all_records()
            for row in rows:
                food = row.get("food", "").strip().lower()
                if food:
                    self._flagged_queries.add(food)
        except Exception as e:
            print(f"PersistentCache load flags error: {e}")

    @staticmethod
    def _key(query: str, bls_version: str) -> str:
        return f"{query.strip().lower()}|{bls_version.strip()}"

    @property
    def prompt_version(self) -> str:
        return self._prompt_hash

    def is_flagged(self, query: str) -> bool:
        """Check if this query has been flagged by a user."""
        self._connect()
        return query.strip().lower() in self._flagged_queries

    def lookup(self, query: str, bls_version: str) -> dict | None:
        """Look up a cached result.

        Returns dict with keys: code, name, confidence, full_result_json, status
        status is "verified" or "unverified"
        Returns None if not found.
        """
        self._connect()
        key = self._key(query, bls_version)

        # Never serve cached results for flagged queries
        if self.is_flagged(query):
            return None

        # Tier 1: verified cache (always valid, prompt-independent)
        if key in self._verified:
            row = self._verified[key]
            return {
                "code": row.get("code", ""),
                "name": row.get("name", ""),
                "confidence": float(row.get("confidence", 0)),
                "full_result_json": row.get("full_result_json", ""),
                "status": "verified",
            }

        # Tier 2: review queue (valid only if prompt matches)
        if key in self._review:
            row = self._review[key]
            if row.get("prompt_hash", "") == self._prompt_hash:
                return {
                    "code": row.get("code", ""),
                    "name": row.get("name", ""),
                    "confidence": float(row.get("confidence", 0)),
                    "full_result_json": row.get("full_result_json", ""),
                    "status": "unverified",
                }
            else:
                # Stale — delete from sheet and memory
                self._delete_from_sheet(self._review_sheet, query, bls_version)
                del self._review[key]
                return None

        return None

    def store(self, query: str, bls_version: str, result: dict,
              confidence: float):
        """Store a re-ranking result in the appropriate tier.

        result should have keys: code, name, confidence, full_result_json
        """
        self._connect()

        # Don't cache flagged queries
        if self.is_flagged(query):
            return

        # Don't cache low-confidence results
        if confidence < 0.60:
            return

        key = self._key(query, bls_version)
        row_data = {
            "query": query.strip().lower(),
            "bls_version": bls_version,
            "code": result.get("code", ""),
            "name": result.get("name", ""),
            "confidence": round(confidence, 3),
            "full_result_json": result.get("full_result_json", ""),
            "prompt_hash": self._prompt_hash,
            "timestamp": datetime.now().isoformat(),
        }
        row_list = [row_data.get(c, "") for c in _COLUMNS]

        try:
            if confidence >= 0.85:
                # Write to verified_cache
                self._ensure_headers(self._verified_sheet)
                self._verified_sheet.append_row(row_list, value_input_option="RAW")
                self._verified[key] = row_data
            else:
                # Write to review_queue (0.60–0.84)
                self._ensure_headers(self._review_sheet)
                self._review_sheet.append_row(row_list, value_input_option="RAW")
                self._review[key] = row_data
        except Exception as e:
            print(f"PersistentCache store error: {e}")

    def confirm(self, query: str, bls_version: str) -> bool:
        """Move a result from review_queue to verified_cache."""
        self._connect()
        key = self._key(query, bls_version)

        if key not in self._review:
            return False

        row_data = self._review[key].copy()
        row_data["timestamp"] = datetime.now().isoformat()
        row_list = [row_data.get(c, "") for c in _COLUMNS]

        try:
            # Add to verified
            self._ensure_headers(self._verified_sheet)
            self._verified_sheet.append_row(row_list, value_input_option="RAW")
            self._verified[key] = row_data

            # Remove from review
            self._delete_from_sheet(self._review_sheet, query, bls_version)
            del self._review[key]
            return True
        except Exception as e:
            print(f"PersistentCache confirm error: {e}")
            return False

    def reject(self, query: str, bls_version: str) -> bool:
        """Delete a result from review_queue."""
        self._connect()
        key = self._key(query, bls_version)

        if key not in self._review:
            return False

        try:
            self._delete_from_sheet(self._review_sheet, query, bls_version)
            del self._review[key]
            return True
        except Exception as e:
            print(f"PersistentCache reject error: {e}")
            return False

    def _ensure_headers(self, sheet):
        """Add header row if sheet is empty."""
        try:
            existing = sheet.row_values(1)
            if not existing:
                sheet.append_row(_COLUMNS, value_input_option="RAW")
        except Exception:
            try:
                sheet.append_row(_COLUMNS, value_input_option="RAW")
            except Exception as e:
                print(f"PersistentCache header error: {e}")

    def _delete_from_sheet(self, sheet, query: str, bls_version: str):
        """Delete matching row(s) from a sheet."""
        try:
            all_values = sheet.get_all_values()
            q_lower = query.strip().lower()
            # Find rows to delete (1-indexed, skip header)
            rows_to_delete = []
            for i, row in enumerate(all_values):
                if i == 0:
                    continue  # skip header
                if (len(row) >= 2
                        and row[0].strip().lower() == q_lower
                        and row[1].strip() == bls_version):
                    rows_to_delete.append(i + 1)  # gspread is 1-indexed
            # Delete from bottom up to preserve indices
            for row_idx in reversed(rows_to_delete):
                sheet.delete_rows(row_idx)
        except Exception as e:
            print(f"PersistentCache delete error: {e}")
