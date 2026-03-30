"""
Claude API Result Cache
=======================
Persistent JSON cache for Claude re-ranking results.
Key: (food_description, bls_version) → list of match dicts.

First query costs ~$0.02, all subsequent queries are free.
Cache can be invalidated per-entry via delete().
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_FILE = CACHE_DIR / "claude_cache.json"


def _make_key(food_description: str, bls_version: str) -> str:
    return f"{food_description.strip().lower()}||{bls_version}"


class ClaudeCache:
    def __init__(self):
        self._lock = Lock()
        self._data: dict[str, list[dict]] = {}
        self._load()

    def _load(self):
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._data = {}

    def _save(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=1)

    def get(self, food_description: str, bls_version: str) -> list[dict] | None:
        key = _make_key(food_description, bls_version)
        return self._data.get(key)

    def put(self, food_description: str, bls_version: str, matches: list[dict]):
        key = _make_key(food_description, bls_version)
        with self._lock:
            self._data[key] = matches
            self._save()

    def delete(self, food_description: str, bls_version: str | None = None):
        with self._lock:
            if bls_version:
                key = _make_key(food_description, bls_version)
                self._data.pop(key, None)
            else:
                # Delete both versions
                for ver in ("BLS 3.02", "BLS 4.0"):
                    key = _make_key(food_description, ver)
                    self._data.pop(key, None)
            self._save()

    @property
    def size(self) -> int:
        return len(self._data)
