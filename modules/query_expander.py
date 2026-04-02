"""
Claude Query Expander
=====================
Before text retrieval, asks Claude (Haiku) to generate smart search terms
for the BLS food database. This fixes cases where the text retriever fails
because it only matches characters, not food concepts.

Cost: ~$0.001 per expansion (Haiku + 150 tokens).
Cached: same food is never expanded twice.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_FILE = CACHE_DIR / "expansion_cache.json"

SYSTEM_PROMPT = """You are a German food database expert. Given a food description from a nutrition study participant, generate search terms that would help find this food in the BLS (Bundeslebensmittelschlüssel) food database. The BLS contains standardized German food names like 'Weizenmischbrot mit Lachs geräuchert', 'Rindfleisch gekocht', 'Joghurt mit Früchten', etc.

Rules:
- Return 5-10 German search terms, one per line
- Include the literal food name if it might exist in a database
- Include German synonyms and alternative names
- For brand names, include what the product actually IS (e.g. Big Mac → Hamburger, Cheeseburger, Rindfleischburger)
- For colloquial/slang terms, include the proper German food name (e.g. Grillfackeln → mariniertes Schweinefleisch, Grillspieß)
- For English food names, include the German translation
- For composite foods, include both the composite term AND individual components
- For drinks disguised as food names, clarify (e.g. heiße Schokolade → Kakaogetränk, Schokolade heißes Getränk, Trinkschokolade)
- Think about what terms a German nutritional database would actually use
- Return ONLY the search terms, no explanations, no numbering"""


class QueryExpander:
    def __init__(self, api_key: str | None = None):
        import anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Set ANTHROPIC_API_KEY or pass api_key parameter.")

        self.client = anthropic.Anthropic(api_key=key, timeout=10.0)
        self._cache: dict[str, list[str]] = {}
        self._lock = Lock()
        self._load_cache()

    def _load_cache(self):
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._cache = {}

    def _save_cache(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=1)

    def expand(self, food_description: str) -> list[str]:
        """Generate smart search terms for a food description.

        Returns list of search term strings, or empty list on failure.
        """
        key = food_description.strip().lower()

        # Check cache
        if key in self._cache:
            return self._cache[key]

        # Call Claude Haiku
        try:
            response = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=150,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": food_description}],
                temperature=0.0,
            )

            text = response.content[0].text.strip()
            terms = [line.strip().lstrip("0123456789.-) ") for line in text.split("\n")]
            terms = [t for t in terms if t and len(t) >= 2]

            # Cache the result
            with self._lock:
                self._cache[key] = terms
                self._save_cache()

            return terms

        except Exception:
            return []

    @property
    def cache_size(self) -> int:
        return len(self._cache)
