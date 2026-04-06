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
- If the food is a modern, specialty, or foreign item that likely does not exist in the BLS, also suggest the closest traditional German food equivalent (e.g. Proteinriegel → Müsliriegel, Energieriegel; Taco Schalen → Maistortilla, Maisfladen; Flohsamenschalen → Leinsamen, Weizenkleie)
- Think about what terms a German nutritional database would actually use
- Return ONLY the search terms, no explanations, no numbering

BLS-specific terminology:
- 'Trüffel' in BLS means the mushroom (Tuber), NOT chocolate truffles. For chocolate truffles use: Praline, Konfekt, Schokolade gefüllt
- For hot/warm drinks, always include 'Getränk' or 'heißes Getränk'. 'Schokolade' alone returns solid chocolate bars.
- 'Mousse' is poorly represented in BLS. Use Creme, Pudding, or Dessert instead.
- Brand names: Buko = Frischkäse, Manner = Haselnuss-Waffelschnitten, Riesen = Karamellbonbon. When you recognize a brand, provide the generic food category."""


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

    # ── V2: Combined spelling + expansion ──

    _V2_SYSTEM = (
        "You are a German food database expert. Given a food description "
        "from a nutrition study participant:\n"
        "1. Correct any spelling errors in the food description.\n"
        "2. Generate 5-8 German search terms that would help find this food "
        "in the BLS (Bundeslebensmittelschlüssel) food database.\n\n"
        "Include: the corrected name, common German synonyms, "
        "brand-to-generic-product mappings, English-to-German translations, "
        "and for composite foods also list the individual components.\n"
        "If the food likely does not exist in BLS, suggest the closest "
        "traditional German food equivalent.\n\n"
        "BLS terminology: 'Trüffel' = mushroom (not chocolate); "
        "for hot drinks include 'Getränk'; 'Mousse' → use 'Creme/Pudding'; "
        "Buko = Frischkäse; Manner = Haselnuss-Waffelschnitten.\n\n"
        'Respond ONLY with JSON, no markdown, no explanation:\n'
        '{"corrected": "corrected food description", '
        '"search_terms": ["term1", "term2", ...]}'
    )

    def expand_with_spelling(
        self,
        food_description: str,
        unknown_tokens: list[str] | None = None,
    ) -> dict:
        """Combined spelling correction + search term expansion.

        Called BEFORE retrieval when Tier 1 flags unknown tokens.

        Returns dict with:
            "corrected": str — spelling-corrected food description
            "search_terms": list[str] — BLS search terms
        """
        cache_key = "v2:" + food_description.strip().lower()

        # Check cache
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if isinstance(cached, dict):
                return cached

        # Build user prompt
        user_msg = f'Food description: "{food_description}"'
        if unknown_tokens:
            user_msg += f"\nUnknown tokens that need attention: {unknown_tokens}"

        fallback = {"corrected": food_description, "search_terms": []}

        try:
            response = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                system=self._V2_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
                temperature=0.0,
            )

            text = response.content[0].text.strip()

            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
                if text.startswith("json"):
                    text = text[4:].strip()

            # Parse JSON
            try:
                result = json.loads(text)
                if not isinstance(result, dict):
                    raise ValueError("not a dict")
                # Ensure expected keys
                result.setdefault("corrected", food_description)
                result.setdefault("search_terms", [])
                # Filter search terms
                result["search_terms"] = [
                    t for t in result["search_terms"]
                    if isinstance(t, str) and len(t) >= 2
                ]
            except (json.JSONDecodeError, ValueError):
                # Fallback: treat as line-separated search terms (old format)
                terms = [
                    line.strip().lstrip("0123456789.-) ")
                    for line in text.split("\n")
                ]
                terms = [t for t in terms if t and len(t) >= 2]
                result = {"corrected": food_description, "search_terms": terms}

            # Cache
            with self._lock:
                self._cache[cache_key] = result
                self._save_cache()

            return result

        except Exception:
            return fallback

    @property
    def cache_size(self) -> int:
        return len(self._cache)
