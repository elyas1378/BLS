"""
Text Match Retriever
====================
Searches the BLS catalog using direct text matching (substring + fuzzy).
Much more accurate than embedding-only search for straightforward food names.

Strategy:
  1. Exact match (food name == BLS entry)
  2. Substring match (food name is contained in BLS entry)  
  3. Fuzzy string match (for misspellings and slight variations)
  4. Fall back to FAISS for creative/unusual descriptions

Then applies BLS code rules to pick the best variant:
  - Fruits/vegetables: prefer "roh" (raw) entries
  - Beverages: prefer "(Getränk)" entries
  - Avoid generic category headers (codes ending in 000/0000)
  - Food diary context: consumed = cooked for meat, raw for produce
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from difflib import SequenceMatcher
from dataclasses import dataclass

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import CATALOG_302, CATALOG_40


@dataclass
class TextCandidate:
    code: str
    name_de: str
    name_en: str
    score: float
    food_group: str
    processing_digit: str
    portion_g: float = None
    match_type: str = ""  # "exact", "substring", "fuzzy", "faiss"

    def to_dict(self):
        return {
            "code": self.code,
            "name_de": self.name_de,
            "name_en": self.name_en,
            "score": round(self.score, 4),
            "food_group": self.food_group,
            "processing_digit": self.processing_digit,
            "portion_g": self.portion_g,
            "match_type": self.match_type,
        }


class TextMatchRetriever:
    """
    Searches BLS catalogs using text matching + BLS domain rules.
    """

    def __init__(self, verbose=True):
        if verbose:
            print("Loading TextMatchRetriever …")

        self.catalog_302 = self._load_catalog(CATALOG_302, "BLS 3.02", verbose)
        self.catalog_40 = self._load_catalog(CATALOG_40, "BLS 4.0", verbose)

        # Build word index for fast lookup
        self._word_index_302 = self._build_word_index(self.catalog_302)
        self._word_index_40 = self._build_word_index(self.catalog_40)

        if verbose:
            print(f"  Ready ✓")

    def _load_catalog(self, path, label, verbose):
        if not path.exists():
            raise FileNotFoundError(f"{label} catalog not found at {path}")
        df = pd.read_parquet(path)
        df["name_lower"] = df["name_de"].str.lower()
        # Pre-split words for fast matching
        df["name_words"] = df["name_lower"].str.findall(r'\w+')
        if verbose:
            print(f"  Loaded {len(df):,} entries for {label}")
        return df

    def _build_word_index(self, df):
        """Build inverted index: word → list of row indices."""
        index = {}
        for idx, words in df["name_words"].items():
            if isinstance(words, list):
                for w in words:
                    if len(w) >= 3:  # skip very short words
                        if w not in index:
                            index[w] = []
                        index[w].append(idx)
        return index

    def _text_search(self, query: str, catalog: pd.DataFrame, word_index: dict,
                     top_k: int = 20) -> list[TextCandidate]:
        """
        Multi-strategy text search:
        1. Exact name match
        2. Word-based candidate retrieval + scoring
        3. Substring matching
        """
        query_lower = query.lower().strip()
        query_words = set(re.findall(r'\w+', query_lower))
        query_words = {w for w in query_words if len(w) >= 3}

        candidates = {}  # code → (score, row, match_type)

        # ── Strategy 1: Exact match ──
        exact = catalog[catalog["name_lower"] == query_lower]
        for _, row in exact.iterrows():
            candidates[row["code"]] = (2.0, row, "exact")

        # ── Strategy 2: Word-based retrieval ──
        # Find all entries that share at least one word with the query
        candidate_indices = set()
        for word in query_words:
            if word in word_index:
                candidate_indices.update(word_index[word])
            # Also try prefix matching for partial words
            for w in word_index:
                if w.startswith(word) or word.startswith(w):
                    candidate_indices.update(word_index[w][:50])  # limit

        for idx in candidate_indices:
            if idx >= len(catalog):
                continue
            row = catalog.iloc[idx]
            code = row["code"]
            if code in candidates:
                continue

            name_lower = row["name_lower"]

            # Score based on multiple factors
            score = 0.0

            # Substring: query in name or name in query
            if query_lower in name_lower:
                score += 0.8
            elif name_lower in query_lower:
                score += 0.6

            # Word overlap
            name_words = set(row["name_words"]) if isinstance(row["name_words"], list) else set()
            overlap = query_words & name_words
            if overlap:
                score += 0.3 * len(overlap) / max(len(query_words), 1)

            # Fuzzy string similarity (weighted higher to catch 1-2 char typos)
            sim = SequenceMatcher(None, query_lower, name_lower).ratio()
            score += sim * 0.6

            # Name length penalty: prefer shorter (more specific) names
            # but not too short (category headers)
            name_len = len(name_lower)
            if 5 <= name_len <= 40:
                score += 0.1
            elif name_len > 60:
                score -= 0.05

            if score > 0.15:
                candidates[code] = (score, row, "text")

        # ── Strategy 3: Substring search for remaining ──
        if len(candidates) < 5 and len(query_lower) >= 4:
            mask = catalog["name_lower"].str.contains(
                re.escape(query_lower), case=False, na=False
            )
            for _, row in catalog[mask].head(20).iterrows():
                code = row["code"]
                if code not in candidates:
                    sim = SequenceMatcher(None, query_lower, row["name_lower"]).ratio()
                    candidates[code] = (sim + 0.3, row, "substring")

        # ── Apply BLS domain rules to adjust scores ──
        scored = []
        for code, (score, row, match_type) in candidates.items():
            adjusted = self._apply_bls_rules(score, code, row, query_lower)
            scored.append((adjusted, code, row, match_type))

        # Sort by adjusted score
        scored.sort(key=lambda x: x[0], reverse=True)

        # Build result
        results = []
        for adj_score, code, row, match_type in scored[:top_k]:
            results.append(TextCandidate(
                code=code,
                name_de=row["name_de"],
                name_en=row.get("name_en", ""),
                score=adj_score,
                food_group=code[0],
                processing_digit=code[-1],
                portion_g=row.get("portion_g"),
                match_type=match_type,
            ))
        return results

    def _apply_bls_rules(self, score: float, code: str, row, query: str) -> float:
        """Apply BLS code structure rules to adjust candidate scores."""
        name = row["name_lower"]
        group = code[0]
        proc = code[-1]

        # ── Penalize broad category headers ──
        if code.endswith("00000"):
            score -= 0.4
        elif code.endswith("0000"):
            score -= 0.2
        elif code.endswith("000") and proc == "0":
            # Could be a valid generic entry (like "Honig S120000")
            # Only penalize if there's a more specific version
            score -= 0.05

        # ── Fruits (F): prefer raw (position 5-6 = "10") ──
        if group == "F":
            if code[4:6] == "10":  # raw, edible portion
                score += 0.15
            elif code[4:6] == "01":  # with kitchen waste
                score -= 0.05
            if "roh" in name:
                score += 0.05

        # ── Vegetables (G): prefer raw unless cooking implied ──
        if group == "G":
            if code[4:6] == "10":
                score += 0.10
            if "roh" in name:
                score += 0.05

        # ── Beverages (N): prefer "(Getränk)" entries ──
        if group == "N":
            if "getränk" in name or "(getränk)" in name:
                score += 0.20
            if "pulver" in name or "trocken" in name:
                score -= 0.15

        # ── Eggs (E): prefer gekocht (boiled) ──
        if group == "E" and "hühnerei" in name:
            if proc == "3":  # gekocht
                score += 0.10
            if "vollei" in name:
                score += 0.05

        # ── Dairy (M): prefer common fat levels ──
        if group == "M":
            if "1,5%" in name or "1,5 %" in name:
                score += 0.03
            if "3,5%" in name or "3,5 %" in name:
                score += 0.02

        # ── Meat cooked (U/Y): prefer gegart/gekocht ──
        if group in ("U", "Y"):
            if proc in ("2", "3"):
                score += 0.05

        # ── Recipe categories (X/Y): prefer Standardrezeptur ──
        if group in ("X", "Y") and "standardrezeptur" in name:
            score += 0.05

        # ── Avoid "Konfitüre" (jam) for raw fruit searches ──
        if "konfitüre" in name and "konfitüre" not in query:
            score -= 0.3
        if "konzentrat" in name and "konzentrat" not in query:
            score -= 0.2
        if "trunk" in name and "trunk" not in query and "saft" not in query:
            score -= 0.2
        if "pulver" in name and "pulver" not in query:
            score -= 0.15
        if "getrocknet" in name and "getrocknet" not in query and "trocken" not in query:
            score -= 0.1

        return score

    def search(self, food_description: str, top_k: int = 20,
               normalize_input: bool = True):
        """
        Search both BLS catalogs for matches.

        Returns dict compatible with SmartReranker:
            {"query": NormalizedQuery, "bls302": [...], "bls40": [...]}
        """
        from modules.normalizer import normalize, NormalizedQuery

        if normalize_input:
            nq = normalize(food_description)
            # Search with BOTH original and cleaned text, merge results
            results_302_orig = self._text_search(
                food_description, self.catalog_302, self._word_index_302, top_k
            )
            results_302_clean = self._text_search(
                nq.cleaned, self.catalog_302, self._word_index_302, top_k
            )
            results_302 = self._merge(results_302_orig, results_302_clean, top_k)

            results_40_orig = self._text_search(
                food_description, self.catalog_40, self._word_index_40, top_k
            )
            results_40_clean = self._text_search(
                nq.cleaned, self.catalog_40, self._word_index_40, top_k
            )
            results_40 = self._merge(results_40_orig, results_40_clean, top_k)

            # Also search compound split/join variants for consistency
            for variant in nq.search_variants:
                v302 = self._text_search(variant, self.catalog_302, self._word_index_302, top_k)
                v40 = self._text_search(variant, self.catalog_40, self._word_index_40, top_k)
                results_302 = self._merge(results_302, v302, top_k)
                results_40 = self._merge(results_40, v40, top_k)
        else:
            nq = NormalizedQuery(original=food_description, cleaned=food_description)
            results_302 = self._text_search(
                food_description, self.catalog_302, self._word_index_302, top_k
            )
            results_40 = self._text_search(
                food_description, self.catalog_40, self._word_index_40, top_k
            )

        return {
            "query": nq,
            "bls302": results_302,
            "bls40": results_40,
        }

    def _merge(self, list_a, list_b, top_k):
        """Merge two candidate lists, keeping higher score for duplicates."""
        by_code = {}
        for c in list_a:
            by_code[c.code] = c
        for c in list_b:
            if c.code not in by_code or c.score > by_code[c.code].score:
                by_code[c.code] = c
        merged = sorted(by_code.values(), key=lambda x: x.score, reverse=True)
        return merged[:top_k]


# =====================================================================
#  CLI test
# =====================================================================

if __name__ == "__main__":
    retriever = TextMatchRetriever()

    tests = [
        "Apfel", "Kaffee", "Tee", "Gurke", "Paprika", "Brötchen",
        "Honig", "Karotte", "Espresso", "Mandarine", "Avocado",
        "Zwiebel", "Leinsamen", "Mozzarella", "Banane", "Eier",
        "Olivenöl", "Haferflocken", "Skyr", "Hafermilch",
        "Bolognese", "Kürbiscurry", "Chicken salad",
    ]

    for food in tests:
        result = retriever.search(food, top_k=5)
        print(f"\n{'─'*60}")
        print(f"  '{food}':")
        for c in result["bls302"][:3]:
            print(f"    [{c.code}] {c.name_de:<50s} score={c.score:.3f} ({c.match_type})")
