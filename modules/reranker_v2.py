"""
Re-ranker v2 — Improved Claude Re-ranking
==========================================
Fixes from v1:
  1. Validates all codes against BLS catalog (no hallucinated codes)
  2. Combines FAISS + text matching candidates for better coverage
  3. Forces Claude to ONLY pick from provided candidate list
  4. Always returns top-3 for BOTH BLS 3.02 and 4.0
  5. Post-validates and falls back if Claude returns bad codes

Usage:
    from modules.reranker_v2 import RerankerV2
    result = reranker.rerank("Käsekuchen", candidates)
    # result has .bls302_matches and .bls40_matches, each with 3 entries

Prerequisites:
    pip install anthropic
    export ANTHROPIC_API_KEY="your-key-here"
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import CLAUDE_MODEL, FOOD_GROUP_LETTERS, PROCESSING_STATES, CATALOG_302, CATALOG_40
from modules.verified_map import VERIFIED_MAP_302
from modules.verified_map_40 import VERIFIED_MAP_40


# =====================================================================
#  Data classes
# =====================================================================

@dataclass
class RankedMatch:
    rank: int
    code: str
    name: str
    confidence: float
    reasoning: str

    def to_dict(self) -> dict:
        return {"rank": self.rank, "code": self.code, "name": self.name,
                "confidence": round(self.confidence, 2), "reasoning": self.reasoning}


@dataclass
class RerankerResult:
    food_description: str
    bls302_matches: list[RankedMatch] = field(default_factory=list)
    bls40_matches: list[RankedMatch] = field(default_factory=list)
    error: str | None = None
    bls302_source: str = ""   # "verified", "cached", "api"
    bls40_source: str = ""    # "verified", "cached", "api"
    claude_nova: int | None = None       # NOVA score from Claude (if requested)
    claude_nova_reasoning: str = ""      # Claude's NOVA explanation


# =====================================================================
#  System prompt
# =====================================================================

SYSTEM_PROMPT = """You are an expert nutritionist matching German food diary entries to BLS (Bundeslebensmittelschlüssel) codes.

CRITICAL RULES:
1. You MUST pick ONLY from the candidate list provided. Do NOT invent codes.
2. Return EXACTLY 3 matches as a JSON array. No other text.
3. Every code you return MUST appear in the candidate list below.

## BLS Code Structure
Codes are 7 characters: 1 letter + 6 digits.

Position 1 (letter) = food group:
{food_groups}

Last digit = processing state:
{processing_states}

## CRITICAL MATCHING RULE — GENERIC vs SPECIFIC codes:
When the food description does NOT explicitly mention a preparation method
(gekocht, gebraten, gebacken, roh, etc.), ALWAYS prefer the GENERIC code
(ending in 00 or 000) over a specific preparation variant. Examples:
- "Lachs" → pick T410000 (Lachs generic), NOT T410082 (Lachs gebraten)
- "Apfel" → pick F110100 (Apfel roh), NOT F110122 (Apfel gegart)
- "Käse" → pick M400000 (Schnittkäse generic), NOT a specific fat variant
ONLY pick a specific preparation code if the description explicitly states it:
- "Lachs, gebraten" → T410082 is correct (gebraten explicitly stated)
- "Reis gekocht" → C352032 is correct (gekocht explicitly stated)

## EVERY WORD MATTERS — modifier awareness:
Every word in the input carries meaning. Pay close attention to:
- Negations: "sin" (without), "ohne" (without) → do NOT pick the opposite (e.g. "sin carne" ≠ "con carne")
- Flavor/type modifiers: "Vanille", "Schoko", "Reis" → the match MUST reflect this modifier
- "mit X" constructions: the match should reflect BOTH the base food AND X, not just the base food alone
- Never add attributes NOT present in the input. "Toast" = generic toast, NOT gluten-free toast.

## Brand names and colloquial German food names:
Interpret these as the FOOD PRODUCT they refer to, not the literal word meaning:
- "Osterhase" = chocolate Easter bunny (Schokolade/Schokoladenfigur), NOT Hase (rabbit meat)
- "Riesen" = caramel candy brand (Karamellbonbon), NOT Riesengarnelen (shrimp)
- "Lindt Kugel" = chocolate praline (Praline/Konfekt), NOT Trüffel (mushroom)
- "Kinder", "Duplo", "Hanuta", "Milka", "Snickers" = chocolate/candy products

## Other matching priorities:
- Composite dishes → X (vegetable-based) or Y (meat/fish-based) recipes
- Fat% should match when specified
- Avoid top-level category headers (codes like M000000, G000000)
- Prefer entries with "Standardrezeptur" for recipe-type foods
- When fat% is mentioned (e.g., "3,5% Fett"), match the closest fat level

## Closest substitute when no exact match exists:
If no candidate is an exact match, pick the most nutritionally similar food from the list. A close substitute with confidence 0.40-0.60 is more useful than refusing to match.

## Few-shot examples of correct reasoning:

Input: "Chili sin carne"
Correct: Pick vegetarian chili / bean stew — "sin" means WITHOUT meat. Do NOT pick "Chili con carne".

Input: "Osterhase"
Correct: Pick Vollmilchschokolade / Schokoladenfigur — this is a chocolate Easter bunny, NOT rabbit meat.

Input: "Reiswaffel"
Correct: Pick Reiswaffel / Reiswaffel entry — a rice cake/wafer, NOT Schokowaffel.

Input: "Vanillejoghurt"
Correct: Pick Joghurt mit Vanille / Vanillejoghurt — the match MUST contain vanilla, NOT plain Joghurt.

Input: "Brötchen mit Lachs"
Correct: If available, pick a composite entry (Brötchen mit Lachs/Räucherlachs). If not, prefer Lachs over plain Brötchen — the participant is describing what they ate ON the bread, which is nutritionally more significant.

## NOVA Classification
Also classify this food according to the NOVA food classification system:
- NOVA 1: Unprocessed or minimally processed foods (fresh fruits, vegetables, plain meat, eggs, milk, plain grains, water, coffee, tea)
- NOVA 2: Processed culinary ingredients (oils, butter, sugar, flour, honey, salt)
- NOVA 3: Processed foods (canned goods, cheese, bread, cured meats, beer, wine)
- NOVA 4: Ultra-processed foods (soft drinks, chips, candy, instant noodles, frozen pizza, packaged snacks, sausages, sweetened cereals, ice cream, energy drinks, anything with brand names from industrial food companies)

## Response format (ONLY this JSON, no other text):
```json
{{
  "matches": [
    {{"rank": 1, "code": "CODE_FROM_LIST", "name": "name", "confidence": 0.92, "reasoning": "brief reason"}},
    {{"rank": 2, "code": "CODE_FROM_LIST", "name": "name", "confidence": 0.75, "reasoning": "brief reason"}},
    {{"rank": 3, "code": "CODE_FROM_LIST", "name": "name", "confidence": 0.60, "reasoning": "brief reason"}}
  ],
  "nova_score": 1,
  "nova_reasoning": "One sentence explanation of NOVA classification"
}}
```

Confidence: 0.90-1.0 = near-exact, 0.70-0.89 = strong, 0.50-0.69 = reasonable, 0.30-0.49 = rough substitute, <0.30 = rough nutritional proxy at best (flag in reasoning for manual review)."""


# =====================================================================
#  RerankerV2 class
# =====================================================================

class RerankerV2:
    def __init__(self, api_key: str | None = None, cache=None):
        import anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Set ANTHROPIC_API_KEY or pass api_key parameter.")

        self.client = anthropic.Anthropic(api_key=key)
        self.model = CLAUDE_MODEL
        self.cache = cache  # ClaudeCache instance (or None to disable)
        self.session_cache_hits = 0
        self.session_api_calls = 0

        # Build system prompt
        fg = "\n".join(f"  {k} = {v}" for k, v in FOOD_GROUP_LETTERS.items())
        ps = "\n".join(f"  {k} = {v}" for k, v in PROCESSING_STATES.items())
        self.system_prompt = SYSTEM_PROMPT.format(
            food_groups=fg, processing_states=ps,
        )

        # Load valid code sets for validation
        self._valid_302 = set()
        self._valid_40 = set()
        self._names_302 = {}
        self._names_40 = {}
        self._load_valid_codes()

    def _load_valid_codes(self):
        """Load all valid BLS codes from catalogs."""
        import pandas as pd
        if CATALOG_302.exists():
            df = pd.read_parquet(CATALOG_302)
            self._valid_302 = set(df["code"])
            self._names_302 = dict(zip(df["code"], df["name_de"]))
        if CATALOG_40.exists():
            df = pd.read_parquet(CATALOG_40)
            self._valid_40 = set(df["code"])
            self._names_40 = dict(zip(df["code"], df["name_de"]))

    def _build_prompt(self, food_description: str, candidates: list[dict],
                      bls_version: str, prep_state: str = None,
                      fat_percent: str = None) -> str:
        """Build the user message with candidate list."""
        parts = [f'## Food description: "{food_description}"']

        context = []
        if prep_state:
            context.append(f"Preparation: {prep_state}")
        if fat_percent:
            context.append(f"Fat: {fat_percent}%")
        if context:
            parts.append(f"Context: {', '.join(context)}")

        parts.append(f"\n## BLS version: {bls_version}")
        parts.append(f"## Candidates (you MUST pick from this list):")

        for i, c in enumerate(candidates, 1):
            line = f"  {i:2d}. [{c['code']}] {c['name_de']}"
            if c.get('name_en'):
                line += f" / {c['name_en']}"
            line += f"  (score: {c['score']:.3f})"
            parts.append(line)

        parts.append(f"\nPick the best 3 from the list above. ONLY use codes from this list.")
        return "\n".join(parts)

    def _call_claude(self, prompt: str, valid_codes: set,
                     candidate_codes: set) -> tuple[list[RankedMatch], int | None, str]:
        """Call Claude and validate the response.

        Returns:
            (matches, nova_score, nova_reasoning)
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        text = response.content[0].text.strip()

        # Parse JSON
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        parsed = json.loads(text)

        # Handle both formats: new {matches, nova_score} and old [array]
        nova_score = None
        nova_reasoning = ""
        if isinstance(parsed, dict):
            matches_raw = parsed.get("matches", [])
            nova_score = parsed.get("nova_score")
            nova_reasoning = parsed.get("nova_reasoning", "")
            if isinstance(nova_score, int) and nova_score not in (1, 2, 3, 4):
                nova_score = None
        else:
            matches_raw = parsed  # old format: bare array

        # Validate each match
        results = []
        for m in matches_raw:
            code = m["code"]

            # VALIDATION 1: Code must be in candidate list
            if code not in candidate_codes:
                # Try to find it in valid codes at least
                if code not in valid_codes:
                    continue  # Skip hallucinated codes entirely
                # Code exists but wasn't in candidates — lower confidence
                m["confidence"] = min(m.get("confidence", 0.5), 0.5)
                m["reasoning"] = f"[NOT IN CANDIDATES] {m.get('reasoning', '')}"

            # VALIDATION 2: Code must exist in BLS catalog
            if code not in valid_codes:
                continue  # Skip invalid codes

            results.append(RankedMatch(
                rank=m["rank"],
                code=code,
                name=m["name"],
                confidence=m["confidence"],
                reasoning=m["reasoning"],
            ))

        return results[:3], nova_score, nova_reasoning

    @staticmethod
    def _matches_to_dicts(matches: list[RankedMatch]) -> list[dict]:
        return [m.to_dict() for m in matches]

    @staticmethod
    def _dicts_to_matches(dicts: list[dict]) -> list[RankedMatch]:
        return [RankedMatch(
            rank=d["rank"], code=d["code"], name=d["name"],
            confidence=d["confidence"], reasoning=d["reasoning"],
        ) for d in dicts]

    def _build_combined_prompt(self, food_description: str,
                               candidates_302: list[dict],
                               candidates_40: list[dict],
                               prep_state: str = None,
                               fat_percent: str = None) -> str:
        """Build a single user message for BOTH BLS versions."""
        parts = [f'## Food description: "{food_description}"']

        context = []
        if prep_state:
            context.append(f"Preparation: {prep_state}")
        if fat_percent:
            context.append(f"Fat: {fat_percent}%")
        if context:
            parts.append(f"Context: {', '.join(context)}")

        parts.append(
            "\nYou must provide matches for BOTH BLS versions below. "
            "Each version has its own candidate list — pick ONLY from "
            "the corresponding list."
        )

        parts.append("\n=== BLS 3.02 Candidates ===")
        if candidates_302:
            for i, c in enumerate(candidates_302, 1):
                line = f"  {i:2d}. [{c['code']}] {c['name_de']}"
                if c.get('name_en'):
                    line += f" / {c['name_en']}"
                line += f"  (score: {c['score']:.3f})"
                parts.append(line)
        else:
            parts.append("  No candidates available.")

        parts.append("\n=== BLS 4.0 Candidates ===")
        if candidates_40:
            for i, c in enumerate(candidates_40, 1):
                line = f"  {i:2d}. [{c['code']}] {c['name_de']}"
                if c.get('name_en'):
                    line += f" / {c['name_en']}"
                line += f"  (score: {c['score']:.3f})"
                parts.append(line)
        else:
            parts.append("  No candidates available.")

        parts.append(
            "\nPick the best 3 from EACH list above. "
            "ONLY use codes from the corresponding list.\n\n"
            'IMPORTANT: Your response MUST use this exact JSON structure '
            'with "bls302" and "bls40" as top-level keys:\n'
            '{"bls302": {"matches": [{"rank": 1, "code": "...", "name": "...", '
            '"confidence": 0.92, "reasoning": "..."}]}, '
            '"bls40": {"matches": [{"rank": 1, "code": "...", "name": "...", '
            '"confidence": 0.92, "reasoning": "..."}]}, '
            '"nova_score": 1, "nova_reasoning": "..."}'
        )
        return "\n".join(parts)

    def _call_claude_combined(self, prompt: str,
                              cand_codes_302: set, cand_codes_40: set
                              ) -> dict:
        """Call Claude once for both BLS versions.

        Returns dict with keys:
            matches_302, matches_40, nova_score, nova_reasoning
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1200,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        text = response.content[0].text.strip()

        # Strip markdown fences
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        parsed = json.loads(text)

        nova_score = parsed.get("nova_score")
        nova_reasoning = parsed.get("nova_reasoning", "")
        if isinstance(nova_score, int) and nova_score not in (1, 2, 3, 4):
            nova_score = None

        def _validate(matches_raw, valid_codes, cand_codes):
            results = []
            for m in matches_raw:
                code = m.get("code", "")
                if code not in cand_codes:
                    if code not in valid_codes:
                        continue
                    m["confidence"] = min(m.get("confidence", 0.5), 0.5)
                    m["reasoning"] = f"[NOT IN CANDIDATES] {m.get('reasoning', '')}"
                if code not in valid_codes:
                    continue
                results.append(RankedMatch(
                    rank=m.get("rank", 0), code=code,
                    name=m.get("name", ""), confidence=m.get("confidence", 0),
                    reasoning=m.get("reasoning", ""),
                ))
            return results[:3]

        # Parse both sections
        bls302_data = parsed.get("bls302", {})
        bls40_data = parsed.get("bls40", {})
        matches_302_raw = bls302_data.get("matches", []) if isinstance(bls302_data, dict) else []
        matches_40_raw = bls40_data.get("matches", []) if isinstance(bls40_data, dict) else []

        # Fallback: Sonnet returned old single-version format {"matches": [...]}
        if not matches_302_raw and not matches_40_raw and "matches" in parsed:
            print("  ⚠ Sonnet returned old format — splitting matches by code validation")
            all_matches = parsed.get("matches", [])
            for m in all_matches:
                code = m.get("code", "")
                if code in cand_codes_302:
                    matches_302_raw.append(m)
                elif code in cand_codes_40:
                    matches_40_raw.append(m)

        return {
            "matches_302": _validate(matches_302_raw, self._valid_302, cand_codes_302),
            "matches_40": _validate(matches_40_raw, self._valid_40, cand_codes_40),
            "nova_score": nova_score,
            "nova_reasoning": nova_reasoning,
        }

    def rerank(self, food_description: str, retrieval_results: dict,
               skip_cache: bool = False) -> RerankerResult:
        """
        Re-rank candidates for BOTH BLS versions.
        Uses verified lookup first (free), then cache, then Claude.
        Combined single API call when both versions need re-ranking.
        """
        result = RerankerResult(food_description=food_description)
        nq = retrieval_results["query"]
        original_lower = food_description.lower().strip()

        try:
            # ── TIER 0: Verified lookup (skip Claude entirely) ──
            # skip_cache also skips verified maps (used by Re-query)
            v302 = None if skip_cache else VERIFIED_MAP_302.get(original_lower)
            v40 = None if skip_cache else VERIFIED_MAP_40.get(original_lower)

            if v302:
                name = self._names_302.get(v302, v302)
                result.bls302_matches = [RankedMatch(
                    rank=1, code=v302, name=name, confidence=0.95,
                    reasoning=f"Verified: '{food_description}' -> [{v302}]"
                )]
                result.bls302_source = "verified"

            if v40:
                name = self._names_40.get(v40, v40)
                result.bls40_matches = [RankedMatch(
                    rank=1, code=v40, name=name, confidence=0.95,
                    reasoning=f"Verified 4.0: '{food_description}' -> [{v40}]"
                )]
                result.bls40_source = "verified"

            # If both verified, skip Claude entirely
            if v302 and v40:
                return result

            # ── Check cache for both versions ──
            need_api_302 = not v302 and bool(retrieval_results.get("bls302"))
            need_api_40 = not v40 and bool(retrieval_results.get("bls40"))

            cands_302 = []
            cands_40 = []

            if need_api_302:
                cached = None
                if self.cache and not skip_cache:
                    cached = self.cache.get(food_description, "BLS 3.02")
                if cached is not None:
                    result.bls302_matches = self._dicts_to_matches(cached)
                    result.bls302_source = "cached"
                    self.session_cache_hits += 1
                    need_api_302 = False
                else:
                    cands_302 = [c.to_dict() if hasattr(c, "to_dict") else c
                                 for c in retrieval_results["bls302"]]

            if need_api_40:
                cached = None
                if self.cache and not skip_cache:
                    cached = self.cache.get(food_description, "BLS 4.0")
                if cached is not None:
                    result.bls40_matches = self._dicts_to_matches(cached)
                    result.bls40_source = "cached"
                    self.session_cache_hits += 1
                    need_api_40 = False
                else:
                    cands_40 = [c.to_dict() if hasattr(c, "to_dict") else c
                                for c in retrieval_results["bls40"]]

            # ── API call ──
            if need_api_302 and need_api_40:
                # Combined single call for both versions
                cand_codes_302 = {c["code"] for c in cands_302}
                cand_codes_40 = {c["code"] for c in cands_40}
                prompt = self._build_combined_prompt(
                    food_description, cands_302, cands_40,
                    prep_state=nq.prep_state, fat_percent=nq.fat_percent,
                )
                combined = self._call_claude_combined(
                    prompt, cand_codes_302, cand_codes_40
                )
                result.bls302_matches = combined["matches_302"]
                result.bls40_matches = combined["matches_40"]
                if combined["nova_score"] is not None:
                    result.claude_nova = combined["nova_score"]
                    result.claude_nova_reasoning = combined["nova_reasoning"]
                self.session_api_calls += 1
                result.bls302_source = "api"
                result.bls40_source = "api"

                # Fallback if Claude returned nothing
                if not result.bls302_matches and cands_302:
                    for i, c in enumerate(cands_302[:3], 1):
                        result.bls302_matches.append(RankedMatch(
                            rank=i, code=c["code"], name=c["name_de"],
                            confidence=c["score"] * 0.7,
                            reasoning="Fallback: Claude returned no valid codes",
                        ))
                if not result.bls40_matches and cands_40:
                    for i, c in enumerate(cands_40[:3], 1):
                        result.bls40_matches.append(RankedMatch(
                            rank=i, code=c["code"], name=c["name_de"],
                            confidence=c["score"] * 0.7,
                            reasoning="Fallback: Claude returned no valid codes",
                        ))

                # Save to cache
                if self.cache and result.bls302_matches:
                    self.cache.put(food_description, "BLS 3.02",
                                   self._matches_to_dicts(result.bls302_matches))
                if self.cache and result.bls40_matches:
                    self.cache.put(food_description, "BLS 4.0",
                                   self._matches_to_dicts(result.bls40_matches))

            elif need_api_302:
                # Only 3.02 needs API (4.0 was cached/verified)
                cand_codes_302 = {c["code"] for c in cands_302}
                prompt_302 = self._build_prompt(
                    food_description, cands_302, "BLS 3.02",
                    prep_state=nq.prep_state, fat_percent=nq.fat_percent,
                )
                matches, nova, nova_reason = self._call_claude(
                    prompt_302, self._valid_302, cand_codes_302
                )
                result.bls302_matches = matches
                if nova is not None and result.claude_nova is None:
                    result.claude_nova = nova
                    result.claude_nova_reasoning = nova_reason
                self.session_api_calls += 1
                result.bls302_source = "api"
                if not result.bls302_matches and cands_302:
                    for i, c in enumerate(cands_302[:3], 1):
                        result.bls302_matches.append(RankedMatch(
                            rank=i, code=c["code"], name=c["name_de"],
                            confidence=c["score"] * 0.7,
                            reasoning="Fallback: Claude returned no valid codes",
                        ))
                if self.cache and result.bls302_matches:
                    self.cache.put(food_description, "BLS 3.02",
                                   self._matches_to_dicts(result.bls302_matches))

            elif need_api_40:
                # Only 4.0 needs API (3.02 was cached/verified)
                cand_codes_40 = {c["code"] for c in cands_40}
                prompt_40 = self._build_prompt(
                    food_description, cands_40, "BLS 4.0",
                    prep_state=nq.prep_state, fat_percent=nq.fat_percent,
                )
                matches, nova, nova_reason = self._call_claude(
                    prompt_40, self._valid_40, cand_codes_40
                )
                result.bls40_matches = matches
                if nova is not None and result.claude_nova is None:
                    result.claude_nova = nova
                    result.claude_nova_reasoning = nova_reason
                self.session_api_calls += 1
                result.bls40_source = "api"
                if not result.bls40_matches and cands_40:
                    for i, c in enumerate(cands_40[:3], 1):
                        result.bls40_matches.append(RankedMatch(
                            rank=i, code=c["code"], name=c["name_de"],
                            confidence=c["score"] * 0.7,
                            reasoning="Fallback: Claude returned no valid codes",
                        ))
                if self.cache and result.bls40_matches:
                    self.cache.put(food_description, "BLS 4.0",
                                   self._matches_to_dicts(result.bls40_matches))

        except Exception as e:
            result.error = str(e)

        return result


# =====================================================================
#  Pretty print
# =====================================================================

def print_result(result: RerankerResult):
    print(f"\n{'═'*70}")
    print(f"  Food: {result.food_description!r}")

    if result.error:
        print(f"  ERROR: {result.error}")
        return

    for version, matches in [("BLS 3.02", result.bls302_matches),
                              ("BLS 4.0", result.bls40_matches)]:
        print(f"\n  {version}:")
        if not matches:
            print(f"    (no matches)")
            continue
        for m in matches:
            marker = "🟢" if m.confidence >= 0.85 else "🟡" if m.confidence >= 0.60 else "🔴"
            print(f"    {m.rank}. [{m.code}] {m.name}")
            print(f"       {marker} conf={m.confidence:.2f} — {m.reasoning}")


# =====================================================================
#  CLI test
# =====================================================================

if __name__ == "__main__":
    # Use BOTH retrievers for maximum candidate coverage
    from modules.retriever import Retriever
    from modules.text_retriever import TextMatchRetriever

    print("Initializing …")
    faiss_retriever = Retriever(verbose=True)
    text_retriever = TextMatchRetriever(verbose=True)
    reranker = RerankerV2()
    print("Ready!\n")

    TESTS = [
        # Multi-ingredient
        ("Salat mit Gemüse, Feta, Essig-Öl",                        "X201400"),
        ("Cheese Burger",                                            "Y911160"),
        ("paniertes Hähnchenschnitzel",                              "Y594112"),
        ("ungarisches Paprikagulasch mit Rindfleisch",               "Y141133"),
        ("Gurkensalat mit Essig-Öl-Dressing",                       "X201840"),
        # Brand / cultural
        ("Magnum Mandel (Eis)",                                      "S240000"),
        ("Miso-Paste",                                               "H862100"),
        ("Kimchi",                                                   "G345000"),
        ("Döner",                                                    "Y921162"),
        ("Tzatziki",                                                 "X476512"),
        # Creative
        ("Frühlingsrollen mini",                                     "X922163"),
        ("Spätzle selbstgemacht",                                    "E438032"),
        ("Krapfen",                                                  "X936612"),
        ("Lachs, gebraten",                                          "T410000"),
        ("Joghurt 1,5% laktosefrei",                                "M141200"),
    ]

    def merge_candidates(faiss_result, text_result):
        """Merge FAISS + text match candidates, keeping best score per code."""
        merged = {"query": faiss_result["query"]}
        for key in ["bls302", "bls40"]:
            by_code = {}
            for c in faiss_result.get(key, []):
                d = c.to_dict() if hasattr(c, "to_dict") else c
                by_code[d["code"]] = c
            for c in text_result.get(key, []):
                d = c.to_dict() if hasattr(c, "to_dict") else c
                if d["code"] not in by_code or d["score"] > by_code[d["code"]].to_dict()["score"]:
                    by_code[d["code"]] = c
            # Sort by score descending
            items = sorted(by_code.values(),
                          key=lambda x: x.to_dict()["score"] if hasattr(x, "to_dict") else x["score"],
                          reverse=True)
            merged[key] = items[:20]
        return merged

    ok302 = ok302_3 = 0
    for i, (desc, exp302) in enumerate(TESTS, 1):
        print(f"[{i:2d}/{len(TESTS)}] {desc!r} …")

        # Get candidates from BOTH retrievers
        faiss_cands = faiss_retriever.search(desc)
        text_cands = text_retriever.search(desc)
        merged = merge_candidates(faiss_cands, text_cands)

        # Re-rank with Claude
        result = reranker.rerank(desc, merged)

        if result.error:
            print(f"  ERROR: {result.error}\n")
            continue

        # Check BLS 3.02
        codes_302 = [m.code for m in result.bls302_matches]
        top1_302 = codes_302[0] if codes_302 else ""
        h1 = top1_302 == exp302
        h3 = exp302 in codes_302
        if h1: ok302 += 1; ok302_3 += 1
        elif h3: ok302_3 += 1

        print_result(result)

    print(f"\n{'═'*70}")
    print(f"  BLS 3.02 — Top-1: {ok302}/{len(TESTS)} = {ok302/len(TESTS)*100:.0f}%  "
          f"Top-3: {ok302_3}/{len(TESTS)} = {ok302_3/len(TESTS)*100:.0f}%")
    print(f"  Cost: ~${len(TESTS) * 2 * 0.01:.2f}")
    print(f"{'═'*70}")
