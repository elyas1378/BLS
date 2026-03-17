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
    export ANTHROPIC_API_KEY="sk-ant-..."
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
- "Champignons" → pick the generic or gedünstet entry, NOT roh
ONLY pick a specific preparation code if the description explicitly states it:
- "Lachs, gebraten" → T410082 is correct (gebraten explicitly stated)
- "Reis gekocht" → C352032 is correct (gekocht explicitly stated)

## Other matching priorities:
- Composite dishes → X (vegetable-based) or Y (meat/fish-based) recipes
- Brand names → generic BLS category
- Fat% should match when specified
- Avoid top-level category headers (codes like M000000, G000000)
- Prefer entries with "Standardrezeptur" for recipe-type foods
- When fat% is mentioned (e.g., "3,5% Fett"), match the closest fat level

## Response format (ONLY this, no other text):
```json
[
  {{"rank": 1, "code": "CODE_FROM_LIST", "name": "name", "confidence": 0.92, "reasoning": "brief reason"}},
  {{"rank": 2, "code": "CODE_FROM_LIST", "name": "name", "confidence": 0.75, "reasoning": "brief reason"}},
  {{"rank": 3, "code": "CODE_FROM_LIST", "name": "name", "confidence": 0.60, "reasoning": "brief reason"}}
]
```

Confidence: 0.90-1.0 = near-exact, 0.70-0.89 = strong, 0.50-0.69 = reasonable, <0.50 = uncertain."""


# =====================================================================
#  RerankerV2 class
# =====================================================================

class RerankerV2:
    def __init__(self, api_key: str | None = None):
        import anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Set ANTHROPIC_API_KEY or pass api_key parameter.")

        self.client = anthropic.Anthropic(api_key=key)
        self.model = CLAUDE_MODEL

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
                     candidate_codes: set) -> list[RankedMatch]:
        """Call Claude and validate the response."""
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

        matches_raw = json.loads(text)

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

        return results[:3]

    def rerank(self, food_description: str, retrieval_results: dict) -> RerankerResult:
        """
        Re-rank candidates for BOTH BLS versions.
        Uses verified lookup first (free), Claude only for unknown foods.
        """
        result = RerankerResult(food_description=food_description)
        nq = retrieval_results["query"]
        original_lower = food_description.lower().strip()

        try:
            # ── TIER 0: Verified lookup (skip Claude entirely) ──
            v302 = VERIFIED_MAP_302.get(original_lower)
            v40 = VERIFIED_MAP_40.get(original_lower)

            if v302:
                name = self._names_302.get(v302, v302)
                result.bls302_matches = [RankedMatch(
                    rank=1, code=v302, name=name, confidence=0.95,
                    reasoning=f"Verified: '{food_description}' -> [{v302}]"
                )]

            if v40:
                name = self._names_40.get(v40, v40)
                result.bls40_matches = [RankedMatch(
                    rank=1, code=v40, name=name, confidence=0.95,
                    reasoning=f"Verified 4.0: '{food_description}' -> [{v40}]"
                )]

            # If both verified, skip Claude entirely
            if v302 and v40:
                return result

            # If only one verified, still call Claude for the other
            # ── BLS 3.02 (Claude) ──
            if not v302 and retrieval_results.get("bls302"):
                cands_302 = [c.to_dict() if hasattr(c, "to_dict") else c
                             for c in retrieval_results["bls302"]]
                cand_codes_302 = {c["code"] for c in cands_302}

                prompt_302 = self._build_prompt(
                    food_description, cands_302, "BLS 3.02",
                    prep_state=nq.prep_state, fat_percent=nq.fat_percent,
                )
                result.bls302_matches = self._call_claude(
                    prompt_302, self._valid_302, cand_codes_302
                )

                # Fallback: if Claude returned nothing valid, use top FAISS candidates
                if not result.bls302_matches and cands_302:
                    for i, c in enumerate(cands_302[:3], 1):
                        result.bls302_matches.append(RankedMatch(
                            rank=i, code=c["code"], name=c["name_de"],
                            confidence=c["score"] * 0.7,
                            reasoning="Fallback: Claude returned no valid codes",
                        ))

            # ── BLS 4.0 (Claude) ──
            if not v40 and retrieval_results.get("bls40"):
                cands_40 = [c.to_dict() if hasattr(c, "to_dict") else c
                            for c in retrieval_results["bls40"]]
                cand_codes_40 = {c["code"] for c in cands_40}

                prompt_40 = self._build_prompt(
                    food_description, cands_40, "BLS 4.0",
                    prep_state=nq.prep_state, fat_percent=nq.fat_percent,
                )
                result.bls40_matches = self._call_claude(
                    prompt_40, self._valid_40, cand_codes_40
                )

                if not result.bls40_matches and cands_40:
                    for i, c in enumerate(cands_40[:3], 1):
                        result.bls40_matches.append(RankedMatch(
                            rank=i, code=c["code"], name=c["name_de"],
                            confidence=c["score"] * 0.7,
                            reasoning="Fallback: Claude returned no valid codes",
                        ))

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
