"""
Re-ranker Module
================
Takes the top-N candidates from FAISS retrieval and uses Claude to
intelligently select the best 3 matches, considering preparation state,
brand mapping, cultural knowledge, and BLS code structure.

Prerequisites:
    pip install anthropic

Usage:
    from modules.reranker import Reranker
    from modules.retriever import Retriever

    retriever = Retriever()
    reranker = Reranker()

    candidates = retriever.search("Eier")
    result = reranker.rerank("Eier", candidates)
    print(result)

Environment:
    Set ANTHROPIC_API_KEY before running.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field

# ── Make config importable ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import CLAUDE_MODEL, FINAL_TOP_K, FOOD_GROUP_LETTERS, PROCESSING_STATES


SYSTEM_PROMPT = """You are an expert nutritionist who specializes in the German Bundeslebensmittelschlüssel (BLS) food composition database. Your task is to match free-text food descriptions from German nutrition study participants to the correct BLS codes.

## BLS Code Structure
All codes are 7 characters: 1 letter + 6 digits (e.g., "E111132").

**First letter = food group:**
{food_groups}

**Last digit (position 7) = processing state:**
{processing_states}

## Matching Rules

1. **Preparation state defaults:**
   - Eggs ("Eier") without prep info → gekocht (boiled), code ends in 3
   - Meat without prep info → gegart (cooked generic), code ends in 2
   - Vegetables without prep info → roh (raw), code ends in 0 or 1
   - Fruits → always roh (raw) unless stated otherwise
   - Bread, cheese, oils → no processing state needed

2. **Synonym awareness:**
   - "Eier" = Hühnerei Vollei
   - "Spiegelei" = Hühnerei gebraten (fried), code ends in 8
   - "Nudeln" = Teigwaren eifrei
   - "Pommes" = Pommes Frites (category K or X)
   - "Poree" = Porree (Lauch)

3. **Brand → generic mapping:**
   - Brand names should map to the generic BLS category
   - "Maggi" → Grundsoße braun aus Trockenprodukt
   - "Nutella" → Nuss-Nougat-Creme
   - "Haribo" → Gummibonbon/Fruchtgummi

4. **Composite dishes:**
   - Multi-ingredient descriptions → look for X (vegetable/carb-based recipes) or Y (meat/fish-based recipes) categories
   - "Kürbiscurry" → X category (vegetable-based recipe)
   - "Bifteki mit Schafskäse" → Y category (meat-based recipe)

5. **Fat percentage:**
   - When fat % is given, match the closest BLS entry with that fat content
   - "Edamer 40% Fett" → "Edamer mind. 40% Fett i. Tr."

6. **Ambiguous terms:**
   - "Käse" without qualifier → use generic cheese code (M400000)
   - "Salat" without qualifier → Kopfsalat/Blattsalat (G category), not mixed salad recipe (X category)

## Response Format
Respond with ONLY a JSON array of exactly {top_k} matches, ordered by confidence. No other text.

```json
[
  {{
    "rank": 1,
    "code": "E111132",
    "name": "Hühnerei Vollei gekocht",
    "confidence": 0.92,
    "reasoning": "Eier is colloquial for Hühnerei; default prep is gekocht"
  }},
  ...
]
```

**Confidence scoring:**
- 0.90–1.0: Near-exact match (e.g., "Banane" → "Banane roh")
- 0.70–0.89: Strong match with minor ambiguity (e.g., "Eier" → gekocht vs gebraten)
- 0.50–0.69: Reasonable but alternatives plausible (e.g., "Käse" → multiple types)
- Below 0.50: Uncertain, needs manual review
"""


@dataclass
class RankedMatch:
    """A single re-ranked match from Claude."""
    rank: int
    code: str
    name: str
    confidence: float
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "code": self.code,
            "name": self.name,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


@dataclass
class RerankerResult:
    """Complete re-ranking result for one food description."""
    food_description: str
    bls302_matches: list[RankedMatch] = field(default_factory=list)
    bls40_matches: list[RankedMatch] = field(default_factory=list)
    error: str | None = None


class Reranker:
    """Uses Claude to re-rank FAISS retrieval candidates."""

    def __init__(self, api_key: str | None = None):
        """
        Initialize the Anthropic client.

        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
        """
        import anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "No API key found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.client = anthropic.Anthropic(api_key=key)
        self.model = CLAUDE_MODEL
        self.top_k = FINAL_TOP_K

        # Build system prompt
        fg = "\n".join(f"  {k} = {v}" for k, v in FOOD_GROUP_LETTERS.items())
        ps = "\n".join(f"  {k} = {v}" for k, v in PROCESSING_STATES.items())
        self.system_prompt = SYSTEM_PROMPT.format(
            food_groups=fg,
            processing_states=ps,
            top_k=FINAL_TOP_K,
        )

    def _build_user_prompt(
        self,
        food_description: str,
        candidates: list[dict],
        bls_version: str,
        prep_state: str | None = None,
        fat_percent: str | None = None,
    ) -> str:
        """Build the user message with the food description and candidate list."""
        parts = [f"## Food description to match:\n\"{food_description}\""]

        # Add context if available
        context_parts = []
        if prep_state:
            context_parts.append(f"Detected preparation state: {prep_state}")
        if fat_percent:
            context_parts.append(f"Detected fat percentage: {fat_percent}%")
        if context_parts:
            parts.append("\n## Additional context:\n" + "\n".join(context_parts))

        parts.append(f"\n## BLS version: {bls_version}")

        # Format candidates
        parts.append(f"\n## Top {len(candidates)} candidates from semantic search:")
        for i, c in enumerate(candidates, 1):
            line = f"  {i:2d}. [{c['code']}] {c['name_de']}"
            if c.get('name_en'):
                line += f" / {c['name_en']}"
            line += f"  (similarity: {c['score']:.3f})"
            if c.get('food_group'):
                fg_label = FOOD_GROUP_LETTERS.get(c['food_group'], '')
                line += f"  [group: {c['food_group']}={fg_label}]"
            parts.append(line)

        parts.append(f"\nSelect the best {self.top_k} matches. Respond with ONLY a JSON array.")
        return "\n".join(parts)

    def _call_claude(self, user_prompt: str) -> list[RankedMatch]:
        """Send prompt to Claude and parse the response."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.0,
        )

        # Extract text content
        text = response.content[0].text.strip()

        # Parse JSON — handle markdown code blocks
        if text.startswith("```"):
            # Remove ```json ... ``` wrapper
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        matches = json.loads(text)

        results = []
        for m in matches:
            results.append(RankedMatch(
                rank=m["rank"],
                code=m["code"],
                name=m["name"],
                confidence=m["confidence"],
                reasoning=m["reasoning"],
            ))
        return results

    def rerank(
        self,
        food_description: str,
        retrieval_results: dict,
    ) -> RerankerResult:
        """
        Re-rank FAISS candidates using Claude.

        Args:
            food_description: original food description text
            retrieval_results: output from Retriever.search() with keys
                               "query", "bls302", "bls40"

        Returns:
            RerankerResult with top-3 matches for both BLS versions
        """
        result = RerankerResult(food_description=food_description)
        nq = retrieval_results["query"]

        try:
            # ── BLS 3.02 ──
            if retrieval_results["bls302"]:
                candidates_302 = [c.to_dict() for c in retrieval_results["bls302"]]
                prompt_302 = self._build_user_prompt(
                    food_description=food_description,
                    candidates=candidates_302,
                    bls_version="BLS 3.02",
                    prep_state=nq.prep_state,
                    fat_percent=nq.fat_percent,
                )
                result.bls302_matches = self._call_claude(prompt_302)

            # ── BLS 4.0 ──
            if retrieval_results["bls40"]:
                candidates_40 = [c.to_dict() for c in retrieval_results["bls40"]]
                prompt_40 = self._build_user_prompt(
                    food_description=food_description,
                    candidates=candidates_40,
                    bls_version="BLS 4.0",
                    prep_state=nq.prep_state,
                    fat_percent=nq.fat_percent,
                )
                result.bls40_matches = self._call_claude(prompt_40)

        except Exception as e:
            result.error = str(e)

        return result

    def rerank_batch(
        self,
        descriptions: list[str],
        retrieval_results_list: list[dict],
    ) -> list[RerankerResult]:
        """Re-rank multiple food descriptions."""
        results = []
        for desc, ret in zip(descriptions, retrieval_results_list):
            results.append(self.rerank(desc, ret))
        return results


# =====================================================================
#  Pretty print helper
# =====================================================================

def print_result(result: RerankerResult):
    """Pretty-print a RerankerResult."""
    print(f"\n{'═'*70}")
    print(f"  Food: {result.food_description!r}")

    if result.error:
        print(f"  ERROR: {result.error}")
        return

    for version, matches in [("BLS 3.02", result.bls302_matches),
                              ("BLS 4.0", result.bls40_matches)]:
        print(f"\n  {version}:")
        for m in matches:
            # Color code confidence
            if m.confidence >= 0.85:
                conf_marker = "🟢"
            elif m.confidence >= 0.60:
                conf_marker = "🟡"
            else:
                conf_marker = "🔴"
            print(f"    {m.rank}. [{m.code}] {m.name}")
            print(f"       {conf_marker} confidence: {m.confidence:.2f}")
            print(f"       {m.reasoning}")


# =====================================================================
#  CLI test
# =====================================================================

if __name__ == "__main__":
    from modules.retriever import Retriever

    print("Initializing …")
    retriever = Retriever(verbose=True)
    reranker = Reranker()
    print("Ready!\n")

    test_queries = [
        "Banane",
        "Eier",
        "Olivenöl",
        "Edamer (40% Fett)",
        "Vollkornnudeln",
        "Chicken salad",
        "Kürbiscurry",
        "Wasser",
        "Poree",
        "Kaffee mit Milch",
    ]

    for query in test_queries:
        print(f"\nSearching for: {query!r} …")
        candidates = retriever.search(query)
        result = reranker.rerank(query, candidates)
        print_result(result)

    print(f"\n{'═'*70}")
    print("All tests complete ✓")
