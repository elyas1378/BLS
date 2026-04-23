"""
NOVA Layer 3 — LLM Verification
================================
Optional verification of the rule-based NOVA classification by Claude.

Called only when the rule-based classifier has low confidence (or no
answer at all). Claude confirms or corrects via a tool-use response.

Design notes
------------
* Structured output via tool_use → Claude cannot return malformed JSON.
* Prompt caching on the system prompt → cheap across large batches.
* No separate storage medium: results are stored in the existing Google
  Sheets log tab (see PersistentCache.get_nova_cache / log_search).
  This module does NOT read or write the cache directly — the caller
  passes in a `cache_lookup` function for reads, and is expected to call
  PersistentCache.log_search() later with the returned NOVA fields to
  populate the cache for future calls.
* Fail-safe: ANY exception (missing key, network, parsing) returns None,
  so the caller falls back to the rule-based result. Never raises.
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Callable, Optional

# ─────────────────────────────────────────────────────────────
#  Prompt (cached by Anthropic for reuse across calls)
# ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You classify foods into NOVA (1–4) for a German nutrition-research tool (FoodMatch).

NOVA levels:
  1 — Unprocessed / minimally processed: fresh, dried, frozen, ground or
      chopped natural foods (fruit, vegetables, meat, fish, eggs, milk,
      water, coffee, tea, pasta, rice, flour, nuts).
  2 — Processed culinary ingredients derived from Group 1: sugar, oils,
      butter, lard, salt, vinegar, baking powder.
  3 — Processed foods = combinations of 1 + 2: bread, cheese, jam, canned
      foods, pickled vegetables, beer, wine, aged cheese, cured/smoked
      meat and fish, yoghurt, pure fruit juice, smoothies.
  4 — Ultra-processed foods: industrial products with additives (soft
      drinks, energy drinks, sweets, chips, margarine, instant meals,
      fast food, ready sauces, sweetened dairy, nectars, mixed juice
      drinks, processed meat like sausages/nuggets).

Principle: classify by the MOST COMMON form of consumption in the German
general population, with preference for minimally processed or
home-prepared equivalents where applicable. Home-baked cakes → 3.
Industrial brand cakes → 4. Generic unlabelled items default to the
typical supermarket product.

You will receive a food description, its BLS code, and the rule-based
guess from FoodMatch. Either confirm the guess or correct it. Be strict:
only disagree when the rule-based answer is clearly wrong by the
principle above. Respond via the nova_verdict tool."""


_NOVA_TOOL = {
    "name": "nova_verdict",
    "description": (
        "Record the verified NOVA classification and whether you agree "
        "with the rule-based guess."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "nova": {
                "type": "integer",
                "enum": [1, 2, 3, 4],
                "description": "Final NOVA level (1–4).",
            },
            "agree": {
                "type": "boolean",
                "description": "True if the final NOVA matches the rule-based guess.",
            },
            "reason": {
                "type": "string",
                "description": "One short sentence justifying the verdict.",
            },
        },
        "required": ["nova", "agree", "reason"],
    },
}


def normalize_key(s: str) -> str:
    """Stable key for matching descriptions in the Google Sheets cache."""
    s = unicodedata.normalize("NFKC", s or "").lower().strip()
    return re.sub(r"\s+", " ", s)


# ─────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────

def verify_nova(
    code: str,
    description: str,
    rule_based_nova: Optional[int],
    rule_based_reason: str = "",
    *,
    cache_lookup: Optional[Callable[[str, str], Optional[dict]]] = None,
    client=None,
    model: Optional[str] = None,
) -> Optional[dict]:
    """Ask Claude to confirm/correct a rule-based NOVA classification.

    Parameters
    ----------
    code, description, rule_based_nova, rule_based_reason
        Context passed to Claude.
    cache_lookup
        Optional function ``(code, description) -> dict | None`` that
        returns a prior LLM verdict for this food. When supplied, it is
        called first; on a hit the API call is skipped entirely.
    client
        Optional preconstructed ``anthropic.Anthropic`` client. If None,
        one is created from ``ANTHROPIC_API_KEY``.
    model
        Override ``config.settings.CLAUDE_MODEL``.

    Returns
    -------
    dict on success:
        { "nova": 1..4, "agree": bool, "reason": str, "source": "cache"|"llm" }
    None on any failure — caller should fall back to rule-based result.
    """
    if not (description or "").strip():
        return None

    # 1) Cache hit (Google Sheets log tab, via injected lookup)
    if cache_lookup is not None:
        try:
            hit = cache_lookup(code, description)
        except Exception:
            hit = None
        if hit:
            return {
                "nova": int(hit["nova"]),
                "agree": bool(hit.get("agree", hit.get("nova") == rule_based_nova)),
                "reason": str(hit.get("reason", "")),
                "source": "cache",
            }

    # 2) Miss → call Claude
    try:
        import anthropic  # lazy import — no cost if verifier is never called
        from config.settings import CLAUDE_MODEL

        if client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                return None
            client = anthropic.Anthropic(api_key=api_key)

        model_name = model or CLAUDE_MODEL

        user_text = (
            f"Food: {description}\n"
            f"BLS code: {code or '(none)'}\n"
            f"Rule-based guess: NOVA "
            f"{rule_based_nova if rule_based_nova else '(no answer)'}\n"
            f"Rule-based reason: {rule_based_reason or '(n/a)'}\n\n"
            "Verify or correct."
        )

        response = client.messages.create(
            model=model_name,
            max_tokens=200,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[_NOVA_TOOL],
            tool_choice={"type": "tool", "name": "nova_verdict"},
            messages=[{"role": "user", "content": user_text}],
        )

        # Extract the tool_use block (guaranteed by tool_choice)
        tool_input = None
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "nova_verdict":
                tool_input = block.input
                break
        if not isinstance(tool_input, dict):
            return None

        nova = int(tool_input.get("nova", 0))
        if nova not in (1, 2, 3, 4):
            return None
        reason = str(tool_input.get("reason", ""))[:300]
        agree = (nova == rule_based_nova)

        return {"nova": nova, "agree": agree, "reason": reason, "source": "llm"}

    except Exception:
        return None
