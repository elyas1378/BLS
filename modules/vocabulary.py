"""
BLS Vocabulary
==============
Extracts all unique words from BLS catalog German food names.
Provides O(1) lookup via BLS_VOCAB_SET and sorted list for difflib matching.

Singleton pattern: builds once on first access, reuses after.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import CATALOG_302, CATALOG_40

_vocab_set: frozenset[str] | None = None
_vocab_list: list[str] | None = None


def _build_vocabulary() -> tuple[frozenset[str], list[str]]:
    """Load both BLS catalogs and extract all unique words from name_de."""
    import pandas as pd

    words = set()

    for path, label in [(CATALOG_302, "BLS 3.02"), (CATALOG_40, "BLS 4.0")]:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        for name in df["name_de"].dropna():
            # Split on spaces, hyphens, slashes, commas, parentheses, periods
            tokens = re.split(r'[\s\-/,\(\)\.]+', name.lower())
            for token in tokens:
                # Strip remaining punctuation at edges
                token = token.strip("'\"%;:")
                if len(token) >= 3:
                    words.add(token)

    vocab_set = frozenset(words)
    vocab_list = sorted(words)
    return vocab_set, vocab_list


def get_vocab_set() -> frozenset[str]:
    """Get the BLS vocabulary as a frozenset for O(1) lookup."""
    global _vocab_set, _vocab_list
    if _vocab_set is None:
        _vocab_set, _vocab_list = _build_vocabulary()
    return _vocab_set


def get_vocab_list() -> list[str]:
    """Get the BLS vocabulary as a sorted list for difflib matching."""
    global _vocab_set, _vocab_list
    if _vocab_list is None:
        _vocab_set, _vocab_list = _build_vocabulary()
    return _vocab_list


def spell_check_tokens(tokens: list[str]) -> tuple[list[str], list[str], bool]:
    """
    Check each token against BLS vocabulary. Auto-correct close matches.

    Returns:
        corrected_tokens: list of tokens with corrections applied
        corrections_log: list of "typo -> correction" strings (for debugging)
        any_unknown: True if any token could NOT be corrected (needs Haiku)
    """
    from difflib import get_close_matches

    vocab_set = get_vocab_set()
    vocab_list = get_vocab_list()

    corrected = []
    log = []
    any_unknown = False

    for token in tokens:
        t = token.lower().strip()

        # Skip short tokens and purely numeric ones
        if len(t) < 3 or t.isdigit():
            corrected.append(t)
            continue

        # Known word — keep as-is
        if t in vocab_set:
            corrected.append(t)
            continue

        # Try difflib close match
        matches = get_close_matches(t, vocab_list, n=1, cutoff=0.82)
        if matches:
            corrected.append(matches[0])
            log.append(f"{t} -> {matches[0]}")
        else:
            # No close match — keep original, flag as unknown
            corrected.append(t)
            any_unknown = True

    return corrected, log, any_unknown


def spell_check_query(normalized_query: str) -> tuple[str, list[str], bool]:
    """
    Takes a full normalized query string, tokenizes it, spell-checks,
    and returns the corrected string reassembled.

    Returns:
        corrected_string: the corrected query
        corrections_log: list of corrections made
        any_unknown: True if any token is unknown
    """
    tokens = normalized_query.split()
    corrected, log, any_unknown = spell_check_tokens(tokens)
    return " ".join(corrected), log, any_unknown


if __name__ == "__main__":
    vocab_set = get_vocab_set()
    vocab_list = get_vocab_list()

    print(f"BLS Vocabulary Size: {len(vocab_set)} unique words\n")

    test_cases = [
        "cappucchino",
        "karoffeln",
        "coleslow",
        "knäckebort",
        "bokwürste",
        "banane",
        "grillfackeln",
        "snickers white",
        "haferflocken mit milch",
        "truthanbrust",
    ]

    print(f"{'Input':35s} {'Corrected':35s} {'Corrections':30s} {'Unknown'}")
    print("-" * 110)
    for tc in test_cases:
        corrected, log, unknown = spell_check_query(tc)
        log_str = "; ".join(log) if log else "-"
        print(f"{tc:35s} {corrected:35s} {log_str:30s} {unknown}")
