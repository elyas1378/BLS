"""
FIT Study Food Group & NOVA Classifier
=======================================

Two-tier classification:
  1. Lookup table (from reference data) — covers all 1,339 known BLS 3.02 codes
  2. Rule-based fallback (from BLS code letter patterns) — for unseen codes

Rules derived from 25,291 rows of FIT study reference data.
No API calls — entirely free.
"""

from __future__ import annotations
from modules.food_group_map import (
    MAIN_GROUP_302, SUB_GROUP_302, NOVA_302,
    MAIN_GROUP_40, SUB_GROUP_40, NOVA_40,
)


# ═══════════════════════════════════════════════════════════
#  Rule-based fallback: BLS code letter → main food group
#  Derived from patterns where one group dominates >=85%
# ═══════════════════════════════════════════════════════════

LETTER_TO_MAIN = {
    "B": "5_Carbohydrate_foods",      # 98.5%
    "C": "5_Carbohydrate_foods",      # 88.4%
    "D": "11_Unfavorable_foods",      # 95.5%
    "F": "3_Fruit",                   # 92.4%
    "G": "1_Vegetables",             # 93.7%
    "K": "5_Carbohydrate_foods",      # 77.4% (best available)
    "M": "6_Milk_milk_products",      # 98.9%
    "N": "12_Beverages",             # 85.9%
    "P": "12_Beverages",             # 98.3%
    "Q": "8_Fats_oils",             # 92.3%
    "R": "13_Miscellaneous",         # 91.8%
    "S": "11_Unfavorable_foods",      # 89.1%
    "T": "10_Fish_fish_products",    # 98.5%
    "U": "9_Meat_meat_products",     # 89.3%
    "V": "9_Meat_meat_products",     # 99.2%
    "W": "9_Meat_meat_products",     # 99.1%
    # E, H, X, Y are too mixed for a single rule — handled below
}

# E: depends on code range (eggs vs pasta vs misc)
# H: nuts dominant but very mixed
# X: vegetable-based recipes — no dominant main group
# Y: meat/fish recipes — no dominant main group

LETTER_TO_SUB = {
    "B": "5_1_Bread_cereals",         # 98.4%
    "D": "11_3_Cakes",               # 61.0% (best available)
    "F": "3_1_Whole_fruit",           # 86.3%
    "G": "xx_none",                   # 99.0%
    "K": "5_2_Starchy_sides",         # 77.4%
    "M": "xx_none",                   # 99.0%
    "S": "11_4_Sugar_confectionary",  # 83.6%
    "T": "xx_none",                   # 98.5%
    "U": "xx_none",                   # 96.8%
    "V": "xx_none",                   # 99.2%
    "W": "xx_none",                   # 98.9%
}

# ═══════════════════════════════════════════════════════════
#  Rule-based NOVA: letter (+ processing digit where clear)
#  Only rules with >=70% confidence
# ═══════════════════════════════════════════════════════════

LETTER_TO_NOVA = {
    "B": 3,  # 92% NOVA 3
    "D": 4,  # 65% NOVA 4 (best guess)
    "F": 1,  # 96% NOVA 1
    "G": 1,  # 90% NOVA 1
    "N": 1,  # 93% NOVA 1
    "P": 3,  # 90% NOVA 3
    "Q": 2,  # 86% NOVA 2
    "V": 1,  # 82% NOVA 1
    "W": 4,  # 91% NOVA 4
}


def _classify_main_E(code: str) -> str:
    """E codes: E1xx = eggs, E4xx = pasta, E1035xx = misc (starch)."""
    if len(code) >= 2:
        second = code[1]
        if second == "1":
            # E10xxxx and E11xxxx
            if code.startswith("E103") or code.startswith("E104"):
                return "13_Miscellaneous"
            return "7_Eggs"
        if second in ("4", "3"):
            return "5_Carbohydrate_foods"
    return "5_Carbohydrate_foods"


def _classify_main_H(code: str) -> str:
    """H codes: mostly nuts, but H8xx can be plant milks (unfavorable/milk)."""
    if code.startswith("H84"):
        return "11_Unfavorable_foods"  # plant milks — mapped as unfavorable in FIT
    if code.startswith("H86"):
        return "13_Miscellaneous"  # soy products, tofu
    return "4_Nuts"


def _classify_sub_E(code: str) -> str:
    if len(code) >= 2 and code[1] in ("4", "3"):
        return "5_2_Starchy_sides"
    return "xx_none"


def _classify_sub_H(code: str) -> str:
    if code.startswith("H84"):
        return "11_6_Processed_alt"
    return "xx_none"


def _classify_sub_N(code: str) -> str:
    """N codes: mostly non-alc beverages, but some are milk-based."""
    return "12_1_Non_alc_bev"


def _classify_sub_P(code: str) -> str:
    """P codes: split between alcoholic and non-alcoholic."""
    # P1-P3 tend to be alcohol-free beer/wine, P4+ alcoholic
    if code.startswith("P1") or code.startswith("P2") or code.startswith("P3"):
        return "12_1_Non_alc_bev"
    return "12_2_Alc_bev"


def _classify_sub_Q(code: str) -> str:
    """Q codes: plant oils vs animal fats."""
    if code.startswith("Q1"):
        return "8_1_Plant_oils"
    if code.startswith("Q2") or code.startswith("Q3"):
        return "8_2_Animal_fats"
    return "8_1_Plant_oils"


def _classify_sub_C(code: str) -> str:
    """C codes: cereals — bread vs starchy sides."""
    if code.startswith("C1"):
        return "5_1_Bread_cereals"  # muesli, oats, cereals
    if code.startswith("C3") or code.startswith("C5"):
        return "5_2_Starchy_sides"  # rice, corn
    return "5_1_Bread_cereals"


def _nova_fallback(code: str) -> int | None:
    """NOVA fallback for letters not in LETTER_TO_NOVA."""
    letter = code[0] if code else ""
    if letter == "C":
        return 1  # 76%
    if letter == "E":
        if code[1:2] == "1":
            return 1  # eggs
        return 1  # pasta — mostly unprocessed
    if letter == "H":
        return 1  # 72% NOVA 1
    if letter == "K":
        return 1  # 70% NOVA 1
    if letter == "M":
        return 3  # 57% NOVA 3 (slight majority)
    if letter == "R":
        return 1  # 44% — very mixed, best guess
    if letter == "S":
        return 4  # 66%
    if letter == "T":
        return 1  # 57%
    if letter == "U":
        return 1  # 54%
    # X, Y: too mixed to assign reliably
    if letter == "X":
        return 3  # 48% — slight majority
    if letter == "Y":
        return 3  # 38% — very mixed, 3 is safest middle ground
    return None


# ═══════════════════════════════════════════════════════════
#  Description-based NOVA overrides
#  Runs AFTER BLS code lookup to correct known wrong cases
# ═══════════════════════════════════════════════════════════

_NOVA4_KEYWORDS = {
    "energy drink", "energydrink", "red bull", "cola", "fanta", "sprite",
    "pepsi", "chips", "pringles", "snickers", "mars", "twix", "oreo",
    "milka", "haribo", "gummibärchen", "ketchup", "fertiggericht",
    "tiefkühl", "instant", "mikrowelle", "dosensuppe", "tütensuppe",
    "cornflakes", "toast",
}

_NOVA1_KEYWORDS = {
    "roh", "frisch",
}
# NOVA 1 only applies when combined with these food-group letters
_NOVA1_LETTERS = {"F", "G", "V", "U", "T", "E", "H", "K", "N"}

_NOVA2_KEYWORDS = {
    "olivenöl", "rapsöl", "butter", "honig", "zucker", "mehl",
    "essig", "salz", "leinöl", "sonnenblumenöl", "kokosöl", "rapsöl",
}


def _nova_override(nova: int | None, food_desc: str | None,
                   brand: str | None, code: str | None) -> int | None:
    """Apply description-based NOVA overrides on top of code-based NOVA."""
    if food_desc is None:
        return nova

    lower = food_desc.lower().strip()
    letter = code[0].upper() if code else ""

    # Rule 1: branded products → NOVA 4
    if brand is not None:
        return 4

    # Rule 2: NOVA 4 keywords
    for kw in _NOVA4_KEYWORDS:
        if kw in lower:
            return 4

    # Rule 3: NOVA 2 keywords (processed culinary ingredients)
    for kw in _NOVA2_KEYWORDS:
        if kw in lower:
            return 2

    # Rule 4: NOVA 1 keywords (only for raw foods in appropriate groups)
    if letter in _NOVA1_LETTERS:
        for kw in _NOVA1_KEYWORDS:
            if kw in lower:
                return 1

    return nova


# ═══════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════

def classify(code: str, bls_version: str = "302",
             food_desc: str | None = None,
             brand: str | None = None) -> dict:
    """
    Classify a BLS code into FIT study food groups and NOVA.

    Args:
        code: BLS code (e.g. "B173000")
        bls_version: "302" or "40"
        food_desc: original food description (for NOVA overrides)
        brand: detected brand name (for NOVA overrides)

    Returns:
        {
            "main_group": str or None,
            "sub_group": str or None,
            "nova": int or None,
            "source": "lookup" | "rule"
        }
    """
    if not code or len(code) < 2:
        return {"main_group": None, "sub_group": None, "nova": None, "source": None}

    # Tier 1: lookup table
    if bls_version == "302":
        main = MAIN_GROUP_302.get(code)
        sub = SUB_GROUP_302.get(code)
        nova = NOVA_302.get(code)
    else:
        main = MAIN_GROUP_40.get(code)
        sub = SUB_GROUP_40.get(code)
        nova = NOVA_40.get(code)

    if main is not None:
        nova = _nova_override(nova, food_desc, brand, code)
        return {"main_group": main, "sub_group": sub, "nova": nova, "source": "lookup"}

    # Tier 2: rule-based fallback
    letter = code[0].upper()

    # Main group
    if letter == "E":
        main = _classify_main_E(code)
    elif letter == "H":
        main = _classify_main_H(code)
    elif letter in ("X", "Y"):
        main = None  # too mixed — can't reliably classify
    else:
        main = LETTER_TO_MAIN.get(letter)

    # Sub group
    sub_fn = {
        "C": _classify_sub_C,
        "E": _classify_sub_E,
        "H": _classify_sub_H,
        "N": _classify_sub_N,
        "P": _classify_sub_P,
        "Q": _classify_sub_Q,
    }.get(letter)
    if sub_fn:
        sub = sub_fn(code)
    else:
        sub = LETTER_TO_SUB.get(letter)

    # NOVA
    nova = LETTER_TO_NOVA.get(letter)
    if nova is None:
        nova = _nova_fallback(code)

    nova = _nova_override(nova, food_desc, brand, code)

    return {"main_group": main, "sub_group": sub, "nova": nova, "source": "rule"}
