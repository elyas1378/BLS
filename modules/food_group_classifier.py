"""
FIT Study Food Group & NOVA Classifier
=======================================

Food groups: lookup table (from reference data) + rule-based fallback.
NOVA: purely rule-based via nova_classifier.py (no reference-data lookups).
No API calls — entirely free.
"""

from __future__ import annotations
from modules.food_group_map import (
    MAIN_GROUP_302, SUB_GROUP_302,
    MAIN_GROUP_40, SUB_GROUP_40,
)
from modules.nova_classifier import classify_nova


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

# NOVA classification has moved to modules/nova_classifier.py


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
    """H codes: nuts vs legumes vs plant-based alternatives."""
    if code.startswith("H84"):
        return "11_Unfavorable_foods"  # plant milks → 11.6 processed dairy alt
    if code.startswith("H86"):
        return "2_Legumes"  # soy products, tofu (Excel Group 2)
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



# ═══════════════════════════════════════════════════════════
#  Excel-scheme corrections (2026-04-21 Standardisierungsbeispiele)
# ═══════════════════════════════════════════════════════════
#
# The Excel food-group scheme disagrees with many entries in food_group_map.py.
# Rather than patch 3400 lookup rows, we apply pattern-based corrections after
# the lookup so the classifier always routes these cases to the new scheme.
#
# Covered:
#   - N2*, N3*              → 11_5_SSB   (Fruchtsaftgetränke, Limonaden, Colas,
#                                         Saftschorle, Isotonic, Getränkepulver)
#   - F code[4] == '7'      → 11_5_SSB   (Fruchtnektare — 7th-digit BLS encoding)
#   - F code[4] == '6'      → 3_2_Juices_smoothies (pure fruit juices + smoothies)
#   - M206*, M216*          → 11_5_SSB   (cocoa milk drinks)
#   - H86*                  → 2_Legumes  (soy products, tofu — already in rule-H)

_SSB = ("11_Unfavorable_foods", "11_5_SSB")
_JUICES = ("3_Fruit", "3_2_Juices_smoothies")
_SWEETS = ("11_Unfavorable_foods", "11_4_Sugar_confectionary")


def _apply_excel_scheme_corrections(code: str, main: str | None, sub: str | None):
    """Pattern-based overrides aligned with the 2026-04-21 Excel Food Groups sheet."""
    # N2*/N3* — sweetened/mixed/soft drinks → SSB
    if code[:2] in ("N2", "N3"):
        return _SSB

    # F-code 5th digit encodes preparation: 7=Nektar, 6=juice/smoothie
    if code.startswith("F") and len(code) >= 5:
        if code[4] == "7":
            return _SSB
        if code[4] == "6":
            return _JUICES

    # Cocoa milk drinks: M206* (Milcherzeugnis/Milchmischgetränk mit Kakao),
    # M216* (Trinkmilch mit Kakao) → SSB (Excel 11.5 includes cocoa).
    if code[:4] in ("M206", "M216"):
        return _SSB

    # Sweetened dairy products (chocolate yoghurt, Dickmilch mit Kakao, etc.)
    # → 11_4_Sugar_confectionary per Excel 11.4 "sweetened milk-products".
    if code[:4] in ("M226", "M236", "M246"):
        return _SWEETS

    # H86* — soy products, tofu → Legumes (Excel Group 2)
    if code.startswith("H86"):
        return "2_Legumes", "xx_none"

    return main, sub


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

    # ── Food group: lookup table then rule-based fallback ──
    if bls_version == "302":
        main = MAIN_GROUP_302.get(code)
        sub = SUB_GROUP_302.get(code)
    else:
        main = MAIN_GROUP_40.get(code)
        sub = SUB_GROUP_40.get(code)

    source = "lookup" if main is not None else "rule"

    if main is None:
        letter = code[0].upper()
        if letter == "E":
            main = _classify_main_E(code)
        elif letter == "H":
            main = _classify_main_H(code)
        elif letter in ("X", "Y"):
            main = None
        else:
            main = LETTER_TO_MAIN.get(letter)

        sub_fn = {
            "C": _classify_sub_C, "E": _classify_sub_E, "H": _classify_sub_H,
            "N": _classify_sub_N, "P": _classify_sub_P, "Q": _classify_sub_Q,
        }.get(letter)
        sub = sub_fn(code) if sub_fn else LETTER_TO_SUB.get(letter)

    # Apply Excel-scheme pattern corrections (SSBs, nectars, juices, cocoa, soy).
    main, sub = _apply_excel_scheme_corrections(code, main, sub)

    # Catch-all: codes that survived all rules without a main group (e.g. X/Y
    # recipe codes not in the lookup) land in Miscellaneous per Sidney 2026-04-23.
    if main is None:
        main = "13_Miscellaneous"
        if sub is None:
            sub = "xx_none"

    # ── NOVA: Layer 1 (code structure) + Layer 2 (description overrides) ──
    nova_result = classify_nova(code, bls_version, food_desc or "", brand)
    nova = nova_result["nova"]

    return {"main_group": main, "sub_group": sub, "nova": nova, "source": source}
