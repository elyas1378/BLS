"""
UPF Classifier — Freiburger NOVA-4 Decision Framework
=====================================================
Reads the Freiburger Ernaehrungsprotokoll Excel table (~190 entries)
and classifies food descriptions as UPF (NOVA 4) or nicht UPF.

Independent from the main NOVA 1-4 classifier — they can disagree.
"""

from __future__ import annotations

from pathlib import Path

import openpyxl

# ── Load Freiburger table at import time ──

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "freiburger_nova.xlsx"

# Dict: lowercased food name -> True (NOVA 4) / False (nicht 4)
_FREIBURGER_TABLE: dict[str, bool] = {}


def _load_table() -> None:
    """Parse the Excel file and populate _FREIBURGER_TABLE."""
    if not _DATA_PATH.exists():
        return
    wb = openpyxl.load_workbook(_DATA_PATH, read_only=True, data_only=True)
    ws = wb.active
    for row in ws.iter_rows(min_row=2, values_only=True):  # skip header row
        food_name = row[0]
        nova_col = row[3]
        if food_name is None or nova_col is None:
            continue  # section header (Brot, OBST, etc.)
        name = str(food_name).strip().lower()
        if not name:
            continue
        if nova_col == 4 or str(nova_col).strip() == "4":
            _FREIBURGER_TABLE[name] = True
        elif str(nova_col).strip().lower() == "nicht 4":
            _FREIBURGER_TABLE[name] = False
    wb.close()


_load_table()

# Pre-sort by length descending so longer (more specific) entries match first
_SORTED_KEYS = sorted(_FREIBURGER_TABLE.keys(), key=len, reverse=True)


# ── Modifier overrides ──

_HOMEMADE_MARKERS = {"selbstgebacken", "selbstgemacht", "hausgemacht", "homemade"}
_INDUSTRIAL_MARKERS = {"konserve", "dose", "fertiggericht", "instant", "fertig-"}


def _check_modifiers(text_lower: str) -> bool | None:
    """Check modifier keywords that override the table lookup.

    Returns True (force UPF), False (force not UPF), or None (no override).
    """
    for marker in _HOMEMADE_MARKERS:
        if marker in text_lower:
            return False
    for marker in _INDUSTRIAL_MARKERS:
        if marker in text_lower:
            return True
    return None


# ── Public API ──

def classify_upf(food_description: str) -> bool | None:
    """Classify a food description against the Freiburger NOVA-4 table.

    Returns:
        True  — UPF (NOVA 4) according to Freiburger framework
        False — nicht UPF (nicht 4)
        None  — no match in the table
    """
    if not food_description or not _FREIBURGER_TABLE:
        return None

    lower = food_description.lower().strip()

    # 1. Check modifier overrides first
    modifier = _check_modifiers(lower)
    if modifier is not None:
        return modifier

    # 2. Substring match against table (longest match wins)
    for key in _SORTED_KEYS:
        if key in lower:
            return _FREIBURGER_TABLE[key]

    return None
