"""
BLS Food Code Matcher — Cloud Version
======================================
Streamlit Cloud deployment. Text matching + concept expansion + Claude API.
Run locally:  streamlit run app.py
"""

import os
import sys
import re
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.food_group_classifier import classify
from modules.nova_classifier import classify_nova, needs_claude_nova
from modules.claude_cache import ClaudeCache


def resolve_nova(code, bls_version, food_desc, brand, claude_nova=None, used_claude=False):
    """Resolve final NOVA: Layer 1+2, then Claude override if low confidence.

    Returns: (nova, low_confidence)
    """
    nova_result = classify_nova(code, bls_version, food_desc or "", brand)
    nova = nova_result["nova"]
    if nova_result["needs_claude"] and used_claude and claude_nova is not None:
        return claude_nova, False
    return nova, nova_result["needs_claude"]

st.set_page_config(
    page_title="BLS Food Code Matcher",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ──
st.markdown("""
<style>
    /* Hide sidebar + clean header */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    /* Centered main area */
    .stMainBlockContainer {
        padding-top: 1rem !important;
        max-width: 900px !important;
        margin: 0 auto !important;
    }

    /* Page background */
    .stApp {
        background: #fafafa !important;
    }

    /* Landing hero */
    .search-hero {
        text-align: center;
        padding-top: 10vh;
        margin-bottom: 0;
    }
    .hero-title {
        font-size: 44px;
        font-weight: 600;
        font-style: normal !important;
        color: #1d1d1f;
        letter-spacing: -0.03em;
        line-height: 1.12;
        margin: 0 0 14px;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', sans-serif;
    }
    .hero-sub {
        font-size: 16px;
        color: #86868b;
        font-weight: 400;
        line-height: 1.6;
        margin: 0 0 44px;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', sans-serif;
    }

    /* Apple-style search bar */
    div[data-testid="stTextInput"] > div {
        max-width: 640px;
        margin: 0 auto;
    }
    div[data-testid="stTextInput"] input {
        background: #f5f5f7 !important;
        border: none !important;
        border-color: transparent !important;
        outline: none !important;
        border-radius: 14px !important;
        padding: 16px 22px !important;
        font-size: 16px !important;
        color: #1d1d1f !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', sans-serif;
        transition: all 0.2s ease;
    }
    div[data-testid="stTextInput"] input::placeholder {
        color: #86868b !important;
    }
    div[data-testid="stTextInput"] input:focus {
        background: #f0f0f2 !important;
        border: none !important;
        box-shadow: 0 0 0 3px rgba(29, 158, 117, 0.12) !important;
    }

    /* Claude-style pills */
    .hero-pills {
        display: flex;
        gap: 8px;
        justify-content: center;
        flex-wrap: wrap;
        margin-top: 28px;
    }
    .hero-pill {
        background: rgba(29, 158, 117, 0.05);
        color: #1a7a5a;
        font-size: 13px;
        padding: 7px 16px;
        border-radius: 10px;
        font-weight: 500;
        cursor: pointer;
        border: 1px solid rgba(29, 158, 117, 0.08);
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', sans-serif;
        transition: all 0.2s ease;
    }
    .hero-pill:hover {
        background: rgba(29, 158, 117, 0.10);
        border-color: rgba(29, 158, 117, 0.18);
    }

    /* Footer */
    .hero-footer {
        margin-top: 52px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        font-size: 11px;
        color: #c7c7cc;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', sans-serif;
    }
    .hero-dot {
        width: 5px;
        height: 5px;
        border-radius: 50%;
        background: #1D9E75;
        display: inline-block;
    }

    /* Result card */
    .result-card-new {
        background: #ffffff !important;
        border-radius: 16px;
        border: 0.5px solid #e5e5ea;
        padding: 24px 28px;
        margin-bottom: 16px;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', sans-serif;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .result-card-new.warning-orange { border-left: 3px solid #ff9500; }
    .result-card-new.warning-red { border-left: 3px solid #ff3b30; }

    .rc-header {
        display: flex; justify-content: space-between;
        align-items: flex-start; margin-bottom: 16px;
    }
    .rc-food-name {
        font-size: 20px; font-weight: 600; color: #1d1d1f;
        letter-spacing: -0.02em;
    }
    .rc-header-left { flex: 1; min-width: 0; }
    .rc-food-meta { font-size: 13px; color: #86868b; margin-top: 4px; }
    .rc-header-right {
        display: flex; gap: 6px; align-items: center; flex-shrink: 0;
    }

    .rc-badge {
        font-size: 11px; padding: 2px 8px; border-radius: 6px; font-weight: 500;
    }
    .rc-badge-verified { background: #f0faf0; color: #248a24; }
    .rc-badge-cached { background: #f0f0ff; color: #5856d6; }
    .rc-badge-api { background: #faf0ff; color: #af52de; }
    .rc-badge-brand { background: #fff8f0; color: #ff9500; }

    .rc-nova {
        font-size: 11px; padding: 4px 10px; border-radius: 8px; font-weight: 500;
    }
    .rc-nova-1 { background: #e8f8e8; color: #248a24; }
    .rc-nova-2 { background: #fffde8; color: #8a7a24; }
    .rc-nova-3 { background: #fff4e8; color: #9a6700; }
    .rc-nova-4 { background: #fff0f0; color: #ff3b30; }

    .rc-foodgroup {
        font-size: 11px; padding: 4px 10px; border-radius: 8px;
        font-weight: 500; background: #f0f0ff; color: #5856d6;
    }

    .rc-warning {
        border-radius: 8px; padding: 10px 14px; margin-bottom: 16px;
        font-size: 12px; line-height: 1.4;
    }
    .rc-warning-orange { background: #fff8f0; color: #9a6700; }
    .rc-warning-red { background: #fff0f0; color: #ff3b30; }
    .rc-warning-yellow { background: #fffbf0; color: #9a6700; }

    .rc-grid {
        display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
    }
    .rc-bls-col {
        background: #f5f5f7; border-radius: 12px; padding: 16px;
    }
    .rc-bls-header {
        display: flex; justify-content: space-between;
        align-items: center; margin-bottom: 10px;
    }
    .rc-bls-label { font-size: 12px; color: #86868b; font-weight: 500; }
    .rc-bls-conf { font-size: 12px; font-weight: 600; }
    .rc-conf-green { color: #34c759; }
    .rc-conf-yellow { color: #ff9500; }
    .rc-conf-red { color: #ff3b30; }

    .rc-progress {
        height: 3px; background: #e5e5ea; border-radius: 2px; margin-bottom: 12px;
    }
    .rc-progress-fill {
        height: 100%; border-radius: 2px; transition: width 0.6s ease;
    }
    .rc-conf-bg-green { background: #34c759; }
    .rc-conf-bg-yellow { background: #ff9500; }
    .rc-conf-bg-red { background: #ff3b30; }

    .rc-bls-code {
        font-size: 15px; font-weight: 600; color: #1d1d1f;
        font-family: SFMono-Regular, Menlo, 'Courier New', monospace;
        letter-spacing: 0.02em; margin-bottom: 4px;
    }
    .rc-bls-name { font-size: 12px; color: #3c3c43; line-height: 1.4; }

    .rc-footer {
        display: flex; justify-content: space-between; align-items: center;
        margin-top: 14px; padding-top: 12px; border-top: 0.5px solid #e5e5ea;
    }
    .rc-source { font-size: 12px; color: #86868b; }

    /* Small action buttons row */
    .action-row {
        display: flex; justify-content: flex-end; align-items: center;
        gap: 8px; margin-top: -8px; margin-bottom: 12px;
    }
    .action-row .stButton > button {
        font-size: 12px !important;
        padding: 4px 16px !important;
        min-height: 0 !important;
        height: auto !important;
        line-height: 1.4 !important;
        border-radius: 8px !important;
        border: 0.5px solid #e5e5ea !important;
        color: #86868b !important;
        background: transparent !important;
        white-space: nowrap !important;
    }
    .action-row .stButton > button:hover {
        background: #f5f5f7 !important;
        color: #1d1d1f !important;
    }

</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  Persistent state
# ═══════════════════════════════════════════════════════════

@st.cache_resource
def get_cache():
    return ClaudeCache()

@st.cache_resource
def load_text_retriever():
    from modules.text_retriever import TextMatchRetriever
    return TextMatchRetriever(verbose=False)

for key, default in [("cache_hits", 0), ("api_calls", 0), ("requery_food", None),
                     ("unmatched_foods", []), ("flagged_this_session", set())]:
    if key not in st.session_state:
        st.session_state[key] = default


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════

FLAG_FILE = Path(__file__).resolve().parent / "cache" / "flagged_results.json"


def save_flag(entry: dict):
    """Append a flagged result to the JSON log."""
    import json
    FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
    flags = []
    if FLAG_FILE.exists():
        try:
            with open(FLAG_FILE, "r", encoding="utf-8") as f:
                flags = json.load(f)
        except (json.JSONDecodeError, IOError):
            flags = []
    flags.append(entry)
    with open(FLAG_FILE, "w", encoding="utf-8") as f:
        json.dump(flags, f, ensure_ascii=False, indent=1)


def load_flags() -> list[dict]:
    """Load all flagged results."""
    import json
    if FLAG_FILE.exists():
        try:
            with open(FLAG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def get_expander(api_key):
    if not api_key:
        return None
    try:
        from modules.query_expander import QueryExpander
        return QueryExpander(api_key=api_key)
    except Exception:
        return None


def get_reranker(api_key):
    if not api_key:
        return None
    try:
        from modules.reranker_v2 import RerankerV2
        return RerankerV2(api_key=api_key, cache=get_cache())
    except Exception as e:
        st.warning(f"Could not load Claude reranker: {e}")
        return None


def get_smart_reranker():
    from modules.smart_reranker import SmartReranker
    return SmartReranker(enable_llm=False)


def safety_flag(result, food_desc=None, brand=None):
    """Compute safety flag from a RerankerResult."""
    m302 = result.bls302_matches
    m40 = result.bls40_matches
    if (m302 and not m40) or (m40 and not m302):
        return "check"
    if not m302 and not m40:
        return "check"
    top302 = m302[0] if m302 else None
    top40 = m40[0] if m40 else None
    if top302 and top302.confidence < 0.60:
        return "check"
    if top40 and top40.confidence < 0.60:
        return "check"
    if top302 and top40 and top302.code[0] != top40.code[0]:
        return "check"
    if top302 and top40:
        cls302 = classify(top302.code, "302", food_desc=food_desc, brand=brand)
        cls40 = classify(top40.code, "40", food_desc=food_desc, brand=brand)
        if cls302["main_group"] and cls40["main_group"]:
            if cls302["main_group"] != cls40["main_group"]:
                return "check"
    return "ok"


def _conf_class(c):
    if c >= 0.85: return "green"
    if c >= 0.60: return "yellow"
    return "red"


def _bls_col_html(matches, label, source):
    """Render one BLS column (3.02 or 4.0)."""
    if not matches:
        return (f'<div class="rc-bls-col">'
                f'<div class="rc-bls-header"><span class="rc-bls-label">{label}</span></div>'
                f'<div class="rc-bls-code" style="color:#86868b;">—</div>'
                f'<div class="rc-bls-name" style="font-style:italic;">No suitable match</div></div>')
    m = matches[0]
    pct = int(m.confidence * 100)
    cc = _conf_class(m.confidence)
    src_map = {"verified": "rc-badge-verified", "cached": "rc-badge-cached", "api": "rc-badge-api"}
    src_lbl = {"verified": "verified", "cached": "cached", "api": "live API"}
    src_html = ""
    if source in src_map:
        src_html = f' <span class="rc-badge {src_map[source]}">{src_lbl[source]}</span>'
    return (f'<div class="rc-bls-col">'
            f'<div class="rc-bls-header">'
            f'<span class="rc-bls-label">{label}{src_html}</span>'
            f'<span class="rc-bls-conf rc-conf-{cc}">{pct}%</span></div>'
            f'<div class="rc-progress"><div class="rc-progress-fill rc-conf-bg-{cc}" style="width:{pct}%"></div></div>'
            f'<div class="rc-bls-code">{m.code}</div>'
            f'<div class="rc-bls-name">{m.name}</div></div>')


def render_result_card(query_text, nq, result, flag, src302, src40,
                       claude_nova=None, used_claude=False, source_text=""):
    """Render the unified result card."""
    m302 = result.bls302_matches or []
    m40 = result.bls40_matches or []
    top1_302 = m302[0].confidence if m302 else 0
    top1_40 = m40[0].confidence if m40 else 0
    top1_conf = max(top1_302, top1_40)

    # Card warning class
    card_class = "result-card-new"
    if top1_conf < 0.30:
        card_class += " warning-red"
    elif top1_conf < 0.60:
        card_class += " warning-orange"

    # NOVA + food group from top BLS 3.02 match
    cls = classify(m302[0].code, "302", food_desc=query_text, brand=nq.brand) if m302 else {}
    nova = None
    if m302:
        nova, _ = resolve_nova(m302[0].code, "302", query_text, nq.brand, claude_nova, used_claude)
    nova_cls = f"rc-nova-{nova}" if nova else ""
    nova_html = f'<span class="rc-nova {nova_cls}">NOVA {nova}</span>' if nova else ""
    fg_raw = (cls.get("main_group") or "").replace("_", " ")
    fg = re.sub(r'^\d+\s*', '', fg_raw)  # strip leading "11 " etc.
    fg_html = f'<span class="rc-foodgroup">{fg}</span>' if fg and fg != "—" else ""

    # Source badge
    src_badge_cls = {"verified": "rc-badge-verified", "cached": "rc-badge-cached", "api": "rc-badge-api"}
    src_badge_lbl = {"verified": "verified", "cached": "cached", "api": "live API"}
    meta_badge = ""
    src_key = src302 or src40
    if src_key in src_badge_cls:
        meta_badge = f' <span class="rc-badge {src_badge_cls[src_key]}">{src_badge_lbl[src_key]}</span>'
    brand_badge = f' <span class="rc-badge rc-badge-brand">brand</span>' if nq.brand else ""

    # Warnings
    warnings_html = ""
    if top1_conf < 0.30:
        warnings_html += '<div class="rc-warning rc-warning-red">No suitable BLS match found — consider rephrasing or decomposing into ingredients.</div>'
        if query_text not in st.session_state.unmatched_foods:
            st.session_state.unmatched_foods.append(query_text)
    elif top1_conf < 0.60:
        warnings_html += '<div class="rc-warning rc-warning-orange">No exact match found — closest alternative shown. Manual review recommended.</div>'
    if flag == "check" and top1_conf >= 0.30:
        warnings_html += '<div class="rc-warning rc-warning-yellow">Versions differ — BLS 3.02 and 4.0 return substantially different matches. Review recommended.</div>'

    # BLS columns
    col302_html = _bls_col_html(m302, "BLS 3.02", src302)
    col40_html = _bls_col_html(m40, "BLS 4.0", src40)

    html = (
        f'<div class="{card_class}">'
        f'<div class="rc-header">'
        f'<div class="rc-header-left">'
        f'<div class="rc-food-name">{query_text}</div>'
        f'<div class="rc-food-meta">{nq.cleaned}{meta_badge}{brand_badge}</div>'
        f'</div>'
        f'<div class="rc-header-right">{nova_html}{fg_html}</div>'
        f'</div>'
        f'{warnings_html}'
        f'<div class="rc-grid">{col302_html}{col40_html}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def get_boosted_candidates(text_ret, query, top_k=30, expander=None):
    from modules.normalizer import normalize
    from modules.concept_expansions import CONCEPT_EXPANSION

    nq = normalize(query)
    all_302, all_40 = {}, {}

    def merge(target, candidates, penalty=1.0):
        for c in candidates:
            cd = c.to_dict() if hasattr(c, "to_dict") else c
            code, score = cd["code"], cd["score"] * penalty
            if code not in target or score > (target[code].to_dict() if hasattr(target[code], "to_dict") else target[code])["score"]:
                target[code] = c

    # Strip noise words/phrases that add no search value
    # Multi-word phrases first (order matters)
    _NOISE_PHRASES = [
        "aus dem glas", "aus der dose", "aus der packung", "aus der tüte",
        "aus der flasche", "vom bäcker", "vom metzger", "vom markt",
        "ohne weitere angaben",
    ]
    _NOISE_WORDS = {
        "selbstgemachte", "selbstgemachter", "selbstgemachtes", "selbstgemacht",
        "hausgemacht", "hausgemachte", "hausgemachter", "homemade",
        "frische", "frischer", "frisches",
        "gekauft", "fertig", "fertige", "fertiger", "fertiges",
        "aufgebacken", "aufgewärmt", "zubereitet",
        "tiefgekühlt", "tiefgefroren",
    }
    query_clean = query
    # Strip parentheticals first (e.g. "(ohne weitere Angaben)", "(aufgebacken)")
    query_clean = re.sub(r'\([^)]*\)', '', query_clean)
    # Strip multi-word phrases
    for phrase in _NOISE_PHRASES:
        query_clean = re.sub(re.escape(phrase), '', query_clean, flags=re.IGNORECASE)
    # Strip single words
    for noise in _NOISE_WORDS:
        query_clean = re.sub(r'\b' + re.escape(noise) + r'\b', '', query_clean, flags=re.IGNORECASE)
    query_clean = re.sub(r'\s+', ' ', query_clean).strip()

    # ── Tier 1: Free spell-check against BLS vocabulary ──
    from modules.vocabulary import spell_check_query, get_vocab_set
    corrected_query, corrections_log, any_unknown = spell_check_query(nq.cleaned)

    # ── Tier 2: Haiku combined call (only if unknown tokens + expander) ──
    haiku_search_terms = []
    pre_retrieval_haiku_ran = False

    if any_unknown and expander is not None:
        unknown_tokens = [t for t in corrected_query.split()
                         if t.lower() not in get_vocab_set()
                         and len(t) >= 3 and not t.isdigit()]
        # Gate: skip Haiku if concept expansion already covers all unknowns
        unknowns_not_in_concepts = [t for t in unknown_tokens
                                    if t.lower() not in CONCEPT_EXPANSION]
        if unknowns_not_in_concepts:
            haiku_result = expander.expand_with_spelling(
                query,  # send original raw query — Haiku does its own correction
                unknown_tokens=unknowns_not_in_concepts
            )
            haiku_corrected = haiku_result.get("corrected", corrected_query)
            haiku_search_terms = haiku_result.get("search_terms", [])
            pre_retrieval_haiku_ran = True
            # Use Haiku's correction as the effective query
            query_clean = haiku_corrected
    elif corrections_log:
        # Tier 1 made corrections — use the corrected query
        query_clean = corrected_query

    # Split into main dish + modifiers at connectors
    _CONNECTORS = r'\s+(?:mit|und|aus|in|auf|ohne|an|nach|vom|zum|dazu|plus|à la)\s+'
    dish_parts = re.split(_CONNECTORS, query_clean, maxsplit=1, flags=re.IGNORECASE)
    main_dish = dish_parts[0].strip()
    modifier_phrases = dish_parts[1].strip() if len(dish_parts) > 1 else ""

    # Search main dish at full weight
    result = text_ret.search(main_dish, top_k=20)
    merge(all_302, result.get("bls302", []))
    merge(all_40, result.get("bls40", []))

    # Also search full query if different from main dish
    if query_clean != main_dish:
        result_full = text_ret.search(query_clean, top_k=20)
        merge(all_302, result_full.get("bls302", []))
        merge(all_40, result_full.get("bls40", []))

    # Search modifier phrases at reduced weight + track codes for slot reservation
    modifier_302_codes = set()
    modifier_40_codes = set()
    if modifier_phrases:
        mod_parts = re.split(r'\s*[,;]\s*|\s+und\s+', modifier_phrases)
        for part in [p.strip() for p in mod_parts[:4] if len(p.strip()) >= 3]:
            try:
                r = text_ret.search(part, top_k=5)
                for c in r.get("bls302", []):
                    modifier_302_codes.add((c.to_dict() if hasattr(c, "to_dict") else c)["code"])
                for c in r.get("bls40", []):
                    modifier_40_codes.add((c.to_dict() if hasattr(c, "to_dict") else c)["code"])
                merge(all_302, r.get("bls302", []), 0.5)
                merge(all_40, r.get("bls40", []), 0.5)
            except Exception:
                pass

    # ── Merge Haiku pre-retrieval search terms (if any) ──
    if haiku_search_terms:
        for term in haiku_search_terms:
            try:
                r = text_ret.search(term, top_k=5)
                merge(all_302, r.get("bls302", []), 0.95)
                merge(all_40, r.get("bls40", []), 0.95)
            except Exception:
                pass

    # ── Vocab-based compound fallback for unknown long tokens ──
    if any_unknown:
        from difflib import get_close_matches as _dfm
        _vset = get_vocab_set()
        _vlst = sorted(_vset)

        def _vocab_compound_split(word):
            """Split a compound word against BLS vocabulary.
            Returns list of (component, quality) or None.
            quality: 2 = both halves food words, 1 = only one half."""
            w = word.lower()
            if w in _vset or len(w) < 6:
                return None
            _JOINERS = ["", "s", "n", "en", "e", "er"]
            both_match = []
            one_match = []
            for i in range(3, len(w) - 2):
                left = w[:i]
                remainder = w[i:]
                for joiner in _JOINERS:
                    if joiner:
                        if not remainder.startswith(joiner):
                            continue
                        right = remainder[len(joiner):]
                    else:
                        right = remainder
                    if len(right) < 3:
                        continue
                    left_in = left in _vset
                    right_in = right in _vset
                    left_resolved = left
                    right_resolved = right
                    if not left_in and len(left) >= 4:
                        fm = _dfm(left, _vlst, n=1, cutoff=0.85)
                        if fm and min(len(left), len(fm[0])) / max(len(left), len(fm[0])) >= 0.5:
                            left_resolved = fm[0]
                            left_in = True
                    if not right_in and len(right) >= 4:
                        fm = _dfm(right, _vlst, n=1, cutoff=0.85)
                        if fm and min(len(right), len(fm[0])) / max(len(right), len(fm[0])) >= 0.5:
                            right_resolved = fm[0]
                            right_in = True
                    if left_in and right_in:
                        both_match.append((left_resolved, right_resolved,
                                           len(left_resolved) * len(right_resolved)))
                    elif left_in:
                        one_match.append((left_resolved,))
                    elif right_in:
                        one_match.append((right_resolved,))
            if both_match:
                best = max(both_match, key=lambda x: x[2])
                return [(best[0], 2), (best[1], 2)]
            if one_match:
                seen = {}
                for parts in one_match:
                    c = parts[0]
                    if c not in seen or len(c) > len(seen[c]):
                        seen[c] = c
                longest = max(seen.values(), key=len)
                return [(longest, 1)]
            return None

        query_tokens = re.findall(r'\w+', corrected_query.lower())
        for token in query_tokens:
            if len(token) >= 6 and token not in _vset and " " not in token:
                split_result = _vocab_compound_split(token)
                if split_result:
                    for comp, quality in split_result:
                        weight = 0.40 if quality == 2 else 0.25
                        try:
                            r = text_ret.search(comp, top_k=10)
                            merge(all_302, r.get("bls302", []), weight)
                            merge(all_40, r.get("bls40", []), weight)
                        except Exception:
                            pass

    # Concept expansion — match triggers against multiple forms of the query
    def _trigger_matches(trigger, text):
        """Check if trigger matches in text. Handles compounds, hyphens, spaces."""
        if len(trigger) <= 4:
            return bool(re.search(r'(?<!\w)' + re.escape(trigger) + r'(?!\w)', text))
        # Try exact substring
        if trigger in text:
            return True
        # Try with hyphens removed (schoko-orange-mousse → schokoorangemousse)
        text_nohyph = text.replace("-", "")
        trigger_nohyph = trigger.replace("-", "")
        if trigger_nohyph in text_nohyph:
            return True
        # Try with hyphens as spaces
        text_spaces = text.replace("-", " ")
        trigger_spaces = trigger.replace("-", " ")
        if trigger_spaces in text_spaces:
            return True
        return False

    query_lower = query_clean.lower().strip()
    for trigger, expansions in CONCEPT_EXPANSION.items():
        if not _trigger_matches(trigger, query_lower):
            continue
        for exp in expansions[:5]:
            try:
                r = text_ret.search(exp, top_k=5)
                merge(all_302, r.get("bls302", []), 0.9)
                merge(all_40, r.get("bls40", []), 0.9)
            except Exception:
                pass

    # Multi-ingredient sub-query split (comma/semicolon only — connectors already handled above)
    comma_parts = re.split(r'\s*[,;]\s*', query_clean)
    if len(comma_parts) > 1:
        for part in [p.strip() for p in comma_parts[:4] if len(p.strip()) >= 3]:
            try:
                r = text_ret.search(part, top_k=5)
                merge(all_302, r.get("bls302", []), 0.85)
                merge(all_40, r.get("bls40", []), 0.85)
            except Exception:
                pass

    # ── Claude query expansion (only when retriever is struggling AND pre-retrieval didn't run) ──
    _old_expansion_ran = False
    if expander is not None and not pre_retrieval_haiku_ran:
        best_302 = max(((c.to_dict() if hasattr(c, "to_dict") else c)["score"]
                        for c in all_302.values()), default=0)
        best_40 = max(((c.to_dict() if hasattr(c, "to_dict") else c)["score"]
                       for c in all_40.values()), default=0)
        if best_302 < 1.5 or best_40 < 1.5:
            expansion_terms = expander.expand(query)
            _old_expansion_ran = True
            for term in expansion_terms:
                try:
                    r = text_ret.search(term, top_k=5)
                    if best_302 < 1.5:
                        merge(all_302, r.get("bls302", []), 0.95)
                    if best_40 < 1.5:
                        merge(all_40, r.get("bls40", []), 0.95)
                except Exception:
                    pass

    # ── Late-stage Haiku rescue ──
    if expander is not None and not haiku_search_terms and not _old_expansion_ran:
        _LSTOP = {"mit", "und", "aus", "von", "für", "ohne", "in", "an", "auf",
                  "der", "die", "das", "dem", "den", "ein", "eine", "einer",
                  "zu", "zum", "zur", "im", "am", "vom"}
        q_content = [t for t in re.findall(r'\w+', query_clean.lower())
                     if len(t) >= 3 and t not in _LSTOP]

        _need_rescue = False
        if len(q_content) >= 2:
            # Multi-token: check if any top-5 candidate contains ALL content tokens
            for pool in (all_302, all_40):
                top5 = sorted(pool.values(),
                              key=lambda x: (x.to_dict() if hasattr(x, "to_dict") else x)["score"],
                              reverse=True)[:5]
                for cand in top5:
                    name_l = (cand.name_de if hasattr(cand, "name_de") else
                              cand.get("name_de", "")).lower()
                    if all(t in name_l for t in q_content):
                        _need_rescue = False
                        break
                else:
                    _need_rescue = True
                    continue
                break

        elif len(q_content) == 1 and len(q_content[0]) >= 8:
            # Single long compound: check if top-1 candidate name contains the word
            _need_rescue = True
            for pool in (all_302, all_40):
                top1 = sorted(pool.values(),
                              key=lambda x: (x.to_dict() if hasattr(x, "to_dict") else x)["score"],
                              reverse=True)[:1]
                if top1:
                    name_l = (top1[0].name_de if hasattr(top1[0], "name_de") else
                              top1[0].get("name_de", "")).lower()
                    if q_content[0] in name_l:
                        _need_rescue = False
                        break

        if _need_rescue:
            try:
                late_terms = expander.expand(query)
                if late_terms:
                    for term in late_terms:
                        try:
                            r = text_ret.search(term, top_k=10)
                            merge(all_302, r.get("bls302", []), 0.95)
                            merge(all_40, r.get("bls40", []), 0.95)
                        except Exception:
                            pass
            except Exception:
                pass

    # ── Cross-version bootstrapping ──
    _best_302 = max(((c.to_dict() if hasattr(c, "to_dict") else c)["score"]
                     for c in all_302.values()), default=0)
    _best_40 = max(((c.to_dict() if hasattr(c, "to_dict") else c)["score"]
                    for c in all_40.values()), default=0)
    _STRONG, _WEAK = 0.8, 0.5
    _STOP_WORDS = {"mit", "und", "aus", "von", "für", "ohne", "roh", "fett",
                   "gegart", "gekocht", "gebraten", "roh"}

    if _best_302 >= _STRONG and _best_40 < _WEAK:
        best_entry = max(all_302.values(),
                         key=lambda c: (c.to_dict() if hasattr(c, "to_dict") else c)["score"])
        best_name = (best_entry.to_dict() if hasattr(best_entry, "to_dict") else best_entry)["name_de"]
        words = [w for w in re.findall(r'\w+', best_name.lower())
                 if len(w) >= 3 and w not in _STOP_WORDS]
        search_term = " ".join(words[:3])
        if search_term:
            try:
                r = text_ret.search(search_term, top_k=10)
                merge(all_40, r.get("bls40", []), 0.70)
            except Exception:
                pass

    elif _best_40 >= _STRONG and _best_302 < _WEAK:
        best_entry = max(all_40.values(),
                         key=lambda c: (c.to_dict() if hasattr(c, "to_dict") else c)["score"])
        best_name = (best_entry.to_dict() if hasattr(best_entry, "to_dict") else best_entry)["name_de"]
        words = [w for w in re.findall(r'\w+', best_name.lower())
                 if len(w) >= 3 and w not in _STOP_WORDS]
        search_term = " ".join(words[:3])
        if search_term:
            try:
                r = text_ret.search(search_term, top_k=10)
                merge(all_302, r.get("bls302", []), 0.70)
            except Exception:
                pass


    # ── Compound salat suppression ──
    _salat_compounds = [t for t in re.findall(r'\w+', query_clean.lower())
                        if t.endswith("salat") and len(t) > 5]
    if _salat_compounds:
        _LEAVES = {"eisbergsalat", "kopfsalat", "feldsalat", "endiviensalat",
                   "lollo", "rucola", "radicchio", "blattsalat"}
        for pool in (all_302, all_40):
            for cand in pool.values():
                nl = (cand.name_de if hasattr(cand, "name_de") else
                      cand.get("name_de", "")).lower()
                if any(leaf in nl for leaf in _LEAVES):
                    if hasattr(cand, "score"):
                        cand.score *= 0.2
                    else:
                        cand["score"] *= 0.2

    # ── Sort, trim, and return ──
    def sort_trim(d, k, mod_codes=None):
        items = sorted(d.values(),
                       key=lambda x: (x.to_dict() if hasattr(x, "to_dict") else x)["score"],
                       reverse=True)
        if not mod_codes:
            return items[:k]
        main_items, mod_items = [], []
        for item in items:
            code = (item.to_dict() if hasattr(item, "to_dict") else item).get("code", "")
            if code in mod_codes:
                mod_items.append(item)
            else:
                main_items.append(item)
        reserved = min(5, len(mod_items))
        combined = main_items[:k - reserved] + mod_items[:reserved]
        combined.sort(key=lambda x: (x.to_dict() if hasattr(x, "to_dict") else x)["score"],
                      reverse=True)
        return combined[:k]

    return {"query": nq,
            "bls302": sort_trim(all_302, top_k, modifier_302_codes),
            "bls40": sort_trim(all_40, top_k, modifier_40_codes)}


# ═══════════════════════════════════════════════════════════
#  Header (rendered after sidebar, so we know if query exists)
# ═══════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════
#  API key + Claude (silent — no UI)
# ═══════════════════════════════════════════════════════════

_api_key = ""
try:
    _api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
except Exception:
    _api_key = os.environ.get("ANTHROPIC_API_KEY", "")
use_claude = bool(_api_key)
cache = get_cache()


# ═══════════════════════════════════════════════════════════
#  Main area
# ═══════════════════════════════════════════════════════════

with st.spinner("Loading BLS catalogs..."):
    text_ret = load_text_retriever()

# Peek at query to decide layout (hero vs compact)
_has_query = bool(st.session_state.get("food_query", ""))

if not _has_query:
    # STATE 1: Landing page — big centered hero
    st.markdown("""<div class="search-hero">
        <div class="hero-title">Search any food.<br>Get the BLS code.</div>
        <p class="hero-sub">Type a food name — we'll match it to the<br>BLS nutritional database instantly.</p>
    </div>""", unsafe_allow_html=True)

query = st.text_input(
    "Food description",
    placeholder="Haferflocken, Döner, Chicken salad...",
    key="food_query",
    label_visibility="collapsed",
)

if not query:
    st.markdown(
        '<div class="hero-pills">'
        '<span class="hero-pill">Haferflocken</span>'
        '<span class="hero-pill">Döner</span>'
        '<span class="hero-pill">Chicken salad</span>'
        '<span class="hero-pill">Vollkornbrot</span>'
        '<span class="hero-pill">Magnum Mandel</span>'
        '</div>'
        '<div class="hero-footer">'
        '<span class="hero-dot"></span>'
        '<span>Hector-Center for Nutrition, Exercise and Sports &middot; '
        'University Hospital Erlangen</span>'
        '</div>',
        unsafe_allow_html=True,
    )

if query:
    # Check verified maps — skip expansion if both versions are verified
    from modules.verified_map import VERIFIED_MAP_302
    from modules.verified_map_40 import VERIFIED_MAP_40
    _is_verified = (query.lower().strip() in VERIFIED_MAP_302
                    and query.lower().strip() in VERIFIED_MAP_40)

    expander = None
    if use_claude and _api_key and not _is_verified:
        expander = get_expander(_api_key)

    with st.spinner("Searching BLS catalogs..."):
        candidates = get_boosted_candidates(text_ret, query, top_k=30, expander=expander)
    nq = candidates["query"]

    # ── Resolve matches ──
    result = None
    result_from_claude = False

    if use_claude and _api_key:
        reranker = get_reranker(_api_key)
        if reranker:
            skip_cache = (st.session_state.requery_food == query)
            if skip_cache:
                cache.delete(query)
                st.session_state.requery_food = None

            old_hits = reranker.session_cache_hits
            old_calls = reranker.session_api_calls

            with st.spinner("Analyzing candidates..."):
                result = reranker.rerank(query, candidates, skip_cache=skip_cache)

            st.session_state.cache_hits += reranker.session_cache_hits - old_hits
            st.session_state.api_calls += reranker.session_api_calls - old_calls
            result_from_claude = True

    if result is None or (result and result.error):
        if result and result.error:
            st.error(f"API error: {result.error}")
        smart = get_smart_reranker()
        result = smart.rerank(query, candidates)

    # ── Extract Claude NOVA if available ──
    claude_nova = getattr(result, "claude_nova", None)

    # ── Safety flag ──
    flag = safety_flag(result, query, nq.brand)

    # ── Source text ──
    src302 = getattr(result, "bls302_source", "")
    src40 = getattr(result, "bls40_source", "")
    if src302 == "verified":
        source_text = "Verified lookup — no API cost"
    elif src302 == "cached":
        source_text = "Served from cache — no API cost"
    elif src302 == "api":
        source_text = "Re-ranked by Claude API"
    else:
        rp = getattr(result, "resolution_path", "")
        source_text = "Verified lookup" if rp == "verified" else "Rule-based matching"

    # ── Unified result card ──
    render_result_card(query, nq, result, flag, src302, src40,
                       claude_nova=claude_nova, used_claude=result_from_claude,
                       source_text=source_text)

    # ── Action row (source + buttons, right-aligned) ──
    has_cached = src302 == "cached" or src40 == "cached"
    already_flagged = query in st.session_state.flagged_this_session

    with st.container():
        st.markdown('<div class="action-row">', unsafe_allow_html=True)
        act_cols = st.columns([5, 1, 1])
        with act_cols[0]:
            st.markdown(f'<span style="font-size:12px; color:#86868b;">{source_text}</span>',
                        unsafe_allow_html=True)
        with act_cols[1]:
            if has_cached:
                if st.button("Re-query", key="requery_btn", type="secondary"):
                    st.session_state.requery_food = query
                    st.rerun()
        with act_cols[2]:
            if already_flagged:
                st.markdown('<span style="font-size:12px; color:#34c759;">Flagged ✓</span>',
                            unsafe_allow_html=True)
            else:
                if st.button("Flag", key="flag_btn", type="secondary"):
                    from datetime import datetime
                    m302 = result.bls302_matches or []
                    m40 = result.bls40_matches or []
                    nova_val = None
                    if m302:
                        nova_val, _ = resolve_nova(m302[0].code, "302", query, nq.brand,
                                                  claude_nova, result_from_claude)
                    save_flag({
                        "timestamp": datetime.now().isoformat(),
                        "food_description": query,
                        "normalized_name": nq.cleaned,
                        "bls302_code": m302[0].code if m302 else None,
                        "bls302_name": m302[0].name if m302 else None,
                        "bls40_code": m40[0].code if m40 else None,
                        "bls40_name": m40[0].name if m40 else None,
                        "nova_score": nova_val,
                        "source": source_text,
                    })
                    st.session_state.flagged_this_session.add(query)
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── More matches (2nd and 3rd choices) ──
    m302 = result.bls302_matches or []
    m40 = result.bls40_matches or []
    if len(m302) > 1 or len(m40) > 1:
        with st.expander("More matches", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                st.caption("BLS 3.02")
                for m in m302[1:3]:
                    pct = int(m.confidence * 100)
                    st.markdown(f"**{m.code}** — {m.name} ({pct}%)")
            with col_b:
                st.caption("BLS 4.0")
                for m in m40[1:3]:
                    pct = int(m.confidence * 100)
                    st.markdown(f"**{m.code}** — {m.name} ({pct}%)")

    # ── Normalization details ──
    with st.expander("Normalization details", expanded=False):
        nc1, nc2, nc3 = st.columns(3)
        with nc1:
            st.markdown(f"**Original:** {nq.original}")
            st.markdown(f"**Cleaned:** {nq.cleaned}")
        with nc2:
            st.markdown(f"**Prep state:** {nq.prep_state or '—'}")
            st.markdown(f"**Fat %:** {nq.fat_percent or '—'}")
        with nc3:
            st.markdown(f"**Brand:** {nq.brand or '—'}")
            st.markdown(f"**Language:** {'English' if nq.is_english else 'German'}")
        st.markdown(
            f"**Candidates:** {len(candidates.get('bls302', []))} (3.02) / "
            f"{len(candidates.get('bls40', []))} (4.0)"
        )

    # ── Unmatched foods (in main area) ──
    if st.session_state.unmatched_foods:
        with st.expander(f"Unmatched foods ({len(st.session_state.unmatched_foods)})", expanded=False):
            for uf in st.session_state.unmatched_foods:
                st.markdown(f"- {uf}")
            if st.button("Clear list", key="clear_unmatched"):
                st.session_state.unmatched_foods = []
                st.rerun()

# ═══════════════════════════════════════════════════════════
#  Admin view (?admin=true)
# ═══════════════════════════════════════════════════════════

if st.query_params.get("admin") == "true":
    st.markdown("---")
    st.markdown("#### Flagged Results")
    flags = load_flags()
    if not flags:
        st.info("No flagged results yet.")
    else:
        df_flags = pd.DataFrame(flags)
        st.dataframe(df_flags, use_container_width=True)
        csv = df_flags.to_csv(index=False).encode("utf-8")
        st.download_button("Download flags as CSV", csv,
                           "flagged_results.csv", "text/csv")
