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

    /* Result cards */
    .result-card {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
        background: #fafbfc;
    }
    .result-card:hover { border-color: #cbd5e1; }
    .result-card .rank-badge {
        display: inline-block; width: 28px; height: 28px; line-height: 28px;
        text-align: center; border-radius: 50%; font-weight: 700;
        font-size: 0.85rem; margin-right: 0.5rem;
    }
    .result-card .code { font-family: monospace; font-weight: 600; font-size: 0.95rem; }
    .result-card .name { color: #334155; }
    .result-card .meta { font-size: 0.82rem; color: #64748b; margin-top: 0.4rem; }

    /* Confidence bar */
    .conf-bar {
        display: inline-block; height: 6px; border-radius: 3px;
        vertical-align: middle; margin-left: 6px;
    }

    /* Source & safety badges */
    .badge {
        display: inline-block; padding: 1px 7px; border-radius: 3px;
        font-size: 0.75rem; font-weight: 500; vertical-align: middle;
    }
    .badge-verified { background: #dcfce7; color: #166534; }
    .badge-cached   { background: #e0e7ff; color: #3730a3; }
    .badge-api      { background: #fef3c7; color: #92400e; }
    .badge-ok       { background: #dcfce7; color: #166534; }
    .badge-check    { background: #fef2f2; color: #991b1b; }

    /* Confidence warning banners */
    .match-warning {
        border-radius: 6px; padding: 0.6rem 1rem; margin-bottom: 0.75rem;
        font-size: 0.85rem;
    }
    .match-warning-orange {
        background: #fef3c7; border: 1px solid #f59e0b; color: #92400e;
    }
    .match-warning-red {
        background: #fee2e2; border: 1px solid #ef4444; color: #991b1b;
    }

    /* NOVA */
    .nova-pill {
        display: inline-block; padding: 1px 8px; border-radius: 4px;
        font-weight: 700; font-size: 0.8rem; color: white;
    }

    /* Summary table styling */
    .summary-table th { background: #f1f5f9; font-size: 0.8rem; }
    .summary-table td { font-size: 0.85rem; }


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
                     ("unmatched_foods", [])]:
    if key not in st.session_state:
        st.session_state[key] = default


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════

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


def conf_color(c):
    if c >= 0.85: return "#16a34a"
    if c >= 0.60: return "#ca8a04"
    return "#dc2626"


def conf_bg(c):
    if c >= 0.85: return "#dcfce7"
    if c >= 0.60: return "#fef9c3"
    return "#fee2e2"


def source_badge(src):
    m = {"verified": "badge-verified", "cached": "badge-cached", "api": "badge-api"}
    labels = {"verified": "verified", "cached": "cached", "api": "live API"}
    cls = m.get(src, "")
    lbl = labels.get(src, "")
    if not lbl:
        return ""
    return f"<span class='badge {cls}'>{lbl}</span>"


def nova_pill(nova, low_confidence=False):
    if nova is None:
        return "<span style='color:#94a3b8;'>—</span>"
    colors = {1: "#16a34a", 2: "#65a30d", 3: "#ca8a04", 4: "#dc2626"}
    bg = colors.get(nova, "#94a3b8")
    if low_confidence:
        label = f"NOVA {nova} (?)"
        return (f"<span class='nova-pill' style='background:{bg}; opacity:0.6;' "
                f"title='Low confidence — enable Claude API for better accuracy'>"
                f"{label}</span>")
    return f"<span class='nova-pill' style='background:{bg};'>NOVA {nova}</span>"


def safety_flag(result, food_desc=None, brand=None):
    """Compute safety flag from a RerankerResult."""
    m302 = result.bls302_matches
    m40 = result.bls40_matches

    # Condition 4: one version has matches, the other has none
    if (m302 and not m40) or (m40 and not m302):
        return "check"

    if not m302 and not m40:
        return "check"

    top302 = m302[0] if m302 else None
    top40 = m40[0] if m40 else None

    # Condition 3: either top-1 confidence < 0.60
    if top302 and top302.confidence < 0.60:
        return "check"
    if top40 and top40.confidence < 0.60:
        return "check"

    # Condition 1: different food group letter
    if top302 and top40 and top302.code[0] != top40.code[0]:
        return "check"

    # Condition 2: food group classification differs
    if top302 and top40:
        cls302 = classify(top302.code, "302", food_desc=food_desc, brand=brand)
        cls40 = classify(top40.code, "40", food_desc=food_desc, brand=brand)
        if cls302["main_group"] and cls40["main_group"]:
            if cls302["main_group"] != cls40["main_group"]:
                return "check"

    return "ok"


def safety_badge_html(flag):
    if flag == "ok":
        return "<span class='badge badge-ok'>OK</span>"
    return "<span class='badge badge-check'>CHECK</span>"


def build_summary_row(query_text, nq, result, flag, claude_nova=None, used_claude=False):
    """Build a dict for the summary table."""
    def _m(matches, idx):
        if idx < len(matches):
            return matches[idx].code, matches[idx].confidence
        return "—", None
    m302 = result.bls302_matches or []
    m40 = result.bls40_matches or []
    c302_1 = _m(m302, 0)
    c302_2 = _m(m302, 1)
    c302_3 = _m(m302, 2)
    c40_1 = _m(m40, 0)
    c40_2 = _m(m40, 1)
    c40_3 = _m(m40, 2)

    # Food group from top-1 BLS 3.02
    cls = classify(m302[0].code, "302", food_desc=query_text, brand=nq.brand) if m302 else {"main_group": None, "sub_group": None, "nova": None}
    if m302:
        nova, _ = resolve_nova(m302[0].code, "302", query_text, nq.brand, claude_nova, used_claude)
        cls["nova"] = nova

    def _fmt_conf(c):
        return f"{c:.0%}" if c is not None else "—"

    return {
        "Food item": query_text,
        "Normalized": nq.cleaned,
        "BLS 3.02 1st": c302_1[0],
        "Conf 1st 3.02": _fmt_conf(c302_1[1]),
        "BLS 3.02 2nd": c302_2[0],
        "Conf 2nd 3.02": _fmt_conf(c302_2[1]),
        "BLS 3.02 3rd": c302_3[0],
        "Conf 3rd 3.02": _fmt_conf(c302_3[1]),
        "BLS 4.0 1st": c40_1[0],
        "Conf 1st 4.0": _fmt_conf(c40_1[1]),
        "BLS 4.0 2nd": c40_2[0],
        "Conf 2nd 4.0": _fmt_conf(c40_2[1]),
        "BLS 4.0 3rd": c40_3[0],
        "Conf 3rd 4.0": _fmt_conf(c40_3[1]),
        "Main group": cls["main_group"] or "—",
        "Sub group": cls["sub_group"] or "—",
        "NOVA": cls["nova"] if cls["nova"] is not None else "—",
        "Safety": "OK" if flag == "ok" else "CHECK",
    }


def render_summary_table(row):
    """Render summary table as styled HTML."""
    def _conf_cell(val):
        if val == "—":
            return "<td style='text-align:center; color:#94a3b8;'>—</td>"
        pct = int(val.replace("%", ""))
        c = conf_color(pct / 100)
        bg = conf_bg(pct / 100)
        return f"<td style='text-align:center; background:{bg}; color:{c}; font-weight:600;'>{val}</td>"

    def _code_cell(val):
        if val == "—":
            return "<td style='text-align:center; color:#94a3b8;'>—</td>"
        return f"<td style='text-align:center; font-family:monospace; font-weight:500;'>{val}</td>"

    def _safety_cell(val):
        if val == "OK":
            return "<td style='text-align:center;'><span class='badge badge-ok'>OK</span></td>"
        return "<td style='text-align:center;'><span class='badge badge-check'>CHECK</span></td>"

    def _nova_cell(val):
        if val == "—":
            return "<td style='text-align:center; color:#94a3b8;'>—</td>"
        return f"<td style='text-align:center;'>{nova_pill(int(val))}</td>"

    html = """<div style="overflow-x:auto; margin-bottom:1.2rem;">
    <table style="width:100%; border-collapse:collapse; font-size:0.82rem; border:1px solid #e2e8f0; border-radius:6px;">
    <thead><tr style="background:#f1f5f9;">"""

    headers = list(row.keys())
    for h in headers:
        html += f"<th style='padding:8px 6px; text-align:center; border-bottom:2px solid #e2e8f0; font-weight:600; color:#475569; white-space:nowrap;'>{h}</th>"
    html += "</tr></thead><tbody><tr>"

    for key, val in row.items():
        val_str = str(val)
        if key.startswith("Conf"):
            html += _conf_cell(val_str)
        elif key.startswith("BLS"):
            html += _code_cell(val_str)
        elif key == "Safety":
            html += _safety_cell(val_str)
        elif key == "NOVA":
            html += _nova_cell(val_str)
        elif key == "Main group" or key == "Sub group":
            display = val_str.replace("_", " ") if val_str != "—" else "—"
            html += f"<td style='padding:6px; font-size:0.78rem; color:#475569;'>{display}</td>"
        else:
            html += f"<td style='padding:6px; color:#334155;'>{val_str}</td>"

    html += "</tr></tbody></table></div>"
    return html


def display_card(m, bls_ver, result_source="", food_desc=None, brand=None,
                 claude_nova=None, used_claude=False):
    """Render a single match as a styled card."""
    color = conf_color(m.confidence)
    bg = conf_bg(m.confidence)
    cls = classify(m.code, bls_ver, food_desc=food_desc, brand=brand)
    main_str = (cls["main_group"] or "—").replace("_", " ")
    sub_str = (cls["sub_group"] or "—").replace("_", " ")
    pct = int(m.confidence * 100)
    bar_w = max(pct, 5)

    nova, nova_low = resolve_nova(m.code, bls_ver, food_desc, brand, claude_nova, used_claude)

    src_html = f" {source_badge(result_source)}" if result_source and m.rank == 1 else ""
    low_conf_note = ""
    if m.rank == 1 and m.confidence < 0.60:
        low_conf_note = ("<div style='color:#92400e; font-size:0.78rem; margin-top:0.3rem; "
                         "font-style:italic;'>Note: Closest available match — may not "
                         "accurately represent this food item.</div>")

    st.markdown(f"""<div class="result-card">
        <div>
            <span class="rank-badge" style="background:{bg}; color:{color};">{m.rank}</span>
            <span class="code" style="color:{color};">{m.code}</span>
            <span class="name">&nbsp;— {m.name}</span>{src_html}
        </div>
        <div style="margin-top:0.4rem;">
            <span style="font-weight:600; color:{color}; font-size:0.9rem;">{m.confidence:.0%}</span>
            <span class="conf-bar" style="width:{bar_w}px; background:{color};"></span>
            <span style="color:#64748b; font-size:0.82rem; margin-left:8px;">{m.reasoning}</span>
        </div>
        <div class="meta">
            {main_str} &nbsp;&middot;&nbsp; {sub_str} &nbsp;&middot;&nbsp; {nova_pill(nova, low_confidence=nova_low)}
        </div>{low_conf_note}
    </div>""", unsafe_allow_html=True)


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

    result = text_ret.search(query, top_k=20)
    merge(all_302, result.get("bls302", []))
    merge(all_40, result.get("bls40", []))

    def _trigger_matches(trigger, text):
        """Check if trigger matches in text. Short triggers use word boundaries."""
        if len(trigger) <= 4:
            return bool(re.search(r'(?<!\w)' + re.escape(trigger) + r'(?!\w)', text))
        return trigger in text

    query_lower = query.lower().strip()
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

    parts = re.split(r'\s*[,;]\s*|\s+und\s+|\s+mit\s+', query)
    if len(parts) > 1:
        for part in [p.strip() for p in parts[:4] if len(p.strip()) >= 3]:
            try:
                r = text_ret.search(part, top_k=5)
                merge(all_302, r.get("bls302", []), 0.85)
                merge(all_40, r.get("bls40", []), 0.85)
            except Exception:
                pass

    # ── Claude query expansion (only when retriever is struggling) ──
    if expander is not None:
        best_302 = max(((c.to_dict() if hasattr(c, "to_dict") else c)["score"]
                        for c in all_302.values()), default=0)
        best_40 = max(((c.to_dict() if hasattr(c, "to_dict") else c)["score"]
                       for c in all_40.values()), default=0)
        if best_302 < 1.5 or best_40 < 1.5:
            expansion_terms = expander.expand(query)
            for term in expansion_terms:
                try:
                    r = text_ret.search(term, top_k=5)
                    if best_302 < 1.5:
                        merge(all_302, r.get("bls302", []), 0.95)
                    if best_40 < 1.5:
                        merge(all_40, r.get("bls40", []), 0.95)
                except Exception:
                    pass

    def sort_trim(d, k):
        items = sorted(d.values(), key=lambda x: (x.to_dict() if hasattr(x, "to_dict") else x)["score"], reverse=True)
        return items[:k]

    return {"query": nq, "bls302": sort_trim(all_302, top_k), "bls40": sort_trim(all_40, top_k)}


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

    # ── Summary table ──
    row = build_summary_row(query, nq, result, flag,
                            claude_nova=claude_nova, used_claude=result_from_claude)
    st.markdown("##### Summary", unsafe_allow_html=True)
    st.markdown(render_summary_table(row), unsafe_allow_html=True)

    # ── Confidence warning banner ──
    top1_conf_302 = result.bls302_matches[0].confidence if result.bls302_matches else 0
    top1_conf_40 = result.bls40_matches[0].confidence if result.bls40_matches else 0
    top1_conf = max(top1_conf_302, top1_conf_40)

    if top1_conf < 0.30:
        st.markdown(
            '<div class="match-warning match-warning-red">'
            "No suitable BLS match found for this food item. Consider: "
            "(1) rephrasing the food description more specifically, or "
            "(2) decomposing into individual ingredients.</div>",
            unsafe_allow_html=True,
        )
        if query not in st.session_state.unmatched_foods:
            st.session_state.unmatched_foods.append(query)
    elif top1_conf < 0.60:
        st.markdown(
            '<div class="match-warning match-warning-orange">'
            "No exact match found. Closest alternative shown "
            "— manual review recommended.</div>",
            unsafe_allow_html=True,
        )

    # ── Status bar: source + safety + re-query ──
    src302 = getattr(result, "bls302_source", "")
    src40 = getattr(result, "bls40_source", "")
    status_parts = []
    if src302 == "verified":
        status_parts.append("Verified lookup — no API cost")
    elif src302 == "cached":
        status_parts.append("Served from cache — no API cost")
    elif src302 == "api":
        status_parts.append("Re-ranked by Claude API")
    else:
        rp = getattr(result, "resolution_path", "")
        if rp == "verified":
            status_parts.append("Verified lookup")
        else:
            status_parts.append("Rule-based matching")

    status_col, requery_col = st.columns([8, 2])
    with status_col:
        st.caption(" | ".join(status_parts))
    with requery_col:
        has_cached = src302 == "cached" or src40 == "cached"
        if has_cached:
            if st.button("Re-query", key="requery_btn", type="secondary"):
                st.session_state.requery_food = query
                st.rerun()

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

    # ── Detailed card view ──
    st.markdown(
        f"##### Detailed Results &nbsp;&nbsp;{safety_badge_html(flag)}",
        unsafe_allow_html=True,
    )

    col302, col40 = st.columns(2)
    with col302:
        st.markdown(
            "<div style='font-weight:600; color:#1e3a5f; margin-bottom:0.5rem;'>BLS 3.02</div>",
            unsafe_allow_html=True,
        )
        if result.bls302_matches:
            for m in result.bls302_matches:
                display_card(m, "302", src302, query, nq.brand,
                             claude_nova=claude_nova, used_claude=result_from_claude)
        else:
            st.markdown(
                "<div class='result-card' style='text-align:center; color:#94a3b8;'>No matches</div>",
                unsafe_allow_html=True,
            )

    with col40:
        st.markdown(
            "<div style='font-weight:600; color:#1e3a5f; margin-bottom:0.5rem;'>BLS 4.0</div>",
            unsafe_allow_html=True,
        )
        if result.bls40_matches:
            for m in result.bls40_matches:
                display_card(m, "40", src40, query, nq.brand,
                             claude_nova=claude_nova, used_claude=result_from_claude)
        else:
            st.markdown(
                "<div class='result-card' style='text-align:center; color:#94a3b8;'>No matches</div>",
                unsafe_allow_html=True,
            )

    # ── Raw candidates ──
    with st.expander("Raw candidates", expanded=False):
        t1, t2 = st.tabs(["BLS 3.02", "BLS 4.0"])
        with t1:
            for i, c in enumerate(candidates.get("bls302", [])[:20], 1):
                cd = c.to_dict() if hasattr(c, "to_dict") else c
                st.text(f"{i:2d}. [{cd['code']}] {cd['name_de'][:55]:<55s} score={cd['score']:.3f}")
        with t2:
            for i, c in enumerate(candidates.get("bls40", [])[:20], 1):
                cd = c.to_dict() if hasattr(c, "to_dict") else c
                st.text(f"{i:2d}. [{cd['code']}] {cd['name_de'][:55]:<55s} score={cd['score']:.3f}")

    # ── Unmatched foods (in main area) ──
    if st.session_state.unmatched_foods:
        with st.expander(f"Unmatched foods ({len(st.session_state.unmatched_foods)})", expanded=False):
            for uf in st.session_state.unmatched_foods:
                st.markdown(f"- {uf}")
            if st.button("Clear list", key="clear_unmatched"):
                st.session_state.unmatched_foods = []
                st.rerun()
