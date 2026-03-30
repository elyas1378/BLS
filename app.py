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
from modules.claude_cache import ClaudeCache

st.set_page_config(
    page_title="BLS Food Code Matcher",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ──
st.markdown("""
<style>
    /* Header bar */
    .app-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5986 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        margin: 0; font-size: 1.6rem; font-weight: 700; color: white;
    }
    .app-header p {
        margin: 0.3rem 0 0 0; font-size: 0.9rem; opacity: 0.85; color: #cbd5e1;
    }
    .app-header .institution {
        margin-top: 0.5rem; font-size: 0.8rem; opacity: 0.7; color: #94a3b8;
        border-top: 1px solid rgba(255,255,255,0.15); padding-top: 0.5rem;
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

    /* NOVA */
    .nova-pill {
        display: inline-block; padding: 1px 8px; border-radius: 4px;
        font-weight: 700; font-size: 0.8rem; color: white;
    }

    /* Summary table styling */
    .summary-table th { background: #f1f5f9; font-size: 0.8rem; }
    .summary-table td { font-size: 0.85rem; }

    /* Footer */
    .app-footer {
        text-align: center; padding: 1rem 0; color: #94a3b8;
        font-size: 0.78rem; border-top: 1px solid #e2e8f0; margin-top: 2rem;
    }
    .app-footer a { color: #64748b; text-decoration: none; }

    /* Sidebar tweaks */
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 0.95rem; color: #475569; text-transform: uppercase;
        letter-spacing: 0.05em; margin-bottom: 0.3rem;
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

for key, default in [("cache_hits", 0), ("api_calls", 0), ("requery_food", None)]:
    if key not in st.session_state:
        st.session_state[key] = default


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════

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


def nova_pill(nova):
    if nova is None:
        return "<span style='color:#94a3b8;'>—</span>"
    colors = {1: "#16a34a", 2: "#65a30d", 3: "#ca8a04", 4: "#dc2626"}
    bg = colors.get(nova, "#94a3b8")
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
        cls302 = classify(top302.code, "302", food_desc, brand)
        cls40 = classify(top40.code, "40", food_desc, brand)
        if cls302["main_group"] and cls40["main_group"]:
            if cls302["main_group"] != cls40["main_group"]:
                return "check"

    return "ok"


def safety_badge_html(flag):
    if flag == "ok":
        return "<span class='badge badge-ok'>OK</span>"
    return "<span class='badge badge-check'>CHECK</span>"


def build_summary_row(query_text, nq, result, flag):
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
    cls = classify(m302[0].code, "302", query_text, nq.brand) if m302 else {"main_group": None, "sub_group": None, "nova": None}

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


def display_card(m, bls_ver, result_source="", food_desc=None, brand=None):
    """Render a single match as a styled card."""
    color = conf_color(m.confidence)
    bg = conf_bg(m.confidence)
    cls = classify(m.code, bls_ver, food_desc, brand)
    main_str = (cls["main_group"] or "—").replace("_", " ")
    sub_str = (cls["sub_group"] or "—").replace("_", " ")
    pct = int(m.confidence * 100)
    bar_w = max(pct, 5)

    src_html = f" {source_badge(result_source)}" if result_source and m.rank == 1 else ""

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
            {main_str} &nbsp;&middot;&nbsp; {sub_str} &nbsp;&middot;&nbsp; {nova_pill(cls['nova'])}
        </div>
    </div>""", unsafe_allow_html=True)


def get_boosted_candidates(text_ret, query, top_k=30):
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

    query_lower = query.lower().strip()
    for trigger, expansions in CONCEPT_EXPANSION.items():
        if trigger in query_lower:
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

    def sort_trim(d, k):
        items = sorted(d.values(), key=lambda x: (x.to_dict() if hasattr(x, "to_dict") else x)["score"], reverse=True)
        return items[:k]

    return {"query": nq, "bls302": sort_trim(all_302, top_k), "bls40": sort_trim(all_40, top_k)}


# ═══════════════════════════════════════════════════════════
#  Header
# ═══════════════════════════════════════════════════════════

st.markdown("""<div class="app-header">
    <h1>BLS Food Code Matcher</h1>
    <p>Automated mapping of free-text food descriptions to BLS 3.02 and BLS 4.0 codes with food group and NOVA classification</p>
    <div class="institution">Hector-Center for Nutrition, Exercise and Sports &mdash; University Hospital Erlangen</div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### API Configuration")
    use_claude = st.toggle("Claude API re-ranking", value=True,
        help="Claude re-ranks candidates for higher accuracy (~$0.02 per new query, free from cache).")
    default_key = ""
    try:
        default_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        default_key = os.environ.get("ANTHROPIC_API_KEY", "")
    api_key_input = st.text_input("Anthropic API Key", value=default_key, type="password")

    st.markdown("---")
    st.markdown("### Matching Pipeline")
    st.markdown(
        "1. Normalize text & expand synonyms\n"
        "2. Search 14,814 (3.02) + 7,140 (4.0) BLS entries\n"
        "3. Concept expansion (251 food mappings)\n"
        "4. Re-rank top candidates\n"
        "5. Classify food group, sub group & NOVA\n"
        "6. Cross-version safety check"
    )

    st.markdown("---")
    st.markdown("### Cache Statistics")
    cache = get_cache()
    c1, c2, c3 = st.columns(3)
    c1.metric("Entries", cache.size)
    c2.metric("Hits", st.session_state.cache_hits)
    c3.metric("API calls", st.session_state.api_calls)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.78rem; color:#94a3b8;'>"
        "<b>Try:</b> Haferflocken, Spiegelei, Chicken salad, "
        "Magnum Mandel, Joghurt 3,5% Fett, Kimchi, Kürbispesto"
        "</div>", unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════
#  Main area
# ═══════════════════════════════════════════════════════════

with st.spinner("Loading BLS catalogs..."):
    text_ret = load_text_retriever()

query = st.text_input(
    "Food description",
    placeholder="Enter a food description, e.g. Haferflocken, Spiegelei, Döner ...",
    key="food_query",
    label_visibility="collapsed",
)

# Subtle search prompt when empty
if not query:
    st.markdown(
        "<p style='color:#94a3b8; font-size:0.9rem; margin-top:-0.5rem;'>"
        "Enter a food description as written by a study participant.</p>",
        unsafe_allow_html=True,
    )

if query:
    with st.spinner("Searching BLS catalogs..."):
        candidates = get_boosted_candidates(text_ret, query, top_k=30)
    nq = candidates["query"]

    # ── Resolve matches ──
    result = None
    result_from_claude = False

    if use_claude and api_key_input:
        reranker = get_reranker(api_key_input)
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

    # ── Safety flag ──
    flag = safety_flag(result, query, nq.brand)

    # ── Summary table ──
    row = build_summary_row(query, nq, result, flag)
    st.markdown("##### Summary", unsafe_allow_html=True)
    st.markdown(render_summary_table(row), unsafe_allow_html=True)

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
                display_card(m, "302", src302, query, nq.brand)
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
                display_card(m, "40", src40, query, nq.brand)
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


# ═══════════════════════════════════════════════════════════
#  Footer
# ═══════════════════════════════════════════════════════════

st.markdown(
    "<div class='app-footer'>"
    "Developed by Elias Adibi&ensp;|&ensp;"
    "Hector-Center for Nutrition, Exercise and Sports&ensp;|&ensp;"
    "University Hospital Erlangen"
    "</div>",
    unsafe_allow_html=True,
)
