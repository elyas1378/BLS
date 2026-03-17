"""
BLS Food Code Matcher — Interactive Search Tool (Cloud Version)
===============================================================
Lightweight version for Streamlit Cloud deployment.
Uses text matching + concept expansion + Claude API.
No PyTorch or FAISS needed.

Run locally:  streamlit run app.py
"""

import os
import sys
import re
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="BLS Food Code Matcher", page_icon="🍽️", layout="wide")


@st.cache_resource
def load_text_retriever():
    from modules.text_retriever import TextMatchRetriever
    return TextMatchRetriever(verbose=False)


@st.cache_resource
def load_catalogs():
    import pandas as pd
    from config.settings import CATALOG_302, CATALOG_40
    cat302 = dict(zip(*pd.read_parquet(CATALOG_302)[["code","name_de"]].values.T)) if CATALOG_302.exists() else {}
    cat40 = dict(zip(*pd.read_parquet(CATALOG_40)[["code","name_de"]].values.T)) if CATALOG_40.exists() else {}
    return cat302, cat40


def get_reranker(api_key):
    if not api_key: return None
    try:
        from modules.reranker_v2 import RerankerV2
        return RerankerV2(api_key=api_key)
    except Exception as e:
        st.warning(f"Could not load Claude reranker: {e}")
        return None


def get_smart_reranker():
    from modules.smart_reranker import SmartReranker
    return SmartReranker(enable_llm=False)


def confidence_color(conf):
    if conf >= 0.85: return "#22c55e"
    elif conf >= 0.60: return "#eab308"
    else: return "#ef4444"


def confidence_label(conf):
    if conf >= 0.85: return "🟢 High"
    elif conf >= 0.60: return "🟡 Medium"
    else: return "🔴 Low"


def display_matches(matches, version_label):
    if not matches:
        st.info(f"No matches found for {version_label}")
        return
    for m in matches:
        color = confidence_color(m.confidence)
        col1, col2 = st.columns([1, 11])
        with col1:
            st.markdown(
                f"<div style='text-align:center; font-size:24px; font-weight:bold; color:{color};'>{m.rank}</div>",
                unsafe_allow_html=True)
        with col2:
            st.markdown(f"**`{m.code}`** — {m.name}")
            st.markdown(
                f"<span style='color:{color}; font-weight:600;'>"
                f"{confidence_label(m.confidence)} ({m.confidence:.0%})</span>"
                f"&nbsp;&nbsp;—&nbsp;&nbsp;{m.reasoning}",
                unsafe_allow_html=True)
        st.divider()


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

    # Strategy 1: Original text search
    result = text_ret.search(query, top_k=20)
    merge(all_302, result.get("bls302", [])); merge(all_40, result.get("bls40", []))

    # Strategy 2: Concept expansion
    query_lower = query.lower().strip()
    for trigger, expansions in CONCEPT_EXPANSION.items():
        if trigger in query_lower:
            for exp in expansions[:5]:
                try:
                    r = text_ret.search(exp, top_k=5)
                    merge(all_302, r.get("bls302", []), 0.9); merge(all_40, r.get("bls40", []), 0.9)
                except Exception: pass

    # Strategy 3: Component search
    parts = re.split(r'\s*[,;]\s*|\s+und\s+|\s+mit\s+', query)
    if len(parts) > 1:
        for part in [p.strip() for p in parts[:4] if len(p.strip()) >= 3]:
            try:
                r = text_ret.search(part, top_k=5)
                merge(all_302, r.get("bls302", []), 0.85); merge(all_40, r.get("bls40", []), 0.85)
            except Exception: pass

    def sort_trim(d, k):
        items = sorted(d.values(), key=lambda x: (x.to_dict() if hasattr(x, "to_dict") else x)["score"], reverse=True)
        return items[:k]

    return {"query": nq, "bls302": sort_trim(all_302, top_k), "bls40": sort_trim(all_40, top_k)}


# ── UI ──
st.title("🍽️ BLS Food Code Matcher")
st.markdown("Type a food description (as written by a study participant) to find the matching BLS codes in both **BLS 3.02** and **BLS 4.0**.")

with st.sidebar:
    st.header("⚙️ Settings")
    use_claude = st.toggle("Use Claude API for re-ranking", value=True,
        help="ON: Claude re-ranks candidates (~$0.02/query). OFF: rule-based only (free).")
    default_key = ""
    try: default_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except: default_key = os.environ.get("ANTHROPIC_API_KEY", "")
    api_key_input = st.text_input("Anthropic API Key", value=default_key, type="password")
    st.divider()
    st.header("📊 Pipeline")
    st.markdown("1. **Normalize** — clean text, expand synonyms\n2. **Text search** — 14,814 + 7,140 BLS entries\n3. **Concept expansion** — 239 food mappings\n4. **Re-rank** — Claude picks best 3\n5. **Validate** — codes checked against catalog")
    st.divider()
    st.markdown("**Try:** Haferflocken, Spiegelei, Chicken salad, Magnum Mandel (Eis), Döner, Joghurt 3,5% Fett, Kimchi, Kürbiscurry, Käsekuchen")

with st.spinner("Loading BLS catalogs..."):
    text_ret = load_text_retriever()

query = st.text_input("🔍 Food description", placeholder="e.g., Haferflocken, Spiegelei, Döner...", key="food_query")

if query:
    with st.spinner("Searching BLS catalogs..."):
        candidates = get_boosted_candidates(text_ret, query, top_k=30)
    nq = candidates["query"]

    with st.expander("🔧 Normalization details", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Original:** {nq.original}"); st.markdown(f"**Cleaned:** {nq.cleaned}")
        with c2:
            if nq.prep_state: st.markdown(f"**Prep state:** {nq.prep_state}")
            if nq.fat_percent: st.markdown(f"**Fat %:** {nq.fat_percent}")
            if nq.brand: st.markdown(f"**Brand:** {nq.brand}")
            if nq.is_english: st.markdown("**Language:** English")
        st.markdown(f"**Candidates:** {len(candidates.get('bls302',[]))} (3.02) / {len(candidates.get('bls40',[]))} (4.0)")

    if use_claude and api_key_input:
        reranker = get_reranker(api_key_input)
        if reranker:
            with st.spinner("🤖 Claude is analyzing candidates..."):
                result = reranker.rerank(query, candidates)
            if result.error:
                st.error(f"API error: {result.error}")
            else:
                is_verified = any("Verified" in m.reasoning for m in (result.bls302_matches or []))
                st.caption("✅ Verified lookup — no API cost" if is_verified else "🤖 Re-ranked by Claude API")
                col302, col40 = st.columns(2)
                with col302: st.subheader("📘 BLS 3.02"); display_matches(result.bls302_matches, "BLS 3.02")
                with col40: st.subheader("📗 BLS 4.0"); display_matches(result.bls40_matches, "BLS 4.0")
        else:
            use_claude = False

    if not use_claude or not api_key_input:
        smart = get_smart_reranker()
        result = smart.rerank(query, candidates)
        st.caption("✅ Verified lookup" if result.resolution_path == "verified" else "📏 Rule-based. Add API key for better results.")
        col302, col40 = st.columns(2)
        with col302: st.subheader("📘 BLS 3.02"); display_matches(result.bls302_matches, "BLS 3.02")
        with col40: st.subheader("📗 BLS 4.0"); display_matches(result.bls40_matches, "BLS 4.0")

    with st.expander("📋 Raw candidates", expanded=False):
        t1, t2 = st.tabs(["BLS 3.02", "BLS 4.0"])
        with t1:
            for i, c in enumerate(candidates.get("bls302", [])[:20], 1):
                cd = c.to_dict() if hasattr(c, "to_dict") else c
                st.text(f"{i:2d}. [{cd['code']}] {cd['name_de'][:55]:<55s} score={cd['score']:.3f}")
        with t2:
            for i, c in enumerate(candidates.get("bls40", [])[:20], 1):
                cd = c.to_dict() if hasattr(c, "to_dict") else c
                st.text(f"{i:2d}. [{cd['code']}] {cd['name_de'][:55]:<55s} score={cd['score']:.3f}")

st.divider()
st.caption("BLS Food Code Matcher v1.0 — Text Search + Concept Expansion + Claude — BLS 3.02 (14,814) • BLS 4.0 (7,140)")
