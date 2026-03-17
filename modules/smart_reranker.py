"""
Smart Re-ranker v2
==================
Three-tier re-ranking:

  Tier 1 — Verified lookup (FREE, instant):
    709 verified BLS code mappings for the most common food descriptions.
    Covers ~74% of all rows in the dataset.

  Tier 2 — FAISS + Rule-based scoring (FREE, instant):
    For descriptions not in the verified map, uses semantic search
    candidates scored with BLS domain rules.

  Tier 3 — Claude LLM re-ranking (PAID, slow):
    Only called when Tier 1+2 confidence is low.

Usage:
    from modules.smart_reranker import SmartReranker
    from modules.retriever import Retriever

    retriever = Retriever()
    reranker = SmartReranker()

    candidates = retriever.search("Eier")
    result = reranker.rerank("Eier", candidates)
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import FOOD_GROUP_LETTERS, PROCESSING_STATES
from modules.normalizer import NormalizedQuery
from modules.verified_map import VERIFIED_MAP_302
from modules.verified_map_40 import VERIFIED_MAP_40


@dataclass
class RankedMatch:
    rank: int
    code: str
    name: str
    confidence: float
    reasoning: str
    def to_dict(self) -> dict:
        return {"rank": self.rank, "code": self.code, "name": self.name,
                "confidence": round(self.confidence, 2), "reasoning": self.reasoning}


@dataclass
class RerankerResult:
    food_description: str
    bls302_matches: list[RankedMatch] = field(default_factory=list)
    bls40_matches: list[RankedMatch] = field(default_factory=list)
    used_llm: bool = False
    resolution_path: str = ""
    error: str | None = None


def _is_category_header(code: str) -> bool:
    return code.endswith("00000") or (code.endswith("0000") and code[1] == "0")


def score_candidate(candidate: dict, query: NormalizedQuery) -> tuple[float, str]:
    code = candidate["code"]
    name_de = candidate["name_de"].lower()
    cleaned = query.cleaned.lower()
    original = query.original.lower().strip()
    bonus = 0.0
    reasons = []
    if cleaned in name_de or original in name_de:
        bonus += 0.15; reasons.append("name match")
    if _is_category_header(code):
        bonus -= 0.20; reasons.append("header penalty")
    for kw in ["suppe","salat","curry","eintopf","auflauf","kuchen","soße","sauce","braten","gulasch"]:
        if kw in original and code[0] in ("X", "Y"):
            bonus += 0.10; reasons.append(f"recipe bonus"); break
    if query.prep_state:
        d = {"roh":"0","gekocht":"3","gegart":"2","gebraten":"8","gebacken":"6",
             "frittiert":"9","gedünstet":"5","gedämpft":"4","geröstet":"7","gegrillt":"8"}
        exp = d.get(query.prep_state)
        if exp and code[-1] == exp:
            bonus += 0.10; reasons.append(f"prep match")
    if query.fat_percent:
        f = query.fat_percent.replace(".", ",")
        if f in name_de or query.fat_percent in name_de:
            bonus += 0.12; reasons.append("fat% match")
    if "standardrezeptur" in name_de:
        bonus += 0.05; reasons.append("Standardrezeptur")
    if set(cleaned.split()) & set(name_de.split()):
        bonus += 0.05; reasons.append("word overlap")
    return bonus, "; ".join(reasons) if reasons else "similarity"


def faiss_rerank(candidates, query, catalog):
    scored = []
    for c in candidates:
        cd = c.to_dict() if hasattr(c, "to_dict") else c
        bonus, reasoning = score_candidate(cd, query)
        scored.append((cd["score"] + bonus, cd, reasoning))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [RankedMatch(rank=i+1, code=s[1]["code"], name=s[1]["name_de"],
            confidence=min(s[0], 0.99), reasoning=s[2]) for i, s in enumerate(scored[:3])]


class SmartReranker:
    def __init__(self, api_key=None, llm_threshold=0.60, enable_llm=True):
        self.llm_threshold = llm_threshold
        self.enable_llm = enable_llm
        self._llm_reranker = None
        if enable_llm:
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if key:
                try:
                    from modules.reranker import Reranker
                    self._llm_reranker = Reranker(api_key=key)
                except Exception: pass
        self._catalog_302 = {}
        self._catalog_40 = {}
        self._load_catalogs()

    def _load_catalogs(self):
        try:
            import pandas as pd
            from config.settings import CATALOG_302, CATALOG_40
            if CATALOG_302.exists():
                df = pd.read_parquet(CATALOG_302)
                self._catalog_302 = dict(zip(df["code"], df["name_de"]))
            if CATALOG_40.exists():
                df = pd.read_parquet(CATALOG_40)
                self._catalog_40 = dict(zip(df["code"], df["name_de"]))
        except Exception: pass

    def rerank(self, food_description, retrieval_results):
        result = RerankerResult(food_description=food_description)
        nq = retrieval_results["query"]
        original_lower = food_description.lower().strip()
        try:
            # TIER 1: Verified lookup
            if original_lower in VERIFIED_MAP_302:
                code = VERIFIED_MAP_302[original_lower]
                name = self._catalog_302.get(code, code)
                result.bls302_matches = [RankedMatch(rank=1, code=code, name=name,
                    confidence=0.95, reasoning=f"Verified: '{food_description}' -> [{code}]")]
                result.resolution_path = "verified"
                if retrieval_results["bls40"]:
                    result.bls40_matches = faiss_rerank(retrieval_results["bls40"], nq, self._catalog_40)
                return result

            # TIER 2: FAISS + Rules
            if retrieval_results["bls302"]:
                result.bls302_matches = faiss_rerank(retrieval_results["bls302"], nq, self._catalog_302)
            if retrieval_results["bls40"]:
                result.bls40_matches = faiss_rerank(retrieval_results["bls40"], nq, self._catalog_40)
            result.resolution_path = "faiss+rules"

            # TIER 3: Claude LLM
            top_conf = result.bls302_matches[0].confidence if result.bls302_matches else 0
            if top_conf < self.llm_threshold and self.enable_llm and self._llm_reranker:
                result.used_llm = True
                result.resolution_path = "llm"
                llm_result = self._llm_reranker.rerank(food_description, retrieval_results)
                if not llm_result.error:
                    if llm_result.bls302_matches: result.bls302_matches = llm_result.bls302_matches
                    if llm_result.bls40_matches: result.bls40_matches = llm_result.bls40_matches
        except Exception as e:
            result.error = str(e)
        return result


def print_result(result):
    print(f"\n{'='*70}")
    print(f"  Food: {result.food_description!r}  [{result.resolution_path}]")
    if result.error: print(f"  ERROR: {result.error}"); return
    for ver, matches in [("BLS 3.02", result.bls302_matches), ("BLS 4.0", result.bls40_matches)]:
        print(f"\n  {ver}:")
        for m in matches:
            mk = "G" if m.confidence >= 0.85 else "Y" if m.confidence >= 0.60 else "R"
            print(f"    {m.rank}. [{m.code}] {m.name}  conf={m.confidence:.2f}  {m.reasoning}")


if __name__ == "__main__":
    from modules.retriever import Retriever
    print("Initializing ..."); retriever = Retriever(verbose=True)
    reranker = SmartReranker(enable_llm=False)
    print(f"Verified map: {len(VERIFIED_MAP_302)} entries\nReady!\n")
    TESTS = [
        ("Kaffee","N410100"),("Apfel","F110100"),("Tee","N600100"),
        ("Paprika","G543100"),("Gurke","G520100"),("Brötchen","B501000"),
        ("Eier","E111132"),("Banane","F503100"),("Wasser","N110000"),
        ("Olivenöl","Q120000"),("Haferflocken","C133000"),("Reis","C352032"),
        ("Nudeln","E401032"),("Milch","M110200"),("Joghurt","M141300"),
        ("Käse","M400000"),("Salat","G103100"),("Hähnchen","Y562032"),
        ("Spiegelei","Y710142"),("Rührei","Y720153"),("Kartoffeln","K100022"),
        ("Lachs","T410000"),("Skyr","M713100"),("Hafermilch","H841100"),
        ("Smoothie","F024600"),("Bolognese","Y038213"),("Schnitzel","Y332132"),
    ]
    ok = 0
    for i,(d,e) in enumerate(TESTS,1):
        c = retriever.search(d); r = reranker.rerank(d, c)
        t = r.bls302_matches[0].code if r.bls302_matches else ""
        hit = t == e; ok += hit
        print(f"[{i:2d}] {'OK' if hit else 'XX'} [{r.resolution_path:>10s}] {d:<25s} got=[{t}] exp=[{e}]")
    print(f"\nTop-1: {ok}/{len(TESTS)} = {ok/len(TESTS)*100:.1f}%")
