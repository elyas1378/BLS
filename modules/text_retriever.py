"""
Text Match Retriever
====================
Searches the BLS catalog using direct text matching (substring + fuzzy).
Much more accurate than embedding-only search for straightforward food names.

Strategy:
  1. Exact match (food name == BLS entry)
  2. Substring match (food name is contained in BLS entry)  
  3. Fuzzy string match (for misspellings and slight variations)
  4. Fall back to FAISS for creative/unusual descriptions

Then applies BLS code rules to pick the best variant:
  - Fruits/vegetables: prefer "roh" (raw) entries
  - Beverages: prefer "(Getränk)" entries
  - Avoid generic category headers (codes ending in 000/0000)
  - Food diary context: consumed = cooked for meat, raw for produce
"""

from __future__ import annotations

import bisect
import re
import sys
from pathlib import Path
from difflib import SequenceMatcher
from dataclasses import dataclass
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import CATALOG_302, CATALOG_40


# ═══════════════════════════════════════════
#  German compound word splitter
# ═══════════════════════════════════════════

# Common German food morphemes — word parts that appear in compound words.
# Sorted longest-first so greedy matching picks the longest part.
_MORPHEMES = sorted([
    # Food types / dish forms
    "brötchen", "brot", "kuchen", "waffel", "stange", "suppe", "soße",
    "aufstrich", "füllung", "stampf", "püree", "salat", "mousse", "creme",
    "curry", "wickel", "auflauf", "pfanne", "bowl", "crunch", "riegel",
    "eintopf", "gulasch", "lasagne", "gratin", "pesto", "dressing",
    "aufschnitt", "pastete", "knödel", "kloß", "klöße", "nockerl", "strudel",
    "fladen", "tortilla", "wrap", "burger", "nuggets", "schnitzel",
    "frikadelle", "bulette", "rösti", "pommes", "chips", "cracker",
    "müsli", "porridge", "smoothie", "shake", "joghurt", "quark",
    "pudding", "eis", "torte", "praline", "bonbon", "dragee",
    "konfitüre", "marmelade", "gelee", "kompott", "mus",
    "teig", "sauerteig", "blätterteig", "hefeteig", "mürbeteig",
    "pfannkuchen", "plätzchen", "gebäck", "keks", "kekse",
    "knäckebrot", "toastbrot", "mischbrot",
    "kroketten", "taschen", "speise", "brei", "einlage",
    # Liquids
    "saft", "schorle", "milch", "drink", "tee", "kaffee", "wasser",
    "sirup", "brühe", "fond", "likör", "wein", "bier",
    # Ingredients / processing
    "mehl", "flocken", "samen", "schalen", "pulver", "extrakt",
    "öl", "fett", "butter", "schmalz", "essig",
    "zucker", "honig", "salz", "senf", "ketchup",
    "sauce", "marinade", "mark",
    "mischung", "zubereitung", "produkt", "geschmack",
    # Grains
    "hafer", "roggen", "weizen", "dinkel", "gerste", "hirse",
    "buchweizen", "amaranth", "quinoa", "mais", "reis",
    # Vegetables
    "kartoffel", "karotten", "karotte", "tomaten", "tomate",
    "spinat", "kürbis", "artischocke", "artischocken",
    "zwiebel", "knoblauch", "lauch", "paprika", "gurke", "gurken",
    "brokkoli", "blumenkohl", "kohlrabi", "rosenkohl", "rotkohl",
    "weißkohl", "spargel", "fenchel", "sellerie", "rucola",
    "champignon", "pilz", "erbsen", "bohnen", "linsen",
    "zucchini", "aubergine", "radieschen",
    "kümmel", "schwarzkümmel", "maronen", "kastanien", "pastinake",
    "kohl", "schoten", "wurzel", "kern",
    # Fruits
    "apfel", "birne", "kirsche", "pflaume", "zwetschge", "beeren",
    "beere", "erdbeere", "himbeere", "heidelbeere", "johannisbeere",
    "orange", "zitrone", "banane", "mango", "ananas", "melone",
    "traube", "feige", "dattel", "cranberry",
    # Nuts/seeds
    "nuss", "nüsse", "mandel", "mandeln", "haselnuss", "walnuss",
    "cashew", "erdnuss", "pistazie", "sesam", "mohn", "kokos",
    "kürbiskern", "sonnenblumenkern", "leinsamen", "flohsamen",
    # Protein
    "schinken", "käse", "fleisch", "hack", "wurst", "speck",
    "lachs", "thunfisch", "hering", "forelle", "garnele",
    "hähnchen", "pute", "rind", "schwein", "kalb", "lamm",
    "ei", "eier", "barsch", "filet", "bauch",
    # Dairy
    "sahne", "schmand", "frischkäse", "mozzarella", "parmesan",
    "gouda", "emmentaler", "camembert", "feta",
    # Sweets/flavorings
    "schoko", "schokolade", "vanille", "karamell", "zimt",
    "nougat", "marzipan", "krokant",
    # Descriptors
    "vollkorn", "voll", "korn", "braun", "schwarz", "weiß", "rot",
    "grün", "dunkel", "hell", "fein", "grob", "zart", "bitter",
    "natur", "bio", "frisch", "trocken", "geräuchert",
    # Regions/styles
    "toskana", "griechisch", "italienisch", "asiatisch",
    # Misc
    "gemüse", "obst", "kräuter", "gewürz",
    "protein", "vitamin", "mineral", "energie",
    "mann", "manner", "maggi", "knorr",
], key=len, reverse=True)


# Weak modifiers: generic descriptors that should never be the sole match reason
_WEAK_MODIFIERS = {
    # Colors
    "braun", "schwarz", "weiß", "rot", "grün", "gelb", "blau",
    "dunkel", "hell",
    # Size/texture
    "groß", "klein", "fein", "grob", "zart", "dick", "dünn",
    "rund", "lang",
    # Taste
    "bitter", "süß", "salzig", "scharf", "mild", "sauer",
    # State
    "frisch", "alt", "neu", "warm", "kalt", "heiß", "roh", "gar",
    # Amount
    "voll", "halb", "ganz", "extra", "mini", "maxi",
}


def split_compound(word: str) -> list[str]:
    """Split a German compound food word into meaningful morphemes.

    Always splits on hyphens. For non-hyphenated compounds >= 8 chars,
    tries greedy left-to-right morpheme matching.

    Returns list of parts, or [word] if no split found.
    """
    word_lower = word.lower().strip()

    # Step 1: Split on hyphens
    if "-" in word_lower:
        parts = [p.strip() for p in word_lower.split("-") if p.strip()]
        # Recursively split each part if it's a long compound
        result = []
        for part in parts:
            if len(part) >= 8:
                sub = _split_no_hyphen(part)
                result.extend(sub)
            else:
                result.append(part)
        return result if len(result) > 1 else [word_lower]

    # Step 2: Non-hyphenated compounds
    if len(word_lower) >= 8:
        parts = _split_no_hyphen(word_lower)
        if len(parts) > 1:
            return parts

    return [word_lower]


def _split_no_hyphen(word: str) -> list[str]:
    """Try to split a non-hyphenated compound word using morpheme list."""
    parts = []
    pos = 0
    while pos < len(word):
        matched = False
        # Try longest morpheme first (list is sorted longest-first)
        for morph in _MORPHEMES:
            if word[pos:].startswith(morph):
                parts.append(morph)
                pos += len(morph)
                # Skip linking letters (common: s, n, en, e, er, es)
                if pos < len(word) and pos + 1 < len(word):
                    tail = word[pos:]
                    skip = 0
                    for link in ('en', 'er', 'es', 's', 'n', 'e'):
                        if tail.startswith(link) and pos + len(link) < len(word):
                            remaining = word[pos + len(link):]
                            if any(remaining.startswith(m) for m in _MORPHEMES):
                                skip = len(link)
                                break
                    if skip:
                        pos += skip
                matched = True
                break
        if not matched:
            # No morpheme matched at this position — unsplittable remainder
            remainder = word[pos:]
            if remainder and len(remainder) >= 3:
                parts.append(remainder)
            pos = len(word)

    # Only return split if we got at least 2 meaningful parts
    if len(parts) >= 2 and all(len(p) >= 2 for p in parts):
        return parts
    return [word]


@dataclass
class TextCandidate:
    code: str
    name_de: str
    name_en: str
    score: float
    food_group: str
    processing_digit: str
    portion_g: float = None
    match_type: str = ""  # "exact", "substring", "fuzzy", "faiss"

    def to_dict(self):
        return {
            "code": self.code,
            "name_de": self.name_de,
            "name_en": self.name_en,
            "score": round(self.score, 4),
            "food_group": self.food_group,
            "processing_digit": self.processing_digit,
            "portion_g": self.portion_g,
            "match_type": self.match_type,
        }


class TextMatchRetriever:
    """
    Searches BLS catalogs using text matching + BLS domain rules.
    """

    def __init__(self, verbose=True):
        if verbose:
            print("Loading TextMatchRetriever …")

        self.catalog_302 = self._load_catalog(CATALOG_302, "BLS 3.02", verbose)
        self.catalog_40 = self._load_catalog(CATALOG_40, "BLS 4.0", verbose)

        # Build word index for fast lookup
        self._word_index_302 = self._build_word_index(self.catalog_302)
        self._word_index_40 = self._build_word_index(self.catalog_40)

        # Build sorted prefix index for O(log n) prefix matching
        self._prefix_keys_302 = sorted(self._word_index_302.keys())
        self._prefix_keys_40 = sorted(self._word_index_40.keys())

        if verbose:
            print(f"  Ready ✓")

    def _load_catalog(self, path, label, verbose):
        if not path.exists():
            raise FileNotFoundError(f"{label} catalog not found at {path}")
        df = pd.read_parquet(path)
        df["name_lower"] = df["name_de"].str.lower()
        # Pre-split words for fast matching
        df["name_words"] = df["name_lower"].str.findall(r'\w+')
        if verbose:
            print(f"  Loaded {len(df):,} entries for {label}")
        return df

    def _build_word_index(self, df):
        """Build inverted index: word → list of row indices."""
        index = {}
        for idx, words in df["name_words"].items():
            if isinstance(words, list):
                for w in words:
                    if len(w) >= 3:  # skip very short words
                        if w not in index:
                            index[w] = []
                        index[w].append(idx)
        return index

    @staticmethod
    def _prefix_lookup(prefix: str, sorted_keys: list, word_index: dict,
                       limit: int = 50) -> list[int]:
        """O(log n) prefix lookup using bisect on a sorted key list."""
        lo = bisect.bisect_left(sorted_keys, prefix)
        results = []
        # Forward scan: keys starting with prefix
        i = lo
        while i < len(sorted_keys) and sorted_keys[i].startswith(prefix):
            results.extend(word_index[sorted_keys[i]][:limit])
            i += 1
        return results

    def _text_search(self, query: str, catalog: pd.DataFrame, word_index: dict,
                     top_k: int = 20, prefix_keys: list = None) -> list[TextCandidate]:
        """
        Multi-strategy text search:
        1. Exact name match
        2. Word-based candidate retrieval + scoring
        3. Substring matching
        """
        query_lower = query.lower().strip()
        query_words = set(re.findall(r'\w+', query_lower))
        query_words = {w for w in query_words if len(w) >= 3}

        # Expand query words with compound splits + head/modifier tagging
        compound_parts = set()
        compound_heads = set()   # last part of each compound (the "what")
        compound_mods = set()    # earlier parts (the "which kind")
        for w in list(query_words):
            parts = split_compound(w)
            if len(parts) > 1:
                valid = [p for p in parts if len(p) >= 3]
                compound_parts.update(valid)
                if valid:
                    compound_heads.add(valid[-1])        # last = head
                    compound_mods.update(valid[:-1])     # rest = modifiers

        candidates = {}  # code → (score, row, match_type)

        # ── Strategy 1: Exact match ──
        exact = catalog[catalog["name_lower"] == query_lower]
        for _, row in exact.iterrows():
            candidates[row["code"]] = (2.0, row, "exact")

        # ── Strategy 2: Word-based retrieval ──
        # Find all entries that share at least one word with the query
        candidate_indices = set()
        all_search_words = query_words | compound_parts
        for word in all_search_words:
            if word in word_index:
                candidate_indices.update(word_index[word])
            # O(log n) prefix matching via sorted key list + bisect
            if prefix_keys is not None:
                candidate_indices.update(
                    self._prefix_lookup(word, prefix_keys, word_index, 50)
                )

        for idx in candidate_indices:
            if idx >= len(catalog):
                continue
            row = catalog.iloc[idx]
            code = row["code"]
            if code in candidates:
                continue

            name_lower = row["name_lower"]

            # Score based on multiple factors
            score = 0.0

            # Substring: query in name or name in query
            if query_lower in name_lower:
                score += 0.8
            elif name_lower in query_lower:
                score += 0.6

            # Word overlap (original query words)
            name_words = set(row["name_words"]) if isinstance(row["name_words"], list) else set()
            overlap = query_words & name_words
            if overlap:
                score += 0.3 * len(overlap) / max(len(query_words), 1)

            # Compound part overlap with head/modifier weighting
            if compound_parts:
                comp_overlap = compound_parts & name_words
                if comp_overlap:
                    score += 0.255 * len(comp_overlap) / max(len(compound_parts), 1)

                    # Head/modifier grammar weighting
                    head_matched = bool(compound_heads & name_words)
                    mod_matched = bool(compound_mods & name_words)
                    if head_matched and mod_matched:
                        score += 0.10   # both head + modifier = best match
                    elif not head_matched and mod_matched:
                        score -= 0.20   # only modifier, missed head = poor match
                else:
                    # No compound overlap at all — check if only weak modifiers match
                    pass

            # Weak modifier penalty: if the ONLY reason this candidate appeared
            # is a generic descriptor word, penalize it
            if compound_mods:
                all_matching = (query_words | compound_parts) & name_words
                if all_matching and all_matching.issubset(_WEAK_MODIFIERS):
                    score -= 0.25

            # Fuzzy string similarity (weighted higher to catch 1-2 char typos)
            sim = SequenceMatcher(None, query_lower, name_lower).ratio()
            score += sim * 0.6

            # Name length penalty: prefer shorter (more specific) names
            # but not too short (category headers)
            name_len = len(name_lower)
            if 5 <= name_len <= 40:
                score += 0.1
            elif name_len > 60:
                score -= 0.05

            if score > 0.15:
                candidates[code] = (score, row, "text")

        # ── Strategy 3: Substring search for remaining ──
        if len(candidates) < 5 and len(query_lower) >= 4:
            mask = catalog["name_lower"].str.contains(
                re.escape(query_lower), case=False, na=False
            )
            for _, row in catalog[mask].head(20).iterrows():
                code = row["code"]
                if code not in candidates:
                    sim = SequenceMatcher(None, query_lower, row["name_lower"]).ratio()
                    candidates[code] = (sim + 0.3, row, "substring")

        # ── Apply BLS domain rules + suffix boosting ──
        scored = []
        for code, (score, row, match_type) in candidates.items():
            adjusted = self._apply_bls_rules(score, code, row, query_lower,
                                             compound_parts)
            scored.append((adjusted, code, row, match_type))

        # Sort by adjusted score
        scored.sort(key=lambda x: x[0], reverse=True)

        # Build result
        results = []
        for adj_score, code, row, match_type in scored[:top_k]:
            results.append(TextCandidate(
                code=code,
                name_de=row["name_de"],
                name_en=row.get("name_en", ""),
                score=adj_score,
                food_group=code[0],
                processing_digit=code[-1],
                portion_g=row.get("portion_g"),
                match_type=match_type,
            ))
        return results

    def _apply_bls_rules(self, score: float, code: str, row, query: str,
                         compound_parts: set = None) -> float:
        """Apply BLS code structure rules + suffix category boosting."""
        name = row["name_lower"]
        group = code[0]
        proc = code[-1]
        if compound_parts is None:
            compound_parts = set()

        # ── Penalize broad category headers ──
        if code.endswith("00000"):
            score -= 0.4
        elif code.endswith("0000"):
            score -= 0.2
        elif code.endswith("000") and proc == "0":
            # Could be a valid generic entry (like "Honig S120000")
            # Only penalize if there's a more specific version
            score -= 0.05

        # ── Fruits (F): prefer raw (position 5-6 = "10") ──
        if group == "F":
            if code[4:6] == "10":  # raw, edible portion
                score += 0.15
            elif code[4:6] == "01":  # with kitchen waste
                score -= 0.05
            if "roh" in name:
                score += 0.05

        # ── Vegetables (G): prefer raw unless cooking implied ──
        if group == "G":
            if code[4:6] == "10":
                score += 0.10
            if "roh" in name:
                score += 0.05

        # ── Beverages (N): prefer "(Getränk)" entries ──
        if group == "N":
            if "getränk" in name or "(getränk)" in name:
                score += 0.20
            if "pulver" in name or "trocken" in name:
                score -= 0.15

        # ── Eggs (E): prefer gekocht (boiled) ──
        if group == "E" and "hühnerei" in name:
            if proc == "3":  # gekocht
                score += 0.10
            if "vollei" in name:
                score += 0.05

        # ── Dairy (M): prefer common fat levels ──
        if group == "M":
            if "1,5%" in name or "1,5 %" in name:
                score += 0.03
            if "3,5%" in name or "3,5 %" in name:
                score += 0.02

        # ── Meat cooked (U/Y): prefer gegart/gekocht ──
        if group in ("U", "Y"):
            if proc in ("2", "3"):
                score += 0.05

        # ── Recipe categories (X/Y): prefer Standardrezeptur ──
        if group in ("X", "Y") and "standardrezeptur" in name:
            score += 0.05

        # ── Recipe categories (X/Y): prefer household preparation ──
        # Position 6 (code[5]): 1/4=household, 2/5=industrial, 3/6=gastronomy
        if group in ("X", "Y") and len(code) >= 6:
            if code[5] in ("1", "4"):
                score += 0.10
            elif code[5] in ("2", "3", "5", "6"):
                score -= 0.05

        # ── Avoid "Konfitüre" (jam) for raw fruit searches ──
        if "konfitüre" in name and "konfitüre" not in query:
            score -= 0.3
        if "konzentrat" in name and "konzentrat" not in query:
            score -= 0.2
        if "trunk" in name and "trunk" not in query and "saft" not in query:
            score -= 0.2
        if "pulver" in name and "pulver" not in query:
            score -= 0.15
        if "getrocknet" in name and "getrocknet" not in query and "trocken" not in query:
            score -= 0.1

        # ═══════════════════════════════════════════
        #  Suffix category boosting
        # ═══════════════════════════════════════════

        # Collect all words to check for suffixes: query words + compound parts
        all_words = set(re.findall(r'\w+', query)) | compound_parts
        query_full = query  # keep original for context checks

        # Suffix → (boost_groups, boost_amount, name_keyword_required)
        _SUFFIX_RULES = {
            "aufstrich":  ({"Q", "R"}, 0.20, None),
            "suppe":      ({"X", "Y"}, 0.20, "suppe"),
            "eintopf":    ({"X", "Y"}, 0.20, "eintopf"),
            "brot":       ({"B"},      0.25, None),
            "brötchen":   ({"B"},      0.25, None),
            "kuchen":     ({"D"},      0.20, None),
            "torte":      ({"D"},      0.20, None),
            "saft":       ({"N", "F"}, 0.20, None),
            "schorle":    ({"N"},      0.20, None),
            "getränk":    ({"N"},      0.20, None),
            "drink":      ({"N", "H"}, 0.20, None),
            "öl":         ({"Q"},      0.25, None),
            "salat":      (None,       0.15, "salat"),  # any group with "salat" in name
            "stampf":     ({"K", "X"}, 0.20, None),
            "püree":      ({"K", "X"}, 0.20, None),
            "mousse":     ({"S", "D"}, 0.15, None),
            "creme":      ({"S", "M"}, 0.15, None),
            "pudding":    ({"S"},      0.15, None),
            "eis":        ({"S"},      0.15, None),
            "wurst":      ({"W"},      0.20, None),
            "curry":      ({"X", "Y"}, 0.15, None),
            "pfanne":     ({"X", "Y"}, 0.15, None),
            "auflauf":    ({"X", "Y"}, 0.15, None),
            "gratin":     ({"X", "Y"}, 0.15, None),
        }

        for word in all_words:
            for suffix, (boost_groups, boost_amt, name_kw) in _SUFFIX_RULES.items():
                if word.endswith(suffix) and word != suffix:
                    # Suffix matched — apply boost or penalty
                    if name_kw:
                        # Boost if the BLS name contains the keyword
                        if name_kw in name:
                            score += boost_amt
                    elif boost_groups and group in boost_groups:
                        score += boost_amt
                    elif boost_groups and group not in boost_groups:
                        # Mild penalty for wrong category
                        score -= 0.08
                    break  # one suffix match per word

        # "-milch" suffix: boost N (beverages) but NOT for "buttermilch"
        for word in all_words:
            if word.endswith("milch") and word != "milch" and word != "buttermilch":
                if group in ("N", "H", "C"):
                    score += 0.20
                break

        # ── Context-aware penalties ──

        # "gemüse" + recipe suffix → penalize meat categories
        if "gemüse" in query_full:
            has_recipe_suffix = any(
                w.endswith(s) for w in all_words
                for s in ("curry", "pfanne", "auflauf", "eintopf")
            )
            if has_recipe_suffix and group in ("U", "V", "W", "Y"):
                score -= 0.15

        # "heiße"/"heißer"/"heißes" → boost beverages
        if any(h in query_full for h in ("heiße", "heißer", "heißes")):
            if group == "N":
                score += 0.25
            elif group in ("B", "D", "S", "U", "V", "W"):
                score -= 0.10

        # "zero"/"light"/"zuckerfrei"/"kalorienarm" → boost diet variants
        if any(d in query_full for d in ("zero", "light", "zuckerfrei",
                                          "kalorienarm", "ohne zucker")):
            if any(d in name for d in ("light", "kalorienreduziert",
                                        "süßstoff", "zuckerfrei", "kalorienarm")):
                score += 0.15

        return score

    def _search_302(self, query: str, top_k: int) -> list[TextCandidate]:
        """Search BLS 3.02 catalog."""
        return self._text_search(query, self.catalog_302, self._word_index_302,
                                 top_k, self._prefix_keys_302)

    def _search_40(self, query: str, top_k: int) -> list[TextCandidate]:
        """Search BLS 4.0 catalog."""
        return self._text_search(query, self.catalog_40, self._word_index_40,
                                 top_k, self._prefix_keys_40)

    def search(self, food_description: str, top_k: int = 20,
               normalize_input: bool = True):
        """
        Search both BLS catalogs for matches.

        Returns dict compatible with SmartReranker:
            {"query": NormalizedQuery, "bls302": [...], "bls40": [...]}
        """
        from modules.normalizer import normalize, NormalizedQuery

        if normalize_input:
            nq = normalize(food_description)

            # Collect all query variants to search
            queries = [food_description]
            if nq.cleaned != food_description.lower().strip():
                queries.append(nq.cleaned)
            queries.extend(nq.search_variants)

            # Search all variants sequentially against both catalogs
            # (outer parallelism is handled by _search_pool in app.py)
            results_302 = []
            results_40 = []
            for q in queries:
                results_302 = self._merge(results_302, self._search_302(q, top_k), top_k)
                results_40 = self._merge(results_40, self._search_40(q, top_k), top_k)
        else:
            nq = NormalizedQuery(original=food_description, cleaned=food_description)
            results_302 = self._search_302(food_description, top_k)
            results_40 = self._search_40(food_description, top_k)

        return {
            "query": nq,
            "bls302": results_302,
            "bls40": results_40,
        }

    def _merge(self, list_a, list_b, top_k):
        """Merge two candidate lists, keeping higher score for duplicates."""
        by_code = {}
        for c in list_a:
            by_code[c.code] = c
        for c in list_b:
            if c.code not in by_code or c.score > by_code[c.code].score:
                by_code[c.code] = c
        merged = sorted(by_code.values(), key=lambda x: x.score, reverse=True)
        return merged[:top_k]


# =====================================================================
#  CLI test
# =====================================================================

if __name__ == "__main__":
    retriever = TextMatchRetriever()

    tests = [
        "Apfel", "Kaffee", "Tee", "Gurke", "Paprika", "Brötchen",
        "Honig", "Karotte", "Espresso", "Mandarine", "Avocado",
        "Zwiebel", "Leinsamen", "Mozzarella", "Banane", "Eier",
        "Olivenöl", "Haferflocken", "Skyr", "Hafermilch",
        "Bolognese", "Kürbiscurry", "Chicken salad",
    ]

    for food in tests:
        result = retriever.search(food, top_k=5)
        print(f"\n{'─'*60}")
        print(f"  '{food}':")
        for c in result["bls302"][:3]:
            print(f"    [{c.code}] {c.name_de:<50s} score={c.score:.3f} ({c.match_type})")
