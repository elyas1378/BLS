"""
NOVA Classification
====================
Layer 1: BLS code structure rules (letter → NOVA)
Layer 2: Description & keyword overrides (correct Layer 1 mistakes)
Layer 3: Claude fallback for low-confidence results (no extra API call —
         piggybacks on the existing BLS re-ranking request)

BLS code format: XNNNNNN (1 letter + 6 digits)
  Position 1 (letter) = food group
  Position 6 (BLS 3.02) = processing state
"""

from __future__ import annotations

NOVA_CONFIDENCE_THRESHOLD = 0.70


def needs_claude_nova(confidence: float) -> bool:
    """Return True if confidence is below threshold and Claude should classify NOVA."""
    return confidence < NOVA_CONFIDENCE_THRESHOLD


# ═══════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════

def classify_nova(bls_code: str, bls_version: str = "302",
                  food_description: str = "", brand: str | None = None,
                  verify_with_llm: bool = True,
                  cache=None) -> dict:
    """
    Classify a BLS code into NOVA (1–4).
    Runs Freiburger override (Layer 0), Layer 1 (code structure), Layer 2
    (description overrides), and a post-cap when Freiburger asserts "NOT_4".

    Layer 3 (Claude verification) fires when verify_with_llm=True AND the
    rule-based result is low-confidence. Freiburger hits (confidence 0.98)
    and other high-confidence rule hits skip the LLM — no wasted calls.
    Set verify_with_llm=False to force a purely offline classification.

    ``cache`` is an optional PersistentCache instance; when provided, the
    LLM call is skipped on cache hits. Cache writes happen via
    ``cache.log_search(..., llm_nova=...)`` which the caller is expected
    to invoke after classification.

    Returns:
        {"nova": int|None, "confidence": float, "method": str, "reason": str,
         "needs_claude": bool, "rule_nova": int|None,
         "llm_agreed": bool (only if verified),
         "llm_reason": str (only if verified),
         "llm_source": "cache"|"llm" (only if verified)}
    """
    # ── Layer 0: Freiburger Ernährungsprotokoll override ──
    # Sidney & Leonie's standardization list (172 items) overrides everything
    # below for exact name matches. "NOT_4" entries fall through and are handled
    # by the cap at the end.
    from modules.freiburger_nova import lookup_nova, is_not_nova4
    freiburger_hit = lookup_nova(food_description)
    if freiburger_hit is not None:
        return {
            "nova": freiburger_hit,
            "confidence": 0.98,
            "method": "freiburger_override",
            "reason": "Freiburger Protokoll standardization list",
            "needs_claude": False,
        }

    layer1 = _layer1_code_structure(bls_code)
    result = _layer2_description_override(
        layer1, bls_code, food_description, brand
    )

    # Cap at 3 when Freiburger explicitly asserts NOT_4
    if result.get("nova") == 4 and is_not_nova4(food_description):
        result["nova"] = 3
        result["method"] = (result.get("method", "") + "+freiburger_cap").strip("+")
        result["reason"] = (
            (result.get("reason", "") or "") + "; capped to 3 (Freiburger NOT_4)"
        ).lstrip("; ")
        result["confidence"] = min(result.get("confidence", 0.8), 0.9)

    result["needs_claude"] = needs_claude_nova(result["confidence"])

    # Preserve the rule-based guess even after LLM verification so the
    # caller can log both (useful for auditing when Claude disagrees).
    result["rule_nova"] = result.get("nova")

    # ── Layer 3: LLM verification (only when the rule-based layers are
    # uncertain). High-confidence rule hits and Freiburger overrides skip
    # this — they don't need a second opinion and we don't pay for one.
    if verify_with_llm and (result["needs_claude"] or result.get("nova") is None):
        from modules.nova_llm_verifier import verify_nova

        cache_lookup = cache.get_nova_cache if cache is not None else None

        verdict = verify_nova(
            code=bls_code,
            description=food_description,
            rule_based_nova=result.get("nova"),
            rule_based_reason=result.get("reason", ""),
            cache_lookup=cache_lookup,
        )

        if verdict is not None:
            result["nova"] = verdict["nova"]
            result["llm_agreed"] = verdict["agree"]
            result["llm_reason"] = verdict["reason"]
            result["llm_source"] = verdict["source"]
            result["method"] = (
                (result.get("method", "") or "") + "+llm_verify"
            ).strip("+")
            result["reason"] = (
                (result.get("reason", "") or "")
                + (f" | LLM: {verdict['reason']}" if verdict["reason"] else "")
            ).strip(" |")
            # Verified results get a confidence bump. Agreement is stronger
            # signal than disagreement (where we're overriding our own rule).
            result["confidence"] = 0.95 if verdict["agree"] else 0.85
            result["needs_claude"] = False

    return result


# ═══════════════════════════════════════════════════════════
#  Layer 1 — BLS code structure rules
# ═══════════════════════════════════════════════════════════

def _layer1_code_structure(bls_code: str) -> dict:
    if not bls_code or len(bls_code) < 2:
        return _result(None, 0.0, "code_structure", "invalid or empty code")

    letter = bls_code[0].upper()

    # NOVA 1
    if letter == "F":
        return _result(1, 0.95, "code_structure", "F = Fruits")
    if letter == "G":
        return _result(1, 0.90, "code_structure", "G = Vegetables")
    if letter == "H":
        return _result(1, 0.85, "code_structure", "H = Legumes/Nuts")
    if letter == "K":
        return _result(1, 0.85, "code_structure", "K = Potatoes")
    if letter == "E":
        if bls_code[1:2] in ("0", "1"):
            return _result(1, 0.90, "code_structure", "E0/E1 = Eggs")
        return _result(3, 0.80, "code_structure", "E2+ = Pasta")
    if letter == "N":
        second = bls_code[1:2]
        if second in ("1", "2"):
            return _result(1, 0.90, "code_structure", "N1/N2 = Water/mineral water")
        if second in ("3",):
            return _result(4, 0.80, "code_structure", "N3 = Soft drinks/juices")
        if second in ("4", "5"):
            return _result(1, 0.85, "code_structure", "N4/N5 = Coffee")
        if second in ("6", "7"):
            return _result(1, 0.85, "code_structure", "N6/N7 = Tea")
        return _result(4, 0.70, "code_structure", "N8+ = Other beverages")
    if letter == "V":
        return _result(1, 0.85, "code_structure", "V = Raw meat/poultry")
    if letter == "T":
        return _result(1, 0.80, "code_structure", "T = Fish")
    if letter == "U":
        return _result(1, 0.75, "code_structure", "U = Meat cooked")

    # NOVA 2
    if letter == "Q":
        return _result(2, 0.90, "code_structure", "Q = Fats/Oils")

    # NOVA 3
    if letter == "B":
        return _result(3, 0.90, "code_structure", "B = Bread/rolls")
    if letter == "M":
        return _result(3, 0.65, "code_structure", "M = Dairy (mixed)")
    if letter == "P":
        return _result(3, 0.85, "code_structure", "P = Alcoholic beverages")
    if letter == "R":
        return _result(3, 0.70, "code_structure", "R = Condiments/sauces")
    if letter == "C":
        return _result(3, 0.65, "code_structure", "C = Cereals (mixed)")

    # NOVA 4
    if letter == "D":
        return _result(4, 0.80, "code_structure", "D = Pastries/Cakes")
    if letter == "W":
        return _result(4, 0.90, "code_structure", "W = Sausages/Processed meat")
    if letter == "S":
        return _result(4, 0.85, "code_structure", "S = Sweets/Candy")

    # Mixed / Recipe
    if letter == "X":
        return _result(3, 0.75, "code_structure", "X = Veggie/carb recipes (mixed)")
    if letter == "Y":
        return _result(3, 0.75, "code_structure", "Y = Meat/fish recipes (mixed)")

    return _result(None, 0.0, "code_structure", f"unknown letter '{letter}'")


# ═══════════════════════════════════════════════════════════
#  Layer 2 — Description & keyword overrides
# ═══════════════════════════════════════════════════════════

# -- Priority 1: Brand detection --

_NOVA4_BRANDS = {
    "snickers", "mars", "twix", "milka", "oreo", "pringles", "red bull",
    "fanta", "pepsi", "coca cola", "coca-cola", "sprite", "haribo", "maggi",
    "duplo", "hanuta", "knoppers", "ferrero", "kinder", "corny", "maoam",
    "ritter sport", "chio", "vitalis", "wagner", "schweppes",
    "magnum", "froop", "knorr", "barilla", "dr. oetker", "iglo", "mccain",
    "miracoli", "müller", "nestlé", "nestle", "tuc", "leibniz", "bahlsen",
    "lorenz", "funny frisch", "capri sun", "granini", "hohes c", "nutella",
    "biscoff", "yogurette", "bionade", "riesen", "buko", "bresso", "meggle",
}

_CLEAN_BRANDS = {
    "gruyère", "gruyere", "leerdammer", "philadelphia", "alpro", "alnatura",
    "landliebe", "kölln", "kolln", "rügenwalder", "rugenwalder",
    "bertolli",
}

# -- Priority 2: NOVA 1 keywords --

_NOVA1_WHOLE_WORDS = {
    "roh", "natur",
    "haferflocken", "oatmeal", "haferkleie", "porridge", "buchweizen",
    "dinkelflocken", "weizenkeime", "bulgur",
    "hühnerei", "vollei",
    "banane", "birne",
    "tomate", "karotte", "möhre", "brokkoli", "spinat",
    "wasser", "mineralwasser",
    "magerquark",
    "naturjoghurt", "joghurt natur", "skyr natur",
    "kakaonibs",
    "ingwer",
}

# Plain dairy → NOVA 1 (M-code items that are minimally processed)
_NOVA1_DAIRY = {
    "joghurt", "skyr", "kefir", "sahne", "schmand", "buttermilch",
    "ayran", "creme fraiche", "saure sahne",
    "griechischer joghurt",
}
# Dairy blocked compounds (flavored/processed dairy stays NOVA 3+)
_NOVA1_DAIRY_BLOCKED = {
    "fruchtjoghurt", "sahnejoghurt", "trinkjoghurt", "joghurtdressing",
    "sahnesoße", "sahnejoghurt",
}

# "frisch" blocked in compound words
_NOVA1_FRISCH_BLOCKED = {
    "frischkäse", "kräuterfrischkäse", "frischkäsezubereitung",
}

# Short words that need word-boundary checking to avoid false matches
_NOVA1_SHORT_WORDS = {"ei", "eier"}

# "milch" needs compound-word protection
_NOVA1_MILCH_BLOCKED = {
    "milchschnitte", "milchshake", "milchreis", "milchbrötchen",
    "milchspeiseeis", "kondensmilch",
    "hafermilch", "mandelmilch", "sojamilch", "kokosmilch", "reismilch",
    "milchkaffee", "kaffee mit milch", "kaffee mit hafermilch",
    "kaffee mit mandelmilch", "kaffee mit schuss milch",
    "alpro hafermilch",
}

# "reis" needs compound-word protection
_NOVA1_REIS_BLOCKED = {
    "milchreis", "reiscracker", "reiswaffel", "reismilch",
}

# "quark" needs compound-word protection
_NOVA1_QUARK_BLOCKED = {
    "fruchtquark", "kräuterquark",
}

# "apfel" blocked in beverages/processed
_NOVA1_APFEL_BLOCKED = {
    "apfelsaft", "apfelschorle", "apfelsaftschorle", "apfelmus",
    "apfelmark", "apfelkuchen", "apfelessig",
}

# "gurke" blocked in pickled
_NOVA1_GURKE_BLOCKED = {
    "gewürzgurke", "gewürzgurken", "essiggurken", "essiggurke",
}

# "salat" as lettuce (NOVA 1) vs prepared salad
_NOVA1_SALAT_BLOCKED = {
    "nudelsalat", "kartoffelsalat", "wurstsalat", "krautsalat",
    "fleischsalat", "heringssalat", "eiersalat", "thunfischsalat",
    "geflügelsalat", "waldorfsalat", "quinoasalat", "bohnensalat",
    "obstsalat",
}

# "linsen" blocked in processed products
_NOVA1_LINSEN_BLOCKED = {
    "linsenwaffeln", "linsenwaffel", "linsenaufstrich",
}

# "erbsen" blocked in processed products
_NOVA1_ERBSEN_BLOCKED = {
    "erbsenproteinpulver", "erbsenpulver",
}

# "traube" blocked in juice
_NOVA1_TRAUBE_BLOCKED = {"traubensaft"}

_NOVA1_LETTERS = {"F", "G", "V", "U", "T", "E", "H", "K", "N", "M", "C"}

# -- Priority 3: NOVA 2 keywords --

_NOVA2_OILS = {
    "olivenöl", "rapsöl", "sonnenblumenöl", "leinöl", "kokosöl",
    "sesamöl", "walnussöl", "kürbiskernöl", "erdnussöl", "maiskeimöl",
}

_NOVA2_FLOUR = {
    "weizenmehl", "roggenmehl", "dinkelmehl", "maismehl",
    "kartoffelmehl", "mandelmehl",
}
# "mehl" blocked in compound words (bread described by flour)
_NOVA2_MEHL_BLOCKED = {
    "vollkornbrot", "brötchen", "dinkelvollkornmehl", "weißmehl brötchen",
}

_NOVA2_OTHER = {
    "ahornsirup", "agavendicksaft",
    "speisestärke", "gelatine", "backpulver",
    "balsamico",
}

# "hefe" blocked — "hefe" in ingredient lists shouldn't trigger NOVA 2
_NOVA2_HEFE_BLOCKED = {"maultasche", "hefeteig"}

# "honig" blocked in compounds
_NOVA2_HONIG_BLOCKED = {"honigmelone", "honigkuchen", "honig-senf"}

# "essig" blocked in salad dressings
_NOVA2_ESSIG_BLOCKED = {
    "essig-öl", "essig/öl", "essigmarinade", "essig-öldressing",
    "essig-öl-dressing", "essiggurke", "essiggurken",
}

# "butter" needs compound protection
_NOVA2_BUTTER_BLOCKED = {
    "butterkeks", "erdnussbutter", "buttermilch", "buttergebäck",
    "buttercreme", "buttercremetorte", "erdnussbutter",
    "butterkäse", "kräuterbutter", "butterhörnchen", "buttercroissant",
}

# "ei"/"eier" needs compound protection — these contain "ei" but aren't plain eggs
_NOVA1_EI_BLOCKED = {
    "milchreis", "eiscreme", "eis ", "speiseeis", "eistee",
    "schweinefleisch", "reis", "heißgetränk",
}

# "zucker" needs compound protection
_NOVA2_ZUCKER_BLOCKED = {
    "zuckerwatte", "zuckerstreusel", "zuckerguss",
    "zuckerfrei", "weniger zucker", "ohne zucker",
}

# "salz" needs compound protection
_NOVA2_SALZ_BLOCKED = {
    "salzstangen", "salzgebäck", "salzbrezeln", "salzcracker",
    "salzkartoffel", "salzkartoffeln",
}

# NOVA 2 spices (R-code items that should be NOVA 2, not 3)
_NOVA2_SPICES = {
    "zimt", "kurkuma", "gewürze", "paprikapulver",
    "currypulver", "oregano", "basilikum getrocknet",
}
# "pfeffer" needs compound blocker
_NOVA2_PFEFFER_BLOCKED = {
    "pfefferminztee", "pfefferminz", "pfefferbeisser", "pfefferbeißer",
}

# -- Priority 4: NOVA 4 keywords --

_NOVA4_ULTRA_PROCESSED = {
    "energy drink", "energydrink", "chips", "fertiggericht",
    "instant", "mikrowelle", "dosensuppe", "tütensuppe",
    "cornflakes", "gummibärchen", "gummibär", "weingummi",
    "schokoriegel", "müsliriegel", "proteinriegel", "energieriegel",
    "eiweißpulver", "proteinpulver", "whey protein",
    # Schorle (Fruchtsaft-Schorle) stays NOVA 4; pure juices + smoothies moved
    # to NOVA 3 per Sidney 2026-04-23 ("juice should be NOVA 3").
    "schorle", "johannisbeerschorle",
    # Nectars and mixed juice drinks per Sidney 2026-04-23 ("Fruchtsaftgetränke
    # → NOVA 4; Nektar including Apfelnektar → NOVA 4").
    "fruchtsaftgetränk", "fruchtsaftgetränke",
    "nektar", "fruchtnektar", "apfelnektar", "orangennektar", "mehrfruchtnektar",
    "hafermilch", "haferdrink", "mandelmilch", "sojamilch", "sojadrink",
    "alpro", "kaffeeweißer",
    "cappuccino", "cappucchino", "milchkaffee", "eiskaffee",
    "latte", "matcha latte",
}

# "tiefkühl" blocked when followed by plain food
_NOVA4_TIEFKUEHL_BLOCKED = {
    "tiefkühlbeeren", "tiefkühlgemüse", "tiefkühlobst",
    "tk himbeeren", "tk beeren",
}

# "schorle" blocked for wine spritzers (NOVA 3, not 4)
_NOVA4_SCHORLE_BLOCKED = {"weißweinschorle", "rotweinschorle", "weinschorle"}

_NOVA4_PROCESSED_MEAT = {
    "wurst", "bratwurst", "bockwurst", "currywurst", "wiener",
    "würstchen", "salami", "mortadella", "leberkäse", "leberwurst",
    "fleischwurst", "mettwurst", "nuggets", "chicken nuggets",
    "fischstäbchen",
}

_NOVA4_SWEETS = {
    "bonbon", "lakritze", "lakritz", "keks", "kekse", "cookie",
    "croissant", "schokocreme", "nuss-nougat-creme",
    "nutella", "konfitüre", "erdnussbutter",
    "kuchen", "torte",
}

# "kuchen" blocked for pfannkuchen/flammkuchen (NOVA 3, not 4)
_NOVA4_KUCHEN_BLOCKED = {"pfannkuchen", "eierpfannkuchen", "flammkuchen"}

# "waffel" blocked for maiswaffeln/linsenwaffeln
_NOVA4_WAFFEL_BLOCKED = {"maiswaffeln", "maiswaffel", "linsenwaffeln", "linsenwaffel", "reiswaffel"}

_NOVA4_DRINKS = {
    "cola", "fanta", "sprite", "limonade", "eistee", "ice tea",
    "spezi", "alkopop", "sirup",
}

# "cola" blocked in rucola
_NOVA4_COLA_BLOCKED = {"rucola"}

_NOVA4_DAIRY = {
    "milchschnitte", "milchshake", "fruchtjoghurt", "fruchtquark",
    "pudding", "sahnejoghurt", "trinkjoghurt", "milchreis",
}

_NOVA4_FROZEN = {
    "pizza", "pommes", "kroketten", "rösti",
    "backfisch", "schlemmerfilet", "cordon bleu",
}

_NOVA4_FAST_FOOD = {
    "döner", "kebab", "gyros", "big mac", "hamburger", "cheeseburger",
    "hot dog", "hotdog", "burrito", "wrap", "frühlingsrolle",
}

_NOVA4_CEREAL = {
    "schokomüsli",
}

_NOVA4_SOY_PLANT = {
    "tofu", "sojasoße", "sojasauce", "erdnussmus", "erdnusssoße",
    "flohsamenschalen", "flohsamenschalenpulver",
    "linsenwaffeln", "linsenwaffel", "linsenaufstrich",
    "mandelmus", "röstzwiebeln",
    "ajvar", "avocadocreme", "guacamole",
    "hummus", "falafel",
    "kondensmilch",
}

_NOVA4_MISC = {
    "bratensoße", "vanilleeis", "apfelmus", "apfelmark",
    "eiweißpulver", "proteinpulver", "whey protein",
    "erbsenproteinpulver", "erbsenpulver",
    "lavita", "multivitaminsaft", "karottensaft",
    "bierschinken", "speck",
    "mangolassi",
    "wasser mit sirup",
    "bratwürste",
    "ramen",
    "honig-senf-dressing",
    "mayonnaise", "remoulade",
}

# -- Priority 5: NOVA 3 keywords --

_NOVA3_KEYWORDS = {
    "käse", "frischkäse",
    "brot", "brötchen", "brezel", "laugengebäck",
    "toast", "marmelade", "ketchup",
    "bier", "sekt", "schnaps", "likör", "obstbrand",
    "whisky", "vodka", "radler",
    "konserve", "dose",
    "räucherlachs", "räucherfisch", "stremellachs", "geräuchert",
    "schinken", "kochschinken", "lachsschinken",
    "gewürzgurke", "gewürzgurken", "cornichons", "essiggurken",
    "sauerkraut", "kimchi", "oliven",
    "gnocchi", "maiswaffeln",
    "butterkäse", "kräuterbutter",
    "passierte tomaten", "stückige tomaten", "gestückelte tomaten",
    "meerrettich", "kloß", "klöße",
    "pfannkuchen", "flammkuchen",
    "thunfisch im eigenen saft",
    "vollkornwrap", "weizenwrap", "wrap",
    "weißweinschorle",
    # Pure fruit juices + smoothies (Sidney 2026-04-23: juice → NOVA 3)
    "apfelsaft", "orangensaft", "traubensaft", "cranberrysaft",
    "fruchtsaft", "mehrfruchtsaft", "multivitaminsaft",
    "smoothie",
}

# "wein" blocked to avoid matching inside "schwein" words
_NOVA3_WEIN_BLOCKED = {
    "schweinebraten", "schweinefilet", "schweinefleisch",
    "schweineschnitzel", "schweinesteak", "schwein",
}

# "gin" blocked to avoid matching inside "aubergine"
_NOVA3_GIN_BLOCKED = {"aubergine"}

# "rum" blocked to avoid matching inside "crumble"
_NOVA3_RUM_BLOCKED = {"crumble", "apple crumble"}


def _layer2_description_override(layer1: dict, bls_code: str,
                                 food_description: str,
                                 brand: str | None) -> dict:
    """Apply description-based overrides on top of Layer 1 result."""
    if not food_description:
        return layer1

    lower = food_description.lower().strip()
    letter = bls_code[0].upper() if bls_code else ""

    # ── Priority 1: Brand detection ──
    if brand is not None:
        brand_lower = brand.lower()
        if brand_lower in _NOVA4_BRANDS:
            return _result(4, 0.95, "description_override",
                           f"brand '{brand}' → NOVA 4")
        if brand_lower not in _CLEAN_BRANDS:
            # Unknown brand — don't override, keep Layer 1
            pass

    # ── Priority 1b: NOVA 1 overrides (any code letter) ──
    _NOVA1_ANY_LETTER = {"kakaonibs", "ingwer", "bulgur", "champignons"}
    for kw in _NOVA1_ANY_LETTER:
        if kw in lower:
            return _result(1, 0.90, "description_override",
                           f"keyword '{kw}' → NOVA 1 (any code)")

    # "bohnen" → NOVA 1 (any code), blocked for recipe compounds
    _NOVA1_BOHNEN_BLOCKED = {
        "bohnensalat", "bohneneintopf", "bohnensuppe", "bohnengemüse",
        "bohnenpfanne", "bohnenragout",
    }
    if "bohnen" in lower or "bohne" in lower:
        if not any(blocked in lower for blocked in _NOVA1_BOHNEN_BLOCKED):
            return _result(1, 0.90, "description_override",
                           "keyword 'bohnen' → NOVA 1 (any code)")

    # "gemüse" → NOVA 1 (any code), blocked for dish compounds
    _NOVA1_GEMUESE_BLOCKED = {
        "gemüsesuppe", "gemüsecurry", "gemüsepfanne", "gemüseauflauf",
        "gemüseeintopf", "gemüselasagne", "gemüseragout", "gemüsegratin",
        "gemüseschnitzel", "gemüsebrühe", "gemüsebouillon",
    }
    if "gemüse" in lower:
        if not any(blocked in lower for blocked in _NOVA1_GEMUESE_BLOCKED):
            return _result(1, 0.90, "description_override",
                           "keyword 'gemüse' → NOVA 1 (any code)")

    # ── Priority 1c: NOVA 4 absolutes (win over NOVA 1 fruit keywords) ──
    # Without this, "Apfelnektar" hits the "apfel" NOVA 1 rule before reaching
    # the "nektar" NOVA 4 rule. These tokens are industrial products regardless
    # of the fruit in the name (Sidney 2026-04-23).
    _NOVA4_ABSOLUTE = (
        "nektar", "fruchtsaftgetränk",
    )
    for kw in _NOVA4_ABSOLUTE:
        if kw in lower:
            return _result(4, 0.92, "description_override",
                           f"keyword '{kw}' → NOVA 4 (absolute)")

    # ── Priority 2: NOVA 1 keywords ──
    if letter in _NOVA1_LETTERS:
        for kw in _NOVA1_WHOLE_WORDS:
            if kw in lower:
                return _result(1, 0.90, "description_override",
                               f"keyword '{kw}' → NOVA 1")

        # Plain dairy → NOVA 1
        for kw in _NOVA1_DAIRY:
            if kw in lower:
                if not any(blocked in lower for blocked in _NOVA1_DAIRY_BLOCKED):
                    return _result(1, 0.90, "description_override",
                                   f"keyword '{kw}' → NOVA 1 (plain dairy)")

        # "frisch" with compound protection
        if "frisch" in lower:
            if not any(blocked in lower for blocked in _NOVA1_FRISCH_BLOCKED):
                return _result(1, 0.90, "description_override",
                               "keyword 'frisch' → NOVA 1")

        import re
        for kw in _NOVA1_SHORT_WORDS:
            if re.search(r'\b' + re.escape(kw) + r'\b', lower):
                if not any(blocked in lower for blocked in _NOVA1_EI_BLOCKED):
                    return _result(1, 0.90, "description_override",
                                   f"keyword '{kw}' (word boundary) → NOVA 1")

        if "milch" in lower:
            if not any(blocked in lower for blocked in _NOVA1_MILCH_BLOCKED):
                return _result(1, 0.90, "description_override",
                               "keyword 'milch' (plain) → NOVA 1")

        if "apfel" in lower:
            if not any(blocked in lower for blocked in _NOVA1_APFEL_BLOCKED):
                return _result(1, 0.90, "description_override",
                               "keyword 'apfel' → NOVA 1")

        if "gurke" in lower:
            if not any(blocked in lower for blocked in _NOVA1_GURKE_BLOCKED):
                return _result(1, 0.90, "description_override",
                               "keyword 'gurke' → NOVA 1")

        if "reis" in lower:
            if not any(blocked in lower for blocked in _NOVA1_REIS_BLOCKED):
                return _result(1, 0.90, "description_override",
                               "keyword 'reis' (plain) → NOVA 1")

        if "quark" in lower:
            if not any(blocked in lower for blocked in _NOVA1_QUARK_BLOCKED):
                return _result(1, 0.90, "description_override",
                               "keyword 'quark' (plain) → NOVA 1")

        if "linsen" in lower:
            if not any(blocked in lower for blocked in _NOVA1_LINSEN_BLOCKED):
                return _result(1, 0.90, "description_override",
                               "keyword 'linsen' → NOVA 1")

        if "erbsen" in lower:
            if not any(blocked in lower for blocked in _NOVA1_ERBSEN_BLOCKED):
                return _result(1, 0.90, "description_override",
                               "keyword 'erbsen' → NOVA 1")

        if "traube" in lower:
            if not any(blocked in lower for blocked in _NOVA1_TRAUBE_BLOCKED):
                return _result(1, 0.90, "description_override",
                               "keyword 'traube' → NOVA 1")

        if "salat" in lower:
            if not any(blocked in lower for blocked in _NOVA1_SALAT_BLOCKED):
                return _result(1, 0.90, "description_override",
                               "keyword 'salat' (lettuce) → NOVA 1")

    # ── Priority 3: NOVA 2 keywords ──
    for kw in _NOVA2_OILS:
        if kw in lower:
            return _result(2, 0.90, "description_override",
                           f"keyword '{kw}' → NOVA 2 (oil)")

    if "butter" in lower:
        if not any(blocked in lower for blocked in _NOVA2_BUTTER_BLOCKED):
            return _result(2, 0.90, "description_override",
                           "keyword 'butter' (plain) → NOVA 2")

    if "zucker" in lower:
        if not any(blocked in lower for blocked in _NOVA2_ZUCKER_BLOCKED):
            return _result(2, 0.90, "description_override",
                           "keyword 'zucker' → NOVA 2 (ingredient)")

    if "rohrzucker" in lower or "puderzucker" in lower:
        return _result(2, 0.90, "description_override",
                       "keyword sugar variant → NOVA 2")

    if "salz" in lower:
        if not any(blocked in lower for blocked in _NOVA2_SALZ_BLOCKED):
            return _result(2, 0.90, "description_override",
                           "keyword 'salz' (plain) → NOVA 2")

    if "honig" in lower:
        if not any(blocked in lower for blocked in _NOVA2_HONIG_BLOCKED):
            return _result(2, 0.90, "description_override",
                           "keyword 'honig' → NOVA 2")

    if "essig" in lower:
        if not any(blocked in lower for blocked in _NOVA2_ESSIG_BLOCKED):
            return _result(2, 0.90, "description_override",
                           "keyword 'essig' → NOVA 2")

    for kw in _NOVA2_FLOUR:
        if kw in lower:
            return _result(2, 0.90, "description_override",
                           f"keyword '{kw}' → NOVA 2 (flour)")

    if "mehl" in lower:
        if not any(blocked in lower for blocked in _NOVA2_MEHL_BLOCKED):
            return _result(2, 0.90, "description_override",
                           "keyword 'mehl' → NOVA 2 (flour)")

    for kw in _NOVA2_SPICES:
        if kw in lower:
            return _result(2, 0.90, "description_override",
                           f"keyword '{kw}' → NOVA 2 (spice)")

    if "pfeffer" in lower:
        if not any(blocked in lower for blocked in _NOVA2_PFEFFER_BLOCKED):
            return _result(2, 0.90, "description_override",
                           "keyword 'pfeffer' → NOVA 2 (spice)")

    for kw in _NOVA2_OTHER:
        if kw in lower:
            return _result(2, 0.90, "description_override",
                           f"keyword '{kw}' → NOVA 2")

    # ── Priority 4: NOVA 4 keywords ──
    for kw in _NOVA4_ULTRA_PROCESSED:
        if kw in lower:
            if ("tiefkühl" in kw and any(blocked in lower for blocked in _NOVA4_TIEFKUEHL_BLOCKED)):
                continue
            if ("schorle" in kw and any(blocked in lower for blocked in _NOVA4_SCHORLE_BLOCKED)):
                continue
            return _result(4, 0.85, "description_override",
                           f"keyword '{kw}' → NOVA 4 (ultra-processed)")

    for kw in _NOVA4_PROCESSED_MEAT:
        if kw in lower:
            return _result(4, 0.85, "description_override",
                           f"keyword '{kw}' → NOVA 4 (processed meat)")

    for kw in _NOVA4_SWEETS:
        if kw in lower:
            if "kuchen" in kw and any(blocked in lower for blocked in _NOVA4_KUCHEN_BLOCKED):
                continue
            return _result(4, 0.85, "description_override",
                           f"keyword '{kw}' → NOVA 4 (sweets)")

    # "waffel" with blocker for maiswaffeln etc
    if "waffel" in lower:
        if not any(blocked in lower for blocked in _NOVA4_WAFFEL_BLOCKED):
            return _result(4, 0.85, "description_override",
                           "keyword 'waffel' → NOVA 4 (sweets)")

    for kw in _NOVA4_DRINKS:
        if kw in lower:
            if not any(blocked in lower for blocked in _NOVA4_COLA_BLOCKED) or "cola" not in kw:
                return _result(4, 0.85, "description_override",
                               f"keyword '{kw}' → NOVA 4 (drink)")

    for kw in _NOVA4_DAIRY:
        if kw in lower:
            return _result(4, 0.85, "description_override",
                           f"keyword '{kw}' → NOVA 4 (dairy ultra-processed)")

    for kw in _NOVA4_FROZEN:
        if kw in lower:
            return _result(4, 0.85, "description_override",
                           f"keyword '{kw}' → NOVA 4 (frozen/convenience)")

    for kw in _NOVA4_FAST_FOOD:
        if kw in lower:
            return _result(4, 0.85, "description_override",
                           f"keyword '{kw}' → NOVA 4 (fast food)")

    for kw in _NOVA4_CEREAL:
        if kw in lower:
            return _result(4, 0.85, "description_override",
                           f"keyword '{kw}' → NOVA 4 (cereal)")

    for kw in _NOVA4_SOY_PLANT:
        if kw in lower:
            return _result(4, 0.85, "description_override",
                           f"keyword '{kw}' → NOVA 4 (processed plant)")

    for kw in _NOVA4_MISC:
        if kw in lower:
            return _result(4, 0.85, "description_override",
                           f"keyword '{kw}' → NOVA 4")

    # ── Priority 5: NOVA 3 keywords ──
    for kw in _NOVA3_KEYWORDS:
        if kw in lower:
            if kw == "gin" and any(blocked in lower for blocked in _NOVA3_GIN_BLOCKED):
                continue
            if kw == "wein" and any(blocked in lower for blocked in _NOVA3_WEIN_BLOCKED):
                continue
            if kw == "rum" and any(blocked in lower for blocked in _NOVA3_RUM_BLOCKED):
                continue
            return _result(3, 0.80, "description_override",
                           f"keyword '{kw}' → NOVA 3")

    # No override — return Layer 1 unchanged
    return layer1


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════

def _result(nova: int | None, confidence: float, method: str,
            reason: str) -> dict:
    return {
        "nova": nova,
        "confidence": confidence,
        "method": method,
        "reason": reason,
    }
