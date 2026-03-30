"""
Normalizer Module
=================
Cleans free-text food descriptions from study participants and extracts
structured metadata (preparation state, fat%, brand, etc.) to improve
downstream embedding search and LLM re-ranking.

Usage:
    from modules.normalizer import normalize
    result = normalize("Edamer (40% Fett)")
    print(result.cleaned)       # "edamer"
    print(result.fat_percent)   # "40"
    print(result.prep_state)    # None
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class NormalizedQuery:
    """Result of normalizing a food description."""
    original: str                          # raw input
    cleaned: str = ""                      # cleaned text for embedding search
    prep_state: str | None = None          # extracted: gekocht, roh, gebraten, …
    fat_percent: str | None = None         # extracted: "40", "3.8", "1.5", …
    brand: str | None = None               # detected brand name
    is_english: bool = False               # input appears to be English
    is_multi_ingredient: bool = False      # contains multiple foods (comma/und)
    components: list[str] = field(default_factory=list)  # split multi-ingredient items
    search_variants: list[str] = field(default_factory=list)  # extra search strings


# =====================================================================
#  Dictionaries
# =====================================================================

# German synonyms / colloquial → BLS nomenclature
SYNONYM_MAP: dict[str, str] = {
    # Eggs
    "eier":         "hühnerei vollei",
    "ei":           "hühnerei vollei",
    "spiegelei":    "hühnerei vollei gebraten",
    "rührei":       "hühnerei vollei rührei",
    "spiegeleier":  "hühnerei vollei gebraten",
    "rühreier":     "hühnerei vollei rührei",
    "omelette":     "hühnerei vollei omelett",
    "omlett":       "hühnerei vollei omelett",

    # Pasta
    "nudeln":       "teigwaren eifrei",
    "pasta":        "teigwaren eifrei",
    "spaghetti":    "teigwaren eifrei spaghetti",
    "penne":        "teigwaren eifrei penne",
    "fusilli":      "teigwaren eifrei fusilli",
    "tagliatelle":  "teigwaren eifrei tagliatelle",
    "makkaroni":    "teigwaren eifrei makkaroni",
    "vollkornnudeln": "teigwaren vollkorn",
    "vollkornpasta":  "teigwaren vollkorn",

    # Potatoes
    "pommes":       "pommes frites",
    "pommes frites": "pommes frites",
    "bratkartoffeln": "kartoffel gebraten",
    "kartoffelpüree": "kartoffelbrei",
    "kartoffelpuree": "kartoffelbrei",
    "stampfkartoffeln": "kartoffelbrei",

    # Meat
    "pute":         "pute brust",
    "putenbrust":   "pute brust",
    "hähnchen":     "hähnchenfleisch",
    "hühnchen":     "hähnchenfleisch",
    "chicken":      "hähnchenfleisch",
    "hackfleisch":  "hackfleisch gemischt",
    "hack":         "hackfleisch gemischt",
    "schnitzel":    "schwein schnitzel",
    "steak":        "rind steak",
    "bratwurst":    "bratwurst schwein",
    "wienerle":     "wiener würstchen",
    "wienerli":     "wiener würstchen",
    "bockwurst":    "bockwurst",
    "leberkäse":    "fleischkäse",
    "leberkäs":     "fleischkäse",

    # Fish
    "lachs":        "lachs atlantik",
    "thunfisch":    "thunfisch",
    "garnelen":     "garnele",
    "shrimps":      "garnele",

    # Dairy
    "joghurt":      "joghurt",
    "jogurt":       "joghurt",
    "yoghurt":      "joghurt",
    "quark":        "speisequark",
    "frischkäse":   "frischkäse",
    "sahne":        "sahne schlagsahne",
    "schmand":      "schmand saure sahne",
    "milch":        "kuhmilch vollmilch",
    "buttermilch":  "buttermilch",
    "skyr":         "skyr",

    # Bread
    "brötchen":     "weizenbrötchen",
    "semmel":       "weizenbrötchen",
    "schrippe":     "weizenbrötchen",
    "toast":        "toastbrot weizen",
    "baguette":     "baguette weizen",
    "croissant":    "croissant",
    "knäckebrot":   "knäckebrot",

    # Vegetables
    "poree":        "porree lauch",
    "lauch":        "porree lauch",
    "salat":        "kopfsalat",
    "eisbergsalat": "eissalat",
    "rucola":       "rucola rauke",
    "mais":         "mais zuckermais",
    "champignons":  "champignon zuchtpilz",
    "pilze":        "champignon zuchtpilz",
    "zucchini":     "zucchini",
    "aubergine":    "aubergine",
    "paprika":      "paprikaschote",
    "tomate":       "tomate rot",
    "tomaten":      "tomate rot",
    "cherry tomaten": "tomate rot",
    "cherrytomaten":  "tomate rot",
    "gurke":        "salatgurke",
    "karotte":      "möhre karotte",
    "karotten":     "möhre karotte",
    "möhre":        "möhre karotte",
    "möhren":       "möhre karotte",
    "brokkoli":     "broccoli brokkoli",
    "broccoli":     "broccoli brokkoli",
    "blumenkohl":   "blumenkohl",
    "spinat":       "spinat",
    "zwiebel":      "zwiebel",
    "zwiebeln":     "zwiebel",
    "knoblauch":    "knoblauch",
    "kürbis":       "kürbis",
    "süßkartoffel": "süßkartoffel batate",
    "süsskartoffel": "süßkartoffel batate",
    "avocado":      "avocado",
    "edamame":      "sojabohne grün",

    # Fruits
    "apfel":        "apfel",
    "banane":       "banane",
    "erdbeeren":    "erdbeere",
    "erdbeere":     "erdbeere",
    "himbeeren":    "himbeere",
    "blaubeeren":   "heidelbeere blaubeere",
    "heidelbeeren": "heidelbeere blaubeere",
    "weintrauben":  "weintraube",
    "trauben":      "weintraube",
    "birne":        "birne",
    "orange":       "apfelsine orange",
    "mandarine":    "mandarine clementine",
    "clementine":   "mandarine clementine",
    "kiwi":         "kiwi",
    "mango":        "mango",
    "ananas":       "ananas",
    "wassermelone": "wassermelone",
    "melone":       "melone",

    # Beverages
    "cola":         "colagetränk",
    "coca cola":    "colagetränk",
    "fanta":        "limonade orange",
    "sprite":       "limonade zitrone",
    "spezi":        "colagetränk mischgetränk",
    "apfelschorle": "apfelsaftschorle",
    "schorle":      "saftschorle",
    "saft":         "fruchtsaft",
    "orangensaft":  "orangensaft",
    "apfelsaft":    "apfelsaft",
    "kaffee":       "bohnenkaffee",
    "espresso":     "bohnenkaffee espresso",
    "cappuccino":   "bohnenkaffee cappuccino",
    "latte macchiato": "bohnenkaffee latte macchiato",
    "milchkaffee":  "bohnenkaffee milchkaffee",
    "tee":          "tee aufguss",
    "wasser":       "trinkwasser",

    # Fats / oils / spreads
    "olivenöl":     "olivenöl",
    "butter":       "butter",
    "margarine":    "margarine",
    "rapsöl":       "rapsöl",
    "sonnenblumenöl": "sonnenblumenöl",
    "kokosöl":      "kokosöl kokosfett",
    "leinöl":       "leinöl",

    # Sweets / snacks
    "schokolade":   "schokolade vollmilch",
    "gummibärchen": "gummibonbon fruchtgummi",
    "haribo":       "gummibonbon fruchtgummi",
    "erdnussflips": "erdnussflips",
    "erdnussfips":  "erdnussflips",
    "chips":        "kartoffelchips",
    "salzstangen":  "salzstangen",
    "kekse":        "keks butterkeks",
    "keks":         "keks butterkeks",
    "nutella":      "nuss-nougat-creme",
    "marmelade":    "konfitüre marmelade",

    # Grains / cereals
    "haferflocken": "haferflocken",
    "müsli":        "müsli",
    "reis":         "reis",
    "basmatireis":  "reis basmati",
    "couscous":     "couscous",
    "bulgur":       "bulgur",
    "quinoa":       "quinoa",
    "hirse":        "hirse",

    # Condiments
    "ketchup":      "ketchup tomatenketchup",
    "senf":         "senf",
    "mayonnaise":   "mayonnaise",
    "mayo":         "mayonnaise",
    "essig":        "essig",
    "sojasoße":     "sojasoße",
    "sojasauce":    "sojasoße",

    # Nuts / seeds
    "mandeln":      "mandel süß",
    "walnüsse":     "walnuss",
    "cashews":      "cashewnuss",
    "erdnüsse":     "erdnuss",
    "erdnussbutter": "erdnussbutter erdnussmus",
    "sonnenblumenkerne": "sonnenblumenkern",
    "leinsamen":    "leinsamen",
    "chiasamen":    "chiasamen chia",
    "chia":         "chiasamen chia",
    "sesam":        "sesamsamen",

    # Legumes
    "linsen":       "linse",
    "kidneybohnen": "kidneybohne",
    "kichererbsen": "kichererbse",
    "bohnen":       "bohne",
    "erbsen":       "erbse",
    "tofu":         "tofu sojabohne",
    "hummus":       "hummus kichererbse",

    # Misc
    "honig":        "honig",
    "zucker":       "zucker haushaltszucker",
    "salz":         "speisesalz kochsalz",
    "mehl":         "weizenmehl",
    "paniermehl":   "paniermehl semmelbrösel",
    "maggi":        "grundsoße braun trockenprodukt",
}

# English → German food terms (for the ~5% English descriptions)
ENGLISH_TO_GERMAN: dict[str, str] = {
    "chicken":       "hähnchenfleisch",
    "chicken breast": "hähnchen brust",
    "chicken salad": "hähnchensalat",
    "beef":          "rindfleisch",
    "pork":          "schweinefleisch",
    "lamb":          "lammfleisch",
    "turkey":        "pute truthahn",
    "salmon":        "lachs atlantik",
    "tuna":          "thunfisch",
    "shrimp":        "garnele",
    "rice":          "reis",
    "bread":         "brot",
    "toast":         "toastbrot",
    "butter":        "butter",
    "cheese":        "käse",
    "cream cheese":  "frischkäse",
    "milk":          "kuhmilch vollmilch",
    "yogurt":        "joghurt",
    "egg":           "hühnerei vollei",
    "eggs":          "hühnerei vollei",
    "scrambled eggs": "rührei",
    "fried egg":     "spiegelei",
    "boiled egg":    "hühnerei vollei gekocht",
    "apple":         "apfel",
    "banana":        "banane",
    "orange":        "apfelsine orange",
    "strawberry":    "erdbeere",
    "salad":         "salat",
    "lettuce":       "kopfsalat",
    "tomato":        "tomate rot",
    "cucumber":      "salatgurke",
    "carrot":        "möhre karotte",
    "potato":        "kartoffel",
    "potatoes":      "kartoffel",
    "french fries":  "pommes frites",
    "curly fries":   "pommes frites",
    "fries":         "pommes frites",
    "noodles":       "teigwaren nudeln",
    "pasta":         "teigwaren",
    "spaghetti":     "teigwaren spaghetti",
    "oatmeal":       "haferflocken haferbrei",
    "oats":          "haferflocken",
    "smoothie":      "smoothie fruchtgetränk",
    "coffee":        "bohnenkaffee",
    "tea":           "tee aufguss",
    "water":         "trinkwasser",
    "juice":         "fruchtsaft",
    "orange juice":  "orangensaft",
    "apple juice":   "apfelsaft",
    "olive oil":     "olivenöl",
    "honey":         "honig",
    "sugar":         "zucker haushaltszucker",
    "chocolate":     "schokolade",
    "cookie":        "keks",
    "cookies":       "keks",
    "cake":          "kuchen torte",
    "ice cream":     "eiscreme speiseeis",
    "nuts":          "nüsse nussmischung",
    "peanut butter": "erdnussbutter erdnussmus",
    "avocado":       "avocado",
    "broccoli":      "broccoli brokkoli",
    "spinach":       "spinat",
    "mushroom":      "champignon pilz",
    "mushrooms":     "champignon pilz",
    "onion":         "zwiebel",
    "garlic":        "knoblauch",
    "pepper":        "paprikaschote",
    "corn":          "mais zuckermais",
    "beans":         "bohne",
    "lentils":       "linse",
    "tofu":          "tofu sojabohne",
    "ham":           "schinken",
    "sausage":       "wurst bratwurst",
    "bacon":         "speck frühstücksspeck",
}

# Known brand names → generic BLS category
BRAND_MAP: dict[str, str] = {
    "maggi":         "grundsoße braun trockenprodukt",
    "knorr":         "grundsoße trockenprodukt",
    "dr. oetker":    "backzutat",
    "dr oetker":     "backzutat",
    "iglo":          "tiefkühlgemüse",
    "frosta":        "tiefkühlgericht",
    "rama":          "margarine halbfett",
    "nutella":       "nuss-nougat-creme",
    "haribo":        "gummibonbon fruchtgummi",
    "coca cola":     "colagetränk",
    "coca-cola":     "colagetränk",
    "pepsi":         "colagetränk",
    "fanta":         "limonade orange",
    "sprite":        "limonade zitrone",
    "schweppes":     "limonade tonic",
    "red bull":      "energydrink",
    "snickers":      "schokoriegel",
    "mars":          "schokoriegel",
    "twix":          "schokoriegel",
    "milka":         "schokolade vollmilch",
    "oreo":          "keks sandwich",
    "pringles":      "kartoffelchips",
    "magnum":        "milchspeiseeis",
    "froop":         "fruchtjoghurt",
    "gruyère":       "hartkäse",
    "gruyere":       "hartkäse",
    "rewe":          "",      # store brand — ignore
    "aldi":          "",
    "lidl":          "",
    "edeka":         "",
    "netto":         "",
    "penny":         "",
    "kaufland":      "",
    "dm":            "",
    "rossmann":      "",
    "ja!":           "",
    "gut & günstig":  "",
}

# Preparation-state keywords to detect and extract
PREP_KEYWORDS: list[tuple[str, str]] = [
    ("frittiert",     "frittiert"),
    ("gebraten",      "gebraten"),
    ("angebraten",    "gebraten"),
    ("gebacken",      "gebacken"),
    ("überbacken",    "gebacken"),
    ("geröstet",      "geröstet"),
    ("getoastet",     "geröstet"),
    ("gedünstet",     "gedünstet"),
    ("gedämpft",      "gedämpft"),
    ("geschmort",     "geschmort"),
    ("gekocht",       "gekocht"),
    ("gegart",        "gegart"),
    ("roh",           "roh"),
    ("frisch",        "roh"),
    ("ungekocht",     "roh"),
    ("tiefgefroren",  "tiefgefroren"),
    ("tiefgekühlt",   "tiefgefroren"),
    ("getrocknet",    "getrocknet"),
    ("eingelegt",     "eingelegt"),
    ("mariniert",     "mariniert"),
    ("paniert",       "paniert"),
    ("gegrillt",      "gegrillt"),
    ("geräuchert",    "geräuchert"),
    ("gesalzen",      "gesalzen"),
    ("gesüßt",        "gesüßt"),
    ("ungesüßt",      "ungesüßt"),
    ("püriert",       "püriert"),
    ("passiert",      "passiert"),
    ("selbstgemacht", "selbstgemacht"),
    ("homemade",      "selbstgemacht"),
]

# Regex for fat percentage: "3,8% Fett", "40 % Fett", "(1.5%-Fett)", etc.
FAT_PATTERN = re.compile(
    r"(\d+[.,]?\d*)\s*%\s*(?:-?\s*)?(?:fett|f\.?\s*i\.?\s*tr\.?)?",
    re.IGNORECASE,
)

# Regex for parenthetical info: "(Bio)", "(40% Fett)", "(Aldi)", "(ca. 200g)"
PAREN_PATTERN = re.compile(r"\([^)]*\)")

# English detection: check if common English words appear
ENGLISH_MARKERS = {
    "with", "and", "the", "from", "fresh", "fried", "baked", "boiled",
    "grilled", "roasted", "cooked", "raw", "salad", "chicken", "beef",
    "pork", "fish", "cream", "sauce", "soup", "bread", "cake",
    "rice", "beans", "butter", "cheese", "milk", "egg", "eggs",
    "ice cream", "french fries", "curly fries", "scrambled",
}


# =====================================================================
#  Core normalization function
# =====================================================================

def normalize(text: str) -> NormalizedQuery:
    """
    Normalize a free-text food description for BLS matching.

    Steps:
        1. Extract fat percentage (before lowercasing)
        2. Detect if English
        3. Lowercase
        4. Extract preparation state
        5. Detect brand names
        6. Strip parentheticals
        7. Expand synonyms (German or English→German)
        8. Detect multi-ingredient entries
        9. Clean whitespace
        10. Generate search variants

    Returns a NormalizedQuery with cleaned text + all extracted metadata.
    """
    result = NormalizedQuery(original=text)
    working = text.strip()
    if not working:
        result.cleaned = ""
        return result

    # 1. Extract fat percentage BEFORE lowercasing (preserve original)
    #    But skip if this is an alcoholic/beverage context (% = alcohol, not fat)
    fat_match = FAT_PATTERN.search(working)
    if fat_match and not _is_beverage_context(working):
        result.fat_percent = fat_match.group(1).replace(",", ".")

    # 2. Lowercase
    working = working.lower().strip()

    # 3. Detect English
    words = set(working.split())
    english_hits = words & ENGLISH_MARKERS
    if len(english_hits) >= 1 and not _has_german_chars(working):
        result.is_english = True

    # 4. Extract preparation state
    for keyword, canonical in PREP_KEYWORDS:
        if keyword in working:
            result.prep_state = canonical
            break  # take the first (most specific) match

    # 5. Detect brand names
    for brand, generic in BRAND_MAP.items():
        if brand in working:
            result.brand = brand
            # Don't replace yet — we want the brand context for search variants
            break

    # 6. Strip parentheticals (but we already extracted fat% and prep from them)
    working = PAREN_PATTERN.sub(" ", working)

    # 7. Strip common noise words/patterns
    working = re.sub(r"\d+[.,]?\d*\s*%\s*(?:-?\s*)?(?:fett|f\.?\s*i\.?\s*tr\.?)?", " ", working)
    working = re.sub(r"\d+[.,]?\d*\s*(?:g|mg|kg|ml|l|cl|el|tl|stück|scheibe|scheiben|portion|portionen)\b", " ", working)
    working = re.sub(r"\b(?:ca|circa|etwa|ungefähr)\b\.?", " ", working)
    working = re.sub(r"\b(?:bio|light|diet|zero|ohne zucker|zuckerfrei)\b", " ", working)

    # Remove store brand names (they add no food information)
    for brand, generic in BRAND_MAP.items():
        if generic == "" and brand in working:
            working = working.replace(brand, " ")

    # 8. Synonym expansion (longest match first)
    cleaned = working.strip()

    # Try English→German first if detected as English
    if result.is_english:
        cleaned = _apply_dict(cleaned, ENGLISH_TO_GERMAN)
    # Then German synonyms
    cleaned = _apply_dict(cleaned, SYNONYM_MAP)

    # 9. Detect multi-ingredient
    separators = re.compile(r"\s*[,;]\s*|\s+und\s+|\s+mit\s+|\s+&\s+")
    parts = separators.split(cleaned)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        result.is_multi_ingredient = True
        result.components = parts

    # 10. Final cleanup
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    result.cleaned = cleaned

    # 11. Generate search variants
    result.search_variants = _build_variants(result)

    return result


# =====================================================================
#  Helper functions
# =====================================================================

_BEVERAGE_KEYWORDS = {
    "bier", "wein", "sekt", "prosecco", "wodka", "vodka", "gin", "rum",
    "likör", "radler", "glühwein", "aperol", "hugo", "whisky", "cognac",
    "schnaps", "grappa", "absinth", "champagner", "cider", "met",
    "weißwein", "rotwein", "rosé", "rose", "weinschorle",
}


def _is_beverage_context(text: str) -> bool:
    """Return True if the text describes an alcoholic beverage (% = ABV, not fat)."""
    lower = text.lower()
    return any(kw in lower for kw in _BEVERAGE_KEYWORDS)


# Common German food word stems for compound splitting/joining
_COMPOUND_PARTS = [
    "kürbis", "tomaten", "tomate", "kartoffel", "apfel", "nuss", "nuß",
    "schinken", "käse", "zwiebel", "knoblauch", "karotten", "möhren",
    "spinat", "brokkoli", "broccoli", "hähnchen", "hühner", "rinds",
    "rinder", "schweine", "lachs", "thunfisch", "gurken", "paprika",
    "pilz", "champignon", "zitronen", "zitrone", "orangen", "erdbeer",
    "himbeer", "heidel", "blau", "johannis", "kirsch", "birnen",
    "pflaume", "mais", "reis", "hafer", "dinkel", "roggen", "weizen",
    "vollkorn", "milch", "sahne", "quark", "joghurt", "butter",
    "kokos", "mandel", "walnuss", "cashew", "erdnuss", "sesam",
    "sonnenblumen", "oliven", "pesto", "soße", "sauce", "suppe",
    "salat", "brot", "kuchen", "creme", "mus", "saft",
]
# Sort longest first so "sonnenblumen" matches before "blumen"
_COMPOUND_PARTS.sort(key=len, reverse=True)


def _compound_split(word: str) -> str | None:
    """
    Try splitting a compound German word at food-word boundaries.
    Returns space-separated version if a split is found, else None.
    E.g., "kürbispesto" → "kürbis pesto"
    """
    lower = word.lower()
    if " " in lower or len(lower) < 6:
        return None
    for part in _COMPOUND_PARTS:
        if lower.startswith(part) and len(lower) > len(part) + 2:
            remainder = lower[len(part):]
            return f"{part} {remainder}"
        if lower.endswith(part) and len(lower) > len(part) + 2:
            prefix = lower[: len(lower) - len(part)]
            return f"{prefix} {part}"
    return None


def _compound_join(text: str) -> str | None:
    """
    Try joining space-separated words into a German compound.
    Returns joined version if the parts look like food compounds, else None.
    E.g., "kürbis pesto" → "kürbispesto"
    """
    words = text.lower().split()
    if len(words) != 2:
        return None
    for part in _COMPOUND_PARTS:
        if words[0] == part or words[1] == part:
            return "".join(words)
    return None


def _has_german_chars(text: str) -> bool:
    """Check for German-specific characters (ä, ö, ü, ß)."""
    return bool(re.search(r"[äöüßÄÖÜ]", text))


def _apply_dict(text: str, mapping: dict[str, str]) -> str:
    """
    Apply a synonym dictionary: replace whole-word matches, longest first.
    Only replaces if the key appears as a whole word (or the entire string).
    """
    # Sort by length descending so longer phrases match first
    for key in sorted(mapping.keys(), key=len, reverse=True):
        # Word-boundary match
        pattern = re.compile(r"\b" + re.escape(key) + r"\b", re.IGNORECASE)
        if pattern.search(text):
            text = pattern.sub(mapping[key], text)
            break  # apply only the first (longest) match to avoid cascading
    return text


def _build_variants(result: NormalizedQuery) -> list[str]:
    """
    Build additional search strings that might help find the right BLS entry.
    These are used as extra queries in the retrieval step.
    """
    variants = []

    # If we have a prep state, create a variant with it appended
    if result.prep_state and result.prep_state not in result.cleaned:
        variants.append(f"{result.cleaned} {result.prep_state}")

    # If brand detected, create a variant with the generic mapping
    if result.brand:
        generic = BRAND_MAP.get(result.brand, "")
        if generic:
            variants.append(generic)

    # If multi-ingredient, add individual components as variants
    if result.is_multi_ingredient:
        for comp in result.components[:3]:  # limit to first 3
            if comp != result.cleaned:
                variants.append(comp)

    # If fat% found, add a variant with it included
    if result.fat_percent:
        variants.append(f"{result.cleaned} {result.fat_percent}% fett")

    # Compound word splitting/joining: try both forms
    split = _compound_split(result.cleaned)
    if split:
        variants.append(split)
    joined = _compound_join(result.cleaned)
    if joined:
        variants.append(joined)

    return variants


# =====================================================================
#  Convenience: batch normalize
# =====================================================================

def normalize_batch(texts: list[str]) -> list[NormalizedQuery]:
    """Normalize a list of food descriptions."""
    return [normalize(t) for t in texts]


# =====================================================================
#  CLI test
# =====================================================================

if __name__ == "__main__":
    test_cases = [
        "Banane",
        "Olivenöl",
        "Eier",
        "Spiegelei",
        "Edamer (40% Fett)",
        "Joghurt 3,8%-Fett",
        "Cherry Tomaten",
        "Curly fries",
        "Poree",
        "Wienerle",
        "Erdnussfips",
        "Maggi",
        "Bolognese aus Tütensoße Maggi",
        "Kaffee mit Milch",
        "Chicken salad",
        "Vollkornnudeln",
        "Zucchini, Aubergine, Tomate",
        "Kornkracher Brötchen mit Zwiebel, Speck und Käse",
        "alternative Sahne Rama 1,5% Fett",
        "Haferflocken",
        "Wasser",
        "Kürbiscurry",
        "Bifteki mit Schafskäse",
        "Wurstsalat (selbstgemacht)",
        "gekochter Schinken",
        "Smoothie",
    ]

    print(f"{'Input':<45} {'Cleaned':<50} {'Prep':<12} {'Fat%':<6} {'Brand':<10} {'EN':<4} {'Multi':<6}")
    print("─" * 140)
    for t in test_cases:
        r = normalize(t)
        print(
            f"{r.original:<45} "
            f"{r.cleaned:<50} "
            f"{r.prep_state or '–':<12} "
            f"{r.fat_percent or '–':<6} "
            f"{r.brand or '–':<10} "
            f"{'Y' if r.is_english else '–':<4} "
            f"{'Y' if r.is_multi_ingredient else '–':<6}"
        )
        if r.search_variants:
            print(f"{'':>45}   variants: {r.search_variants}")
    print()
    print("Done ✓")
