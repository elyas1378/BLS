"""
Dish-Type Suffix Synonyms
=========================
Maps German food dish-type suffixes to BLS terminology synonyms.
Used to bridge vocabulary gaps: "Stampf" → "Püree", "Mousse" → "Creme", etc.
"""

DISH_TYPE_SYNONYMS = {
    # Mashed/puréed
    "stampf": ["püree", "mus", "brei"],
    "püree": ["stampf", "mus", "brei"],

    # Spreads
    "aufstrich": ["brotaufstrich", "fettaufstrich", "paste", "creme"],

    # Sauces
    "soße": ["sauce", "dressing", "tunke"],
    "sauce": ["soße", "dressing"],
    "dip": ["aufstrich", "soße"],

    # Dessert textures
    "mousse": ["creme", "schaum", "dessert"],
    "creme": ["mousse", "pudding", "dessert"],
    "grütze": ["kompott", "gelee"],

    # Soups
    "suppe": ["eintopf", "brühe"],
    "eintopf": ["suppe", "topf"],

    # Curry/stew-style
    "curry": ["eintopf", "ragout", "pfanne", "gericht"],
    "ragout": ["gulasch", "eintopf", "geschnetzeltes"],
    "gulasch": ["ragout", "eintopf"],
    "pfanne": ["gericht", "ragout"],

    # Salad/bowl
    "bowl": ["salat", "schüssel"],
    "salat": ["rohkost", "bowl"],

    # Baked goods shape
    "schnecke": ["plunder", "hörnchen", "tasche", "gebäck"],
    "stange": ["gebäck", "grissini", "brötchen"],
    "zopf": ["hefezopf", "hefekranz", "gebäck"],

    # Drinks
    "getränk": ["drink", "saft", "nektar", "schorle"],
    "schorle": ["saftschorle", "getränk"],
    "shake": ["milchmischgetränk", "drink"],

    # Oils
    "öl": ["pflanzenöl", "speiseöl"],
}
