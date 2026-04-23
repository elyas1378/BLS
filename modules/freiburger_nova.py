"""
Freiburger Ernährungsprotokoll — NOVA overrides.

Source: 20260421_Standardisierungsbeispiele.xlsx, sheet "NOVA Assignment".
Authority: Leonie Burgard + Sidney Marie Pruß (2026-04-23).
Principle: "Foods classified according to their most common form of consumption
           in the general population, with preference for minimally processed or
           home-prepared equivalents where applicable."

Values:
    int (4, 2)   — explicit NOVA verdict
    "NOT_4"      — assertion that item is NOT NOVA 4 (could be 1/2/3,
                   resolved by the rule-based classifier, capped at 3)
"""
from __future__ import annotations

import re
import unicodedata


def _normalize(s: str) -> str:
    """Match normalization used when building the lookup keys."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).lower().strip()
    s = re.sub(r'\s+', ' ', s)
    # normalize fancy dashes to plain hyphen
    for ch in ("\u2010", "\u2011", "\u2013", "\u2014"):
        s = s.replace(ch, "-")
    return s


FREIBURGER_NOVA: dict[str, object] = {
    'brötchen': "NOT_4",  # Brötchen
    'croissant aus blätterteig': 4,  # Croissant aus Blätterteig
    'graubrot': "NOT_4",  # Graubrot
    'hefezopf': "NOT_4",  # Hefezopf
    'knäckebrot': "NOT_4",  # Knäckebrotow]
    'weizentoastbrot': 4,  # Weizentoastbrot
    'vollkornbrötchen': "NOT_4",  # Vollkornbrötchen
    'vollkornbrot': "NOT_4",  # Vollkornbrot
    'weißbrot': "NOT_4",  # Weißbrot
    'zwieback': 4,  # Zwieback
    'butter': "NOT_4",  # Butter
    'margarine pflanzlich (linolsäure 30-50 %)': 4,  # Margarine pflanzlich (Linolsäure 30–50 %)
    'margarine halbfett (linolsäure > 50 %)': 4,  # Margarine halbfett (Linolsäure > 50 %)
    'edelpilzkäse': "NOT_4",  # Edelpilzkäse
    'frischkäse': "NOT_4",  # Frischkäse
    'schmelzkäse': 4,  # Schmelzkäse
    'schnittkäse (30% f. i. tr.)': "NOT_4",  # Schnittkäse (30% F. i. Tr.)
    'schnittkäse (50% f. i. tr.)': "NOT_4",  # Schnittkäse (50% F. i. Tr.)
    'camembert (45% f. i. tr. )': "NOT_4",  # Camembert (45% F. i. Tr. )
    'camembert (60% f. i. tr.)': "NOT_4",  # Camembert (60% F. i. Tr.)
    'bierschinken': 4,  # Bierschinken
    'corned beef': 4,  # Corned Beef
    'fleischwurst': 4,  # Fleischwurst
    'fleischkäse (aufschnitt)': 4,  # Fleischkäse (Aufschnitt)
    'fleischsalat': 4,  # Fleischsalat
    'leberwurst': 4,  # Leberwurst
    'mettwurst': 4,  # Mettwurst
    'teewurst': 4,  # Teewurst
    'salami': 4,  # Salami
    'schinken (roh geräuchert, lachsschinken)': "NOT_4",  # Schinken (roh geräuchert, Lachsschinken)
    'schinken (gekocht, ungeräuchert)': "NOT_4",  # Schinken (gekocht, ungeräuchert)
    'speck': "NOT_4",  # Speck
    'honig': "NOT_4",  # Honig
    'konfitüre': "NOT_4",  # Konfitüre
    'nuss-nougat-creme (süß)': 4,  # Nuss‐Nougat‐Creme (süß)
    'vegetabiler brotaufstrich': 4,  # Vegetabiler Brotaufstrich
    'hühnerei (gekocht)': "NOT_4",  # Hühnerei (gekocht)
    'cornflakes': 4,  # Cornflakes
    'cornflakes gezuckert (geröstet)': 4,  # Cornflakes gezuckert (geröstet)
    'haferflocken': "NOT_4",  # Haferflocken
    'müsli': "NOT_4",  # Müsliow]
    'buttermilch': "NOT_4",  # Buttermilch
    'joghurt natur fettarm 1,5 % f. i. tr.': "NOT_4",  # Joghurt natur fettarm 1,5 % F. i. Tr.
    'joghurt vollfett 3,5% f. i. tr.': "NOT_4",  # Joghurt vollfett 3,5% F. i. Tr.
    'joghurt fettarm 1,5% f. i. tr. mit fruchtzubereitung': 4,  # Joghurt fettarm 1,5% F. i. Tr.  mit Fruchtzubereitung
    'joghurt vollfett 3,5 % f.i. tr. mit fruchtzubereitung': 4,  # Joghurt vollfett 3,5 % F.i. Tr. mit Fruchtzubereitung
    'kuhmilch (1,5% f. i. tr.)': "NOT_4",  # Kuhmilch (1,5% F. i. Tr.)
    'kuhmilch (3,5% f. i. tr)': "NOT_4",  # Kuhmilch (3,5% F. i. Tr)
    'kakao/trinkschokolade': "NOT_4",  # Kakao/Trinkschokolade
    'quark (magerstufe)': "NOT_4",  # Quark (Magerstufe)
    'quark (halbfettstufe)': "NOT_4",  # Quark (Halbfettstufe)
    'schlagsahne (30 % fett)': "NOT_4",  # Schlagsahne (30 % Fett)
    'kondensmilch (7,5 % fett)': "NOT_4",  # Kondensmilch (7,5 % Fett)
    'beerenobst': "NOT_4",  # Beerenobst
    'weintrauben (frisch)': "NOT_4",  # Weintrauben (frisch)
    'kernobst': "NOT_4",  # Kernobst
    'steinobst': "NOT_4",  # Steinobst
    'banane (frisch)': "NOT_4",  # Banane (frisch)
    'südfrüchte': "NOT_4",  # Südfrüchte
    'zitrusfrüchte': "NOT_4",  # Zitrusfrüchte
    'rosinen, trockenfrüchte': "NOT_4",  # Rosinen, Trockenfrüchte
    'cornichons, saure gurken': "NOT_4",  # Cornichons, saure Gurken
    'nüsse': "NOT_4",  # Nüsse
    'oliven': "NOT_4",  # Oliven
    'erdnüsse gesalzen': "NOT_4",  # Erdnüsse gesalzen
    'erdnussflips': 4,  # Erdnussflips
    'chips': 4,  # Chips
    'salzstangen': 4,  # Salzstangen
    'suppen klar': "NOT_4",  # Suppen klar
    'suppe hell gebunden': "NOT_4",  # Suppe hell gebunden
    'cremesuppe': "NOT_4",  # Cremesuppe
    'gulaschsuppe': "NOT_4",  # Gulaschsuppe
    'nudelsuppe mit hühnerfleisch': "NOT_4",  # Nudelsuppe mit Hühnerfleisch
    'gemüsesuppe': "NOT_4",  # Gemüsesuppe
    'kartoffelsuppe': "NOT_4",  # Kartoffelsuppe
    'linsen-eintopf': "NOT_4",  # Linsen-Eintopf
    'hackfleisch': "NOT_4",  # Hackfleisch
    'kalbfleisch': "NOT_4",  # Kalbfleisch
    'rindfleisch': "NOT_4",  # Rindfleisch
    'schweinefleisch': "NOT_4",  # Schweinefleisch
    'innereien': "NOT_4",  # Innereien
    'kotelett': "NOT_4",  # Kotelett
    'schnitzel paniert': "NOT_4",  # Schnitzel paniert
    'würstchen': 4,  # Würstchen
    'brathähnchen': "NOT_4",  # Brathähnchen
    'geflügel': "NOT_4",  # Geflügel
    'fisch': "NOT_4",  # Fisch
    'fischfilet paniert': "NOT_4",  # Fischfilet paniert
    'fischkonserve abgetropft': "NOT_4",  # Fischkonserve abgetropft
    'kartoffeln (gekocht)': "NOT_4",  # Kartoffeln (gekocht)
    'pellkartoffeln': "NOT_4",  # Pellkartoffeln
    'bratkartoffeln': "NOT_4",  # Bratkartoffeln
    'kartoffelbrei': "NOT_4",  # Kartoffelbrei
    'kartoffelknödel (gekocht)': "NOT_4",  # Kartoffelknödel (gekocht)
    'kartoffelpuffer': "NOT_4",  # Kartoffelpuffer
    'kartoffelsalat': "NOT_4",  # Kartoffelsalat
    'pommes frites': 4,  # Pommes Frites
    'weißer reis gekocht': "NOT_4",  # weißer Reis gekocht
    'naturreis (gekocht)': "NOT_4",  # Naturreis (gekocht)
    'nudeln eifrei gekocht': "NOT_4",  # Nudeln eifrei gekocht
    'vollkornnudeln gekocht': "NOT_4",  # Vollkornnudeln gekocht
    'semmelknödel': "NOT_4",  # Semmelknödel
    'schupfnudeln': 4,  # Schupfnudeln
    'spätzle, eiernudeln gekocht': "NOT_4",  # Spätzle, Eiernudeln gekocht
    'joghurt-salat-soße': "NOT_4",  # Joghurt-Salat-Soßeow]
    'essig-öl-marinade': "NOT_4",  # Essig-Öl-Marinade
    'bechamelsoße': "NOT_4",  # Bechamelsoße
    'grundsoße': "NOT_4",  # Grundsoße
    'hackfleischsoße': "NOT_4",  # Hackfleischsoße
    'jägersoße': "NOT_4",  # Jägersoße
    'käsesoße': "NOT_4",  # Käsesoße
    'grüne soße, kräuterquark': "NOT_4",  # Grüne Soße, Kräuterquark
    'tomatensauce': "NOT_4",  # Tomatensauce
    'grillsauce': 4,  # Grillsauce
    'tomatenketchup': 4,  # Tomatenketchup
    'tomatenmark': "NOT_4",  # Tomatenmark
    'senf': "NOT_4",  # Senf
    'mayonnaise': 4,  # Mayonnaise
    'bratfett': 2,  # Bratfett
    'öl (pflanzenöl)': "NOT_4",  # Öl (Pflanzenöl)
    'blattsalat (mit dressing)': "NOT_4",  # Blattsalat (mit Dressing)
    'rohkostsalat mit dressing': "NOT_4",  # Rohkostsalat mit Dressing
    'bleichsellerie, mangold, spinat': "NOT_4",  # Bleichsellerie, Mangold, Spinat
    'grüne bohnen': "NOT_4",  # Grüne Bohnen
    'aubergine, gruke, paprika, tomate , zucchini': "NOT_4",  # Aubergine, Gruke, Paprika, Tomate , Zucchini
    'gemüsemais': "NOT_4",  # Gemüsemais
    'blumenkohl, broccoli, kohl (rot-, grün-, weiß-), kohlrabi, rosenkohl, wirsing': "NOT_4",  # Blumenkohl, Broccoli, Kohl (Rot-, Grün-, Weiß-), Kohlrabi, Rosenkohl, Wirsing
    'sauerkraut': "NOT_4",  # Sauerkraut
    'fenchel, lauch, spargel, zwiebel': "NOT_4",  # Fenchel, Lauch, Spargel, Zwiebel
    'möhre, radieschen, rettich, rote beete, rüben, sellerie, schwarzwurzel': "NOT_4",  # Möhre, Radieschen, Rettich, Rote Beete, Rüben, Sellerie, Schwarzwurzel
    'pilze': "NOT_4",  # Pilze
    'nudelsalat': "NOT_4",  # Nudelsalat
    'wurstsalat': 4,  # Wurstsalat
    'griechischer salat': "NOT_4",  # Griechischer Salat
    'italienischer salat': "NOT_4",  # Italienischer Salat
    'bratwurst ohne brötchen': 4,  # Bratwurst ohne Brötchen
    'currywurst ohne brötchen': 4,  # Currywurst ohne Brötchen
    'hamburger': 4,  # Hamburger
    'cheeseburger': 4,  # Cheeseburger
    'big mac': 4,  # Big Mac
    'maultaschen/ravioli': 4,  # Maultaschen/Ravioli
    'pizza': 4,  # Pizza
    'pfannkuchen': "NOT_4",  # Pfannkuchen
    'erbsen (gekocht)': "NOT_4",  # Erbsen (gekocht)
    'linsen (gekocht)': "NOT_4",  # Linsen (gekocht)
    'bohnen (gekocht)': "NOT_4",  # Bohnen (gekocht)
    'pudding (mit milch)': 4,  # Pudding (mit Milch)
    'eiscreme': 4,  # Eiscreme
    'obstkuchen': "NOT_4",  # Obstkuchen
    'cremekuchen': "NOT_4",  # Cremekuchen
    'rührkuchen': "NOT_4",  # Rührkuchen
    'plätzchen, kekse': 4,  # Plätzchen, Kekse
    'schokolade': 4,  # Schokolade
    'praline': 4,  # Praline
    'bonbon, hartkaramelle': 4,  # Bonbon, Hartkaramelle
    'fruchtgummi': 4,  # Fruchtgummi
    'zucker': "NOT_4",  # Zucker
    'kaffee (schwarz)': "NOT_4",  # Kaffee (schwarz)
    'tee (ungesüßt)': "NOT_4",  # Tee (ungesüßt)
    'mineralwasser (mit kohlensäure)': "NOT_4",  # Mineralwasser (mit Kohlensäure)
    'limonade': 4,  # Limonade
    'cola getränke': 4,  # Cola Getränke
    'fruchtsaft': "NOT_4",  # Fruchtsaft
    'obst- fruchtnektar': 4,  # Obst- Fruchtnektar
    'bier alkoholfrei': "NOT_4",  # Bier alkoholfrei
    'bier': "NOT_4",  # Bier
    'weizenbier': "NOT_4",  # Weizenbier
    'weißwein': "NOT_4",  # Weißwein
    'rotwein': "NOT_4",  # Rotwein
    'sekt': "NOT_4",  # Sekt
    'likör': 4,  # Likör
    'schnaps, branntwein': "NOT_4",  # Schnaps, Branntwein
}


def lookup_nova(food_desc: str):
    """Return explicit NOVA value (int) if Freiburger asserts one, else None.

    For "NOT_4" entries returns None (use is_not_nova4 to check that separately).
    """
    v = FREIBURGER_NOVA.get(_normalize(food_desc))
    return v if isinstance(v, int) else None


def is_not_nova4(food_desc: str) -> bool:
    """True when the Freiburger Protokoll explicitly asserts item is NOT NOVA 4."""
    return FREIBURGER_NOVA.get(_normalize(food_desc)) == "NOT_4"
