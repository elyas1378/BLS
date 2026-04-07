"""
Claude Query Expander
=====================
Before text retrieval, asks Claude (Haiku) to generate smart search terms
for the BLS food database. This fixes cases where the text retriever fails
because it only matches characters, not food concepts.

Cost: ~$0.001 per expansion (Haiku + 150 tokens).
"""

from __future__ import annotations

import json
import os

SYSTEM_PROMPT = """You are a German food database expert. Given a food description from a nutrition study participant, generate search terms that would help find this food in the BLS (Bundeslebensmittelschlüssel) food database. The BLS contains standardized German food names like 'Weizenmischbrot mit Lachs geräuchert', 'Rindfleisch gekocht', 'Joghurt mit Früchten', etc.

Rules:
- Return 5-10 German search terms, one per line
- Include the literal food name if it might exist in a database
- Include German synonyms and alternative names
- For brand names, include what the product actually IS (e.g. Big Mac → Hamburger, Cheeseburger, Rindfleischburger)
- For colloquial/slang terms, include the proper German food name (e.g. Grillfackeln → mariniertes Schweinefleisch, Grillspieß)
- For English food names, include the German translation
- For composite foods, include both the composite term AND individual components
- For drinks disguised as food names, clarify (e.g. heiße Schokolade → Kakaogetränk, Schokolade heißes Getränk, Trinkschokolade)
- If the food is a modern, specialty, or foreign item that likely does not exist in the BLS, also suggest the closest traditional German food equivalent (e.g. Proteinriegel → Müsliriegel, Energieriegel; Taco Schalen → Maistortilla, Maisfladen; Flohsamenschalen → Leinsamen, Weizenkleie)
- Think about what terms a German nutritional database would actually use
- Return ONLY the search terms, no explanations, no numbering

BLS-specific terminology:
- 'Trüffel' in BLS means the mushroom (Tuber), NOT chocolate truffles. For chocolate truffles use: Praline, Konfekt, Schokolade gefüllt
- For hot/warm drinks, always include 'Getränk' or 'heißes Getränk'. 'Schokolade' alone returns solid chocolate bars.
- 'Mousse' is poorly represented in BLS. Use Creme, Pudding, or Dessert instead.
- Brand names: Buko = Frischkäse, Manner = Haselnuss-Waffelschnitten, Riesen = Karamellbonbon. When you recognize a brand, provide the generic food category.

BLS naming patterns — your search terms MUST match these:
The BLS database does NOT use compound words the way German speakers naturally write them. You MUST decompose compounds into BLS-style phrases:
- "Vanillejoghurt" → search "Joghurt mit Vanille", "Joghurt Vanillegeschmack"
- "Fruchtjoghurt" → search "Joghurt mit Fruchtzubereitung", "Joghurt Frucht"
- "Hühnersuppe" → search "Suppe mit Hühnerfleisch", "Nudelsuppe Huhn"
- "Gemüserisotto" → search "Gemüsereis", "Reis mit Gemüse" (BLS uses "Reis" not "Risotto")
- "Käsestange" → search "Käsegebäck", "Gebäck mit Käse", "Käsegebäck Blätterteig"
- "Nusszopf" → search "Nusskuchen", "Hefeteiggebäck mit Nüssen"
General rule: split [Modifier][FoodType] compounds into "FoodType mit Modifier" AND search for BLS synonyms of FoodType:
- Riegel → Energieriegel, Müsliriegel
- Risotto → Reis
- Zopf/Schnecke/Stange (baked goods) → Kuchen, Gebäck, Plundergebäck, Hefeteig
- Suppe → also try Eintopf, Brühe
- Curry → Curryreis, Currysoße"""


class QueryExpander:
    def __init__(self, api_key: str | None = None):
        import anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Set ANTHROPIC_API_KEY or pass api_key parameter.")

        self.client = anthropic.Anthropic(api_key=key, timeout=30.0)

    def expand(self, food_description: str) -> list[str]:
        """Generate smart search terms for a food description.

        Returns list of search term strings, or empty list on failure.
        """
        try:
            response = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=150,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": food_description}],
                temperature=0.0,
            )

            text = response.content[0].text.strip()
            terms = [line.strip().lstrip("0123456789.-) ") for line in text.split("\n")]
            terms = [t for t in terms if t and len(t) >= 2]

            return terms

        except Exception as e:
            print(f"  ⚠ Haiku expansion failed for '{food_description[:40]}': {type(e).__name__}: {e}")
            return []

    # ── V2: Combined spelling + expansion ──

    _V2_SYSTEM = (
        "You are a German food database expert. Given a food description "
        "from a nutrition study participant:\n"
        "1. Correct any spelling errors in the food description.\n"
        "2. Generate 5-8 German search terms that would help find this food "
        "in the BLS (Bundeslebensmittelschlüssel) food database.\n\n"
        "Include: the corrected name, common German synonyms, "
        "brand-to-generic-product mappings, English-to-German translations, "
        "and for composite foods also list the individual components.\n"
        "If the food likely does not exist in BLS, suggest the closest "
        "traditional German food equivalent.\n\n"
        "BLS terminology: 'Trüffel' = mushroom (not chocolate); "
        "for hot drinks include 'Getränk'; 'Mousse' → use 'Creme/Pudding'; "
        "Buko = Frischkäse; Manner = Haselnuss-Waffelschnitten.\n\n"
        "BLS naming patterns — decompose compounds into BLS-style phrases:\n"
        "- 'Vanillejoghurt' → 'Joghurt mit Vanille', 'Joghurt Vanillegeschmack'\n"
        "- 'Fruchtjoghurt' → 'Joghurt mit Fruchtzubereitung', 'Joghurt Frucht'\n"
        "- 'Hühnersuppe' → 'Suppe mit Hühnerfleisch', 'Nudelsuppe Huhn'\n"
        "- 'Gemüserisotto' → 'Gemüsereis', 'Reis mit Gemüse' (BLS uses 'Reis' not 'Risotto')\n"
        "- 'Käsestange' → 'Käsegebäck', 'Gebäck mit Käse'\n"
        "- 'Nusszopf' → 'Nusskuchen', 'Hefeteiggebäck mit Nüssen'\n"
        "Split [Modifier][FoodType] → 'FoodType mit Modifier' + BLS synonyms:\n"
        "Riegel → Energieriegel, Müsliriegel; Risotto → Reis; "
        "Zopf/Schnecke/Stange → Kuchen, Gebäck, Plundergebäck; "
        "Suppe → Eintopf, Brühe; Curry → Curryreis, Currysoße.\n\n"
        'Respond ONLY with JSON, no markdown, no explanation:\n'
        '{"corrected": "corrected food description", '
        '"search_terms": ["term1", "term2", ...]}'
    )

    def expand_with_spelling(
        self,
        food_description: str,
        unknown_tokens: list[str] | None = None,
    ) -> dict:
        """Combined spelling correction + search term expansion.

        Called BEFORE retrieval when Tier 1 flags unknown tokens.

        Returns dict with:
            "corrected": str — spelling-corrected food description
            "search_terms": list[str] — BLS search terms
        """
        user_msg = f'Food description: "{food_description}"'
        if unknown_tokens:
            user_msg += f"\nUnknown tokens that need attention: {unknown_tokens}"

        fallback = {"corrected": food_description, "search_terms": []}

        try:
            response = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                system=self._V2_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
                temperature=0.0,
            )

            text = response.content[0].text.strip()

            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
                if text.startswith("json"):
                    text = text[4:].strip()

            # Parse JSON
            try:
                result = json.loads(text)
                if not isinstance(result, dict):
                    raise ValueError("not a dict")
                # Ensure expected keys
                result.setdefault("corrected", food_description)
                result.setdefault("search_terms", [])
                # Filter search terms
                result["search_terms"] = [
                    t for t in result["search_terms"]
                    if isinstance(t, str) and len(t) >= 2
                ]
            except (json.JSONDecodeError, ValueError):
                # Fallback: treat as line-separated search terms (old format)
                terms = [
                    line.strip().lstrip("0123456789.-) ")
                    for line in text.split("\n")
                ]
                terms = [t for t in terms if t and len(t) >= 2]
                result = {"corrected": food_description, "search_terms": terms}

            return result

        except Exception as e:
            print(f"  ⚠ Haiku expansion (v2) failed for '{food_description[:40]}': {type(e).__name__}: {e}")
            return fallback

    # ── Gemini Flash expansion ──

    def expand_gemini(self, food_description: str) -> list[str]:
        """Generate search terms using Gemini Flash.

        Runs in parallel with Haiku — provides complementary terms.
        Returns list of search term strings, or empty list on failure.
        """
        # Get API key
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            try:
                import streamlit as st
                gemini_key = st.secrets.get("GEMINI_API_KEY")
            except Exception:
                pass
        if not gemini_key:
            return []

        try:
            from google import genai

            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=food_description,
                config=genai.types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=150,
                ),
            )

            text = response.text.strip()
            terms = [line.strip().lstrip("0123456789.-) ") for line in text.split("\n")]
            terms = [t for t in terms if t and len(t) >= 2]

            return terms

        except Exception as e:
            print(f"  ⚠ Gemini expansion failed for '{food_description[:40]}': {type(e).__name__}: {e}")
            return []
