"""
ai_model.py
Real API recipe reccommendation using Claude API (claude-haiku).
Falls back to a local mock when ANTHROPIC_API_KEY is not set - useful for offline dev / CI without secrets

"""

import json
import os
import random
import re
import time


#------------------------------------------------------------------------------------------
# Real Claude API implementation
#------------------------------------------------------------------------------------------

def _call_claude(ingredients: list[str]) -> dict:
    """Call Clude Haiku and parse the JSON response."""
    import anthropic # imported here so the module loads without the package

    client = anthropic.Anthropic()   #reads ANTHROPIC_API_KEY from env

    prompt = (
        f"You are a recipe assistant. "
        f"Given these ingredients: {', '.join(ingredients)}, "
        f"recommend ONE recipe. "
        f"Respond ONLY with valid JSON in this exact format, no extra text:\n"
        f'{{"recipe_name": "...", "time_min": <int>, "confidence": <float 0-1>, '
        f'"ingredients": [...]}}'
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if present (```json ... ```)
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)

# ---------------------------------------------------------------------------
# Mock fallback (no API key required)
# ---------------------------------------------------------------------------


_KNOWN_RECIPES = {
    frozenset(["eggs", "flour", "milk"]): ("Pancakes", 15),
    frozenset(["potato", "salt", "oil"]): ("French fries", 25),
    frozenset(["chicken fillet", "salt", "pepper"]): ("Chicken fillet", 20),
    frozenset(["tomato", "cheese", "basil"]): ("Caprese salad", 10),
    frozenset(["pasta", "tomato sauce", "garlic"]): ("Pasta with tomato sauce", 18),
    frozenset(["rice", "chicken", "broccoli"]): ("Chicken and rice bowl", 30),
    frozenset(["flour", "sugar", "butter", "eggs"]): ("Cake", 45),
    frozenset(["tuna", "mayonnaise", "corn"]): ("Tuna salad", 8),
    frozenset(["spinach", "eggs", "feta cheese"]): ("Spinach omelette", 12),
    frozenset(["shrimp", "garlic", "lemon", "butter"]): ("Garlic butter shrimp", 15),
}


def _call_mock(ingredients: list[str]) -> dict:
    """Deterministic mock that simulates realistic AI behaviour."""
    time.sleep(random.uniform(0.05, 0.3))

    key = frozenset(ingredients)
    if key in _KNOWN_RECIPES:
        recipe_name, time_min = _KNOWN_RECIPES[key]
        # 10 % flakiness — intentional, to exercise retry tests
        if random.random() < 0.1:
            recipe_name = "Wrong recipe (flaky simulation)"
        confidence = round(random.uniform(0.75, 0.95), 2)
    else:
        recipe_name = "Generic meal"
        time_min = random.randint(10, 40)
        confidence = round(random.uniform(0.3, 0.6), 2)

    return {
        "recipe_name": recipe_name,
        "time_min": time_min,
        "confidence": confidence,
        "ingredients": ingredients,
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def recommend_recipe(ingredients, system_prompt: str = "") -> dict | str:
    """
    Recommend a recipe for the given ingredients.

    Parameters
    ----------
    ingredients : list[str] | str | None
        Ingredients to cook with.
    system_prompt : str
        Ignored when using the real API (kept for test-suite compatibility).
        When the mock is active it controls output format ("JSON" → dict).

    Returns
    -------
    dict  when ingredients are valid and JSON format is requested / API is used
    str   in mock mode without "JSON" in system_prompt
    """

    # ---- input validation (same behaviour regardless of backend) ----------
    if ingredients is None or (isinstance(ingredients, list) and len (ingredients) == 0):
        if random.random() < 0.3:
            raise ValueError ("Model crashed on empty input")
        return {"error": "Empty string not allowed", "status":400}

    if isinstance(ingredients, str) and ingredients.strip() == "":
        return {"error": "Empty string not allowed", "status": 400}

    if isinstance(ingredients, str) and len(ingredients) > 100:
        time.sleep(0.5)
        return{"recipe_name": "Input too long", "confidence": 0.3, "time_min":0}

    # ---- choose backend ---------------------------------------------------
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    use_real_api = bool(api_key and api_key != "dummy")

    if use_real_api:
        try:
            ing_list = ingredients if isinstance(ingredients, list) else [ingredients]
            return _call_claude(ing_list)
        except Exception as exc:  #noqa BLE001
            #Gracefull degradation: faill through to mock on any API error
            print(f"[ai_model] Claude API error ({exc}), failing back to mock.")

   # ---- mock path --------------------------------------------------------
    ing_list = ingredients if isinstance(ingredients, list) else [ingredients]
    result = _call_mock(ing_list)

    if "JSON" in system_prompt.upper():
        return result
    return f"Recipe: {result['recipe_name']}, time: {result['time_min']} min"
