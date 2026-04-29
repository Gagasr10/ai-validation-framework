"""
test_accuracy.py

Gold-standard accuracy tests: 10 known ingredient combinations
must produce the correct recipe name with >= 70 % confidence.

When ANTHROPIC_API_KEY is set the tests run against the real Claude API.
Without a key the deterministic mock is used instead.
"""

import random

import pytest
from ai_model import recommend_recipe

GOLD_DATA = [
    (["eggs", "flour", "milk"], "Pancakes"),
    (["potato", "salt", "oil"], "French fries"),
    (["chicken fillet", "salt", "pepper"], "Chicken fillet"),
    (["tomato", "cheese", "basil"], "Caprese salad"),
    (["pasta", "tomato sauce", "garlic"], "Pasta with tomato sauce"),
    (["rice", "chicken", "broccoli"], "Chicken and rice bowl"),
    (["flour", "sugar", "butter", "eggs"], "Cake"),
    (["tuna", "mayonnaise", "corn"], "Tuna salad"),
    (["spinach", "eggs", "feta cheese"], "Spinach omelette"),
    (["shrimp", "garlic", "lemon", "butter"], "Garlic butter shrimp"),
]


@pytest.mark.parametrize("ingredients, expected", GOLD_DATA)
def test_accuracy_exact_match(ingredients, expected, monkeypatch):
    """Model must return the correct recipe name and confidence >= 0.7."""
    monkeypatch.setattr(random, "random", lambda: 0.5)  # neutralise 10% flakiness
    result = recommend_recipe(ingredients, system_prompt="Always return JSON")

    assert isinstance(result, dict), (
        f"Expected dict response, got {type(result).__name__}: {result}"
    )
    assert result.get("recipe_name") == expected, (
        f"Expected '{expected}', got '{result.get('recipe_name')}'"
    )
    assert result.get("confidence", 0) >= 0.7, (
        f"Confidence too low: {result.get('confidence')}"
    )


def test_overall_accuracy(monkeypatch):
    """At least 90 % of gold recipes must pass (allows for 1 flaky failure)."""
    monkeypatch.setattr(random, "random", lambda: 0.5)  # neutralise 10% flakiness
    passed = 0
    for ingredients, expected in GOLD_DATA:
        result = recommend_recipe(ingredients, system_prompt="Always return JSON")
        if isinstance(result, dict) and result.get("recipe_name") == expected:
            passed += 1

    accuracy = passed / len(GOLD_DATA)
    assert accuracy >= 0.9, (
        f"Overall accuracy {accuracy:.0%} is below the 90 % threshold"
    )