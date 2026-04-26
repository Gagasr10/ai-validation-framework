"""
test_soft_assertions.py

Uses pytest-check for soft assertions: all checks run even when
one fails, giving a full picture of what the model got right/wrong.
Useful for partial-correctness scoring of AI outputs.
"""

import pytest
import pytest_check as check
from ai_model import recommend_recipe


def test_partial_correctness_pancakes():
    """All four fields of the Pancakes recipe must be correct."""
    result = recommend_recipe(
        ["eggs", "flour", "milk"], system_prompt="Always return JSON"
    )

    check.is_instance(result, dict, "Response must be a dict")

    if isinstance(result, dict):
        check.equal(result.get("recipe_name"), "Pancakes",
                    f"Wrong recipe name: {result.get('recipe_name')}")
        check.between(result.get("time_min", 0), 5, 30,
                      f"Unexpected cook time: {result.get('time_min')}")
        check.greater(result.get("confidence", 0), 0.7,
                      f"Confidence too low: {result.get('confidence')}")
        check.is_in("ingredients", result,
                    "Response is missing 'ingredients' key")


def test_partial_correctness_unknown_recipe():
    """
    Unknown ingredients produce a Generic meal —
    soft assertions capture what IS correct even when name is unexpected.
    """
    result = recommend_recipe(
        ["unobtainium", "dark_matter"], system_prompt="Always return JSON"
    )

    check.is_not_none(result, "Result must not be None")

    if isinstance(result, dict):
        check.is_in("recipe_name", result, "Missing 'recipe_name'")
        check.is_in("time_min", result, "Missing 'time_min'")
        check.is_in("confidence", result, "Missing 'confidence'")
        # Unknown recipe → low confidence expected
        check.less(result.get("confidence", 1), 0.7,
                   "Expected low confidence for unknown ingredients")


@pytest.mark.parametrize("ingredients, field", [
    (["eggs", "flour", "milk"], "recipe_name"),
    (["potato", "salt", "oil"], "time_min"),
    (["tuna", "mayonnaise", "corn"], "confidence"),
])
def test_required_fields_present(ingredients, field):
    """Each known recipe response must contain the specified field."""
    result = recommend_recipe(ingredients, system_prompt="Always return JSON")

    check.is_instance(result, dict, "Response must be a dict")
    if isinstance(result, dict):
        check.is_in(field, result, f"Field '{field}' missing from response")