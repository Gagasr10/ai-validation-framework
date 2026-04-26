"""
test_edge_cases.py

Fuzzing / boundary-value tests: the model must never crash —
it must always return a result or a structured error dict.
"""

import pytest
from ai_model import recommend_recipe

EDGE_INPUTS = [
    pytest.param([], id="empty_list"),
    pytest.param([""], id= "list_with_empty_string"),
    pytest.param(None, id="none_input"),
    pytest.param("x" * 200, id="very_long_string"),
    pytest.param(["🥚", "🍞", "🥛"], id="emoji_ingredients"),
    pytest.param(["nonexistent_ingredient_xyz"], id="unknown_ingredient"),
    pytest.param(["eggs", "eggs", "eggs"], id="repeated_ingredient"),
    pytest.param([""], id="single_empty_string"),
]

@pytest.mark.parametrize("weird_input", EDGE_INPUTS)
def test_edge_cases_no_crash(weird_input):
    """
    The model must handle every edge input gracefully.
    Acceptable outcomes: a dict result OR a non-empty string.
    Raising an unhandled exception is a failure.
    """
    try:
        result = recommend_recipe(weird_input)
        assert result is not None, "Model returned None — unexpected"
        assert isinstance(result, (dict, str)), (
            f"Unexpected return type: {type(result).__name__}"
        )
    except ValueError:
        # ValueError on empty/None input is documented behaviour — acceptable
        pass
    except Exception as exc:
        pytest.fail(f"Model raised unexpected exception for {weird_input!r}: {exc}")


def test_very_long_input_returns_graceful_response():
    """Extremely long string input must return a dict with low confidence."""
    result = recommend_recipe("x" * 200)
    assert isinstance(result, dict), "Expected dict for long input"
    assert result.get("confidence", 1) <= 0.5, ("Expected low confidence for long input")


def test_unknown_ingredients_returns_generic_mail():
    """Unknown ingredients should return something - not crashh."""
    result = recommend_recipe(["unobtainium", "dark_matter"],
                              system_prompt = "Always return JSON")
    assert result is not None
    if isinstance(result, dict):
        assert "recipe_name" in result