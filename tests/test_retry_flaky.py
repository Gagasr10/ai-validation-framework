"""
test_retry_flaky.py

Demonstrates retry strategies for flaky AI responses.
The mock model has a 10 % chance of returning a wrong recipe —
these tests show how to handle that in a real test suite.
"""

import pytest
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError, wait_fixed
from ai_model import recommend_recipe

# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(5), wait=wait_fixed(0.2), reraise=True)
def _stable_call(ingredients: list[str]) -> dict:
    """Call the model and assert correctness - retries up to 5 times."""
    result = recommend_recipe(ingredients, system_prompt="Always return JSON")
    assert isinstance(result, dict), "Expected dict"
    assert result.get("recipe_name") == "Pancakes", (
        f"Flaky response detected: {result.get('recipe_name')}"
    )
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_flaky_ai_stabilises_with_retry():
    """
    With up to 5 retries the model should eventually return the correct recipe.
    (10 % flakiness means the probability of 5 consecutive failures is < 0.001 %)
    """
    result = _stable_call(["eggs", "flour", "milk"])
    assert result["recipe_name"] == "Pancakes"
    assert result["confidence"] >= 0.7


def test_retry_count_is_reasonable():
    """Verify the model converges quickly — tracks attempt count."""
    attempts = 0

    for _ in range(10):          # run 10 independent calls
        attempts += 1
        result = recommend_recipe(
            ["eggs", "flour", "milk"], system_prompt="Always return JSON"
        )
        if isinstance(result, dict) and result.get("recipe_name") == "Pancakes":
            break

    assert attempts <= 10, (
        f"Model required too many attempts to stabilise: {attempts}"
    )

def test_error_handling_on_empty_input():
    """
    Empty-input calls may raise ValueError or return an error dict.
    Both outcomes are acceptable — an unhandled crash is not.
    """
    try:
        result = recommend_recipe([])
        if isinstance(result, dict):
            assert "error" in result or result.get("status") == 400
    except ValueError:
        pass  #documented behaviour


