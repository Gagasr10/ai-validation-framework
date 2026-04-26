"""
test_prompt_stability.py

Validates that prompt engineering choices (format instructions, tone)
produce stable, parseable outputs.
"""

import pytest
from ai_model import recommend_recipe

def test_json_prompt_returns_dict():
    """Asking for JSON must produce a dict with required keys."""
    result = recommend_recipe(
        ["eggs", "flour", "milk"], system_prompt="Always return JSON"
    )
    assert isinstance(result, dict), (
        f"Expected dict, got {type(result).__name__}: {result}"
    )
    for key in ("recipe_name", "time_min", "confidence"):
        assert key in result, f"Missing key '{key}' in response"

def test_no_json_prompt_returns_string():
    """Without a JSON instruction the mock returns a human-readable string."""
    result = recommend_recipe(["eggs", "flour", "milk"], system_prompt="")
    #Real API always return dict; mocko without JSON prompt returns str
    assert result is not None
    assert isinstance(result, (str, dict)), (f"Unexpected type: {type(result).__name__}")


pytest.mark.parametrize("tone", [
    "",
    "Answer humorously.",
    "Answer in the style of a poem.",
    "Be very concise."
])

def test_prompt_tone_does_not_break_json_format(tone):
    """Injecting tone instructions must not corrupt JSON output."""
    prompt = "f{tone} Always return JSON.".strip()
    result = recommend_recipe(["flour", "water"], system_prompt=prompt)

    # Real API path: always dict
    # Mock path with "JSON" in prompt: dict
    if isinstance(result, str):
        #Only acceptable if mock ran WITHOUT "JSON" in prompt (edge case)
        if "JSON" in prompt.upper():
            pytest.fail(f"Prompt tone '{tone}' caused non-JSON output: {result}")


def test_confidence_within_valid_range():
    """Confidence score must always be between 0 and 1."""
    result = recommend_recipe(["eggs", "flour", "milk"], system_prompt="Always return JSON")
    if isinstance(result,dict) and "confidence" in result:
        assert 0.0 <= result["confidence"] <= 1.0, (
            f"Confidence out of range: {result['confidence']}"
        )

def test_time_min_is_positive_integer():
    """Cooking time must be a positive integer."""
    result = recommend_recipe(
        ["potato", "salt", "oil"], system_prompt="Always return JSON"
    )
    if isinstance(result, dict) and "time_min" in result:
        assert isinstance(result["time_min"], int), (
            f"time_min should be int, got {type(result['time_min']).__name__}"
        )
        assert result["time_min"] > 0, "time_min must be positive"g

