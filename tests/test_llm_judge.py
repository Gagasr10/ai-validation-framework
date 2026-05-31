"""
test_llm_judge.py

Statistical LLM-as-judge: scores every response against a multi-dimension
rubric and asserts that aggregate quality meets defined thresholds.

Scoring rubric (each dimension 0.0 or 1.0):
  schema_ok      — all four required keys are present
  types_ok       — recipe_name: str, time_min: int|float, confidence: float, ingredients: list
  range_ok       — confidence ∈ [0.0, 1.0]  AND  time_min > 0
  name_nonempty  — recipe_name is a non-empty, non-whitespace string
  no_injection   — recipe_name does not contain known jailbreak indicators

Composite score = mean of all five dimensions (0.0–1.0).

Suite thresholds:
  • Mean composite across golden dataset  >= 0.85
  • Per-response composite               >= 0.60
  • Per-dimension mean                   >= 0.75
"""

from __future__ import annotations

import random
import statistics
from typing import Any

import pytest
from ai_model import recommend_recipe
from golden_dataset import GOLD_DATA

GOLD_INPUTS: list[list[str]] = [ingredients for ingredients, _ in GOLD_DATA]

REQUIRED_KEYS = ("recipe_name", "time_min", "confidence", "ingredients")
BANNED_TOKENS = ("hacked", "pwned", "drop table", "alert(", "exception")


# ---------------------------------------------------------------------------
# Rubric
# ---------------------------------------------------------------------------

def score_response(result: Any) -> dict[str, float]:
    """
    Score a single model response on five dimensions.
    Non-dict responses fail all dimensions (0.0 each).
    """
    zero = {k: 0.0 for k in ("schema_ok", "types_ok", "range_ok", "name_nonempty", "no_injection")}

    if not isinstance(result, dict):
        return zero

    schema_ok = float(all(k in result for k in REQUIRED_KEYS))

    types_ok = float(
        isinstance(result.get("recipe_name"), str)
        and isinstance(result.get("time_min"), (int, float))
        and isinstance(result.get("confidence"), (int, float))
        and isinstance(result.get("ingredients"), list)
    )

    confidence = float(result.get("confidence", -1))
    time_min = float(result.get("time_min", 0))
    range_ok = float(0.0 <= confidence <= 1.0 and time_min > 0)

    name = result.get("recipe_name", "") or ""
    name_nonempty = float(bool(name.strip()))

    no_injection = float(
        not any(tok in name.lower() for tok in BANNED_TOKENS)
    )

    return {
        "schema_ok": schema_ok,
        "types_ok": types_ok,
        "range_ok": range_ok,
        "name_nonempty": name_nonempty,
        "no_injection": no_injection,
    }


def composite(scores: dict[str, float]) -> float:
    return statistics.mean(scores.values())


# ---------------------------------------------------------------------------
# Helper — collect scores across the full golden dataset
# ---------------------------------------------------------------------------

def _collect_all_scores(monkeypatch) -> list[dict[str, float]]:
    monkeypatch.setattr(random, "random", lambda: 0.5)  # neutralise 10% mock flakiness
    all_scores = []
    for ingredients in GOLD_INPUTS:
        result = recommend_recipe(ingredients, system_prompt="Always return JSON")
        all_scores.append(score_response(result))
    return all_scores


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_judge_mean_composite_above_threshold(monkeypatch):
    """
    Mean composite score across all golden inputs must be >= 0.85.
    A lower value indicates widespread schema/type/range violations.
    """
    all_scores = _collect_all_scores(monkeypatch)
    composites = [composite(s) for s in all_scores]
    mean = statistics.mean(composites)

    assert mean >= 0.85, (
        f"Mean judge score {mean:.2%} is below the 85% threshold.\n"
        f"Per-call composites: {[f'{c:.2f}' for c in composites]}"
    )


@pytest.mark.parametrize("ingredients, expected_name", GOLD_DATA)
def test_judge_per_response_quality(ingredients, expected_name, monkeypatch):
    """Every golden response must score >= 0.60 composite."""
    monkeypatch.setattr(random, "random", lambda: 0.5)
    result = recommend_recipe(ingredients, system_prompt="Always return JSON")
    scores = score_response(result)
    c = composite(scores)

    assert c >= 0.60, (
        f"Response quality too low for {ingredients!r}.\n"
        f"composite={c:.2f}  breakdown={scores}\n"
        f"result={result}"
    )


def test_judge_dimension_averages(monkeypatch):
    """
    Each rubric dimension must average >= 0.75 across the golden dataset.
    Reports all failing dimensions before asserting.
    """
    all_scores = _collect_all_scores(monkeypatch)
    failures = []

    for dim in ("schema_ok", "types_ok", "range_ok", "name_nonempty", "no_injection"):
        avg = statistics.mean(s[dim] for s in all_scores)
        if avg < 0.75:
            failures.append(
                f"  '{dim}': {avg:.2%}  (values: {[s[dim] for s in all_scores]})"
            )

    assert not failures, (
        "The following rubric dimensions are below the 75% threshold:\n"
        + "\n".join(failures)
    )


def test_judge_confidence_never_out_of_range(monkeypatch):
    """Confidence must be in [0.0, 1.0] for every golden response."""
    monkeypatch.setattr(random, "random", lambda: 0.5)
    for ingredients in GOLD_INPUTS:
        result = recommend_recipe(ingredients, system_prompt="Always return JSON")
        if isinstance(result, dict) and "confidence" in result:
            c = result["confidence"]
            assert 0.0 <= c <= 1.0, (
                f"Confidence {c!r} is out of range for {ingredients!r}"
            )


def test_judge_time_min_always_positive(monkeypatch):
    """time_min must be a positive number for every golden response."""
    monkeypatch.setattr(random, "random", lambda: 0.5)
    for ingredients in GOLD_INPUTS:
        result = recommend_recipe(ingredients, system_prompt="Always return JSON")
        if isinstance(result, dict) and "time_min" in result:
            t = result["time_min"]
            assert isinstance(t, (int, float)), (
                f"time_min {t!r} is not numeric for {ingredients!r}"
            )
            assert t > 0, f"time_min {t!r} is not positive for {ingredients!r}"


def test_judge_score_function_unit():
    """Unit test the scorer itself against known good and bad inputs."""
    perfect = {
        "recipe_name": "Pancakes",
        "time_min": 15,
        "confidence": 0.85,
        "ingredients": ["eggs", "flour", "milk"],
    }
    scores = score_response(perfect)
    assert composite(scores) == 1.0, f"Perfect response scored < 1.0: {scores}"

    missing_key = {"recipe_name": "X", "time_min": 5}
    scores_bad = score_response(missing_key)
    assert scores_bad["schema_ok"] == 0.0, "schema_ok should be 0 for incomplete dict"

    not_a_dict = "some string response"
    scores_str = score_response(not_a_dict)
    assert all(v == 0.0 for v in scores_str.values()), (
        "Non-dict response should score 0.0 on all dimensions"
    )
