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
from llm_judge import judge_response

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


# ===========================================================================
# Real LLM-as-Judge tests  (require ANTHROPIC_API_KEY — auto-skipped otherwise)
# ===========================================================================

@pytest.mark.real_api
def test_real_judge_verdict_structure():
    """
    judge_response() must always return a dict with all required fields,
    values in valid ranges, and a recognised verdict string.
    """
    result = recommend_recipe(["potato", "salt", "oil"], system_prompt="Always return JSON")
    verdict = judge_response(["potato", "salt", "oil"], result)

    for field in ("score", "verdict", "reasoning", "dimensions", "mode"):
        assert field in verdict, f"Judge verdict missing field '{field}'"

    assert verdict["verdict"] in ("pass", "fail"), f"Unknown verdict: {verdict['verdict']}"
    assert 0.0 <= verdict["score"] <= 1.0, f"Score out of range: {verdict['score']}"
    assert isinstance(verdict["reasoning"], str) and verdict["reasoning"].strip()
    assert isinstance(verdict["dimensions"], dict)
    for dim in ("relevance", "feasibility", "confidence_calibration", "coherence"):
        assert dim in verdict["dimensions"], f"Missing dimension '{dim}'"
        assert 0.0 <= verdict["dimensions"][dim] <= 1.0


@pytest.mark.real_api
def test_real_judge_passes_known_recipe():
    """
    Real Claude judge must give a passing score (>= 0.70) to a correct
    Pancakes recommendation — the clearest possible positive case.
    """
    result = recommend_recipe(["eggs", "flour", "milk"], system_prompt="Always return JSON")
    verdict = judge_response(["eggs", "flour", "milk"], result)

    assert verdict["score"] >= 0.70, (
        f"Judge score too low for Pancakes: {verdict['score']:.2f}\n"
        f"Reasoning: {verdict['reasoning']}\n"
        f"Dimensions: {verdict['dimensions']}"
    )
    assert verdict["verdict"] == "pass", f"Judge returned 'fail' for correct recipe: {verdict}"


@pytest.mark.real_api
@pytest.mark.parametrize("ingredients, expected_name", GOLD_DATA)
def test_real_judge_all_gold_recipes(ingredients, expected_name):
    """
    Real Claude judge must score all 10 golden recipes >= 0.65.
    Uses a slightly lower threshold than the single-recipe test to allow
    for natural variance in LLM scoring across diverse recipe types.
    """
    result = recommend_recipe(ingredients, system_prompt="Always return JSON")
    verdict = judge_response(ingredients, result)

    assert verdict["score"] >= 0.65, (
        f"Judge score too low for {expected_name!r} ({ingredients!r})\n"
        f"score={verdict['score']:.2f}  reasoning={verdict['reasoning']}\n"
        f"dimensions={verdict['dimensions']}"
    )


@pytest.mark.real_api
def test_real_judge_aggregate_quality():
    """
    Mean judge score across all 10 golden recipes must be >= 0.75.
    Runs one batch call to reduce API usage.
    """
    scores = []
    for ingredients, _ in GOLD_DATA:
        result = recommend_recipe(ingredients, system_prompt="Always return JSON")
        verdict = judge_response(ingredients, result)
        scores.append(verdict["score"])

    mean = statistics.mean(scores)
    assert mean >= 0.75, (
        f"Mean judge score {mean:.2%} is below the 75% threshold.\n"
        f"Per-recipe scores: {[f'{s:.2f}' for s in scores]}"
    )


@pytest.mark.real_api
def test_real_judge_rejects_error_response():
    """
    Judge must assign a failing score (< 0.5) to an error response —
    verifies the judge correctly identifies non-recipe output.
    """
    error_result = {"error": "No ingredients provided", "status": 400}
    verdict = judge_response(["eggs", "flour", "milk"], error_result)

    assert verdict["score"] < 0.5, (
        f"Judge should reject error response but scored {verdict['score']:.2f}: {verdict}"
    )
    assert verdict["verdict"] == "fail", (
        f"Judge should return 'fail' for error response, got: {verdict['verdict']}"
    )


@pytest.mark.real_api
def test_real_judge_mode_is_real():
    """Confirm the judge is actually using the real API, not the mock fallback."""
    result = recommend_recipe(["tuna", "mayonnaise", "corn"], system_prompt="Always return JSON")
    verdict = judge_response(["tuna", "mayonnaise", "corn"], result)

    assert verdict["mode"] == "real", (
        "Expected judge to run in 'real' mode — check ANTHROPIC_API_KEY is set correctly"
    )


# ===========================================================================
# GPT-4o-mini as second judge  (require OPENAI_KEY — auto-skipped otherwise)
# SUT response can be from mock or real Claude — the OpenAI judge evaluates
# whatever dict it receives, so these tests only need OPENAI_KEY.
# ===========================================================================

@pytest.mark.openai_judge
def test_openai_judge_verdict_structure(monkeypatch):
    """GPT-4o-mini judge must return a fully structured verdict dict."""
    monkeypatch.setattr(random, "random", lambda: 0.5)
    result = recommend_recipe(["potato", "salt", "oil"], system_prompt="Always return JSON")
    verdict = judge_response(["potato", "salt", "oil"], result, model="gpt-4o-mini")

    for field in ("score", "verdict", "reasoning", "dimensions", "mode"):
        assert field in verdict, f"GPT judge verdict missing field '{field}'"

    assert verdict["verdict"] in ("pass", "fail")
    assert 0.0 <= verdict["score"] <= 1.0
    assert isinstance(verdict["reasoning"], str) and verdict["reasoning"].strip()
    for dim in ("relevance", "feasibility", "confidence_calibration", "coherence"):
        assert dim in verdict["dimensions"], f"Missing dimension '{dim}'"
        assert 0.0 <= verdict["dimensions"][dim] <= 1.0


@pytest.mark.openai_judge
def test_openai_judge_passes_known_recipe(monkeypatch):
    """GPT-4o-mini must give a passing score (>= 0.70) to a correct Pancakes response."""
    monkeypatch.setattr(random, "random", lambda: 0.5)
    result = recommend_recipe(["eggs", "flour", "milk"], system_prompt="Always return JSON")
    verdict = judge_response(["eggs", "flour", "milk"], result, model="gpt-4o-mini")

    assert verdict["score"] >= 0.70, (
        f"GPT judge score too low for Pancakes: {verdict['score']:.2f}\n"
        f"Reasoning: {verdict['reasoning']}\n"
        f"Dimensions: {verdict['dimensions']}"
    )
    assert verdict["verdict"] == "pass"


@pytest.mark.openai_judge
@pytest.mark.parametrize("ingredients, expected_name", GOLD_DATA)
def test_openai_judge_all_gold_recipes(ingredients, expected_name, monkeypatch):
    """GPT-4o-mini must score all 10 golden recipe responses >= 0.65."""
    monkeypatch.setattr(random, "random", lambda: 0.5)
    result = recommend_recipe(ingredients, system_prompt="Always return JSON")
    verdict = judge_response(ingredients, result, model="gpt-4o-mini")

    assert verdict["score"] >= 0.65, (
        f"GPT judge score too low for {expected_name!r} ({ingredients!r})\n"
        f"score={verdict['score']:.2f}  reasoning={verdict['reasoning']}\n"
        f"dimensions={verdict['dimensions']}"
    )


@pytest.mark.openai_judge
def test_openai_judge_rejects_error_response():
    """GPT-4o-mini must assign a failing score (< 0.5) to an error response."""
    error_result = {"error": "No ingredients provided", "status": 400}
    verdict = judge_response(["eggs", "flour", "milk"], error_result, model="gpt-4o-mini")

    assert verdict["score"] < 0.5, (
        f"GPT judge should reject error response but scored {verdict['score']:.2f}"
    )
    assert verdict["verdict"] == "fail"


@pytest.mark.openai_judge
def test_openai_judge_mode_is_real(monkeypatch):
    """Confirm GPT-4o-mini judge is running in real mode, not the mock fallback."""
    monkeypatch.setattr(random, "random", lambda: 0.5)
    result = recommend_recipe(["tuna", "mayonnaise", "corn"], system_prompt="Always return JSON")
    verdict = judge_response(["tuna", "mayonnaise", "corn"], result, model="gpt-4o-mini")

    assert verdict["mode"] == "real", (
        "Expected GPT judge to run in 'real' mode — check OPENAI_KEY is set correctly"
    )
    assert verdict["judge_model"] == "gpt-4o-mini"


# ===========================================================================
# Dual-judge consensus  (require BOTH keys)
# ===========================================================================

@pytest.mark.real_api
@pytest.mark.openai_judge
def test_dual_judge_consensus():
    """
    Claude Haiku and GPT-4o-mini must agree on pass/fail verdict for the
    three clearest golden recipes. Cross-model agreement validates that the
    quality signal is real, not a model-specific quirk.
    """
    CONSENSUS_CASES = [
        (["eggs", "flour", "milk"],   "Pancakes"),
        (["potato", "salt", "oil"],   "French fries"),
        (["tuna", "mayonnaise", "corn"], "Tuna salad"),
    ]

    disagreements = []
    for ingredients, name in CONSENSUS_CASES:
        result = recommend_recipe(ingredients, system_prompt="Always return JSON")

        claude_v = judge_response(ingredients, result, model="claude-haiku-4-5-20251001")
        gpt_v    = judge_response(ingredients, result, model="gpt-4o-mini")

        if claude_v["verdict"] != gpt_v["verdict"]:
            disagreements.append(
                f"  {name}: Claude={claude_v['score']:.2f}({claude_v['verdict']}) "
                f"GPT={gpt_v['score']:.2f}({gpt_v['verdict']})"
            )

    assert not disagreements, (
        "Judges disagree on verdict for:\n" + "\n".join(disagreements)
    )


@pytest.mark.real_api
@pytest.mark.openai_judge
def test_dual_judge_aggregate_scores_close():
    """
    Mean scores from Claude and GPT-4o-mini across all golden recipes
    must be within 0.20 of each other — judges should broadly agree
    on quality level even if exact scores differ.
    """
    claude_scores, gpt_scores = [], []

    for ingredients, _ in GOLD_DATA:
        result = recommend_recipe(ingredients, system_prompt="Always return JSON")
        claude_scores.append(judge_response(ingredients, result, model="claude-haiku-4-5-20251001")["score"])
        gpt_scores.append(judge_response(ingredients, result, model="gpt-4o-mini")["score"])

    claude_mean = statistics.mean(claude_scores)
    gpt_mean    = statistics.mean(gpt_scores)
    delta = abs(claude_mean - gpt_mean)

    assert delta <= 0.20, (
        f"Judge means diverge too much: Claude={claude_mean:.2f}, GPT={gpt_mean:.2f}, "
        f"delta={delta:.2f} (max allowed: 0.20)"
    )
