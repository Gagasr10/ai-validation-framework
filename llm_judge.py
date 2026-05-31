"""
llm_judge.py

LLM-as-Judge: evaluates recipe recommendations using a second LLM call.
Supports two judge providers:
  - Anthropic  (Claude Haiku, default)  — requires ANTHROPIC_API_KEY
  - OpenAI     (GPT-4o-mini)            — requires OPENAI_API_KEY or OPENAI_KEY

Public API
----------
judge_response(ingredients, result, model="claude-haiku-4-5-20251001") -> dict

Provider is auto-detected from the model name:
  "gpt-*" / "o1-*" / "o3-*"  → OpenAI
  everything else             → Anthropic

The verdict dict always contains:
  score        float 0-1    overall quality score
  verdict      str          "pass" (score >= 0.70) | "fail"
  reasoning    str          one-sentence explanation from the judge
  dimensions   dict         per-criterion scores (0-1 each)
  mode         str          "real" | "mock"
  judge_model  str | None   model used (None in mock mode)

Falls back to _mock_judge() when the required API key is absent.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

PASS_THRESHOLD = 0.70

_JUDGE_PROMPT = """\
You are a strict quality judge evaluating AI recipe recommendations.

RECOMMENDATION TO EVALUATE
Input ingredients  : {ingredients_str}
Recommended recipe : {recipe_name}
Cooking time       : {time_min} minutes
Model confidence   : {confidence}

SCORING CRITERIA — score each dimension 0.0 to 1.0:
1. relevance             : Does the recipe genuinely use the input ingredients as core components?
2. feasibility           : Is the cooking time realistic for this recipe type?
3. confidence_calibration: Is the confidence score appropriate? \
(high for common recipes, lower for unusual ones)
4. coherence             : Is the recipe name a recognisable, sensible dish?

RULES
- Score 1.0 only when clearly correct; 0.5 when uncertain; 0.0 when clearly wrong.
- overall = relevance*0.4 + feasibility*0.2 + confidence_calibration*0.2 + coherence*0.2
- verdict: "pass" if overall >= {threshold}, else "fail"

Respond ONLY with valid JSON — no markdown fences, no extra text:
{{"score": <float>, "verdict": "<pass|fail>", "reasoning": "<max 20 words>",
  "dimensions": {{"relevance": <float>, "feasibility": <float>,
                  "confidence_calibration": <float>, "coherence": <float>}}}}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_prompt(ingredients: list[str], result: Any) -> str:
    return _JUDGE_PROMPT.format(
        ingredients_str=", ".join(str(i) for i in ingredients),
        recipe_name=result.get("recipe_name", "N/A") if isinstance(result, dict) else "N/A",
        time_min=result.get("time_min", "N/A") if isinstance(result, dict) else "N/A",
        confidence=result.get("confidence", "N/A") if isinstance(result, dict) else "N/A",
        threshold=PASS_THRESHOLD,
    )


def _parse_judge_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Anthropic judge
# ---------------------------------------------------------------------------

def _call_claude_judge(ingredients: list[str], result: Any, model: str) -> dict:
    import anthropic  # deferred import — module loads without the package

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=300,
        system="You are a strict but fair recipe quality judge. Always return valid JSON.",
        messages=[{"role": "user", "content": _build_prompt(ingredients, result)}],
    )
    verdict = _parse_judge_json(response.content[0].text)
    verdict["mode"] = "real"
    verdict["judge_model"] = model
    return verdict


# ---------------------------------------------------------------------------
# OpenAI judge
# ---------------------------------------------------------------------------

def _call_openai_judge(ingredients: list[str], result: Any, model: str) -> dict:
    from openai import OpenAI  # deferred import

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY", "")
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        max_tokens=300,
        response_format={"type": "json_object"},  # GPT-4o-mini native JSON mode
        messages=[
            {
                "role": "system",
                "content": "You are a strict but fair recipe quality judge. Always return valid JSON.",
            },
            {"role": "user", "content": _build_prompt(ingredients, result)},
        ],
    )
    verdict = json.loads(response.choices[0].message.content)
    verdict["mode"] = "real"
    verdict["judge_model"] = model
    return verdict


# ---------------------------------------------------------------------------
# Structural mock judge (no API key required)
# ---------------------------------------------------------------------------

def _mock_judge(ingredients: list[str], result: Any) -> dict:
    """
    Deterministic structural scorer — used when the relevant API key is absent.
    Evaluates schema, types, and value ranges only (no semantic quality check).
    """
    if not isinstance(result, dict) or "error" in result:
        return {
            "score": 0.0,
            "verdict": "fail",
            "reasoning": "Response is not a valid recipe dict.",
            "dimensions": {
                "relevance": 0.0,
                "feasibility": 0.0,
                "confidence_calibration": 0.0,
                "coherence": 0.0,
            },
            "mode": "mock",
            "judge_model": None,
        }

    required = ("recipe_name", "time_min", "confidence", "ingredients")
    schema_ok = all(k in result for k in required)
    types_ok = (
        isinstance(result.get("recipe_name"), str)
        and isinstance(result.get("time_min"), (int, float))
        and isinstance(result.get("confidence"), (int, float))
        and isinstance(result.get("ingredients"), list)
    )
    conf = float(result.get("confidence", -1))
    t = float(result.get("time_min", 0))
    range_ok = 0.0 <= conf <= 1.0 and t > 0
    name_ok = bool(str(result.get("recipe_name", "")).strip())

    dims = {
        "relevance": float(schema_ok and name_ok),
        "feasibility": float(range_ok),
        "confidence_calibration": float(range_ok),
        "coherence": float(types_ok and name_ok),
    }
    score = round(
        dims["relevance"] * 0.4
        + dims["feasibility"] * 0.2
        + dims["confidence_calibration"] * 0.2
        + dims["coherence"] * 0.2,
        4,
    )
    return {
        "score": score,
        "verdict": "pass" if score >= PASS_THRESHOLD else "fail",
        "reasoning": "Structural mock judge: schema/type/range checks only.",
        "dimensions": dims,
        "mode": "mock",
        "judge_model": None,
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def _is_openai_model(model: str) -> bool:
    return model.startswith(("gpt-", "o1", "o3", "o4"))


def judge_response(
    ingredients: list[str],
    result: Any,
    model: str = "claude-haiku-4-5-20251001",
) -> dict:
    """
    Evaluate a recipe recommendation with an LLM judge.

    Provider is auto-detected from the model name.
    Falls back to _mock_judge() when the required API key is absent.
    """
    if _is_openai_model(model):
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY", "")
        if api_key:
            try:
                return _call_openai_judge(ingredients, result, model)
            except Exception as exc:
                logger.warning("OpenAI judge error (%s) — falling back to mock.", exc)
    else:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if api_key and api_key != "dummy":
            try:
                return _call_claude_judge(ingredients, result, model)
            except Exception as exc:
                logger.warning("Claude judge error (%s) — falling back to mock.", exc)

    return _mock_judge(ingredients, result)
