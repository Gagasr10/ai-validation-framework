"""
test_performance.py

Validates that the AI model meets response-time SLAs.
"""

import time

import pytest
from ai_model import recommend_recipe

SLA_MS = 5000  # 5 s — generous for real API; mock is well under 1 s


def test_response_time_sla():
    """Single call must complete within the defined SLA."""
    start = time.perf_counter()
    result = recommend_recipe(["eggs", "flour", "milk"], system_prompt="Always return JSON")
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < SLA_MS, (
        f"Response too slow: {elapsed_ms:.0f} ms  (SLA: {SLA_MS} ms)"
    )
    assert isinstance(result, dict), "Expected dict response"


def test_response_time_benchmark(benchmark):
    """pytest-benchmark: measures average latency over multiple runs."""
    def call():
        return recommend_recipe(["eggs", "flour", "milk"], system_prompt="Always return JSON")

    result = benchmark(call)
    assert isinstance(result, dict)


def test_concurrent_calls_within_sla():
    """Three sequential calls must each stay within SLA (no cumulative drift)."""
    ingredients_list = [
        ["eggs", "flour", "milk"],
        ["potato", "salt", "oil"],
        ["tuna", "mayonnaise", "corn"],
    ]
    for ingredients in ingredients_list:
        start = time.perf_counter()
        recommend_recipe(ingredients, system_prompt="Always return JSON")
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < SLA_MS, (
            f"Call for {ingredients} exceeded SLA: {elapsed_ms:.0f} ms"
        )
