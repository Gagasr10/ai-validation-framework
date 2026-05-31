"""
conftest.py

Shared pytest configuration and fixtures.
"""

import os
from datetime import datetime

import pytest
from golden_dataset import GOLD_DATA

# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "smoke: fast, critical-path tests")
    config.addinivalue_line("markers", "regression: full regression suite")
    config.addinivalue_line("markers", "slow: tests that take > 2 s")
    config.addinivalue_line(
        "markers",
        "real_api: requires ANTHROPIC_API_KEY — skipped automatically when absent",
    )
    config.addinivalue_line(
        "markers",
        "openai_judge: requires OPENAI_KEY — skipped automatically when absent",
    )


def pytest_collection_modifyitems(items):
    """Layer 1: skip real_api / openai_judge tests immediately when key is absent."""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY", "")

    has_anthropic = bool(anthropic_key and anthropic_key != "dummy")
    has_openai = bool(openai_key)

    skip_anthropic = pytest.mark.skip(reason="ANTHROPIC_API_KEY not set")
    skip_openai = pytest.mark.skip(reason="OPENAI_KEY not set")

    for item in items:
        if item.get_closest_marker("real_api") and not has_anthropic:
            item.add_marker(skip_anthropic)
        if item.get_closest_marker("openai_judge") and not has_openai:
            item.add_marker(skip_openai)


# ---------------------------------------------------------------------------
# Session-level API health probes
# Cached so the real API call is made only ONCE per session regardless of
# how many real_api / openai_judge tests are collected.
# ---------------------------------------------------------------------------

_api_health: dict[str, dict] = {}


def _probe_anthropic() -> dict:
    """Make the smallest possible Anthropic API call to confirm key + credits work."""
    try:
        import anthropic
        anthropic.Anthropic().messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=5,
            messages=[{"role": "user", "content": "hi"}],
        )
        return {"ok": True, "reason": None}
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


def _probe_openai() -> dict:
    """Make the smallest possible OpenAI API call to confirm key + credits work."""
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY", "")
        OpenAI(api_key=api_key).chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=5,
            messages=[{"role": "user", "content": "hi"}],
        )
        return {"ok": True, "reason": None}
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


def pytest_runtest_setup(item):
    """
    Layer 2: before the first real_api / openai_judge test in the session,
    fire a live API probe and cache the result. Every subsequent test of the
    same type reads from cache (no extra API calls). If the probe fails,
    the test is skipped with the exact error — e.g. 'credit balance too low'.
    """
    if item.get_closest_marker("real_api"):
        if "anthropic" not in _api_health:
            print("\n[api-probe] Checking Anthropic API connectivity...", flush=True)
            _api_health["anthropic"] = _probe_anthropic()
            status = "OK ✓" if _api_health["anthropic"]["ok"] else f"FAILED — {_api_health['anthropic']['reason']}"
            print(f"[api-probe] Anthropic: {status}", flush=True)
        if not _api_health["anthropic"]["ok"]:
            pytest.skip(f"Anthropic API not reachable: {_api_health['anthropic']['reason']}")

    if item.get_closest_marker("openai_judge"):
        if "openai" not in _api_health:
            print("\n[api-probe] Checking OpenAI API connectivity...", flush=True)
            _api_health["openai"] = _probe_openai()
            status = "OK ✓" if _api_health["openai"]["ok"] else f"FAILED — {_api_health['openai']['reason']}"
            print(f"[api-probe] OpenAI: {status}", flush=True)
        if not _api_health["openai"]["ok"]:
            pytest.skip(f"OpenAI API not reachable: {_api_health['openai']['reason']}")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Print a timestamped result line for every test call phase."""
    outcome = yield
    report = outcome.get_result()
    if report.when == "call":
        status = report.outcome.upper()
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {status:>6}  {item.name}")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def api_mode() -> str:
    """
        Returns 'real' when ANTHROPIC_API_KEY is present, 'mock' otherwise.
        Tests can use this to skip or adjust assertions.
    """
    key = os.getenv("ANTHROPIC_API_KEY", "")
    mode = "real" if (key and key != "dummy") else "mock"
    print(f"\n[conftest] Running in {mode.upper()} API mode")
    return mode

@pytest.fixture
def known_ingredients():
    """Three representative gold entries — kept for backward compatibility."""
    return [
        (["eggs", "flour", "milk"], "Pancakes"),
        (["potato", "salt", "oil"], "French fries"),
        (["tuna", "mayonnaise", "corn"], "Tuna salad"),
    ]


@pytest.fixture(scope="session")
def golden_dataset():
    """Full gold-standard dataset — 10 ingredient→recipe pairs."""
    return GOLD_DATA





