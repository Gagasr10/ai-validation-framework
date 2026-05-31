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


def pytest_collection_modifyitems(items):
    """Auto-skip real_api tests when ANTHROPIC_API_KEY is not configured."""
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if key and key != "dummy":
        return  # key present — let all tests run
    skip = pytest.mark.skip(reason="ANTHROPIC_API_KEY not set — real API test skipped")
    for item in items:
        if item.get_closest_marker("real_api"):
            item.add_marker(skip)

@pytest.hookimpl(tryfirst=True,hookwrapper=True)
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





