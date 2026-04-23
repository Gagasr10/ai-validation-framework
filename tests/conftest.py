"""
conftest.py

Shared pytest configuration and fixtures.
"""

import os
from datetime import datetime

import pytest

# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line("markers", "smoke: fast, critical-path tests" )
    config.addinivalue_line("markers", "regression: full regression suite")
    config.addinivalue_line("markers", "slow: tests that take > 2 s")

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
    """A set of ingredient lists that have deterministic gold answers."""
    return [
        (["eggs", "flour", "milk"], "Pancakes"),
        (["potato", "salt", "oil"], "French fries"),
        (["tuna", "mayonnaise", "corn"], "Tuna salad"),
    ]





