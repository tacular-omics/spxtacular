"""Shared test configuration.

The default run is kept fast. Two switches restore the full, slower run (CI uses both):

- ``--run-slow`` (or ``RUN_SLOW=1``) runs tests marked ``@pytest.mark.slow``;
- ``HYPOTHESIS_PROFILE=thorough`` runs 300 hypothesis examples per test instead of 30
  (``exhaustive`` runs 2000; give it a longer ``--timeout``).
"""

from __future__ import annotations

import os

import pytest
from hypothesis import HealthCheck, settings

settings.register_profile(
    "default",
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
# CI: at least the per-test counts the suite used before the fast default (60-300),
# and still inside the 15 s per-test timeout.
settings.register_profile(
    "thorough",
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
# Manual soak run; needs a longer timeout, e.g. --timeout=600.
settings.register_profile(
    "exhaustive",
    max_examples=2000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-slow", action="store_true", default=False, help="also run tests marked slow")


def _run_slow(config: pytest.Config) -> bool:
    return bool(config.getoption("--run-slow")) or os.environ.get("RUN_SLOW") == "1"


def pytest_configure(config: pytest.Config) -> None:
    # Modules whose *import* is slow (test_matchms_interop.py) check this at collection.
    if _run_slow(config):
        os.environ["RUN_SLOW"] = "1"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _run_slow(config):
        return
    skip_slow = pytest.mark.skip(reason="slow: pass --run-slow or set RUN_SLOW=1")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
