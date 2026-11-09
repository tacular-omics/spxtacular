"""Tests for the spectrl gate itself, in a module that never imports spectrl.

The gate tests cannot live in ``tests/test_spectrl_bridge.py``, which is where the matchms
equivalents live for matchms: that module's own skip is at module scope, so with spectrl absent
its tests are removed from the collected total, gate tests included, and the tests proving the
gate works would disappear in exactly the environment they are about. Here they run either way.

Both directions of both cases are driven through the injectable hooks of
:func:`tests.spectrl_gate.spectrl_or_skip`, because an in-range spectrl is installed in every
environment this suite normally runs in, so the absent and out-of-range branches are otherwise
unreachable from inside the suite.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest

from tests.spectrl_gate import REQUIRE_SPECTRL, declared_specifier, spectrl_or_skip

_FAKE = ModuleType("spectrl")


def _missing_spectrl() -> ModuleType:
    raise ImportError("No module named 'spectrl'")


def _present_spectrl() -> ModuleType:
    return _FAKE


def _lowest_declared() -> str:
    """The floor the pyproject declares, so this test needs no version written into it."""
    floors = [spec.version for spec in declared_specifier() if spec.operator in {">=", "=="}]
    assert len(floors) == 1, f"expected one floor in the declared range, got {floors}"
    return floors[0]


def test_absent_spectrl_fails_when_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the switch set, an absent spectrl must raise, not skip.

    The ``except pytest.skip.Exception`` is the point of the test: an escaped ``Skipped`` is
    reported as a skip rather than a failure, so a dead gate would read as a green suite.
    """
    monkeypatch.setenv(REQUIRE_SPECTRL, "1")
    try:
        with pytest.raises(ImportError, match="spectrl"):
            spectrl_or_skip(_missing_spectrl)
    except pytest.skip.Exception as exc:
        pytest.fail(f"{REQUIRE_SPECTRL}=1 skipped instead of failing: {exc}")


def test_absent_spectrl_skips_without_the_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(REQUIRE_SPECTRL, raising=False)
    with pytest.raises(pytest.skip.Exception, match="spectrl is not installed"):
        spectrl_or_skip(_missing_spectrl)


@pytest.mark.parametrize("switch", [None, "0", "1"])
def test_out_of_range_spectrl_fails_whatever_the_switch_says(
    monkeypatch: pytest.MonkeyPatch, switch: str | None
) -> None:
    """An installed-but-out-of-range spectrl is a failure, and the switch is not its gate.

    ``pytest.importorskip`` succeeds for spectrl 4.x, so this is the silence the switch cannot
    close. The ``Skipped`` catch is here for the same reason as in the test above.
    """
    if switch is None:
        monkeypatch.delenv(REQUIRE_SPECTRL, raising=False)
    else:
        monkeypatch.setenv(REQUIRE_SPECTRL, switch)
    try:
        with pytest.raises(pytest.fail.Exception) as excinfo:
            spectrl_or_skip(_present_spectrl, lambda: "4.0.0")
    except pytest.skip.Exception as exc:
        pytest.fail(f"an out-of-range spectrl skipped instead of failing: {exc}")
    message = str(excinfo.value)
    assert "4.0.0" in message
    assert str(declared_specifier()) in message


def test_in_range_spectrl_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(REQUIRE_SPECTRL, raising=False)
    assert spectrl_or_skip(_present_spectrl, _lowest_declared) is _FAKE


def test_declared_range_covers_the_installed_version() -> None:
    """The real pyproject is parsed, and the version actually installed satisfies it."""
    spectrl = spectrl_or_skip()
    assert spectrl.__name__ == "spectrl"


def test_declarations_that_disagree_are_a_failure(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "x"\ndependencies = []\n'
        '[project.optional-dependencies]\nspectrl = ["spectrl>=3.0,<4"]\n'
        'all = ["spectrl>=3.0,<5"]\n',
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="declares spectrl inconsistently"):
        declared_specifier(pyproject)


def test_no_declaration_at_all_is_a_failure(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "x"\ndependencies = ["numpy"]\n', encoding="utf-8")
    with pytest.raises(AssertionError, match="declares no spectrl requirement"):
        declared_specifier(pyproject)
