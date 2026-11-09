"""The one gate for the optional spectrl dependency, shared by every test module that needs it.

spectrl is the ``spectrl`` extra. The gate lives here rather than in a test module because two
modules need it -- ``tests/test_spectrl_bridge.py`` and two tests in ``tests/test_ionization.py``
-- and a gate duplicated is a gate that can rot in one copy, which is exactly what happened to
matchms: ``tests/test_ionization.py`` carried its own ``pytest.importorskip`` and so ignored
``SPXTACULAR_REQUIRE_MATCHMS`` until ``tests/matchms_gate.py`` existed.

The other half of that module's reasoning does **not** transfer. matchms must not be imported at
module scope because the first import JIT-compiles pynndescent for 15-20 s; ``import spectrl``
takes about 0.23 s (measured 2026-11-08 with ``python -X importtime``, three runs), so nothing
here is about collection cost, and ``tests/test_spectrl_bridge.py`` keeps its module-scope import.

Two different silences are closed, and they are not the same failure:

* An **absent** spectrl skips, unless :data:`REQUIRE_SPECTRL` is ``"1"``, which makes it a
  failure. This is the case a job that installs the extra and silently fails to get spectrl
  would otherwise report as green.
* An **installed** spectrl outside the range ``pyproject.toml`` declares is always a failure,
  switch or no switch. ``pytest.importorskip`` succeeds for spectrl 4.x, so without this check
  these tests would pass against a token format the cap says is untested: each spectrl major
  changes the format (``spectrl.v3`` today), which is why the cap is there.

The range is read from ``pyproject.toml`` rather than written down here, because it is declared
in three places (the ``spectrl`` extra, the ``all`` extra and the ``dev`` group) and a number
copied into a test is a fourth place to forget. Disagreement between the three is itself a
failure.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

#: Set to "1" to make an absent spectrl a failure instead of a skip, as
#: ``SPXTACULAR_REQUIRE_MATCHMS`` does for matchms and ``SPXTACULAR_REQUIRE_THERMO`` for
#: fisher-py. CI sets it on every job that installs the ``spectrl`` extra and runs the whole
#: suite.
REQUIRE_SPECTRL = "SPXTACULAR_REQUIRE_SPECTRL"

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def declared_specifier(pyproject: Path = _PYPROJECT) -> SpecifierSet:
    """Version range the package declares for spectrl, from ``pyproject.toml``.

    Returns
    -------
    SpecifierSet
        The one range every declaration agrees on.

    Raises
    ------
    AssertionError
        If no declaration is found, or if the declarations disagree. Both mean the cap this
        gate enforces is not the cap the package publishes.
    """
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data["project"]
    declarations: dict[str, set[str]] = {}
    sources: list[tuple[str, list[str]]] = [
        *((f"optional-dependencies.{name}", reqs) for name, reqs in project.get("optional-dependencies", {}).items()),
        *((f"dependency-groups.{name}", reqs) for name, reqs in data.get("dependency-groups", {}).items()),
        ("dependencies", project.get("dependencies", [])),
    ]
    for where, requirements in sources:
        for raw in requirements:
            if not isinstance(raw, str):
                continue
            requirement = Requirement(raw)
            if requirement.name == "spectrl":
                declarations.setdefault(str(requirement.specifier), set()).add(where)

    assert declarations, f"{pyproject} declares no spectrl requirement"
    assert len(declarations) == 1, f"{pyproject} declares spectrl inconsistently: {declarations}"
    return SpecifierSet(next(iter(declarations)))


def _import_spectrl() -> ModuleType:
    import spectrl

    return spectrl


def _installed_version() -> str:
    from importlib.metadata import version

    return version("spectrl")


def spectrl_or_skip(
    importer: Callable[[], ModuleType] = _import_spectrl,
    installed_version: Callable[[], str] = _installed_version,
) -> ModuleType:
    """Return the spectrl module, or skip, or fail, per the two cases in this module's docstring.

    Parameters
    ----------
    importer, installed_version
        Injectable so that the gate tests in ``tests/test_spectrl_gate.py`` can drive the absent
        and out-of-range branches in an environment where an in-range spectrl is installed,
        which is every environment this suite normally runs in.

    Raises
    ------
    ImportError
        spectrl is absent and :data:`REQUIRE_SPECTRL` is ``"1"``.
    Failed
        spectrl is installed but outside the declared range, whatever the switch says.
    """
    try:
        module = importer()
    except ImportError:
        if os.environ.get(REQUIRE_SPECTRL) == "1":
            raise
        # `allow_module_level` because `tests/test_spectrl_bridge.py` calls this at module
        # scope, where a plain `pytest.skip` is a usage error, not a skip. Measured
        # 2026-11-08 in a spectrl-free environment: without it the module is reported as a
        # collection ERROR, which is the same shape as the switch firing.
        pytest.skip(
            f"spectrl is not installed; set {REQUIRE_SPECTRL}=1 to make this a failure",
            allow_module_level=True,
        )

    specifier = declared_specifier()
    found = installed_version()
    if not specifier.contains(Version(found), prereleases=True):
        pytest.fail(
            f"spectrl {found} is installed but out of range: pyproject.toml declares "
            f"spectrl{specifier}, and each spectrl major changes the token format these tests "
            f"pin. (The other case, spectrl absent, is a skip gated by {REQUIRE_SPECTRL}.)"
        )
    return module
