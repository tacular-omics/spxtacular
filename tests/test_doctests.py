"""Execute every docstring example in the package.

This covers the ``>>>`` examples in ``src/spxtacular/*.py``. The fenced ``python`` blocks in
``docs/*.md`` and the README are a separate surface, executed by :mod:`tests.test_docs`;
neither file replaces the other.
"""

import doctest
import importlib
import pkgutil
import sys
from types import ModuleType
from typing import NoReturn

import spxtacular

# Measured 2026-11-08. Raise these when you add modules or examples; never lower them
# without saying why in the commit message.
MIN_MODULES = 38
MIN_EXAMPLES = 0
# MIN_EXAMPLES is 0 because the one ``>>>`` example in src (mzspeclib.py, opening a library
# file) carries ``# doctest: +SKIP``, and ``attempted`` cannot be relied on to notice it: a
# skipped example counts toward ``attempted`` on Python 3.13 and 3.14 but not on 3.12, which
# CI also tests (measured 2026-11-08: attempted=0 on 3.12.3, 1 on 3.13.7 and 3.14.0rc2).
# DocTestFinder counts examples it parsed, skipped or not, so this floor is the one that
# notices the example being deleted on every version we support.
MIN_EXAMPLES_FOUND = 1

# No module is excluded. Every optional dependency in src is imported lazily or inside a
# ``try/except ImportError``, so the whole walk imports with no extra installed.
EXCLUDED: frozenset[str] = frozenset()


def _reraise(name: str) -> NoReturn:
    """Fail the walk on an unimportable module instead of skipping it silently.

    ``pkgutil.walk_packages`` catches and discards :exc:`ImportError` when ``onerror`` is
    ``None``, which would hide a whole subpackage's examples.
    """
    raise ImportError(f"{name} failed to import during the module walk") from sys.exception()


def _all_modules() -> list[ModuleType]:
    mods: list[ModuleType] = [spxtacular]
    for _finder, name, _ispkg in pkgutil.walk_packages(
        spxtacular.__path__, prefix=f"{spxtacular.__name__}.", onerror=_reraise
    ):
        if name in EXCLUDED:
            continue
        mods.append(importlib.import_module(name))
    return mods


def test_docstring_examples():
    mods = _all_modules()
    assert len(mods) >= MIN_MODULES, f"module walk found only {len(mods)}"
    finder = doctest.DocTestFinder()
    found = attempted = failed = 0
    for mod in mods:
        found += sum(len(test.examples) for test in finder.find(mod))
        result = doctest.testmod(mod, verbose=False)
        attempted += result.attempted
        failed += result.failed
    assert failed == 0, f"{failed} of {attempted} docstring examples failed"
    assert attempted >= MIN_EXAMPLES, f"only {attempted} examples ran, expected >= {MIN_EXAMPLES}"
    assert found >= MIN_EXAMPLES_FOUND, f"only {found} examples were found, expected >= {MIN_EXAMPLES_FOUND}"
