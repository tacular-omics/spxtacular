"""The one gate for the optional matchms dependency, shared by every test module that needs it.

matchms is the ``interop`` extra. Importing it pulls in pynndescent, which JIT-compiles numba
kernels at import time (15-20 s), so the import must not sit at module scope in a test module:
that cost lands at collection, for every run, whether or not any matchms test is selected.

It lives here rather than in one of the test modules because two modules gate on it, and a gate
duplicated is a gate that can rot in one copy: ``tests/test_ionization.py`` carried its own
``pytest.importorskip`` and so ignored :data:`REQUIRE_MATCHMS` until this module existed.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from typing import Any, NamedTuple

import pytest

#: Set to "1" to make an absent matchms a failure instead of a skip, as
#: ``SPXTACULAR_REQUIRE_THERMO`` does for fisher-py in ``tests/test_thermo.py``. CI sets it on
#: every job that installs the ``interop`` extra and runs the whole suite.
REQUIRE_MATCHMS = "SPXTACULAR_REQUIRE_MATCHMS"


class MatchmsApi(NamedTuple):
    """The matchms symbols the tests build objects with.

    Typed ``Any`` deliberately: annotating them precisely would need ``import matchms`` at
    module scope, which is the collection cost this module exists to avoid.
    """

    spectrum: Any
    normalize_intensities: Any


def import_matchms() -> MatchmsApi:
    """Import matchms, raising :class:`ImportError` when it is not installed.

    psims, which matchms imports, warns when the optional hdf5plugin is missing; under
    ``filterwarnings = error`` that failed whichever test imported matchms first, so the import
    is made here with that one warning suppressed.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="hdf5plugin is missing", category=UserWarning)
        import matchms
        from matchms.filtering import normalize_intensities

    return MatchmsApi(spectrum=matchms.Spectrum, normalize_intensities=normalize_intensities)


def matchms_api_or_skip(importer: Callable[[], MatchmsApi] = import_matchms) -> MatchmsApi:
    """Return the matchms symbols, skipping unless :data:`REQUIRE_MATCHMS` asks for a failure.

    ``importer`` is injectable so that the gate tests in ``tests/test_matchms_interop.py`` can
    drive the absent-matchms branch in an environment where matchms is installed, which is every
    environment this suite normally runs in.
    """
    try:
        return importer()
    except ImportError:
        if os.environ.get(REQUIRE_MATCHMS) == "1":
            raise
        pytest.skip(f"matchms is not installed; set {REQUIRE_MATCHMS}=1 to make this a failure")
