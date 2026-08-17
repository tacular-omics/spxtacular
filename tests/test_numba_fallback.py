"""
Tests that deconvolve_spectrum works correctly without numba.

numba is simulated as absent by patching sys.modules and reloading the decon
modules, then genuinely restored afterwards.

The restore matters more than it looks.  ``importlib.reload`` re-executes a
module into its *existing* ``__dict__``, and ``core._deconvolve`` is a reference
to the function object living in that dict — so a reload with numba blocked
swaps the JIT kernel out from under every later test in the session.  Restoring
by rewriting ``sys.modules`` does nothing (the dict was mutated in place); the
modules have to be reloaded a second time with numba visible again.
"""

import importlib
import sys
from contextlib import contextmanager

import numpy as np
import pytest


@contextmanager
def numba_blocked():
    """Reload the decon modules with numba absent, then genuinely restore them."""
    import spxtacular.decon.greedy as greedy_mod
    import spxtacular.decon.scored as scored_mod

    real_numba = sys.modules.get("numba")
    sys.modules["numba"] = None  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
    try:
        importlib.reload(greedy_mod)
        importlib.reload(scored_mod)
        yield scored_mod.deconvolve_spectrum
    finally:
        if real_numba is None:
            sys.modules.pop("numba", None)
        else:
            sys.modules["numba"] = real_numba
        importlib.reload(greedy_mod)
        importlib.reload(scored_mod)


def _two_cluster_spectrum() -> tuple[np.ndarray, np.ndarray]:
    """A 2+ and a 1+ isotope cluster plus an isolated singleton."""
    mz = np.array(
        [400.00, 400.5017, 401.0033, 401.5050, 700.00, 701.0033, 702.0067, 900.12],
        dtype=np.float64,
    )
    intensity = np.array(
        [100000.0, 62000.0, 24000.0, 7000.0, 80000.0, 35000.0, 9000.0, 5000.0],
        dtype=np.float64,
    )
    return mz, intensity


# ---------------------------------------------------------------------------
# Without numba
# ---------------------------------------------------------------------------


def test_deconvolve_works_without_numba() -> None:
    """deconvolve_spectrum must produce valid output even when numba is absent."""
    mz = np.array([500.0, 500.501, 501.002], dtype=np.float64)
    intensity = np.array([100000.0, 70000.0, 30000.0], dtype=np.float64)

    with numba_blocked() as deconvolve_spectrum:
        result = deconvolve_spectrum(mz, intensity, charge_range=(1, 3), tolerance=50.0, is_ppm=True)

    assert len(result) == 4
    assert len(result[0]) > 0


def test_numba_and_pure_python_produce_identical_results() -> None:
    """The JIT and fallback paths must agree exactly — not merely both 'work'."""
    mz, intensity = _two_cluster_spectrum()

    with numba_blocked() as pure_deconvolve:
        pure = pure_deconvolve(mz, intensity, charge_range=(1, 4), tolerance=20.0, is_ppm=True)

    from spxtacular.decon.scored import deconvolve_spectrum as current

    jitted = current(mz, intensity, charge_range=(1, 4), tolerance=20.0, is_ppm=True)

    for name, a, b in zip(("mz", "charge", "intensity", "score"), pure, jitted, strict=True):
        assert len(a) == len(b), f"{name}: length differs between backends"
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=0.0, err_msg=f"{name} differs between backends")


def test_blocking_numba_does_not_leak_into_later_tests() -> None:
    """After the context manager exits, the real backend must be live again.

    Without this, whichever implementation gets exercised depends on test
    collection order, and the JIT path silently stops being covered at all.
    """
    numba = pytest.importorskip("numba")

    import spxtacular.decon.scored as scored_mod

    before = type(scored_mod._match_apex_cluster).__name__
    with numba_blocked():
        pass
    after = type(scored_mod._match_apex_cluster).__name__

    assert after == before, f"backend leaked: {before!r} -> {after!r}"
    assert isinstance(scored_mod._match_apex_cluster, numba.core.registry.CPUDispatcher)


# ---------------------------------------------------------------------------
# Core correctness — with the currently installed environment (numba present or not)
# ---------------------------------------------------------------------------


def test_empty_spectrum_returns_four_empty_arrays() -> None:
    from spxtacular.decon.scored import deconvolve_spectrum

    result = deconvolve_spectrum(
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        charge_range=(1, 3),
        tolerance=50.0,
        is_ppm=True,
    )
    assert len(result) == 4
    for arr in result:
        assert len(arr) == 0


def test_result_is_4_tuple() -> None:
    from spxtacular.decon.scored import deconvolve_spectrum

    mz = np.array([500.0, 500.501, 501.002], dtype=np.float64)
    intensity = np.array([100000.0, 70000.0, 30000.0], dtype=np.float64)
    result = deconvolve_spectrum(mz, intensity, charge_range=(1, 3), tolerance=50.0, is_ppm=True)

    assert len(result) == 4


def test_result_arrays_have_same_length() -> None:
    from spxtacular.decon.scored import deconvolve_spectrum

    mz = np.array([500.0, 500.501, 501.002, 800.0], dtype=np.float64)
    intensity = np.array([100000.0, 70000.0, 30000.0, 5000.0], dtype=np.float64)
    mz_out, charges_out, intensity_out, scores_out = deconvolve_spectrum(
        mz, intensity, charge_range=(1, 3), tolerance=50.0, is_ppm=True
    )

    n = len(mz_out)
    assert len(charges_out) == n
    assert len(intensity_out) == n
    assert len(scores_out) == n


def test_result_mz_sorted_ascending() -> None:
    from spxtacular.decon.scored import deconvolve_spectrum

    mz = np.array([500.0, 500.501, 501.002, 200.0, 201.0], dtype=np.float64)
    intensity = np.array([100000.0, 70000.0, 30000.0, 5000.0, 4000.0], dtype=np.float64)
    mz_out, _, _, _ = deconvolve_spectrum(mz, intensity, charge_range=(1, 3), tolerance=50.0, is_ppm=True)

    assert list(mz_out) == sorted(mz_out.tolist())


def test_scores_in_zero_to_one_range() -> None:
    from spxtacular.decon.scored import deconvolve_spectrum

    mz = np.array([500.0, 500.501, 501.002], dtype=np.float64)
    intensity = np.array([100000.0, 70000.0, 30000.0], dtype=np.float64)
    _, _, _, scores_out = deconvolve_spectrum(mz, intensity, charge_range=(1, 3), tolerance=50.0, is_ppm=True)

    assert np.all(scores_out >= 0.0)
    assert np.all(scores_out <= 1.0)


def test_singletons_have_charge_minus_one() -> None:
    from spxtacular.decon.scored import deconvolve_spectrum

    # Single isolated peak — no isotope cluster possible
    mz = np.array([500.0], dtype=np.float64)
    intensity = np.array([100000.0], dtype=np.float64)
    _, charges_out, _, scores_out = deconvolve_spectrum(mz, intensity, charge_range=(1, 3), tolerance=50.0, is_ppm=True)

    assert len(charges_out) == 1
    assert charges_out[0] == -1
    assert scores_out[0] == pytest.approx(0.0)


def test_da_tolerance_mode() -> None:
    from spxtacular.decon.scored import deconvolve_spectrum

    mz = np.array([500.0, 500.501, 501.002], dtype=np.float64)
    intensity = np.array([100000.0, 70000.0, 30000.0], dtype=np.float64)
    result = deconvolve_spectrum(mz, intensity, charge_range=(1, 3), tolerance=0.01, is_ppm=False)

    assert len(result) == 4
    assert len(result[0]) > 0
