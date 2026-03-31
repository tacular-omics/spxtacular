"""
Tests that deconvolve_spectrum works correctly without numba.

numba is simulated as absent by patching sys.modules and reloading the
decon modules.  Each test reloads fresh to avoid cross-test contamination.
"""

import importlib
import sys

import numpy as np
import pytest


def _reload_deconvolve():
    """Reload greedy and scored with numba blocked; return deconvolve_spectrum."""
    with_numba_blocked = dict(sys.modules)
    with_numba_blocked["numba"] = None  # type: ignore[assignment]  # ty: ignore[invalid-assignment]

    import spxtacular.decon.greedy as greedy_mod
    import spxtacular.decon.scored as scored_mod

    # Temporarily suppress numba from modules then reload
    saved = sys.modules.copy()
    sys.modules["numba"] = None  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
    try:
        importlib.reload(greedy_mod)
        importlib.reload(scored_mod)
        from spxtacular.decon.scored import deconvolve_spectrum

        return deconvolve_spectrum
    finally:
        # Restore original module state
        sys.modules.clear()
        sys.modules.update(saved)


# ---------------------------------------------------------------------------
# Without numba
# ---------------------------------------------------------------------------


def test_deconvolve_works_without_numba() -> None:
    """deconvolve_spectrum must produce valid output even when numba is absent."""
    deconvolve_spectrum = _reload_deconvolve()

    mz = np.array([500.0, 500.501, 501.002], dtype=np.float64)
    intensity = np.array([100000.0, 70000.0, 30000.0], dtype=np.float64)
    result = deconvolve_spectrum(mz, intensity, charge_range=(1, 3), tolerance=50.0, is_ppm=True)

    assert len(result) == 4
    assert len(result[0]) > 0


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
