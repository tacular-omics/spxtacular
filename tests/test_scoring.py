"""
Tests for spxtacular.scoring.score.

Fragments are mocked with MagicMock — score() only accesses .mz, .ion_type,
and .position through match_fragments and the internal helpers.
"""
from unittest.mock import MagicMock

import numpy as np
import pytest

from spxtacular.core import Spectrum
from spxtacular.scoring import score

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXPECTED_KEYS = {
    "hyperscore",
    "probability_score",
    "total_matched_intensity",
    "matched_fraction",
    "intensity_fraction",
    "mean_ppm_error",
    "spectral_angle",
    "longest_run",
}


def _make_frag(mz: float, ion_type: str = "b", position: int = 1) -> MagicMock:
    f = MagicMock()
    f.mz = mz
    f.ion_type = ion_type
    f.position = position
    return f


def _spectrum() -> Spectrum:
    mz = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float64)
    intensity = np.array([10.0, 50.0, 20.0, 15.0], dtype=np.float64)
    return Spectrum(mz=mz, intensity=intensity)


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------


def test_score_returns_all_expected_keys() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="Da")
    assert set(result.keys()) == _EXPECTED_KEYS


def test_score_all_values_are_floats() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="Da")
    for key, val in result.items():
        assert isinstance(val, float), f"{key} is not float"


# ---------------------------------------------------------------------------
# No-match baseline
# ---------------------------------------------------------------------------


def test_score_no_matches_hyperscore_is_zero() -> None:
    spec = _spectrum()
    frag = _make_frag(999.0)  # far outside spectrum
    result = score(spec, [frag], tolerance=0.02, tolerance_type="Da")
    assert result["hyperscore"] == pytest.approx(0.0)


def test_score_no_matches_total_matched_intensity_is_zero() -> None:
    spec = _spectrum()
    frag = _make_frag(999.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="Da")
    assert result["total_matched_intensity"] == pytest.approx(0.0)


def test_score_no_matches_matched_fraction_is_zero() -> None:
    spec = _spectrum()
    frag = _make_frag(999.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="Da")
    assert result["matched_fraction"] == pytest.approx(0.0)


def test_score_no_matches_intensity_fraction_is_zero() -> None:
    spec = _spectrum()
    frag = _make_frag(999.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="Da")
    assert result["intensity_fraction"] == pytest.approx(0.0)


def test_score_no_matches_spectral_angle_is_zero() -> None:
    spec = _spectrum()
    frag = _make_frag(999.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="Da")
    assert result["spectral_angle"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Matched fragment — positive scores
# ---------------------------------------------------------------------------


def test_hyperscore_positive_when_fragment_matches() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="Da")
    assert result["hyperscore"] > 0.0


def test_total_matched_intensity_equals_matched_peak_intensity() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0)  # matches peak at index 1 with intensity 50.0
    result = score(spec, [frag], tolerance=0.02, tolerance_type="Da")
    assert result["total_matched_intensity"] == pytest.approx(50.0)


def test_mean_ppm_error_zero_for_perfect_match() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0)  # exact m/z
    result = score(spec, [frag], tolerance=0.02, tolerance_type="Da")
    assert result["mean_ppm_error"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Bounds checks
# ---------------------------------------------------------------------------


def test_matched_fraction_in_zero_to_one() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="Da")
    assert 0.0 <= result["matched_fraction"] <= 1.0


def test_intensity_fraction_in_zero_to_one() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="Da")
    assert 0.0 <= result["intensity_fraction"] <= 1.0


def test_spectral_angle_in_minus_one_to_one() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="Da")
    assert -1.0 <= result["spectral_angle"] <= 1.0


# ---------------------------------------------------------------------------
# longest_run — consecutive ion series
# ---------------------------------------------------------------------------


def test_longest_run_three_consecutive_b_ions() -> None:
    spec = _spectrum()
    # b1=100, b2=200, b3=300 — all match spectrum peaks
    frags = [_make_frag(float(pos * 100), ion_type="b", position=pos) for pos in [1, 2, 3]]
    result = score(spec, frags, tolerance=0.02, tolerance_type="Da")
    assert result["longest_run"] >= 3.0


def test_longest_run_zero_when_no_match() -> None:
    spec = _spectrum()
    frag = _make_frag(999.0, ion_type="b", position=1)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="Da")
    assert result["longest_run"] == pytest.approx(0.0)


def test_longest_run_one_for_single_match() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0, ion_type="b", position=5)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="Da")
    assert result["longest_run"] >= 1.0


# ---------------------------------------------------------------------------
# Empty fragments list
# ---------------------------------------------------------------------------


def test_score_empty_fragments_returns_all_zeros() -> None:
    spec = _spectrum()
    result = score(spec, [], tolerance=0.02, tolerance_type="Da")
    for key in ("hyperscore", "total_matched_intensity", "matched_fraction",
                "intensity_fraction", "mean_ppm_error", "spectral_angle", "longest_run"):
        assert result[key] == pytest.approx(0.0), f"{key} should be 0 with no fragments"
