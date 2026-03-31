"""
Tests for spxtacular.scoring.score.

Fragments are mocked with MagicMock — score() only accesses .mz, .ion_type,
and .position through match_fragments and the internal helpers.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from spxtacular.core import Spectrum
from spxtacular.scoring import _binom_log10_survival, _count_unique_ions, score

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
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert set(result.keys()) == _EXPECTED_KEYS


def test_score_all_values_are_floats() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    for key, val in result.items():
        assert isinstance(val, float), f"{key} is not float"


# ---------------------------------------------------------------------------
# No-match baseline
# ---------------------------------------------------------------------------


def test_score_no_matches_hyperscore_is_zero() -> None:
    spec = _spectrum()
    frag = _make_frag(999.0)  # far outside spectrum
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert result["hyperscore"] == pytest.approx(0.0)


def test_score_no_matches_total_matched_intensity_is_zero() -> None:
    spec = _spectrum()
    frag = _make_frag(999.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert result["total_matched_intensity"] == pytest.approx(0.0)


def test_score_no_matches_matched_fraction_is_zero() -> None:
    spec = _spectrum()
    frag = _make_frag(999.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert result["matched_fraction"] == pytest.approx(0.0)


def test_score_no_matches_intensity_fraction_is_zero() -> None:
    spec = _spectrum()
    frag = _make_frag(999.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert result["intensity_fraction"] == pytest.approx(0.0)


def test_score_no_matches_spectral_angle_is_zero() -> None:
    spec = _spectrum()
    frag = _make_frag(999.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert result["spectral_angle"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Matched fragment — positive scores
# ---------------------------------------------------------------------------


def test_hyperscore_positive_when_fragment_matches() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert result["hyperscore"] > 0.0


def test_total_matched_intensity_equals_matched_peak_intensity() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0)  # matches peak at index 1 with intensity 50.0
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert result["total_matched_intensity"] == pytest.approx(50.0)


def test_mean_ppm_error_zero_for_perfect_match() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0)  # exact m/z
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert result["mean_ppm_error"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Bounds checks
# ---------------------------------------------------------------------------


def test_matched_fraction_in_zero_to_one() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert 0.0 <= result["matched_fraction"] <= 1.0


def test_intensity_fraction_in_zero_to_one() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert 0.0 <= result["intensity_fraction"] <= 1.0


def test_spectral_angle_in_minus_one_to_one() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert -1.0 <= result["spectral_angle"] <= 1.0


# ---------------------------------------------------------------------------
# longest_run — consecutive ion series
# ---------------------------------------------------------------------------


def test_longest_run_three_consecutive_b_ions() -> None:
    spec = _spectrum()
    # b1=100, b2=200, b3=300 — all match spectrum peaks
    frags = [_make_frag(float(pos * 100), ion_type="b", position=pos) for pos in [1, 2, 3]]
    result = score(spec, frags, tolerance=0.02, tolerance_type="da")
    assert result["longest_run"] >= 3.0


def test_longest_run_zero_when_no_match() -> None:
    spec = _spectrum()
    frag = _make_frag(999.0, ion_type="b", position=1)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert result["longest_run"] == pytest.approx(0.0)


def test_longest_run_one_for_single_match() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0, ion_type="b", position=5)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert result["longest_run"] >= 1.0


# ---------------------------------------------------------------------------
# Empty fragments list
# ---------------------------------------------------------------------------


def test_score_empty_fragments_returns_all_zeros() -> None:
    spec = _spectrum()
    result = score(spec, [], tolerance=0.02, tolerance_type="da")
    for key in (
        "hyperscore",
        "total_matched_intensity",
        "matched_fraction",
        "intensity_fraction",
        "mean_ppm_error",
        "spectral_angle",
        "longest_run",
    ):
        assert result[key] == pytest.approx(0.0), f"{key} should be 0 with no fragments"


# ---------------------------------------------------------------------------
# _binom_log10_survival edge cases
# ---------------------------------------------------------------------------


def test_binom_log10_survival_k_zero_returns_zero() -> None:
    assert _binom_log10_survival(0, 10, 0.5) == pytest.approx(0.0)


def test_binom_log10_survival_k_negative_returns_zero() -> None:
    assert _binom_log10_survival(-1, 10, 0.5) == pytest.approx(0.0)


def test_binom_log10_survival_k_greater_than_n_returns_neg_inf() -> None:
    import math

    assert _binom_log10_survival(11, 10, 0.5) == -math.inf


def test_binom_log10_survival_p_zero_returns_neg_inf() -> None:
    import math

    assert _binom_log10_survival(1, 10, 0.0) == -math.inf


def test_binom_log10_survival_p_one_returns_zero() -> None:
    assert _binom_log10_survival(1, 10, 1.0) == pytest.approx(0.0)


def test_binom_log10_survival_p_greater_than_one_returns_zero() -> None:
    assert _binom_log10_survival(1, 10, 1.5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _count_unique_ions with dict input
# ---------------------------------------------------------------------------


def test_count_unique_ions_dict_sums_lengths() -> None:
    from peptacular import IonType

    frag_dict: dict = {
        (IonType.B, 1): [100.0, 200.0, 300.0],
        (IonType.Y, 1): [400.0, 500.0],
    }
    result = _count_unique_ions(frag_dict)
    assert result == 5


def test_count_unique_ions_dict_empty_returns_zero() -> None:
    result = _count_unique_ions({})
    assert result == 0


# ---------------------------------------------------------------------------
# _probability_score edge cases via public score()
# ---------------------------------------------------------------------------


def test_probability_score_zero_for_empty_spectrum() -> None:
    spec = Spectrum(
        mz=np.array([], dtype=np.float64),
        intensity=np.array([], dtype=np.float64),
    )
    frag = _make_frag(200.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert result["probability_score"] == pytest.approx(0.0)


def test_probability_score_zero_for_single_peak_zero_range() -> None:
    spec = Spectrum(
        mz=np.array([200.0], dtype=np.float64),
        intensity=np.array([100.0], dtype=np.float64),
    )
    frag = _make_frag(200.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert result["probability_score"] == pytest.approx(0.0)


def test_probability_score_ppm_tolerance_path() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0)
    result = score(spec, [frag], tolerance=10.0, tolerance_type="ppm")
    assert result["probability_score"] >= 0.0


# ---------------------------------------------------------------------------
# score() with dict fragments input
# ---------------------------------------------------------------------------


def test_score_dict_fragments_returns_expected_keys() -> None:
    from peptacular import IonType

    spec = _spectrum()
    frag_dict: dict = {
        (IonType.B, 1): [100.0, 200.0],
    }
    result = score(spec, frag_dict, tolerance=0.02, tolerance_type="da")
    assert set(result.keys()) == _EXPECTED_KEYS
