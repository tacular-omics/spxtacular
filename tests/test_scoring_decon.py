"""
Tests for the score array attached to Spectrum after deconvolution.

The scored deconvolution uses a Bhattacharyya-based isotopic pattern score.
Singletons (no isotope cluster found) always get score=0.0.
"""
import numpy as np
import pytest

from spxtacular.core import Spectrum, SpectrumType

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# Charge-2 isotope envelope (four peaks ~0.501 Da apart) plus one singleton.
_MZ = np.array([300.0, 500.0, 500.501, 501.002, 501.503], dtype=np.float64)
_INTENSITY = np.array([1000.0, 100000.0, 70000.0, 30000.0, 8000.0], dtype=np.float64)


def _raw() -> Spectrum:
    return Spectrum(mz=_MZ.copy(), intensity=_INTENSITY.copy())


def _decon() -> Spectrum:
    return _raw().deconvolute(charge_range=(1, 3), tolerance=50, tolerance_type="ppm")


# ---------------------------------------------------------------------------
# Score array basics
# ---------------------------------------------------------------------------


def test_score_array_is_none_on_raw_spectrum() -> None:
    spec = _raw()
    assert spec.score is None


def test_score_array_populated_after_deconvolute() -> None:
    decon = _decon()
    assert decon.score is not None
    assert decon.score.dtype == np.float64
    assert len(decon.score) == len(decon.mz)
    assert np.all(decon.score >= 0.0)
    assert np.all(decon.score <= 1.0)


def test_singletons_have_score_zero() -> None:
    decon = _decon()
    assert decon.charge is not None
    singleton_mask = decon.charge == -1
    assert singleton_mask.any(), "expected at least one singleton"
    assert np.all(decon.score[singleton_mask] == 0.0)  # type: ignore[index]


def test_assigned_cluster_has_positive_score() -> None:
    decon = _decon()
    assert decon.charge is not None
    assigned_mask = decon.charge > 0
    assert assigned_mask.any(), "expected at least one assigned cluster"
    assert np.all(decon.score[assigned_mask] > 0.0)  # type: ignore[index]


# ---------------------------------------------------------------------------
# filter() with score bounds
# ---------------------------------------------------------------------------


def test_filter_min_score_removes_low_scoring_peaks() -> None:
    decon = _decon()
    filtered = decon.filter(min_score=0.4)
    assert len(filtered.mz) < len(decon.mz)
    assert filtered.score is not None
    assert np.all(filtered.score >= 0.4)


def test_filter_max_score_removes_high_scoring_peaks() -> None:
    decon = _decon()
    filtered = decon.filter(max_score=0.3)
    assert len(filtered.mz) < len(decon.mz)
    assert filtered.score is not None
    assert np.all(filtered.score <= 0.3)


def test_filter_score_preserves_score_array() -> None:
    decon = _decon()
    filtered = decon.filter(min_score=0.1)
    assert filtered.score is not None


def test_filter_min_score_keeps_all_when_threshold_is_zero() -> None:
    decon = _decon()
    filtered = decon.filter(min_score=0.0)
    assert len(filtered.mz) == len(decon.mz)


# ---------------------------------------------------------------------------
# decharge() propagates scores
# ---------------------------------------------------------------------------


def test_decharge_propagates_score() -> None:
    decon = _decon()
    decharged = decon.decharge()
    assert decharged.score is not None
    assert len(decharged.score) == len(decharged.mz)


def test_decharge_score_values_match_assigned_cluster_scores() -> None:
    decon = _decon()
    assert decon.charge is not None
    known_scores = sorted(decon.score[decon.charge != -1].tolist())  # type: ignore[index]
    decharged = decon.decharge()
    assert decharged.score is not None
    # values should be the same set (possibly reordered by ascending neutral mass)
    assert sorted(decharged.score.tolist()) == pytest.approx(known_scores)


# ---------------------------------------------------------------------------
# peaks property
# ---------------------------------------------------------------------------


def test_peak_score_field_filled_for_scored_spectrum() -> None:
    decon = _decon()
    peaks = decon.peaks
    assert len(peaks) == len(decon.mz)
    for peak, expected_score in zip(peaks, decon.score, strict=True):  # type: ignore[arg-type]
        assert peak.score == pytest.approx(float(expected_score))


def test_peak_score_is_none_for_raw_spectrum() -> None:
    spec = _raw()
    peaks = spec.peaks
    for peak in peaks:
        assert peak.score is None


# ---------------------------------------------------------------------------
# min_intensity sentinel
# ---------------------------------------------------------------------------


def test_min_intensity_sentinel_string_runs_without_error() -> None:
    spec = _raw()
    decon = spec.deconvolute(charge_range=(1, 3), tolerance=50, tolerance_type="ppm", min_intensity="min")
    assert decon.score is not None


# ---------------------------------------------------------------------------
# min_score threshold — high threshold forces all singletons
# ---------------------------------------------------------------------------


def test_min_score_rejects_all_clusters_when_threshold_is_very_high() -> None:
    spec = _raw()
    decon = spec.deconvolute(
        charge_range=(1, 3), tolerance=50, tolerance_type="ppm", min_score=0.9999
    )
    assert decon.charge is not None
    assert np.all(decon.charge == -1), "expected all singletons with min_score=0.9999"


def test_min_score_zero_allows_best_cluster_to_be_assigned() -> None:
    spec = _raw()
    decon = spec.deconvolute(
        charge_range=(1, 3), tolerance=50, tolerance_type="ppm", min_score=0.0
    )
    assert decon.charge is not None
    assert np.any(decon.charge > 0), "expected at least one assigned cluster"


# ---------------------------------------------------------------------------
# Manually constructed scored spectrum
# ---------------------------------------------------------------------------


def test_manually_scored_spectrum_filter_works() -> None:
    """A Spectrum constructed with an explicit score array filters correctly."""
    spec = Spectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype=np.float64),
        intensity=np.array([10.0, 50.0, 20.0], dtype=np.float64),
        charge=np.array([2, -1, 3], dtype=np.int32),
        score=np.array([0.9, 0.0, 0.5], dtype=np.float64),
        spectrum_type=SpectrumType.DECONVOLUTED,
    )
    filtered = spec.filter(min_score=0.6)
    assert len(filtered.mz) == 1
    assert filtered.mz[0] == pytest.approx(100.0)
    assert filtered.score is not None
    assert filtered.score[0] == pytest.approx(0.9)
