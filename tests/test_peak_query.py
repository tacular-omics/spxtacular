"""
Tests for the peak-query API: ``has_peak``, ``get_peak``, ``get_peaks`` and
their shared ``_find_matching_peaks`` helper.
"""

from typing import Any, cast

import numpy as np
import pytest

from spxtacular.core import Peak, Spectrum
from spxtacular.enums import ToleranceType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _query_spec() -> Spectrum:
    """Three peaks clustered around 100 Da plus two far away.

    The cluster is deliberately arranged so that the *closest* peak to 100.0 is
    not the *most intense* one, which is what makes the two ``collision`` modes
    distinguishable. Charge, IM and iso_score all differ per peak so a query
    that returned the wrong peak cannot look right by accident.
    """
    return Spectrum(
        mz=np.array([100.000, 100.004, 100.008, 200.0, 300.0], dtype=np.float64),
        intensity=np.array([10.0, 100.0, 50.0, 999.0, 7.0], dtype=np.float64),
        charge=np.array([1, 2, 2, 3, 1], dtype=np.int32),
        im=np.array([0.90, 1.10, 0.92, 1.50, 0.80], dtype=np.float64),
        iso_score=np.array([0.10, 0.80, 0.55, 0.99, 0.20], dtype=np.float64),
    )


def _mzs(peaks: list[Peak]) -> list[float]:
    return [p.mz for p in peaks]


def _boundary_spec() -> Spectrum:
    """m/z values that are exact in binary, so an edge test is not fuzzy."""
    return Spectrum(
        mz=np.array([100.0, 100.25, 100.5], dtype=np.float64),
        intensity=np.array([1.0, 2.0, 3.0], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Tolerance: Da vs ppm
# ---------------------------------------------------------------------------


class TestTolerance:
    def test_da_tolerance_window_selects_exact_peaks(self) -> None:
        spec = _query_spec()
        assert _mzs(spec.get_peaks(100.0, tolerance=0.01)) == [100.000, 100.004, 100.008]
        # a narrower window must actually shrink the result
        assert _mzs(spec.get_peaks(100.0, tolerance=0.005)) == [100.000, 100.004]
        assert _mzs(spec.get_peaks(100.0, tolerance=0.001)) == [100.000]

    def test_ppm_tolerance_scales_with_target_mz(self) -> None:
        spec = _query_spec()
        # 50 ppm of 100 Da == 0.005 Da -> two peaks.
        assert _mzs(spec.get_peaks(100.0, tolerance=50, tolerance_type="ppm")) == [100.000, 100.004]

    def test_ppm_and_da_differ_for_the_same_tolerance_number(self) -> None:
        """The unit must be honoured: 50 Da is a vastly wider window than 50 ppm."""
        spec = _query_spec()
        ppm_hits = spec.get_peaks(100.0, tolerance=50, tolerance_type="ppm")
        da_hits = spec.get_peaks(100.0, tolerance=50, tolerance_type="da")
        assert _mzs(ppm_hits) == [100.000, 100.004]  # 0.005 Da window
        assert _mzs(da_hits) == [100.000, 100.004, 100.008]  # +-50 Da window (reaches 150)

    def test_ppm_window_grows_with_the_target(self) -> None:
        # 20 ppm is 0.002 Da at m/z 100 but 0.02 Da at m/z 1000, so the same
        # ppm value must accept a 0.01 Da offset only at the high m/z.
        spec = Spectrum(
            mz=np.array([100.01, 1000.01], dtype=np.float64),
            intensity=np.array([1.0, 1.0], dtype=np.float64),
        )
        assert spec.get_peaks(100.0, tolerance=20, tolerance_type="ppm") == []
        assert len(spec.get_peaks(1000.0, tolerance=20, tolerance_type="ppm")) == 1

    def test_tolerance_type_is_case_insensitive(self) -> None:
        """``"PPM"`` used to fall through to Da — a window 1e6 times too wide."""
        spec = _query_spec()
        expected = _mzs(spec.get_peaks(100.0, tolerance=50, tolerance_type=ToleranceType.PPM))
        assert expected == [100.000, 100.004]
        # cast: the Literal alias only spells the lower-case forms, but the
        # runtime coerces through the enum, so these must agree.
        for spelling in ("ppm", "PPM", "Ppm"):
            assert _mzs(spec.get_peaks(100.0, tolerance=50, tolerance_type=cast(Any, spelling))) == expected

    def test_da_tolerance_type_is_case_insensitive(self) -> None:
        spec = _query_spec()
        expected = _mzs(spec.get_peaks(100.0, tolerance=0.005, tolerance_type=ToleranceType.DA))
        for spelling in ("da", "DA", "Da"):
            assert _mzs(spec.get_peaks(100.0, tolerance=0.005, tolerance_type=cast(Any, spelling))) == expected

    def test_default_tolerance_type_is_da(self) -> None:
        spec = _query_spec()
        assert _mzs(spec.get_peaks(100.0, tolerance=0.005)) == _mzs(
            spec.get_peaks(100.0, tolerance=0.005, tolerance_type="da")
        )

    @pytest.mark.parametrize("method", ["has_peak", "get_peak", "get_peaks"])
    def test_unknown_tolerance_type_raises(self, method: str) -> None:
        """An unrecognised unit must raise, not silently mean "Da"."""
        spec = _query_spec()
        with pytest.raises(ValueError, match="ToleranceType"):
            getattr(spec, method)(100.0, 0.01, cast(Any, "bogus"))


class TestToleranceBoundary:
    def test_peak_exactly_at_da_tolerance_edge_is_included(self) -> None:
        # |100.25 - 100.0| == 0.25 exactly (both values are binary-exact), so
        # this discriminates ``<=`` from ``<``.
        spec = _boundary_spec()
        assert _mzs(spec.get_peaks(100.0, tolerance=0.25)) == [100.0, 100.25]

    def test_peak_just_outside_da_tolerance_is_excluded(self) -> None:
        spec = _boundary_spec()
        assert _mzs(spec.get_peaks(100.0, tolerance=0.2)) == [100.0]

    def test_peak_exactly_at_ppm_tolerance_edge_is_included(self) -> None:
        # 2500 ppm of 100.0 is exactly 0.25 Da.
        spec = _boundary_spec()
        assert _mzs(spec.get_peaks(100.0, tolerance=2500, tolerance_type="ppm")) == [100.0, 100.25]

    def test_peak_exactly_at_im_tolerance_edge_is_included(self) -> None:
        spec = Spectrum(
            mz=np.array([100.0, 100.0], dtype=np.float64),
            intensity=np.array([1.0, 2.0], dtype=np.float64),
            im=np.array([1.0, 1.25], dtype=np.float64),
        )
        peaks = spec.get_peaks(100.0, tolerance=0.001, target_im=1.0, im_tol=0.25)
        assert [p.im for p in peaks] == [1.0, 1.25]
        assert [p.im for p in spec.get_peaks(100.0, tolerance=0.001, target_im=1.0, im_tol=0.2)] == [1.0]


# ---------------------------------------------------------------------------
# Extra filters: charge and ion mobility
# ---------------------------------------------------------------------------


class TestTargetCharge:
    def test_target_charge_keeps_only_that_charge(self) -> None:
        spec = _query_spec()
        peaks = spec.get_peaks(100.0, tolerance=0.01, target_charge=2)
        assert _mzs(peaks) == [100.004, 100.008]
        assert {p.charge for p in peaks} == {2}

    def test_target_charge_narrows_the_unfiltered_result(self) -> None:
        spec = _query_spec()
        assert _mzs(spec.get_peaks(100.0, tolerance=0.01, target_charge=1)) == [100.000]

    def test_target_charge_with_no_matching_charge_returns_nothing(self) -> None:
        spec = _query_spec()
        assert spec.get_peaks(100.0, tolerance=0.01, target_charge=7) == []
        assert spec.get_peak(100.0, tolerance=0.01, target_charge=7) is None
        assert spec.has_peak(100.0, tolerance=0.01, target_charge=7) is False

    def test_target_charge_raises_when_spectrum_has_no_charge_array(self) -> None:
        """Asking for a dimension the spectrum lacks is an error, not a no-op.

        Silently skipping the filter returns every m/z match, which reads as
        "these are the charge-3 peaks". Same rule as ``filter()``.
        """
        spec = _boundary_spec()
        for call in (
            lambda: spec.get_peaks(100.0, tolerance=0.25, target_charge=3),
            lambda: spec.get_peak(100.0, tolerance=0.25, target_charge=3),
            lambda: spec.has_peak(100.0, tolerance=0.25, target_charge=3),
        ):
            with pytest.raises(ValueError, match="no charge array"):
                call()


class TestTargetIm:
    def test_target_im_keeps_only_peaks_in_the_mobility_window(self) -> None:
        spec = _query_spec()
        # IM values in the cluster are 0.90 / 1.10 / 0.92; +-0.02 around 0.91
        # keeps the first and third and drops the (most intense) second.
        peaks = spec.get_peaks(100.0, tolerance=0.01, target_im=0.91, im_tol=0.02)
        assert _mzs(peaks) == [100.000, 100.008]

    def test_default_im_tol_is_narrow(self) -> None:
        spec = _query_spec()
        # default im_tol=0.01 around 0.90 excludes the 0.92 peak.
        assert _mzs(spec.get_peaks(100.0, tolerance=0.01, target_im=0.90)) == [100.000]

    def test_target_im_with_no_match_returns_nothing(self) -> None:
        spec = _query_spec()
        assert spec.get_peaks(100.0, tolerance=0.01, target_im=5.0) == []
        assert spec.get_peak(100.0, tolerance=0.01, target_im=5.0) is None
        assert spec.has_peak(100.0, tolerance=0.01, target_im=5.0) is False

    def test_charge_and_im_filters_compose(self) -> None:
        spec = _query_spec()
        # charge==2 alone gives 100.004 and 100.008; the IM window then keeps
        # only 100.008.
        peaks = spec.get_peaks(100.0, tolerance=0.01, target_charge=2, target_im=0.92)
        assert _mzs(peaks) == [100.008]

    def test_target_im_raises_when_spectrum_has_no_im_array(self) -> None:
        """As for charge: a request for a missing dimension must not pass silently."""
        spec = _boundary_spec()
        with pytest.raises(ValueError, match="no im array"):
            spec.get_peaks(100.0, tolerance=0.25, target_im=42.0)


# ---------------------------------------------------------------------------
# get_peak: collision handling
# ---------------------------------------------------------------------------


class TestGetPeakCollision:
    def test_largest_and_closest_disagree_on_the_same_query(self) -> None:
        """The two modes must resolve the same three-way collision differently."""
        spec = _query_spec()
        largest = spec.get_peak(100.0, tolerance=0.01, collision="largest")
        closest = spec.get_peak(100.0, tolerance=0.01, collision="closest")
        assert largest is not None
        assert closest is not None
        assert largest.mz == pytest.approx(100.004)  # highest intensity (100.0)
        assert closest.mz == pytest.approx(100.000)  # smallest |Δm/z|
        assert largest.mz != closest.mz

    def test_default_collision_is_largest(self) -> None:
        spec = _query_spec()
        default = spec.get_peak(100.0, tolerance=0.01)
        assert default is not None
        assert default.mz == pytest.approx(100.004)

    def test_closest_picks_the_nearer_peak_from_either_side(self) -> None:
        # nearest match is *below* the target here, so "closest" cannot be
        # faked by always taking the first or last index in the window.
        spec = Spectrum(
            mz=np.array([99.996, 100.006], dtype=np.float64),
            intensity=np.array([1.0, 500.0], dtype=np.float64),
        )
        closest = spec.get_peak(100.0, tolerance=0.01, collision="closest")
        largest = spec.get_peak(100.0, tolerance=0.01, collision="largest")
        assert closest is not None and largest is not None
        assert closest.mz == pytest.approx(99.996)
        assert largest.mz == pytest.approx(100.006)

    def test_collision_respects_the_charge_filter(self) -> None:
        """The winner is chosen among filtered matches, not all m/z matches."""
        spec = _query_spec()
        # unrestricted, "largest" is the z=2 peak at 100.004; restricted to
        # z=1 the only candidate is 100.000.
        peak = spec.get_peak(100.0, tolerance=0.01, target_charge=1, collision="largest")
        assert peak is not None
        assert peak.mz == pytest.approx(100.000)
        assert peak.charge == 1


# ---------------------------------------------------------------------------
# Misses
# ---------------------------------------------------------------------------


class TestNoMatch:
    def test_get_peak_returns_none_when_nothing_matches(self) -> None:
        assert _query_spec().get_peak(150.0, tolerance=0.01) is None

    def test_get_peaks_returns_empty_list_when_nothing_matches(self) -> None:
        assert _query_spec().get_peaks(150.0, tolerance=0.01) == []

    def test_has_peak_is_false_when_nothing_matches(self) -> None:
        assert _query_spec().has_peak(150.0, tolerance=0.01) is False

    def test_has_peak_is_true_only_when_a_peak_is_in_range(self) -> None:
        spec = _boundary_spec()
        assert spec.has_peak(100.5, tolerance=0.01) is True
        assert spec.has_peak(100.4, tolerance=0.01) is False

    @pytest.mark.parametrize("target", [100.0, 100.004, 200.0, 150.0, 0.0])
    def test_has_peak_agrees_with_get_peaks(self, target: float) -> None:
        spec = _query_spec()
        assert spec.has_peak(target, tolerance=0.01) == (len(spec.get_peaks(target, tolerance=0.01)) > 0)


# ---------------------------------------------------------------------------
# Returned Peak objects
# ---------------------------------------------------------------------------


class TestReturnedPeak:
    def test_get_peak_carries_every_optional_field(self) -> None:
        """Regression: ``iso_score`` used to be dropped from returned peaks."""
        spec = _query_spec()
        peak = spec.get_peak(100.0, tolerance=0.01, collision="closest")
        assert peak is not None
        assert peak.mz == pytest.approx(100.000)
        assert peak.intensity == pytest.approx(10.0)
        assert peak.charge == 1
        assert peak.im == pytest.approx(0.90)
        assert peak.iso_score == pytest.approx(0.10)

    def test_get_peaks_carry_every_optional_field(self) -> None:
        spec = _query_spec()
        peaks = spec.get_peaks(100.0, tolerance=0.01)
        assert [p.charge for p in peaks] == [1, 2, 2]
        assert [p.im for p in peaks] == pytest.approx([0.90, 1.10, 0.92])
        assert [p.iso_score for p in peaks] == pytest.approx([0.10, 0.80, 0.55])

    def test_optional_fields_are_none_when_the_spectrum_lacks_them(self) -> None:
        spec = _boundary_spec()
        peak = spec.get_peak(100.0, tolerance=0.001)
        assert peak is not None
        assert peak.charge is None
        assert peak.im is None
        assert peak.iso_score is None

    def test_scalars_are_python_types_not_numpy_scalars(self) -> None:
        """``type(...) is`` on purpose: ``np.float64`` passes ``isinstance(x, float)``.

        Leaking numpy scalars out of the API makes ``Peak`` unhashable-adjacent
        surprises (e.g. json serialisation) show up far from here.
        """
        spec = _query_spec()
        peak = spec.get_peak(100.0, tolerance=0.01)
        assert peak is not None
        assert type(peak.mz) is float
        assert type(peak.intensity) is float
        assert type(peak.charge) is int
        assert type(peak.im) is float
        assert type(peak.iso_score) is float

        for p in spec.get_peaks(100.0, tolerance=0.01):
            assert type(p.mz) is float
            assert type(p.charge) is int


# ---------------------------------------------------------------------------
# Degenerate spectra
# ---------------------------------------------------------------------------


class TestDegenerateSpectra:
    def test_empty_spectrum_matches_nothing(self) -> None:
        spec = Spectrum(
            mz=np.empty(0, dtype=np.float64),
            intensity=np.empty(0, dtype=np.float64),
        )
        assert spec.has_peak(100.0) is False
        assert spec.get_peak(100.0) is None
        assert spec.get_peaks(100.0) == []

    def test_empty_spectrum_with_optional_arrays_matches_nothing(self) -> None:
        spec = Spectrum(
            mz=np.empty(0, dtype=np.float64),
            intensity=np.empty(0, dtype=np.float64),
            charge=np.empty(0, dtype=np.int32),
            im=np.empty(0, dtype=np.float64),
            iso_score=np.empty(0, dtype=np.float64),
        )
        assert spec.get_peaks(100.0, tolerance=1.0, target_charge=2, target_im=1.0) == []
        assert spec.get_peak(100.0, tolerance=1.0, collision="closest") is None

    def test_single_peak_spectrum_hit_and_miss(self) -> None:
        spec = Spectrum(
            mz=np.array([250.0], dtype=np.float64),
            intensity=np.array([42.0], dtype=np.float64),
        )
        hit = spec.get_peak(250.004, tolerance=0.01)
        assert hit is not None
        assert hit.intensity == pytest.approx(42.0)
        assert spec.has_peak(250.02, tolerance=0.01) is False
        assert spec.get_peaks(250.02, tolerance=0.01) == []

    def test_single_peak_spectrum_same_answer_for_both_collision_modes(self) -> None:
        spec = Spectrum(
            mz=np.array([250.0], dtype=np.float64),
            intensity=np.array([42.0], dtype=np.float64),
        )
        assert spec.get_peak(250.0, collision="largest") == spec.get_peak(250.0, collision="closest")


# ---------------------------------------------------------------------------
# _find_matching_peaks (shared helper)
# ---------------------------------------------------------------------------


class TestFindMatchingPeaks:
    def test_returns_indices_in_storage_order(self) -> None:
        """Indices come back ascending, i.e. in the spectrum's own peak order."""
        spec = Spectrum(
            mz=np.array([100.008, 100.000, 100.004], dtype=np.float64),
            intensity=np.array([50.0, 10.0, 100.0], dtype=np.float64),
        )
        idx = spec._find_matching_peaks(100.0, 0.01, ToleranceType.DA, None, None, 0.01)
        assert idx.tolist() == [0, 1, 2]
        # get_peaks inherits that order — it does not re-sort by m/z.
        assert _mzs(spec.get_peaks(100.0, tolerance=0.01)) == [100.008, 100.000, 100.004]

    def test_returns_empty_index_array_on_a_miss(self) -> None:
        spec = _query_spec()
        idx = spec._find_matching_peaks(150.0, 0.01, ToleranceType.DA, None, None, 0.01)
        assert idx.tolist() == []


class TestCollisionValidation:
    def test_unknown_collision_mode_raises(self) -> None:
        """An unrecognised mode used to fall through to "closest" silently."""
        spec = _boundary_spec()
        with pytest.raises(ValueError, match="collision must be"):
            spec.get_peak(100.0, tolerance=0.5, collision="nearest")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    def test_collision_mode_is_case_insensitive(self) -> None:
        spec = _boundary_spec()
        upper = spec.get_peak(100.0, tolerance=0.5, collision="LARGEST")  # ty: ignore[invalid-argument-type]
        assert upper == spec.get_peak(100.0, tolerance=0.5, collision="largest")
