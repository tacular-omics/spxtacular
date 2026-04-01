"""
Tests for new features: remove_precursor_peak, scale_intensity, round_mz,
mass_error_plot, facet_plot, da_to_ppm, ppm_to_da.
"""

import numpy as np
import pytest

from spxtacular.core import Spectrum
from spxtacular.utils import da_to_ppm, ppm_to_da

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spec() -> Spectrum:
    return Spectrum(
        mz=np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64),
        intensity=np.array([10.0, 50.0, 30.0, 20.0, 5.0], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# remove_precursor_peak
# ---------------------------------------------------------------------------


class TestRemovePrecursorPeak:
    def test_removes_precursor_within_tolerance(self) -> None:
        spec = _spec()
        result = spec.remove_precursor_peak(precursor_mz=300.0, tolerance=0.5)
        assert 300.0 not in result.mz
        assert len(result.mz) == 4

    def test_keeps_peaks_outside_tolerance(self) -> None:
        spec = _spec()
        result = spec.remove_precursor_peak(precursor_mz=300.0, tolerance=0.5)
        np.testing.assert_array_equal(result.mz, [100.0, 200.0, 400.0, 500.0])

    def test_removes_isotope_peaks(self) -> None:
        neutron = 1.003355
        mz = np.array([300.0, 300.0 + neutron, 300.0 + 2 * neutron, 500.0], dtype=np.float64)
        spec = Spectrum(mz=mz, intensity=np.ones(4, dtype=np.float64))
        result = spec.remove_precursor_peak(precursor_mz=300.0, tolerance=0.01, isotopes=2)
        assert len(result.mz) == 1
        assert result.mz[0] == pytest.approx(500.0)

    def test_no_isotopes_by_default(self) -> None:
        neutron = 1.003355
        mz = np.array([300.0, 300.0 + neutron, 500.0], dtype=np.float64)
        spec = Spectrum(mz=mz, intensity=np.ones(3, dtype=np.float64))
        result = spec.remove_precursor_peak(precursor_mz=300.0, tolerance=0.01, isotopes=0)
        assert len(result.mz) == 2

    def test_ppm_tolerance(self) -> None:
        spec = _spec()
        # 10 ppm at m/z 300 = 0.003 Da
        result = spec.remove_precursor_peak(precursor_mz=300.0, tolerance=10, tolerance_type="ppm")
        assert 300.0 not in result.mz

    def test_inplace(self) -> None:
        spec = _spec()
        original_id = id(spec)
        result = spec.remove_precursor_peak(precursor_mz=300.0, tolerance=0.5, inplace=True)
        assert id(result) == original_id
        assert 300.0 not in result.mz

    def test_no_match_returns_all_peaks(self) -> None:
        spec = _spec()
        result = spec.remove_precursor_peak(precursor_mz=999.0, tolerance=0.01)
        assert len(result.mz) == 5


# ---------------------------------------------------------------------------
# scale_intensity
# ---------------------------------------------------------------------------


class TestScaleIntensity:
    def test_root_default_sqrt(self) -> None:
        spec = _spec()
        result = spec.scale_intensity(method="root")
        np.testing.assert_allclose(result.intensity, np.sqrt(spec.intensity))

    def test_root_custom_degree(self) -> None:
        spec = _spec()
        result = spec.scale_intensity(method="root", degree=3)
        np.testing.assert_allclose(result.intensity, np.power(spec.intensity, 1.0 / 3))

    def test_log_default_base2(self) -> None:
        spec = _spec()
        result = spec.scale_intensity(method="log")
        expected = np.log1p(spec.intensity) / np.log(2)
        np.testing.assert_allclose(result.intensity, expected)

    def test_log_custom_base(self) -> None:
        spec = _spec()
        result = spec.scale_intensity(method="log", base=10.0)
        expected = np.log1p(spec.intensity) / np.log(10)
        np.testing.assert_allclose(result.intensity, expected)

    def test_rank(self) -> None:
        spec = _spec()
        result = spec.scale_intensity(method="rank")
        # Original intensities: [10, 50, 30, 20, 5]
        # Sorted order: 5(1), 10(2), 20(3), 30(4), 50(5)
        # Ranks at positions: [2, 5, 4, 3, 1]
        np.testing.assert_array_equal(result.intensity, [2.0, 5.0, 4.0, 3.0, 1.0])

    def test_unknown_method_raises(self) -> None:
        spec = _spec()
        with pytest.raises(ValueError, match="Unknown scaling method"):
            spec.scale_intensity(method="invalid")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    def test_inplace(self) -> None:
        spec = _spec()
        original_id = id(spec)
        result = spec.scale_intensity(method="root", inplace=True)
        assert id(result) == original_id

    def test_preserves_mz(self) -> None:
        spec = _spec()
        result = spec.scale_intensity(method="log")
        np.testing.assert_array_equal(result.mz, spec.mz)


# ---------------------------------------------------------------------------
# round_mz
# ---------------------------------------------------------------------------


class TestRoundMz:
    def test_round_combines_sum(self) -> None:
        mz = np.array([100.1, 100.2, 200.3], dtype=np.float64)
        intensity = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        spec = Spectrum(mz=mz, intensity=intensity)
        result = spec.round_mz(decimals=0, combine="sum")
        assert len(result.mz) == 2
        np.testing.assert_array_equal(result.mz, [100.0, 200.0])
        np.testing.assert_array_equal(result.intensity, [30.0, 30.0])

    def test_round_combines_max(self) -> None:
        mz = np.array([100.1, 100.2, 200.3], dtype=np.float64)
        intensity = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        spec = Spectrum(mz=mz, intensity=intensity)
        result = spec.round_mz(decimals=0, combine="max")
        assert len(result.mz) == 2
        np.testing.assert_array_equal(result.mz, [100.0, 200.0])
        np.testing.assert_array_equal(result.intensity, [20.0, 30.0])

    def test_round_with_decimals(self) -> None:
        mz = np.array([100.11, 100.12, 100.19], dtype=np.float64)
        intensity = np.array([5.0, 10.0, 15.0], dtype=np.float64)
        spec = Spectrum(mz=mz, intensity=intensity)
        result = spec.round_mz(decimals=1, combine="sum")
        # 100.11 -> 100.1, 100.12 -> 100.1, 100.19 -> 100.2
        assert len(result.mz) == 2
        np.testing.assert_allclose(result.mz, [100.1, 100.2])
        np.testing.assert_allclose(result.intensity, [15.0, 15.0])

    def test_drops_optional_arrays(self) -> None:
        mz = np.array([100.1, 100.2], dtype=np.float64)
        intensity = np.array([10.0, 20.0], dtype=np.float64)
        charge = np.array([1, 2], dtype=np.int32)
        spec = Spectrum(mz=mz, intensity=intensity, charge=charge)
        result = spec.round_mz(decimals=0)
        assert result.charge is None

    def test_unknown_combine_raises(self) -> None:
        spec = _spec()
        with pytest.raises(ValueError, match="Unknown combine method"):
            spec.round_mz(combine="mean")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


# ---------------------------------------------------------------------------
# da_to_ppm / ppm_to_da
# ---------------------------------------------------------------------------


class TestConversions:
    def test_da_to_ppm(self) -> None:
        result = da_to_ppm(0.001, 1000.0)
        assert result == pytest.approx(1.0)

    def test_ppm_to_da(self) -> None:
        result = ppm_to_da(10.0, 500.0)
        assert result == pytest.approx(0.005)

    def test_roundtrip(self) -> None:
        mz = 800.0
        da = 0.004
        ppm = da_to_ppm(da, mz)
        da_back = ppm_to_da(ppm, mz)
        assert da_back == pytest.approx(da)

    def test_zero_delta(self) -> None:
        assert da_to_ppm(0.0, 500.0) == 0.0
        assert ppm_to_da(0.0, 500.0) == 0.0


# ---------------------------------------------------------------------------
# mass_error_plot
# ---------------------------------------------------------------------------


class TestMassErrorPlot:
    def test_returns_figure_with_matches(self) -> None:
        import plotly.graph_objects as go
        from peptacular import IonType

        mz = np.array([200.0, 300.0, 400.0], dtype=np.float64)
        intensity = np.array([100.0, 200.0, 150.0], dtype=np.float64)
        spec = Spectrum(mz=mz, intensity=intensity)

        fragments = {(IonType.B, 1): [200.001, 300.002]}
        fig = spec.mass_error_plot(fragments, tolerance=0.01, tolerance_type="da")
        assert isinstance(fig, go.Figure)

    def test_returns_empty_figure_no_matches(self) -> None:
        import plotly.graph_objects as go
        from peptacular import IonType

        spec = _spec()
        fragments = {(IonType.B, 1): [999.0]}
        fig = spec.mass_error_plot(fragments, tolerance=0.001, tolerance_type="da")
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# facet_plot
# ---------------------------------------------------------------------------


class TestFacetPlot:
    def test_spectrum_only(self) -> None:
        import plotly.graph_objects as go

        spec = _spec()
        fig = spec.facet_plot()
        assert isinstance(fig, go.Figure)

    def test_with_fragments(self) -> None:
        import plotly.graph_objects as go
        from peptacular import IonType

        spec = _spec()
        fragments = {(IonType.B, 1): [200.001, 300.002]}
        fig = spec.facet_plot(fragments=fragments, tolerance=0.01, tolerance_type="da")
        assert isinstance(fig, go.Figure)

    def test_with_mirror(self) -> None:
        import plotly.graph_objects as go

        spec = _spec()
        mirror = Spectrum(
            mz=np.array([150.0, 250.0, 350.0], dtype=np.float64),
            intensity=np.array([20.0, 40.0, 10.0], dtype=np.float64),
        )
        fig = spec.facet_plot(mirror_spectrum=mirror)
        assert isinstance(fig, go.Figure)
