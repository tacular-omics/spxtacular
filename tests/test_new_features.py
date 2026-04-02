"""
Tests for new features: remove_precursor_peak, scale_intensity, round_mz,
mass_error_plot, facet_plot, da_to_ppm, ppm_to_da.
"""

import numpy as np
import pytest

from spxtacular.core import MsnSpectrum, Precursor, Spectrum, SpectrumType
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

    def test_profile_raises(self) -> None:
        spec = Spectrum(
            mz=np.array([100.0, 200.0], dtype=np.float64),
            intensity=np.array([10.0, 20.0], dtype=np.float64),
            spectrum_type=SpectrumType.PROFILE,
        )
        with pytest.raises(ValueError, match="centroid"):
            spec.remove_precursor_peak(precursor_mz=100.0)

    def test_no_precursor_info_raises(self) -> None:
        spec = _spec()
        with pytest.raises(ValueError, match="precursor_mz is required"):
            spec.remove_precursor_peak()

    def test_auto_from_msn_spectrum(self) -> None:
        import peptacular as pt

        NEUTRON = pt.C13_NEUTRON_MASS
        prec_mz = 500.0
        prec_z = 2
        # Build peaks: precursor at z=2, its isotopes, plus a fragment peak
        mz = np.array(
            [prec_mz, prec_mz + NEUTRON / prec_z, prec_mz + 2 * NEUTRON / prec_z, 250.0],
            dtype=np.float64,
        )
        spec = MsnSpectrum(
            mz=mz,
            intensity=np.ones(4, dtype=np.float64),
            precursors=[Precursor(mz=prec_mz, intensity=1.0, charge=prec_z, is_monoisotopic=True)],
        )
        result = spec.remove_precursor_peak(tolerance=0.01)
        # Only the fragment peak at 250.0 should survive
        assert len(result.mz) == 1
        assert result.mz[0] == pytest.approx(250.0)

    def test_multi_charge_removal(self) -> None:
        import peptacular as pt

        PROTON = pt.PROTON_MASS
        prec_mz = 500.0
        prec_z = 2
        neutral = (prec_mz * prec_z) - (prec_z * PROTON)
        # m/z at z=1
        mz_z1 = (neutral + PROTON) / 1
        # m/z at z=2 = prec_mz
        mz = np.array([mz_z1, prec_mz, 250.0], dtype=np.float64)
        spec = Spectrum(mz=mz, intensity=np.ones(3, dtype=np.float64))
        result = spec.remove_precursor_peak(
            precursor_mz=prec_mz, precursor_charge=prec_z, tolerance=0.01, isotopes=0,
        )
        # Both z=1 and z=2 peaks removed, only 250.0 survives
        assert len(result.mz) == 1
        assert result.mz[0] == pytest.approx(250.0)

    def test_auto_isotopes_uses_distribution(self) -> None:
        import peptacular as pt

        PROTON = pt.PROTON_MASS
        NEUTRON = pt.C13_NEUTRON_MASS
        prec_mz = 500.0
        prec_z = 2
        neutral = (prec_mz * prec_z) - (prec_z * PROTON)
        # Get expected isotope count
        dist = pt.estimate_isotopic_distribution(
            neutral, min_abundance_threshold=0.01, use_neutron_count=True,
        )
        n_isotopes = len(dist)
        # Build peaks: precursor + all expected isotopes at z=2
        mz_list = [prec_mz + i * NEUTRON / prec_z for i in range(n_isotopes)]
        mz_list.append(250.0)  # fragment
        mz = np.array(mz_list, dtype=np.float64)
        spec = Spectrum(mz=mz, intensity=np.ones(len(mz_list), dtype=np.float64))
        result = spec.remove_precursor_peak(
            precursor_mz=prec_mz,
            precursor_charge=prec_z,
            tolerance=0.01,
            isotopes="auto",
            remove_charge_states=False,  # only z=2 to simplify
        )
        assert len(result.mz) == 1
        assert result.mz[0] == pytest.approx(250.0)

    def test_deconvoluted_charge_aware(self) -> None:
        # Deconvoluted spectrum: monoisotopic peaks with known charges
        mz = np.array([300.0, 500.0, 500.0], dtype=np.float64)
        intensity = np.array([100.0, 200.0, 150.0], dtype=np.float64)
        charge = np.array([1, 2, 3], dtype=np.int32)
        spec = Spectrum(mz=mz, intensity=intensity, charge=charge)
        # Only remove the peak at mz=500 with charge=2
        result = spec.remove_precursor_peak(
            precursor_mz=500.0, precursor_charge=2, tolerance=0.01,
        )
        assert len(result.mz) == 2
        # Peak at mz=300 z=1 and mz=500 z=3 should remain
        np.testing.assert_allclose(result.mz, [300.0, 500.0])
        np.testing.assert_array_equal(result.charge, [1, 3])

    def test_decharged_neutral_mass(self) -> None:
        import peptacular as pt

        PROTON = pt.PROTON_MASS
        prec_mz = 500.0
        prec_z = 2
        neutral = (prec_mz * prec_z) - (prec_z * PROTON)
        # Decharged spectrum: m/z = neutral mass, charge = 0
        mz = np.array([neutral, 250.0, 800.0], dtype=np.float64)
        charge = np.array([0, 0, 0], dtype=np.int32)
        spec = Spectrum(mz=mz, intensity=np.ones(3, dtype=np.float64), charge=charge)
        result = spec.remove_precursor_peak(
            precursor_mz=prec_mz, precursor_charge=prec_z, tolerance=0.01,
        )
        assert len(result.mz) == 2
        np.testing.assert_allclose(result.mz, [250.0, 800.0])

    def test_multiple_precursors(self) -> None:
        prec1_mz, prec1_z = 400.0, 2
        prec2_mz, prec2_z = 600.0, 3
        # Place peaks at both precursor m/z values plus a fragment
        mz = np.array([250.0, prec1_mz, prec2_mz], dtype=np.float64)
        spec = MsnSpectrum(
            mz=mz,
            intensity=np.ones(3, dtype=np.float64),
            precursors=[
                Precursor(mz=prec1_mz, intensity=1.0, charge=prec1_z, is_monoisotopic=True),
                Precursor(mz=prec2_mz, intensity=1.0, charge=prec2_z, is_monoisotopic=True),
            ],
        )
        result = spec.remove_precursor_peak(tolerance=0.01, isotopes=0)
        assert len(result.mz) == 1
        assert result.mz[0] == pytest.approx(250.0)

    def test_explicit_mz_bypasses_auto_detection(self) -> None:
        # MsnSpectrum with precursors, but explicit mz overrides
        spec = MsnSpectrum(
            mz=np.array([300.0, 500.0], dtype=np.float64),
            intensity=np.ones(2, dtype=np.float64),
            precursors=[Precursor(mz=300.0, intensity=1.0, charge=2, is_monoisotopic=True)],
        )
        # Only remove 500.0 (not 300.0 from precursors)
        result = spec.remove_precursor_peak(precursor_mz=500.0, tolerance=0.01, isotopes=0)
        assert len(result.mz) == 1
        assert result.mz[0] == pytest.approx(300.0)


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
