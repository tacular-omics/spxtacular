"""Tests for Spectrum.save() / Spectrum.load() and MsnSpectrum.save() / MsnSpectrum.load()."""

import numpy as np
import pytest

from spxtacular.core import MsnSpectrum, Spectrum, SpectrumType, TargetIon

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _basic_spectrum(**kwargs) -> Spectrum:
    return Spectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype=np.float64),
        intensity=np.array([1000.0, 5000.0, 2000.0], dtype=np.float64),
        spectrum_type=SpectrumType.CENTROID,
        **kwargs,
    )


def _basic_msn(**kwargs) -> MsnSpectrum:
    return MsnSpectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype=np.float64),
        intensity=np.array([1000.0, 5000.0, 2000.0], dtype=np.float64),
        spectrum_type=SpectrumType.CENTROID,
        ms_level=2,
        scan_number=42,
        native_id="scan=42",
        rt=120.5,
        polarity="positive",
        collision_energy=28.0,
        activation_type="HCD",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Spectrum roundtrip
# ---------------------------------------------------------------------------


def test_spectrum_save_load_arrays(tmp_path):
    spec = _basic_spectrum()
    spec.save(tmp_path / "spec")
    restored = Spectrum.load(tmp_path / "spec.npz")
    np.testing.assert_array_equal(restored.mz, spec.mz)
    np.testing.assert_array_equal(restored.intensity, spec.intensity)


def test_spectrum_save_load_spectrum_type(tmp_path):
    spec = _basic_spectrum()
    spec.save(tmp_path / "spec")
    restored = Spectrum.load(tmp_path / "spec.npz")
    assert restored.spectrum_type == spec.spectrum_type


def test_spectrum_save_load_optional_arrays(tmp_path):
    charge = np.array([1, 2, -1], dtype=np.int32)
    im = np.array([0.9, 1.1, 1.05], dtype=np.float64)
    score = np.array([0.8, 0.95, 0.0], dtype=np.float64)
    spec = _basic_spectrum(charge=charge, im=im, score=score)
    spec.save(tmp_path / "spec")
    restored = Spectrum.load(tmp_path / "spec.npz")
    np.testing.assert_array_equal(restored.charge, charge)
    np.testing.assert_array_equal(restored.im, im)
    np.testing.assert_array_equal(restored.score, score)


def test_spectrum_save_load_none_optional_arrays(tmp_path):
    spec = _basic_spectrum()
    spec.save(tmp_path / "spec")
    restored = Spectrum.load(tmp_path / "spec.npz")
    assert restored.charge is None
    assert restored.im is None
    assert restored.score is None


def test_spectrum_save_load_none_spectrum_type(tmp_path):
    spec = Spectrum(
        mz=np.array([100.0], dtype=np.float64),
        intensity=np.array([1000.0], dtype=np.float64),
        spectrum_type=None,
    )
    spec.save(tmp_path / "spec")
    restored = Spectrum.load(tmp_path / "spec.npz")
    assert restored.spectrum_type is None


# ---------------------------------------------------------------------------
# MsnSpectrum roundtrip
# ---------------------------------------------------------------------------


def test_msn_save_load_arrays(tmp_path):
    spec = _basic_msn()
    spec.save(tmp_path / "msn")
    restored = MsnSpectrum.load(tmp_path / "msn.npz")
    np.testing.assert_array_equal(restored.mz, spec.mz)
    np.testing.assert_array_equal(restored.intensity, spec.intensity)


def test_msn_save_load_scalars(tmp_path):
    spec = _basic_msn()
    spec.save(tmp_path / "msn")
    restored = MsnSpectrum.load(tmp_path / "msn.npz")
    assert restored.ms_level == spec.ms_level
    assert restored.scan_number == spec.scan_number
    assert restored.native_id == spec.native_id
    assert restored.rt == pytest.approx(spec.rt)
    assert restored.polarity == spec.polarity
    assert restored.collision_energy == pytest.approx(spec.collision_energy)
    assert restored.activation_type == spec.activation_type


def test_msn_save_load_precursors(tmp_path):
    precursor = TargetIon(mz=500.25, intensity=1e5, charge=2, im=None, score=None, is_monoisotopic=True)
    spec = _basic_msn(precursors=[precursor])
    spec.save(tmp_path / "msn")
    restored = MsnSpectrum.load(tmp_path / "msn.npz")
    assert restored.precursors is not None
    assert len(restored.precursors) == 1
    p = restored.precursors[0]
    assert p.mz == pytest.approx(precursor.mz)
    assert p.charge == precursor.charge
    assert p.is_monoisotopic == precursor.is_monoisotopic


def test_msn_save_load_no_precursors(tmp_path):
    spec = _basic_msn(precursors=None)
    spec.save(tmp_path / "msn")
    restored = MsnSpectrum.load(tmp_path / "msn.npz")
    assert restored.precursors is None


def test_msn_save_load_mz_range(tmp_path):
    spec = _basic_msn(mz_range=(50.0, 1500.0))
    spec.save(tmp_path / "msn")
    restored = MsnSpectrum.load(tmp_path / "msn.npz")
    assert restored.mz_range == pytest.approx((50.0, 1500.0))
