import numpy as np
import pytest

from spxtacular.core import MsnSpectrum, Spectrum, SpectrumType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_spec(mz, intensity, **kwargs):
    return Spectrum(
        mz=np.array(mz, dtype=np.float64),
        intensity=np.array(intensity, dtype=np.float64),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Basic combination
# ---------------------------------------------------------------------------


def test_combine_two_spectra_basic():
    s1 = make_spec([100.0, 200.0], [10.0, 20.0])
    s2 = make_spec([150.0, 300.0], [5.0, 15.0])
    result = Spectrum.combine([s1, s2])

    assert len(result) == 4
    assert np.array_equal(result.mz, [100.0, 150.0, 200.0, 300.0])
    assert np.array_equal(result.intensity, [10.0, 5.0, 20.0, 15.0])


def test_combine_three_spectra():
    s1 = make_spec([100.0], [1.0])
    s2 = make_spec([200.0], [2.0])
    s3 = make_spec([300.0], [3.0])
    result = Spectrum.combine([s1, s2, s3])

    assert len(result) == 3
    assert np.array_equal(result.mz, [100.0, 200.0, 300.0])


def test_combine_sorted_by_mz():
    s1 = make_spec([300.0, 100.0], [3.0, 1.0])
    s2 = make_spec([250.0, 50.0], [2.5, 0.5])
    result = Spectrum.combine([s1, s2])

    assert np.array_equal(result.mz, [50.0, 100.0, 250.0, 300.0])
    assert np.array_equal(result.intensity, [0.5, 1.0, 2.5, 3.0])


# ---------------------------------------------------------------------------
# Single spectrum
# ---------------------------------------------------------------------------


def test_combine_single_spectrum():
    s = make_spec([200.0, 100.0], [2.0, 1.0])
    result = Spectrum.combine([s])

    assert len(result) == 2
    assert np.array_equal(result.mz, [100.0, 200.0])
    assert np.array_equal(result.intensity, [1.0, 2.0])


# ---------------------------------------------------------------------------
# Empty list
# ---------------------------------------------------------------------------


def test_combine_empty_raises():
    with pytest.raises(ValueError):
        Spectrum.combine([])


# ---------------------------------------------------------------------------
# Optional arrays: charge
# ---------------------------------------------------------------------------


def test_combine_all_have_charge():
    s1 = make_spec([100.0], [1.0], charge=np.array([2], dtype=np.int32))
    s2 = make_spec([200.0], [2.0], charge=np.array([3], dtype=np.int32))
    result = Spectrum.combine([s1, s2])

    assert result.charge is not None
    assert np.array_equal(result.charge, [2, 3])


def test_combine_mixed_charge_dropped():
    s1 = make_spec([100.0], [1.0], charge=np.array([2], dtype=np.int32))
    s2 = make_spec([200.0], [2.0])
    result = Spectrum.combine([s1, s2])

    assert result.charge is None


# ---------------------------------------------------------------------------
# Optional arrays: im
# ---------------------------------------------------------------------------


def test_combine_all_have_im():
    s1 = make_spec([100.0], [1.0], im=np.array([0.8]))
    s2 = make_spec([200.0], [2.0], im=np.array([1.2]))
    result = Spectrum.combine([s1, s2])

    assert result.im is not None
    assert np.array_equal(result.im, [0.8, 1.2])


def test_combine_mixed_im_dropped():
    s1 = make_spec([100.0], [1.0], im=np.array([0.8]))
    s2 = make_spec([200.0], [2.0])
    result = Spectrum.combine([s1, s2])

    assert result.im is None


# ---------------------------------------------------------------------------
# Optional arrays: score
# ---------------------------------------------------------------------------


def test_combine_all_have_score():
    s1 = make_spec([100.0], [1.0], score=np.array([0.9]))
    s2 = make_spec([200.0], [2.0], score=np.array([0.7]))
    result = Spectrum.combine([s1, s2])

    assert result.score is not None
    assert np.array_equal(result.score, [0.9, 0.7])


def test_combine_mixed_score_dropped():
    s1 = make_spec([100.0], [1.0], score=np.array([0.9]))
    s2 = make_spec([200.0], [2.0])
    result = Spectrum.combine([s1, s2])

    assert result.score is None


# ---------------------------------------------------------------------------
# Optional arrays sorted together with mz
# ---------------------------------------------------------------------------


def test_combine_optional_arrays_follow_mz_sort():
    s1 = make_spec([300.0], [3.0], im=np.array([1.0]), score=np.array([0.9]))
    s2 = make_spec([100.0], [1.0], im=np.array([0.5]), score=np.array([0.5]))
    result = Spectrum.combine([s1, s2])

    assert np.array_equal(result.mz, [100.0, 300.0])
    assert np.array_equal(result.im, [0.5, 1.0])
    assert np.array_equal(result.score, [0.5, 0.9])


# ---------------------------------------------------------------------------
# Scalar metadata: spectrum_type
# ---------------------------------------------------------------------------


def test_combine_same_spectrum_type_preserved():
    s1 = make_spec([100.0], [1.0], spectrum_type=SpectrumType.CENTROID)
    s2 = make_spec([200.0], [2.0], spectrum_type=SpectrumType.CENTROID)
    result = Spectrum.combine([s1, s2])

    assert result.spectrum_type == SpectrumType.CENTROID


def test_combine_mixed_spectrum_type_becomes_none():
    s1 = make_spec([100.0], [1.0], spectrum_type=SpectrumType.CENTROID)
    s2 = make_spec([200.0], [2.0], spectrum_type=SpectrumType.PROFILE)
    result = Spectrum.combine([s1, s2])

    assert result.spectrum_type is None


# ---------------------------------------------------------------------------
# Scalar metadata: normalized
# ---------------------------------------------------------------------------


def test_combine_same_normalized_preserved():
    s1 = Spectrum(mz=np.array([100.0]), intensity=np.array([1.0]), normalized="max")
    s2 = Spectrum(mz=np.array([200.0]), intensity=np.array([2.0]), normalized="max")
    result = Spectrum.combine([s1, s2])

    assert result.normalized == "max"


def test_combine_mixed_normalized_becomes_none():
    s1 = Spectrum(mz=np.array([100.0]), intensity=np.array([1.0]), normalized="max")
    s2 = Spectrum(mz=np.array([200.0]), intensity=np.array([2.0]), normalized="tic")
    result = Spectrum.combine([s1, s2])

    assert result.normalized is None


def test_combine_all_normalized_none_stays_none():
    s1 = make_spec([100.0], [1.0])
    s2 = make_spec([200.0], [2.0])
    result = Spectrum.combine([s1, s2])

    assert result.normalized is None


# ---------------------------------------------------------------------------
# Scalar metadata: denoised
# ---------------------------------------------------------------------------


def test_combine_same_denoised_preserved():
    s1 = Spectrum(mz=np.array([100.0]), intensity=np.array([1.0]), denoised="mad")
    s2 = Spectrum(mz=np.array([200.0]), intensity=np.array([2.0]), denoised="mad")
    result = Spectrum.combine([s1, s2])

    assert result.denoised == "mad"


def test_combine_mixed_denoised_becomes_none():
    s1 = Spectrum(mz=np.array([100.0]), intensity=np.array([1.0]), denoised="mad")
    s2 = Spectrum(mz=np.array([200.0]), intensity=np.array([2.0]))
    result = Spectrum.combine([s1, s2])

    assert result.denoised is None


# ---------------------------------------------------------------------------
# MsnSpectrum inputs
# ---------------------------------------------------------------------------


def test_combine_msn_returns_base_spectrum():
    s1 = MsnSpectrum(mz=np.array([100.0]), intensity=np.array([1.0]), scan_number=1)
    s2 = MsnSpectrum(mz=np.array([200.0]), intensity=np.array([2.0]), scan_number=2)
    result = Spectrum.combine([s1, s2])

    assert type(result) is Spectrum
    assert not hasattr(result, "scan_number")


# ---------------------------------------------------------------------------
# Zero-peak spectra
# ---------------------------------------------------------------------------


def test_combine_zero_peak_spectrum_with_nonempty():
    s_empty = make_spec([], [])
    s_full = make_spec([100.0, 200.0], [1.0, 2.0])
    result = Spectrum.combine([s_empty, s_full])

    assert len(result) == 2
    assert np.array_equal(result.mz, [100.0, 200.0])


def test_combine_all_zero_peak_spectra():
    s1 = make_spec([], [])
    s2 = make_spec([], [])
    result = Spectrum.combine([s1, s2])

    assert len(result) == 0
