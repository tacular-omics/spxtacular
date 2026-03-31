"""
Additional coverage tests for spxtacular.core.
"""

import warnings
from typing import Any

import numpy as np
import pytest

from spxtacular.core import Peak, Spectrum, SpectrumType, _centroid_peaks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spec(
    n: int = 4,
    charge: bool = False,
    im: bool = False,
    iso_score: bool = False,
) -> Spectrum:
    mz = np.linspace(100.0, 400.0, n, dtype=np.float64)
    intensity = np.array([10.0, 50.0, 20.0, 15.0][:n], dtype=np.float64)
    kw = {}
    if charge:
        kw["charge"] = np.array([1, 2, 1, 2][:n], dtype=np.int32)
    if im:
        kw["im"] = np.array([0.9, 1.0, 1.1, 1.2][:n], dtype=np.float64)
    if iso_score:
        kw["iso_score"] = np.array([0.8, 0.9, 0.7, 0.6][:n], dtype=np.float64)
    return Spectrum(mz=mz, intensity=intensity, **kw)


# ---------------------------------------------------------------------------
# _centroid_peaks
# ---------------------------------------------------------------------------


def test_centroid_peaks_fewer_than_four_points_returns_empty() -> None:
    mz = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    intensity = np.array([1.0, 5.0, 1.0], dtype=np.float64)
    c_mz, c_int, c_im = _centroid_peaks(mz, intensity)
    assert len(c_mz) == 0
    assert len(c_int) == 0
    assert c_im is None


def test_centroid_peaks_fewer_than_four_with_im_returns_empty_im() -> None:
    mz = np.array([100.0, 200.0], dtype=np.float64)
    intensity = np.array([1.0, 5.0], dtype=np.float64)
    im = np.array([0.9, 1.0], dtype=np.float64)
    c_mz, c_int, c_im = _centroid_peaks(mz, intensity, im)
    assert len(c_mz) == 0
    assert c_im is not None
    assert len(c_im) == 0


def test_centroid_peaks_gaussian_returns_centroided() -> None:
    x = np.array([490, 492, 494, 496, 498, 499, 500, 501, 502, 504, 506, 508, 510], dtype=np.float64)
    y = np.array([1, 2, 3, 5, 20, 80, 200, 80, 20, 5, 3, 2, 1], dtype=np.float64)
    c_mz, c_int, _ = _centroid_peaks(x, y)
    assert len(c_mz) > 0
    assert float(c_mz[0]) == pytest.approx(500.0, abs=1.0)


def test_centroid_peaks_with_im_returns_im_apex() -> None:
    x = np.linspace(490.0, 510.0, 80, dtype=np.float64)
    y = np.exp(-0.5 * ((x - 500.0) / 1.0) ** 2) * 1000.0
    im = np.ones_like(x) * 1.23
    c_mz, c_int, c_im = _centroid_peaks(x, y, im)
    assert c_im is not None
    assert len(c_im) == len(c_mz)


# ---------------------------------------------------------------------------
# Peak.__repr__
# ---------------------------------------------------------------------------


def test_peak_repr_minimal() -> None:
    p = Peak(mz=123.456, intensity=1000.0)
    r = repr(p)
    assert "mz=123.4560" in r
    assert "int=" in r
    assert "im=" not in r
    assert "score=" not in r
    assert ", z=" not in r


def test_peak_repr_with_charge() -> None:
    p = Peak(mz=200.0, intensity=500.0, charge=2)
    assert "z=2" in repr(p)


def test_peak_repr_with_im() -> None:
    p = Peak(mz=200.0, intensity=500.0, im=1.234)
    assert "im=1.234" in repr(p)


def test_peak_repr_with_iso_score() -> None:
    p = Peak(mz=200.0, intensity=500.0, iso_score=0.987)
    assert "score=0.987" in repr(p)


# ---------------------------------------------------------------------------
# Spectrum validation errors
# ---------------------------------------------------------------------------


def test_spectrum_wrong_length_charge_raises() -> None:
    with pytest.raises(ValueError, match="charge array"):
        Spectrum(
            mz=np.array([1.0, 2.0], dtype=np.float64),
            intensity=np.array([1.0, 2.0], dtype=np.float64),
            charge=np.array([1], dtype=np.int32),
        )


def test_spectrum_wrong_length_im_raises() -> None:
    with pytest.raises(ValueError, match="im array"):
        Spectrum(
            mz=np.array([1.0, 2.0], dtype=np.float64),
            intensity=np.array([1.0, 2.0], dtype=np.float64),
            im=np.array([1.0], dtype=np.float64),
        )


def test_spectrum_wrong_length_iso_score_raises() -> None:
    with pytest.raises(ValueError, match="score array"):
        Spectrum(
            mz=np.array([1.0, 2.0], dtype=np.float64),
            intensity=np.array([1.0, 2.0], dtype=np.float64),
            iso_score=np.array([0.5], dtype=np.float64),
        )


# ---------------------------------------------------------------------------
# top_peaks by non-intensity keys
# ---------------------------------------------------------------------------


def test_top_peaks_by_mz_returns_highest_mz() -> None:
    spec = _spec()
    peaks = spec.top_peaks(2, by="mz")
    mzs = [p.mz for p in peaks]
    assert mzs[0] == pytest.approx(400.0)


def test_top_peaks_by_charge_returns_highest_charge() -> None:
    spec = _spec(charge=True)
    peaks = spec.top_peaks(1, by="charge")
    assert peaks[0].charge == 2


def test_top_peaks_by_im_returns_highest_im() -> None:
    spec = _spec(im=True)
    peaks = spec.top_peaks(1, by="im")
    p = peaks[0]
    assert p.im is not None
    assert p.im == pytest.approx(1.2)


def test_top_peaks_by_score_returns_highest_score() -> None:
    spec = _spec(charge=True, iso_score=True)
    peaks = spec.top_peaks(1, by="score")
    assert peaks[0].iso_score is not None
    assert peaks[0].iso_score == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# _argsort ValueError paths
# ---------------------------------------------------------------------------


def test_argsort_charge_raises_when_no_charge() -> None:
    spec = _spec()
    with pytest.raises(ValueError, match="no charge"):
        _ = spec._argsort_charge


def test_argsort_im_raises_when_no_im() -> None:
    spec = _spec()
    with pytest.raises(ValueError, match="no ion mobility"):
        _ = spec._argsort_im


def test_argsort_score_raises_when_no_score() -> None:
    spec = _spec()
    with pytest.raises(ValueError, match="no score"):
        _ = spec._argsort_score


# ---------------------------------------------------------------------------
# filter — charge, im, top_n
# ---------------------------------------------------------------------------


def test_filter_min_charge_keeps_high_charge_only() -> None:
    spec = _spec(charge=True)
    filtered = spec.filter(min_charge=2)
    assert filtered.charge is not None
    assert all(filtered.charge >= 2)


def test_filter_max_charge_keeps_low_charge_only() -> None:
    spec = _spec(charge=True)
    filtered = spec.filter(max_charge=1)
    assert filtered.charge is not None
    assert all(filtered.charge <= 1)


def test_filter_min_im_keeps_high_im_peaks() -> None:
    spec = _spec(im=True)
    filtered = spec.filter(min_im=1.05)
    assert filtered.im is not None
    assert all(filtered.im >= 1.05)


def test_filter_max_im_keeps_low_im_peaks() -> None:
    spec = _spec(im=True)
    filtered = spec.filter(max_im=1.0)
    assert filtered.im is not None
    assert all(filtered.im <= 1.0)


def test_filter_top_n_keeps_n_highest_intensity_peaks() -> None:
    spec = _spec()
    filtered = spec.filter(top_n=2)
    assert len(filtered.mz) == 2
    assert set(filtered.intensity.tolist()) == {50.0, 20.0}


def test_filter_inplace_modifies_spectrum() -> None:
    spec = _spec()
    original_id = id(spec)
    result = spec.filter(min_mz=200.0, inplace=True)
    assert id(result) == original_id
    assert all(result.mz >= 200.0)


# ---------------------------------------------------------------------------
# normalize — tic, median, warning when already normalized
# ---------------------------------------------------------------------------


def test_normalize_tic_sums_to_one() -> None:
    spec = _spec()
    normed = spec.normalize(method="tic")
    assert float(normed.intensity.sum()) == pytest.approx(1.0)


def test_normalize_median_sets_median_to_one() -> None:
    spec = _spec()
    normed = spec.normalize(method="median")
    assert float(np.median(normed.intensity)) == pytest.approx(1.0)


def test_normalize_already_normalized_emits_warning() -> None:
    spec = _spec()
    normed = spec.normalize(method="max")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        normed.normalize(method="max")
        assert len(w) == 1
        assert "already normalized" in str(w[0].message)


# ---------------------------------------------------------------------------
# denoise — warning when already denoised
# ---------------------------------------------------------------------------


def test_denoise_already_denoised_emits_warning() -> None:
    spec = _spec()
    denoised = spec.denoise(method="mad")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        denoised.denoise(method="mad")
        assert len(w) == 1
        assert "already denoised" in str(w[0].message)


# ---------------------------------------------------------------------------
# centroid — profile to centroid conversion, and warning when already centroid
# ---------------------------------------------------------------------------


def test_centroid_converts_profile_to_centroid_type() -> None:
    x = np.linspace(490.0, 510.0, 80, dtype=np.float64)
    y = np.exp(-0.5 * ((x - 500.0) / 1.0) ** 2) * 1000.0
    spec = Spectrum(mz=x, intensity=y, spectrum_type=SpectrumType.PROFILE)
    result = spec.centroid()
    assert result.spectrum_type == SpectrumType.CENTROID


def test_centroid_already_centroided_emits_warning() -> None:
    spec = Spectrum(
        mz=np.array([100.0, 200.0], dtype=np.float64),
        intensity=np.array([10.0, 20.0], dtype=np.float64),
        spectrum_type=SpectrumType.CENTROID,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        spec.centroid()
        assert len(w) == 1
        assert "already centroided" in str(w[0].message)


# ---------------------------------------------------------------------------
# _apply_mask inplace=True
# ---------------------------------------------------------------------------


def test_apply_mask_inplace_true_modifies_in_place() -> None:
    spec = _spec(charge=True, im=True, iso_score=True)
    mask = np.array([True, False, True, False])
    original_id = id(spec)
    result = spec._apply_mask(mask, inplace=True)
    assert id(result) == original_id
    assert len(result.mz) == 2


# ---------------------------------------------------------------------------
# _apply_index inplace=True
# ---------------------------------------------------------------------------


def test_apply_index_inplace_true_modifies_in_place() -> None:
    spec = _spec(charge=True, im=True, iso_score=True)
    idx = np.array([3, 1, 0, 2], dtype=np.intp)
    original_id = id(spec)
    result = spec._apply_index(idx, inplace=True)
    assert id(result) == original_id
    assert result.mz[0] == pytest.approx(spec.mz[0])


# ---------------------------------------------------------------------------
# sort — by charge, im, score, reverse=True, unknown key raises
# ---------------------------------------------------------------------------


def test_sort_by_charge_ascending() -> None:
    spec = _spec(charge=True)
    sorted_spec = spec.sort(by="charge")
    assert sorted_spec.charge is not None
    assert list(sorted_spec.charge) == sorted(sorted_spec.charge.tolist())


def test_sort_by_im_ascending() -> None:
    spec = _spec(im=True)
    shuffled = spec.sort(by="im", reverse=True)
    resorted = shuffled.sort(by="im")
    assert resorted.im is not None
    assert list(resorted.im) == sorted(resorted.im.tolist())


def test_sort_by_score_ascending() -> None:
    spec = _spec(charge=True, iso_score=True)
    sorted_spec = spec.sort(by="score")
    assert sorted_spec.iso_score is not None
    assert list(sorted_spec.iso_score) == sorted(sorted_spec.iso_score.tolist())


def test_sort_reverse_true_descends() -> None:
    spec = _spec()
    sorted_spec = spec.sort(by="mz", reverse=True)
    mzs = sorted_spec.mz.tolist()
    assert mzs == sorted(mzs, reverse=True)


def test_sort_unknown_key_raises() -> None:
    from typing import cast

    spec = _spec()
    bad: Any = cast(Any, "bogus")
    with pytest.raises(ValueError, match="Unknown sort key"):
        spec.sort(by=bad)


# ---------------------------------------------------------------------------
# copy
# ---------------------------------------------------------------------------


def test_copy_returns_independent_arrays() -> None:
    spec = _spec(charge=True, im=True, iso_score=True)
    c = spec.copy()
    c.mz[0] = 9999.0
    assert spec.mz[0] != 9999.0


def test_copy_charge_is_independent() -> None:
    spec = _spec(charge=True)
    c = spec.copy()
    assert c.charge is not None
    c.charge[0] = 99
    assert spec.charge is not None
    assert spec.charge[0] != 99


# ---------------------------------------------------------------------------
# update inplace=True
# ---------------------------------------------------------------------------


def test_update_inplace_modifies_spectrum() -> None:
    spec = _spec()
    original_id = id(spec)
    new_intensity = np.ones(4, dtype=np.float64)
    result = spec.update(intensity=new_intensity, inplace=True)
    assert id(result) == original_id
    assert all(result.intensity == 1.0)


# ---------------------------------------------------------------------------
# merge — ValueError for invalid tolerance types
# ---------------------------------------------------------------------------


def test_merge_invalid_mz_tolerance_type_raises() -> None:
    from typing import cast

    spec = _spec()
    bad: Any = cast(Any, "invalid")
    with pytest.raises(ValueError, match="mz_tolerance_type"):
        spec.merge(mz_tolerance_type=bad)


def test_merge_invalid_im_tolerance_type_raises() -> None:
    from typing import cast

    spec = _spec()
    bad: Any = cast(Any, "invalid")
    with pytest.raises(ValueError, match="im_tolerance_type"):
        spec.merge(im_tolerance_type=bad)
