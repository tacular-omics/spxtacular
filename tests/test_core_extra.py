"""
Additional coverage tests for spxtacular.core.
"""

import warnings
from typing import Any

import numpy as np
import pytest

from spxtacular.core import MsnSpectrum, Peak, Spectrum, SpectrumType, _centroid_peaks

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


def _multi_spec() -> Spectrum:
    """Five peaks whose five arrays each impose a *different* peak order.

    Every value within a column is distinct and no column is sorted the same
    way as another, so an operation that reordered or masked one array without
    the others cannot coincidentally produce the expected answer.
    """
    return Spectrum(
        mz=np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64),
        intensity=np.array([30.0, 50.0, 10.0, 40.0, 20.0], dtype=np.float64),
        charge=np.array([3, 1, 5, 2, 4], dtype=np.int32),
        im=np.array([1.4, 1.2, 0.8, 1.0, 0.6], dtype=np.float64),
        iso_score=np.array([0.55, 0.95, 0.35, 0.75, 0.15], dtype=np.float64),
    )


def _mzs(spec: Spectrum) -> list[float]:
    """The surviving m/z values, in storage order."""
    return [float(v) for v in spec.mz]


def _rows(spec: Spectrum) -> list[tuple[float, float, int, float, float]]:
    """One tuple per peak, tying each peak's five values together.

    Comparing lists of these tuples is what makes co-permutation testable: a
    row can only stay intact if every array moved with the sort key.
    """
    assert spec.charge is not None
    assert spec.im is not None
    assert spec.iso_score is not None
    return [
        (float(mz), float(inten), int(z), float(im), float(score))
        for mz, inten, z, im, score in zip(spec.mz, spec.intensity, spec.charge, spec.im, spec.iso_score, strict=True)
    ]


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
    # 81 samples put one point exactly on the apex (500.0). An even-sized grid
    # straddles the apex with two *equal* intensities, which the strict
    # ``i_prev < i_curr > i_next`` peak test rejects — that yields zero
    # centroids and makes every assertion below vacuously true.
    x = np.linspace(490.0, 510.0, 81, dtype=np.float64)
    y = np.exp(-0.5 * ((x - 500.0) / 1.0) ** 2) * 1000.0
    # A distinct IM per sample, so "the apex sample's IM" is distinguishable
    # from a neighbour's IM and from any average across the peak.
    im = np.linspace(0.5, 1.5, 81, dtype=np.float64)
    c_mz, c_int, c_im = _centroid_peaks(x, y, im)
    assert len(c_mz) == 1
    assert float(c_mz[0]) == pytest.approx(500.0, abs=0.5)
    assert float(c_int[0]) == pytest.approx(1000.0, rel=0.05)
    assert c_im is not None
    assert len(c_im) == len(c_mz)
    apex = int(np.argmax(y))
    assert float(c_im[0]) == pytest.approx(float(im[apex]))
    # ...and not a neighbour's IM (the grid step is 0.0125, far above the
    # default relative tolerance, so this really discriminates).
    assert float(c_im[0]) != pytest.approx(float(im[apex - 1]))


def test_centroid_peaks_with_im_takes_each_peaks_own_apex() -> None:
    """Two resolved peaks must each report the IM of *their own* apex."""
    x = np.linspace(490.0, 530.0, 161, dtype=np.float64)
    y = np.exp(-0.5 * ((x - 500.0) / 1.0) ** 2) * 1000.0 + np.exp(-0.5 * ((x - 520.0) / 1.0) ** 2) * 500.0
    # IM is flat within each peak but differs between them, so mixing the two
    # peaks' IM values up (or emitting a global apex) is detectable.
    im = np.where(x < 510.0, 0.75, 1.25)
    c_mz, _, c_im = _centroid_peaks(x, y, im)
    assert len(c_mz) == 2
    assert c_im is not None
    assert [float(v) for v in c_im] == [0.75, 1.25]


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
# filter — every criterion, pinned to the exact set of survivors
#
# Every test below asserts the exact surviving m/z values (never just a
# predicate such as ``all(filtered.charge >= 2)``, which an over-aggressive
# filter satisfies vacuously by returning nothing).
# ---------------------------------------------------------------------------


def test_filter_min_mz_keeps_peaks_at_or_above_bound() -> None:
    filtered = _multi_spec().filter(min_mz=300.0)
    assert _mzs(filtered) == [300.0, 400.0, 500.0]  # bound itself is inclusive
    assert all(filtered.mz >= 300.0)


def test_filter_max_mz_keeps_peaks_at_or_below_bound() -> None:
    filtered = _multi_spec().filter(max_mz=300.0)
    assert _mzs(filtered) == [100.0, 200.0, 300.0]  # bound itself is inclusive
    assert all(filtered.mz <= 300.0)


def test_filter_mz_window_keeps_both_boundary_peaks() -> None:
    filtered = _multi_spec().filter(min_mz=200.0, max_mz=400.0)
    assert _mzs(filtered) == [200.0, 300.0, 400.0]


def test_filter_min_intensity_keeps_exact_survivors() -> None:
    filtered = _multi_spec().filter(min_intensity=30.0)
    # intensities are 30/50/10/40/20 — the 30.0 peak is on the bound and stays.
    assert _mzs(filtered) == [100.0, 200.0, 400.0]
    assert all(filtered.intensity >= 30.0)


def test_filter_max_intensity_keeps_exact_survivors() -> None:
    filtered = _multi_spec().filter(max_intensity=30.0)
    assert _mzs(filtered) == [100.0, 300.0, 500.0]
    assert all(filtered.intensity <= 30.0)


def test_filter_min_charge_keeps_exact_survivors() -> None:
    filtered = _multi_spec().filter(min_charge=3)
    # charges are 3/1/5/2/4 — a filter on the wrong column cannot produce this.
    assert _mzs(filtered) == [100.0, 300.0, 500.0]
    assert filtered.charge is not None
    assert all(filtered.charge >= 3)


def test_filter_max_charge_keeps_exact_survivors() -> None:
    filtered = _multi_spec().filter(max_charge=3)
    assert _mzs(filtered) == [100.0, 200.0, 400.0]
    assert filtered.charge is not None
    assert all(filtered.charge <= 3)


def test_filter_charge_window_keeps_exact_survivors() -> None:
    filtered = _multi_spec().filter(min_charge=2, max_charge=4)
    assert _mzs(filtered) == [100.0, 400.0, 500.0]


def test_filter_min_im_keeps_exact_survivors() -> None:
    filtered = _multi_spec().filter(min_im=1.0)
    # im values are 1.4/1.2/0.8/1.0/0.6.
    assert _mzs(filtered) == [100.0, 200.0, 400.0]
    assert filtered.im is not None
    assert all(filtered.im >= 1.0)


def test_filter_max_im_keeps_exact_survivors() -> None:
    filtered = _multi_spec().filter(max_im=1.0)
    assert _mzs(filtered) == [300.0, 400.0, 500.0]
    assert filtered.im is not None
    assert all(filtered.im <= 1.0)


def test_filter_min_score_keeps_exact_survivors() -> None:
    filtered = _multi_spec().filter(min_score=0.5)
    # iso_scores are 0.55/0.95/0.35/0.75/0.15.
    assert _mzs(filtered) == [100.0, 200.0, 400.0]
    assert filtered.iso_score is not None
    assert all(filtered.iso_score >= 0.5)


def test_filter_max_score_keeps_exact_survivors() -> None:
    filtered = _multi_spec().filter(max_score=0.5)
    assert _mzs(filtered) == [300.0, 500.0]
    assert filtered.iso_score is not None
    assert all(filtered.iso_score <= 0.5)


def test_filter_top_n_keeps_n_highest_intensity_peaks() -> None:
    filtered = _multi_spec().filter(top_n=2)
    # highest intensities are 50 (mz 200) and 40 (mz 400); survivors stay in
    # storage order rather than intensity order.
    assert _mzs(filtered) == [200.0, 400.0]


def test_filter_top_n_larger_than_spectrum_keeps_everything() -> None:
    spec = _multi_spec()
    filtered = spec.filter(top_n=99)
    assert _mzs(filtered) == _mzs(spec)


def test_filter_top_n_zero_keeps_nothing() -> None:
    filtered = _multi_spec().filter(top_n=0)
    assert _mzs(filtered) == []


def test_filter_top_n_applies_after_the_other_criteria() -> None:
    # Globally the two most intense peaks are mz 200 (50) and 400 (40); once
    # mz < 250 is excluded first, the winners are 400 (40) and 500 (20).
    filtered = _multi_spec().filter(min_mz=250.0, top_n=2)
    assert _mzs(filtered) == [400.0, 500.0]


def test_filter_combined_criteria_intersect() -> None:
    filtered = _multi_spec().filter(min_charge=2, min_intensity=20.0)
    # charge >= 2 -> mz 100/300/400/500; intensity >= 20 drops mz 300 (10).
    assert _mzs(filtered) == [100.0, 400.0, 500.0]


def test_filter_matching_nothing_returns_empty_spectrum() -> None:
    spec = _multi_spec()
    filtered = spec.filter(min_mz=10_000.0)
    assert _mzs(filtered) == []
    assert filtered.charge is not None
    assert len(filtered.charge) == 0
    assert len(spec.mz) == 5  # original untouched


def test_filter_returns_new_spectrum_without_mutating_original() -> None:
    spec = _multi_spec()
    before = _rows(spec)
    filtered = spec.filter(min_mz=300.0)
    assert filtered is not spec
    assert _rows(spec) == before
    assert _rows(filtered) == before[2:]


def test_filter_inplace_modifies_spectrum() -> None:
    spec = _multi_spec()
    before = _rows(spec)
    result = spec.filter(min_mz=200.0, inplace=True)
    assert result is spec
    assert _rows(spec) == before[1:]  # every array was masked, not just mz


# filter() raises rather than silently ignoring a criterion for a dimension the
# spectrum does not carry — returning every peak would read as "nothing was
# filtered out".


def test_filter_min_charge_without_charge_array_raises() -> None:
    with pytest.raises(ValueError, match="no charge array"):
        _spec().filter(min_charge=2)


def test_filter_max_charge_without_charge_array_raises() -> None:
    with pytest.raises(ValueError, match="no charge array"):
        _spec().filter(max_charge=2)


def test_filter_min_im_without_im_array_raises() -> None:
    with pytest.raises(ValueError, match="no im array"):
        _spec().filter(min_im=1.0)


def test_filter_max_im_without_im_array_raises() -> None:
    with pytest.raises(ValueError, match="no im array"):
        _spec().filter(max_im=1.0)


def test_filter_min_score_without_score_array_raises() -> None:
    with pytest.raises(ValueError, match="no iso_score array"):
        _spec().filter(min_score=0.5)


def test_filter_max_score_without_score_array_raises() -> None:
    with pytest.raises(ValueError, match="no iso_score array"):
        _spec().filter(max_score=0.5)


def test_filter_raises_before_masking_anything() -> None:
    """The guard must fire even when other criteria would have matched."""
    spec = _spec()
    with pytest.raises(ValueError, match="no iso_score array"):
        spec.filter(min_mz=200.0, min_score=0.5, inplace=True)
    assert len(spec.mz) == 4  # inplace filter did not run


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


def test_normalize_unknown_method_raises() -> None:
    """Regression: an unrecognised method fell through to median normalisation
    while recording the caller's spelling, so ``method="Max"`` lied."""
    from typing import cast

    spec = _spec()
    bad: Any = cast(Any, "Max")
    with pytest.raises(ValueError, match="Unknown normalization method"):
        spec.normalize(method=bad)


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


def test_centroid_deconvoluted_raises() -> None:
    """A deconvoluted spectrum has no profile left to fit."""
    spec = Spectrum(
        mz=np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float64),
        intensity=np.array([10.0, 50.0, 20.0, 15.0], dtype=np.float64),
        charge=np.array([1, 2, 1, 2], dtype=np.int32),
        spectrum_type=SpectrumType.DECONVOLUTED,
    )
    with pytest.raises(ValueError, match="requires profile data"):
        spec.centroid()


def test_centroid_clears_the_normalized_flag() -> None:
    """Regression: fitted apex intensities are not the normalised profile
    intensities, so the stale flag blocked re-normalisation."""
    x = np.linspace(490.0, 510.0, 80, dtype=np.float64)
    y = np.exp(-0.5 * ((x - 500.0) / 1.0) ** 2) * 1000.0
    spec = Spectrum(mz=x, intensity=y, spectrum_type=SpectrumType.PROFILE)
    renormalized = spec.normalize().centroid().normalize()

    assert renormalized.normalized == "max"
    assert float(renormalized.intensity.max()) == pytest.approx(1.0)


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


# A sort must *co-permute* every array. Checking only that the sort key comes
# back ordered (``list(s.charge) == sorted(s.charge)``) is self-referential: a
# sort that reordered the key column alone and left mz/intensity/im/iso_score
# where they were would pass. The tests below compare whole peak rows instead.


@pytest.mark.parametrize(
    ("by", "column"),
    [("mz", 0), ("intensity", 1), ("charge", 2), ("im", 3), ("score", 4)],
)
def test_sort_ascending_co_permutes_every_array(by: Any, column: int) -> None:
    spec = _multi_spec()
    before = _rows(spec)
    result = _rows(spec.sort(by=by))

    # 1. no peak was invented, dropped or split: same multiset of rows...
    assert sorted(result) == sorted(before)
    # 2. ...each row still ties its own five values together, and the rows are
    #    ordered by the requested column. Because every value in a column is
    #    unique, (1) + (2) pin the permutation exactly — the expected result is
    #    computed from the *input*, never from the output.
    assert result == sorted(before, key=lambda row: row[column])


@pytest.mark.parametrize(
    ("by", "column"),
    [("mz", 0), ("intensity", 1), ("charge", 2), ("im", 3), ("score", 4)],
)
def test_sort_reverse_co_permutes_every_array(by: Any, column: int) -> None:
    spec = _multi_spec()
    before = _rows(spec)
    result = _rows(spec.sort(by=by, reverse=True))

    assert sorted(result) == sorted(before)
    assert result == sorted(before, key=lambda row: row[column], reverse=True)


def test_sort_returns_new_spectrum_without_mutating_original() -> None:
    spec = _multi_spec()
    before = _rows(spec)
    result = spec.sort(by="intensity")

    assert result is not spec
    assert _rows(spec) == before  # original left in its original order
    assert _rows(result) != before  # and the returned copy really was reordered
    # arrays must not be shared, or a later edit to one would corrupt the other
    result.mz[0] = 9999.0
    assert _rows(spec) == before


def test_sort_inplace_mutates_the_spectrum_and_returns_it() -> None:
    spec = _multi_spec()
    before = _rows(spec)
    result = spec.sort(by="im", inplace=True)

    assert result is spec
    assert _rows(spec) == sorted(before, key=lambda row: row[3])


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


def test_update_inplace_coerces_dtypes_like_construction() -> None:
    """Regression: the inplace path skipped ``__post_init__``, so a list stayed a
    list (breaking the next array operation) and a float charge array stayed
    float64 instead of int32."""
    spec = _spec()
    spec.update(mz=[400.0, 300.0, 200.0, 100.0], charge=np.array([1.0, 2.0, 1.0, 2.0]), inplace=True)

    assert isinstance(spec.mz, np.ndarray)
    assert spec.mz.dtype == np.float64
    assert spec.charge is not None
    assert spec.charge.dtype == np.int32
    assert _mzs(spec.filter(min_mz=250.0)) == [400.0, 300.0]


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


def test_msn_str_repr_with_rt() -> None:
    msn = MsnSpectrum(
        mz=np.array([100.0], dtype=np.float64),
        intensity=np.array([1.0], dtype=np.float64),
        scan_number=7,
        ms_level=2,
        rt=42.5,
        polarity="positive",
    )
    s = str(msn)
    assert "scan=7" in s and "rt=42.50s" in s
    assert repr(msn) == s


def test_msn_str_repr_with_rt_none() -> None:
    """Regression: ``__str__`` used to crash on ``rt=None`` via ``:.2f``."""
    msn = MsnSpectrum(
        mz=np.array([100.0], dtype=np.float64),
        intensity=np.array([1.0], dtype=np.float64),
        scan_number=1,
        ms_level=1,
        rt=None,
    )
    s = str(msn)
    assert "rt=None" in s
    assert repr(msn) == s


# ---------------------------------------------------------------------------
# Centroid intensity threshold and flat apexes
# ---------------------------------------------------------------------------


class TestCentroidThreshold:
    def _noisy_profile(self):
        """Six real Gaussian peaks over a noise floor."""
        mz = np.linspace(400.0, 412.0, 2400)
        peaks = [(402.10, 1e5), (402.60, 5.2e4), (403.10, 1.6e4), (405.85, 7e4), (406.35, 3.1e4), (409.40, 4e4)]
        prof = sum(a * np.exp(-0.5 * ((mz - c) / 0.013) ** 2) for c, a in peaks)
        prof = prof + np.abs(np.random.default_rng(1).normal(0.0, 120.0, mz.size))
        return Spectrum(mz=mz, intensity=prof, spectrum_type=SpectrumType.PROFILE), 6

    def test_without_a_threshold_noise_becomes_peaks(self) -> None:
        """Documents why the threshold exists: every local maximum is a peak."""
        spec, real = self._noisy_profile()
        assert len(spec.centroid()) > 100 * real

    def test_an_absolute_threshold_recovers_the_real_peaks(self) -> None:
        spec, real = self._noisy_profile()
        assert len(spec.centroid(min_intensity=2000)) == real

    def test_noise_threshold_cuts_the_count_dramatically(self) -> None:
        spec, _ = self._noisy_profile()
        assert len(spec.centroid(min_intensity="noise")) < len(spec.centroid()) / 10

    def test_threshold_never_invents_peaks(self) -> None:
        spec, _ = self._noisy_profile()
        assert len(spec.centroid(min_intensity=1e9)) == 0

    def test_a_higher_threshold_never_keeps_more(self) -> None:
        spec, _ = self._noisy_profile()
        counts = [len(spec.centroid(min_intensity=t)) for t in (0, 500, 2000, 10000, 50000)]
        assert counts == sorted(counts, reverse=True)

    def test_default_is_unchanged(self) -> None:
        """The floor is opt-in; existing callers keep their behaviour."""
        spec, _ = self._noisy_profile()
        assert len(spec.centroid()) == len(spec.centroid(min_intensity=None))


class TestCentroidFlatApex:
    def test_a_two_sample_plateau_is_still_a_peak(self) -> None:
        """A strict prev < curr > next test drops these, and they are routine in
        quantised or saturated data."""
        mz = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        plateau = Spectrum(mz=mz, intensity=np.array([1.0, 5.0, 9.0, 9.0, 5.0, 1.0]))
        assert len(plateau.centroid()) == 1

    def test_a_sharp_apex_still_works(self) -> None:
        mz = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        sharp = Spectrum(mz=mz, intensity=np.array([1.0, 5.0, 9.0, 8.0, 5.0, 1.0]))
        assert len(sharp.centroid()) == 1

    def test_a_rising_plateau_is_not_a_peak(self) -> None:
        """Relaxing the test must not turn every flat shoulder into a peak."""
        mz = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        rising = Spectrum(mz=mz, intensity=np.array([1.0, 2.0, 3.0, 3.0, 3.0, 3.0]))
        assert len(rising.centroid()) == 0


# ---------------------------------------------------------------------------
# deconvolute — profile guard, normalized flag
# ---------------------------------------------------------------------------


def _gaussian_profile() -> Spectrum:
    """One ion, drawn as a profile trace."""
    x = np.linspace(499.0, 501.0, 200, dtype=np.float64)
    y = np.exp(-0.5 * ((x - 500.0) / 0.05) ** 2) * 1e4
    return Spectrum(mz=x, intensity=y, spectrum_type=SpectrumType.PROFILE)


def test_deconvolute_profile_raises() -> None:
    """Regression: profile input was processed happily, and every sample of a
    single peak became a candidate — one Gaussian yielded 32 'peaks'."""
    with pytest.raises(ValueError, match="requires centroid data"):
        _gaussian_profile().deconvolute(charge_range=(1, 2))


def test_deconvolute_clears_the_normalized_flag() -> None:
    """Regression: cluster intensities are sums of the input peaks, so the stale
    flag made the following normalize() warn and return unnormalised data."""
    spec = Spectrum(
        mz=np.array([500.0, 500.501, 501.002, 300.0], dtype=np.float64),
        intensity=np.array([100000.0, 70000.0, 30000.0, 1000.0], dtype=np.float64),
        spectrum_type=SpectrumType.CENTROID,
    )
    renormalized = spec.normalize().deconvolute(charge_range=(1, 3)).normalize()

    assert renormalized.normalized == "max"
    assert float(renormalized.intensity.max()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# remove_precursor_peak — tolerance type is case-insensitive
# ---------------------------------------------------------------------------


def test_remove_precursor_peak_uppercase_ppm_is_still_ppm() -> None:
    """Regression: ``tolerance_type="PPM"`` failed a case-sensitive comparison
    and was treated as Da, so 20 ppm became a 20 Da window."""
    from typing import cast

    spec = Spectrum(
        mz=np.array([500.0, 510.0], dtype=np.float64),
        intensity=np.array([1000.0, 2000.0], dtype=np.float64),
        spectrum_type=SpectrumType.CENTROID,
    )
    # The Literal type spells the canonical lowercase values; the case-insensitive
    # spellings are a runtime affordance, so the cast is the point of the test.
    upper: Any = cast(Any, "PPM")
    kept = spec.remove_precursor_peak(precursor_mz=500.0, tolerance=20, tolerance_type=upper, isotopes=0)

    assert _mzs(kept) == [510.0]
