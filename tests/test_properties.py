"""Property-based tests over generated spectra.

Spectra cover sorted and unsorted m/z, empty spectra, duplicate m/z, and zero or
NaN intensity. Set ``HYPOTHESIS_PROFILE=thorough`` (or ``exhaustive``) for a longer run;
the profiles are in ``conftest.py``.
"""

import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from peptacular import IonType

from spxtacular import MgfReader, MsnSpectrum, Precursor, Spectrum, score, write_mgf

MZ = st.floats(min_value=50.0, max_value=3000.0, allow_nan=False, allow_infinity=False)
FINITE_INTENSITY = st.one_of(st.just(0.0), st.floats(min_value=0.0, max_value=1e9, allow_nan=False))
INTENSITY = st.one_of(FINITE_INTENSITY, st.just(float("nan")))


@st.composite
def peak_arrays(draw, *, allow_nan: bool = True, sort: bool | None = None, min_size: int = 0, max_size: int = 40):
    """Draw (mz, intensity). Duplicates come from drawing m/z out of a small pool."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    pool = draw(st.lists(MZ, min_size=1, max_size=max(1, n)))
    mz = np.asarray(draw(st.lists(st.sampled_from(pool), min_size=n, max_size=n)), dtype=np.float64)
    intensity = np.asarray(
        draw(st.lists(INTENSITY if allow_nan else FINITE_INTENSITY, min_size=n, max_size=n)), dtype=np.float64
    )
    if sort is None:
        sort = draw(st.booleans())
    if sort:
        order = np.argsort(mz, kind="stable")
        mz, intensity = mz[order], intensity[order]
    return mz, intensity


@st.composite
def spectra(draw, *, allow_nan: bool = True, sort: bool | None = None, extras: bool = True):
    mz, intensity = draw(peak_arrays(allow_nan=allow_nan, sort=sort))
    n = mz.size
    charge = im = None
    if extras and draw(st.booleans()):
        charge = np.asarray(draw(st.lists(st.sampled_from([-1, 1, 2, 3, 4]), min_size=n, max_size=n)))
    if extras and draw(st.booleans()):
        im = np.asarray(draw(st.lists(st.floats(0.5, 1.5), min_size=n, max_size=n)))
    return Spectrum(
        mz=mz, intensity=intensity, charge=charge, im=im, spectrum_type="centroid" if charge is None else None
    )


def _assert_same(a: Spectrum, b: Spectrum) -> None:
    assert type(a) is type(b)
    for name in ("mz", "intensity", "charge", "im", "iso_score"):
        x, y = getattr(a, name), getattr(b, name)
        if x is None or y is None:
            assert x is None and y is None, name
        else:
            assert x.dtype == y.dtype, name
            np.testing.assert_array_equal(x, y, err_msg=name)
    assert a.spectrum_type == b.spectrum_type


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


@given(spectrum=spectra())
def test_save_load_round_trip(spectrum: Spectrum) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "spectrum"
        spectrum.save(path)
        _assert_same(Spectrum.load(path), spectrum)


@given(spectrum=spectra(allow_nan=False))
def test_json_round_trip(spectrum: Spectrum) -> None:
    restored = Spectrum.from_json(spectrum.to_json())
    _assert_same(restored, spectrum)
    assert restored == spectrum


@given(spectrum=spectra(extras=False))
def test_json_rejects_nan_intensity_or_round_trips(spectrum: Spectrum) -> None:
    # Standard JSON has no NaN, so a spectrum holding one must fail loudly
    # rather than write a file other parsers refuse.
    if np.isnan(spectrum.intensity).any():
        with pytest.raises(ValueError):
            spectrum.to_json()
    else:
        _assert_same(Spectrum.from_json(spectrum.to_json()), spectrum)


@st.composite
def msn_spectra(draw):
    mz, intensity = draw(peak_arrays(allow_nan=False))
    precursors = None
    if draw(st.booleans()):
        precursors = [
            Precursor(
                precursor_mz=draw(MZ),
                intensity=draw(FINITE_INTENSITY),
                charge=draw(st.one_of(st.none(), st.integers(1, 6))),
                is_monoisotopic=None,
            )
        ]
    return MsnSpectrum(
        mz=mz,
        intensity=intensity,
        spectrum_type="centroid",
        scan_number=draw(st.one_of(st.none(), st.integers(1, 10**7))),
        rt=draw(st.one_of(st.none(), st.floats(0.0, 1e5, allow_nan=False))),
        ms_level=2,
        precursors=precursors,
    )


@given(batch=st.lists(msn_spectra(), min_size=1, max_size=4))
def test_mgf_write_read_round_trip(batch: list[MsnSpectrum]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.mgf"
        write_mgf(batch, path)
        with MgfReader(path) as reader:
            restored = list(reader)
    assert len(restored) == len(batch)
    for got, want in zip(restored, batch, strict=True):
        np.testing.assert_array_equal(got.mz, want.mz)
        np.testing.assert_array_equal(got.intensity, want.intensity)
        assert got.scan_number == want.scan_number
        assert got.rt == want.rt
        if want.precursors:
            assert got.precursors is not None
            assert got.precursors[0].precursor_mz == want.precursors[0].precursor_mz
            assert got.precursors[0].intensity == want.precursors[0].intensity
            assert got.precursors[0].charge == want.precursors[0].charge
        else:
            assert not got.precursors


def test_mzml_write_read_round_trip() -> None:
    # spxtacular has no spectrum-to-mzML writer; its one mzML write path re-indexes an
    # existing file. Every spectrum must read back the same through it.
    pytest.importorskip("mzmlpy")
    from spxtacular import write_indexed_mzml_gzip
    from spxtacular.reader import MzmlReader

    source = Path(__file__).parent / "data" / "example.mzML"
    with MzmlReader(source) as reader:
        original = list(reader)
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "example.mzML.gz"
        write_indexed_mzml_gzip(source, output)
        with MzmlReader(output) as reader:
            restored = list(reader)
    assert len(restored) == len(original) > 0
    for got, want in zip(restored, original, strict=True):
        assert got == want


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------


@given(
    spectrum=spectra(),
    bounds=st.tuples(st.one_of(st.none(), MZ), st.one_of(st.none(), MZ)),
    min_intensity=st.one_of(st.none(), st.floats(0.0, 1e9)),
)
def test_filter_is_idempotent(spectrum: Spectrum, bounds, min_intensity) -> None:
    kwargs = {"min_mz": bounds[0], "max_mz": bounds[1], "min_intensity": min_intensity}
    once = spectrum.filter(**kwargs)
    _assert_same(once.filter(**kwargs), once)


@given(spectrum=spectra(), n=st.integers(0, 50))
def test_top_n_is_idempotent(spectrum: Spectrum, n: int) -> None:
    once = spectrum.filter(top_n=n)
    assert len(once.mz) == min(n, len(spectrum.mz))
    _assert_same(once.filter(top_n=n), once)


@given(spectrum=spectra(), n=st.integers(1, 10), width=st.floats(1.0, 500.0))
def test_top_n_per_window_is_idempotent(spectrum: Spectrum, n: int, width: float) -> None:
    once = spectrum.filter(top_n_per_window=(n, width))
    _assert_same(once.filter(top_n_per_window=(n, width)), once)


@given(spectrum=spectra(), by=st.sampled_from(["mz", "intensity"]), reverse=st.booleans())
def test_sort_is_idempotent(spectrum: Spectrum, by: str, reverse: bool) -> None:
    once = spectrum.sort(by=by, reverse=reverse)  # type: ignore[arg-type]
    _assert_same(once.sort(by=by, reverse=reverse), once)  # type: ignore[arg-type]
    if by == "mz" and len(once.mz):
        steps = np.diff(once.mz)
        assert np.all(steps <= 0) if reverse else np.all(steps >= 0)


# ---------------------------------------------------------------------------
# Edge spectra do not crash
# ---------------------------------------------------------------------------


@given(spectrum=spectra(extras=False), method=st.sampled_from(["max", "tic", "median"]))
def test_normalize_does_not_crash(spectrum: Spectrum, method: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = spectrum.normalize(method=method)  # type: ignore[arg-type]
    assert len(out.mz) == len(spectrum.mz)


@given(
    spectrum=spectra(extras=False),
    method=st.sampled_from(["mad", "percentile", "histogram", "baseline", "iterative_median"]),
)
def test_denoise_does_not_crash(spectrum: Spectrum, method: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = spectrum.denoise(method=method)  # type: ignore[arg-type]
    assert len(out.mz) <= len(spectrum.mz)


@given(spectrum=spectra(), tolerance=st.floats(0.0, 1.0))
def test_merge_does_not_crash(spectrum: Spectrum, tolerance: float) -> None:
    out = spectrum.merge(mz_tolerance=tolerance)
    assert len(out.mz) <= len(spectrum.mz)


@given(arrays=peak_arrays(sort=True))
def test_centroid_does_not_crash(arrays) -> None:
    mz, intensity = arrays
    profile = Spectrum(mz=mz, intensity=intensity, spectrum_type="profile")
    out = profile.centroid()
    assert np.all(np.isfinite(out.mz))
    assert np.all(np.isfinite(out.intensity))


@given(spectrum=spectra(extras=False))
def test_deconvolute_does_not_crash(spectrum: Spectrum) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = spectrum.deconvolute(tolerance=20, charge_range=(1, 4))
    assert out.charge is not None
    assert len(out.charge) == len(out.mz)


FRAGMENTS = {(IonType("b"), 1): [98.06004031562, 227.10263340359], (IonType("y"), 1): [148.06043, 263.08737]}


@given(spectrum=spectra(), tolerance=st.floats(0.0, 1.0), unit=st.sampled_from(["da", "ppm"]))
def test_match_fragments_and_score_do_not_crash(spectrum: Spectrum, tolerance: float, unit: str) -> None:
    matches = spectrum.match_fragments(FRAGMENTS, tolerance=tolerance, tolerance_unit=unit)
    assert all(0 <= m.peak_index < len(spectrum.mz) for m in matches)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = score(spectrum, FRAGMENTS, tolerance=tolerance, tolerance_unit=unit)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Regressions: minimised Hypothesis failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reverse", [False, True])
def test_sort_keeps_tied_peaks_in_input_order(reverse: bool) -> None:
    # np.argsort's default quicksort is not stable, and reversing an ascending
    # order flips ties, so sorting a sorted spectrum again reordered tied peaks.
    spectrum = Spectrum(mz=np.full(40, 50.0), intensity=np.arange(40.0))
    for by in ("mz", "intensity"):
        once = spectrum.sort(by=by, reverse=reverse)  # type: ignore[arg-type]
        _assert_same(once.sort(by=by, reverse=reverse), once)  # type: ignore[arg-type]
    np.testing.assert_array_equal(spectrum.sort(by="mz", reverse=reverse).intensity, np.arange(40.0))
    # Two intensity levels, many ties: within a level, m/z must stay ascending.
    levels = np.where(np.arange(40) % 3 == 0, 1.0, 0.0)
    tied = Spectrum(mz=np.arange(100.0, 140.0), intensity=levels).sort(by="intensity", reverse=reverse)
    for level in (0.0, 1.0):
        assert np.all(np.diff(tied.mz[tied.intensity == level]) > 0)


def test_sort_descending_is_idempotent_with_nan_tie() -> None:
    spectrum = Spectrum(mz=np.array([50.0, 50.0]), intensity=np.array([0.0, np.nan]))
    once = spectrum.sort(by="mz", reverse=True)
    _assert_same(once.sort(by="mz", reverse=True), once)
    np.testing.assert_array_equal(once.intensity, [0.0, np.nan])


def test_sort_descending_by_intensity_orders_ties_by_input() -> None:
    spectrum = Spectrum(mz=np.array([100.0, 200.0, 300.0, 400.0]), intensity=np.array([5.0, 9.0, 5.0, 9.0]))
    np.testing.assert_array_equal(spectrum.sort(by="intensity", reverse=True).mz, [200.0, 400.0, 100.0, 300.0])


@pytest.mark.parametrize("method", ["mad", "percentile", "histogram", "baseline", "iterative_median"])
def test_denoise_ignores_nan_intensity(method: str) -> None:
    # One NaN made histogram raise and every other estimator return NaN, and a
    # NaN threshold then removed every peak.
    intensity = np.array([1.0, 2.0, 1.5, np.nan, 1000.0, 1.2, 0.8, 2000.0])
    spectrum = Spectrum(mz=np.arange(100.0, 100.0 + intensity.size), intensity=intensity)
    finite = Spectrum(mz=spectrum.mz[np.isfinite(intensity)], intensity=intensity[np.isfinite(intensity)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        denoised = spectrum.denoise(method=method)  # type: ignore[arg-type]
    np.testing.assert_array_equal(denoised.mz, finite.denoise(method=method).mz)  # type: ignore[arg-type]
    assert 2000.0 in denoised.intensity


def test_denoise_all_nan_spectrum_does_not_crash() -> None:
    spectrum = Spectrum(mz=np.array([50.0]), intensity=np.array([np.nan]))
    assert len(spectrum.denoise(method="histogram").mz) <= 1


@pytest.mark.parametrize("intensity", [[0.0, 5e-324], [1.0, 1.0 + 2**-52], [1e300, np.nextafter(1e300, np.inf)]])
def test_histogram_denoise_on_a_range_too_narrow_to_bin(intensity: list[float]) -> None:
    # numpy cannot cut a range this narrow into 100 finite bins and raised.
    spectrum = Spectrum(mz=np.array([50.0, 50.0]), intensity=np.array(intensity))
    assert len(spectrum.denoise(method="histogram").mz) <= 2
