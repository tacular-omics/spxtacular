"""
Tests for run-level extraction: chromatograms and XICs.

These are the first functions in the library that consume a *run* rather than a
spectrum, which brings two constraints worth pinning: the iterable is walked
exactly once (readers are expensive and may be generators), and the m/z order is
whatever the instrument gave us -- a timsTOF frame is ordered by ion-mobility
scan, so it is not globally sorted.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from spxtacular.chromatogram import Chromatogram, extract_chromatogram, extract_xic
from spxtacular.core import MsnSpectrum, Spectrum
from spxtacular.visualization import plot_chromatogram, plot_xic

TARGETS = [500.0, 700.0, 900.0]
REAL_DDA_DATA = Path(__file__).parent / "data" / "example_dda.d"


def _run(n_scans: int = 12, shuffle: bool = False, with_im: bool = False) -> list[MsnSpectrum]:
    """A synthetic run: each target elutes as a Gaussian at a different time."""
    rng = np.random.default_rng(0)
    scans: list[MsnSpectrum] = []
    for i in range(n_scans):
        rt = 100.0 + i * 10.0
        mz = np.array([*TARGETS, 333.3, 444.4], dtype=np.float64)
        # Each target peaks at a different scan.
        inten = np.array(
            [1e5 * np.exp(-0.5 * ((i - c) / 2.0) ** 2) for c in (2.0, 6.0, 9.0)] + [50.0, 50.0],
            dtype=np.float64,
        )
        im = np.array([1.0, 1.1, 1.2, 0.8, 0.9], dtype=np.float64) if with_im else None
        if shuffle:
            perm = rng.permutation(mz.size)
            mz, inten = mz[perm], inten[perm]
            im = None if im is None else im[perm]
        scans.append(MsnSpectrum(mz=mz, intensity=inten, im=im, ms_level=1, rt=rt))
    return scans


class _CountingIterable:
    """Yields a run once and records how many times iteration was started."""

    def __init__(self, items):
        self._items = items
        self.passes = 0

    def __iter__(self):
        self.passes += 1
        return iter(self._items)


@pytest.fixture(scope="module")
def real_dda_frames() -> list[Spectrum]:
    """Load the large Bruker fixture once for all real-run assertions."""
    pytest.importorskip("tdfpy")
    from spxtacular.reader import DReader

    with DReader(str(REAL_DDA_DATA)) as reader:
        return [
            Spectrum(
                mz=s.mz.copy(),
                intensity=s.intensity.copy(),
                im=None if s.im is None else s.im.copy(),
            )
            for s in reader.ms1
        ]


class TestChromatogramModel:
    def test_list_inputs_are_coerced_to_float_arrays(self) -> None:
        chromatogram = Chromatogram(rt=[1, 2], intensity=[3, 4])  # ty: ignore[invalid-argument-type]

        assert chromatogram.rt.dtype == np.float64
        assert chromatogram.intensity.dtype == np.float64
        assert chromatogram.total == 7.0
        assert chromatogram.apex_rt == 2.0

    def test_mismatched_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            Chromatogram(rt=np.array([1.0, 2.0]), intensity=np.array([3.0]))

    @pytest.mark.parametrize(
        ("field", "rt", "intensity"),
        [
            ("rt", np.array([[1.0, 2.0]]), np.array([3.0, 4.0])),
            ("intensity", np.array([1.0, 2.0]), np.array([[3.0, 4.0]])),
        ],
    )
    def test_non_one_dimensional_arrays_are_rejected(
        self,
        field: str,
        rt: np.ndarray,
        intensity: np.ndarray,
    ) -> None:
        with pytest.raises(ValueError, match=rf"{field} array must be one-dimensional"):
            Chromatogram(rt=rt, intensity=intensity)


# ---------------------------------------------------------------------------
# Chromatogram extraction
# ---------------------------------------------------------------------------


class TestExtractChromatogram:
    def test_tic_sums_each_scan(self) -> None:
        run = _run()
        tic = extract_chromatogram(run)
        assert len(tic) == len(run)
        np.testing.assert_allclose(tic.intensity, [float(s.intensity.sum()) for s in run])

    def test_bpc_takes_each_scan_maximum(self) -> None:
        run = _run()
        bpc = extract_chromatogram(run, mode="bpc")
        np.testing.assert_allclose(bpc.intensity, [float(s.intensity.max()) for s in run])
        # A base peak can never exceed the total.
        assert (bpc.intensity <= extract_chromatogram(run).intensity + 1e-9).all()

    def test_mz_range_restricts_the_sum(self) -> None:
        run = _run()
        full = extract_chromatogram(run)
        narrow = extract_chromatogram(run, mz_range=(690.0, 710.0))
        assert narrow.total < full.total
        assert narrow.total > 0

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode must be"):
            extract_chromatogram(_run(), mode="xic")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    def test_output_is_sorted_by_retention_time(self) -> None:
        run = list(reversed(_run()))
        tic = extract_chromatogram(run)
        assert (np.diff(tic.rt) > 0).all()

    def test_falls_back_to_scan_index_when_rt_is_missing(self) -> None:
        plain = [Spectrum(mz=np.array([100.0]), intensity=np.array([float(i)])) for i in range(4)]
        tic = extract_chromatogram(plain)
        np.testing.assert_allclose(tic.rt, [0.0, 1.0, 2.0, 3.0])

    def test_empty_run(self) -> None:
        assert len(extract_chromatogram([])) == 0

    def test_consumes_the_iterable_exactly_once(self) -> None:
        """Readers are expensive and `reader.ms1` may not be replayable."""
        src = _CountingIterable(_run())
        extract_chromatogram(src)
        assert src.passes == 1


# ---------------------------------------------------------------------------
# XIC extraction
# ---------------------------------------------------------------------------


class TestExtractXic:
    def _brute_force(self, run, target, tol_ppm=20.0):
        out = []
        for s in run:
            tol = target * tol_ppm / 1e6
            out.append(float(s.intensity[np.abs(s.mz - target) <= tol].sum()))
        return np.array(out)

    def test_matches_a_brute_force_reference_exactly(self) -> None:
        run = _run()
        chroms = extract_xic(run, TARGETS, tolerance=20)
        for target, chrom in zip(TARGETS, chroms, strict=True):
            np.testing.assert_array_equal(chrom.intensity, self._brute_force(run, target))

    def test_unsorted_spectra_give_the_same_answer(self) -> None:
        """timsTOF frames are ordered by mobility scan, not by m/z."""
        ordered = extract_xic(_run(shuffle=False), TARGETS, tolerance=20)
        shuffled = extract_xic(_run(shuffle=True), TARGETS, tolerance=20)
        for a, b in zip(ordered, shuffled, strict=True):
            np.testing.assert_allclose(a.intensity, b.intensity)

    def test_every_target_extracted_in_one_pass(self) -> None:
        src = _CountingIterable(_run())
        chroms = extract_xic(src, TARGETS, tolerance=20)
        assert len(chroms) == len(TARGETS)
        assert src.passes == 1, "extracting N targets must not mean N walks of the reader"

    def test_apex_lands_where_the_analyte_elutes(self) -> None:
        # Targets peak at scans 2, 6 and 9 -> rt 120, 160, 190.
        chroms = extract_xic(_run(), TARGETS, tolerance=20)
        assert [c.apex_rt for c in chroms] == [120.0, 160.0, 190.0]

    def test_a_scalar_target_returns_a_single_chromatogram(self) -> None:
        chroms = extract_xic(_run(), 500.0, tolerance=20)
        assert len(chroms) == 1 and chroms[0].mz == pytest.approx(500.0)

    def test_da_and_ppm_windows_differ(self) -> None:
        run = _run()
        tight = extract_xic(run, [500.0], tolerance=1.0, tolerance_type="ppm")[0]
        wide = extract_xic(run, [500.0], tolerance=250.0, tolerance_type="da")[0]
        assert wide.total > tight.total

    def test_max_aggregate_never_exceeds_sum(self) -> None:
        run = _run()
        s = extract_xic(run, TARGETS, tolerance=20, aggregate="sum")
        m = extract_xic(run, TARGETS, tolerance=20, aggregate="max")
        for a, b in zip(m, s, strict=True):
            assert (a.intensity <= b.intensity + 1e-9).all()

    def test_unknown_aggregate_raises(self) -> None:
        with pytest.raises(ValueError, match="aggregate must be"):
            extract_xic(_run(), TARGETS, aggregate="mean")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    def test_ion_mobility_window_restricts_the_trace(self) -> None:
        run = _run(with_im=True)
        full = extract_xic(run, [500.0], tolerance=20)[0]
        # 500.0 sits at im 1.0; a window that excludes it must zero the trace.
        excluded = extract_xic(run, [500.0], tolerance=20, im_window=(1.15, 1.30))[0]
        included = extract_xic(run, [500.0], tolerance=20, im_window=(0.95, 1.05))[0]
        assert full.total > 0
        assert excluded.total == 0.0
        assert included.total == pytest.approx(full.total)

    def test_ion_mobility_window_is_recorded_as_applied(self) -> None:
        chrom = extract_xic(_run(with_im=True), [500.0], tolerance=20, im_window=(0.95, 1.05))[0]
        assert chrom.meta["im_window"] == (0.95, 1.05)
        assert chrom.meta["im_window_applied"] is True
        assert chrom.meta["im_window_skipped"] == 0

    def test_ion_mobility_window_without_im_warns_and_says_so(self) -> None:
        """An IM-less run cannot be gated; the metadata must not claim it was."""
        run = _run(with_im=False)
        with pytest.warns(UserWarning, match="no spectrum carries ion mobility"):
            chrom = extract_xic(run, [500.0], tolerance=20, im_window=(0.0, 0.1))[0]
        # The window excludes everything, yet the trace is the ungated one.
        assert chrom.total == pytest.approx(extract_xic(run, [500.0], tolerance=20)[0].total)
        assert chrom.meta["im_window"] == (0.0, 0.1)
        assert chrom.meta["im_window_applied"] is False
        assert chrom.meta["im_window_skipped"] == len(run)

    def test_ion_mobility_window_warns_about_the_scans_it_could_not_gate(self) -> None:
        mixed = [*_run(n_scans=3, with_im=True), *_run(n_scans=2, with_im=False)]
        with pytest.warns(UserWarning, match="not applied to 2 of 5 spectra"):
            chrom = extract_xic(mixed, [500.0], tolerance=20, im_window=(1.15, 1.30))[0]
        assert chrom.meta["im_window_applied"] is True
        assert chrom.meta["im_window_skipped"] == 2

    def test_no_ion_mobility_window_means_no_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            extract_xic(_run(with_im=False), TARGETS, tolerance=20)

    def test_no_targets_returns_nothing(self) -> None:
        assert extract_xic(_run(), []) == []

    def test_a_target_absent_from_the_run_gives_a_flat_zero_trace(self) -> None:
        chrom = extract_xic(_run(), [1234.5678], tolerance=20)[0]
        assert len(chrom) > 0
        assert chrom.total == 0.0
        assert chrom.apex_rt is not None  # still a well-formed trace


class TestChromatogramObject:
    def test_apex_and_total(self) -> None:
        c = Chromatogram(rt=np.array([1.0, 2.0, 3.0]), intensity=np.array([1.0, 9.0, 2.0]))
        assert c.apex_rt == 2.0
        assert c.total == pytest.approx(12.0)

    def test_empty_chromatogram_has_no_apex(self) -> None:
        c = Chromatogram(rt=np.zeros(0), intensity=np.zeros(0))
        assert c.apex_rt is None
        assert len(c) == 0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


class TestChromatogramPlots:
    def test_accepts_a_single_chromatogram(self) -> None:
        fig = plot_chromatogram(extract_chromatogram(_run()))
        assert len(fig.data) == 1
        assert fig.layout.xaxis.title.text == "Retention time (s)"

    def test_accepts_a_list_of_chromatograms(self) -> None:
        fig = plot_chromatogram(extract_xic(_run(), TARGETS, tolerance=20))
        assert len(fig.data) == len(TARGETS)
        assert fig.layout.showlegend is True

    def test_accepts_raw_spectra_and_extracts_a_tic(self) -> None:
        fig = plot_chromatogram(_run())
        assert len(fig.data) == 1
        assert "TIC" in fig.layout.title.text

    def test_single_trace_is_filled_but_several_are_not(self) -> None:
        """Overlapping washes obscure each other, so fill is for one trace."""
        assert plot_chromatogram(extract_chromatogram(_run())).data[0].fill == "tozeroy"
        multi = plot_chromatogram(extract_xic(_run(), TARGETS, tolerance=20))
        assert all(t.fill is None for t in multi.data)

    def test_apex_is_labelled_for_few_traces_only(self) -> None:
        one = plot_chromatogram(extract_chromatogram(_run()))
        assert any("s" in str(a.text) for a in one.layout.annotations)
        many = plot_chromatogram(extract_xic(_run(), [500.0, 700.0, 900.0, 333.3, 444.4], tolerance=500))
        assert not many.layout.annotations

    def test_plot_xic_extracts_and_plots_in_one_call(self) -> None:
        fig = plot_xic(_run(), TARGETS, tolerance=20)
        assert len(fig.data) == len(TARGETS)
        assert "±20 ppm" in fig.layout.title.text

    def test_plot_xic_walks_the_run_once(self) -> None:
        src = _CountingIterable(_run())
        plot_xic(src, TARGETS, tolerance=20)
        assert src.passes == 1


# ---------------------------------------------------------------------------
# Against real instrument data
# ---------------------------------------------------------------------------


class TestRealRun:
    """A real 65-frame timsTOF run.

    Synthetic runs cannot catch the thing that actually bites here: a Bruker MS1
    frame is ordered by ion-mobility scan, so roughly half its m/z steps descend.
    That is what made a searchsorted-based extraction disagree with brute force
    by 12.9 million when this was first prototyped.
    """

    def test_frames_really_are_unsorted(self, real_dda_frames: list[Spectrum]) -> None:
        """If this ever fails the fixture changed, and the tests below lose their point."""
        assert any(bool(np.any(np.diff(f.mz) < 0)) for f in real_dda_frames)

    def test_xic_matches_brute_force_on_real_frames(self, real_dda_frames: list[Spectrum]) -> None:
        targets = [599.3262, 621.3293, 733.4002]
        chroms = extract_xic(real_dda_frames, targets, tolerance=20, tolerance_type="ppm")
        for target, chrom in zip(targets, chroms, strict=True):
            expected = np.array(
                [float(f.intensity[np.abs(f.mz - target) <= target * 20 / 1e6].sum()) for f in real_dda_frames]
            )
            np.testing.assert_array_equal(chrom.intensity, expected)
            assert chrom.total > 0, "these targets are real peaks from the first frame"

    def test_tic_covers_every_frame(self, real_dda_frames: list[Spectrum]) -> None:
        tic = extract_chromatogram(real_dda_frames)
        assert len(tic) == len(real_dda_frames)
        assert (tic.intensity > 0).all()
