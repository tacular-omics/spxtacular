"""
Tests for the visual theme and the rendering behaviour of the plotting layer.

These assert the properties that actually make a spectrum plot correct and
readable -- trace counts, label thinning, sign of the mirrored half, colour
assignment -- rather than that a Figure object came back. Every check here
corresponds to a real defect the previous implementation had.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spxtacular import theme
from spxtacular.core import Spectrum
from spxtacular.plot_table import build_plot_table, plot_from_table
from spxtacular.visualization import facet_plot, mass_error_plot, mirror_plot, plot_spectrum


def _spectrum(n: int = 6) -> Spectrum:
    rng = np.random.default_rng(0)
    mz = np.sort(rng.uniform(100.0, 1000.0, n))
    return Spectrum(mz=mz, intensity=rng.uniform(1e3, 1e5, n))


def _stick_points(fig) -> int:
    """Count real (non-separator) stick endpoints across every trace."""
    total = 0
    for tr in fig.data:
        if tr.x is None:
            continue
        total += sum(1 for v in tr.x if v is not None and v == v)
    return total


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------


class TestTheme:
    def test_charge_ramp_is_monotone_and_does_not_wrap(self) -> None:
        # Charge is ordinal, so the ramp must never repeat: the old categorical
        # cycle made z=1 and z=11 the same colour.
        colors = [theme.charge_color(z) for z in range(1, 6)]
        assert len(set(colors)) == len(colors)
        assert theme.charge_color(11) == theme.charge_color(5), "beyond the ramp should clamp, not wrap"
        assert theme.charge_color(11) != theme.charge_color(1)

    def test_non_positive_charge_is_neutral(self) -> None:
        assert theme.charge_color(-1) == theme.neutral_color()
        assert theme.charge_color(0) == theme.neutral_color()

    def test_ion_slots_are_unique_and_ordered(self) -> None:
        colors = [theme.ion_color(i) for i in theme._ION_SLOTS]
        assert len(set(colors)) == len(colors)

    def test_unknown_ion_folds_to_neutral_rather_than_a_new_hue(self) -> None:
        assert theme.ion_color("by") == theme.neutral_color()
        assert theme.ion_color("zzz") == theme.neutral_color()

    def test_modes_have_distinct_surfaces_and_hues(self) -> None:
        assert theme.surface("light") != theme.surface("dark")
        assert theme.ion_color("b", "light") != theme.ion_color("b", "dark")

    def test_set_plot_theme_round_trips(self) -> None:
        original = theme.resolve_mode()
        try:
            theme.set_plot_theme("dark")
            assert theme.resolve_mode() == "dark"
            assert theme.surface() == theme.surface("dark")
            assert theme.ion_color("b") == theme.ion_color("b", "dark")
            theme.set_plot_theme("light")
            assert theme.resolve_mode() == "light"
            assert theme.surface() == theme.surface("light")
        finally:
            theme.set_plot_theme(original)

    def test_explicit_mode_overrides_the_global_default(self) -> None:
        original = theme.resolve_mode()
        try:
            theme.set_plot_theme("dark")
            assert theme.surface("light") == theme.surface("light")
            assert theme.surface("light") != theme.surface()
        finally:
            theme.set_plot_theme(original)

    def test_set_plot_theme_rejects_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match=r"light.*dark"):
            theme.set_plot_theme("solarized")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    def test_template_has_no_panel_fill_and_a_recessive_grid(self) -> None:
        tpl = theme.template("light")
        assert tpl.layout.plot_bgcolor == theme.surface("light")
        assert tpl.layout.paper_bgcolor == theme.surface("light")
        # Horizontal grid only; a vertical grid on a spectrum is pure ink.
        assert tpl.layout.yaxis.showgrid is True
        assert tpl.layout.xaxis.showgrid is False


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRendering:
    def test_facet_plot_does_not_make_one_trace_per_peak(self) -> None:
        # This was the pathology: 3000 peaks -> 3000 traces and a figure the
        # browser struggles to render, for the same picture a few traces give.
        spec = _spectrum(400)
        fig = facet_plot(spec)
        assert len(fig.data) < 10, f"expected a handful of grouped traces, got {len(fig.data)}"

    def test_facet_plot_draws_every_peak(self) -> None:
        spec = _spectrum(40)
        fig = facet_plot(spec)
        # Two endpoints per stick (base and tip).
        assert _stick_points(fig) == 2 * len(spec)

    def test_plot_from_table_keeps_rows_with_missing_series_or_color(self) -> None:
        # groupby drops NA keys by default, which silently deleted peaks from
        # the figure -- easy to hit after a merge/reindex on a user-edited table.
        table = build_plot_table(_spectrum(4))
        table.loc[0, "series"] = None
        table.loc[1, "color"] = None
        fig = plot_from_table(table)
        assert _stick_points(fig) == 8

    def test_plot_from_table_rejects_a_missing_required_column(self) -> None:
        table = build_plot_table(_spectrum(3)).drop(columns=["label_size"])
        with pytest.raises(ValueError, match="label_size"):
            plot_from_table(table)

    def test_na_label_is_not_rendered_as_the_text_nan(self) -> None:
        table = build_plot_table(_spectrum(3))
        table["label"] = [np.nan, "keep", ""]
        fig = plot_from_table(table)
        texts = [a.text for a in fig.layout.annotations]
        assert "nan" not in texts
        assert texts == ["keep"]

    def test_single_series_gets_no_legend_box(self) -> None:
        # One colour: the title already says what is plotted.
        fig = plot_spectrum(_spectrum(5), color=None)
        assert fig.layout.showlegend is False


# ---------------------------------------------------------------------------
# Label thinning
# ---------------------------------------------------------------------------


class TestLabelCapping:
    def _scored(self, n: int) -> Spectrum:
        mz = np.linspace(100.0, 2000.0, n)
        return Spectrum(
            mz=mz,
            intensity=np.linspace(1e3, 1e5, n),
            charge=np.ones(n, dtype=np.int32),
            iso_score=np.full(n, 0.9),
        )

    def test_labels_are_capped(self) -> None:
        table = build_plot_table(self._scored(200), max_labels=10)
        assert (table["label"] != "").sum() <= 10

    def test_labels_do_not_collide(self) -> None:
        # Even under the count cap, labels must not stack on top of each other.
        spec = self._scored(200)
        table = build_plot_table(spec, max_labels=None)
        labelled = table.loc[table["label"] != "", "mz"].to_numpy()
        span = float(spec.mz.max() - spec.mz.min())
        gaps = np.diff(np.sort(labelled))
        assert (gaps >= span * 0.02).all(), "labels placed closer than the minimum separation"

    def test_highest_intensity_peak_always_keeps_its_label(self) -> None:
        spec = self._scored(200)
        table = build_plot_table(spec, max_labels=5)
        top = int(np.argmax(spec.intensity))
        assert table.loc[top, "label"] != ""


# ---------------------------------------------------------------------------
# Mirror plot
# ---------------------------------------------------------------------------


class TestMirrorPlot:
    def test_lower_half_is_negated(self) -> None:
        # The defining property of a mirror plot, and previously never asserted.
        raw = Spectrum(mz=np.array([100.0, 200.0]), intensity=np.array([5e4, 2.5e4]))
        dec = raw.deconvolute(charge_range=(1, 2), tolerance=100, tolerance_type="ppm")
        fig = mirror_plot(raw, dec)
        raw_y = [v for v in fig.data[0].y if v == v]
        assert min(raw_y) < 0
        assert max(raw_y) <= 0

    def test_hover_reports_true_intensity_not_the_normalised_value(self) -> None:
        raw = Spectrum(mz=np.array([100.0, 200.0]), intensity=np.array([5e4, 2.5e4]))
        dec = raw.deconvolute(charge_range=(1, 2), tolerance=100, tolerance_type="ppm")
        fig = mirror_plot(raw, dec, normalize=True)
        shown = [v for v in fig.data[0].customdata if v == v]
        assert max(shown) == pytest.approx(5e4)

    def test_all_zero_half_does_not_produce_nan(self) -> None:
        raw = Spectrum(mz=np.array([100.0, 200.0]), intensity=np.zeros(2))
        dec = raw.deconvolute(charge_range=(1, 2), tolerance=100, tolerance_type="ppm")
        fig = mirror_plot(raw, dec, normalize=True)
        ys = [v for v in fig.data[0].y if v == v]
        assert np.isfinite(ys).all()

    def test_charge_colours_match_plot_spectrum(self) -> None:
        # Put the two figures side by side and z=1 must be the same colour.
        mz = np.array([100.0, 100.5, 101.0])
        raw = Spectrum(mz=mz, intensity=np.array([1e5, 5e4, 2e4]))
        dec = raw.deconvolute(charge_range=(1, 3), tolerance=500, tolerance_type="ppm")
        mirror = mirror_plot(raw, dec)
        by_name = {tr.name: tr.line.color for tr in mirror.data if tr.name}
        for name, color in by_name.items():
            if name.startswith("z="):
                assert color == theme.charge_color(int(name[2:]))


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


class TestDegenerateInputs:
    def test_all_zero_intensity_does_not_raise(self) -> None:
        import peptacular as pt

        frags = pt.fragment("PEPTIDE", ion_types="by", charges=[1])
        spec = Spectrum(
            mz=np.sort(np.array([f.mz for f in frags])),
            intensity=np.zeros(len(frags)),
        )
        # Bubble sizes divide by the max intensity; an all-zero match set used to
        # raise ZeroDivisionError, and zero-intensity peaks are real.
        mass_error_plot(spec, frags, tolerance=0.5, tolerance_type="da")
        facet_plot(spec)

    def test_empty_spectrum_renders(self) -> None:
        empty = Spectrum(mz=np.array([]), intensity=np.array([]))
        plot_spectrum(empty)
        facet_plot(empty)

    def test_plot_functions_do_not_mutate_their_input(self) -> None:
        spec = _spectrum(8)
        mz_before = spec.mz.copy()
        int_before = spec.intensity.copy()
        table = build_plot_table(spec)
        table_before = table.copy()
        plot_spectrum(spec)
        plot_from_table(table)
        np.testing.assert_array_equal(spec.mz, mz_before)
        np.testing.assert_array_equal(spec.intensity, int_before)
        pd.testing.assert_frame_equal(table, table_before)
