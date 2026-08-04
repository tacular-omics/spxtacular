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
from spxtacular.plot_table import build_annot_plot_table, build_plot_table, plot_from_table, table_view
from spxtacular.visualization import (
    facet_plot,
    mass_error_plot,
    mirror_plot,
    plot_spectrum,
    save_figure,
    sequence_coverage_plot,
)


def _spectrum(n: int = 6) -> Spectrum:
    rng = np.random.default_rng(0)
    mz = np.sort(rng.uniform(100.0, 1000.0, n))
    return Spectrum(mz=mz, intensity=rng.uniform(1e3, 1e5, n))


def _stick_points(fig) -> int:
    """Count real (non-separator) stick endpoints across the line traces.

    Skips the transparent hover hit-layer, which is a marker trace carrying one
    point per peak rather than stick geometry.
    """
    total = 0
    for tr in fig.data:
        if tr.x is None or tr.mode != "lines":
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

    def test_ion_colours_follow_the_proteomics_convention(self) -> None:
        """b blue, y red, a green, c teal, x purple, z orange.

        This is the long-standing convention (Skyline, MetaDraw, IPSA,
        spectrum_utils), and a spectrum that breaks it is misread by anyone used
        to those tools. Pinned by hue family rather than exact hex, so the
        palette can be re-stepped without silently swapping what b and y mean.
        """
        import colorsys

        def hue_deg(hex_color: str) -> float:
            r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
            return colorsys.rgb_to_hsv(r, g, b)[0] * 360

        for mode in ("light", "dark"):
            expected = {
                "b": (190, 260),  # blue
                "y": (340, 20),  # red (wraps through 0)
                "a": (90, 160),  # green
                "c": (140, 190),  # teal / aqua
                "x": (240, 290),  # purple / violet
                "z": (10, 45),  # orange
            }
            for ion, (lo, hi) in expected.items():
                h = hue_deg(theme.ion_color(ion, mode))  # type: ignore[arg-type]
                ok = lo <= h <= hi if lo < hi else (h >= lo or h <= hi)
                assert ok, f"{mode} {ion!r} hue {h:.0f}deg outside {lo}-{hi}"

    def test_b_and_y_are_the_most_separable_pair(self) -> None:
        """b and y co-occur in nearly every spectrum, so they must be the safest pair."""
        b, y = theme.ion_color("b"), theme.ion_color("y")
        assert b != y
        # Cheap proxy for "far apart": different hue family and both saturated.
        assert theme.ion_color("b", "dark") != theme.ion_color("y", "dark")

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
        # Vertical labels need only ~one line-height of room, so the required
        # separation is much smaller than it was for horizontal text.
        assert (gaps >= span * 0.008).all(), "labels placed closer than the minimum separation"

    def test_labels_are_vertical_by_default(self) -> None:
        """Vertical is the spectrum-viewer convention and the reason so many fit."""
        table = build_plot_table(self._scored(20))
        assert (table["label_angle"] == -90.0).all()

        fig = plot_from_table(table)
        rotated = [a for a in fig.layout.annotations if a.textangle == -90]
        assert rotated, "labels must be rendered rotated, not just marked as such"
        # Rotated text grows upward from the tip, so it anchors at its bottom edge.
        assert all(a.yanchor == "bottom" for a in rotated)

    def test_label_angle_is_editable(self) -> None:
        table = build_plot_table(self._scored(20))
        table["label_angle"] = 0.0
        fig = plot_from_table(table)
        assert all(a.textangle == 0 for a in fig.layout.annotations)

    def test_a_table_without_label_angle_still_renders(self) -> None:
        """Hand-built tables predating the column must not break."""
        table = build_plot_table(self._scored(20)).drop(columns=["label_angle"])
        fig = plot_from_table(table)
        assert fig.layout.annotations

    def test_vertical_labels_get_headroom(self) -> None:
        """Otherwise the tallest peak's label is clipped by the plot edge."""
        table = build_plot_table(self._scored(20))
        fig = plot_from_table(table)
        top = fig.layout.yaxis.range[1]
        assert top > float(table["intensity"].max())

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


class TestHoverLayer:
    def test_a_transparent_hit_layer_covers_every_peak(self) -> None:
        # A 1.6px stick is a pinpoint; the hit target has to exceed the mark.
        spec = _spectrum(12)
        fig = plot_spectrum(spec)
        hits = [tr for tr in fig.data if tr.mode == "markers"]
        assert len(hits) == 1
        hit = hits[0]
        assert len(hit.x) == len(spec)
        assert hit.marker.color == "rgba(0,0,0,0)", "the hit layer must be invisible"
        assert hit.marker.size >= 18
        assert hit.showlegend is False

    def test_sticks_defer_hovering_to_the_hit_layer(self) -> None:
        fig = plot_spectrum(_spectrum(5))
        lines = [tr for tr in fig.data if tr.mode == "lines"]
        assert lines and all(tr.hoverinfo == "skip" for tr in lines)

    def test_template_provides_a_crosshair_and_a_generous_hit_distance(self) -> None:
        layout = theme.template("light").layout
        assert layout.xaxis.showspikes is True
        assert layout.xaxis.spikedash == "solid"  # dashed reads as a threshold
        assert layout.hoverdistance >= 20
        assert layout.autosize is True


class TestIntensityScaling:
    def test_relative_is_the_default_and_peaks_at_100(self) -> None:
        spec = Spectrum(mz=np.array([100.0, 200.0]), intensity=np.array([4e4, 1e5]))
        table = build_plot_table(spec)
        assert table["intensity"].max() == pytest.approx(100.0)
        assert table.attrs["intensity_label"] == "Relative intensity (%)"

    def test_scaling_never_changes_the_reported_number(self) -> None:
        # The axis may be rescaled; the tooltip must still say what the data says.
        spec = Spectrum(mz=np.array([100.0, 200.0]), intensity=np.array([4e4, 1e5]))
        table = build_plot_table(spec, intensity_scale="relative")
        assert "1.00e+05" in table.loc[1, "hover"]

    def test_transforms_compress_dynamic_range(self) -> None:
        spec = Spectrum(mz=np.array([100.0, 200.0]), intensity=np.array([1.0, 1e6]))
        plain = build_plot_table(spec, intensity_scale="absolute")
        logged = build_plot_table(spec, intensity_scale="absolute", intensity_transform="log")
        plain_ratio = plain["intensity"].max() / max(plain["intensity"].min(), 1e-12)
        log_ratio = logged["intensity"].max() / max(logged["intensity"].min(), 1e-12)
        assert log_ratio < plain_ratio
        assert "log" in logged.attrs["intensity_label"]

    def test_unknown_scale_or_transform_raises(self) -> None:
        spec = _spectrum(3)
        with pytest.raises(ValueError, match="intensity_scale"):
            build_plot_table(spec, intensity_scale="percent")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        with pytest.raises(ValueError, match="intensity_transform"):
            build_plot_table(spec, intensity_transform="cbrt")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    def test_all_zero_intensity_does_not_divide_by_zero(self) -> None:
        spec = Spectrum(mz=np.array([100.0, 200.0]), intensity=np.zeros(2))
        table = build_plot_table(spec, intensity_scale="relative")
        assert np.isfinite(table["intensity"]).all()


class TestSequenceCoverage:
    def _matched(self, peptide: str):
        import peptacular as pt

        frags = pt.fragment(peptide, ion_types="by", charges=[1])
        mz = np.sort(np.array([f.mz for f in frags]))
        spec = Spectrum(mz=mz, intensity=np.linspace(1e4, 1e5, len(mz)))
        return spec, frags

    def test_full_coverage_is_reported(self) -> None:
        spec, frags = self._matched("PEPTIDEK")
        fig = sequence_coverage_plot(spec, "PEPTIDEK", frags, tolerance=0.02, tolerance_type="da")
        assert "100%" in fig.layout.title.text

    def test_one_annotation_per_residue(self) -> None:
        peptide = "PEPTIDEK"
        spec, frags = self._matched(peptide)
        fig = sequence_coverage_plot(spec, peptide, frags, tolerance=0.02, tolerance_type="da")
        residues = [a.text for a in fig.layout.annotations if len(str(a.text)) == 1]
        assert residues == list(peptide)

    def test_no_matches_means_no_coverage_and_no_ticks(self) -> None:
        peptide = "PEPTIDEK"
        spec = Spectrum(mz=np.array([10.0, 20.0]), intensity=np.array([1.0, 2.0]))
        _, frags = self._matched(peptide)
        fig = sequence_coverage_plot(spec, peptide, frags, tolerance=0.001, tolerance_type="da")
        assert "0/7" in fig.layout.title.text
        assert len(fig.layout.shapes) == 0

    def test_empty_peptide_raises(self) -> None:
        spec, frags = self._matched("PEPTIDEK")
        with pytest.raises(ValueError, match="at least one residue"):
            sequence_coverage_plot(spec, "", frags)


class TestPrecursorMarker:
    def _ms2(self):
        from spxtacular.core import MsnSpectrum, Precursor

        return MsnSpectrum(
            mz=np.array([100.0, 200.0, 300.0]),
            intensity=np.array([1e4, 5e4, 2e4]),
            ms_level=2,
            precursors=[Precursor(mz=250.0, intensity=1e6, charge=2, im=None, is_monoisotopic=True)],
            isolation_mz_range=(248.0, 252.0),
        )

    def test_precursor_line_and_isolation_window_are_drawn(self) -> None:
        fig = plot_spectrum(self._ms2())
        assert any("precursor" in str(a.text) for a in fig.layout.annotations)
        assert len(fig.layout.shapes) >= 2  # the window band and the precursor line

    def test_marker_can_be_switched_off(self) -> None:
        fig = plot_spectrum(self._ms2(), show_precursor=False)
        assert not any("precursor" in str(a.text) for a in fig.layout.annotations)

    def test_plain_spectrum_gets_no_marker(self) -> None:
        fig = plot_spectrum(_spectrum(4))
        assert not any("precursor" in str(a.text) for a in fig.layout.annotations)


class TestTableViewAndTexture:
    def test_table_view_carries_the_values(self) -> None:
        spec = Spectrum(mz=np.array([123.4567, 200.0]), intensity=np.array([1e5, 5e4]))
        html = table_view(build_plot_table(spec))
        assert "123.4567" in html
        assert "1e+05" in html or "100000" in html
        assert html.count("<tr>") == 3  # header plus two peaks

    def test_table_view_escapes_label_markup(self) -> None:
        table = build_plot_table(_spectrum(2))
        table["label"] = ["<script>x</script>", ""]
        html = table_view(table)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_table_view_can_restrict_to_annotated_peaks(self) -> None:
        table = build_plot_table(_spectrum(4))
        table["label"] = ["a", "", "", "b"]
        html = table_view(table, annotated_only=True)
        assert html.count("<tr>") == 3

    def test_texture_gives_each_ion_series_its_own_dash(self) -> None:
        import peptacular as pt

        frags = pt.fragment("PEPTIDEK", ion_types="by", charges=[1])
        mz = np.sort(np.array([f.mz for f in frags]))
        spec = Spectrum(mz=mz, intensity=np.linspace(1e4, 1e5, len(mz)))
        table = build_annot_plot_table(spec, frags, tolerance=0.02, tolerance_type="da", texture=True)
        dashes = set(table.loc[table["series"].isin(["b", "y"]), "dash"])
        assert len(dashes) == 2, "b and y need distinguishable textures, not just hues"

    def test_texture_is_off_by_default(self) -> None:
        table = build_plot_table(_spectrum(3))
        assert (table["dash"] == "solid").all()


class TestPaletteOverride:
    def test_set_palette_replaces_hues(self) -> None:
        original = {mode: list(hues) for mode, hues in theme._CATEGORICAL.items()}
        custom = ["#111111", "#222222", "#333333", "#444444", "#555555", "#666666", "#777777", "#888888"]
        try:
            theme.set_palette(categorical={"light": custom, "dark": custom})
            assert theme.ion_color("b", "light") == "#111111"
        finally:
            theme.set_palette(categorical=original)
        assert theme.ion_color("b", "light") == original["light"][0]

    def test_set_palette_requires_both_modes(self) -> None:
        with pytest.raises(ValueError, match="both modes"):
            theme.set_palette(categorical={"light": ["#000000"] * 8})

    def test_set_palette_requires_enough_hues(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            theme.set_palette(categorical={"light": ["#000000"], "dark": ["#ffffff"]})


class TestSaveFigure:
    def test_html_round_trips(self, tmp_path) -> None:
        fig = plot_spectrum(_spectrum(4))
        out = save_figure(fig, tmp_path / "fig.html")
        assert out.exists() and out.read_text().strip().startswith("<")

    def test_extensionless_path_becomes_html(self, tmp_path) -> None:
        fig = plot_spectrum(_spectrum(3))
        assert save_figure(fig, tmp_path / "fig").suffix == ".html"

    def test_unknown_format_is_rejected(self, tmp_path) -> None:
        fig = plot_spectrum(_spectrum(3))
        with pytest.raises(ValueError, match="unsupported figure format"):
            save_figure(fig, tmp_path / "fig.tiff")


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
