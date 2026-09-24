"""Tests for the backend-neutral figure layer: styles, rich text, layout, and both renderers.

Content-layer checks run on the :class:`FigureSpec` (``backend="spec"``) and the
resolved layout, so they hold for every backend. The backend tests are smoke
tests: the figure draws and saves, and a few properties that matter for
publication (vector text, embedded TrueType fonts) hold. No pixel comparisons.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import peptacular as pt
import pytest

import spxtacular as spx
from spxtacular._layout import label_boxes, resolve_figure, tick_values
from spxtacular._text import RichText, best_label
from spxtacular.core import MsnSpectrum, Precursor, Spectrum
from spxtacular.errors import SpxtacularError
from spxtacular.figspec import FigureSpec

HAS_MPL = importlib.util.find_spec("matplotlib") is not None
needs_mpl = pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")

PEPTIDE = "PEPTIDEK"


def _psm() -> tuple[MsnSpectrum, list]:
    frags = pt.fragment(PEPTIDE, ion_types=("b", "y"), charges=[1, 2])
    rng = np.random.default_rng(1)
    matched = np.array([f.mz for f in frags])
    noise = rng.uniform(100.0, 900.0, 40)
    mz = np.concatenate([matched + rng.normal(0, 0.002, len(matched)), noise])
    inten = np.concatenate([rng.uniform(2e4, 1e5, len(matched)), rng.uniform(1e3, 1e4, len(noise))])
    order = np.argsort(mz)
    spec = MsnSpectrum(
        mz=mz[order],
        intensity=inten[order],
        ms_level=2,
        precursors=[Precursor(precursor_mz=464.73, intensity=1e6, charge=2, im=None, is_monoisotopic=True)],
    )
    return spec, frags


def _any_overlap(boxes: list[tuple[float, float, float, float]], tol: float = 0.01) -> bool:
    for i, (ax0, ay0, ax1, ay1) in enumerate(boxes):
        for bx0, by0, bx1, by1 in boxes[i + 1 :]:
            if ax0 < bx1 - tol and bx0 < ax1 - tol and ay0 < by1 - tol and by0 < ay1 - tol:
                return True
    return False


# ---------------------------------------------------------------------------
# Styles and sizes
# ---------------------------------------------------------------------------


class TestStyle:
    def test_presets_exist_and_differ(self) -> None:
        paper, screen, talk = (spx.get_style(n) for n in ("paper", "screen", "talk"))
        assert paper.font_size < screen.font_size <= talk.font_size
        assert paper.print_ink and not screen.print_ink

    def test_unknown_style_raises(self) -> None:
        with pytest.raises(SpxtacularError, match="unknown figure style"):
            spx.get_style("poster-glossy")

    def test_with_returns_a_modified_copy(self) -> None:
        paper = spx.get_style("paper")
        small = paper.with_(font_size=6.0)
        assert small.font_size == 6.0 and paper.font_size != 6.0

    @pytest.mark.parametrize(
        ("size", "width_mm"), [("single", 85.0), ("onehalf", 114.0), ("double", 175.0), (120, 120.0)]
    )
    def test_journal_column_widths(self, size, width_mm) -> None:
        spec, _ = _psm()
        fs = spx.plot_spectrum(spec, backend="spec", style="paper", size=size)
        assert fs.width_mm == pytest.approx(width_mm)

    def test_explicit_height(self) -> None:
        spec, _ = _psm()
        fs = spx.plot_spectrum(spec, backend="spec", size=(100, 40))
        assert (fs.width_mm, fs.height_mm) == pytest.approx((100.0, 40.0))

    def test_bad_size_raises(self) -> None:
        spec, _ = _psm()
        with pytest.raises(SpxtacularError, match="size"):
            spx.plot_spectrum(spec, backend="spec", size="quarter")

    def test_default_style_follows_backend(self) -> None:
        spec, _ = _psm()
        assert spx.plot_spectrum(spec, backend="spec").style.name == "paper"


# ---------------------------------------------------------------------------
# Rich text and ion labels
# ---------------------------------------------------------------------------


class TestRichText:
    def test_charge_is_superscript_and_ordinal_subscript(self) -> None:
        label = best_label("b2^2")
        assert label.rich.runs == (("b", "n"), ("2", "sub"), ("2+", "sup"))
        assert label.series == "b"

    def test_neutral_loss_formula_gets_subscripts(self) -> None:
        runs = best_label("y7-H2O").rich.runs
        assert ("2", "sub") in runs[2:]
        assert "\u2212" in "".join(t for t, _ in runs), "a loss uses a real minus sign"

    def test_ambiguous_annotation_prefers_the_simpler_ion(self) -> None:
        assert best_label("b2-H2O,y3^2").rich.text == "y32+"

    def test_html_and_mathtext(self) -> None:
        rich = best_label("y3^2").rich
        assert rich.html() == "y<sub>3</sub><sup>2+</sup>"
        assert rich.mathtext() == r"y$_{\mathrm{3}}$$^{\mathrm{2}{+}}$"

    def test_italic_mz_stays_italic_in_mathtext(self) -> None:
        assert RichText((("m/z", "it"),)).mathtext() == r"$\mathit{m/z}$"

    def test_literal_dollar_cannot_start_mathtext(self) -> None:
        assert RichText.plain("$5").mathtext() == r"\$5"

    def test_scripts_are_narrower_than_plain_text(self) -> None:
        plain = RichText.plain("y32+").width(10)
        scripted = best_label("y3^2").rich.width(10)
        assert scripted < plain


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


class TestLayout:
    def test_value_axes_get_at_least_three_ticks(self) -> None:
        ticks, _ = tick_values(0.0, 1.9, length_pt=100.0, spacing_pt=60.0)
        assert len(ticks) >= 3

    def test_annotated_labels_never_overlap(self) -> None:
        spec, frags = _psm()
        fs = spx.annotate_spectrum(spec, frags, backend="spec", style="paper")
        panel = resolve_figure(fs).panels[0]
        boxes = label_boxes(panel)
        assert boxes
        assert not _any_overlap(boxes)

    def test_labels_stay_inside_the_panel(self) -> None:
        spec, frags = _psm()
        fs = spx.annotate_spectrum(spec, frags, backend="spec", style="paper")
        panel = resolve_figure(fs).panels[0]
        _, _, w, h = panel.rect
        for x0, y0, x1, y1 in label_boxes(panel):
            assert x0 >= -0.5 and x1 <= w + 0.5
            assert y0 >= -0.5 and y1 <= h + 0.5

    def test_sequence_header_adds_a_band_above_the_spectrum(self) -> None:
        spec, frags = _psm()
        plain = spx.annotate_spectrum(spec, frags, backend="spec")
        headed = spx.annotate_spectrum(spec, frags, peptide=PEPTIDE, backend="spec")
        assert headed.cells[0].panels[0].header_height > plain.cells[0].panels[0].header_height
        assert headed.height_mm > plain.height_mm

    def test_error_panel_is_symmetric_around_zero(self) -> None:
        spec, frags = _psm()
        fs = spx.annotate_spectrum(spec, frags, mass_error_panel=True, backend="spec")
        panels = fs.cells[0].panels
        assert len(panels) == 2
        err_axis = panels[1].y
        assert err_axis.lo == pytest.approx(-err_axis.hi)
        assert err_axis.ticks is not None and 0.0 in err_axis.ticks

    def test_error_axis_uses_the_tolerance_when_units_match(self) -> None:
        spec, frags = _psm()
        fs = spx.mass_error_plot(spec, frags, tolerance=15, tolerance_type="ppm", unit="ppm", backend="spec")
        y = fs.cells[0].panels[0].y
        # The tolerance band fits with a little headroom, and the axis stays symmetric.
        assert 15.0 <= y.hi <= 20.0
        assert y.lo == pytest.approx(-y.hi)

    def test_precursor_is_marked(self) -> None:
        spec, _ = _psm()
        fs = spx.plot_spectrum(spec, backend="spec")
        names = {getattr(m, "name", None) for m in fs.cells[0].panels[0].marks}
        assert "precursor" in names
        fs = spx.plot_spectrum(spec, backend="spec", show_precursor=False)
        names = {getattr(m, "name", None) for m in fs.cells[0].panels[0].marks}
        assert "precursor" not in names

    def test_mirror_halves_share_one_scale(self) -> None:
        spec, frags = _psm()
        fs = spx.mirror_plot(spec, spec, fragments=frags, backend="spec")
        panel = resolve_figure(fs).panels[0]
        assert panel.y.lo == pytest.approx(-panel.y.hi)


# ---------------------------------------------------------------------------
# Reporter ions
# ---------------------------------------------------------------------------


class TestReporterIons:
    """The plot draws what :func:`extract_reporter_ions` reads; extraction itself is tested in test_reporter."""

    def _spectrum(self, *, drop: int | None = None) -> Spectrum:
        ions = spx.extract_reporter_ions(Spectrum(mz=np.array([100.0]), intensity=np.array([1.0])), "TMT6")
        mz = [m for i, m in enumerate(ions.reporter_mz) if i != drop]
        inten = [(i + 1) * 1e4 for i in range(6) if i != drop]
        mz.append(127.5)  # an interfering peak between channels
        inten.append(1e6)
        order = np.argsort(mz)
        return Spectrum(mz=np.asarray(mz)[order], intensity=np.asarray(inten)[order])

    def _bars(self, fs: FigureSpec):
        return [m for p in fs.cells[0].panels for m in p.marks if type(m).__name__ == "Bars"]

    def test_one_bar_per_channel_scaled_to_the_strongest(self) -> None:
        fs = spx.reporter_ion_plot(self._spectrum(), "TMT6", backend="spec")
        (bars,) = self._bars(fs)
        assert len(bars.x) == 6
        assert np.asarray(bars.height) == pytest.approx([100 / 6 * k for k in range(1, 7)])

    def test_interfering_peak_is_not_a_channel(self) -> None:
        fs = spx.reporter_ion_plot(self._spectrum(), "TMT6", backend="spec")
        (bars,) = self._bars(fs)
        assert max(bars.height) == pytest.approx(100.0)  # the 1e6 peak at 127.5 is ignored

    def test_missing_channel_is_marked_not_detected(self) -> None:
        fs = spx.reporter_ion_plot(self._spectrum(drop=2), "TMT6", backend="spec")
        texts = [t.text for p in fs.cells[0].panels for m in p.marks if type(m).__name__ == "LabelSet" for t in m.texts]
        assert texts == ["n.d."]

    def test_accepts_extracted_reporter_ions(self) -> None:
        spec = self._spectrum()
        ions = spx.extract_reporter_ions(spec, "TMT6", tolerance=0.005, tolerance_unit="da")
        (bars,) = self._bars(spx.reporter_ion_plot(spec, ions, backend="spec", normalize=False))
        assert np.asarray(bars.height) == pytest.approx(ions.intensity)

    def test_unknown_plex_raises(self) -> None:
        with pytest.raises(SpxtacularError):
            spx.reporter_ion_plot(self._spectrum(), "TMT99", backend="spec")


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class TestBackends:
    def test_unknown_backend_raises(self) -> None:
        spec, _ = _psm()
        with pytest.raises(SpxtacularError, match="backend"):
            spx.plot_spectrum(spec, backend="bokeh")  # ty: ignore[invalid-argument-type]

    def test_spec_backend_returns_a_figure_spec(self) -> None:
        spec, frags = _psm()
        assert isinstance(spx.annotate_spectrum(spec, frags, backend="spec"), FigureSpec)

    def test_plotly_is_the_default(self) -> None:
        spec, _ = _psm()
        fig = spx.plot_spectrum(spec)
        assert type(fig).__module__.startswith("plotly")

    def test_spectrum_methods_forward_backend(self) -> None:
        spec, frags = _psm()
        assert isinstance(spec.plot(backend="spec"), FigureSpec)
        assert isinstance(spec.annotate(frags, backend="spec"), FigureSpec)

    def test_spec_renders_with_plotly(self) -> None:
        spec, frags = _psm()
        fig = spx.annotate_spectrum(spec, frags, peptide=PEPTIDE, backend="spec").render("plotly")
        assert fig.layout.annotations

    @needs_mpl
    @pytest.mark.parametrize(
        "make",
        [
            lambda s, f: spx.plot_spectrum(s, backend="matplotlib"),
            lambda s, f: spx.annotate_spectrum(s, f, peptide=PEPTIDE, mass_error_panel=True, backend="matplotlib"),
            lambda s, f: spx.mirror_plot(s, s, fragments=f, backend="matplotlib"),
            lambda s, f: spx.mass_error_plot(s, f, backend="matplotlib"),
            lambda s, f: spx.sequence_coverage_plot(s, PEPTIDE, f, backend="matplotlib"),
            lambda s, f: spx.facet_plot(s, fragments=f, mirror_spectrum=s, backend="matplotlib"),
            lambda s, f: spx.reporter_ion_plot(s, "TMT6", backend="matplotlib"),
        ],
        ids=["spectrum", "annotated", "mirror", "mass_error", "coverage", "facet", "reporter"],
    )
    def test_matplotlib_draws_every_figure(self, make, tmp_path) -> None:
        spec, frags = _psm()
        fig = make(spec, frags)
        out = spx.save_figure(fig, tmp_path / "fig.png", dpi=72)
        assert out.stat().st_size > 1000

    @needs_mpl
    def test_pdf_embeds_truetype_fonts(self, tmp_path) -> None:
        spec, frags = _psm()
        fig = spx.annotate_spectrum(spec, frags, backend="matplotlib")
        data = spx.save_figure(fig, tmp_path / "fig.pdf").read_bytes()
        assert b"/FontFile2" in data, "fonts must be embedded as TrueType (Type 42), not Type 3"
        assert b"/Subtype /Type3" not in data

    @needs_mpl
    def test_svg_keeps_text_as_text(self, tmp_path) -> None:
        spec, frags = _psm()
        fig = spx.annotate_spectrum(spec, frags, backend="matplotlib")
        svg = spx.save_figure(fig, tmp_path / "fig.svg").read_text()
        assert "<text" in svg

    @needs_mpl
    def test_saving_a_spec_uses_matplotlib(self, tmp_path) -> None:
        spec, _ = _psm()
        out = spx.save_figure(spx.plot_spectrum(spec, backend="spec"), tmp_path / "fig.svg")
        assert "<text" in out.read_text()

    @needs_mpl
    def test_compose_letters_the_parts(self) -> None:
        spec, frags = _psm()
        parts = [
            spx.plot_spectrum(spec, backend="spec"),
            spx.mass_error_plot(spec, frags, backend="spec"),
            spx.reporter_ion_plot(spec, "TMT6", backend="spec"),
        ]
        composed = spx.compose_figure(parts, ncols=2, backend="spec")
        assert [c.letter for c in composed.cells] == ["a", "b", "c"]
        assert composed.width_mm == pytest.approx(175.0)
        fig = spx.compose_figure(parts, labels="ABC", backend="matplotlib")
        texts = {t.get_text() for t in fig.texts}
        assert {"A", "B", "C"} <= texts

    def test_compose_rejects_rendered_figures(self) -> None:
        spec, _ = _psm()
        with pytest.raises(SpxtacularError):
            spx.compose_figure([spx.plot_spectrum(spec)], backend="spec")
