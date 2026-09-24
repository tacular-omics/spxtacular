"""
Figures for mass spectrometry data.

Every function here builds a backend-neutral :class:`~spxtacular.figspec.FigureSpec`
and draws it with the engine you ask for:

``backend="plotly"`` (default)
    An interactive ``plotly.graph_objects.Figure`` with hover on every peak.
``backend="matplotlib"``
    A ``matplotlib.figure.Figure`` for print: vector PDF/SVG with embedded
    TrueType fonts. Needs ``pip install 'spxtacular[matplotlib]'``.
``backend="spec"``
    The :class:`~spxtacular.figspec.FigureSpec` itself, to edit, render later,
    or pass to :func:`~spxtacular.figspec.compose_figure`.

``style=`` picks the typography and line weights (``"screen"``, ``"paper"``,
``"talk"``; see :mod:`spxtacular.style`) and ``size=`` the physical size
(``"single"``, ``"onehalf"``, ``"double"`` journal columns, or millimetres).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from . import theme
from ._text import RichText, best_label
from .chromatogram import Chromatogram
from .core import Spectrum, SpectrumType
from .enums import (
    DEFAULT_FRAGMENT_TOLERANCE,
    DEFAULT_FRAGMENT_TOLERANCE_UNIT,
    PeakSelection,
    PeakSelectionLike,
    check_tolerance_unit,
)
from .errors import SpxtacularError
from .figspec import (
    Axis,
    AxSegments,
    AxText,
    Backend,
    Band,
    Bars,
    Cell,
    Colorbar,
    FigureSpec,
    HitLayer,
    LabelSet,
    Line,
    Mark,
    Panel,
    Points,
    RefLine,
    Sticks,
    check_backend,
    finish,
    new_spec,
)
from .matching import FragmentInput, MatchedFragment, match_fragments
from .plot_table import (
    _HIT_TARGET_SIZE,
    _MAX_LABELS_DEFAULT,
    _PROFILE_MAX_POINTS,
    _cap_labels,
    _charge_series,
    _decimate_profile,
    _fragment_label,
    _scaled_intensity,
    build_annot_plot_table,
    build_plot_table,
    figure_title,
    intensity_axis,
    mz_label,
    table_marks,
    table_panel,
)
from .reporter import (
    DEFAULT_REPORTER_TOLERANCE,
    DEFAULT_REPORTER_TOLERANCE_UNIT,
    ReporterIons,
    extract_reporter_ions,
)
from .style import PT_PER_MM, FigureStyle, SizeLike, StyleName, resolve_size, resolve_style
from .utils import format_precursor_charge, signed_precursor_charge

if TYPE_CHECKING:
    import pandas as pd
    from peptacular.annotation.annotation import ProFormaAnnotation
    from tacular import IsobaricTagInfo
    from tacular.types import ToleranceUnit

    from .reporter import ImpurityTable

StyleLike = StyleName | str | FigureStyle | None

__all__ = [
    "annotate_spectrum",
    "facet_plot",
    "mass_error_plot",
    "mirror_plot",
    "plot_chromatogram",
    "plot_spectrum",
    "plot_xic",
    "profile_centroid_plot",
    "reporter_ion_plot",
    "save_figure",
    "sequence_coverage_plot",
]


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------


def _setup(backend: str, style: StyleLike, theme_mode: theme.ThemeMode | None) -> tuple[str, FigureStyle, Any]:
    key = check_backend(backend)
    return key, resolve_style(style, key), theme.resolve_mode(theme_mode)


def _build(
    cell: Cell,
    *,
    key: str,
    fig_style: FigureStyle,
    size: SizeLike,
    mode: theme.ThemeMode,
    layout_kwargs: dict[str, Any],
) -> Any:
    spec = new_spec(cell, style=fig_style, backend=key, size=size, theme_mode=mode, layout_kwargs=layout_kwargs)
    return finish(spec, key)


def _first_precursor_mz(spectrum: Spectrum) -> float | None:
    precursors = getattr(spectrum, "precursors", None) or []
    return float(precursors[0].precursor_mz) if precursors else None


def _precursor_label(mz: float, charge: int | None, polarity: Any, style: FigureStyle) -> RichText:
    """``precursor 500.2500 (2+)`` on screen, ``[M+2H]²⁺`` in print."""
    if style.name == "screen":
        charge_text = format_precursor_charge(charge, polarity)
        return RichText.plain(f"precursor {mz:.4f}" + (f" ({charge_text})" if charge_text is not None else ""))
    signed = signed_precursor_charge(charge, polarity)
    if signed is None or signed == 0:
        return RichText.plain("precursor")
    z = abs(signed)
    sign = "+" if signed > 0 else "\u2212"
    count = "" if z == 1 else str(z)
    return RichText((("[M" + sign + count + "H]", "n"), (f"{count}{'+' if signed > 0 else '\u2212'}", "sup")))


def _precursor_marks(spectrum: Spectrum, style: FigureStyle, mode: theme.ThemeMode) -> list[Mark]:
    """The isolation window as a faint band and each precursor as a hairline.

    Reference furniture, not data: both sit behind the peaks. Nothing is drawn
    for a spectrum without precursor information.
    """
    precursors = getattr(spectrum, "precursors", None)
    if not precursors:
        return []
    muted = theme.text_color("muted", mode)
    marks: list[Mark] = []
    window = getattr(spectrum, "isolation_mz_range", None)
    if window is not None and len(window) == 2:
        lo, hi = float(window[0]), float(window[1])
        if hi > lo:
            marks.append(Band("v", lo, hi, muted, alpha=0.08))
    for prec in precursors:
        mz_val = getattr(prec, "precursor_mz", None)
        if mz_val is None:
            continue
        marks.append(
            RefLine(
                "v",
                float(mz_val),
                muted,
                width=style.axis_width,
                dash="dash" if style.print_ink else "solid",
                label=_precursor_label(
                    float(mz_val), getattr(prec, "charge", None), getattr(spectrum, "polarity", None), style
                ),
                label_color=theme.text_color("secondary", mode),
                name="precursor",
            )
        )
    return marks


def save_figure(
    fig: Any,
    path: str | Path,
    *,
    scale: float | None = None,
    dpi: float | None = None,
    **kwargs: Any,
) -> Path:
    """Write a figure to disk, choosing the writer from the file extension.

    Works for plotly figures, matplotlib figures and figure specs (a spec is
    drawn with matplotlib when it is installed, else plotly).

    ``.pdf`` and ``.svg`` are vector output; matplotlib embeds the fonts as
    TrueType (Type 42), which journals accept and which stays editable in
    Illustrator or Inkscape. ``.png`` is rendered at the figure style's
    resolution (600 dpi for ``"paper"``) unless ``dpi`` says otherwise.
    ``.html`` is plotly only and always works. Plotly static export needs
    ``kaleido``: ``pip install 'spxtacular[plotly-export]'``.

    Parameters
    ----------
    fig:
        A plotly figure, a matplotlib figure, or a :class:`~spxtacular.figspec.FigureSpec`.
    path:
        Destination. The suffix picks the format.
    scale:
        Plotly raster only: device pixel ratio. Overrides ``dpi``.
    dpi:
        Raster resolution in dots per inch. Defaults to the figure style's ``dpi``.
    **kwargs:
        Forwarded to the backend writer (``write_html``/``write_image`` or ``savefig``).

    Returns
    -------
    The path written.
    """
    out = Path(path)
    suffix = out.suffix.lower()

    if isinstance(fig, FigureSpec):
        backend = "matplotlib" if importlib.util.find_spec("matplotlib") is not None else "plotly"
        if suffix in ("", ".html"):
            backend = "plotly"
        fig = fig.render(backend)  # type: ignore[arg-type]

    if hasattr(fig, "savefig") and not hasattr(fig, "write_image"):
        mpl_formats = (".png", ".svg", ".pdf", ".eps", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")
        if suffix not in mpl_formats:
            raise SpxtacularError(
                f"unsupported format {suffix!r} for a matplotlib figure; expected one of {', '.join(mpl_formats)}"
            )
        if dpi is not None:
            kwargs.setdefault("dpi", dpi)
        fig.savefig(str(out), **kwargs)
        return out

    if suffix in ("", ".html"):
        out = out.with_suffix(".html")
        fig.write_html(str(out), **kwargs)
        return out

    static = (".png", ".svg", ".pdf", ".jpg", ".jpeg", ".webp")
    if suffix not in static:
        raise SpxtacularError(f"unsupported figure format {suffix!r}; expected .html or one of {', '.join(static)}")

    try:
        importlib.import_module("kaleido")
    except (ImportError, OSError) as exc:
        raise ImportError(
            f"writing {suffix} requires the kaleido package: pip install 'spxtacular[plotly-export]' "
            "(or save to .html, which needs nothing extra)"
        ) from exc
    layout = getattr(fig, "layout", None)
    meta = getattr(layout, "meta", None)
    meta = meta if isinstance(meta, dict) else {}
    if scale is None:
        target_dpi = dpi if dpi is not None else meta.get("spx_dpi")
        # Plotly lays out in CSS pixels at 96 per inch.
        scale = float(target_dpi) / 96.0 if target_dpi else 2.0
    if getattr(layout, "width", None) is None and "spx_width" in meta:
        # An autosized screen figure has no width until a browser gives it one;
        # export it at its design width rather than plotly's 700 px default.
        kwargs.setdefault("width", meta["spx_width"])
    fig.write_image(str(out), scale=scale, **kwargs)
    return out


def _ion_type(fragment: Any) -> str:
    ion = getattr(fragment, "ion_type", "")
    return str(ion.value if hasattr(ion, "value") else ion)


def _unit_of(tolerance_unit: ToleranceUnit) -> ToleranceUnit:
    return check_tolerance_unit(tolerance_unit)


def _error_unit(unit: str) -> Literal["ppm", "da"]:
    """Normalise a mass-error unit to ``"ppm"`` or ``"da"``, rejecting anything else."""
    normalised = str(unit).lower()
    if normalised not in ("ppm", "da"):
        raise SpxtacularError(f"Unsupported error unit {unit!r}; expected 'ppm' or 'da'")
    return normalised


def _error_title(unit: str, short: bool = False) -> RichText:
    """``Mass error (ppm)``; ``Error (ppm)`` for a strip too short to hold the long form."""
    return RichText.plain(f"{'Error' if short else 'Mass error'} ({'ppm' if unit == 'ppm' else 'Da'})")


def _error_marks(
    matches: Sequence[MatchedFragment],
    unit: str,
    *,
    style: FigureStyle,
    mode: theme.ThemeMode,
    labels: bool,
    max_labels: int | None,
    max_size: float,
) -> list[Mark]:
    """Mass-error dots (area proportional to intensity), a zero line, and optional labels."""
    muted = theme.text_color("muted", mode)
    marks: list[Mark] = [RefLine("h", 0.0, muted, width=style.axis_width * 0.8, name="zero")]
    if not matches:
        return marks
    mzs = np.asarray([m.peak_mz for m in matches], dtype=np.float64)
    errors = np.asarray([m.ppm_error if unit == "ppm" else m.da_error for m in matches], dtype=np.float64)
    inten = np.asarray([m.peak_intensity for m in matches], dtype=np.float64)
    top = float(inten.max()) if len(inten) else 1.0
    top = top or 1.0
    rel = np.clip(inten / top, 0.0, 1.0)
    # Area, not diameter, tracks intensity, so a 4x stronger peak does not look 16x bigger.
    min_size = max_size * 0.28
    sizes = min_size + (max_size - min_size) * np.sqrt(rel)
    ion_types = [_ion_type(m.fragment) for m in matches]
    colors = [theme.ion_color(t, mode) for t in ion_types]
    names = [_fragment_label(m.fragment, False) for m in matches]
    order = np.argsort(-inten)  # small dots drawn last, on top
    marks.append(
        Points(
            x=mzs[order],
            y=errors[order],
            sizes=sizes[order],
            colors=[colors[i] for i in order],
            outline=theme.marker_outline(mode),
            outline_width=max(0.3, style.axis_width * 0.6),
            opacity=0.85,
            name="errors",
            customdata=[[float(inten[i]), names[i]] for i in order],
            hovertemplate=(
                f"m/z: %{{x:.4f}}<br>error ({'ppm' if unit == 'ppm' else 'Da'}): %{{y:.4f}}<br>"
                "intensity: %{customdata[0]:.2e}<br>%{customdata[1]}<extra></extra>"
            ),
        )
    )
    if labels:
        capped = _cap_labels(list(names), inten, max_labels)
        keep = [i for i, t in enumerate(capped) if t]
        if keep:
            parsed = [best_label(names[i]) for i in keep]
            marks.append(
                LabelSet(
                    x=mzs[keep],
                    y=errors[keep],
                    texts=[p.rich for p in parsed],
                    colors=[theme.label_color(colors[i], mode) if style.label_series_color else muted for i in keep],
                    priority=np.asarray([p.priority * (0.2 + rel[i]) for p, i in zip(parsed, keep, strict=True)]),
                    size=style.label_size,
                    anchor_offset=sizes[keep] / 2.0,
                    gap=style.label_gap,
                )
            )
    return marks


def _errors(matches: Sequence[MatchedFragment], unit: str) -> list[float]:
    return [m.ppm_error if unit == "ppm" else m.da_error for m in matches]


def _nice_ceil(value: float) -> float:
    """Smallest 1, 2, 2.5 or 5 times a power of ten that is >= ``value``."""
    if value <= 0 or not np.isfinite(value):
        return 1.0
    exp = np.floor(np.log10(value))
    for step in (1.0, 2.0, 2.5, 5.0, 10.0):
        if step * 10**exp >= value * (1 - 1e-9):
            return float(step * 10**exp)
    return float(10 ** (exp + 1))


def _error_axis(
    unit: str,
    tolerance: float | None,
    tolerance_unit: str | None,
    errors: Sequence[float] | NDArray[np.float64] = (),
    *,
    short: bool = False,
) -> Axis:
    """Symmetric error axis with ticks at -span, 0, +span.

    The span is the matching tolerance when it is in the displayed unit (the
    window edges are the numbers a reader checks), else a round number just
    above the largest error.
    """
    if tolerance is not None and tolerance_unit == unit and tolerance > 0:
        span = float(tolerance)
    else:
        finite = np.abs(np.asarray(errors, dtype=np.float64))
        finite = finite[np.isfinite(finite)]
        span = _nice_ceil(float(finite.max()) if len(finite) else 0.0)
    text = f"{span:g}"
    return Axis(
        label=_error_title(unit, short),
        lo=-span * 1.15,
        hi=span * 1.15,
        ticks=[-span, 0.0, span],
        ticktext=[f"\u2212{text}", "0", text],
        zeroline=False,
    )


# ---------------------------------------------------------------------------
# Spectrum plots
# ---------------------------------------------------------------------------


def _im_panel(
    spectrum: Spectrum,
    *,
    show_scores: bool,
    max_labels: int | None,
    style: FigureStyle,
    mode: theme.ThemeMode,
    intensity_scale: Literal["absolute", "relative"],
    intensity_transform: Literal["sqrt", "log"] | None,
    absolute_axis: bool,
) -> Panel:
    """Sticks coloured by ion mobility, quantised into 20 bins of a single-hue ramp.

    Each stick takes one flat colour and the colour bar carries the mapping.
    Intensity is scaled as the plot-table path scales it; the hover always
    reports the unscaled value.
    """
    import plotly.colors as pc

    mz = np.asarray(spectrum.mz, dtype=np.float64)
    intensity = np.asarray(spectrum.intensity, dtype=np.float64)
    plotted, intensity_label = _scaled_intensity(intensity, intensity_scale, intensity_transform)
    im = spectrum.im
    assert im is not None

    im_label = getattr(spectrum, "im_type", None) or "im"
    if im_label == "ook0":
        im_label = "1/K0"

    im_arr = np.asarray(im, dtype=np.float64)
    n_bins = 20
    if len(im_arr) == 0 or np.all(np.isnan(im_arr)):
        im_min = im_max = 0.0
        norm = np.zeros(len(im_arr))
    else:
        im_min, im_max = float(np.nanmin(im_arr)), float(np.nanmax(im_arr))
        norm = np.zeros(len(im_arr)) if im_min == im_max else np.nan_to_num((im_arr - im_min) / (im_max - im_min))
    bin_idx = np.clip((norm * n_bins).astype(int), 0, n_bins - 1)
    # Single-hue sequential ramp rather than Viridis: ion mobility is a
    # magnitude, and a multi-hue ramp invents banding that is not in the data.
    scale = theme.sequential_scale(mode)
    bin_colors: list[str] = pc.sample_colorscale(scale, n_bins, colortype="rgb")
    bin_hex = [_rgb_to_hex(c) for c in bin_colors]

    marks: list[Mark] = []
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        hover = [
            f"m/z: {m:.4f}<br>intensity: {i:.2e}<br>{im_label}: {v:.4f}"
            for m, i, v in zip(mz[mask], intensity[mask], im_arr[mask], strict=True)
        ]
        marks.append(
            Sticks(
                x=mz[mask],
                y=plotted[mask],
                color=bin_hex[b],
                width=style.stick_width,
                customdata=hover,
                hovertemplate="%{customdata}<extra></extra>",
            )
        )

    if show_scores and spectrum.iso_score is not None:
        texts = _cap_labels(
            [f"{float(s):.2f}" if float(s) > 0.0 else "" for s in spectrum.iso_score], intensity, max_labels
        )
        keep = [i for i, t in enumerate(texts) if t]
        if keep:
            marks.append(
                LabelSet(
                    x=mz[keep],
                    y=plotted[keep],
                    texts=[RichText.plain(texts[i]) for i in keep],
                    colors=[theme.text_color("secondary", mode)] * len(keep),
                    priority=intensity[keep],
                    size=style.label_size,
                    gap=style.label_gap,
                )
            )
    relative = intensity_scale == "relative" and intensity_transform is None
    y_axis, secondary = intensity_axis(
        intensity_label,
        relative=relative,
        base_peak=float(intensity.max()) if len(intensity) else None,
        absolute_axis=absolute_axis,
    )
    return Panel(
        marks=marks,
        x=Axis(label=mz_label()),
        y=y_axis,
        y_secondary=secondary,
        colorbar=Colorbar(lo=im_min, hi=im_max, scale=scale, title=RichText.plain(im_label)),
    )


def _rgb_to_hex(color: str) -> str:
    if color.startswith("#"):
        return color
    inner = color[color.index("(") + 1 : color.index(")")]
    parts = [float(p) for p in inner.split(",")[:3]]
    return "#" + "".join(f"{round(p):02x}" for p in parts)


def plot_spectrum(
    spectrum: Spectrum,
    *,
    title: str | None = None,
    color: Literal["charge", "im"] | None = "charge",
    show_scores: bool = True,
    max_labels: int | None = _MAX_LABELS_DEFAULT,
    theme_mode: theme.ThemeMode | None = None,
    intensity_scale: Literal["absolute", "relative"] = "relative",
    intensity_transform: Literal["sqrt", "log"] | None = None,
    show_precursor: bool = True,
    render: Literal["sticks", "profile"] | None = None,
    max_points: int | None = _PROFILE_MAX_POINTS,
    absolute_axis: bool = False,
    backend: Backend = "plotly",
    style: StyleLike = None,
    size: SizeLike = None,
    **layout_kwargs: Any,
) -> Any:
    """Plot a spectrum: sticks for centroid data, a continuous trace for profile.

    Parameters
    ----------
    spectrum:
        Spectrum to plot.
    title:
        Plot title. Defaults to the spectrum type in the ``"screen"`` and
        ``"talk"`` styles; the ``"paper"`` style draws a title only when given one.
    color:
        ``"charge"`` (default) colours sticks by charge state on an ordinal
        ramp when charge data is present. ``"im"`` colours sticks by ion
        mobility on a 20-step single-hue ramp with a colour bar; it falls back
        to ``"charge"`` without an IM array, and is rejected for profile data
        (centroid first, or pass ``render="sticks"``). ``None`` draws every
        stick in one colour.
    show_scores:
        Label peaks with their isotope-profile score (score > 0 only).
    max_labels:
        Cap on direct labels, highest intensity first (default 60). The layout
        also drops labels that cannot be placed without an overlap; every value
        stays on hover and in :func:`~spxtacular.plot_table.build_plot_table`.
    theme_mode:
        ``"light"`` or ``"dark"``. Defaults to the global plot theme.
    intensity_scale:
        ``"relative"`` scales the base peak to 100 %. ``"absolute"`` keeps raw values.
    intensity_transform:
        Optional ``"sqrt"`` or ``"log"`` display transform.
    show_precursor:
        Mark the precursor m/z and isolation window when available.
    render:
        ``"sticks"`` or ``"profile"``. ``None`` picks from ``spectrum.spectrum_type``.
    max_points:
        Cap on samples drawn for a profile trace (min/max decimation keeps
        every apex). ``None`` draws every sample.
    absolute_axis:
        With relative intensities, add a right-hand axis in absolute counts.
    backend:
        ``"plotly"`` (default), ``"matplotlib"``, or ``"spec"``.
    style:
        ``"screen"``, ``"paper"``, ``"talk"`` or a :class:`~spxtacular.style.FigureStyle`.
        ``None`` is ``"screen"`` for plotly and ``"paper"`` for matplotlib.
    size:
        ``"single"`` (85 mm), ``"onehalf"`` (114 mm), ``"double"`` (175 mm), a
        width in mm, or ``(width_mm, height_mm)``.
    **layout_kwargs:
        Plotly only: forwarded to ``fig.update_layout`` last.
    """
    key, fig_style, mode = _setup(backend, style, theme_mode)
    default_title = str(spectrum.spectrum_type or "Spectrum")
    if color == "im" and spectrum.im is not None and len(spectrum.im) == len(spectrum.mz):
        # The type check comes first: the im path only draws sticks, and routing
        # profile data there would draw one bar per sample and lose the peak shape.
        resolved = (
            render
            if render is not None
            else ("profile" if spectrum.spectrum_type == SpectrumType.PROFILE else "sticks")
        )
        if resolved == "profile":
            raise SpxtacularError(
                "color='im' draws sticks, which discards the peak shape of a profile spectrum. "
                "Centroid it first (spectrum.centroid()), or pass render='sticks' to draw every "
                "sample as a stick anyway."
            )
        panel = _im_panel(
            spectrum,
            show_scores=show_scores,
            max_labels=max_labels,
            style=fig_style,
            mode=mode,
            intensity_scale=intensity_scale,
            intensity_transform=intensity_transform,
            absolute_axis=absolute_axis,
        )
    else:
        table = build_plot_table(
            spectrum,
            show_charges=color == "charge",
            show_scores=show_scores,
            max_labels=max_labels,
            theme_mode=mode,
            intensity_scale=intensity_scale,
            intensity_transform=intensity_transform,
        )
        render_mode = render if render is not None else table.attrs.get("render", "sticks")
        if render_mode not in ("sticks", "profile"):
            raise SpxtacularError(f"render must be 'sticks' or 'profile', got {render_mode!r}")
        panel = table_panel(
            table,
            style=fig_style,
            theme_mode=mode,
            render=render_mode,
            max_points=max_points,
            absolute_axis=absolute_axis,
        )
    if show_precursor:
        panel.marks[:0] = _precursor_marks(spectrum, fig_style, mode)
    cell = Cell(panels=[panel], title=figure_title(title, default_title, fig_style), aspect=fig_style.aspect)
    return _build(cell, key=key, fig_style=fig_style, size=size, mode=mode, layout_kwargs=layout_kwargs)


def _similarity_value(
    top: Spectrum,
    bottom: Spectrum,
    similarity: Literal["cosine", "modified_cosine", "entropy"] | float | None,
    tolerance: float,
    tolerance_unit: ToleranceUnit,
) -> tuple[str, float] | None:
    if similarity is None:
        return None
    if isinstance(similarity, int | float) and not isinstance(similarity, bool):
        return "similarity", float(similarity)
    from .similarity import cosine, entropy_similarity, modified_cosine

    name = str(similarity).lower()
    if name == "cosine":
        return "cosine", cosine(top, bottom, tolerance=tolerance, tolerance_unit=tolerance_unit)
    if name == "modified_cosine":
        mzs = [_first_precursor_mz(s) for s in (top, bottom)]
        if mzs[0] is None or mzs[1] is None:
            raise SpxtacularError("similarity='modified_cosine' needs a precursor m/z on both spectra")
        return "modified cosine", modified_cosine(
            top, bottom, mzs[0], mzs[1], tolerance=tolerance, tolerance_unit=tolerance_unit
        )
    if name == "entropy":
        return "entropy similarity", entropy_similarity(top, bottom, tolerance=tolerance, tolerance_unit=tolerance_unit)
    raise SpxtacularError(
        f"similarity must be 'cosine', 'modified_cosine', 'entropy', a number, or None; got {similarity!r}"
    )


def mirror_plot(
    raw: Spectrum,
    deconvoluted: Spectrum,
    *,
    fragments: FragmentInput | None = None,
    names: tuple[str, str] | None = None,
    similarity: Literal["cosine", "modified_cosine", "entropy"] | float | None = None,
    title: str | None = None,
    normalize: bool = True,
    show_charges: bool = True,
    show_scores: bool = True,
    tolerance: float = DEFAULT_FRAGMENT_TOLERANCE,
    tolerance_unit: ToleranceUnit = DEFAULT_FRAGMENT_TOLERANCE_UNIT,
    peak_selection: PeakSelectionLike = PeakSelection.CLOSEST,
    max_labels: int | None = _MAX_LABELS_DEFAULT,
    theme_mode: theme.ThemeMode | None = None,
    backend: Backend = "plotly",
    style: StyleLike = None,
    size: SizeLike = None,
    **layout_kwargs: Any,
) -> Any:
    """Mirror plot: one spectrum above the axis, the other reflected below it.

    Built for two jobs:

    * **Raw vs deconvoluted** (the default reading of the arguments): the
      deconvoluted spectrum above, coloured by charge, the raw spectrum below,
      so you can trace which raw peaks fed each cluster.
    * **Experimental vs reference** (pass ``fragments``): both halves are
      annotated with the same ion colours and labels, styled identically, so
      matched ions line up across the axis.

    The lower half's tick labels read positive: both halves run 0-100 %.

    Parameters
    ----------
    raw:
        Spectrum drawn below the axis.
    deconvoluted:
        Spectrum drawn above the axis.
    fragments:
        Annotate both halves with these fragments.
    names:
        ``(upper, lower)`` names written inside each half. Defaults to
        ``("deconvoluted", "raw")`` without fragments and no names with them.
    similarity:
        ``"cosine"``, ``"modified_cosine"`` or ``"entropy"`` to compute and
        show a similarity score between the halves, or a number to show as is.
    title:
        Plot title.
    normalize:
        Scale each half to its own base peak (relative %). ``False`` plots raw values.
    show_charges:
        Without fragments: colour the upper sticks by charge state.
    show_scores:
        Without fragments: label upper peaks with isotope-profile scores.
    tolerance, tolerance_unit, peak_selection:
        Fragment matching and similarity parameters.
    max_labels:
        Label cap per half.
    theme_mode, backend, style, size, **layout_kwargs:
        As for :func:`plot_spectrum`.
    """
    key, fig_style, mode = _setup(backend, style, theme_mode)
    scale: Literal["absolute", "relative"] = "relative" if normalize else "absolute"
    marks: list[Mark] = []

    if fragments is not None:
        tables = [
            build_annot_plot_table(
                s,
                fragments,
                tolerance=tolerance,
                tolerance_unit=tolerance_unit,
                peak_selection=peak_selection,
                max_labels=max_labels,
                theme_mode=mode,
                intensity_scale=scale,
            )
            for s in (raw, deconvoluted)
        ]
        marks += table_marks(tables[0], style=fig_style, theme_mode=mode, direction="down", legend=False)
        marks += table_marks(tables[1], style=fig_style, theme_mode=mode, direction="up", legend=True)
        label = str(tables[1].attrs["intensity_label"])
    else:
        raw_true = np.asarray(raw.intensity, dtype=np.float64)
        dec_true = np.asarray(deconvoluted.intensity, dtype=np.float64)
        raw_plot, label = _scaled_intensity(raw_true, scale, None)
        dec_plot, _ = _scaled_intensity(dec_true, scale, None)
        raw_mz = np.asarray(raw.mz, dtype=np.float64)
        dec_mz = np.asarray(deconvoluted.mz, dtype=np.float64)
        context = theme.neutral_color(mode) if fig_style.strong_context else theme.unmatched_color(mode)
        marks.append(
            Sticks(
                x=raw_mz,
                y=-raw_plot,
                color=context,
                width=fig_style.stick_width_context * 1.2,
                name="raw",
                customdata=raw_true.tolist(),
                hovertemplate="m/z: %{x:.4f}<br>intensity: %{customdata:.2e}<extra>raw</extra>",
            )
        )
        charge = deconvoluted.charge
        hover = "m/z: %{x:.4f}<br>intensity: %{customdata:.2e}<extra></extra>"
        if show_charges and charge is not None:
            zs = sorted({int(c) for c in charge})
            for z in zs:
                mask = charge == z
                marks.append(
                    Sticks(
                        x=dec_mz[mask],
                        y=dec_plot[mask],
                        color=theme.charge_color(z, mode),
                        width=fig_style.stick_width,
                        name=_charge_series(z),
                        legend=len(zs) > 1,
                        customdata=dec_true[mask].tolist(),
                        hovertemplate=hover,
                    )
                )
        else:
            marks.append(
                Sticks(
                    x=dec_mz,
                    y=dec_plot,
                    color=theme.charge_color(1, mode),
                    width=fig_style.stick_width,
                    name="deconvoluted",
                    customdata=dec_true.tolist(),
                    hovertemplate=hover,
                )
            )
        if show_scores and deconvoluted.iso_score is not None:
            texts = _cap_labels(
                [f"{float(s):.2f}" if float(s) > 0.0 else "" for s in deconvoluted.iso_score], dec_true, max_labels
            )
            keep = [i for i, t in enumerate(texts) if t]
            if keep:
                marks.append(
                    LabelSet(
                        x=dec_mz[keep],
                        y=dec_plot[keep],
                        texts=[RichText.plain(texts[i]) for i in keep],
                        colors=[theme.text_color("secondary", mode)] * len(keep),
                        priority=dec_true[keep],
                        size=fig_style.label_size,
                        gap=fig_style.label_gap,
                    )
                )
        if names is None:
            names = ("deconvoluted", "raw")

    # Names and the score sit in the right-hand corners: the low-m/z end of a
    # peptide spectrum holds immonium ions and is rarely empty, the high end usually is.
    name_color = theme.text_color("secondary", mode)
    inset = fig_style.font_size * 0.6
    line = fig_style.font_size * 1.3
    upper, lower = names if names is not None else ("", "")
    if upper:
        marks.append(
            AxText(1.0, 1.0, RichText.plain(upper), fig_style.font_size, name_color, dx=-inset, dy=-inset,
                   ha="right", va="top", name="half_name")
        )  # fmt: skip
    if lower:
        marks.append(
            AxText(1.0, 0.0, RichText.plain(lower), fig_style.font_size, name_color, dx=-inset, dy=inset,
                   ha="right", va="bottom", name="half_name")
        )  # fmt: skip
    sim = _similarity_value(deconvoluted, raw, similarity, tolerance, tolerance_unit)
    if sim is not None:
        marks.append(
            AxText(1.0, 1.0, RichText.plain(f"{sim[0]} {sim[1]:.3f}"), fig_style.font_size, name_color,
                   dx=-inset, dy=-inset - (line if upper else 0.0), ha="right", va="top", name="similarity")
        )  # fmt: skip

    y_axis, _ = intensity_axis(label, relative=normalize, base_peak=None, absolute_axis=False, mirrored=True)
    if not normalize:
        y_axis.scale_exponent = True
    panel = Panel(marks=marks, x=Axis(label=mz_label()), y=y_axis)
    cell = Cell(
        panels=[panel],
        title=figure_title(title, "Raw vs deconvoluted" if fragments is None else "Mirror plot", fig_style),
        aspect=fig_style.aspect * 1.25,
    )
    return _build(cell, key=key, fig_style=fig_style, size=size, mode=mode, layout_kwargs=layout_kwargs)


# ---------------------------------------------------------------------------
# Sequence header and coverage
# ---------------------------------------------------------------------------

_N_TERM = {"a", "b", "c"}
_C_TERM = {"x", "y", "z"}


def _as_annotation(peptide: str | ProFormaAnnotation) -> ProFormaAnnotation:
    import peptacular as pt

    if isinstance(peptide, str):
        return pt.parse(peptide)
    return peptide


def _bond_evidence(matches: Sequence[MatchedFragment], n_res: int) -> tuple[dict[int, str], dict[int, str]]:
    """Bonds evidenced by N- and C-terminal fragments, as ``{bond: ion type}``.

    Bond ``k`` is the one after residue ``k`` (1-based). When several ion types
    evidence a bond, the one earliest in the fixed series order wins, so the
    colour does not depend on input order.
    """
    n_bonds: dict[int, str] = {}
    c_bonds: dict[int, str] = {}
    slots = theme._ION_SLOTS
    rank = {s: i for i, s in enumerate(slots)}
    for m in matches:
        frag = m.fragment
        ion = _ion_type(frag).lower()
        pos = getattr(frag, "position", None)
        if not isinstance(pos, int) or pos <= 0 or pos >= n_res:
            continue
        if ion in _N_TERM:
            bond, target = pos, n_bonds
        elif ion in _C_TERM:
            bond, target = n_res - pos, c_bonds
        else:
            continue
        if bond not in target or rank.get(ion, 99) < rank.get(target[bond], 99):
            target[bond] = ion
    return n_bonds, c_bonds


def _ladder_marks(
    residues: list[str],
    modified: list[bool],
    n_bonds: dict[int, str],
    c_bonds: dict[int, str],
    *,
    letter_size: float,
    step: float,
    xf: float,
    yf: float,
    dy: float,
    mode: theme.ThemeMode,
    tick_width: float,
    first_index: int = 0,
    total: int | None = None,
) -> list[Mark]:
    """Residue letters with b-style ticks above-left and y-style ticks below-right of each bond.

    Letters are centred on ``(xf, yf)`` offset by ``dy`` pt, ``step`` pt apart.
    ``first_index`` and ``total`` let a long sequence wrap over several rows.
    """
    n = len(residues)
    total = n if total is None else total
    ink = theme.text_color("primary", mode)
    mod_color = theme.text_color("secondary", mode)
    marks: list[Mark] = []
    centre = (n - 1) / 2.0
    for i, (res, is_mod) in enumerate(zip(residues, modified, strict=True)):
        marks.append(
            AxText(
                xf,
                yf,
                RichText.plain(res),
                letter_size,
                ink,
                dx=(i - centre) * step,
                dy=dy,
                ha="center",
                va="middle",
                bold=is_mod,
                name="residue",
            )
        )
        if is_mod:
            # A small dot under a modified residue: bold alone is easy to miss.
            marks.append(
                AxText(
                    xf,
                    yf,
                    RichText.plain("•"),
                    letter_size * 0.55,
                    mod_color,
                    dx=(i - centre) * step,
                    dy=dy - letter_size * 0.78,
                    ha="center",
                    va="middle",
                    name="mod_marker",
                )
            )
    h = letter_size * 0.62  # tick reach from the letter centre line
    foot = step * 0.32
    by_color: dict[str, list[tuple[float, float, float, float, float, float]]] = {}
    for bonds, up in ((n_bonds, True), (c_bonds, False)):
        for bond, ion in bonds.items():
            local = bond - first_index  # bond after local residue (local - 1)
            if local < 1 or local > n - 1 + (1 if first_index + n < total else 0):
                continue
            x = (local - 1 - centre) * step + step / 2.0
            color = theme.ion_color(ion, mode)
            segs = by_color.setdefault(color, [])
            if up:
                segs.append((xf, yf, x, dy + h * 0.15, x, dy + h))
                segs.append((xf, yf, x, dy + h, x - foot, dy + h))
            else:
                segs.append((xf, yf, x, dy - h * 0.15, x, dy - h))
                segs.append((xf, yf, x, dy - h, x + foot, dy - h))
    for color, segs in by_color.items():
        marks.append(AxSegments(segs, color, tick_width, name="coverage_tick"))
    return marks


def _residues(annotation: ProFormaAnnotation) -> tuple[list[str], list[bool]]:
    seq = str(annotation.stripped_sequence)
    mods = []
    for i in range(len(seq)):
        try:
            mods.append(bool(annotation.has_internal_mods_at_index(i)))
        except (AttributeError, IndexError, TypeError):
            mods.append(False)
    if seq and _flag(annotation, "has_nterm_mods"):
        mods[0] = True
    if seq and _flag(annotation, "has_cterm_mods"):
        mods[-1] = True
    return list(seq), mods


def _flag(obj: object, name: str) -> bool:
    """A peptacular boolean that may be a property or a method, depending on version."""
    value = getattr(obj, name, False)
    return bool(value() if callable(value) else value)


def _header_geometry(n_res: int, width_pt: float, style: FigureStyle) -> tuple[float, float]:
    """Letter size and spacing for a one-line sequence header that fits ``width_pt``."""
    letter = style.axis_title_size * 1.3
    step = letter * 1.45
    avail = width_pt * 0.8
    if n_res * step > avail:
        step = avail / max(n_res, 1)
        letter = min(letter, step / 1.2)
    return letter, step


def annotate_spectrum(
    spectrum: Spectrum,
    fragments: FragmentInput,
    *,
    tolerance: float = DEFAULT_FRAGMENT_TOLERANCE,
    tolerance_unit: ToleranceUnit = DEFAULT_FRAGMENT_TOLERANCE_UNIT,
    title: str | None = None,
    peak_selection: PeakSelectionLike = PeakSelection.CLOSEST,
    include_sequence: bool = False,
    max_labels: int | None = _MAX_LABELS_DEFAULT,
    theme_mode: theme.ThemeMode | None = None,
    intensity_scale: Literal["absolute", "relative"] = "relative",
    intensity_transform: Literal["sqrt", "log"] | None = None,
    texture: bool = False,
    show_precursor: bool = True,
    peptide: str | ProFormaAnnotation | None = None,
    mass_error_panel: bool = False,
    absolute_axis: bool = False,
    backend: Backend = "plotly",
    style: StyleLike = None,
    size: SizeLike = None,
    **layout_kwargs: Any,
) -> Any:
    """Plot a spectrum with matched fragment ions coloured and labelled.

    Unmatched peaks are thin grey context; matched peaks take their ion-series
    colour (b blue, y red, ...) and a label such as y₇²⁺. Labels are placed
    without overlaps: straight above the peak, or beside it with a leader
    line, and dropped only when nothing nearby is free. Plain b/y ions are
    placed before neutral losses and isotopes.

    Parameters
    ----------
    spectrum:
        Centroid spectrum to plot.
    fragments:
        Fragment objects from peptacular to match against peaks.
    tolerance, tolerance_unit:
        Matching tolerance and its unit (``"ppm"`` or ``"Da"``).
    title:
        Plot title.
    peak_selection:
        ``"closest"``, ``"largest"`` or ``"all"``; see :func:`~spxtacular.matching.match_fragments`.
    include_sequence:
        Embed the residue sequence in each label's hover and table text.
    max_labels:
        Label cap, strongest peaks first.
    theme_mode:
        ``"light"`` or ``"dark"``.
    intensity_scale, intensity_transform:
        As for :func:`plot_spectrum`.
    texture:
        Give each ion series a distinct dash pattern (for greyscale print).
    show_precursor:
        Mark the precursor m/z and isolation window when available.
    peptide:
        A sequence or ProForma string (or a peptacular annotation). Draws the
        sequence above the spectrum with a tick at each bond a matched
        fragment covers: above-left for a/b/c ions, below-right for x/y/z.
    mass_error_panel:
        Add a strip under the spectrum with each match's mass error, in the
        unit of ``tolerance_unit``, spanning the tolerance.
    absolute_axis:
        Add a right-hand axis in absolute intensity.
    backend, style, size, **layout_kwargs:
        As for :func:`plot_spectrum`.
    """
    key, fig_style, mode = _setup(backend, style, theme_mode)
    table = build_annot_plot_table(
        spectrum,
        fragments,
        tolerance=tolerance,
        tolerance_unit=tolerance_unit,
        peak_selection=peak_selection,
        include_sequence=include_sequence,
        max_labels=max_labels,
        theme_mode=mode,
        intensity_scale=intensity_scale,
        intensity_transform=intensity_transform,
        texture=texture,
    )
    panel = table_panel(table, style=fig_style, theme_mode=mode, absolute_axis=absolute_axis)
    if show_precursor:
        panel.marks[:0] = _precursor_marks(spectrum, fig_style, mode)

    need_matches = peptide is not None or mass_error_panel
    matches = (
        match_fragments(
            spectrum, fragments, tolerance=tolerance, tolerance_unit=tolerance_unit, peak_selection=peak_selection
        )
        if need_matches
        else []
    )
    extra_mm = 0.0
    if peptide is not None:
        annotation = _as_annotation(peptide)
        residues, modified = _residues(annotation)
        if residues:
            n_bonds, c_bonds = _bond_evidence(matches, len(residues))
            width_mm, _ = resolve_size(size, fig_style)
            letter, step = _header_geometry(len(residues), width_mm * PT_PER_MM, fig_style)
            header = letter * 2.9
            panel.header_height = header
            panel.marks += _ladder_marks(
                residues,
                modified,
                n_bonds,
                c_bonds,
                letter_size=letter,
                step=step,
                xf=0.5,
                yf=1.0,
                dy=header * 0.5,
                mode=mode,
                tick_width=max(fig_style.stick_width, fig_style.axis_width * 1.4),
            )
            extra_mm += header / PT_PER_MM

    panels = [panel]
    if mass_error_panel:
        unit = _unit_of(tolerance_unit)
        err_marks = _error_marks(
            matches,
            unit,
            style=fig_style,
            mode=mode,
            labels=False,
            max_labels=max_labels,
            max_size=fig_style.font_size * 0.8,
        )
        strip = fig_style.font_size * 6.5
        panels.append(
            Panel(
                marks=err_marks,
                x=Axis(label=mz_label()),
                y=_error_axis(unit, tolerance, unit, _errors(matches, unit), short=True),
                fixed_height=strip,
                share_x=True,
            )
        )
        extra_mm += (strip + fig_style.font_size) / PT_PER_MM
    cell = Cell(
        panels=panels,
        title=figure_title(title, "Annotated spectrum", fig_style),
        aspect=fig_style.aspect,
        extra_height_mm=extra_mm,
    )
    return _build(cell, key=key, fig_style=fig_style, size=size, mode=mode, layout_kwargs=layout_kwargs)


def sequence_coverage_plot(
    spectrum: Spectrum,
    peptide: str | ProFormaAnnotation,
    fragments: FragmentInput,
    *,
    tolerance: float = DEFAULT_FRAGMENT_TOLERANCE,
    tolerance_unit: ToleranceUnit = DEFAULT_FRAGMENT_TOLERANCE_UNIT,
    peak_selection: PeakSelectionLike = PeakSelection.CLOSEST,
    title: str | None = None,
    theme_mode: theme.ThemeMode | None = None,
    backend: Backend = "plotly",
    style: StyleLike = None,
    size: SizeLike = None,
    **layout_kwargs: Any,
) -> Any:
    """Sequence coverage ladder: which backbone bonds the spectrum evidences.

    Residues run left to right. A tick above and to the left of a bond marks an
    N-terminal (a/b/c) fragment ending there; a tick below and to the right a
    C-terminal (x/y/z) fragment starting there. Ticks take the ion-series
    colour; modified residues are bold with a dot beneath. Long sequences wrap.

    Parameters
    ----------
    spectrum:
        The spectrum the fragments were matched against.
    peptide:
        Residue sequence or ProForma string (modifications are marked, not spelled out).
    fragments:
        Fragment objects to match, as for :func:`~spxtacular.matching.match_fragments`.
    tolerance, tolerance_unit, peak_selection:
        Matching parameters.
    title:
        Plot title. Defaults to the covered-bond count.
    theme_mode, backend, style, size, **layout_kwargs:
        As for :func:`plot_spectrum`.
    """
    key, fig_style, mode = _setup(backend, style, theme_mode)
    annotation = _as_annotation(peptide)
    residues, modified = _residues(annotation)
    n_res = len(residues)
    if n_res == 0:
        raise SpxtacularError("peptide must contain at least one residue")
    matches = match_fragments(
        spectrum, fragments, tolerance=tolerance, tolerance_unit=tolerance_unit, peak_selection=peak_selection
    )
    n_bonds, c_bonds = _bond_evidence(matches, n_res)

    width_mm, _ = resolve_size(size, fig_style)
    width_pt = width_mm * PT_PER_MM
    letter = fig_style.axis_title_size * 1.6
    step = letter * 1.5
    per_row = max(8, int(width_pt * 0.9 // step))
    rows = -(-n_res // per_row)
    row_height = letter * 2.8

    marks: list[Mark] = []
    for r in range(rows):
        lo = r * per_row
        hi = min(n_res, lo + per_row)
        row_n = {b: t for b, t in n_bonds.items() if lo < b <= hi}
        row_c = {b: t for b, t in c_bonds.items() if lo < b <= hi}
        yf = 1.0 - (r + 0.5) / rows
        chunk = residues[lo:hi]
        # Left-align wrapped rows so residue columns line up across rows.
        centre_shift = ((per_row - len(chunk)) / 2.0) * step if rows > 1 else 0.0
        row_marks = _ladder_marks(
            chunk,
            modified[lo:hi],
            row_n,
            row_c,
            letter_size=letter,
            step=step,
            xf=0.5,
            yf=yf,
            dy=0.0,
            mode=mode,
            tick_width=max(fig_style.stick_width * 1.3, fig_style.axis_width * 1.6),
            first_index=lo,
            total=n_res,
        )
        for m in row_marks:
            if isinstance(m, AxText):
                m.dx -= centre_shift
            elif isinstance(m, AxSegments):
                m.segments = [
                    (a, b, x0 - centre_shift, y0, x1 - centre_shift, y1) for a, b, x0, y0, x1, y1 in m.segments
                ]
        marks += row_marks

    n_color = theme.ion_color("b", mode)
    c_color = theme.ion_color("y", mode)
    # One NaN point, not an empty array: plotly leaves empty traces out of the legend.
    empty = np.full(1, np.nan)
    for name, color in (("N-terminal (a/b/c)", n_color), ("C-terminal (x/y/z)", c_color)):
        marks.append(Line(x=empty, y=empty, color=color, width=fig_style.stick_width * 1.3, name=name, legend=True))

    n_bond_total = max(n_res - 1, 1)
    covered = len(set(n_bonds) | set(c_bonds))
    summary = f"{covered}/{n_bond_total} backbone bonds covered ({covered / n_bond_total:.0%})"
    if not (title or fig_style.show_title):
        marks.append(
            AxText(0.0, 0.0, RichText.plain(summary), fig_style.font_size, theme.text_color("secondary", mode),
                   ha="left", va="bottom", name="summary")
        )  # fmt: skip
    panel = Panel(
        marks=marks,
        x=Axis(lo=0.0, hi=1.0, visible=False),
        y=Axis(lo=0.0, hi=1.0, visible=False),
        fixed_height=rows * row_height + (fig_style.font_size * 1.6 if not fig_style.show_title else 0.0),
    )
    height_mm = (rows * row_height) / PT_PER_MM + 14.0
    cell = Cell(
        panels=[panel],
        title=figure_title(title, f"Sequence coverage — {summary}", fig_style),
        aspect=0.0,
        extra_height_mm=height_mm,
    )
    return _build(cell, key=key, fig_style=fig_style, size=size, mode=mode, layout_kwargs=layout_kwargs)


# ---------------------------------------------------------------------------
# Mass errors and facets
# ---------------------------------------------------------------------------


def mass_error_plot(
    spectrum: Spectrum,
    fragments: FragmentInput,
    *,
    tolerance: float = DEFAULT_FRAGMENT_TOLERANCE,
    tolerance_unit: ToleranceUnit = DEFAULT_FRAGMENT_TOLERANCE_UNIT,
    peak_selection: PeakSelectionLike = PeakSelection.CLOSEST,
    unit: str = "ppm",
    title: str | None = None,
    max_labels: int | None = _MAX_LABELS_DEFAULT,
    theme_mode: theme.ThemeMode | None = None,
    backend: Backend = "plotly",
    style: StyleLike = None,
    size: SizeLike = None,
    **layout_kwargs: Any,
) -> Any:
    """Mass error of each matched fragment against its m/z.

    Dot area follows peak intensity and colour the ion series. Labels are
    sub/superscripted ion names placed without overlaps; every match stays on
    hover.

    Parameters
    ----------
    spectrum:
        Spectrum to plot.
    fragments:
        Fragment objects from peptacular to match against peaks.
    tolerance, tolerance_unit, peak_selection:
        Matching parameters.
    unit:
        Error unit: ``"ppm"`` or ``"da"``.
    title:
        Plot title.
    max_labels:
        Label cap, strongest peaks first.
    theme_mode, backend, style, size, **layout_kwargs:
        As for :func:`plot_spectrum`.
    """
    key, fig_style, mode = _setup(backend, style, theme_mode)
    err_unit = _error_unit(unit)
    matches = match_fragments(
        spectrum, fragments, tolerance=tolerance, tolerance_unit=tolerance_unit, peak_selection=peak_selection
    )
    marks = _error_marks(
        matches,
        err_unit,
        style=fig_style,
        mode=mode,
        labels=True,
        max_labels=max_labels,
        max_size=fig_style.font_size * 1.6,
    )
    y_axis = _error_axis(err_unit, tolerance, _unit_of(tolerance_unit), _errors(matches, err_unit))
    y_axis.headroom = True
    panel = Panel(marks=marks, x=Axis(label=mz_label(), pad=0.04), y=y_axis)
    default = "Mass errors" if matches else "Mass errors (no matches)"
    cell = Cell(panels=[panel], title=figure_title(title, default, fig_style), aspect=fig_style.aspect)
    return _build(cell, key=key, fig_style=fig_style, size=size, mode=mode, layout_kwargs=layout_kwargs)


def facet_plot(
    spectrum: Spectrum,
    *,
    fragments: FragmentInput | None = None,
    mirror_spectrum: Spectrum | None = None,
    title: str | None = None,
    tolerance: float = DEFAULT_FRAGMENT_TOLERANCE,
    tolerance_unit: ToleranceUnit = DEFAULT_FRAGMENT_TOLERANCE_UNIT,
    peak_selection: PeakSelectionLike = PeakSelection.CLOSEST,
    include_sequence: bool = False,
    unit: str = "ppm",
    max_labels: int | None = _MAX_LABELS_DEFAULT,
    theme_mode: theme.ThemeMode | None = None,
    backend: Backend = "plotly",
    style: StyleLike = None,
    size: SizeLike = None,
    **layout_kwargs: Any,
) -> Any:
    """Stacked panels on one m/z axis: spectrum, mass errors, mirror.

    1. The (annotated) spectrum, always.
    2. Mass errors, when ``fragments`` is given.
    3. ``mirror_spectrum`` reflected below, when given.

    Parameters
    ----------
    spectrum:
        Primary spectrum.
    fragments:
        Fragments for annotation and the mass-error panel.
    mirror_spectrum:
        Optional second spectrum, drawn downward in the last panel.
    title:
        Plot title.
    tolerance, tolerance_unit, peak_selection, include_sequence:
        Matching parameters.
    unit:
        Mass-error unit: ``"ppm"`` or ``"da"``.
    max_labels:
        Label cap per panel.
    theme_mode, backend, style, size, **layout_kwargs:
        As for :func:`plot_spectrum`.
    """
    key, fig_style, mode = _setup(backend, style, theme_mode)
    err_unit = _error_unit(unit)
    if fragments is not None:
        table = build_annot_plot_table(
            spectrum,
            fragments,
            tolerance=tolerance,
            tolerance_unit=tolerance_unit,
            peak_selection=peak_selection,
            include_sequence=include_sequence,
            max_labels=max_labels,
            theme_mode=mode,
        )
    else:
        table = build_plot_table(spectrum, max_labels=max_labels, theme_mode=mode)
    top = table_panel(table, style=fig_style, theme_mode=mode)
    top.marks = [m for m in top.marks if not isinstance(m, HitLayer)] + [
        m for m in top.marks if isinstance(m, HitLayer)
    ]
    panels = [top]
    extra = 0.0
    if fragments is not None:
        matches = match_fragments(
            spectrum, fragments, tolerance=tolerance, tolerance_unit=tolerance_unit, peak_selection=peak_selection
        )
        err = _error_marks(
            matches,
            err_unit,
            style=fig_style,
            mode=mode,
            labels=False,
            max_labels=max_labels,
            max_size=fig_style.font_size * 1.1,
        )
        panels.append(
            Panel(
                marks=err,
                x=Axis(label=mz_label()),
                y=_error_axis(err_unit, tolerance, _unit_of(tolerance_unit), _errors(matches, err_unit), short=True),
                weight=0.45,
                share_x=True,
            )
        )
        extra += 0.3
    if mirror_spectrum is not None:
        # Same colouring as the top panel, so matched ions read identically on both halves.
        if fragments is not None:
            mirror_table = build_annot_plot_table(
                mirror_spectrum,
                fragments,
                tolerance=tolerance,
                tolerance_unit=tolerance_unit,
                peak_selection=peak_selection,
                max_labels=max_labels,
                theme_mode=mode,
            )
        else:
            mirror_table = build_plot_table(mirror_spectrum, max_labels=max_labels, theme_mode=mode)
        marks = table_marks(mirror_table, style=fig_style, theme_mode=mode, direction="down", legend=False)
        y_axis, _ = intensity_axis(
            str(mirror_table.attrs.get("intensity_label", "Intensity")),
            relative=True,
            base_peak=None,
            absolute_axis=False,
            mirrored=True,
        )
        y_axis.hi = 0.0
        # No tick label at 0: it would sit against the panel above.
        y_axis.ticks = [-100.0, -50.0]
        y_axis.ticktext = ["100", "50"]
        panels.append(Panel(marks=marks, x=Axis(label=mz_label()), y=y_axis, weight=0.7, share_x=True))
        extra += 0.45
    cell = Cell(
        panels=panels,
        title=figure_title(title, "Facet plot", fig_style),
        aspect=fig_style.aspect * (1.0 + extra),
    )
    return _build(cell, key=key, fig_style=fig_style, size=size, mode=mode, layout_kwargs=layout_kwargs)


# ---------------------------------------------------------------------------
# Chromatograms and profile data
# ---------------------------------------------------------------------------


def plot_chromatogram(
    chromatograms: Chromatogram | Sequence[Chromatogram] | Iterable[Spectrum],
    *,
    title: str | None = None,
    theme_mode: theme.ThemeMode | None = None,
    show_apex: bool = True,
    fill: bool | None = None,
    backend: Backend = "plotly",
    style: StyleLike = None,
    size: SizeLike = None,
    **layout_kwargs: Any,
) -> Any:
    """Plot one or more chromatograms against retention time.

    Accepts :class:`~spxtacular.chromatogram.Chromatogram` objects, or an
    iterable of spectra, from which a TIC is extracted::

        with spx.Reader("run.d") as reader:
            spx.plot_chromatogram(reader.ms1).show()

    Parameters
    ----------
    chromatograms:
        A chromatogram, a sequence of them, or an iterable of spectra.
    title:
        Plot title.
    theme_mode:
        ``"light"`` or ``"dark"``.
    show_apex:
        Label each trace's apex with its retention time (up to four traces).
    fill:
        Fill under the trace. Defaults to on for one trace, off for several.
    backend, style, size, **layout_kwargs:
        As for :func:`plot_spectrum`.
    """
    from .chromatogram import Chromatogram as _Chrom
    from .chromatogram import extract_chromatogram

    key, fig_style, mode = _setup(backend, style, theme_mode)
    if isinstance(chromatograms, _Chrom):
        traces_in = [chromatograms]
    elif isinstance(chromatograms, Sequence) and all(isinstance(c, _Chrom) for c in chromatograms):
        traces_in = list(cast("Sequence[_Chrom]", chromatograms))
    else:
        traces_in = [extract_chromatogram(cast("Iterable[Spectrum]", chromatograms))]

    if fill is None:
        fill = len(traces_in) == 1
    units = {chrom.meta.get("rt_unit", "s") for chrom in traces_in if len(chrom)}
    if len(units) > 1:
        raise SpxtacularError("Cannot plot retention times and scan indices on the same axis")
    unit = next(iter(units), "s")
    if unit not in ("s", "scan_index"):
        raise SpxtacularError(f"Unsupported chromatogram time unit: {unit!r}")
    axis_title = "Scan index" if unit == "scan_index" else "Retention time (s)"
    time_label = "Scan index" if unit == "scan_index" else "RT"
    time_suffix = "" if unit == "scan_index" else " s"

    marks: list[Mark] = []
    apex_x: list[float] = []
    apex_y: list[float] = []
    apex_t: list[RichText] = []
    apex_c: list[str] = []
    for i, chrom in enumerate(traces_in):
        if len(traces_in) == 1:
            color = theme.charge_color(1, mode)
        else:
            color = theme.ion_color(theme._ION_SLOTS[i % len(theme._ION_SLOTS)], mode)
        name = chrom.label or f"trace {i + 1}"
        marks.append(
            Line(
                x=np.asarray(chrom.rt, dtype=np.float64),
                y=np.asarray(chrom.intensity, dtype=np.float64),
                color=color,
                width=fig_style.line_width,
                fill=bool(fill),
                fill_alpha=0.12,
                name=name,
                legend=len(traces_in) > 1,
                hovertemplate=(
                    time_label
                    + ": %{x:.2f}"
                    + time_suffix
                    + "<br>intensity: %{y:.4g}<extra>"
                    + (chrom.label or "")
                    + "</extra>"
                ),
            )
        )
        if show_apex and len(traces_in) <= 4 and len(chrom):
            apex = int(np.argmax(chrom.intensity))
            if chrom.intensity[apex] > 0:
                apex_x.append(float(chrom.rt[apex]))
                apex_y.append(float(chrom.intensity[apex]))
                apex_t.append(RichText.plain(f"{chrom.rt[apex]:.1f}{time_suffix}"))
                apex_c.append(
                    theme.label_color(color, mode) if len(traces_in) > 1 else theme.text_color("secondary", mode)
                )
    if apex_x:
        marks.append(
            LabelSet(
                x=np.asarray(apex_x),
                y=np.asarray(apex_y),
                texts=apex_t,
                colors=apex_c,
                priority=np.asarray(apex_y),
                size=fig_style.label_size,
                gap=fig_style.label_gap,
            )
        )
    panel = Panel(
        marks=marks,
        x=Axis(label=RichText.plain(axis_title)),
        y=Axis(label=RichText.plain("Intensity"), lo=0.0, headroom=True, scale_exponent=True),
    )
    default = traces_in[0].label if len(traces_in) == 1 and traces_in[0].label else "Chromatograms"
    cell = Cell(panels=[panel], title=figure_title(title, default, fig_style), aspect=fig_style.aspect)
    return _build(cell, key=key, fig_style=fig_style, size=size, mode=mode, layout_kwargs=layout_kwargs)


def plot_xic(
    spectra: Iterable[Spectrum],
    targets: Sequence[float] | float,
    *,
    tolerance: float = 20.0,
    tolerance_unit: ToleranceUnit = "ppm",
    im_window: tuple[float, float] | None = None,
    aggregate: Literal["sum", "max"] = "sum",
    title: str | None = None,
    theme_mode: theme.ThemeMode | None = None,
    backend: Backend = "plotly",
    style: StyleLike = None,
    size: SizeLike = None,
    **layout_kwargs: Any,
) -> Any:
    """Extract and plot ion chromatograms in one call.

    Every target is extracted in a single pass over ``spectra``::

        with spx.Reader("run.d") as reader:
            spx.plot_xic(reader.ms1, [500.2649, 622.0290], tolerance=20).show()

    See :func:`~spxtacular.chromatogram.extract_xic` for the extraction
    parameters, and :func:`plot_chromatogram` for the rest.
    """
    from .chromatogram import extract_xic

    chroms = extract_xic(
        spectra,
        targets,
        tolerance=tolerance,
        tolerance_unit=tolerance_unit,
        im_window=im_window,
        aggregate=aggregate,
    )
    unit = "ppm" if str(tolerance_unit).lower() == "ppm" else "Da"
    default_title = f"Extracted ion chromatogram{'s' if len(chroms) > 1 else ''} (±{tolerance:g} {unit})"
    return plot_chromatogram(
        chroms,
        title=title or default_title,
        theme_mode=theme_mode,
        backend=backend,
        style=style,
        size=size,
        **layout_kwargs,
    )


def profile_centroid_plot(
    profile: Spectrum,
    *,
    centroids: Spectrum | None = None,
    title: str | None = None,
    theme_mode: theme.ThemeMode | None = None,
    max_points: int | None = _PROFILE_MAX_POINTS,
    backend: Backend = "plotly",
    style: StyleLike = None,
    size: SizeLike = None,
    **layout_kwargs: Any,
) -> Any:
    """Profile trace with the centroided peaks drawn on top.

    The check that centroiding did the right thing: a stick off its apex is a
    mis-assigned centre, and an apex with no stick is a dropped peak.
    Flat-topped peaks give one centroid at the middle of the plateau.

    Parameters
    ----------
    profile:
        The profile-mode spectrum.
    centroids:
        Centroided peaks to overlay. Defaults to ``profile.centroid()``.
    title:
        Plot title.
    theme_mode:
        ``"light"`` or ``"dark"``.
    max_points:
        Cap on drawn profile samples (min/max decimation).
    backend, style, size, **layout_kwargs:
        As for :func:`plot_spectrum`.
    """
    key, fig_style, mode = _setup(backend, style, theme_mode)
    if centroids is None:
        centroids = profile.centroid()
    prof_mz, prof_int = _decimate_profile(
        np.asarray(profile.mz, dtype=np.float64),
        np.asarray(profile.intensity, dtype=np.float64),
        max_points,
    )
    profile_color = theme.neutral_color(mode) if fig_style.print_ink else theme.unmatched_color(mode)
    centroid_color = theme.charge_color(1, mode)
    marks: list[Mark] = [
        Line(
            x=prof_mz,
            y=prof_int,
            color=profile_color,
            width=fig_style.line_width,
            fill=True,
            fill_alpha=0.18,
            name="profile",
            legend=True,
            hovertemplate="m/z: %{x:.4f}<br>intensity: %{y:.4g}<extra>profile</extra>",
        )
    ]
    if len(centroids) > 0:
        c_mz = np.asarray(centroids.mz, dtype=np.float64)
        c_int = np.asarray(centroids.intensity, dtype=np.float64)
        marks.append(Sticks(x=c_mz, y=c_int, color=centroid_color, width=fig_style.stick_width, name="centroids",
                            legend=True))  # fmt: skip
        marks.append(
            HitLayer(
                x=c_mz,
                y=c_int,
                hover=[f"m/z: {m:.4f}<br>intensity: {i:.4g}" for m, i in zip(c_mz, c_int, strict=True)],
                size_px=_HIT_TARGET_SIZE,
            )
        )
    panel = Panel(
        marks=marks,
        x=Axis(label=mz_label()),
        y=Axis(label=RichText.plain("Intensity"), lo=0.0, pad=0.05, scale_exponent=True),
    )
    default = f"Profile vs centroids — {len(centroids)} peaks from {len(profile)} samples"
    cell = Cell(panels=[panel], title=figure_title(title, default, fig_style), aspect=fig_style.aspect)
    return _build(cell, key=key, fig_style=fig_style, size=size, mode=mode, layout_kwargs=layout_kwargs)


# ---------------------------------------------------------------------------
# Reporter ions
# ---------------------------------------------------------------------------


def reporter_ion_plot(
    spectrum: Spectrum,
    plex: str | IsobaricTagInfo | ReporterIons = "TMT10",
    *,
    tolerance: float = DEFAULT_REPORTER_TOLERANCE,
    tolerance_unit: ToleranceUnit = DEFAULT_REPORTER_TOLERANCE_UNIT,
    impurities: ImpurityTable | pd.DataFrame | NDArray[np.float64] | None = None,
    show_spectrum: bool = True,
    normalize: bool = True,
    title: str | None = None,
    theme_mode: theme.ThemeMode | None = None,
    backend: Backend = "plotly",
    style: StyleLike = None,
    size: SizeLike = None,
    **layout_kwargs: Any,
) -> Any:
    """Isobaric-label reporter ions: channel intensities, with the reporter region above.

    The lower panel is one bar per channel, in channel order, as % of the
    strongest channel. The upper panel (``show_spectrum``) is the raw reporter
    region, with the peak picked for each channel highlighted, so an
    interfering or missing reporter is visible rather than hidden in a bar.
    Channels with no peak are marked "n.d.".

    Parameters
    ----------
    spectrum:
        An MS2 or MS3 spectrum carrying reporter ions.
    plex:
        A plex name from tacular (``"TMT6"``, ``"TMT10"``, ``"TMT11"``, ``"TMT16"``,
        ``"TMT18"``, ``"iTRAQ4"``, ``"iTRAQ8"``, ...), an :class:`~tacular.IsobaricTagInfo`,
        or a :class:`~spxtacular.reporter.ReporterIons` already extracted from ``spectrum``.
    tolerance, tolerance_unit, impurities:
        Passed to :func:`~spxtacular.reporter.extract_reporter_ions`; ignored when
        ``plex`` is a :class:`~spxtacular.reporter.ReporterIons`.
    show_spectrum:
        Draw the reporter m/z region above the bars.
    normalize:
        Bars as % of the strongest channel. ``False`` plots the intensities as extracted.
    title:
        Plot title.
    theme_mode, backend, style, size, **layout_kwargs:
        As for :func:`plot_spectrum`.

    Raises
    ------
    SpxtacularError
        As :func:`~spxtacular.reporter.extract_reporter_ions` (unknown plex, bad unit, ...).
    """
    key, fig_style, mode = _setup(backend, style, theme_mode)
    if isinstance(plex, ReporterIons):
        ions = plex
    else:
        ions = extract_reporter_ions(
            spectrum, plex, tolerance=tolerance, tolerance_unit=tolerance_unit, impurities=impurities
        )
    names = list(ions.channels)
    reporter_mz = np.asarray(ions.reporter_mz, dtype=np.float64)
    heights = np.asarray(ions.intensity, dtype=np.float64)
    observed = np.asarray(ions.observed_mz, dtype=np.float64)
    top = float(heights.max()) if len(heights) else 0.0
    plotted = heights / top * 100.0 if normalize and top > 0 else heights
    # The dark end of the ordinal blue ramp: the light end washes out as a filled bar.
    bar_color = theme.charge_color(3, mode)
    missing = theme.text_color("muted", mode)
    positions = np.arange(len(names), dtype=np.float64)
    bar_marks: list[Mark] = [
        Bars(
            x=positions,
            height=plotted,
            width=0.72,
            colors=[bar_color] * len(names),
            name="reporters",
            customdata=[[n, float(m), float(h)] for n, m, h in zip(names, reporter_mz, heights, strict=True)],
            hovertemplate=(
                "%{customdata[0]} (m/z %{customdata[1]:.4f})<br>intensity: %{customdata[2]:.3e}<extra></extra>"
            ),
        )
    ]
    absent = [i for i, found in enumerate(ions.found) if not found]
    if absent:
        bar_marks.append(
            LabelSet(
                x=positions[absent],
                y=np.zeros(len(absent)),
                texts=[RichText.plain("n.d.")] * len(absent),
                colors=[missing] * len(absent),
                priority=np.ones(len(absent)),
                size=fig_style.label_size * 0.9,
                leaders=False,
                gap=fig_style.label_gap,
            )
        )
    y_label = "Relative intensity (%)" if normalize else "Intensity"
    bar_axis = Axis(
        label=RichText.plain(y_label),
        lo=0.0,
        hi=None,
        tick_max=100.0 if normalize else None,
        headroom=True,
        scale_exponent=not normalize,
    )
    bar_panel = Panel(
        marks=bar_marks,
        x=Axis(
            label=RichText.plain(f"{ions.plex} channel"),
            lo=-0.6,
            hi=len(names) - 0.4,
            ticks=positions.tolist(),
            ticktext=names,
        ),
        y=bar_axis,
    )
    panels = [bar_panel]
    if show_spectrum:
        lo_mz = float(reporter_mz.min()) - 0.4
        hi_mz = float(reporter_mz.max()) + 0.4
        mz = np.asarray(spectrum.mz, dtype=np.float64)
        inten = np.asarray(spectrum.intensity, dtype=np.float64)
        window = (mz >= lo_mz) & (mz <= hi_mz)
        w_mz, w_int = mz[window], inten[window]
        w_top = float(w_int.max()) if len(w_int) else 0.0
        w_rel = w_int / w_top * 100.0 if w_top > 0 else w_int
        # The peak extract_reporter_ions picked for each channel, not everything in the window.
        assigned = np.isin(w_mz, observed[~np.isnan(observed)])
        context = theme.neutral_color(mode) if fig_style.strong_context else theme.unmatched_color(mode)
        spec_marks: list[Mark] = [
            Sticks(x=w_mz[~assigned], y=w_rel[~assigned], color=context, width=fig_style.stick_width_context,
                   name="other peaks", legend=bool((~assigned).any()) and bool(assigned.any())),
            Sticks(x=w_mz[assigned], y=w_rel[assigned], color=bar_color, width=fig_style.stick_width,
                   name="reporter", legend=bool((~assigned).any()) and bool(assigned.any())),
        ]  # fmt: skip
        if len(w_mz):
            spec_marks.append(
                HitLayer(
                    x=w_mz,
                    y=w_rel,
                    hover=[f"m/z: {m:.4f}<br>intensity: {i:.3e}" for m, i in zip(w_mz, w_int, strict=True)],
                )
            )
        panels.insert(
            0,
            Panel(
                marks=spec_marks,
                x=Axis(label=mz_label(), lo=lo_mz, hi=hi_mz),
                y=Axis(label=RichText.plain("Rel. int. (%)"), lo=0.0, tick_max=100.0, pad=0.08),
                weight=0.8,
            ),
        )
    cell = Cell(
        panels=panels,
        title=figure_title(title, "Reporter ions", fig_style),
        aspect=fig_style.aspect * (1.55 if show_spectrum else 1.0),
    )
    return _build(cell, key=key, fig_style=fig_style, size=size, mode=mode, layout_kwargs=layout_kwargs)
