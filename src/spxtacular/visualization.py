"""
Visualization tools for mass spectrometry data.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    import plotly.graph_objects as go

import numpy as np
from numpy.typing import NDArray

from . import theme
from .chromatogram import Chromatogram
from .core import Spectrum, SpectrumType
from .enums import (
    DEFAULT_FRAGMENT_TOLERANCE,
    DEFAULT_FRAGMENT_TOLERANCE_TYPE,
    PeakSelection,
    PeakSelectionLike,
    ToleranceLike,
)
from .matching import FragmentInput
from .plot_table import (
    _HIT_TARGET_SIZE,
    _LABEL_ANGLE_DEFAULT,
    _MAX_LABELS_DEFAULT,
    _PROFILE_MAX_POINTS,
    _cap_labels,
    _charge_series,
    _decimate_profile,
    _fragment_label,
    _rgba,
    _scaled_intensity,
    _sticks,
    build_annot_plot_table,
    build_plot_table,
    plot_from_table,
)
from .utils import format_precursor_charge


def _add_precursor_marker(
    fig: go.Figure,
    spectrum: Spectrum,
    theme_mode: theme.ThemeMode | None = None,
) -> None:
    """Mark the precursor m/z and its isolation window on an MSn figure.

    Reference chrome, not data: the window is a faint band and the precursor a
    hairline, both behind the peaks. Silently does nothing for a spectrum that
    carries no precursor information.
    """
    precursors = getattr(spectrum, "precursors", None)
    if not precursors:
        return

    muted = theme.text_color("muted", theme_mode)
    window = getattr(spectrum, "isolation_mz_range", None)
    if window is not None and len(window) == 2:
        lo, hi = float(window[0]), float(window[1])
        if hi > lo:
            fig.add_vrect(
                x0=lo,
                x1=hi,
                fillcolor=muted,
                opacity=0.08,
                line_width=0,
                layer="below",
            )

    for prec in precursors:
        mz_val = getattr(prec, "mz", None)
        if mz_val is None:
            continue
        charge = getattr(prec, "charge", None)
        charge_text = format_precursor_charge(charge, getattr(spectrum, "polarity", None))
        text = f"precursor {float(mz_val):.4f}" + (f" ({charge_text})" if charge_text is not None else "")
        fig.add_vline(
            x=float(mz_val),
            line_width=1,
            line_color=muted,
            layer="below",
            annotation_text=text,
            annotation_position="top right",
            annotation_font={"size": 10, "color": muted},
        )


def save_figure(fig: go.Figure, path: str | Path, scale: float = 2.0, **kwargs) -> Path:
    """Write a figure to disk, choosing the writer from the file extension.

    ``.html`` always works. Static formats (``.png``, ``.svg``, ``.pdf``,
    ``.jpg``, ``.jpeg``, ``.webp``) go through plotly's static export, which needs the
    ``kaleido`` package. Missing-package errors include an install command;
    export failures such as an invalid destination remain their original type.

    Parameters
    ----------
    fig:
        Figure to write.
    path:
        Destination. The suffix picks the format.
    scale:
        Device pixel ratio for raster formats; ``2.0`` gives a figure that still
        looks sharp in a paper or on a high-density display.
    **kwargs:
        Forwarded to Plotly's HTML or static-image writer.

    Returns
    -------
    The path written.
    """
    out = Path(path)
    suffix = out.suffix.lower()

    if suffix in ("", ".html"):
        out = out.with_suffix(".html")
        fig.write_html(str(out), **kwargs)
        return out

    static = (".png", ".svg", ".pdf", ".jpg", ".jpeg", ".webp")
    if suffix not in static:
        raise ValueError(f"unsupported figure format {suffix!r}; expected .html or one of {', '.join(static)}")

    try:
        import importlib

        importlib.import_module("kaleido")
    except (ImportError, OSError) as exc:
        raise ImportError(
            f"writing {suffix} requires the kaleido package: pip install kaleido "
            "(or save to .html, which needs nothing extra)"
        ) from exc
    fig.write_image(str(out), scale=scale, **kwargs)
    return out


def requires_plotly(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to check if plotly is installed."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            import plotly.graph_objects  # noqa: F401
        except ImportError as exc:
            raise ImportError("plotly required for plotting: pip install plotly") from exc
        return func(*args, **kwargs)

    return wrapper


def _plot_spectrum_im(
    spectrum: Spectrum,
    title: str | None = None,
    show_scores: bool = True,
    max_labels: int | None = _MAX_LABELS_DEFAULT,
    theme_mode: theme.ThemeMode | None = None,
    intensity_scale: Literal["absolute", "relative"] = "relative",
    intensity_transform: Literal["sqrt", "log"] | None = None,
    **layout_kwargs,
) -> go.Figure:
    """Stick plot with sticks coloured by ion mobility.

    The ramp is quantised into 20 bins of a single-hue sequential scale, so each
    stick takes one flat colour; the colourbar carries the mapping.

    Intensity is scaled the same way the plot-table path scales it, so switching
    ``color=`` does not silently switch the y-axis between relative and absolute.
    As there, the tooltip always reports the unscaled value.
    """
    import plotly.colors as pc
    import plotly.graph_objects as go

    mz = spectrum.mz
    intensity = spectrum.intensity
    # Scaled values are what gets drawn; `intensity` stays the number the hover
    # reports, so rescaling only ever moves the axis.
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
        if im_min == im_max:
            norm = np.zeros(len(im_arr))
        else:
            norm = np.nan_to_num((im_arr - im_min) / (im_max - im_min), nan=0.0)
    bin_idx = np.clip((norm * n_bins).astype(int), 0, n_bins - 1)
    # Single-hue sequential ramp rather than Viridis: ion mobility is a magnitude,
    # and a multi-hue ramp invents banding that isn't in the data.
    scale = theme.sequential_scale(theme_mode)
    bin_colors: list[str] = pc.sample_colorscale(scale, n_bins)

    traces: list[go.Scatter] = []
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        mz_b = mz[mask]
        int_b = intensity[mask]
        plot_b = plotted[mask]
        im_b = im_arr[mask]
        xs, ys = _sticks(mz_b, plot_b)
        hover_data: list[str] = []
        for i in range(len(mz_b)):
            tip = f"m/z: {float(mz_b[i]):.4f}<br>intensity: {float(int_b[i]):.2e}<br>{im_label}: {float(im_b[i]):.4f}"
            hover_data += [tip, tip, ""]
        traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"color": bin_colors[b], "width": 1},
                customdata=hover_data,
                hovertemplate="%{customdata}<extra></extra>",
                showlegend=False,
            )
        )

    # Invisible dummy trace whose sole purpose is rendering the colorbar
    traces.append(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={
                "colorscale": scale,
                "showscale": True,
                "cmin": im_min,
                "cmax": im_max,
                "colorbar": {
                    "title": {"text": im_label, "font": {"size": 11}},
                    "thickness": 12,
                    "outlinewidth": 0,
                    "tickfont": {"size": 10},
                    "len": 0.8,
                },
                "size": 0,
            },
            hoverinfo="none",
            showlegend=False,
        )
    )

    annotations = []
    if show_scores and spectrum.iso_score is not None:
        texts = _cap_labels(
            [f"{float(s):.2f}" if float(s) > 0.0 else "" for s in spectrum.iso_score],
            intensity,
            max_labels,
            mz,
        )
        for i, text in enumerate(texts):
            if text:
                annotations.append(
                    dict(
                        x=float(mz[i]),
                        y=float(plotted[i]),
                        text=text,
                        showarrow=False,
                        yshift=6,
                        yanchor="bottom",
                        textangle=_LABEL_ANGLE_DEFAULT,
                        font={"size": 11, "color": theme.text_color("secondary", theme_mode)},
                        xanchor="center",
                    )
                )

    fig = go.Figure(traces)
    fig.update_layout(
        template=theme.template(theme_mode),
        title=title or str(spectrum.spectrum_type or "Spectrum"),
        xaxis_title="m/z",
        yaxis_title=intensity_label,
        annotations=annotations,
        **layout_kwargs,
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


@requires_plotly
def plot_spectrum(
    spectrum: Spectrum,
    title: str | None = None,
    *,
    color: Literal["charge", "im"] | None = "charge",
    show_scores: bool = True,
    show_charges: bool | None = None,
    max_labels: int | None = _MAX_LABELS_DEFAULT,
    theme_mode: theme.ThemeMode | None = None,
    intensity_scale: Literal["absolute", "relative"] = "relative",
    intensity_transform: Literal["sqrt", "log"] | None = None,
    show_precursor: bool = True,
    render: Literal["sticks", "profile"] | None = None,
    max_points: int | None = _PROFILE_MAX_POINTS,
    **layout_kwargs,
) -> go.Figure:
    """Plot a spectrum: sticks for centroid data, a continuous trace for profile.

    Parameters
    ----------
    spectrum:
        Spectrum to plot.
    title:
        Plot title. Defaults to the spectrum type.
    color:
        Coloring mode for peaks.  ``"charge"`` (default) colours sticks by
        charge state on an ordinal ramp when charge data is present.  ``"im"``
        colours sticks by ion mobility, quantised into 20 bins of a single-hue
        sequential scale, when IM data is present; falls back to ``"charge"``
        when no IM array is available.  ``None`` renders every stick in one colour.

        ``"im"`` is a stick encoding, so it is rejected for profile data --
        centroid first, or pass ``render="sticks"`` if you really mean to draw
        every sample as a bar.
    show_scores:
        Annotate peaks with their isotope profile score when score data is
        present. Only peaks with score > 0 are labelled. Defaults to True.
    show_charges:
        Deprecated. Use ``color="charge"`` or ``color=None`` instead.
    intensity_scale:
        ``"relative"`` scales the base peak to 100. ``"absolute"`` preserves
        raw intensities on the y-axis.
    intensity_transform:
        Optional ``"sqrt"`` or ``"log"`` display transform.
    show_precursor:
        Draw precursor m/z and isolation-window markers when available.
    render:
        ``"sticks"`` or ``"profile"``. ``None`` (default) picks from
        ``spectrum.spectrum_type``: profile data is drawn as a continuous trace
        with a light fill, everything else as sticks.
    max_points:
        Cap on samples drawn for a profile trace (default 4000). Thinning keeps
        the minimum and maximum of each bucket, so no peak apex is lost;
        ``None`` draws every sample. Applies to profile renders only; a stick
        render draws every peak.
    max_labels:
        Cap on directly-drawn labels, highest-intensity peaks first (default
        ``_MAX_LABELS_DEFAULT``, currently 60). Labels are also thinned by
        collision, so a dense region keeps fewer than the cap. ``None`` labels
        every scored peak, which on a large spectrum produces an unreadable
        pile-up along the baseline -- the remaining values stay available on
        hover.
    theme_mode:
        ``"light"`` or ``"dark"``. Defaults to the global mode set by
        :func:`~spxtacular.theme.set_plot_theme`.
    **layout_kwargs:
        Forwarded to ``fig.update_layout``.
    """
    if show_charges is not None:
        warnings.warn(
            "show_charges is deprecated; use color='charge' or color=None instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        color = "charge" if show_charges else None

    if color == "im" and spectrum.im is not None and len(spectrum.im) == len(spectrum.mz):
        # The type check has to come *first*. The im path only knows how to draw
        # sticks, so routing to it before asking what kind of spectrum this is
        # drew profile data as one bar per sample -- the peak shape thrown away,
        # and the very thing profile data exists to carry.
        resolved = (
            render
            if render is not None
            else ("profile" if spectrum.spectrum_type == SpectrumType.PROFILE else "sticks")
        )
        if resolved == "profile":
            raise ValueError(
                "color='im' draws sticks, which discards the peak shape of a profile spectrum. "
                "Centroid it first (spectrum.centroid()), or pass render='sticks' to draw every "
                "sample as a stick anyway."
            )
        fig = _plot_spectrum_im(
            spectrum,
            title=title,
            show_scores=show_scores,
            max_labels=max_labels,
            theme_mode=theme_mode,
            intensity_scale=intensity_scale,
            intensity_transform=intensity_transform,
            **layout_kwargs,
        )
        if show_precursor:
            _add_precursor_marker(fig, spectrum, theme_mode)
        return fig
    table = build_plot_table(
        spectrum,
        show_charges=color == "charge",
        show_scores=show_scores,
        max_labels=max_labels,
        theme_mode=theme_mode,
        intensity_scale=intensity_scale,
        intensity_transform=intensity_transform,
    )
    fig = plot_from_table(
        table,
        title=title or str(spectrum.spectrum_type or "Spectrum"),
        theme_mode=theme_mode,
        render=render,
        max_points=max_points,
        **layout_kwargs,
    )
    if show_precursor:
        _add_precursor_marker(fig, spectrum, theme_mode)
    return fig


@requires_plotly
def mirror_plot(
    raw: Spectrum,
    deconvoluted: Spectrum,
    title: str | None = None,
    normalize: bool = True,
    show_charges: bool = True,
    show_scores: bool = True,
    max_labels: int | None = _MAX_LABELS_DEFAULT,
    theme_mode: theme.ThemeMode | None = None,
    **layout_kwargs,
) -> go.Figure:
    """Mirror plot: raw spectrum (upside-down, below) vs deconvoluted (above).

    Both spectra share the same m/z axis.  The raw spectrum is reflected below
    y=0 so you can visually trace which raw peaks contributed to each
    deconvoluted cluster.

    Parameters
    ----------
    raw:
        The undeconvoluted spectrum.
    deconvoluted:
        The deconvoluted spectrum (output of ``raw.deconvolute()``).
    title:
        Plot title.
    normalize:
        If True (default), each half is independently scaled to its own
        maximum so both fill their half of the plot symmetrically.
    show_charges:
        Colour deconvoluted sticks by charge state when charge data is present.
    show_scores:
        Annotate deconvoluted peaks with isotope profile scores (score > 0).
    max_labels:
        Maximum number of score labels, strongest peaks first.
    theme_mode:
        ``"light"`` or ``"dark"``. ``None`` uses the global plot theme.
    **layout_kwargs:
        Forwarded to ``fig.update_layout``.
    """
    import plotly.graph_objects as go

    raw_mz = raw.mz
    raw_int = raw.intensity
    dec_mz = deconvoluted.mz
    dec_int = deconvoluted.intensity
    charge = deconvoluted.charge

    # Keep the pre-normalisation values: the hover must report the intensity the
    # data actually has, not the 0-1 figure used to lay the halves out.
    raw_true = np.asarray(raw_int, dtype=np.float64)
    dec_true = np.asarray(dec_int, dtype=np.float64)

    # Normalise each half independently so they fill their half symmetrically.
    # `or 1.0` guards an all-zero (or empty) half: dividing by zero produced an
    # all-NaN array and a silently blank panel.
    if normalize:
        raw_scale = (float(raw_int.max()) if len(raw_int) > 0 else 1.0) or 1.0
        dec_scale = (float(dec_int.max()) if len(dec_int) > 0 else 1.0) or 1.0
        raw_int = raw_int / raw_scale
        dec_int = dec_int / dec_scale

    traces: list[go.Scatter] = []

    def _tri(values: NDArray[np.float64]) -> list[float]:
        """Repeat each value across the (base, tip, gap) triple `_sticks` emits."""
        out: list[float] = []
        for v in values:
            out += [float(v), float(v), float("nan")]
        return out

    # ── raw spectrum: sticks pointing downward ─────────────────────────────────
    x_raw, y_raw = _sticks(raw_mz, -raw_int)
    traces.append(
        go.Scatter(
            x=x_raw,
            y=y_raw,
            mode="lines",
            line={"color": theme.unmatched_color(theme_mode), "width": 1.0},
            name="raw",
            hovertemplate="m/z: %{x:.4f}<br>intensity: %{customdata:.2e}<extra>raw</extra>",
            customdata=_tri(raw_true),
        )
    )

    # ── deconvoluted spectrum: sticks pointing upward, coloured by charge ──────
    # Colours come from the same ordinal ramp plot_spectrum uses, so a spectrum
    # keeps its colours when you put the two figures side by side.
    has_charge = show_charges and charge is not None
    if has_charge and charge is not None:
        unique_charges = sorted(set(int(c) for c in charge))
        for z in unique_charges:
            mask = charge == z
            label = _charge_series(z)
            x, y = _sticks(dec_mz[mask], dec_int[mask])
            traces.append(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=label,
                    line={"color": theme.charge_color(z, theme_mode), "width": 1.6},
                    customdata=_tri(dec_true[mask]),
                    hovertemplate="m/z: %{x:.4f}<br>intensity: %{customdata:.2e}<extra></extra>",
                )
            )
    else:
        x, y = _sticks(dec_mz, dec_int)
        traces.append(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line={"color": theme.charge_color(1, theme_mode), "width": 1.6},
                name="deconvoluted",
                customdata=_tri(dec_true),
                hovertemplate="m/z: %{x:.4f}<br>intensity: %{customdata:.2e}<extra></extra>",
            )
        )

    # ── score annotations above deconvoluted peaks ─────────────────────────────
    annotations = []
    if show_scores and deconvoluted.iso_score is not None:
        texts = _cap_labels(
            [f"{float(s):.2f}" if float(s) > 0.0 else "" for s in deconvoluted.iso_score],
            dec_true,
            max_labels,
            dec_mz,
        )
        for i, text in enumerate(texts):
            if text:
                annotations.append(
                    dict(
                        x=float(dec_mz[i]),
                        y=float(dec_int[i]),
                        text=text,
                        showarrow=False,
                        yshift=6,
                        yanchor="bottom",
                        textangle=_LABEL_ANGLE_DEFAULT,
                        font={"size": 11, "color": theme.text_color("secondary", theme_mode)},
                        xanchor="center",
                    )
                )

    fig = go.Figure(traces)
    fig.update_layout(
        template=theme.template(theme_mode),
        title=title or "Raw vs Deconvoluted",
        xaxis_title="m/z",
        yaxis_title="Normalised intensity" if normalize else "Intensity",
        # The mirror axis is the one place a zero line is data, not chrome: it is
        # the boundary the two spectra are reflected across.
        yaxis={
            "zeroline": True,
            "zerolinewidth": 1,
            "zerolinecolor": theme.text_color("muted", theme_mode),
        },
        showlegend=True,
        annotations=annotations,
        **layout_kwargs,
    )
    return fig


@requires_plotly
def annotate_spectrum(
    spectrum: Spectrum,
    fragments: FragmentInput,
    tolerance: float = DEFAULT_FRAGMENT_TOLERANCE,
    tolerance_type: ToleranceLike = DEFAULT_FRAGMENT_TOLERANCE_TYPE,
    title: str | None = None,
    peak_selection: PeakSelectionLike = PeakSelection.CLOSEST,
    include_sequence: bool = False,
    max_labels: int | None = _MAX_LABELS_DEFAULT,
    theme_mode: theme.ThemeMode | None = None,
    intensity_scale: Literal["absolute", "relative"] = "relative",
    intensity_transform: Literal["sqrt", "log"] | None = None,
    texture: bool = False,
    show_precursor: bool = True,
    **layout_kwargs,
) -> go.Figure:
    """Plot a spectrum with matched fragment ion annotations.

    Unmatched peaks are drawn in light grey.  Matched peaks are coloured by
    ion series (b=blue, y=red, a=green, …) and labelled.

    Parameters
    ----------
    spectrum:
        Centroid spectrum to plot.
    fragments:
        Fragment objects from peptacular to match against peaks.
    tolerance:
        Matching tolerance.
    tolerance_type:
        ``"Da"`` or ``"ppm"``.
    title:
        Plot title.
    peak_selection:
        Which peak(s) to annotate per fragment — ``"closest"``, ``"largest"``,
        or ``"all"``.  See :func:`~spxtacular.matching.match_fragments`.
    include_sequence:
        Embed the residue sequence in each label (e.g. ``b3{PEP}``).
        Set to ``False`` for compact labels (``b3``).
    max_labels:
        Maximum number of direct ion labels, strongest peaks first.
    theme_mode:
        ``"light"`` or ``"dark"``. ``None`` uses the global plot theme.
    intensity_scale:
        ``"relative"`` scales the base peak to 100. ``"absolute"`` preserves
        raw intensities on the y-axis.
    intensity_transform:
        Optional ``"sqrt"`` or ``"log"`` display transform.
    texture:
        Give each ion series a distinct dash pattern.
    show_precursor:
        Draw precursor m/z and isolation-window markers when available.
    **layout_kwargs:
        Forwarded to ``fig.update_layout``.

    Returns
    -------
    plotly ``Figure``.
    """
    table = build_annot_plot_table(
        spectrum,
        fragments,
        tolerance,
        tolerance_type,
        peak_selection,
        include_sequence,
        max_labels=max_labels,
        theme_mode=theme_mode,
        intensity_scale=intensity_scale,
        intensity_transform=intensity_transform,
        texture=texture,
    )
    fig = plot_from_table(table, title=title or "Annotated spectrum", theme_mode=theme_mode, **layout_kwargs)
    if show_precursor:
        _add_precursor_marker(fig, spectrum, theme_mode)
    return fig


@requires_plotly
def plot_chromatogram(
    chromatograms: Chromatogram | Sequence[Chromatogram] | Iterable[Spectrum],
    title: str | None = None,
    theme_mode: theme.ThemeMode | None = None,
    show_apex: bool = True,
    fill: bool | None = None,
    **layout_kwargs,
) -> go.Figure:
    """Plot one or more chromatograms against retention time.

    Accepts :class:`~spxtacular.chromatogram.Chromatogram` objects, or an iterable
    of spectra -- in which case a TIC is extracted for you, which is the usual
    "what does this run look like" first glance::

        with spx.Reader("run.d") as reader:
            spx.plot_chromatogram(reader.ms1).show()

    Parameters
    ----------
    chromatograms:
        A chromatogram, a sequence of them, or an iterable of spectra to
        extract a TIC from.
    title:
        Plot title.
    theme_mode:
        ``"light"`` or ``"dark"``.
    show_apex:
        Label each trace's apex with its retention time. Suppressed above four
        traces, where the labels start competing with the data.
    fill:
        Fill under the trace. Defaults to on for a single trace and off for
        several, where overlapping washes obscure each other.
    **layout_kwargs:
        Forwarded to ``fig.update_layout``.

    Returns
    -------
    plotly ``Figure``.
    """
    import plotly.graph_objects as go

    from .chromatogram import Chromatogram as _Chrom
    from .chromatogram import extract_chromatogram

    if isinstance(chromatograms, _Chrom):
        traces_in = [chromatograms]
    elif isinstance(chromatograms, Sequence) and all(isinstance(c, _Chrom) for c in chromatograms):
        traces_in = list(cast("Sequence[_Chrom]", chromatograms))
    else:
        spectra = cast("Iterable[Spectrum]", chromatograms)
        traces_in = [extract_chromatogram(spectra)]

    if fill is None:
        fill = len(traces_in) == 1

    units = {chrom.meta.get("rt_unit", "s") for chrom in traces_in if len(chrom)}
    if len(units) > 1:
        raise ValueError("Cannot plot retention times and scan indices on the same axis")
    unit = next(iter(units), "s")
    if unit not in ("s", "scan_index"):
        raise ValueError(f"Unsupported chromatogram time unit: {unit!r}")
    axis_title = "Scan index" if unit == "scan_index" else "Retention time (s)"
    time_label = "Scan index" if unit == "scan_index" else "RT"
    time_suffix = "" if unit == "scan_index" else " s"
    fig = go.Figure()
    for i, chrom in enumerate(traces_in):
        # Several traces are distinct series, so they take categorical slots; a
        # lone trace has no identity to signal and takes the default hue.
        color = theme.ion_color(theme._ION_SLOTS[i % len(theme._ION_SLOTS)], theme_mode)
        if len(traces_in) == 1:
            color = theme.charge_color(1, theme_mode)

        fig.add_trace(
            go.Scatter(
                x=chrom.rt,
                y=chrom.intensity,
                mode="lines",
                name=chrom.label or f"trace {i + 1}",
                line={"color": color, "width": 1.8},
                fill="tozeroy" if fill else None,
                fillcolor=_rgba(color, 0.12) if fill else None,
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
                fig.add_annotation(
                    x=float(chrom.rt[apex]),
                    y=float(chrom.intensity[apex]),
                    text=f"{chrom.rt[apex]:.1f}{time_suffix}",
                    showarrow=False,
                    yshift=10,
                    font={"size": 10, "color": theme.text_color("secondary", theme_mode)},
                    xanchor="center",
                )

    fig.update_layout(
        template=theme.template(theme_mode),
        title=title or (traces_in[0].label if len(traces_in) == 1 else "Chromatograms"),
        xaxis_title=axis_title,
        yaxis_title="Intensity",
        showlegend=len(traces_in) > 1,
        **layout_kwargs,
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


@requires_plotly
def plot_xic(
    spectra: Iterable[Spectrum],
    targets: Sequence[float] | float,
    tolerance: float = 20.0,
    tolerance_type: ToleranceLike = "ppm",
    im_window: tuple[float, float] | None = None,
    aggregate: Literal["sum", "max"] = "sum",
    title: str | None = None,
    theme_mode: theme.ThemeMode | None = None,
    **layout_kwargs,
) -> go.Figure:
    """Extract and plot ion chromatograms in one call.

    Every target is extracted in a single pass over ``spectra``, so tracing ten
    m/z values costs one walk of the reader rather than ten::

        with spx.Reader("run.d") as reader:
            spx.plot_xic(reader.ms1, [500.2649, 622.0290], tolerance=20).show()

    See :func:`~spxtacular.chromatogram.extract_xic` for the extraction
    parameters, including ``im_window``, which is what makes a trace selective on
    ion-mobility data.
    """
    from .chromatogram import extract_xic

    chroms = extract_xic(
        spectra,
        targets,
        tolerance=tolerance,
        tolerance_type=tolerance_type,
        im_window=im_window,
        aggregate=aggregate,
    )
    unit = "ppm" if str(tolerance_type).lower() == "ppm" else "Da"
    default_title = f"Extracted ion chromatogram{'s' if len(chroms) > 1 else ''} (±{tolerance:g} {unit})"
    return plot_chromatogram(chroms, title=title or default_title, theme_mode=theme_mode, **layout_kwargs)


@requires_plotly
def profile_centroid_plot(
    profile: Spectrum,
    centroids: Spectrum | None = None,
    title: str | None = None,
    theme_mode: theme.ThemeMode | None = None,
    max_points: int | None = _PROFILE_MAX_POINTS,
    **layout_kwargs,
) -> go.Figure:
    """Profile trace with the centroided peaks drawn on top.

    The view for checking that centroiding did the right thing: the continuous
    signal underneath, and a stick at each m/z the centroider decided was a peak.
    A stick off the apex means a mis-assigned centre; an apex with no stick means
    a peak was dropped.

    Use this view to verify that thresholding did not remove a real peak and
    that fitted centers remain aligned with their profile apexes. Flat-topped
    peaks are supported and produce one centroid at the middle of the plateau.

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
        Cap on drawn profile samples; see :func:`~spxtacular.plot_table.plot_from_table`.
    **layout_kwargs:
        Forwarded to ``fig.update_layout``.

    Returns
    -------
    plotly ``Figure``.
    """
    import plotly.graph_objects as go

    if centroids is None:
        centroids = profile.centroid()

    prof_mz, prof_int = _decimate_profile(
        np.asarray(profile.mz, dtype=np.float64),
        np.asarray(profile.intensity, dtype=np.float64),
        max_points,
    )

    profile_color = theme.unmatched_color(theme_mode)
    centroid_color = theme.charge_color(1, theme_mode)

    fig = go.Figure()
    # Profile underneath and recessive: it is the context the centroids are
    # checked against, not the subject.
    fig.add_trace(
        go.Scatter(
            x=prof_mz,
            y=prof_int,
            mode="lines",
            line={"color": profile_color, "width": 1.2},
            fill="tozeroy",
            fillcolor=_rgba(profile_color, 0.18),
            name="profile",
            hovertemplate="m/z: %{x:.4f}<br>intensity: %{y:.4g}<extra>profile</extra>",
        )
    )

    if len(centroids) > 0:
        xs, ys = _sticks(
            np.asarray(centroids.mz, dtype=np.float64),
            np.asarray(centroids.intensity, dtype=np.float64),
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"color": centroid_color, "width": 1.6},
                name="centroids",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=centroids.mz,
                y=centroids.intensity,
                mode="markers",
                marker={"size": _HIT_TARGET_SIZE, "color": "rgba(0,0,0,0)"},
                customdata=[
                    f"m/z: {m:.4f}<br>intensity: {i:.4g}"
                    for m, i in zip(centroids.mz, centroids.intensity, strict=True)
                ],
                hovertemplate="%{customdata}<extra>centroid</extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        template=theme.template(theme_mode),
        title=title or f"Profile vs centroids — {len(centroids)} peaks from {len(profile)} samples",
        xaxis_title="m/z",
        yaxis_title="Intensity",
        showlegend=True,
        **layout_kwargs,
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


@requires_plotly
def sequence_coverage_plot(
    spectrum: Spectrum,
    peptide: str,
    fragments: FragmentInput,
    tolerance: float = DEFAULT_FRAGMENT_TOLERANCE,
    tolerance_type: ToleranceLike = DEFAULT_FRAGMENT_TOLERANCE_TYPE,
    peak_selection: PeakSelectionLike = PeakSelection.CLOSEST,
    title: str | None = None,
    theme_mode: theme.ThemeMode | None = None,
    **layout_kwargs,
) -> go.Figure:
    """Sequence coverage ladder: which backbone bonds the spectrum actually evidences.

    The residues run left to right. A tick above and to the *left* of a residue
    marks an N-terminal (a/b/c) fragment ending at that bond; a tick below and to
    the *right* marks a C-terminal (x/y/z) fragment starting there. A bond with
    ticks on both sides is confirmed from both directions.

    This is the standard companion to an annotated spectrum: the spectrum shows
    that peaks matched, the ladder shows *where along the peptide* they matched,
    which is what tells you whether an identification is localised or leaning on
    one end of the molecule.

    Parameters
    ----------
    spectrum:
        The spectrum the fragments were matched against.
    peptide:
        Residue sequence, one character per residue. Modifications in ProForma
        brackets are not rendered -- pass the stripped sequence.
    fragments:
        Fragment objects to match, as for :func:`~spxtacular.matching.match_fragments`.
    tolerance, tolerance_type, peak_selection:
        Matching parameters.
    title:
        Plot title.
    theme_mode:
        ``"light"`` or ``"dark"``.
    **layout_kwargs:
        Forwarded to ``fig.update_layout``.

    Returns
    -------
    plotly ``Figure``.
    """
    import plotly.graph_objects as go

    from .matching import match_fragments

    residues = list(peptide)
    n_res = len(residues)
    if n_res == 0:
        raise ValueError("peptide must contain at least one residue")

    matches = match_fragments(spectrum, fragments, tolerance, tolerance_type, peak_selection)

    # A fragment of length k evidences the bond after residue k (N-terminal
    # series) or before residue n-k (C-terminal series).
    n_term_bonds: set[int] = set()
    c_term_bonds: set[int] = set()
    n_series = {"a", "b", "c"}
    c_series = {"x", "y", "z"}

    for m in matches:
        frag = m.fragment
        ion = str(getattr(frag, "ion_type", "")).lower()
        pos = getattr(frag, "position", None)
        if not isinstance(pos, int) or pos <= 0 or pos >= n_res:
            continue
        if ion in n_series:
            n_term_bonds.add(pos)
        elif ion in c_series:
            c_term_bonds.add(n_res - pos)

    fig = go.Figure()
    ink = theme.text_color("primary", theme_mode)
    n_color = theme.ion_color("b", theme_mode)
    c_color = theme.ion_color("y", theme_mode)

    for i, residue in enumerate(residues):
        fig.add_annotation(
            x=i,
            y=0,
            text=residue,
            showarrow=False,
            font={"size": 15, "color": ink, "family": theme._FONT_FAMILY},
            xanchor="center",
            yanchor="middle",
        )

    # Ticks sit on the bond, i.e. halfway between two residues.
    for bond in n_term_bonds:
        x = bond - 0.5
        fig.add_shape(type="line", x0=x, x1=x, y0=0.18, y1=0.55, line={"color": n_color, "width": 2})
        fig.add_shape(type="line", x0=x, x1=x - 0.28, y0=0.55, y1=0.55, line={"color": n_color, "width": 2})
    for bond in c_term_bonds:
        x = bond - 0.5
        fig.add_shape(type="line", x0=x, x1=x, y0=-0.18, y1=-0.55, line={"color": c_color, "width": 2})
        fig.add_shape(type="line", x0=x, x1=x + 0.28, y0=-0.55, y1=-0.55, line={"color": c_color, "width": 2})

    n_bonds = max(n_res - 1, 1)
    covered = len(n_term_bonds | c_term_bonds)
    subtitle = f"{covered}/{n_bonds} backbone bonds covered ({covered / n_bonds:.0%})"

    # Legend proxies: two invisible traces so the ion-series colours are named
    # rather than left for the reader to infer from the ticks.
    for name, color in (("N-term (a/b/c)", n_color), ("C-term (x/y/z)", c_color)):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line={"color": color, "width": 2}, name=name))

    fig.update_layout(
        template=theme.template(theme_mode),
        # The count belongs in the title, not repeated underneath it.
        title=title or f"Sequence coverage — {subtitle}",
        showlegend=True,
        height=210,
        # Wider right margin so the legend keys are not clipped by the paper edge.
        margin={"l": 28, "r": 64, "t": 64, "b": 28},
        **layout_kwargs,
    )
    fig.update_xaxes(range=[-0.8, n_res - 0.2], visible=False)
    fig.update_yaxes(range=[-1.0, 1.0], visible=False)
    return fig


@requires_plotly
def mass_error_plot(
    spectrum: Spectrum,
    fragments: FragmentInput,
    tolerance: float = DEFAULT_FRAGMENT_TOLERANCE,
    tolerance_type: ToleranceLike = DEFAULT_FRAGMENT_TOLERANCE_TYPE,
    peak_selection: PeakSelectionLike = PeakSelection.CLOSEST,
    unit: str = "ppm",
    title: str | None = None,
    max_labels: int | None = _MAX_LABELS_DEFAULT,
    theme_mode: theme.ThemeMode | None = None,
    **layout_kwargs,
) -> go.Figure:
    """Bubble plot of mass errors vs m/z.

    Each matched fragment is shown as a bubble whose x-position is the
    observed m/z, y-position is the mass error (ppm or Da), and size is
    proportional to the peak intensity.  Bubbles are coloured by ion series.

    Parameters
    ----------
    spectrum:
        Spectrum to plot.
    fragments:
        Fragment objects from peptacular to match against peaks.
    tolerance:
        Matching tolerance.
    tolerance_type:
        ``"Da"`` or ``"ppm"``.
    peak_selection:
        ``"closest"``, ``"largest"``, or ``"all"``.
    unit:
        Error unit: ``"ppm"`` or ``"da"``.
    title:
        Plot title.
    max_labels:
        Cap on directly-drawn mzPAF labels, highest-intensity first (default
        ``_MAX_LABELS_DEFAULT``, currently 60), with the same collision
        avoidance the spectrum plots use. ``None`` labels every bubble, which
        for a few hundred matches is an unreadable smear -- the labels stay on
        hover regardless.
    theme_mode:
        ``"light"`` or ``"dark"``. ``None`` uses the global plot theme.
    **layout_kwargs:
        Forwarded to ``fig.update_layout``.
    """
    import plotly.graph_objects as go

    from .matching import match_fragments

    matches = match_fragments(spectrum, fragments, tolerance, tolerance_type, peak_selection)

    if not matches:
        # The empty case is still a figure someone looks at: without the template
        # a dark-mode caller got a white plotly default, and without axis titles
        # an unlabelled empty box.
        fig = go.Figure()
        fig.update_layout(
            template=theme.template(theme_mode),
            title=title or "Mass Errors (no matches)",
            xaxis_title="m/z",
            yaxis_title=f"Error ({unit})",
            showlegend=False,
            **layout_kwargs,
        )
        return fig

    mzs = [m.peak_mz for m in matches]
    errors = [m.ppm_error if unit == "ppm" else m.da_error for m in matches]
    intensities = [m.peak_intensity for m in matches]
    ion_types = [
        m.fragment.ion_type.value if hasattr(m.fragment.ion_type, "value") else str(m.fragment.ion_type)
        for m in matches
    ]

    # Normalise bubble sizes. `or 1.0` catches an all-zero-intensity match set,
    # which is real (thresholded or fully background-subtracted data) and used to
    # raise ZeroDivisionError -- the `if intensities` guard only caught an empty list.
    max_int = (max(intensities) if intensities else 1.0) or 1.0
    sizes = [max(6, 36 * i / max_int) for i in intensities]

    colors = [theme.ion_color(it, theme_mode) for it in ion_types]

    # mzPAF labels, so a 2+ and a 1+ of the same ion don't both render as "b3".
    # Thinned exactly as the spectrum plots thin theirs: a label on every bubble
    # is the same unreadable smear here as it is along a baseline.
    all_labels = [_fragment_label(m.fragment, False) for m in matches]
    labels = _cap_labels(
        list(all_labels),
        np.asarray(intensities, dtype=np.float64),
        max_labels,
        np.asarray(mzs, dtype=np.float64),
    )
    # Thinning drops labels off the plot, so the hover has to carry the full set
    # -- otherwise capping would make a bubble's identity unreachable.
    hover_data = [[float(i), lab] for i, lab in zip(intensities, all_labels, strict=True)]

    fig = go.Figure(
        go.Scatter(
            x=mzs,
            y=errors,
            mode="markers+text",
            marker={
                "size": sizes,
                "color": colors,
                "opacity": 0.7,
                # Separates two overlapping bubbles. Theme-aware: a fixed dark
                # grey disappeared into the dark surface.
                "line": {"width": 1, "color": theme.marker_outline(theme_mode)},
            },
            text=labels,
            textposition="top center",
            textfont={"size": 9},
            hovertemplate=(
                f"m/z: %{{x:.4f}}<br>error ({unit}): %{{y:.4f}}<br>"
                "intensity: %{customdata[0]:.2e}<br>%{customdata[1]}<extra></extra>"
            ),
            customdata=hover_data,
        )
    )

    # Solid hairline, not dashed: dashing reads as "threshold" or "projection"
    # when this is just the zero-error reference.
    fig.add_hline(y=0, line_color=theme.text_color("muted", theme_mode), line_width=1)
    fig.update_layout(
        template=theme.template(theme_mode),
        title=title or "Mass Errors",
        xaxis_title="m/z",
        yaxis_title=f"Error ({unit})",
        showlegend=False,
        **layout_kwargs,
    )
    return fig


def _add_stick_traces(
    fig: go.Figure,
    table,
    row: int,
    col: int,
    theme_mode: theme.ThemeMode | None = None,
    negate: bool = False,
) -> None:
    """Add a plot table to a subplot as one trace per (series, colour) group.

    One trace *per group*, not per peak. Drawing a separate trace for every peak
    makes plotly allocate per-trace state thousands of times over -- a 5000-peak
    spectrum became 5000 traces and a figure the browser struggles to render,
    for a picture identical to the handful of traces this produces.
    """
    import pandas as pd
    import plotly.graph_objects as go

    for (series, color), group in table.groupby(["series", "color"], sort=False, dropna=False):
        if pd.isna(color):
            color = theme.unmatched_color(theme_mode)
        mz_arr = group["mz"].to_numpy(dtype=np.float64)
        int_arr = group["intensity"].to_numpy(dtype=np.float64)
        if negate:
            int_arr = -int_arr
        xs, ys = _sticks(mz_arr, int_arr)

        hover_data: list[str] = []
        for h in group["hover"].tolist():
            hover_data += [h, h, ""]

        first = group.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=str(series),
                line={"color": str(color), "width": float(first["linewidth"])},
                opacity=float(first["opacity"]),
                customdata=hover_data,
                hovertemplate="%{customdata}<extra></extra>",
                showlegend=False,
            ),
            row=row,
            col=col,
        )


@requires_plotly
def facet_plot(
    spectrum: Spectrum,
    fragments: FragmentInput | None = None,
    mirror_spectrum: Spectrum | None = None,
    title: str | None = None,
    tolerance: float = DEFAULT_FRAGMENT_TOLERANCE,
    tolerance_type: ToleranceLike = DEFAULT_FRAGMENT_TOLERANCE_TYPE,
    peak_selection: PeakSelectionLike = PeakSelection.CLOSEST,
    include_sequence: bool = False,
    unit: str = "ppm",
    max_labels: int | None = _MAX_LABELS_DEFAULT,
    theme_mode: theme.ThemeMode | None = None,
    **layout_kwargs,
) -> go.Figure:
    """Multi-panel facet plot combining spectrum, mass errors, and mirror.

    Panels (top to bottom):
    1. Annotated spectrum (always shown)
    2. Mass errors bubble chart (shown if ``fragments`` is provided)
    3. Mirror spectrum (shown if ``mirror_spectrum`` is provided)

    Parameters
    ----------
    spectrum:
        Primary spectrum to plot.
    fragments:
        Fragment objects for annotation and mass error panels.
    mirror_spectrum:
        Optional second spectrum shown as a mirror below.
    title:
        Plot title.
    tolerance:
        Matching tolerance.
    tolerance_type:
        ``"Da"`` or ``"ppm"``.
    peak_selection:
        ``"closest"``, ``"largest"``, or ``"all"``.
    include_sequence:
        Embed residue sequence in annotation labels.
    unit:
        Error unit for mass error panel: ``"ppm"`` or ``"da"``.
    max_labels:
        Maximum number of direct labels in the annotated and mass-error panels.
    theme_mode:
        ``"light"`` or ``"dark"``. ``None`` uses the global plot theme.
    **layout_kwargs:
        Forwarded to ``fig.update_layout``.
    """
    from plotly.subplots import make_subplots

    n_rows = 1
    subtitles = ["Spectrum"]
    if fragments is not None:
        n_rows += 1
        subtitles.append("Mass Errors")
    if mirror_spectrum is not None:
        n_rows += 1
        subtitles.append("Mirror")

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=subtitles,
    )

    # Panel 1: annotated spectrum (or plain spectrum)
    if fragments is not None:
        table = build_annot_plot_table(
            spectrum,
            fragments,
            tolerance,
            tolerance_type,
            peak_selection,
            include_sequence,
            max_labels=max_labels,
            theme_mode=theme_mode,
        )
    else:
        table = build_plot_table(spectrum, max_labels=max_labels, theme_mode=theme_mode)

    import plotly.graph_objects as go

    _add_stick_traces(fig, table, row=1, col=1, theme_mode=theme_mode)

    # Carry the ion labels through. The panel exists to show the annotation, so
    # silently dropping the labels defeated its purpose.
    label_mask = table["label"].notna() & (table["label"] != "")
    for _, lrow in table[label_mask].iterrows():
        fig.add_annotation(
            x=float(lrow["mz"]),
            y=float(lrow["intensity"]),
            text=str(lrow["label"]),
            showarrow=False,
            yshift=6,
            yanchor="bottom",
            textangle=_LABEL_ANGLE_DEFAULT,
            font={"size": 10, "color": theme.text_color("secondary", theme_mode)},
            xanchor="center",
            row=1,
            col=1,
        )
    # The table knows what scaling it applied; hardcoding "Intensity" here
    # mislabels the panel, which is relative-scaled by default.
    fig.update_yaxes(title_text=table.attrs.get("intensity_label", "Intensity"), row=1, col=1, rangemode="tozero")

    current_row = 2

    # Panel 2: mass errors
    if fragments is not None:
        from .matching import match_fragments

        matches = match_fragments(spectrum, fragments, tolerance, tolerance_type, peak_selection)
        if matches:
            mzs = [m.peak_mz for m in matches]
            errors = [m.ppm_error if unit == "ppm" else m.da_error for m in matches]
            intensities = [m.peak_intensity for m in matches]
            max_int = max(intensities) or 1.0
            sizes = [max(6, 28 * i / max_int) for i in intensities]

            ion_types = [
                m.fragment.ion_type.value if hasattr(m.fragment.ion_type, "value") else str(m.fragment.ion_type)
                for m in matches
            ]
            colors = [theme.ion_color(it, theme_mode) for it in ion_types]

            fig.add_trace(
                go.Scatter(
                    x=mzs,
                    y=errors,
                    mode="markers",
                    marker={"size": sizes, "color": colors, "opacity": 0.7},
                    showlegend=False,
                ),
                row=current_row,
                col=1,
            )
        fig.update_yaxes(title_text=f"Error ({unit})", row=current_row, col=1)
        current_row += 1

    # Panel 3: mirror spectrum
    if mirror_spectrum is not None:
        mirror_table = build_plot_table(mirror_spectrum, max_labels=max_labels, theme_mode=theme_mode)
        _add_stick_traces(fig, mirror_table, row=current_row, col=1, theme_mode=theme_mode, negate=True)
        # As for panel 1: take the label from the table rather than hardcoding
        # "Intensity" onto axis values that are relative-scaled by default.
        fig.update_yaxes(title_text=mirror_table.attrs.get("intensity_label", "Intensity"), row=current_row, col=1)

    fig.update_xaxes(title_text="m/z", row=n_rows, col=1)
    fig.update_layout(
        template=theme.template(theme_mode),
        title=title or "Facet Plot",
        height=300 * n_rows,
        showlegend=False,
        **layout_kwargs,
    )
    # Subplot titles are annotations; style them as headings rather than leaving
    # them at plotly's default so they don't compete with the figure title.
    for annotation in fig.layout.annotations[:n_rows]:
        annotation.font = {"size": 12, "color": theme.text_color("secondary", theme_mode)}
    return fig
