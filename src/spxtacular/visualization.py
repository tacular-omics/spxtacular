"""
Visualization tools for mass spectrometry data.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import plotly.graph_objects as go

import numpy as np
from numpy.typing import NDArray

from . import theme
from .core import Spectrum
from .enums import (
    DEFAULT_FRAGMENT_TOLERANCE,
    DEFAULT_FRAGMENT_TOLERANCE_TYPE,
    PeakSelection,
    PeakSelectionLike,
    ToleranceLike,
)
from .matching import FragmentInput
from .plot_table import (
    _MAX_LABELS_DEFAULT,
    _cap_labels,
    _charge_series,
    _fragment_label,
    _sticks,
    build_annot_plot_table,
    build_plot_table,
    plot_from_table,
)


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
    **layout_kwargs,
) -> go.Figure:
    """Stick plot with sticks coloured by ion mobility.

    The ramp is quantised into 20 bins of a single-hue sequential scale, so each
    stick takes one flat colour; the colourbar carries the mapping.
    """
    import plotly.colors as pc
    import plotly.graph_objects as go

    mz = spectrum.mz
    intensity = spectrum.intensity
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
        im_b = im_arr[mask]
        xs, ys = _sticks(mz_b, int_b)
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
                        y=float(intensity[i]),
                        text=text,
                        showarrow=False,
                        yshift=8,
                        font={"size": 11, "color": theme.text_color("secondary", theme_mode)},
                        xanchor="center",
                    )
                )

    fig = go.Figure(traces)
    fig.update_layout(
        template=theme.template(theme_mode),
        title=title or str(spectrum.spectrum_type or "Spectrum"),
        xaxis_title="m/z",
        yaxis_title="Intensity",
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
    **layout_kwargs,
) -> go.Figure:
    """Plot spectrum as a stick plot using plotly.

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
    show_scores:
        Annotate peaks with their isotope profile score when score data is
        present. Only peaks with score > 0 are labelled. Defaults to True.
    show_charges:
        Deprecated. Use ``color="charge"`` or ``color=None`` instead.
    max_labels:
        Cap on directly-drawn labels, highest-intensity peaks first
        (default 25). ``None`` labels every scored peak, which on a large
        spectrum produces an unreadable pile-up along the baseline -- the
        remaining values stay available on hover.
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
        return _plot_spectrum_im(
            spectrum,
            title=title,
            show_scores=show_scores,
            max_labels=max_labels,
            theme_mode=theme_mode,
            **layout_kwargs,
        )
    table = build_plot_table(
        spectrum,
        show_charges=color == "charge",
        show_scores=show_scores,
        max_labels=max_labels,
        theme_mode=theme_mode,
    )
    return plot_from_table(
        table,
        title=title or str(spectrum.spectrum_type or "Spectrum"),
        theme_mode=theme_mode,
        **layout_kwargs,
    )


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
                        yshift=8,
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
    )
    fig = plot_from_table(table, title=title or "Annotated spectrum", theme_mode=theme_mode, **layout_kwargs)
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
    **layout_kwargs:
        Forwarded to ``fig.update_layout``.
    """
    import plotly.graph_objects as go

    from .matching import match_fragments

    matches = match_fragments(spectrum, fragments, tolerance, tolerance_type, peak_selection)

    if not matches:
        fig = go.Figure()
        fig.update_layout(title=title or "Mass Errors (no matches)", **layout_kwargs)
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
    labels = [_fragment_label(m.fragment, False) for m in matches]

    fig = go.Figure(
        go.Scatter(
            x=mzs,
            y=errors,
            mode="markers+text",
            marker={"size": sizes, "color": colors, "opacity": 0.7, "line": {"width": 1, "color": "#333"}},
            text=labels,
            textposition="top center",
            textfont={"size": 9},
            hovertemplate=(
                f"m/z: %{{x:.4f}}<br>error ({unit}): %{{y:.4f}}<br>intensity: %{{customdata:.2e}}<extra></extra>"
            ),
            customdata=intensities,
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
            yshift=8,
            font={"size": 10, "color": theme.text_color("secondary", theme_mode)},
            xanchor="center",
            row=1,
            col=1,
        )
    fig.update_yaxes(title_text="Intensity", row=1, col=1, rangemode="tozero")

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
        fig.update_yaxes(title_text="Intensity", row=current_row, col=1)

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
