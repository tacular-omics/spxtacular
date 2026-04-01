"""
Visualization tools for mass spectrometry data.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import plotly.graph_objects as go

import numpy as np

from .core import Spectrum
from .enums import PeakSelection, PeakSelectionLike, ToleranceLike, ToleranceType
from .matching import FragmentInput
from .plot_table import build_annot_plot_table, build_plot_table, plot_from_table


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


@requires_plotly
def plot_spectrum(
    spectrum: Spectrum,
    title: str | None = None,
    show_charges: bool = True,
    show_scores: bool = True,
    **layout_kwargs,
) -> go.Figure:
    """Plot spectrum as a stick plot using plotly.

    Parameters
    ----------
    spectrum:
        Spectrum to plot.
    title:
        Plot title. Defaults to the spectrum type.
    show_charges:
        Colour sticks by charge state when charge data is present.
    show_scores:
        Annotate peaks with their isotope profile score when score data is
        present. Only peaks with score > 0 are labelled. Defaults to True.
    **layout_kwargs:
        Forwarded to ``fig.update_layout``.
    """
    table = build_plot_table(spectrum, show_charges=show_charges, show_scores=show_scores)
    fig = plot_from_table(
        table,
        title=title or str(spectrum.spectrum_type or "Spectrum"),
        **layout_kwargs,
    )
    return fig


@requires_plotly
def mirror_plot(
    raw: Spectrum,
    deconvoluted: Spectrum,
    title: str | None = None,
    normalize: bool = True,
    show_charges: bool = True,
    show_scores: bool = True,
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

    # Normalise each half independently so they fill their half symmetrically
    if normalize:
        raw_scale = float(raw_int.max()) if len(raw_int) > 0 else 1.0
        dec_scale = float(dec_int.max()) if len(dec_int) > 0 else 1.0
        raw_int = raw_int / raw_scale
        dec_int = dec_int / dec_scale

    traces: list[go.Scatter] = []

    # ── raw spectrum: sticks pointing downward ─────────────────────────────────
    x_raw, y_raw = _sticks(raw_mz, -raw_int)
    traces.append(
        go.Scatter(
            x=x_raw,
            y=y_raw,
            mode="lines",
            line={"color": "#aaaaaa"},
            name="raw",
            hovertemplate="m/z: %{x:.4f}<br>intensity: %{customdata:.2e}<extra>raw</extra>",
            customdata=np.abs(y_raw),
        )
    )

    # ── deconvoluted spectrum: sticks pointing upward, coloured by charge ──────
    has_charge = show_charges and charge is not None
    if has_charge and charge is not None:
        unique_charges = sorted(set(int(c) for c in charge))
        for z in unique_charges:
            mask = charge == z
            label = "singleton" if z == -1 else f"z={z}"
            x, y = _sticks(dec_mz[mask], dec_int[mask])
            traces.append(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=label,
                    hovertemplate="m/z: %{x:.4f}<br>intensity: %{y:.2e}<extra></extra>",
                )
            )
    else:
        x, y = _sticks(dec_mz, dec_int)
        traces.append(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line={"color": "steelblue"},
                name="deconvoluted",
                hovertemplate="m/z: %{x:.4f}<br>intensity: %{y:.2e}<extra></extra>",
            )
        )

    # ── score annotations above deconvoluted peaks ─────────────────────────────
    annotations = []
    if show_scores and deconvoluted.iso_score is not None:
        for i, s in enumerate(deconvoluted.iso_score):
            if s > 0.0:
                annotations.append(
                    dict(
                        x=float(dec_mz[i]),
                        y=float(dec_int[i]),
                        text=f"{s:.2f}",
                        showarrow=False,
                        yshift=6,
                        font={"size": 9, "color": "#555555"},
                        xanchor="center",
                    )
                )

    fig = go.Figure(traces)
    fig.update_layout(
        title=title or "Raw vs Deconvoluted",
        xaxis_title="m/z",
        yaxis_title="Normalised intensity" if normalize else "Intensity",
        yaxis={"zeroline": True, "zerolinewidth": 1, "zerolinecolor": "#333333"},
        showlegend=True,
        annotations=annotations,
        **layout_kwargs,
    )
    return fig


@requires_plotly
def annotate_spectrum(
    spectrum: Spectrum,
    fragments: FragmentInput,
    tolerance: float = 0.02,
    tolerance_type: ToleranceLike = ToleranceType.PPM,
    title: str | None = None,
    peak_selection: PeakSelectionLike = PeakSelection.CLOSEST,
    include_sequence: bool = False,
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
    table = build_annot_plot_table(spectrum, fragments, tolerance, tolerance_type, peak_selection, include_sequence)
    fig = plot_from_table(table, title=title or "Annotated spectrum", **layout_kwargs)
    return fig


@requires_plotly
def mass_error_plot(
    spectrum: Spectrum,
    fragments: FragmentInput,
    tolerance: float = 0.02,
    tolerance_type: ToleranceLike = ToleranceType.PPM,
    peak_selection: PeakSelectionLike = PeakSelection.CLOSEST,
    unit: str = "ppm",
    title: str | None = None,
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

    # Normalise bubble sizes
    max_int = max(intensities) if intensities else 1.0
    sizes = [max(5, 40 * i / max_int) for i in intensities]

    ion_colors = {
        "b": "#1f77b4", "y": "#d62728", "a": "#2ca02c",
        "c": "#9467bd", "z": "#ff7f0e", "x": "#8c564b",
    }
    colors = [ion_colors.get(it, "#aaaaaa") for it in ion_types]

    labels = []
    for m in matches:
        frag = m.fragment
        ion = frag.ion_type.value if hasattr(frag.ion_type, "value") else str(frag.ion_type)
        pos = frag.position if hasattr(frag, "position") else ""
        labels.append(f"{ion}{pos}")

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
                "m/z: %{x:.4f}<br>"
                f"error ({unit}): %{{y:.4f}}<br>"
                "intensity: %{customdata:.2e}<extra></extra>"
            ),
            customdata=intensities,
        )
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.update_layout(
        title=title or "Mass Errors",
        xaxis_title="m/z",
        yaxis_title=f"Error ({unit})",
        showlegend=False,
        **layout_kwargs,
    )
    return fig


@requires_plotly
def facet_plot(
    spectrum: Spectrum,
    fragments: FragmentInput | None = None,
    mirror_spectrum: Spectrum | None = None,
    title: str | None = None,
    tolerance: float = 0.02,
    tolerance_type: ToleranceLike = ToleranceType.PPM,
    peak_selection: PeakSelectionLike = PeakSelection.CLOSEST,
    include_sequence: bool = False,
    unit: str = "ppm",
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
        table = build_annot_plot_table(spectrum, fragments, tolerance, tolerance_type, peak_selection, include_sequence)
    else:
        table = build_plot_table(spectrum)

    import plotly.graph_objects as go

    for _, row in table.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[float(row["mz"]), float(row["mz"])],
                y=[0, float(row["intensity"])],
                mode="lines",
                line={"color": row["color"], "width": float(row["linewidth"])},
                showlegend=False,
                hovertext=row.get("hover", ""),
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )
    fig.update_yaxes(title_text="Intensity", row=1, col=1)

    current_row = 2

    # Panel 2: mass errors
    if fragments is not None:
        from .matching import match_fragments

        matches = match_fragments(spectrum, fragments, tolerance, tolerance_type, peak_selection)
        if matches:
            mzs = [m.peak_mz for m in matches]
            errors = [m.ppm_error if unit == "ppm" else m.da_error for m in matches]
            intensities = [m.peak_intensity for m in matches]
            max_int = max(intensities)
            sizes = [max(5, 30 * i / max_int) for i in intensities]

            ion_colors = {
                "b": "#1f77b4", "y": "#d62728", "a": "#2ca02c",
                "c": "#9467bd", "z": "#ff7f0e", "x": "#8c564b",
            }
            ion_types = [
                m.fragment.ion_type.value if hasattr(m.fragment.ion_type, "value")
                else str(m.fragment.ion_type)
                for m in matches
            ]
            colors = [ion_colors.get(it, "#aaaaaa") for it in ion_types]

            fig.add_trace(
                go.Scatter(
                    x=mzs, y=errors, mode="markers",
                    marker={"size": sizes, "color": colors, "opacity": 0.7},
                    showlegend=False,
                ),
                row=current_row, col=1,
            )
        fig.update_yaxes(title_text=f"Error ({unit})", row=current_row, col=1)
        current_row += 1

    # Panel 3: mirror spectrum
    if mirror_spectrum is not None:
        mirror_table = build_plot_table(mirror_spectrum)
        for _, row in mirror_table.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[float(row["mz"]), float(row["mz"])],
                    y=[0, -float(row["intensity"])],
                    mode="lines",
                    line={"color": row["color"], "width": float(row["linewidth"])},
                    showlegend=False,
                ),
                row=current_row, col=1,
            )
        fig.update_yaxes(title_text="Intensity", row=current_row, col=1)

    fig.update_xaxes(title_text="m/z", row=n_rows, col=1)
    fig.update_layout(
        title=title or "Facet Plot",
        height=300 * n_rows,
        showlegend=False,
        **layout_kwargs,
    )
    return fig


def _sticks(mz: np.ndarray, intensity: np.ndarray) -> tuple[list, list]:
    """Interleave (mz, 0, mz, intensity, None) triples for a stick plot."""
    n = len(mz)
    x = np.empty(n * 3, dtype=np.float64)
    y = np.empty(n * 3, dtype=np.float64)
    x[0::3] = mz
    x[1::3] = mz
    x[2::3] = np.nan
    y[0::3] = 0.0
    y[1::3] = intensity
    y[2::3] = np.nan
    return x.tolist(), y.tolist()
