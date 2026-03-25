"""
Visualization tools for mass spectrometry data.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import plotly.graph_objects as go

import numpy as np
from peptacular.annotation.frag import Fragment

from .core import Spectrum
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
    if show_scores and deconvoluted.score is not None:
        for i, s in enumerate(deconvoluted.score):
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
    fragments: Sequence[Fragment],
    tolerance: float = 0.02,
    tolerance_type: Literal["Da", "ppm"] = "Da",
    title: str | None = None,
    peak_selection: Literal["closest", "largest", "all"] = "closest",
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
    table = build_annot_plot_table(
        spectrum, fragments, tolerance, tolerance_type, peak_selection, include_sequence
    )
    fig = plot_from_table(table, title=title or "Annotated spectrum", **layout_kwargs)
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


