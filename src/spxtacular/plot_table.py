"""
Intermediate plot-table API for spectrum visualisation.

The plot table is a pandas DataFrame with one row per peak.  Each row carries
both the raw data (m/z, intensity, charge, …) and all visual properties
(color, linewidth, label, font settings, …).  Users can freely modify the
DataFrame before passing it to :func:`plot_from_table`.

Public API
----------
build_plot_table        -- plain spectrum → DataFrame
build_annot_plot_table  -- spectrum + fragments → DataFrame with ion labels
plot_from_table         -- DataFrame → plotly Figure
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from peptacular.annotation.frag import Fragment

from .core import Spectrum
from .matching import match_fragments

if TYPE_CHECKING:
    import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Colour constants (kept in sync with visualization.py)
# ---------------------------------------------------------------------------

_ION_COLORS: dict[str, str] = {
    "b": "#1f77b4",
    "y": "#d62728",
    "a": "#2ca02c",
    "c": "#9467bd",
    "z": "#ff7f0e",
    "x": "#8c564b",
}
_DEFAULT_ION_COLOR = "#aaaaaa"

# Qualitative colour cycle for charge states (plotly G10 palette)
_CHARGE_COLORS: list[str] = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
_SINGLETON_COLOR = "#aaaaaa"

# ---------------------------------------------------------------------------
# Column defaults
# ---------------------------------------------------------------------------

_LABEL_SIZE_DEFAULT: float = 10.0
_LABEL_FONT_DEFAULT: str = "Arial"
_LABEL_COLOR_DEFAULT: str = "#333333"
_LABEL_YSHIFT_DEFAULT: float = 6.0
_LABEL_XANCHOR_DEFAULT: str = "center"
_LINEWIDTH_DEFAULT: float = 1.0
_OPACITY_DEFAULT: float = 1.0


def _hover(mz: float, intensity: float) -> str:
    return f"m/z: {mz:.4f}<br>intensity: {intensity:.2e}"


# ---------------------------------------------------------------------------
# build_plot_table
# ---------------------------------------------------------------------------


def build_plot_table(
    spectrum: Spectrum,
    show_charges: bool = True,
    show_scores: bool = True,
) -> pd.DataFrame:
    """Build a plot table from a plain spectrum (no fragment annotations).

    Parameters
    ----------
    spectrum:
        Source spectrum.
    show_charges:
        When ``True`` (default) and charge data is present, each charge state
        gets its own colour; the ``series`` column is set to ``"z=N"`` or
        ``"singleton"``.  When ``False``, all peaks use ``"steelblue"``.
    show_scores:
        When ``True`` (default) and score data is present, peaks with
        ``score > 0`` are labelled with their score value.

    Returns
    -------
    pd.DataFrame with columns:
    ``mz``, ``intensity``, ``charge``, ``score``, ``im``,
    ``color``, ``linewidth``, ``opacity``, ``series``,
    ``label``, ``label_size``, ``label_font``, ``label_color``,
    ``label_yshift``, ``label_xanchor``, ``hover``.
    """
    mz = spectrum.mz
    intensity = spectrum.intensity
    n = len(mz)

    charge_arr = spectrum.charge
    score_arr = spectrum.score
    im_arr = spectrum.im

    has_charge = show_charges and charge_arr is not None

    # Build charge/score/im columns
    if charge_arr is not None:
        charge_col = pd.array(charge_arr.tolist(), dtype="Int64")
    else:
        charge_col = pd.array([pd.NA] * n, dtype="Int64")

    if score_arr is not None:
        score_col = score_arr.astype(np.float64).tolist()
    else:
        score_col = [float("nan")] * n

    if im_arr is not None:
        im_col = im_arr.astype(np.float64).tolist()
    else:
        im_col = [float("nan")] * n

    # Colours and series
    if has_charge and charge_arr is not None:
        unique_charges = sorted(set(int(c) for c in charge_arr))
        charge_to_color: dict[int, str] = {}
        color_idx = 0
        for z in unique_charges:
            if z == -1:
                charge_to_color[z] = _SINGLETON_COLOR
            else:
                charge_to_color[z] = _CHARGE_COLORS[color_idx % len(_CHARGE_COLORS)]
                color_idx += 1
        colors = [charge_to_color[int(c)] for c in charge_arr]
        series = ["singleton" if int(c) == -1 else f"z={int(c)}" for c in charge_arr]
    else:
        colors = ["steelblue"] * n
        series = ["peaks"] * n

    # Labels
    if show_scores and score_arr is not None:
        labels = [f"{float(s):.2f}" if float(s) > 0.0 else "" for s in score_arr]
    else:
        labels = [""] * n

    hovers = [_hover(float(mz[i]), float(intensity[i])) for i in range(n)]

    return pd.DataFrame(
        {
            "mz": mz.astype(np.float64),
            "intensity": intensity.astype(np.float64),
            "charge": charge_col,
            "score": score_col,
            "im": im_col,
            "color": colors,
            "linewidth": [_LINEWIDTH_DEFAULT] * n,
            "opacity": [_OPACITY_DEFAULT] * n,
            "series": series,
            "label": labels,
            "label_size": [_LABEL_SIZE_DEFAULT] * n,
            "label_font": [_LABEL_FONT_DEFAULT] * n,
            "label_color": [_LABEL_COLOR_DEFAULT] * n,
            "label_yshift": [_LABEL_YSHIFT_DEFAULT] * n,
            "label_xanchor": [_LABEL_XANCHOR_DEFAULT] * n,
            "hover": hovers,
        }
    )


# ---------------------------------------------------------------------------
# build_annot_plot_table
# ---------------------------------------------------------------------------


def _fragment_label(fragment: Fragment, include_sequence: bool) -> str:
    import paftacular as pft

    return pft.to_mzpaf(fragment, include_annotation=include_sequence).serialize()


def build_annot_plot_table(
    spectrum: Spectrum,
    fragments: Sequence[Fragment],
    mz_tol: float = 0.02,
    mz_tol_type: Literal["Da", "ppm"] = "Da",
    peak_selection: Literal["closest", "largest", "all"] = "closest",
    include_sequence: bool = False,
) -> pd.DataFrame:
    """Build a plot table with fragment-ion annotations.

    Matched peaks are coloured by ion series (b=blue, y=red, …) and labelled.
    Unmatched peaks are grey with no label.

    Parameters
    ----------
    spectrum:
        Centroid spectrum to annotate.
    fragments:
        Fragment objects from peptacular.
    mz_tol:
        Matching tolerance.
    mz_tol_type:
        ``"Da"`` or ``"ppm"``.
    peak_selection:
        How to resolve multiple peaks per fragment — ``"closest"``,
        ``"largest"``, or ``"all"``.
    include_sequence:
        Embed the residue sequence in each label (e.g. ``b3{PEP}``).

    Returns
    -------
    pd.DataFrame with the same columns as :func:`build_plot_table`.
    """
    matches = match_fragments(spectrum, fragments, mz_tol, mz_tol_type, peak_selection)

    # Group matches by peak index
    peak_frags: dict[int, list[Fragment]] = {}
    for peak_idx, frag in matches:
        peak_frags.setdefault(peak_idx, []).append(frag)

    mz = spectrum.mz
    intensity = spectrum.intensity
    n = len(mz)

    charge_arr = spectrum.charge
    score_arr = spectrum.score
    im_arr = spectrum.im

    if charge_arr is not None:
        charge_col = pd.array(charge_arr.tolist(), dtype="Int64")
    else:
        charge_col = pd.array([pd.NA] * n, dtype="Int64")

    if score_arr is not None:
        score_col = score_arr.astype(np.float64).tolist()
    else:
        score_col = [float("nan")] * n

    if im_arr is not None:
        im_col = im_arr.astype(np.float64).tolist()
    else:
        im_col = [float("nan")] * n

    colors: list[str] = []
    series_list: list[str] = []
    labels: list[str] = []
    hovers: list[str] = []

    for i in range(n):
        mz_val = float(mz[i])
        int_val = float(intensity[i])
        frags = peak_frags.get(i)
        if frags:
            ion_type = str(frags[0].ion_type)
            color = _ION_COLORS.get(ion_type, _DEFAULT_ION_COLOR)
            label_text = "<br>".join(_fragment_label(f, include_sequence) for f in frags)
            hover_text = f"m/z: {mz_val:.4f}<br>intensity: {int_val:.2e}<br>{label_text}"
            colors.append(color)
            series_list.append(ion_type)
            labels.append(label_text)
            hovers.append(hover_text)
        else:
            colors.append("#cccccc")
            series_list.append("unmatched")
            labels.append("")
            hovers.append(_hover(mz_val, int_val))

    return pd.DataFrame(
        {
            "mz": mz.astype(np.float64),
            "intensity": intensity.astype(np.float64),
            "charge": charge_col,
            "score": score_col,
            "im": im_col,
            "color": colors,
            "linewidth": [_LINEWIDTH_DEFAULT] * n,
            "opacity": [_OPACITY_DEFAULT] * n,
            "series": series_list,
            "label": labels,
            "label_size": [_LABEL_SIZE_DEFAULT] * n,
            "label_font": [_LABEL_FONT_DEFAULT] * n,
            "label_color": [_LABEL_COLOR_DEFAULT] * n,
            "label_yshift": [_LABEL_YSHIFT_DEFAULT] * n,
            "label_xanchor": [_LABEL_XANCHOR_DEFAULT] * n,
            "hover": hovers,
        }
    )


# ---------------------------------------------------------------------------
# plot_from_table
# ---------------------------------------------------------------------------


def _sticks(
    mz: NDArray[np.float64],
    intensity: NDArray[np.float64],
) -> tuple[list, list]:
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


def plot_from_table(
    table: pd.DataFrame,
    title: str | None = None,
    **layout_kwargs,
) -> go.Figure:
    """Render a stick plot from a plot table DataFrame.

    One ``go.Scatter`` trace is created per unique ``(series, color)`` group.
    All peaks in a group share the linewidth and opacity of the *first* row
    in that group.  Annotations are added for every row where ``label != ""``.

    Parameters
    ----------
    table:
        DataFrame produced by :func:`build_plot_table` or
        :func:`build_annot_plot_table`, or a user-modified copy thereof.
        Required columns: ``mz``, ``intensity``, ``series``, ``color``,
        ``linewidth``, ``opacity``, ``hover``, ``label``, ``label_size``,
        ``label_font``, ``label_color``, ``label_yshift``, ``label_xanchor``.
    title:
        Plot title.
    **layout_kwargs:
        Forwarded to ``fig.update_layout``.

    Returns
    -------
    plotly ``Figure``.
    """
    import plotly.graph_objects as go

    traces: list[go.Scatter] = []

    # One trace per (series, color) — preserves legend grouping and colour
    for (series, color), group in table.groupby(["series", "color"], sort=False):
        mz_arr = group["mz"].to_numpy(dtype=np.float64)
        int_arr = group["intensity"].to_numpy(dtype=np.float64)
        hover_arr = group["hover"].tolist()

        xs, ys = _sticks(mz_arr, int_arr)

        # Hover: repeat each hover text twice (base + tip) then empty for None
        hover_data: list[str] = []
        for h in hover_arr:
            hover_data += [h, h, ""]

        first = group.iloc[0]
        linewidth = float(first["linewidth"])
        opacity = float(first["opacity"])

        traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=str(series),
                line={"color": str(color), "width": linewidth},
                opacity=opacity,
                customdata=hover_data,
                hovertemplate="%{customdata}<extra></extra>",
            )
        )

    # Annotations for labelled peaks
    annotations = []
    label_mask = table["label"] != ""
    for _, row in table[label_mask].iterrows():
        annotations.append(
            dict(
                x=float(row["mz"]),
                y=float(row["intensity"]),
                text=str(row["label"]),
                showarrow=False,
                yshift=float(row["label_yshift"]),
                font={
                    "size": float(row["label_size"]),
                    "family": str(row["label_font"]),
                    "color": str(row["label_color"]),
                },
                xanchor=str(row["label_xanchor"]),
            )
        )

    fig = go.Figure(traces)
    unique_series = table["series"].nunique()
    fig.update_layout(
        title=title or "Spectrum",
        xaxis_title="m/z",
        yaxis_title="Intensity",
        showlegend=unique_series > 1,
        annotations=annotations,
        **layout_kwargs,
    )
    return fig
