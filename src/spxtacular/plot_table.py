"""
Intermediate plot-table API for spectrum visualisation.

The plot table is a pandas DataFrame with one row per peak.  Each row carries
both the raw data (m/z, intensity, charge, …) and all visual properties
(color, linewidth, label, font settings, …).  Users can freely modify the
DataFrame before passing it to :func:`plot_from_table`.

What the renderer reads
-----------------------
:func:`plot_from_table` draws from ``mz``, ``intensity``, ``series``, ``color``,
``linewidth``, ``opacity``, ``hover``, and the ``label*`` columns *only*.  The
``charge``, ``score``, and ``im`` columns are inputs to the *builders* and are
carried along for reference; editing them after the table is built changes
nothing on the figure.  In particular ``hover`` is baked in by the builder, so
to change a tooltip edit ``hover`` directly rather than the value behind it.

Public API
----------
build_plot_table        -- plain spectrum → DataFrame
build_annot_plot_table  -- spectrum + fragments → DataFrame with ion labels
plot_from_table         -- DataFrame → plotly Figure
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from peptacular.annotation.frag import Fragment

from . import theme
from .core import Spectrum, SpectrumType
from .enums import (
    DEFAULT_FRAGMENT_TOLERANCE,
    DEFAULT_FRAGMENT_TOLERANCE_TYPE,
    PeakSelection,
    PeakSelectionLike,
    ToleranceLike,
)
from .matching import FragmentInput, match_fragments

if TYPE_CHECKING:
    import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Column defaults
#
# Colour itself lives in theme.py -- this module reads from there so a palette
# change lands in one place. See that module for why charge is an ordinal ramp
# and ion type is categorical.
# ---------------------------------------------------------------------------

_LABEL_SIZE_DEFAULT: float = 11.0
_LABEL_FONT_DEFAULT: str = theme._FONT_FAMILY
_LABEL_YSHIFT_DEFAULT: float = 6.0
_LABEL_XANCHOR_DEFAULT: str = "center"
#: Rotation for direct labels, degrees. -90 reads bottom-to-top.
#:
#: Vertical is the spectrum-viewer convention for a reason: a horizontal label
#: occupies its full text width (~50px for "b13^2"), so neighbouring labels
#: collide almost immediately and most have to be dropped. Rotated, each occupies
#: about one line-height (~14px), which is why several times as many peaks can be
#: labelled before anything overlaps.
_LABEL_ANGLE_DEFAULT: float = -90.0
#: Matched peaks. Thin, but heavier than the unmatched context behind them.
_LINEWIDTH_DEFAULT: float = 1.6
#: Unmatched peaks are context: thinner and dimmer so the annotated peaks lead.
_LINEWIDTH_UNMATCHED: float = 1.0
_OPACITY_DEFAULT: float = 1.0
_OPACITY_UNMATCHED: float = 0.55

#: Default cap on directly-drawn labels, highest-intensity first.
#:
#: Labelling every annotated peak is the single worst thing a spectrum plot can
#: do: a deconvoluted 5000-peak spectrum emits 5000 overlapping annotations that
#: render as an unreadable smear along the baseline and cost ~10x the build time.
#: Direct labels work precisely because they are sparing; the rest of the values
#: stay one hover away.
_MAX_LABELS_DEFAULT: int = 60

#: Columns :func:`plot_from_table` requires. Validated up front so a missing
#: column fails immediately with a clear message rather than part-way through
#: rendering, or -- worse -- only on a dataset that happens to have labels.
_REQUIRED_COLUMNS: tuple[str, ...] = (
    "mz",
    "intensity",
    "series",
    "color",
    "linewidth",
    "opacity",
    "hover",
    "label",
    "label_size",
    "label_font",
    "label_color",
    "label_yshift",
    "label_xanchor",
)


#: Diameter of the transparent hover target placed on each peak tip.
#: The guidance is that a mark's hit area must exceed the mark; a 1.6px stick is
#: a pinpoint, so an invisible marker carries the tooltip instead.
_HIT_TARGET_SIZE: float = 22.0


def _scaled_intensity(
    intensity: NDArray[np.float64],
    scale: Literal["absolute", "relative"],
    transform: Literal["sqrt", "log"] | None,
) -> tuple[NDArray[np.float64], str]:
    """Return the intensity to *plot* plus the axis label for it.

    The unscaled values stay in ``intensity_abs`` and drive every tooltip, so
    rescaling only ever changes the axis, never the number the reader is told.
    """
    values = np.asarray(intensity, dtype=np.float64)
    label = "Intensity"

    if scale == "relative":
        peak = float(np.nanmax(values)) if len(values) else 0.0
        if peak > 0:
            values = values / peak * 100.0
        label = "Relative intensity (%)"
    elif scale != "absolute":
        raise ValueError(f"intensity_scale must be 'absolute' or 'relative', got {scale!r}")

    if transform == "sqrt":
        values = np.sqrt(np.clip(values, 0.0, None))
        label = f"√ {label[0].lower()}{label[1:]}"
    elif transform == "log":
        values = np.log10(np.clip(values, 0.0, None) + 1.0)
        label = f"log₁₀ {label[0].lower()}{label[1:]}"
    elif transform is not None:
        raise ValueError(f"intensity_transform must be None, 'sqrt' or 'log', got {transform!r}")

    return values, label


#: Cap on samples drawn for a profile trace. Roughly twice a typical plot width in
#: pixels, which is the most a screen can resolve.
_PROFILE_MAX_POINTS: int = 4000


def _decimate_profile(
    mz: NDArray[np.float64],
    intensity: NDArray[np.float64],
    max_points: int | None = _PROFILE_MAX_POINTS,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Thin a profile trace to ``max_points`` while keeping every peak apex.

    Splits the samples into buckets of equal width and keeps the **minimum and
    maximum** of each. This is the standard waveform-drawing technique, and the
    reason for it is that the obvious alternative is dangerous: taking every Nth
    sample can step straight over the two or three samples that form a peak, so a
    real peak silently disappears from the plot. Min/max keeps each bucket's
    extremes, so an apex survives no matter where it falls.

    Returns the samples unchanged when they already fit.
    """
    n = len(mz)
    if max_points is None or n <= max_points:
        return mz, intensity

    n_buckets = max(1, max_points // 2)
    bounds = np.linspace(0, n, n_buckets + 1).astype(np.int64)

    keep = np.empty(n_buckets * 2, dtype=np.int64)
    for b in range(n_buckets):
        lo, hi = int(bounds[b]), int(bounds[b + 1])
        if hi <= lo:
            keep[2 * b] = keep[2 * b + 1] = lo if lo < n else n - 1
            continue
        seg = intensity[lo:hi]
        keep[2 * b] = lo + int(np.argmin(seg))
        keep[2 * b + 1] = lo + int(np.argmax(seg))

    # Sorted and de-duplicated so the trace stays monotonic in m/z.
    idx = np.unique(keep)
    return mz[idx], intensity[idx]


def _charge_series(charge: int) -> str:
    """Legend label for a charge state, using the library's charge conventions."""
    if charge == -1:
        return "singleton"
    if charge == 0:
        return "decharged"
    return f"z={charge}"


#: Minimum gap between two direct labels, as a fraction of the m/z axis span.
#: Plotly does no collision avoidance for layout annotations, so without this a
#: cluster of matched peaks renders its labels on top of each other regardless of
#: how few there are.
_LABEL_MIN_SEPARATION: float = 0.009


def _cap_labels(
    labels: list[str],
    intensity: NDArray[np.float64],
    max_labels: int | None,
    mz: NDArray[np.float64] | None = None,
) -> list[str]:
    """Thin direct labels down to a readable set; blank the rest.

    Two passes, both needed:

    * **Collision** -- walking the candidates from most to least intense, a label
      is dropped when a stronger one already sits within
      ``_LABEL_MIN_SEPARATION`` of the m/z axis. Intensity order means the peak a
      reader cares about wins the space.
    * **Count** -- at most ``max_labels`` survive.

    The dropped values are not lost; they stay in the hover text and in the plot
    table itself.
    """
    idx = [i for i, text in enumerate(labels) if text]
    if not idx:
        return labels

    idx.sort(key=lambda i: float(intensity[i]), reverse=True)

    if mz is not None and len(mz) > 1:
        span = float(np.nanmax(mz)) - float(np.nanmin(mz))
        min_sep = span * _LABEL_MIN_SEPARATION
        if min_sep > 0:
            placed: list[float] = []
            survivors: list[int] = []
            for i in idx:
                x = float(mz[i])
                if all(abs(x - p) >= min_sep for p in placed):
                    placed.append(x)
                    survivors.append(i)
            idx = survivors

    if max_labels is not None:
        idx = idx[:max_labels]

    keep = set(idx)
    return [text if i in keep else "" for i, text in enumerate(labels)]


def _hover(mz: float, intensity: float, im: float | None = None) -> str:
    base = f"m/z: {mz:.4f}<br>intensity: {intensity:.2e}"
    return base if im is None else f"{base}<br>im: {im:.4f}"


def _im_value(im_col: list[float], im_arr: NDArray[np.float64] | None, i: int) -> float | None:
    """Peak `i`'s ion mobility, or None if no IM array is present or the value is NaN."""
    if im_arr is None:
        return None
    val = float(im_col[i])
    return None if np.isnan(val) else val


# ---------------------------------------------------------------------------
# build_plot_table
# ---------------------------------------------------------------------------


def build_plot_table(
    spectrum: Spectrum,
    show_charges: bool = True,
    show_scores: bool = True,
    max_labels: int | None = _MAX_LABELS_DEFAULT,
    theme_mode: theme.ThemeMode | None = None,
    intensity_scale: Literal["absolute", "relative"] = "relative",
    intensity_transform: Literal["sqrt", "log"] | None = None,
) -> pd.DataFrame:
    """Build a plot table from a plain spectrum (no fragment annotations).

    There is no ``texture`` option here: the texture channel distinguishes ion
    *series*, and a plain spectrum has none. See :func:`build_annot_plot_table`.

    Parameters
    ----------
    spectrum:
        Source spectrum.
    show_charges:
        When ``True`` (default) and charge data is present, each charge state
        gets its own colour from the ordinal charge ramp; the ``series`` column
        is set to ``"z=N"`` or ``"singleton"``.  When ``False``, every peak
        takes the ramp's 1+ step (:func:`spxtacular.theme.charge_color` at
        ``charge=1``) and the ``series`` column is ``"peaks"``.
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

    plotted, intensity_label = _scaled_intensity(intensity, intensity_scale, intensity_transform)

    charge_arr = spectrum.charge
    score_arr = spectrum.iso_score
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

    # Colours and series. Charge is ordinal, so it takes a single-hue ramp keyed
    # directly on the charge value -- not a cycle over encounter order, which
    # made the colours depend on which charges happened to be present and
    # repeated itself after ten distinct states.
    if has_charge and charge_arr is not None:
        colors = [theme.charge_color(int(c), theme_mode) for c in charge_arr]
        series = [_charge_series(int(c)) for c in charge_arr]
    else:
        colors = [theme.charge_color(1, theme_mode)] * n
        series = ["peaks"] * n

    # Profile data is a continuous trace, so a per-sample label is meaningless --
    # there is no "peak" at a sample, only a point on a curve.
    is_profile = spectrum.spectrum_type == SpectrumType.PROFILE

    # Labels
    if show_scores and score_arr is not None and not is_profile:
        labels = [f"{float(s):.2f}" if float(s) > 0.0 else "" for s in score_arr]
        labels = _cap_labels(labels, intensity, max_labels, mz)
    else:
        labels = [""] * n

    hovers = [
        _hover(
            float(mz[i]),
            float(intensity[i]),
            _im_value(im_col, im_arr, i),
        )
        for i in range(n)
    ]

    table = pd.DataFrame(
        {
            "mz": mz.astype(np.float64),
            "intensity": plotted,
            "intensity_abs": intensity.astype(np.float64),
            "charge": charge_col,
            "score": score_col,
            "im": im_col,
            "color": colors,
            "linewidth": [_LINEWIDTH_DEFAULT] * n,
            "opacity": [_OPACITY_DEFAULT] * n,
            "dash": ["solid"] * n,
            "series": series,
            "label": labels,
            "label_size": [_LABEL_SIZE_DEFAULT] * n,
            "label_font": [_LABEL_FONT_DEFAULT] * n,
            "label_color": [theme.text_color("secondary", theme_mode)] * n,
            "label_yshift": [_LABEL_YSHIFT_DEFAULT] * n,
            "label_xanchor": [_LABEL_XANCHOR_DEFAULT] * n,
            "label_angle": [_LABEL_ANGLE_DEFAULT] * n,
            "hover": hovers,
        }
    )
    # Carried on the frame so the renderer can title the axis correctly
    # without re-deriving what scaling was applied.
    table.attrs["intensity_label"] = intensity_label
    table.attrs["render"] = "profile" if is_profile else "sticks"
    return table


# ---------------------------------------------------------------------------
# build_annot_plot_table
# ---------------------------------------------------------------------------


def _ion_priority(ion_type: str) -> int:
    """Sort key giving each ion series a fixed rank, unknown series last."""
    key = str(ion_type).lower()
    return theme._ION_SLOTS.index(key) if key in theme._ION_SLOTS else len(theme._ION_SLOTS)


def _fragment_label(fragment: Fragment, include_sequence: bool) -> str:
    import paftacular as pft

    return pft.to_mzpaf(fragment, include_annotation=include_sequence).serialize()


def build_annot_plot_table(
    spectrum: Spectrum,
    fragments: FragmentInput,
    tolerance: float = DEFAULT_FRAGMENT_TOLERANCE,
    tolerance_type: ToleranceLike = DEFAULT_FRAGMENT_TOLERANCE_TYPE,
    peak_selection: PeakSelectionLike = PeakSelection.CLOSEST,
    include_sequence: bool = False,
    max_labels: int | None = _MAX_LABELS_DEFAULT,
    theme_mode: theme.ThemeMode | None = None,
    intensity_scale: Literal["absolute", "relative"] = "relative",
    intensity_transform: Literal["sqrt", "log"] | None = None,
    texture: bool = False,
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
    tolerance:
        Matching tolerance.
    tolerance_type:
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
    matches = match_fragments(spectrum, fragments, tolerance, tolerance_type, peak_selection)

    # Group matches by peak index
    peak_frags: dict[int, list[Fragment]] = {}
    for m in matches:
        peak_frags.setdefault(m.peak_index, []).append(m.fragment)

    mz = spectrum.mz
    intensity = spectrum.intensity
    n = len(mz)

    plotted, intensity_label = _scaled_intensity(intensity, intensity_scale, intensity_transform)

    charge_arr = spectrum.charge
    score_arr = spectrum.iso_score
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
    linewidths: list[float] = []
    opacities: list[float] = []
    dashes: list[str] = []

    unmatched = theme.unmatched_color(theme_mode)

    for i in range(n):
        mz_val = float(mz[i])
        int_val = float(intensity[i])
        frags = peak_frags.get(i)
        im_val = _im_value(im_col, im_arr, i)
        if frags:
            # When one peak matches several ions, pick the colour deterministically
            # by the fixed series order rather than taking whichever fragment the
            # caller happened to list first -- otherwise reordering the input
            # fragment list silently repaints the plot.
            ion_type = min((str(f.ion_type) for f in frags), key=_ion_priority)
            label_text = "<br>".join(_fragment_label(f, include_sequence) for f in frags)
            hover_text = _hover(mz_val, int_val, im_val) + f"<br>{label_text}"
            colors.append(theme.ion_color(ion_type, theme_mode))
            series_list.append(ion_type)
            labels.append(label_text)
            hovers.append(hover_text)
            linewidths.append(_LINEWIDTH_DEFAULT)
            opacities.append(_OPACITY_DEFAULT)
            dashes.append(theme.ion_dash(ion_type) if texture else "solid")
        else:
            colors.append(unmatched)
            series_list.append("unmatched")
            labels.append("")
            hovers.append(_hover(mz_val, int_val, im_val))
            linewidths.append(_LINEWIDTH_UNMATCHED)
            opacities.append(_OPACITY_UNMATCHED)
            dashes.append("solid")

    labels = _cap_labels(labels, intensity, max_labels, mz)

    table = pd.DataFrame(
        {
            "mz": mz.astype(np.float64),
            "intensity": plotted,
            "intensity_abs": intensity.astype(np.float64),
            "charge": charge_col,
            "score": score_col,
            "im": im_col,
            "color": colors,
            "linewidth": linewidths,
            "opacity": opacities,
            "dash": dashes,
            "series": series_list,
            "label": labels,
            "label_size": [_LABEL_SIZE_DEFAULT] * n,
            "label_font": [_LABEL_FONT_DEFAULT] * n,
            "label_color": [theme.text_color("secondary", theme_mode)] * n,
            "label_yshift": [_LABEL_YSHIFT_DEFAULT] * n,
            "label_xanchor": [_LABEL_XANCHOR_DEFAULT] * n,
            "label_angle": [_LABEL_ANGLE_DEFAULT] * n,
            "hover": hovers,
        }
    )
    # Carried on the frame so the renderer can title the axis correctly
    # without re-deriving what scaling was applied.
    table.attrs["intensity_label"] = intensity_label
    # Fragment matching is a centroid operation, so this table always draws sticks.
    table.attrs["render"] = "sticks"
    return table


# ---------------------------------------------------------------------------
# plot_from_table
# ---------------------------------------------------------------------------


def _sticks(
    mz: NDArray[np.float64],
    intensity: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Interleave (mz, 0, mz, NaN) triples for a stick plot.

    Returns numpy arrays rather than lists. Plotly validates a Python list
    element by element -- around 2,000 calls per figure for a modest spectrum,
    which dominated figure construction -- but takes a numpy array through a
    fast path. NaN reads as a line break exactly as ``None`` does.
    """
    n = len(mz)
    x = np.empty(n * 3, dtype=np.float64)
    y = np.empty(n * 3, dtype=np.float64)
    x[0::3] = mz
    x[1::3] = mz
    x[2::3] = np.nan
    y[0::3] = 0.0
    y[1::3] = intensity
    y[2::3] = np.nan
    return x, y


def table_view(
    table: pd.DataFrame,
    max_rows: int | None = None,
    annotated_only: bool = False,
) -> str:
    """Render a plot table as an HTML table -- the accessible companion to the figure.

    A tooltip enhances, it never gates: every value the figure shows should be
    reachable without hovering. That matters here because label capping
    deliberately drops labels off the plot, and a hover is unusable for keyboard
    and screen-reader users.

    Parameters
    ----------
    table:
        A table from :func:`build_plot_table` or :func:`build_annot_plot_table`.
    max_rows:
        Keep only the this many most intense peaks. ``None`` (default) keeps all.
    annotated_only:
        Keep only peaks carrying a label. Useful beside an annotated spectrum,
        where the unmatched peaks are context rather than results.

    Returns
    -------
    An HTML ``<table>`` as a string.
    """
    from html import escape

    view = table
    if annotated_only and "label" in view.columns:
        view = view[view["label"].notna() & (view["label"] != "")]
    if max_rows is not None:
        sort_col = "intensity_abs" if "intensity_abs" in view.columns else "intensity"
        view = view.nlargest(max_rows, sort_col)
    view = view.sort_values("mz")

    columns: list[tuple[str, str]] = [("mz", "m/z"), ("intensity_abs", "Intensity")]
    if "intensity_abs" not in view.columns:
        columns = [("mz", "m/z"), ("intensity", "Intensity")]
    for col, heading in (("charge", "z"), ("score", "Score"), ("im", "Ion mobility")):
        if col in view.columns and view[col].notna().any():
            columns.append((col, heading))
    if "label" in view.columns and (view["label"] != "").any():
        columns.append(("label", "Annotation"))

    def _fmt(col: str, value) -> str:
        if pd.isna(value):
            return ""
        if col == "mz":
            return f"{float(value):.4f}"
        if col in ("intensity", "intensity_abs"):
            return f"{float(value):.4g}"
        if col in ("score", "im"):
            return f"{float(value):.3f}"
        if col == "charge":
            return str(int(value))
        # Labels are data and may contain markup; escape, and turn the <br>
        # separators the plot uses into commas.
        return escape(str(value).replace("<br>", ", "))

    head = "".join(f"<th scope='col'>{escape(h)}</th>" for _, h in columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{_fmt(col, row[col])}</td>" for col, _ in columns) + "</tr>"
        for _, row in view.iterrows()
    )
    return f"<table><caption>Peak list</caption><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


def _plot_profile_trace(
    table: pd.DataFrame,
    title: str | None,
    theme_mode: theme.ThemeMode | None,
    max_points: int | None,
    **layout_kwargs,
) -> go.Figure:
    """Draw a profile spectrum as a continuous trace rather than sticks.

    Profile data samples a continuous signal, so the peak *shape* is the
    information -- which is exactly what a stick plot throws away, since it draws
    every sample as its own bar from the baseline. A connected line keeps the
    shape, and costs a third of the coordinates.
    """
    import plotly.graph_objects as go

    mz = table["mz"].to_numpy(dtype=np.float64)
    intensity = table["intensity"].to_numpy(dtype=np.float64)
    mz, intensity = _decimate_profile(mz, intensity, max_points)

    color = str(table["color"].iloc[0]) if len(table) else theme.charge_color(1, theme_mode)
    fig = go.Figure(
        go.Scatter(
            x=mz,
            y=intensity,
            mode="lines",
            line={"color": color, "width": 1.4},
            # A wash, not a saturated block: the fill says "area under the trace"
            # without competing with the trace itself.
            fill="tozeroy",
            fillcolor=_rgba(color, 0.10),
            name="profile",
            hovertemplate="m/z: %{x:.4f}<br>intensity: %{y:.4g}<extra></extra>",
        )
    )
    fig.update_layout(
        template=theme.template(theme_mode),
        title=title or "Profile spectrum",
        xaxis_title="m/z",
        yaxis_title=table.attrs.get("intensity_label", "Intensity"),
        showlegend=False,
        **layout_kwargs,
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    """``#rrggbb`` -> ``rgba(r, g, b, alpha)`` for plotly fills."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def plot_from_table(
    table: pd.DataFrame,
    title: str | None = None,
    theme_mode: theme.ThemeMode | None = None,
    render: Literal["sticks", "profile"] | None = None,
    max_points: int | None = _PROFILE_MAX_POINTS,
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

    missing = [c for c in _REQUIRED_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"plot table is missing required column(s): {', '.join(missing)}")

    mode = render if render is not None else table.attrs.get("render", "sticks")
    if mode not in ("sticks", "profile"):
        raise ValueError(f"render must be 'sticks' or 'profile', got {mode!r}")

    if mode == "profile":
        return _plot_profile_trace(table, title, theme_mode, max_points, **layout_kwargs)

    traces: list[go.Scatter] = []

    # One trace per (series, color) — preserves legend grouping and colour.
    #
    # dropna=False matters: groupby drops NA keys by default, so a peak whose
    # series or colour came back NA (easy to produce with merge/reindex/concat on
    # a user-edited table) would vanish from the figure with no error at all.
    for (series, color), group in table.groupby(["series", "color"], sort=False, dropna=False):  # type: ignore
        if pd.isna(color):
            color = theme.unmatched_color(theme_mode)
        if pd.isna(series):
            series = "unlabelled"
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
        line: dict = {"color": str(color), "width": linewidth}
        if "dash" in group.columns and str(first["dash"]) != "solid":
            line["dash"] = str(first["dash"])

        traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=str(series),
                line=line,
                opacity=opacity,
                # The hit layer below carries the tooltip; letting the hairline
                # itself answer hovers would mean the reader has to land on it.
                hoverinfo="skip",
            )
        )

    # Transparent hover targets on the peak tips. One trace for the whole
    # spectrum, invisible, sized well beyond the sticks so the pointer only has
    # to be near a peak rather than on it.
    traces.append(
        go.Scatter(
            x=table["mz"].to_numpy(dtype=np.float64),
            y=table["intensity"].to_numpy(dtype=np.float64),
            mode="markers",
            marker={"size": _HIT_TARGET_SIZE, "color": "rgba(0,0,0,0)"},
            customdata=table["hover"].tolist(),
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=False,
            name="",
        )
    )

    # Annotations for labelled peaks. notna() as well as != "" — an NA label is
    # not empty, and str(NaN) renders the literal text "nan" onto the plot.
    annotations = []
    label_mask = table["label"].notna() & (table["label"] != "")
    has_angle = "label_angle" in table.columns
    for _, row in table[label_mask].iterrows():
        # Fall back for tables built before label_angle existed, or hand-made ones.
        angle = float(row["label_angle"]) if has_angle and pd.notna(row["label_angle"]) else 0.0
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
                # Rotated labels grow upward from the peak tip, so anchor their
                # bottom edge there; an unrotated one keeps plotly's centring.
                yanchor="bottom" if angle else "middle",
                textangle=angle,
            )
        )

    fig = go.Figure(traces)
    unique_series = table["series"].nunique(dropna=False)
    fig.update_layout(
        template=theme.template(theme_mode),
        title=title or "Spectrum",
        xaxis_title="m/z",
        yaxis_title=table.attrs.get("intensity_label", "Intensity"),
        # A single series needs no legend box: one colour, and the title already
        # names what is plotted.
        showlegend=unique_series > 1,
        annotations=annotations,
        **layout_kwargs,
    )
    # Anchor the baseline so sticks read as growing from zero, and keep the
    # y-axis off the data. Vertical labels grow upward from the peak tip, so the
    # tallest peak needs headroom or its label is clipped by the plot edge.
    fig.update_yaxes(rangemode="tozero")
    if annotations and any(a.get("textangle") for a in annotations):
        overall = float(table["intensity"].max()) if len(table) else 0.0
        # Only the *labelled* peaks need room above them. Padding from the overall
        # maximum wastes the top of the plot whenever the tallest peak is unlabelled.
        labelled = table.loc[label_mask, "intensity"]
        tallest_labelled = float(labelled.max()) if len(labelled) else 0.0
        top = max(overall, tallest_labelled * 1.22)
        if top > 0:
            fig.update_yaxes(range=[0, top])
    return fig
