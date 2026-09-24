"""
Intermediate plot-table API for spectrum visualisation.

The plot table is a pandas DataFrame with one row per peak. Each row carries
peak data and all visual properties. ``intensity`` is the plotted value, while
``intensity_abs`` preserves the unscaled value used by hover and table output.
Users can modify the DataFrame before passing it to :func:`plot_from_table`.

What the renderer reads
-----------------------
:func:`plot_from_table` draws from ``mz``, ``intensity``, ``series``, ``color``,
``linewidth`` (relative to the default 1.6, scaled to the figure style's stick
width), ``opacity``, optional ``dash``, ``hover``, and the ``label*`` columns.
The ``charge``, ``score``, ``im``, and ``intensity_abs`` columns are carried for
reference and accessible output. Editing them after the table is
built does not change the figure. In particular, ``hover`` is baked in by the
builder, so change that column directly to change a tooltip. The renderer also
uses ``table.attrs["intensity_label"]`` for the y-axis title and
``table.attrs["render"]`` to choose sticks or a profile trace. When
``table.attrs["label_format"]`` is ``"mzpaf"`` the labels are parsed as mzPAF
and drawn with sub- and superscripts (``y7^2`` becomes y₇²⁺).

Public API
----------
build_plot_table        -- plain spectrum → DataFrame
build_annot_plot_table  -- spectrum + fragments → DataFrame with ion labels
plot_from_table         -- DataFrame → plotly or matplotlib Figure
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from peptacular.annotation.frag import Fragment

from . import theme
from ._text import RichText, best_label
from .core import Spectrum, SpectrumType
from .enums import (
    DEFAULT_FRAGMENT_TOLERANCE,
    DEFAULT_FRAGMENT_TOLERANCE_TYPE,
    PeakSelection,
    PeakSelectionLike,
    ToleranceLike,
)
from .errors import SpxtacularError
from .figspec import (
    Axis,
    Backend,
    Cell,
    HitLayer,
    LabelSet,
    Line,
    Mark,
    Panel,
    Sticks,
    check_backend,
    finish,
    new_spec,
)
from .matching import FragmentInput, match_fragments
from .style import FigureStyle, SizeLike, StyleName, resolve_style

# ---------------------------------------------------------------------------
# Column defaults
#
# Colour itself lives in theme.py -- this module reads from there so a palette
# change lands in one place. See that module for why charge is an ordinal ramp
# and ion type is categorical.
# ---------------------------------------------------------------------------

#: Label size in pt. NaN means "the figure style's label size".
_LABEL_SIZE_DEFAULT: float = float("nan")
#: Rotation for direct labels, degrees; 0 is horizontal, -90 reads bottom-to-top.
#:
#: Horizontal is the default because labels are now placed with collision
#: avoidance: a label that does not fit straight above its peak moves beside it
#: with a leader line instead of being dropped, so the horizontal layout's width
#: cost no longer throws labels away, and horizontal text is what journals and
#: readers expect.
_LABEL_ANGLE_DEFAULT: float = 0.0
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
    "label_color",
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
        raise SpxtacularError(f"intensity_scale must be 'absolute' or 'relative', got {scale!r}")

    if transform == "sqrt":
        values = np.sqrt(np.clip(values, 0.0, None))
        label = f"√ {label[0].lower()}{label[1:]}"
    elif transform == "log":
        values = np.log10(np.clip(values, 0.0, None) + 1.0)
        label = f"log₁₀ {label[0].lower()}{label[1:]}"
    elif transform is not None:
        raise SpxtacularError(f"intensity_transform must be None, 'sqrt' or 'log', got {transform!r}")

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


def _cap_labels(
    labels: list[str],
    intensity: NDArray[np.float64],
    max_labels: int | None,
) -> list[str]:
    """Keep the ``max_labels`` most intense labels; blank the rest.

    Collisions are not handled here: the figure layout places every surviving
    label clear of the others and of the peaks, moving it aside with a leader
    line when the spot above its peak is taken, and drops only what cannot fit.
    The dropped values stay in the hover text and in the plot table.
    """
    if max_labels is None:
        return labels
    idx = [i for i, text in enumerate(labels) if text]
    if len(idx) <= max_labels:
        return labels
    idx.sort(key=lambda i: float(intensity[i]), reverse=True)
    keep = set(idx[:max_labels])
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
    *,
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
    max_labels:
        Maximum number of direct labels, strongest peaks first. ``None``
        draws every label that the layout can fit without an overlap.
    theme_mode:
        ``"light"`` or ``"dark"``. ``None`` uses the global plot theme.
    intensity_scale:
        ``"relative"`` scales the base peak to 100. ``"absolute"`` preserves
        raw values in the plotted intensity column.
    intensity_transform:
        Optional ``"sqrt"`` or ``"log"`` display transform.

    Returns
    -------
    pd.DataFrame with columns:
    ``mz``, ``intensity``, ``charge``, ``score``, ``im``,
    ``intensity_abs``, ``color``, ``linewidth``, ``opacity``, ``dash``,
    ``series``, ``label``, ``label_size`` (pt; NaN uses the figure style),
    ``label_color``, ``label_angle`` (degrees, 0 horizontal), ``hover``.
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
        labels = _cap_labels(labels, intensity, max_labels)
    else:
        labels = [""] * n

    label_colors = [theme.text_color("secondary", theme_mode)] * n

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
            "label_color": label_colors,
            "label_angle": [_LABEL_ANGLE_DEFAULT] * n,
            "hover": hovers,
        }
    )
    # Carried on the frame so the renderer can title the axis correctly
    # without re-deriving what scaling was applied.
    table.attrs["intensity_label"] = intensity_label
    table.attrs["render"] = "profile" if is_profile else "sticks"
    table.attrs["label_format"] = "text"
    table.attrs["intensity_scale"] = intensity_scale
    return table


# ---------------------------------------------------------------------------
# build_annot_plot_table
# ---------------------------------------------------------------------------


def _ion_priority(ion_type: str) -> int:
    """Sort key giving each ion series a fixed rank, unknown series last."""
    key = str(ion_type).lower()
    return theme._ION_SLOTS.index(key) if key in theme._ION_SLOTS else len(theme._ION_SLOTS)


def _fragment_label(fragment: Fragment, include_sequence: bool) -> str:
    """Return the fragment's mzPAF label (no mass error; signed charge for negative ions)."""
    import paftacular as pft

    return pft.to_mzpaf(fragment, include_sequence=include_sequence).serialize()


def build_annot_plot_table(
    spectrum: Spectrum,
    fragments: FragmentInput,
    *,
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
    max_labels:
        Maximum number of direct labels, strongest peaks first. ``None``
        draws every label that the layout can fit without an overlap.
    theme_mode:
        ``"light"`` or ``"dark"``. ``None`` uses the global plot theme.
    intensity_scale:
        ``"relative"`` scales the base peak to 100. ``"absolute"`` preserves
        raw values in the plotted intensity column.
    intensity_transform:
        Optional ``"sqrt"`` or ``"log"`` display transform.
    texture:
        Give each matched ion series a distinct dash pattern.

    Returns
    -------
    pd.DataFrame with the same columns as :func:`build_plot_table`.
    """
    matches = match_fragments(
        spectrum, fragments, tolerance=tolerance, tolerance_type=tolerance_type, peak_selection=peak_selection
    )

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

    labels = _cap_labels(labels, intensity, max_labels)
    label_colors = [
        theme.label_color(c, theme_mode) if lab else theme.text_color("secondary", theme_mode)
        for c, lab in zip(colors, labels, strict=True)
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
            "linewidth": linewidths,
            "opacity": opacities,
            "dash": dashes,
            "series": series_list,
            "label": labels,
            "label_size": [_LABEL_SIZE_DEFAULT] * n,
            "label_color": label_colors,
            "label_angle": [_LABEL_ANGLE_DEFAULT] * n,
            "hover": hovers,
        }
    )
    # Carried on the frame so the renderer can title the axis correctly
    # without re-deriving what scaling was applied.
    table.attrs["intensity_label"] = intensity_label
    # Fragment matching is a centroid operation, so this table always draws sticks.
    table.attrs["render"] = "sticks"
    # Labels are mzPAF; the renderer turns them into sub/superscripted ion names
    # and ranks them (a plain b/y ion outranks a neutral loss or an isotope).
    table.attrs["label_format"] = "mzpaf"
    table.attrs["intensity_scale"] = intensity_scale
    return table


# ---------------------------------------------------------------------------
# plot_from_table
# ---------------------------------------------------------------------------


def table_view(
    table: pd.DataFrame,
    *,
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
        Keep only this many most intense peaks. ``None`` (default) keeps all.
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


def _is_blank(value: object) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA or value == ""


def intensity_axis(
    label: str,
    *,
    relative: bool,
    base_peak: float | None,
    absolute_axis: bool,
    mirrored: bool = False,
) -> tuple[Axis, tuple[RichText, float] | None]:
    """The y axis for an intensity panel, plus an optional absolute secondary axis.

    Relative axes stop their ticks at 100 % even when the range grows to fit
    labels. Absolute axes divide by a power of ten and say so in the title.
    """
    title = RichText.plain(label)
    secondary = None
    if relative:
        axis = Axis(
            label=title,
            lo=None if mirrored else 0.0,
            tick_min=-100.0 if mirrored else 0.0,
            tick_max=100.0,
            headroom=True,
            abs_ticklabels=mirrored,
            zeroline=mirrored,
        )
        if absolute_axis and base_peak and base_peak > 0:
            secondary = (RichText.plain("Intensity"), base_peak / 100.0)
    else:
        axis = Axis(
            label=title,
            lo=None if mirrored else 0.0,
            headroom=True,
            scale_exponent=True,
            abs_ticklabels=mirrored,
            zeroline=mirrored,
        )
    return axis, secondary


def mz_label() -> RichText:
    """The m/z axis title, with *m/z* in italics as IUPAC recommends."""
    return RichText((("m/z", "it"),))


def table_marks(
    table: pd.DataFrame,
    *,
    style: FigureStyle,
    theme_mode: theme.ThemeMode,
    render: Literal["sticks", "profile"] = "sticks",
    max_points: int | None = _PROFILE_MAX_POINTS,
    direction: Literal["up", "down"] = "up",
    legend: bool = True,
    hit_layer: bool = True,
    labels: bool = True,
) -> list[Mark]:
    """Marks for one plot table: sticks per ``(series, color)`` group, a hover layer, labels.

    ``direction="down"`` negates the intensities for the lower half of a mirror plot.
    """
    sign = -1.0 if direction == "down" else 1.0
    marks: list[Mark] = []
    mode = theme_mode
    unmatched = theme.unmatched_color(mode)

    if render == "profile":
        mz = table["mz"].to_numpy(dtype=np.float64)
        inten = table["intensity"].to_numpy(dtype=np.float64)
        mz, inten = _decimate_profile(mz, inten, max_points)
        color = str(table["color"].iloc[0]) if len(table) else theme.charge_color(1, mode)
        marks.append(
            Line(
                x=mz,
                y=inten * sign,
                color=color,
                width=style.line_width,
                fill=True,
                fill_alpha=0.10,
                name="profile",
                hovertemplate="m/z: %{x:.4f}<br>intensity: %{y:.4g}<extra></extra>",
            )
        )
        return marks

    groups = cast(
        "list[tuple[tuple[Any, Any], pd.DataFrame]]",
        list(table.groupby(["series", "color"], sort=False, dropna=False)),  # type: ignore[arg-type]
    )
    n_series = table["series"].nunique(dropna=False)
    show_legend = legend and n_series > 1
    # Draw context first so matched sticks sit on top of it.
    groups.sort(key=lambda g: 0 if g[0][0] == "unmatched" else 1)
    for (series, color), group in groups:
        color = unmatched if pd.isna(color) else str(color)
        series = "unlabelled" if pd.isna(series) else str(series)
        first = group.iloc[0]
        width_scale = float(first["linewidth"]) / _LINEWIDTH_DEFAULT
        width = width_scale * style.stick_width
        opacity = float(first["opacity"])
        is_context = series == "unmatched"
        if is_context and style.strong_context and color == unmatched:
            # Print needs context peaks that survive a laser printer: the darker
            # neutral grey, fully opaque, thin.
            color = theme.neutral_color(mode)
            opacity = min(1.0, opacity + 0.25)
        dash = str(first["dash"]) if "dash" in group.columns and not _is_blank(first["dash"]) else "solid"
        marks.append(
            Sticks(
                x=group["mz"].to_numpy(dtype=np.float64),
                y=group["intensity"].to_numpy(dtype=np.float64) * sign,
                color=color,
                width=width,
                opacity=opacity,
                dash=dash,
                name=series,
                # A print legend lists what is highlighted; "unmatched" is self-evident.
                legend=show_legend and not (is_context and style.print_ink),
            )
        )

    if hit_layer and len(table):
        marks.append(
            HitLayer(
                x=table["mz"].to_numpy(dtype=np.float64),
                y=table["intensity"].to_numpy(dtype=np.float64) * sign,
                hover=[str(h) for h in table["hover"].tolist()],
                size_px=_HIT_TARGET_SIZE,
            )
        )

    if labels:
        label_set = table_labels(table, style=style, theme_mode=mode, direction=direction)
        if label_set is not None:
            marks.append(label_set)
    return marks


def table_labels(
    table: pd.DataFrame,
    *,
    style: FigureStyle,
    theme_mode: theme.ThemeMode,
    direction: Literal["up", "down"] = "up",
) -> LabelSet | None:
    """The table's non-empty labels as a :class:`LabelSet`, ranked for placement."""
    mask = table["label"].notna() & (table["label"] != "")
    if not mask.any():
        return None
    rows = table[mask]
    mzpaf = table.attrs.get("label_format") == "mzpaf"
    sign = -1.0 if direction == "down" else 1.0
    inten = rows["intensity"].to_numpy(dtype=np.float64)
    top = float(np.nanmax(np.abs(table["intensity"].to_numpy(dtype=np.float64)))) or 1.0
    texts: list[RichText] = []
    prio: list[float] = []
    colors: list[str] = []
    for text, color, rel in zip(rows["label"].tolist(), rows["label_color"].tolist(), inten / top, strict=True):
        if mzpaf:
            lab = best_label(str(text))
            texts.append(lab.rich)
            base = lab.priority
        else:
            texts.append(RichText.plain(str(text).replace("<br>", ", ")))
            base = 1.0
        prio.append(base * (0.2 + max(0.0, float(rel))))
        if _is_blank(color):
            color = theme.text_color("secondary", theme_mode)
        colors.append(str(color) if style.label_series_color else theme.text_color("secondary", theme_mode))
    size = style.label_size
    if "label_size" in rows.columns:
        sizes = rows["label_size"].to_numpy(dtype=np.float64)
        finite = sizes[np.isfinite(sizes)]
        if len(finite):
            size = float(finite[0])
    rotation = 0.0
    if "label_angle" in rows.columns:
        angles = rows["label_angle"].to_numpy(dtype=np.float64)
        finite = angles[np.isfinite(angles)]
        if len(finite):
            rotation = -float(finite[0])
    return LabelSet(
        x=rows["mz"].to_numpy(dtype=np.float64),
        y=inten * sign,
        texts=texts,
        colors=colors,
        priority=np.asarray(prio, dtype=np.float64),
        size=size,
        direction=direction,
        rotation=rotation,
        gap=style.label_gap,
    )


def table_panel(
    table: pd.DataFrame,
    *,
    style: FigureStyle,
    theme_mode: theme.ThemeMode,
    render: Literal["sticks", "profile"] = "sticks",
    max_points: int | None = _PROFILE_MAX_POINTS,
    absolute_axis: bool = False,
) -> Panel:
    """A one-panel spectrum from a plot table."""
    marks = table_marks(table, style=style, theme_mode=theme_mode, render=render, max_points=max_points)
    relative = table.attrs.get("intensity_scale") == "relative" and table.attrs.get("intensity_label", "").startswith(
        "Relative"
    )
    base_peak = None
    if "intensity_abs" in table.columns and len(table):
        base_peak = float(np.nanmax(table["intensity_abs"].to_numpy(dtype=np.float64)))
    y_axis, secondary = intensity_axis(
        table.attrs.get("intensity_label", "Intensity"),
        relative=relative,
        base_peak=base_peak,
        absolute_axis=absolute_axis,
    )
    if render == "profile":
        y_axis.headroom = False
        y_axis.pad = 0.04
    return Panel(marks=marks, x=Axis(label=mz_label()), y=y_axis, y_secondary=secondary)


def plot_from_table(
    table: pd.DataFrame,
    *,
    title: str | None = None,
    theme_mode: theme.ThemeMode | None = None,
    render: Literal["sticks", "profile"] | None = None,
    max_points: int | None = _PROFILE_MAX_POINTS,
    backend: Backend = "plotly",
    style: StyleName | FigureStyle | None = None,
    size: SizeLike = None,
    absolute_axis: bool = False,
    **layout_kwargs,
) -> Any:
    """Draw a plot table: sticks per ``(series, color)`` group, placed labels, hover.

    Every row with a non-empty ``label`` is a label candidate. The layout places
    each one above its peak, or beside it with a leader line when that spot is
    taken, and drops it only when nothing nearby is free. Stronger and more
    informative ions are placed first.

    Parameters
    ----------
    table:
        DataFrame produced by :func:`build_plot_table` or
        :func:`build_annot_plot_table`, or a user-modified copy thereof.
        Required columns: ``mz``, ``intensity``, ``series``, ``color``,
        ``linewidth``, ``opacity``, ``hover``, ``label``, ``label_size``,
        ``label_color``.
    title:
        Plot title. Drawn by the ``"screen"`` and ``"talk"`` styles; the
        ``"paper"`` style draws it only when you pass one.
    theme_mode:
        ``"light"`` or ``"dark"``. ``None`` uses the global plot theme.
    render:
        ``"sticks"`` or ``"profile"``. ``None`` uses the table metadata.
    max_points:
        Maximum profile samples to draw after min/max decimation. ``None``
        draws every sample.
    backend:
        ``"plotly"`` (default), ``"matplotlib"``, or ``"spec"`` for the
        backend-neutral :class:`~spxtacular.figspec.FigureSpec`.
    style:
        ``"screen"``, ``"paper"`` or ``"talk"``, or a
        :class:`~spxtacular.style.FigureStyle`. ``None`` is ``"screen"`` for
        plotly and ``"paper"`` for matplotlib.
    size:
        ``"single"`` (85 mm), ``"onehalf"`` (114 mm), ``"double"`` (175 mm), a
        width in mm, or ``(width_mm, height_mm)``. ``None`` uses the style's size.
    absolute_axis:
        With relative intensities, add a right-hand axis in absolute intensity.
    **layout_kwargs:
        Plotly only: forwarded to ``fig.update_layout`` last.

    Returns
    -------
    A plotly ``Figure``, a matplotlib ``Figure``, or a ``FigureSpec``.
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in table.columns]
    if missing:
        raise SpxtacularError(f"plot table is missing required column(s): {', '.join(missing)}")

    mode = render if render is not None else table.attrs.get("render", "sticks")
    if mode not in ("sticks", "profile"):
        raise SpxtacularError(f"render must be 'sticks' or 'profile', got {mode!r}")
    key = check_backend(backend)
    fig_style = resolve_style(style, key)
    theme_resolved = theme.resolve_mode(theme_mode)
    panel = table_panel(
        table,
        style=fig_style,
        theme_mode=theme_resolved,
        render=mode,
        max_points=max_points,
        absolute_axis=absolute_axis,
    )
    default_title = "Profile spectrum" if mode == "profile" else "Spectrum"
    spec = new_spec(
        Cell(panels=[panel], title=figure_title(title, default_title, fig_style)),
        style=fig_style,
        backend=key,
        size=size,
        theme_mode=theme_resolved,
        layout_kwargs=layout_kwargs,
    )
    return finish(spec, key)


def figure_title(title: str | None, default: str, style: FigureStyle) -> RichText | None:
    """The title to draw: the caller's, else the default when the style shows titles."""
    if title:
        return RichText.plain(title)
    return RichText.plain(default) if style.show_title else None
