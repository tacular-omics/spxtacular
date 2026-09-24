"""
Resolve a :class:`~spxtacular.figspec.FigureSpec` into concrete geometry.

Everything here works in points (1/72 inch) with the origin at the bottom-left
of the figure, which is the convention both matplotlib (figure fractions) and
plotly (paper domains) use. The backends only convert units.

What gets decided here, once, for both backends:

* **Margins** from measured text: the widest tick label, the axis titles, the
  title row, the legend row and any header above a panel.
* **Ticks** at round numbers, spaced for the style's font size.
* **Stacked panels** share their left edge, so y axes line up.
* **Label placement**: every label is placed above its peak if that spot is
  free, otherwise nearby with a leader line, otherwise dropped. See
  :func:`place_labels`.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, replace
from itertools import pairwise
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from . import theme
from ._text import RichText, text_width
from .figspec import (
    Axis,
    AxSegments,
    AxText,
    Band,
    Bars,
    Cell,
    FigureSpec,
    LabelSet,
    Line,
    Panel,
    Points,
    RefLine,
    Sticks,
)
from .style import PT_PER_MM, FigureStyle

if TYPE_CHECKING:
    from collections.abc import Sequence

_PAD = 3.0  #: outer padding, pt
_PANEL_GAP_SHARED = 5.0  #: gap between stacked panels that share an x axis, pt
_HEADROOM_STEPS: tuple[float, ...] = (1.08, 1.16, 1.26, 1.38, 1.52)


# ---------------------------------------------------------------------------
# Resolved structures
# ---------------------------------------------------------------------------


@dataclass
class ResolvedAxis:
    lo: float
    hi: float
    ticks: list[float]
    ticktext: list[str]
    title: RichText | None
    show_ticklabels: bool = True
    grid: bool = False
    zeroline: bool = False
    visible: bool = True
    #: Plotly may choose its own ticks (interactive zooming) for this axis.
    auto_ticks: bool = False
    #: Axis title font size, pt; ``None`` means the style's ``axis_title_size``.
    #: Smaller when the full-size title would not fit along the axis.
    title_size: float | None = None
    #: Tick label rotation in degrees (90 reads bottom to top), for crowded category labels.
    tick_angle: float = 0.0


@dataclass
class PlacedLabel:
    x: float  #: data-space anchor
    y: float
    text: RichText
    color: str
    size: float
    #: Offset of the label box's anchor edge centre from the data anchor, pt.
    dx: float
    dy: float
    va: Literal["bottom", "top"]
    rotation: float
    #: Leader line in pt relative to the anchor, or None.
    leader: tuple[float, float, float, float] | None
    name: str
    #: Box in panel pt coordinates, for tests and debugging.
    box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    #: The label covers context sticks; backends draw it on a background patch.
    knockout: bool = False


@dataclass
class ResolvedPanel:
    panel: Panel
    index: int  #: 1-based, across the whole figure
    rect: tuple[float, float, float, float]  #: left, bottom, width, height in pt
    x: ResolvedAxis
    y: ResolvedAxis
    y2: ResolvedAxis | None
    labels: list[PlacedLabel] = field(default_factory=list)
    dropped: int = 0
    #: RefLine label placement: (mark index, dx pt, dy pt, ha)
    refline_labels: dict[int, tuple[float, float, str]] = field(default_factory=dict)
    shared_with: int | None = None  #: index of the panel whose x axis this one matches


@dataclass
class LegendItem:
    name: str
    color: str
    kind: Literal["line", "marker", "bar"]
    width: float
    dash: str = "solid"


@dataclass
class ResolvedCell:
    cell: Cell
    rect: tuple[float, float, float, float]
    panels: list[ResolvedPanel]
    title_xy: tuple[float, float] | None  #: left, top of the title, pt
    letter_xy: tuple[float, float] | None  #: left, top of the letter, pt
    legend: list[LegendItem]
    legend_xy: tuple[float, float] | None  #: right, bottom of the legend row, pt
    colorbar_rect: tuple[float, float, float, float] | None = None


@dataclass
class Ink:
    """Resolved chrome colours."""

    surface: str
    axis: str
    tick: str
    ticklabel: str
    title: str
    axis_title: str
    grid: str
    muted: str
    primary: str


@dataclass
class ResolvedFigure:
    spec: FigureSpec
    width: float  #: pt
    height: float  #: pt
    cells: list[ResolvedCell]
    ink: Ink

    @property
    def style(self) -> FigureStyle:
        return self.spec.style

    @property
    def panels(self) -> list[ResolvedPanel]:
        return [p for c in self.cells for p in c.panels]


# ---------------------------------------------------------------------------
# Ticks
# ---------------------------------------------------------------------------


def nice_step(span: float, target: float) -> float:
    """A 1/2/2.5/5 x 10^k step giving about ``target`` intervals over ``span``."""
    if span <= 0 or not math.isfinite(span):
        return 1.0
    raw = span / max(target, 1.0)
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if m * mag >= raw * 0.999:
            return m * mag
    return 10.0 * mag


def tick_values(
    lo: float, hi: float, length_pt: float, spacing_pt: float, min_sep_pt: float = 0.0
) -> tuple[list[float], float]:
    """Round tick positions inside ``[lo, hi]`` about ``spacing_pt`` apart.

    A short axis gets fewer ticks, not denser ones: the step is only refined
    towards three ticks while neighbours stay at least ``min_sep_pt`` apart.
    """
    target = max(1.0, length_pt / max(spacing_pt, 1.0))
    span = hi - lo

    def build(step: float) -> list[float]:
        out = []
        v = math.ceil(lo / step - 1e-9) * step
        while v <= hi + step * 1e-9:
            out.append(round(v, 12) + 0.0)
            v += step
        return out

    step = nice_step(span, target)
    ticks = build(step)
    # A value axis with only "0" and one other label is hard to read: densify
    # (at most a few times) until there are at least three ticks, as long as
    # the labels still have room.
    for _ in range(3):
        if len(ticks) >= 3:
            break
        target *= 1.6
        finer = nice_step(span, target)
        if span > 0 and finer / span * length_pt < min_sep_pt:
            break
        step, ticks = finer, build(finer)
    return ticks, step


_TICK_GAP_EM = 0.25  #: minimum clear space between neighbouring tick labels, in font sizes


def fit_tick_labels(
    values: Sequence[float],
    texts: Sequence[str],
    lo: float,
    hi: float,
    length_pt: float,
    font_size: float,
    *,
    vertical: bool,
    rotate: bool = False,
) -> tuple[list[float], list[str], float]:
    """Thin (or, for category labels, rotate) tick labels so neighbours do not overlap.

    Returns the kept values, their texts and the label angle (0 or 90). A y
    axis compares label heights, an x axis label widths; ``rotate`` lets an x
    axis stand its labels on end first when that is enough. Thinning keeps a
    regular stride and both end ticks when it can; an axis too short for two
    labels keeps the last one.
    """
    vals = [float(v) for v in values]
    txts = [str(t) for t in texts]
    if len(vals) < 2 or hi <= lo:
        return vals, txts, 0.0
    line = font_size * 1.18
    gap = font_size * _TICK_GAP_EM
    inside = [i for i, v in enumerate(vals) if lo - 1e-9 * abs(hi - lo) <= v <= hi + 1e-9 * abs(hi - lo)]
    pos = {i: (vals[i] - lo) / (hi - lo) * length_pt for i in inside}

    def extent(i: int, angle: float) -> float:
        if vertical or angle:
            return line
        return text_width(txts[i], font_size)

    def fits(keep: list[int], angle: float) -> bool:
        return all(abs(pos[b] - pos[a]) >= (extent(a, angle) + extent(b, angle)) / 2 + gap for a, b in pairwise(keep))

    angles = [0.0, 90.0] if rotate and not vertical else [0.0]
    for angle in angles:
        if fits(inside, angle):
            return vals, txts, angle
    angle = angles[-1]
    n = len(inside)
    best: list[int] | None = None
    for k in range(2, n):
        # Prefer strides that keep both ends; fall back to any stride.
        starts = [0] if (n - 1) % k == 0 else [0, (n - 1) % k]
        for start in starts:
            keep = inside[start::k]
            if len(keep) >= 2 and fits(keep, angle):
                best = keep
                break
        if best is not None:
            break
    if best is None:
        best = [inside[0], inside[-1]] if n >= 2 and fits([inside[0], inside[-1]], angle) else inside[-1:]
    return [vals[i] for i in best], [txts[i] for i in best], angle


def format_tick(value: float, step: float) -> str:
    if step >= 1 or step <= 0:
        text = f"{value:.0f}"
    else:
        decimals = min(6, max(0, math.ceil(-math.log10(step) - 1e-9)))
        if step * 10**decimals % 1 > 1e-6:  # e.g. 0.25 needs 2 decimals
            decimals += 1
        text = f"{value:.{decimals}f}"
    if text.startswith("-") and float(text) == 0:
        text = text[1:]
    return text.replace("-", "\u2212")


# ---------------------------------------------------------------------------
# Data extents
# ---------------------------------------------------------------------------


def _finite(a: NDArray[np.float64]) -> NDArray[np.float64]:
    a = np.asarray(a, dtype=np.float64)
    return a[np.isfinite(a)]


def _data_extent(panel: Panel, axis: Literal["x", "y"]) -> tuple[float, float] | None:
    vals: list[NDArray[np.float64]] = []
    for m in panel.marks:
        if isinstance(m, Sticks):
            if axis == "x":
                vals.append(_finite(m.x))
            else:
                vals.append(_finite(m.y))
                if len(m.y):
                    vals.append(np.array([m.base]))
        elif isinstance(m, Line):
            vals.append(_finite(m.x if axis == "x" else m.y))
            if axis == "y" and m.fill and len(m.y):
                vals.append(np.array([0.0]))
        elif isinstance(m, Points):
            vals.append(_finite(m.x if axis == "x" else m.y))
        elif isinstance(m, Bars):
            if axis == "x":
                x = _finite(m.x)
                vals.append(x - m.width / 2)
                vals.append(x + m.width / 2)
            else:
                vals.append(_finite(m.height))
                if len(m.height):
                    vals.append(np.array([0.0]))
        elif isinstance(m, Band) and m.orient == ("v" if axis == "x" else "h"):
            vals.append(np.array([m.lo, m.hi]))
    vals = [v for v in vals if len(v)]
    if not vals:
        return None
    allv = np.concatenate(vals)
    return float(allv.min()), float(allv.max())


def _x_range(panel: Panel) -> tuple[float, float]:
    ax = panel.x
    ext = _data_extent(panel, "x")
    if ext is None:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = ext
        span = hi - lo
        if span <= 0:
            half = max(abs(lo) * 0.05, 1.0)
            lo, hi = lo - half, hi + half
        else:
            pad = ax.pad if ax.pad else 0.04
            lo, hi = lo - span * pad, hi + span * pad
    if ax.lo is not None:
        lo = ax.lo
    if ax.hi is not None:
        hi = ax.hi
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _y_base_range(panel: Panel) -> tuple[float, float]:
    """The y range before label headroom."""
    ax = panel.y
    ext = _data_extent(panel, "y")
    if ext is None:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = ext
        if hi <= lo:
            hi = lo + (abs(lo) or 1.0)
        if ax.pad:
            span = hi - lo
            lo = lo - span * ax.pad if lo < 0 else lo
            hi = hi + span * ax.pad
    if ax.lo is not None:
        lo = ax.lo
    if ax.hi is not None:
        hi = ax.hi
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


# ---------------------------------------------------------------------------
# Tick text and axis sizes
# ---------------------------------------------------------------------------


def _resolve_ticks(
    ax: Axis, lo: float, hi: float, length_pt: float, spacing: float, min_sep: float = 0.0
) -> tuple[list[float], list[str], RichText | None, bool]:
    """Tick values, texts, and the (possibly exponent-extended) title."""
    title = ax.label
    if ax.ticks is not None:
        values = [float(v) for v in ax.ticks]
        texts = list(ax.ticktext) if ax.ticktext is not None else [f"{v:g}" for v in values]
        return values, texts, title, False
    tlo = lo if ax.tick_min is None else max(lo, ax.tick_min)
    thi = hi if ax.tick_max is None else min(hi, ax.tick_max)
    # Ticks are spaced over the whole axis so the density does not change when
    # the upper limit is capped for headroom.
    values, step = tick_values(tlo, thi, length_pt * (thi - tlo) / max(hi - lo, 1e-300), spacing, min_sep)
    shown = [abs(v) for v in values] if ax.abs_ticklabels else values
    exponent = 0
    if ax.scale_exponent and values:
        top = max(abs(v) for v in values)
        if top >= 1e4:
            exponent = math.floor(math.log10(top))
    if exponent:
        scale = 10.0**exponent
        texts = [format_tick(v / scale, step / scale) for v in shown]
        if title is not None:
            title = title + RichText(((" (\u00d710", "n"), (str(exponent), "sup"), (")", "n")))
    else:
        texts = [format_tick(v, step) for v in shown]
    return values, texts, title, not ax.abs_ticklabels and not exponent


def _is_category(ax: Axis) -> bool:
    """Explicit tick text that is not just numbers (channel names, sample names)."""
    if ax.ticktext is None:
        return False
    for t in ax.ticktext:
        try:
            float(str(t).replace("\u2212", "-"))
        except ValueError:
            return True
    return False


def _axis_ticks(
    ax: Axis, lo: float, hi: float, length_pt: float, style: FigureStyle, *, vertical: bool
) -> tuple[list[float], list[str], RichText | None, bool, float]:
    """:func:`_resolve_ticks` plus :func:`fit_tick_labels`: ticks whose labels fit the axis.

    Returns values, texts, title, whether plotly may pick its own ticks, and the label angle.
    """
    fs = style.font_size
    spacing = style.y_tick_spacing if vertical else style.x_tick_spacing
    min_sep = fs * (1.18 + _TICK_GAP_EM) if vertical else 0.0
    values, texts, title, auto = _resolve_ticks(ax, lo, hi, length_pt, spacing, min_sep)
    kept, ktexts, angle = fit_tick_labels(
        values, texts, lo, hi, length_pt, fs, vertical=vertical, rotate=not vertical and _is_category(ax)
    )
    if len(kept) != len(values):
        auto = False
    return kept, ktexts, title, auto, angle


#: Word abbreviations tried, in order, when an axis title is longer than its axis.
_TITLE_ABBREVIATIONS: tuple[tuple[str, str], ...] = (
    ("Relative intensity", "Rel. int."),
    ("Relative abundance", "Rel. abund."),
    ("Relative", "Rel."),
    ("Intensity", "Int."),
    ("intensity", "int."),
    ("Mass error", "Error"),
    ("Retention time", "RT"),
    ("Ion mobility", "IM"),
)
_MIN_TITLE_SCALE = 0.75


def _abbreviate(title: RichText) -> RichText:
    runs = []
    for text, role in title.runs:
        if role == "n":
            for long, short in _TITLE_ABBREVIATIONS:
                text = text.replace(long, short)
        runs.append((text, role))
    return RichText(tuple(runs))


def fit_axis_title(title: RichText | None, length_pt: float, size: float) -> tuple[RichText | None, float | None]:
    """An axis title that fits along ``length_pt``: as given, abbreviated, then set smaller.

    Returns the title and its font size (``None`` for the style's size). The
    size never drops below 75 % of the style's.
    """
    if title is None or title.width(size) <= length_pt:
        return title, None
    short = _abbreviate(title)
    if short.width(size) <= length_pt:
        return short, None
    scaled = max(size * _MIN_TITLE_SCALE, size * length_pt / max(short.width(size), 1e-9))
    return short, scaled


# ---------------------------------------------------------------------------
# Label placement
# ---------------------------------------------------------------------------


def _range_max_table(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sparse table: row ``k`` holds the max of ``v[i : i + 2**k]`` (``-inf`` past the end)."""
    n = len(v)
    levels = max(1, n.bit_length())
    t = np.full((levels, n), -np.inf)
    t[0] = v
    for k in range(1, levels):
        half, m = 1 << (k - 1), n - (1 << k) + 1
        if m <= 0:
            break
        t[k, :m] = np.maximum(t[k - 1, :m], t[k - 1, half : half + m])
    return t


def _range_max(t: NDArray[np.float64], i0: NDArray[np.intp], i1: NDArray[np.intp]) -> NDArray[np.float64]:
    """Max over ``[i0, i1)`` for each pair; ``-inf`` for an empty range."""
    out = np.full(i0.shape, -np.inf)
    ok = i1 > i0
    if ok.any():
        a, b = i0[ok], i1[ok]
        k = np.floor(np.log2(b - a)).astype(np.intp)
        out[ok] = np.maximum(t[k, a], t[k, b - np.left_shift(1, k)])
    return out


class _StickSet:
    """Vertical segments (x, lo, hi) in the canonical frame, sorted by x.

    :meth:`prepare` splits them at the lowest point any label or leader can
    reach: a stick starting below it ("grounded", the usual peak) blocks a box
    exactly when its top reaches the box, which a range-max table answers in
    O(1) per candidate. The rest are checked one by one.
    """

    def __init__(self, x: NDArray[np.float64], lo: NDArray[np.float64], hi: NDArray[np.float64]) -> None:
        ok = np.isfinite(x) & np.isfinite(lo) & np.isfinite(hi)
        x, lo, hi = x[ok], lo[ok], hi[ok]
        order = np.argsort(x, kind="stable")
        self.x, self.lo, self.hi = x[order], lo[order], hi[order]
        self.prepare(-np.inf)

    def __len__(self) -> int:
        return len(self.x)

    def prepare(self, ground: float) -> None:
        g = self.lo < ground
        self.gx, self.glo, self.ghi = self.x[g], self.lo[g], self.hi[g]
        self.gt = _range_max_table(self.ghi) if len(self.gx) else None
        f = ~g
        self.fx, self.flo, self.fhi = self.x[f], self.lo[f], self.hi[f]

    def boxes_hit(
        self,
        x0: NDArray[np.float64],
        x1: NDArray[np.float64],
        y0: NDArray[np.float64],
        y1: NDArray[np.float64],
        pad: float,
    ) -> NDArray[np.bool_]:
        """Per box: does a stick with x in [x0 - pad, x1 + pad] overlap (y0 - 0.3, y1)?"""
        hit = np.zeros(len(x0), dtype=bool)
        if not len(x0):
            return hit
        if self.gt is not None:
            i0 = np.searchsorted(self.gx, x0 - pad)
            i1 = np.searchsorted(self.gx, x1 + pad, side="right")
            hit |= _range_max(self.gt, i0, i1) > y0 - 0.3
        if len(self.fx):
            k0 = int(np.searchsorted(self.fx, float(x0.min()) - pad))
            k1 = int(np.searchsorted(self.fx, float(x1.max()) + pad, side="right"))
            if k1 > k0:
                sx, lo, hi = self.fx[k0:k1], self.flo[k0:k1], self.fhi[k0:k1]
                m = (
                    (sx[None, :] >= (x0 - pad)[:, None])
                    & (sx[None, :] <= (x1 + pad)[:, None])
                    & (lo[None, :] < y1[:, None])
                    & (hi[None, :] > (y0 - 0.3)[:, None])
                )
                hit |= m.any(axis=1)
        return hit

    def leader_hit(self, ax: float, start: tuple[float, float], end: tuple[float, float]) -> bool:
        """Does a leader from ``start`` to ``end`` cross a stick that stands taller than it?"""
        lo_x, hi_x = min(start[0], end[0]), max(start[0], end[0])
        sloped = hi_x - lo_x > 1e-6
        for sxs, los, his, table in ((self.gx, self.glo, self.ghi, self.gt), (self.fx, self.flo, self.fhi, None)):
            if not len(sxs):
                continue
            k0 = int(np.searchsorted(sxs, lo_x - 0.3))
            k1 = int(np.searchsorted(sxs, hi_x + 0.3, side="right"))
            if k1 <= k0:
                continue
            if table is not None:
                k = (k1 - k0).bit_length() - 1
                top = max(float(table[k, k0]), float(table[k, k1 - (1 << k)]))
                if sloped:
                    slope = (end[1] - start[1]) / (end[0] - start[0])
                    low = min(start[1] + slope * (x - start[0]) for x in (lo_x - 0.3, hi_x + 0.3)) - 0.3
                else:
                    low = start[1]
                if top <= low:
                    continue
            sx = sxs[k0:k1]
            own = np.abs(sx - ax) < 1e-6
            if sloped:
                ly = start[1] + (sx - start[0]) / (end[0] - start[0]) * (end[1] - start[1])
                if np.any(~own & (his[k0:k1] > ly - 0.3) & (los[k0:k1] < ly)):
                    return True
            elif np.any(~own & (his[k0:k1] > start[1])):
                return True
        return False


@dataclass
class _Obstacles:
    """What labels keep clear of, in the canonical frame."""

    #: Sticks and traces a label never covers.
    hard: _StickSet
    #: Context sticks (unmatched peaks): avoided first, covered when nothing else fits.
    soft: _StickSet
    #: Dots and bars, (n, 4) as (x0, y0, x1, y1).
    boxes: NDArray[np.float64]


def _seg_box_matrix(segs: NDArray[np.float64], boxes: NDArray[np.float64]) -> NDArray[np.bool_]:
    """(boxes x segs): does segment j cross box i? Liang-Barsky over both axes."""
    if len(segs) == 0 or len(boxes) == 0:
        return np.zeros((len(boxes), len(segs)), dtype=bool)
    sx0, sy0 = segs[:, 0], segs[:, 1]
    dx, dy = segs[:, 2] - sx0, segs[:, 3] - sy0
    t0 = np.zeros((len(boxes), len(segs)))
    t1 = np.ones((len(boxes), len(segs)))
    ok = np.ones((len(boxes), len(segs)), dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for p, q in (
            (-dx, sx0[None, :] - boxes[:, 0, None]),
            (dx, boxes[:, 2, None] - sx0[None, :]),
            (-dy, sy0[None, :] - boxes[:, 1, None]),
            (dy, boxes[:, 3, None] - sy0[None, :]),
        ):
            zero = p == 0
            if zero.any():
                ok &= ~zero[None, :] | (q >= 0)
            r = q / p[None, :]
            t0 = np.where((p < 0)[None, :], np.maximum(t0, r), t0)
            t1 = np.where((p > 0)[None, :], np.minimum(t1, r), t1)
    return ok & (t0 <= t1)


def _segs_hit_boxes(segs: NDArray[np.float64], boxes: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Per box (rows of ``boxes``): does any of ``segs`` cross it?"""
    return _seg_box_matrix(segs, boxes).any(axis=1)


def _segs_cross(segs: NDArray[np.float64], others: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Per row of ``segs`` (N x 4): does it properly cross any of ``others`` (M x 4)?"""
    if len(segs) == 0 or len(others) == 0:
        return np.zeros(len(segs), dtype=bool)
    ax, ay, bx, by = (segs[:, i, None] for i in range(4))
    cx, cy, dx, dy = (others[None, :, i] for i in range(4))

    def orient(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)

    o1 = orient(ax, ay, bx, by, cx, cy)
    o2 = orient(ax, ay, bx, by, dx, dy)
    o3 = orient(cx, cy, dx, dy, ax, ay)
    o4 = orient(cx, cy, dx, dy, bx, by)
    return ((o1 * o2 < 0) & (o3 * o4 < 0)).any(axis=1)


def _candidates(w: float, h: float) -> list[tuple[float, float, bool, float]]:
    """(dx, dy, leader, cost), cheapest first."""
    out: list[tuple[float, float, bool, float]] = []
    for fx in (0.0, 0.3, -0.3, 0.6, -0.6, 1.1, -1.1, 1.7, -1.7, 2.4, -2.4):
        for fy in (0.0, 0.9, 1.8, 2.8, 4.0, 5.5):
            dx, dy = fx * w, fy * h
            leader = abs(fx) > 0.35 or fy > 0.0
            if abs(fx) > 0.35 and fy == 0.0:
                continue  # a sideways label with no lift reads as belonging to the neighbour
            cost = abs(fx) + 0.8 * fy + (0.6 if leader else 0.0)
            out.append((dx, dy, leader, cost))
    out.sort(key=lambda c: c[3])
    return out


def _candidate_arrays(
    w: float, h: float, leaders: bool
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    cands = [c for c in _candidates(w, h) if leaders or not c[2]]
    return (
        np.array([c[0] for c in cands], dtype=np.float64),
        np.array([c[1] for c in cands], dtype=np.float64),
        np.array([c[2] for c in cands], dtype=bool),
    )


def _rotated_extent(w: float, h: float, rotation: float) -> tuple[float, float]:
    """Width and height of the axis-aligned box around a ``w`` x ``h`` box turned by ``rotation`` degrees."""
    r = math.radians(rotation % 180)
    c, s = abs(math.cos(r)), abs(math.sin(r))
    return w * c + h * s, w * s + h * c


#: Gap, pt, between a peak label and the left or right edge of the plot area.
_SPINE_CLEAR = 0.6


def place_labels(
    labelsets: Sequence[LabelSet],
    to_pt_x,
    to_pt_y,
    width: float,
    height: float,
    obstacles_up: _Obstacles,
    obstacles_down: _Obstacles | None,
    reserved: list[tuple[float, float, float, float]],
    stop_below: float = -math.inf,
    skip: set[tuple[int, int]] | None = None,
    dropped_keys: set[tuple[int, int]] | None = None,
) -> tuple[list[PlacedLabel], int, float]:
    """Greedy, priority-ordered placement. Returns labels, dropped count, placed priority.

    Works in a canonical "up" frame: for a downward label set (the lower half of
    a mirror plot) the panel is flipped so the same search applies. Each label
    first tries every spot that keeps clear of all marks; if none is free it may
    cover context sticks (unmatched peaks), drawn with a knockout background.

    ``to_pt_x``/``to_pt_y`` map data arrays to panel pt. The run stops early,
    returning score ``-inf``, once it cannot reach ``stop_below``. Labels in
    ``skip`` (``(labelset index, label index)``) count as dropped without a
    search; the keys of labels dropped here are added to ``dropped_keys``.
    """
    pad = 0.6
    # --- per-label geometry, computed once ------------------------------------
    rows: list[tuple[float, int, int, float, float, float, float, float, float, float, float, bool]] = []
    skipped = 0
    ground = {False: math.inf, True: math.inf}
    for s_i, ls in enumerate(labelsets):
        n = len(ls.texts)
        if not n:
            continue
        down = ls.direction == "down"
        xs = np.asarray(to_pt_x(np.asarray(ls.x, dtype=np.float64)), dtype=np.float64)
        ys = np.asarray(to_pt_y(np.asarray(ls.y, dtype=np.float64)), dtype=np.float64)
        prio = np.nan_to_num(np.asarray(ls.priority, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        for j in range(n):
            text = ls.texts[j]
            if not text:
                continue
            if skip is not None and (s_i, j) in skip:
                skipped += 1
                continue
            size = ls.size
            if ls.sizes is not None and math.isfinite(float(ls.sizes[j])) and float(ls.sizes[j]) > 0:
                size = float(ls.sizes[j])
            rotation = ls.rotation
            if ls.rotations is not None and math.isfinite(float(ls.rotations[j])):
                rotation = float(ls.rotations[j])
            tw, th = _rotated_extent(text.width(size), text.height(size), rotation)
            ax_pt, ay_panel = float(xs[j]), float(ys[j])
            ay = height - ay_panel if down else ay_panel
            off = float(ls.anchor_offset[j]) if ls.anchor_offset is not None else 0.0
            if math.isfinite(ax_pt) and math.isfinite(ay):
                ground[down] = min(ground[down], ay + off + ls.gap * 0.35 - 1e-6)
            rows.append((float(prio[j]), s_i, j, ax_pt, ay_panel, ay, off, tw, th, size, rotation, down))
    rows.sort(key=lambda t: -t[0])
    remaining = [*np.cumsum([r[0] for r in rows][::-1])[::-1].tolist(), 0.0]

    up_ground = ground[False] if obstacles_down is not None else min(ground[False], ground[True])
    obstacles_up.hard.prepare(up_ground)
    obstacles_up.soft.prepare(up_ground)
    if obstacles_down is not None:
        obstacles_down.hard.prepare(ground[True])
        obstacles_down.soft.prepare(ground[True])

    placed: list[PlacedLabel] = []
    dropped = skipped
    score = 0.0
    # Boxes and leaders in the *panel* frame (y up from the panel bottom).
    box_arr = np.zeros((len(rows) + len(reserved) + 1, 4))
    n_boxes = 0
    for b in reserved:
        box_arr[n_boxes] = b
        n_boxes += 1
    leaders = np.zeros((len(rows) + 1, 4))
    n_leaders = 0
    cand_cache: dict[tuple[float, float, bool], tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]] = {}

    for idx, (prio, s_i, j, ax_pt, ay_panel, ay, off, tw, th, size, rotation, down) in enumerate(rows):
        if score + remaining[idx] < stop_below:
            return placed, dropped + len(rows) - idx, -math.inf
        ls = labelsets[s_i]
        if not (math.isfinite(ax_pt) and math.isfinite(ay)):
            dropped += 1
            continue
        obst = obstacles_down if (down and obstacles_down is not None) else obstacles_up
        base = ay + off + ls.gap
        key = (round(tw, 2), round(th, 2), ls.leaders)
        cands = cand_cache.get(key)
        if cands is None:
            cands = _candidate_arrays(tw, th, ls.leaders)
            cand_cache[key] = cands
        cdx, cdy, clead = cands

        # --- box stage, vectorised over candidates ----------------------------
        cx = ax_pt + cdx
        x0, x1 = cx - tw / 2, cx + tw / 2
        y0 = base + cdy
        y1 = y0 + th
        # Keep clear of the y spines: the knockout background (0.1 em in
        # matplotlib) plus half a spine must not reach the plot edge.
        edge = size * 0.1 + _SPINE_CLEAR
        sel = np.flatnonzero((x0 >= edge) & (x1 <= width - edge) & (y1 <= height) & (y0 >= 0))
        if len(sel) and n_boxes:
            py0, py1 = (height - y1[sel], height - y0[sel]) if down else (y0[sel], y1[sel])
            b = box_arr[:n_boxes]
            hit = (
                (b[None, :, 0] < (x1[sel] + pad)[:, None])
                & (b[None, :, 2] > (x0[sel] - pad)[:, None])
                & (b[None, :, 1] < (py1 + pad)[:, None])
                & (b[None, :, 3] > (py0 - pad)[:, None])
            ).any(axis=1)
            sel = sel[~hit]
        if len(sel) and len(obst.boxes):
            ob = obst.boxes
            hit = (
                (ob[None, :, 0] < x1[sel, None])
                & (ob[None, :, 2] > x0[sel, None])
                & (ob[None, :, 1] < y1[sel, None])
                & (ob[None, :, 3] > y0[sel, None])
            ).any(axis=1)
            sel = sel[~hit]
        if len(sel):
            sel = sel[~obst.hard.boxes_hit(x0[sel], x1[sel], y0[sel], y1[sel], pad)]
        if len(sel) and n_leaders:
            py0, py1 = (height - y1[sel], height - y0[sel]) if down else (y0[sel], y1[sel])
            # Only leaders whose extent meets the candidates' joint extent can cross one.
            lead = leaders[:n_leaders]
            near = (
                (np.minimum(lead[:, 0], lead[:, 2]) <= x1[sel].max())
                & (np.maximum(lead[:, 0], lead[:, 2]) >= x0[sel].min())
                & (np.minimum(lead[:, 1], lead[:, 3]) <= py1.max())
                & (np.maximum(lead[:, 1], lead[:, 3]) >= py0.min())
            )
            if near.any():
                hit = _segs_hit_boxes(lead[near], np.column_stack([x0[sel], py0, x1[sel], py1]))
                sel = sel[~hit]
        soft_hit = obst.soft.boxes_hit(x0[sel], x1[sel], y0[sel], y1[sel], pad) if len(sel) else np.zeros(0, bool)

        # --- leader stage, cheapest candidate first -----------------------------
        chosen = None
        # Pass 2 retries only candidates that context sticks alone blocked.
        retry: list[int] = []
        soft_set = set(sel[soft_hit].tolist())
        for knockout in (False, True):
            if knockout:
                if not retry and not soft_hit.any():
                    break
                pool = np.union1d(sel[soft_hit], np.asarray(retry, dtype=np.intp))
            else:
                pool = sel[~soft_hit]
            if not len(pool):
                continue
            # Leaders for the whole pool at once; stick checks only for survivors, in cost order.
            px0, px1, py0_ = x0[pool], x1[pool], y0[pool]
            lead = clead[pool].copy()
            lx = np.minimum(np.maximum(ax_pt, px0 + 1.0), px1 - 1.0)
            l_sy = ay + off + ls.gap * 0.35
            l_ey = py0_ - 0.4
            lead &= ~((l_ey - l_sy < 1.0) & (np.abs(lx - ax_pt) < 1.0))
            bad = np.zeros(len(pool), dtype=bool)
            seg_of: dict[int, list[float]] = {}
            li = np.flatnonzero(lead)
            if len(li):
                sy0 = height - l_sy if down else l_sy
                ey = height - l_ey[li] if down else l_ey[li]
                segs = np.column_stack([np.full(len(li), ax_pt), np.full(len(li), sy0), lx[li], ey])
                if n_boxes:
                    b = box_arr[:n_boxes]
                    near = (
                        (b[:, 0] <= max(ax_pt, float(lx[li].max())))
                        & (b[:, 2] >= min(ax_pt, float(lx[li].min())))
                        & (b[:, 1] <= max(sy0, float(ey.max())))
                        & (b[:, 3] >= min(sy0, float(ey.min())))
                    )
                    if near.any():
                        bad[li] |= _seg_box_matrix(segs, b[near]).any(axis=0)
                if n_leaders:
                    bad[li] |= _segs_cross(segs, leaders[:n_leaders])
                seg_of = dict(zip(li.tolist(), segs.tolist(), strict=True))
            for k in np.flatnonzero(~bad).tolist():
                c = int(pool[k])
                bx0, bx1, by0, by1 = float(x0[c]), float(x1[c]), float(y0[c]), float(y1[c])
                pby0, pby1 = (height - by1, height - by0) if down else (by0, by1)
                seg = None
                if lead[k]:
                    l_start = (ax_pt, l_sy)
                    l_end = (float(lx[k]), by0 - 0.4)
                    if obst.hard.leader_hit(ax_pt, l_start, l_end):
                        continue
                    if not knockout and obst.soft.leader_hit(ax_pt, l_start, l_end):
                        retry.append(c)
                        continue
                    seg = tuple(seg_of[k])
                chosen = (float(cx[c]), by0, bx0, bx1, pby0, pby1, seg, knockout and c in soft_set)
                break
            if chosen is not None:
                break

        if chosen is None:
            dropped += 1
            if dropped_keys is not None:
                dropped_keys.add((s_i, j))
            continue
        ccx, by0, bx0, bx1, pby0, pby1, seg, knock = chosen
        box_arr[n_boxes] = (bx0, pby0, bx1, pby1)
        n_boxes += 1
        rel_leader = None
        if seg is not None:
            leaders[n_leaders] = seg
            n_leaders += 1
            rel_leader = (seg[0] - ax_pt, seg[1] - ay_panel, seg[2] - ax_pt, seg[3] - ay_panel)
        if down:
            dy_rel = (height - by0) - ay_panel  # top edge of the box, panel frame
            va: Literal["bottom", "top"] = "top"
        else:
            dy_rel = by0 - ay_panel
            va = "bottom"
        placed.append(
            PlacedLabel(
                x=float(ls.x[j]),
                y=float(ls.y[j]),
                text=ls.texts[j],
                color=ls.colors[j],
                size=size,
                dx=ccx - ax_pt,
                dy=dy_rel,
                va=va,
                rotation=rotation,
                leader=rel_leader,
                name=ls.name,
                box=(bx0, pby0, bx1, pby1),
                knockout=knock,
            )
        )
        score += prio
    return placed, dropped, score


def _stick_obstacles(panel: Panel, to_pt_x, to_pt_y, height: float, flip: bool) -> _Obstacles:
    parts: dict[bool, tuple[list, list, list]] = {False: ([], [], []), True: ([], [], [])}
    boxes: list[NDArray[np.float64]] = []
    for m in panel.marks:
        if isinstance(m, Sticks) and m.obstacle != "none" and len(m.x):
            x = to_pt_x(np.asarray(m.x, dtype=np.float64))
            a = to_pt_y(np.full(len(m.x), m.base))
            b = to_pt_y(np.asarray(m.y, dtype=np.float64))
            xs, los, his = parts[m.obstacle == "soft"]
            xs.append(x)
            los.append(np.minimum(a, b))
            his.append(np.maximum(a, b))
        elif isinstance(m, Line) and len(m.x):
            x = to_pt_x(np.asarray(m.x, dtype=np.float64))
            b = to_pt_y(np.asarray(m.y, dtype=np.float64))
            a = to_pt_y(np.zeros(len(m.x))) if m.fill else b - 0.5
            xs, los, his = parts[False]
            xs.append(x)
            los.append(np.minimum(a, b))
            his.append(np.maximum(a, b))
        elif isinstance(m, Points) and len(m.x):
            x = to_pt_x(np.asarray(m.x, dtype=np.float64))
            y = to_pt_y(np.asarray(m.y, dtype=np.float64))
            r = np.nan_to_num(np.asarray(m.sizes, dtype=np.float64), nan=0.0) / 2
            ok = np.isfinite(x) & np.isfinite(y)
            boxes.append(np.column_stack([x - r, y - r, x + r, y + r])[ok])
        elif isinstance(m, Bars) and len(m.x):
            xc = np.asarray(m.x, dtype=np.float64)
            a, b = to_pt_x(xc - m.width / 2), to_pt_x(xc + m.width / 2)
            y0 = np.broadcast_to(to_pt_y(0.0), a.shape)
            y1 = to_pt_y(np.asarray(m.height, dtype=np.float64))
            arr = np.column_stack([np.minimum(a, b), np.minimum(y0, y1), np.maximum(a, b), np.maximum(y0, y1)])
            boxes.append(arr[np.isfinite(arr).all(axis=1)])
    box_arr = np.concatenate(boxes) if boxes else np.zeros((0, 4))

    def sticks(soft: bool) -> _StickSet:
        xs, los, his = parts[soft]
        if not xs:
            empty = np.zeros(0)
            return _StickSet(empty, empty, empty)
        sx, slo, shi = np.concatenate(xs), np.concatenate(los), np.concatenate(his)
        if flip:
            slo, shi = height - shi, height - slo
        return _StickSet(sx, slo, shi)

    if flip and len(box_arr):
        box_arr = np.column_stack([box_arr[:, 0], height - box_arr[:, 3], box_arr[:, 2], height - box_arr[:, 1]])
    return _Obstacles(sticks(False), sticks(True), box_arr)


# ---------------------------------------------------------------------------
# Figure resolution
# ---------------------------------------------------------------------------


def ink_for(style: FigureStyle, mode: theme.ThemeMode) -> Ink:
    primary = theme.text_color("primary", mode)
    secondary = theme.text_color("secondary", mode)
    muted = theme.text_color("muted", mode)
    if style.print_ink:
        return Ink(
            surface=theme.print_surface(mode),
            axis=primary,
            tick=primary,
            ticklabel=primary,
            title=primary,
            axis_title=primary,
            grid=theme.grid_color(mode),
            muted=secondary,
            primary=primary,
        )
    return Ink(
        surface=theme.surface(mode),
        axis=theme.axis_color(mode),
        tick=theme.axis_color(mode),
        ticklabel=muted,
        title=primary,
        axis_title=secondary,
        grid=theme.grid_color(mode),
        muted=muted,
        primary=primary,
    )


def legend_items(cell: Cell, mode: theme.ThemeMode | None = None) -> list[LegendItem]:
    items: list[LegendItem] = []
    seen: set[str] = set()
    for panel in cell.panels:
        for m in panel.marks:
            if not getattr(m, "legend", False):
                continue
            name = getattr(m, "name", None)
            if not name or name in seen:
                continue
            seen.add(name)
            if isinstance(m, Sticks | Line):
                items.append(LegendItem(name, m.color, "line", m.width, m.dash))
            elif isinstance(m, Points):
                color = m.colors[0] if m.colors else theme.text_color("primary", mode)
                items.append(LegendItem(name, color, "marker", 0.0))
    return items


def _legend_width(items: list[LegendItem], size: float) -> float:
    if not items:
        return 0.0
    return sum(text_width(it.name, size) + size * 1.9 for it in items) + size * 0.8 * (len(items) - 1)


def _axis_title_height(title: RichText | None, size: float) -> float:
    return title.height(size) if title else 0.0


def resolve_figure(spec: FigureSpec) -> ResolvedFigure:
    style = spec.style
    mode = spec.theme_mode
    width = spec.width_mm * PT_PER_MM
    height = spec.height_mm * PT_PER_MM
    ncols = max(1, spec.ncols)
    n = len(spec.cells)
    nrows = -(-n // ncols)

    # Row heights: proportional to each row's preferred height.
    cell_w = width / ncols
    prefs = []
    for r in range(nrows):
        row = spec.cells[r * ncols : (r + 1) * ncols]
        prefs.append(
            max(
                fitted_height(c, style, cell_w) if c.fit_height else c.aspect * cell_w + c.extra_height_mm * PT_PER_MM
                for c in row
            )
        )
    total = sum(prefs) or 1.0
    row_h = [height * p / total for p in prefs]

    cells: list[ResolvedCell] = []
    panel_index = 1
    top = height
    for r in range(nrows):
        for c in range(ncols):
            k = r * ncols + c
            if k >= n:
                break
            rect = (c * cell_w, top - row_h[r], cell_w, row_h[r])
            rc = _resolve_cell(spec.cells[k], rect, style, panel_index, first_cell=(k == 0), mode=spec.theme_mode)
            panel_index += len(rc.panels)
            cells.append(rc)
        top -= row_h[r]
    if spec.check_panel_size:
        _warn_small_panels(cells, style)
    return ResolvedFigure(spec=spec, width=width, height=height, cells=cells, ink=ink_for(style, mode))


#: A panel shorter than this many font sizes cannot show two y tick labels.
_MIN_PANEL_EM = 2.6


def _warn_small_panels(cells: list[ResolvedCell], style: FigureStyle) -> None:
    small = []
    for rc in cells:
        for rp in rc.panels:
            if rp.y.visible and rp.panel.fixed_height is None and rp.rect[3] < style.font_size * _MIN_PANEL_EM:
                small.append(f"{rc.cell.letter or rp.index} ({rp.rect[3]:.0f} pt)")
    if small:
        warnings.warn(
            f"compose_figure: panels {', '.join(small)} are too short for {style.name!r} text "
            f"({style.font_size:g} pt); pass a larger size, fewer parts per column, or a smaller style",
            UserWarning,
            stacklevel=4,
        )


def _top_band(
    cell: Cell, style: FigureStyle, cw: float, mode: theme.ThemeMode | None = None
) -> tuple[float, list[LegendItem], float, float, bool]:
    """Height of the title/letter/legend band above a cell's panels, pt.

    Also returns the legend items, the first row's height, the letter's width
    and whether the legend needs a row of its own.
    """
    fs = style.font_size
    items = legend_items(cell, mode)
    if len(items) <= 1:
        items = []
    row1 = 0.0
    letter_w = 0.0
    if cell.letter:
        letter_w = text_width(cell.letter, style.panel_letter_size) + style.panel_letter_size * 0.5
        row1 = style.panel_letter_size * 1.2
    title_w = 0.0
    if cell.title:
        row1 = max(row1, cell.title.height(style.title_size))
        title_w = cell.title.width(style.title_size)
    leg_w = _legend_width(items, fs)
    legend_own_row = False
    if items:
        if row1 and letter_w + title_w + leg_w + fs * 2 > cw - 2 * _PAD:
            legend_own_row = True
        else:
            row1 = max(row1, fs * 1.5)
    top_used = _PAD + row1 + (fs * 1.6 if legend_own_row else 0.0)
    if top_used > _PAD:
        top_used += fs * 0.3
    return top_used, items, row1, letter_w, legend_own_row


def fitted_height(cell: Cell, style: FigureStyle, width: float) -> float:
    """Height, pt, of a cell whose panels all have a fixed height and hide their x axes.

    The title and legend band plus the panels plus the outer padding: no
    empty space below.
    """
    top_used, *_ = _top_band(cell, style, width)
    panels = cell.panels
    body = sum(p.fixed_height or 0.0 for p in panels) + sum(p.header_height for p in panels)
    return top_used + body + _PAD


def _fit_marks_to_width(panel: Panel, width: float) -> Panel:
    """Shrink a panel's ``fit_width`` text group (a sequence header) to fit ``width`` pt.

    Letters, spacing and tick marks scale together about the group's centre;
    a header above the axes (``yf == 1``) keeps its bottom edge and the
    panel's ``header_height`` shrinks with it. Returns the panel itself when
    the group already fits.
    """
    group = [m for m in panel.marks if isinstance(m, AxText | AxSegments) and m.fit_width]
    if not group:
        return panel
    x_lo, x_hi = math.inf, -math.inf
    dy_by_yf: dict[float, list[float]] = {}
    for m in group:
        if isinstance(m, AxText):
            w = m.text.width(m.size) * (1.06 if m.bold else 1.0)
            x = m.xf * width + m.dx
            x0 = x - (w if m.ha == "right" else w / 2 if m.ha == "center" else 0)
            x_lo, x_hi = min(x_lo, x0), max(x_hi, x0 + w)
            dy_by_yf.setdefault(m.yf, []).append(m.dy)
        else:
            for xf, yf, sx0, sy0, sx1, sy1 in m.segments:
                x_lo = min(x_lo, xf * width + sx0, xf * width + sx1)
                x_hi = max(x_hi, xf * width + sx0, xf * width + sx1)
                dy_by_yf.setdefault(yf, []).extend((sy0, sy1))
    avail = width - 2.0
    span = x_hi - x_lo
    if span <= avail or span <= 0:
        return panel
    k = avail / span
    cx = (x_lo + x_hi) / 2
    header = panel.header_height > 0
    # Header rows scale towards the axes' top edge; other rows towards their own middle.
    cy = {yf: 0.0 if (header and yf >= 1.0) else (min(v) + max(v)) / 2 for yf, v in dy_by_yf.items()}

    def sx(xf: float, d: float) -> float:
        return cx + (xf * width + d - cx) * k - xf * width

    def sy(yf: float, d: float) -> float:
        return cy[yf] + (d - cy[yf]) * k

    marks: list = []
    for m in panel.marks:
        if isinstance(m, AxText) and m.fit_width:
            marks.append(replace(m, size=m.size * k, dx=sx(m.xf, m.dx), dy=sy(m.yf, m.dy)))
        elif isinstance(m, AxSegments) and m.fit_width:
            segs = [(xf, yf, sx(xf, a), sy(yf, b), sx(xf, c), sy(yf, d)) for xf, yf, a, b, c, d in m.segments]
            marks.append(replace(m, segments=segs))
        else:
            marks.append(m)
    return replace(panel, marks=marks, header_height=panel.header_height * k if header else panel.header_height)


def _ytick_extent(rp: ResolvedPanel, fs: float) -> tuple[float, float, float]:
    """Widest y tick label, and how far the top and bottom labels reach past the panel, pt."""
    ya = rp.y
    ph = rp.rect[3]
    half = fs * 1.18 / 2
    w, above, below = 0.0, -math.inf, -math.inf
    if not (ya.visible and ya.show_ticklabels):
        return 0.0, -math.inf, -math.inf
    for v, t in zip(ya.ticks, ya.ticktext, strict=False):
        if not t or not ya.lo <= v <= ya.hi:
            continue
        pos = (v - ya.lo) / (ya.hi - ya.lo) * ph
        w = max(w, text_width(t, fs))
        above = max(above, pos + half - ph)
        below = max(below, half - pos)
    return w, above, below


def _prune_stack_ticks(upper: ResolvedPanel, lower: ResolvedPanel, fs: float) -> None:
    """Blank edge tick labels of two stacked panels that would run into each other.

    The lower panel's top label goes first, then the upper panel's bottom one.
    """
    half = fs * 1.18 / 2
    gap = fs * _TICK_GAP_EM

    def edge(rp: ResolvedPanel, top: bool) -> tuple[int, float] | None:
        ya = rp.y
        best: tuple[int, float] | None = None
        for i, (v, t) in enumerate(zip(ya.ticks, ya.ticktext, strict=False)):
            if not t or not ya.lo <= v <= ya.hi:
                continue
            y = rp.rect[1] + (v - ya.lo) / (ya.hi - ya.lo) * rp.rect[3]
            if best is None or (y > best[1] if top else y < best[1]):
                best = (i, y)
        return best

    if not (upper.y.visible and lower.y.visible):
        return
    for victim in (lower, upper, lower, upper):
        lo_edge, up_edge = edge(lower, top=True), edge(upper, top=False)
        if lo_edge is None or up_edge is None or up_edge[1] - half - gap >= lo_edge[1] + half:
            return
        i = lo_edge[0] if victim is lower else up_edge[0]
        texts = list(victim.y.ticktext)
        texts[i] = ""
        victim.y.ticktext = texts
        victim.y.auto_ticks = False


def _resolve_cell(
    cell: Cell,
    rect: tuple[float, float, float, float],
    style: FigureStyle,
    first_index: int,
    first_cell: bool,
    mode: theme.ThemeMode | None = None,
) -> ResolvedCell:
    cx, cy, cw, ch = rect
    fs = style.font_size
    ats = style.axis_title_size
    tick_pad = fs * 0.35
    tick_text_h = fs * 1.2
    title_gap = fs * 0.4

    # --- top furniture -----------------------------------------------------
    top_used, items, row1, letter_w, legend_own_row = _top_band(cell, style, cw, mode)
    band_gap = fs * 0.3 if top_used > _PAD else _PAD
    title = cell.title
    letter = cell.letter

    spec_panels = cell.panels

    # --- provisional ranges and tick widths ---------------------------------
    y_ranges = [_y_base_range(p) for p in spec_panels]
    x_ranges = [_x_range(p) for p in spec_panels]
    # Link shared x ranges to the first panel of the stack.
    for i, p in enumerate(spec_panels):
        if p.share_x and i > 0:
            lo = min(x_ranges[i - 1][0], x_ranges[i][0]) if p.x.lo is None else x_ranges[i][0]
            hi = max(x_ranges[i - 1][1], x_ranges[i][1]) if p.x.hi is None else x_ranges[i][1]
            x_ranges[i] = (lo, hi)
            for j in range(i):
                if spec_panels[j + 1].share_x or j == i - 1:
                    x_ranges[j] = (lo, hi)

    def ytick_width(p: Panel, rng: tuple[float, float], length: float) -> tuple[float, RichText | None]:
        lo, hi = rng
        if p.y.headroom:
            hi = hi * 1.3 if hi > 0 else hi
            if p.y.abs_ticklabels:
                lo = lo * 1.3 if lo < 0 else lo
        _, texts, ttl, _, _ = _axis_ticks(p.y, lo, hi, length, style, vertical=True)
        w = max((text_width(t, fs) for t in texts), default=0.0)
        return w, ttl

    header0 = spec_panels[0].header_height if spec_panels else 0.0
    est_h = max(20.0, (ch - top_used - header0 - 40.0) / max(len(spec_panels), 1))
    left_parts = []
    for p, rng in zip(spec_panels, y_ranges, strict=True):
        if not p.y.visible:
            left_parts.append(_PAD)
            continue
        w, ttl = ytick_width(p, rng, est_h)
        left_parts.append(
            _PAD + _axis_title_height(ttl, ats) + (title_gap if ttl else 0) + w + tick_pad + style.tick_length
        )
    left = max(left_parts, default=_PAD)

    right = _PAD
    last = spec_panels[-1] if spec_panels else None
    # Room for half of the last x tick label.
    if last is not None and last.x.visible:
        _, xtexts, _, _, xangle = _axis_ticks(last.x, *x_ranges[-1], cw - left - _PAD, style, vertical=False)
        if xtexts and not xangle:
            right = max(right, text_width(xtexts[-1], fs) / 2 + 1.0)
    for p, rng in zip(spec_panels, y_ranges, strict=True):
        if p.y_secondary is not None:
            sec_title, factor = p.y_secondary
            lo, hi = rng
            _, texts, _, _ = _resolve_ticks(
                Axis(scale_exponent=True, tick_min=p.y.tick_min, tick_max=p.y.tick_max),
                lo * factor,
                hi * factor,
                est_h,
                style.y_tick_spacing,
            )
            w = max((text_width(t, fs) for t in texts), default=0.0)
            right = max(right, _PAD + sec_title.height(ats) + title_gap + w + tick_pad + style.tick_length + 4)
        if p.colorbar is not None:
            cb = p.colorbar
            _, texts, _, _ = _resolve_ticks(Axis(), cb.lo, cb.hi if cb.hi > cb.lo else cb.lo + 1, est_h, 22.0)
            w = max((text_width(t, fs) for t in texts), default=0.0)
            right = max(right, _PAD + fs * 1.0 + 8.0 + tick_pad + 2.0 + w + title_gap + cb.title.height(fs) + 6)

    extra_left = 0.0
    extra_top = 0.0
    resolved: list[ResolvedPanel] = []
    plot_left = cx + left
    plot_w = max(cw - left - right, 10.0)
    # Margins come from estimates; resolve, measure what the ticks actually
    # need, and go round again (at most twice) if the estimate was short.
    for _attempt in range(3):
        cur_left = left + extra_left
        cur_top = top_used + extra_top
        plot_left = cx + cur_left
        plot_w = max(cw - cur_left - right, 10.0)
        panels = [_fit_marks_to_width(p, plot_w) for p in spec_panels]
        header_top = panels[0].header_height if panels else 0.0

        def bottom_space(i: int, panels: list[Panel] = panels, plot_w: float = plot_w) -> float:
            p = panels[i]
            if not p.x.visible:
                return 0.0
            _, texts, _, _, angle = _axis_ticks(p.x, *x_ranges[i], plot_w, style, vertical=False)
            text_h = max((text_width(t, fs) for t in texts), default=0.0) + 1.0 if angle else tick_text_h
            return text_h + tick_pad + style.tick_length + (title_gap + p.x.label.height(ats) if p.x.label else 0.0)

        bottom = _PAD + (bottom_space(len(panels) - 1) if panels else 0.0)

        # --- vertical stack -------------------------------------------------
        gaps = []
        for i in range(1, len(panels)):
            p = panels[i]
            g = _PANEL_GAP_SHARED if p.share_x else bottom_space(i - 1) + fs * 0.8
            gaps.append(g + p.header_height)
        avail = ch - cur_top - header_top - bottom - sum(gaps)
        fixed = sum(p.fixed_height or 0.0 for p in panels)
        weights = sum(p.weight for p in panels if p.fixed_height is None) or 1.0
        flex = max(avail - fixed, 10.0)
        heights = [p.fixed_height if p.fixed_height is not None else flex * p.weight / weights for p in panels]

        y_cursor = cy + ch - cur_top - header_top
        resolved = []
        for i, p in enumerate(panels):
            if i > 0:
                y_cursor -= gaps[i - 1]
            h = heights[i]
            prect = (plot_left, y_cursor - h, plot_w, h)
            y_cursor -= h
            index = first_index + i
            shows_x_labels = not (i + 1 < len(panels) and panels[i + 1].share_x)
            rp = _resolve_panel(p, index, prect, x_ranges[i], y_ranges[i], style, shows_x_labels)
            if p.share_x and i > 0:
                rp.shared_with = first_index + i - 1 if resolved[-1].shared_with is None else resolved[-1].shared_with
            resolved.append(rp)

        # --- measure ----------------------------------------------------------
        need_left = _PAD
        for rp in resolved:
            if not rp.y.visible:
                continue
            w, _, _ = _ytick_extent(rp, fs)
            size = rp.y.title_size or ats
            t_h = _axis_title_height(rp.y.title, size) + (title_gap if rp.y.title else 0.0)
            need_left = max(need_left, _PAD + t_h + w + tick_pad + style.tick_length)
        need_top = 0.0
        if resolved:
            _, above, _ = _ytick_extent(resolved[0], fs)
            need_top = above - header_top - band_gap
        grow_left = need_left - cur_left
        if grow_left <= 0.5 and need_top <= 0.5:
            break
        extra_left += max(grow_left, 0.0)
        extra_top += max(need_top, 0.0)

    for i in range(1, len(resolved)):
        if resolved[i].panel.share_x:
            _prune_stack_ticks(resolved[i - 1], resolved[i], fs)

    # --- furniture positions ------------------------------------------------
    top_y = cy + ch - _PAD
    letter_xy = (cx + _PAD, top_y) if letter else None
    title_xy = (cx + _PAD + letter_w, top_y) if title else None
    if letter and not title:
        title_xy = None
    if title and not letter:
        # Align the title with the plot's left edge when there is no letter.
        title_xy = (cx + _PAD, top_y)
    legend_xy = None
    if items:
        right_edge = plot_left + plot_w
        if legend_own_row:
            legend_xy = (right_edge, top_y - row1 - fs * 1.5)
        else:
            legend_xy = (right_edge, top_y - max(row1, fs * 1.5) + fs * 0.1)

    colorbar_rect = None
    for rp in resolved:
        if rp.panel.colorbar is not None:
            left, b, w, h = rp.rect
            colorbar_rect = (left + w + 8.0, b + h * 0.1, fs * 1.0, h * 0.8)

    return ResolvedCell(
        cell=cell,
        rect=rect,
        panels=resolved,
        title_xy=title_xy,
        letter_xy=letter_xy,
        legend=items,
        legend_xy=legend_xy,
        colorbar_rect=colorbar_rect,
    )


def _resolve_panel(
    panel: Panel,
    index: int,
    rect: tuple[float, float, float, float],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    style: FigureStyle,
    shows_x_labels: bool,
) -> ResolvedPanel:
    _, _, pw, ph = rect
    xlo, xhi = x_range
    ylo, yhi = y_range
    labelsets = [m for m in panel.marks if isinstance(m, LabelSet)]
    has_down = any(ls.direction == "down" for ls in labelsets)
    symmetric = panel.y.abs_ticklabels or has_down

    def fx(v):
        return (np.asarray(v, dtype=np.float64) - xlo) / (xhi - xlo) * pw

    reserved_base: list[tuple[float, float, float, float]] = []

    def run(
        lo: float,
        hi: float,
        stop_below: float = -math.inf,
        skip: set[tuple[int, int]] | None = None,
        dropped_keys: set[tuple[int, int]] | None = None,
    ) -> tuple[list[PlacedLabel], int, float, dict[int, tuple[float, float, str]]]:
        def fy(v):
            return (np.asarray(v, dtype=np.float64) - lo) / (hi - lo) * ph

        reserved = list(reserved_base)
        ref_pos: dict[int, tuple[float, float, str]] = {}
        # Reference-line labels sit at the top of the panel beside the line.
        for k, m in enumerate(panel.marks):
            if isinstance(m, RefLine) and m.label and m.orient == "v":
                size = style.label_size
                w, h = m.label.width(size), m.label.height(size)
                x = float(fx(m.value))
                gap = 2.0
                ha = "left"
                x0 = x + gap
                if x0 + w > pw:
                    ha = "right"
                    x0 = x - gap - w
                y1 = ph - 1.0
                reserved.append((x0, y1 - h, x0 + w, y1))
                ref_pos[k] = (gap if ha == "left" else -gap, -1.0, ha)
            elif isinstance(m, AxText) and 0 <= m.xf <= 1 and 0 <= m.yf <= 1:
                w, h = m.text.width(m.size), m.text.height(m.size)
                x = m.xf * pw + m.dx
                y = m.yf * ph + m.dy
                x0 = x - (w if m.ha == "right" else w / 2 if m.ha == "center" else 0)
                y0 = y - (h if m.va == "top" else h / 2 if m.va == "middle" else 0)
                reserved.append((x0, y0, x0 + w, y0 + h))
        if not labelsets:
            return [], 0, 0.0, ref_pos
        up = _stick_obstacles(panel, fx, fy, ph, flip=False)
        down = _stick_obstacles(panel, fx, fy, ph, flip=True) if has_down else None
        placed, dropped, score = place_labels(
            labelsets, fx, fy, pw, ph, up, down, reserved, stop_below, skip=skip, dropped_keys=dropped_keys
        )
        return placed, dropped, score, ref_pos

    candidates: list[tuple[float, float]] = []
    if panel.y.headroom and labelsets and panel.y.hi is None:
        for f in _HEADROOM_STEPS:
            hi = yhi * f if yhi > 0 else yhi
            lo = ylo * f if (symmetric and ylo < 0 and panel.y.lo is None) else ylo
            candidates.append((lo, hi))
    else:
        if panel.y.headroom and panel.y.hi is None and yhi > 0:
            yhi = yhi * _HEADROOM_STEPS[0]
            if symmetric and ylo < 0 and panel.y.lo is None:
                ylo = ylo * _HEADROOM_STEPS[0]
        candidates.append((ylo, yhi))

    # Pick the least headroom that places (nearly) as much as the most would.
    # The tallest candidate runs first to set the bar; the others stop as soon
    # as they cannot reach 97% of it, and once the bar is the total priority no
    # later candidate can raise it. Less headroom leaves less room, so a label
    # the tallest candidate had to drop is not searched for again.
    total = sum(
        float(np.nansum(np.clip(np.nan_to_num(np.asarray(ls.priority, dtype=np.float64)), 0, None))) for ls in labelsets
    )
    hopeless: set[tuple[int, int]] = set()
    first = run(*candidates[-1], stop_below=-math.inf, dropped_keys=hopeless)
    best = first[2]
    results: dict[int, tuple] = {len(candidates) - 1: (*candidates[-1], *first)}
    for i, (clo, chi) in enumerate(candidates[:-1]):
        res = run(clo, chi, stop_below=best * 0.97 - 1e-9, skip=hopeless)
        results[i] = (clo, chi, *res)
        best = max(best, res[2])
        if res[2] >= best * 0.97 - 1e-12 and best >= total - 1e-9:
            break
    ordered = [results[i] for i in sorted(results)]
    chosen = next((r for r in ordered if r[4] >= best * 0.97 - 1e-12), ordered[-1])
    lo, hi, placed, dropped, _score, ref_pos = chosen

    # --- axes ---------------------------------------------------------------
    xt, xtext, xtitle, xauto, xangle = _axis_ticks(panel.x, xlo, xhi, pw, style, vertical=False)
    yt, ytext, ytitle, yauto, _ = _axis_ticks(panel.y, lo, hi, ph, style, vertical=True)
    xtitle, xtitle_size = fit_axis_title(xtitle, pw, style.axis_title_size)
    ytitle, ytitle_size = fit_axis_title(ytitle, ph, style.axis_title_size)
    x_axis = ResolvedAxis(
        lo=xlo,
        hi=xhi,
        ticks=xt,
        ticktext=xtext,
        title=xtitle if shows_x_labels else None,
        show_ticklabels=shows_x_labels,
        grid=bool(panel.x.grid) if panel.x.grid is not None else False,
        zeroline=panel.x.zeroline,
        visible=panel.x.visible,
        auto_ticks=xauto and style.name == "screen",
        title_size=xtitle_size,
        tick_angle=xangle,
    )
    y_axis = ResolvedAxis(
        lo=lo,
        hi=hi,
        ticks=yt,
        ticktext=ytext,
        title=ytitle,
        grid=style.grid if panel.y.grid is None else panel.y.grid,
        zeroline=panel.y.zeroline,
        visible=panel.y.visible,
        auto_ticks=yauto and style.name == "screen" and panel.y.tick_max is None,
        title_size=ytitle_size,
    )
    y2 = None
    if panel.y_secondary is not None:
        sec_title, factor = panel.y_secondary
        tick_lo = lo if panel.y.tick_min is None else max(lo, panel.y.tick_min)
        tick_hi = hi if panel.y.tick_max is None else min(hi, panel.y.tick_max)
        vals, step = tick_values(
            tick_lo * factor, tick_hi * factor, ph * (tick_hi - tick_lo) / (hi - lo), style.y_tick_spacing
        )
        shown = [abs(v) for v in vals] if panel.y.abs_ticklabels else vals
        top = max((abs(v) for v in vals), default=0.0)
        title2 = sec_title
        exponent = math.floor(math.log10(top)) if top >= 1e4 else 0
        if exponent:
            texts = [format_tick(v / 10**exponent, step / 10**exponent) for v in shown]
            title2 = sec_title + RichText(((" (\u00d710", "n"), (str(exponent), "sup"), (")", "n")))
        else:
            texts = [format_tick(v, step) for v in shown]
        y2 = ResolvedAxis(
            lo=lo,
            hi=hi,
            ticks=[v / factor for v in vals],
            ticktext=texts,
            title=title2,
        )

    return ResolvedPanel(
        panel=panel,
        index=index,
        rect=rect,
        x=x_axis,
        y=y_axis,
        y2=y2,
        labels=placed,
        dropped=dropped,
        refline_labels=ref_pos,
    )


@dataclass
class TextBox:
    """One piece of text in a resolved figure, as the layout models it."""

    kind: str  #: title, letter, legend, xtick, ytick, xtitle, ytitle, y2tick, y2title, label, text, refline
    text: str
    box: tuple[float, float, float, float]  #: x0, y0, x1, y1 in figure pt (y up)
    panel: int | None = None  #: 1-based panel index, None for cell furniture


def _tick_label_widths(ax: ResolvedAxis, size: float) -> list[float]:
    return [text_width(t, size) for t, v in zip(ax.ticktext, ax.ticks, strict=False) if ax.lo <= v <= ax.hi]


def text_boxes(fig: ResolvedFigure) -> list[TextBox]:
    """Every text box the layout placed, in figure pt: titles, letters, legends,
    tick labels, axis titles, peak labels, panel text and reference-line labels.

    The boxes use the same metrics the layout used to make room (line-box
    heights, estimated widths), so a test can check that nothing overlaps and
    nothing leaves the figure without rendering pixels. Colour bars are not
    included.
    """
    style = fig.style
    fs = style.font_size
    tick_off = style.tick_length + fs * 0.35
    title_gap = fs * 0.4
    tick_h = fs * 1.18
    out: list[TextBox] = []
    for cell in fig.cells:
        if cell.letter_xy is not None and cell.cell.letter:
            lx, ly = cell.letter_xy
            w = text_width(RichText((((cell.cell.letter), "bf"),)), style.panel_letter_size)
            out.append(TextBox("letter", cell.cell.letter, (lx, ly - style.panel_letter_size * 1.18, lx + w, ly)))
        if cell.title_xy is not None and cell.cell.title:
            tx, ty = cell.title_xy
            t = cell.cell.title
            out.append(
                TextBox("title", t.text, (tx, ty - t.height(style.title_size), tx + t.width(style.title_size), ty))
            )
        if cell.legend and cell.legend_xy is not None:
            rx, by = cell.legend_xy
            w = _legend_width(cell.legend, fs)
            out.append(TextBox("legend", " ".join(i.name for i in cell.legend), (rx - w, by, rx, by + tick_h)))
        for rp in cell.panels:
            left, bottom, pw, ph = rp.rect
            idx = rp.index
            xa, ya = rp.x, rp.y
            # y tick labels and title (left side).
            ytw = 0.0
            if ya.visible and ya.show_ticklabels:
                for v, t in zip(ya.ticks, ya.ticktext, strict=False):
                    if not ya.lo <= v <= ya.hi or not t:
                        continue
                    y = bottom + (v - ya.lo) / (ya.hi - ya.lo) * ph
                    w = text_width(t, fs)
                    ytw = max(ytw, w)
                    x1 = left - tick_off
                    out.append(TextBox("ytick", t, (x1 - w, y - tick_h / 2, x1, y + tick_h / 2), idx))
            if ya.visible and ya.title:
                size = ya.title_size or style.axis_title_size
                h, w = ya.title.width(size), ya.title.height(size)
                x1 = left - tick_off - ytw - title_gap
                yc = bottom + ph / 2
                out.append(TextBox("ytitle", ya.title.text, (x1 - w, yc - h / 2, x1, yc + h / 2), idx))
            # x tick labels and title (below).
            xth = 0.0
            if xa.visible and xa.show_ticklabels:
                rot = abs(xa.tick_angle) % 180 == 90
                for v, t in zip(xa.ticks, xa.ticktext, strict=False):
                    if not xa.lo <= v <= xa.hi or not t:
                        continue
                    x = left + (v - xa.lo) / (xa.hi - xa.lo) * pw
                    w = text_width(t, fs)
                    bw, bh = (tick_h, w) if rot else (w, tick_h)
                    xth = max(xth, bh)
                    y1 = bottom - tick_off
                    out.append(TextBox("xtick", t, (x - bw / 2, y1 - bh, x + bw / 2, y1), idx))
            if xa.visible and xa.title:
                size = xa.title_size or style.axis_title_size
                w, h = xa.title.width(size), xa.title.height(size)
                y1 = bottom - tick_off - max(xth, tick_h) - title_gap
                xc = left + pw / 2
                out.append(TextBox("xtitle", xa.title.text, (xc - w / 2, y1 - h, xc + w / 2, y1), idx))
            if rp.y2 is not None:
                y2 = rp.y2
                w2 = 0.0
                for v, t in zip(y2.ticks, y2.ticktext, strict=False):
                    if not y2.lo <= v <= y2.hi or not t:
                        continue
                    y = bottom + (v - y2.lo) / (y2.hi - y2.lo) * ph
                    w = text_width(t, fs)
                    w2 = max(w2, w)
                    x0 = left + pw + tick_off
                    out.append(TextBox("y2tick", t, (x0, y - tick_h / 2, x0 + w, y + tick_h / 2), idx))
                if y2.title:
                    h, w = y2.title.width(style.axis_title_size), y2.title.height(style.axis_title_size)
                    x0 = left + pw + tick_off + w2 + title_gap
                    yc = bottom + ph / 2
                    out.append(TextBox("y2title", y2.title.text, (x0, yc - h / 2, x0 + w, yc + h / 2), idx))
            # Peak labels.
            for lab in rp.labels:
                x0, y0, x1, y1 = lab.box
                out.append(TextBox("label", lab.text.text, (left + x0, bottom + y0, left + x1, bottom + y1), idx))
            # Panel text and reference-line labels.
            for k, m in enumerate(rp.panel.marks):
                if isinstance(m, AxText) and m.text:
                    w, h = m.text.width(m.size), m.text.height(m.size)
                    if m.bold:
                        w *= 1.06
                    x = left + m.xf * pw + m.dx
                    y = bottom + m.yf * ph + m.dy
                    x0 = x - (w if m.ha == "right" else w / 2 if m.ha == "center" else 0)
                    y0 = y - (h if m.va == "top" else h / 2 if m.va == "middle" else 0)
                    out.append(TextBox("text", m.text.text, (x0, y0, x0 + w, y0 + h), idx))
                elif isinstance(m, RefLine) and m.label and k in rp.refline_labels:
                    dx, dy, ha = rp.refline_labels[k]
                    size = style.label_size
                    w, h = m.label.width(size), m.label.height(size)
                    x = left + (m.value - xa.lo) / (xa.hi - xa.lo) * pw + dx
                    x0 = x if ha == "left" else x - w
                    y1 = bottom + ph + dy
                    out.append(TextBox("refline", m.label.text, (x0, y1 - h, x0 + w, y1), idx))
    return out


def text_problems(fig: ResolvedFigure, tol: float = 0.25) -> list[str]:
    """Overlapping text boxes, boxes outside the figure and peak labels on a y spine, described for a test failure."""
    boxes = text_boxes(fig)
    problems: list[str] = []
    for b in boxes:
        x0, y0, x1, y1 = b.box
        if x0 < -tol or y0 < -tol or x1 > fig.width + tol or y1 > fig.height + tol:
            problems.append(f"outside: {b.kind} {b.text!r} (panel {b.panel}) {tuple(round(v, 1) for v in b.box)}")
    for rp in fig.panels:
        w = rp.rect[2]
        for lab in rp.labels:
            x0, _, x1, _ = lab.box
            if x0 < _SPINE_CLEAR - tol or x1 > w - _SPINE_CLEAR + tol:
                problems.append(
                    f"spine: label {lab.text.text!r} (panel {rp.index}) x {x0:.1f}-{x1:.1f} of plot width {w:.1f}"
                )
    arr = np.array([b.box for b in boxes]) if boxes else np.zeros((0, 4))
    for i in range(len(boxes)):
        a = arr[i]
        hit = (
            (arr[i + 1 :, 0] < a[2] - tol)
            & (arr[i + 1 :, 2] > a[0] + tol)
            & (arr[i + 1 :, 1] < a[3] - tol)
            & (arr[i + 1 :, 3] > a[1] + tol)
        )
        for j in np.flatnonzero(hit):
            o = boxes[i + 1 + int(j)]
            problems.append(
                f"overlap: {boxes[i].kind} {boxes[i].text!r} (panel {boxes[i].panel}) "
                f"x {o.kind} {o.text!r} (panel {o.panel})"
            )
    return problems


def label_boxes(panel: ResolvedPanel) -> list[tuple[float, float, float, float]]:
    """Placed label boxes in panel pt coordinates (for tests)."""
    return [lab.box for lab in panel.labels]


__all__ = [
    "Ink",
    "PlacedLabel",
    "ResolvedAxis",
    "ResolvedCell",
    "ResolvedFigure",
    "ResolvedPanel",
    "TextBox",
    "fit_axis_title",
    "fit_tick_labels",
    "format_tick",
    "label_boxes",
    "nice_step",
    "place_labels",
    "resolve_figure",
    "text_boxes",
    "text_problems",
    "tick_values",
]
