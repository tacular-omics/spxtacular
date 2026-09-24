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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from . import theme
from ._text import RichText, text_width
from .figspec import (
    Axis,
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


def tick_values(lo: float, hi: float, length_pt: float, spacing_pt: float) -> tuple[list[float], float]:
    """Round tick positions inside ``[lo, hi]`` about ``spacing_pt`` apart."""
    target = max(1.0, length_pt / max(spacing_pt, 1.0))
    ticks: list[float] = []
    step = 1.0
    # A value axis with only "0" and one other label is hard to read: densify
    # (at most a few times) until there are at least three ticks.
    for _ in range(4):
        step = nice_step(hi - lo, target)
        ticks = []
        v = math.ceil(lo / step - 1e-9) * step
        while v <= hi + step * 1e-9:
            ticks.append(round(v, 12) + 0.0)
            v += step
        if len(ticks) >= 3:
            break
        target *= 1.6
    return ticks, step


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
    ax: Axis, lo: float, hi: float, length_pt: float, spacing: float
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
    values, step = tick_values(tlo, thi, length_pt * (thi - tlo) / max(hi - lo, 1e-300), spacing)
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


# ---------------------------------------------------------------------------
# Label placement
# ---------------------------------------------------------------------------


@dataclass
class _Obstacles:
    #: Vertical segments: x, lo, hi (sorted by x), pt, in the canonical frame.
    sx: NDArray[np.float64]
    slo: NDArray[np.float64]
    shi: NDArray[np.float64]
    boxes: list[tuple[float, float, float, float]]


def _seg_boxes_hit(x0: float, y0: float, x1: float, y1: float, boxes: NDArray[np.float64]) -> bool:
    """Does the segment cross any box? Liang-Barsky, vectorised over boxes."""
    if len(boxes) == 0:
        return False
    dx, dy = x1 - x0, y1 - y0
    t0 = np.zeros(len(boxes))
    t1 = np.ones(len(boxes))
    ok = np.ones(len(boxes), dtype=bool)
    for p, q in (
        (-dx, x0 - boxes[:, 0]),
        (dx, boxes[:, 2] - x0),
        (-dy, y0 - boxes[:, 1]),
        (dy, boxes[:, 3] - y0),
    ):
        if p == 0:
            ok &= q >= 0
            continue
        r = q / p
        if p < 0:
            t0 = np.maximum(t0, r)
        else:
            t1 = np.minimum(t1, r)
    return bool(np.any(ok & (t0 <= t1)))


def _segs_cross(p: tuple[float, float, float, float], segs: NDArray[np.float64]) -> bool:
    """Does segment ``p`` properly cross any of ``segs`` (N x 4)?"""
    if len(segs) == 0:
        return False
    ax, ay, bx, by = p
    cx, cy, dx, dy = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]

    def orient(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)

    o1 = orient(ax, ay, bx, by, cx, cy)
    o2 = orient(ax, ay, bx, by, dx, dy)
    o3 = orient(cx, cy, dx, dy, ax, ay)
    o4 = orient(cx, cy, dx, dy, bx, by)
    return bool(np.any((o1 * o2 < 0) & (o3 * o4 < 0)))


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


def place_labels(
    labelsets: Sequence[LabelSet],
    to_pt_x,
    to_pt_y,
    width: float,
    height: float,
    obstacles_up: _Obstacles,
    obstacles_down: _Obstacles | None,
    reserved: list[tuple[float, float, float, float]],
) -> tuple[list[PlacedLabel], int, float]:
    """Greedy, priority-ordered placement. Returns labels, dropped count, placed priority.

    Works in a canonical "up" frame: for a downward label set (the lower half of
    a mirror plot) the panel is flipped so the same search applies.
    """
    items: list[tuple[float, int, int]] = []
    for s_i, ls in enumerate(labelsets):
        for j in range(len(ls.texts)):
            if not ls.texts[j]:
                continue
            items.append((float(ls.priority[j]), s_i, j))
    items.sort(key=lambda t: -t[0])

    placed: list[PlacedLabel] = []
    dropped = 0
    score = 0.0
    # Boxes and leaders in the *panel* frame (y up from the panel bottom).
    box_arr = np.zeros((len(items) + len(reserved) + 1, 4))
    n_boxes = 0
    for b in reserved:
        box_arr[n_boxes] = b
        n_boxes += 1
    leaders = np.zeros((len(items) + 1, 4))
    n_leaders = 0
    cand_cache: dict[tuple[float, float], list[tuple[float, float, bool, float]]] = {}

    for prio, s_i, j in items:
        ls = labelsets[s_i]
        down = ls.direction == "down"
        obst = obstacles_down if (down and obstacles_down is not None) else obstacles_up
        text = ls.texts[j]
        tw, th = text.width(ls.size), text.height(ls.size)
        if ls.rotation % 180:
            tw, th = th, tw
        ax_pt = to_pt_x(float(ls.x[j]))
        ay_panel = to_pt_y(float(ls.y[j]))
        if not (math.isfinite(ax_pt) and math.isfinite(ay_panel)):
            dropped += 1
            continue
        ay = height - ay_panel if down else ay_panel
        off = float(ls.anchor_offset[j]) if ls.anchor_offset is not None else 0.0
        base = ay + off + ls.gap
        key = (round(tw, 2), round(th, 2))
        cands = cand_cache.get(key)
        if cands is None:
            cands = _candidates(tw, th)
            cand_cache[key] = cands
        if not ls.leaders:
            cands = [c for c in cands if not c[2]]

        chosen = None
        for dx, dy, leader, _cost in cands:
            cx = ax_pt + dx
            x0, x1 = cx - tw / 2, cx + tw / 2
            y0, y1 = base + dy, base + dy + th
            if x0 < 0 or x1 > width or y1 > height or y0 < 0:
                continue
            # Frame conversion for the shared (panel-frame) box list.
            py0, py1 = (height - y1, height - y0) if down else (y0, y1)
            pad = 0.6
            if n_boxes:
                b = box_arr[:n_boxes]
                if np.any((b[:, 0] < x1 + pad) & (b[:, 2] > x0 - pad) & (b[:, 1] < py1 + pad) & (b[:, 3] > py0 - pad)):
                    continue
            # Sticks under the box.
            i0 = int(np.searchsorted(obst.sx, x0 - pad))
            i1 = int(np.searchsorted(obst.sx, x1 + pad, side="right"))
            if i1 > i0 and np.any((obst.slo[i0:i1] < y1) & (obst.shi[i0:i1] > y0 - 0.3)):
                continue
            if obst.boxes:
                ob = np.asarray(obst.boxes)
                if np.any((ob[:, 0] < x1) & (ob[:, 2] > x0) & (ob[:, 1] < y1) & (ob[:, 3] > y0)):
                    continue
            # Existing leaders through this box.
            if n_leaders and _seg_boxes_hit_any(leaders[:n_leaders], (x0, py0, x1, py1)):
                continue
            seg = None
            if leader:
                lx = min(max(ax_pt, x0 + 1.0), x1 - 1.0)
                l_start = (ax_pt, ay + off + ls.gap * 0.35)
                l_end = (lx, y0 - 0.4)
                if l_end[1] - l_start[1] < 1.0 and abs(l_end[0] - l_start[0]) < 1.0:
                    leader = False
                else:
                    p0 = (l_start[0], height - l_start[1]) if down else l_start
                    p1 = (l_end[0], height - l_end[1]) if down else l_end
                    if n_boxes and _seg_boxes_hit(p0[0], p0[1], p1[0], p1[1], box_arr[:n_boxes]):
                        continue
                    if n_leaders and _segs_cross((p0[0], p0[1], p1[0], p1[1]), leaders[:n_leaders]):
                        continue
                    # Leader crossing a stick that stands taller than it.
                    lo_x, hi_x = min(l_start[0], l_end[0]), max(l_start[0], l_end[0])
                    k0 = int(np.searchsorted(obst.sx, lo_x - 0.3))
                    k1 = int(np.searchsorted(obst.sx, hi_x + 0.3, side="right"))
                    if k1 > k0:
                        sx = obst.sx[k0:k1]
                        if hi_x - lo_x > 1e-6:
                            ly = l_start[1] + (sx - l_start[0]) / (l_end[0] - l_start[0]) * (l_end[1] - l_start[1])
                            own = np.abs(sx - ax_pt) < 1e-6
                            if np.any(~own & (obst.shi[k0:k1] > ly - 0.3) & (obst.slo[k0:k1] < ly)):
                                continue
                        else:
                            own = np.abs(sx - ax_pt) < 1e-6
                            if np.any(~own & (obst.shi[k0:k1] > l_start[1])):
                                continue
                    seg = (p0[0], p0[1], p1[0], p1[1])
            chosen = (cx, y0, y1, x0, x1, py0, py1, leader, seg)
            break

        if chosen is None:
            dropped += 1
            continue
        cx, y0, y1, x0, x1, py0, py1, leader, seg = chosen
        box_arr[n_boxes] = (x0, py0, x1, py1)
        n_boxes += 1
        rel_leader = None
        if seg is not None:
            leaders[n_leaders] = seg
            n_leaders += 1
            rel_leader = (seg[0] - ax_pt, seg[1] - ay_panel, seg[2] - ax_pt, seg[3] - ay_panel)
        if down:
            dy_rel = (height - y0) - ay_panel  # top edge of the box, panel frame
            va: Literal["bottom", "top"] = "top"
        else:
            dy_rel = y0 - ay_panel
            va = "bottom"
        placed.append(
            PlacedLabel(
                x=float(ls.x[j]),
                y=float(ls.y[j]),
                text=text,
                color=ls.colors[j],
                size=ls.size,
                dx=cx - ax_pt,
                dy=dy_rel,
                va=va,
                rotation=ls.rotation,
                leader=rel_leader,
                name=ls.name,
                box=(x0, py0, x1, py1),
            )
        )
        score += prio
    return placed, dropped, score


def _seg_boxes_hit_any(segs: NDArray[np.float64], box: tuple[float, float, float, float]) -> bool:
    """Does any segment cross ``box``?"""
    x0, y0, x1, y1 = box
    arr = np.array([[x0, y0, x1, y1]])
    return any(_seg_boxes_hit(float(s[0]), float(s[1]), float(s[2]), float(s[3]), arr) for s in segs)


def _stick_obstacles(panel: Panel, to_pt_x, to_pt_y, height: float, flip: bool) -> _Obstacles:
    xs: list[NDArray[np.float64]] = []
    los: list[NDArray[np.float64]] = []
    his: list[NDArray[np.float64]] = []
    boxes: list[tuple[float, float, float, float]] = []
    for m in panel.marks:
        if isinstance(m, Sticks) and m.obstacle and len(m.x):
            x = to_pt_x(np.asarray(m.x, dtype=np.float64))
            a = to_pt_y(np.full(len(m.x), m.base))
            b = to_pt_y(np.asarray(m.y, dtype=np.float64))
            xs.append(x)
            los.append(np.minimum(a, b))
            his.append(np.maximum(a, b))
        elif isinstance(m, Line) and len(m.x):
            x = to_pt_x(np.asarray(m.x, dtype=np.float64))
            b = to_pt_y(np.asarray(m.y, dtype=np.float64))
            a = to_pt_y(np.zeros(len(m.x))) if m.fill else b - 0.5
            xs.append(x)
            los.append(np.minimum(a, b))
            his.append(np.maximum(a, b))
        elif isinstance(m, Points) and len(m.x):
            x = to_pt_x(np.asarray(m.x, dtype=np.float64))
            y = to_pt_y(np.asarray(m.y, dtype=np.float64))
            r = np.asarray(m.sizes, dtype=np.float64) / 2
            for xi, yi, ri in zip(x, y, r, strict=True):
                if math.isfinite(xi) and math.isfinite(yi):
                    boxes.append((xi - ri, yi - ri, xi + ri, yi + ri))
        elif isinstance(m, Bars) and len(m.x):
            for xi, hi in zip(np.asarray(m.x, dtype=np.float64), np.asarray(m.height, dtype=np.float64), strict=True):
                a, b = to_pt_x(xi - m.width / 2), to_pt_x(xi + m.width / 2)
                y0, y1 = to_pt_y(0.0), to_pt_y(hi)
                boxes.append((min(a, b), min(y0, y1), max(a, b), max(y0, y1)))
    if xs:
        sx = np.concatenate(xs)
        slo = np.concatenate(los)
        shi = np.concatenate(his)
        ok = np.isfinite(sx) & np.isfinite(slo) & np.isfinite(shi)
        sx, slo, shi = sx[ok], slo[ok], shi[ok]
    else:
        sx = slo = shi = np.zeros(0)
    if flip:
        slo, shi = height - shi, height - slo
        boxes = [(b[0], height - b[3], b[2], height - b[1]) for b in boxes]
    order = np.argsort(sx, kind="stable")
    return _Obstacles(sx[order], slo[order], shi[order], boxes)


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


def legend_items(cell: Cell) -> list[LegendItem]:
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
                items.append(LegendItem(name, m.colors[0] if m.colors else "#000000", "marker", 0.0))
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
        prefs.append(max(c.aspect * cell_w + c.extra_height_mm * PT_PER_MM for c in row))
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
            rc = _resolve_cell(spec.cells[k], rect, style, panel_index, first_cell=(k == 0))
            panel_index += len(rc.panels)
            cells.append(rc)
        top -= row_h[r]
    return ResolvedFigure(spec=spec, width=width, height=height, cells=cells, ink=ink_for(style, mode))


def _resolve_cell(
    cell: Cell,
    rect: tuple[float, float, float, float],
    style: FigureStyle,
    first_index: int,
    first_cell: bool,
) -> ResolvedCell:
    cx, cy, cw, ch = rect
    fs = style.font_size
    ats = style.axis_title_size
    tick_pad = fs * 0.35
    tick_text_h = fs * 1.2
    title_gap = fs * 0.4

    # --- top furniture -----------------------------------------------------
    items = legend_items(cell)
    show_legend = len(items) > 1
    if not show_legend:
        items = []
    title = cell.title
    letter = cell.letter
    row1 = 0.0
    letter_w = 0.0
    if letter:
        letter_w = text_width(letter, style.panel_letter_size) + style.panel_letter_size * 0.5
        row1 = style.panel_letter_size * 1.2
    title_w = 0.0
    if title:
        row1 = max(row1, title.height(style.title_size))
        title_w = title.width(style.title_size)
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

    panels = cell.panels
    header_top = panels[0].header_height if panels else 0.0

    # --- provisional ranges and tick widths ---------------------------------
    y_ranges = [_y_base_range(p) for p in panels]
    x_ranges = [_x_range(p) for p in panels]
    # Link shared x ranges to the first panel of the stack.
    for i, p in enumerate(panels):
        if p.share_x and i > 0:
            lo = min(x_ranges[i - 1][0], x_ranges[i][0]) if p.x.lo is None else x_ranges[i][0]
            hi = max(x_ranges[i - 1][1], x_ranges[i][1]) if p.x.hi is None else x_ranges[i][1]
            x_ranges[i] = (lo, hi)
            for j in range(i):
                if panels[j + 1].share_x or j == i - 1:
                    x_ranges[j] = (lo, hi)

    def ytick_width(p: Panel, rng: tuple[float, float], length: float) -> tuple[float, RichText | None]:
        lo, hi = rng
        if p.y.headroom:
            hi = hi * 1.3 if hi > 0 else hi
            if p.y.abs_ticklabels:
                lo = lo * 1.3 if lo < 0 else lo
        _, texts, ttl, _ = _resolve_ticks(p.y, lo, hi, length, style.y_tick_spacing)
        w = max((text_width(t, fs) for t in texts), default=0.0)
        return w, ttl

    est_h = max(20.0, (ch - top_used - header_top - 40.0) / max(len(panels), 1))
    left_parts = []
    for p, rng in zip(panels, y_ranges, strict=True):
        if not p.y.visible:
            left_parts.append(_PAD)
            continue
        w, ttl = ytick_width(p, rng, est_h)
        left_parts.append(
            _PAD + _axis_title_height(ttl, ats) + (title_gap if ttl else 0) + w + tick_pad + style.tick_length
        )
    left = max(left_parts, default=_PAD)

    right = _PAD
    last = panels[-1] if panels else None
    # Room for half of the last x tick label.
    if last is not None and last.x.visible:
        _, xtexts, _, _ = _resolve_ticks(last.x, *x_ranges[-1], cw - left - _PAD, style.x_tick_spacing)
        if xtexts:
            right = max(right, text_width(xtexts[-1], fs) / 2 + 1.0)
    for p, rng in zip(panels, y_ranges, strict=True):
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

    def bottom_space(p: Panel) -> float:
        if not p.x.visible:
            return 0.0
        return tick_text_h + tick_pad + style.tick_length + (title_gap + p.x.label.height(ats) if p.x.label else 0.0)

    bottom = _PAD + (bottom_space(last) if last is not None else 0.0)

    # --- vertical stack -----------------------------------------------------
    gaps = []
    for i in range(1, len(panels)):
        p = panels[i]
        g = _PANEL_GAP_SHARED if p.share_x else bottom_space(panels[i - 1]) + fs * 0.8
        gaps.append(g + p.header_height)
    avail = ch - top_used - header_top - bottom - sum(gaps)
    fixed = sum(p.fixed_height or 0.0 for p in panels)
    weights = sum(p.weight for p in panels if p.fixed_height is None) or 1.0
    flex = max(avail - fixed, 10.0)
    heights = [p.fixed_height if p.fixed_height is not None else flex * p.weight / weights for p in panels]

    plot_left = cx + left
    plot_w = max(cw - left - right, 10.0)
    y_cursor = cy + ch - top_used - header_top
    resolved: list[ResolvedPanel] = []
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

    def run(lo: float, hi: float) -> tuple[list[PlacedLabel], int, float, dict[int, tuple[float, float, str]]]:
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
            labelsets, lambda v: float(fx(v)), lambda v: float(fy(v)), pw, ph, up, down, reserved
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

    results = [(lo, hi, *run(lo, hi)) for lo, hi in candidates]
    best = max(r[4] for r in results)
    chosen = next(r for r in results if r[4] >= best * 0.97 - 1e-12)
    lo, hi, placed, dropped, _score, ref_pos = chosen

    # --- axes ---------------------------------------------------------------
    xt, xtext, xtitle, xauto = _resolve_ticks(panel.x, xlo, xhi, pw, style.x_tick_spacing)
    yt, ytext, ytitle, yauto = _resolve_ticks(panel.y, lo, hi, ph, style.y_tick_spacing)
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
    "format_tick",
    "label_boxes",
    "nice_step",
    "place_labels",
    "resolve_figure",
    "tick_values",
]
