"""
Backend-neutral figure specifications.

Every spxtacular plot is built in three steps:

1. **Content.** A plotting function turns spectra into a :class:`FigureSpec`:
   panels, axes and marks (sticks, lines, points, labels) with data-space
   coordinates and theme colours. Nothing here knows which engine will draw it.
2. **Layout.** :mod:`spxtacular._layout` resolves the spec at a concrete size in
   points: margins from the real text extents, "nice" ticks, aligned panels,
   and collision-free label positions with leader lines.
3. **Drawing.** A backend (plotly or matplotlib) draws the resolved spec. The
   backends only translate; they make no layout decisions, so the two cannot
   drift apart.

Pass ``backend="spec"`` to any plotting function to get the spec itself, edit
it, then :func:`render` it -- or hand several to :func:`compose_figure` to build
a multi-panel figure with a/b/c labels.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from . import theme
from ._text import RichText
from .errors import SpxtacularError
from .style import PT_PER_MM, FigureStyle, SizeLike, StyleName, resolve_size, resolve_style

if TYPE_CHECKING:
    from ._layout import ResolvedFigure

Backend = Literal["plotly", "matplotlib", "spec"]
BACKENDS: tuple[str, ...] = ("plotly", "matplotlib", "spec")


# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------


@dataclass
class Sticks:
    """Vertical sticks from ``base`` to ``y`` at each ``x``: the centroid-spectrum mark."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    color: str
    width: float  #: pt
    opacity: float = 1.0
    dash: str = "solid"
    name: str | None = None
    legend: bool = False
    #: Plotly hover. ``None`` hands hovering to a :class:`HitLayer`.
    customdata: Sequence[Any] | None = None
    hovertemplate: str | None = None
    base: float = 0.0
    #: How labels treat these sticks: ``"hard"`` never covered, ``"soft"``
    #: (context such as unmatched peaks) avoided but covered, on a background
    #: patch, when nothing else fits, ``"none"`` ignored.
    obstacle: Literal["hard", "soft", "none"] = "hard"


@dataclass
class Line:
    """A connected trace, optionally filled to zero."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    color: str
    width: float
    fill: bool = False
    fill_alpha: float = 0.12
    dash: str = "solid"
    name: str | None = None
    legend: bool = False
    customdata: Sequence[Any] | None = None
    hovertemplate: str | None = None


@dataclass
class Points:
    """Filled circles, e.g. mass-error dots. ``sizes`` are diameters in pt."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    sizes: NDArray[np.float64]
    colors: list[str]
    outline: str | None = None
    outline_width: float = 0.4
    opacity: float = 0.85
    name: str | None = None
    legend: bool = False
    customdata: Sequence[Any] | None = None
    hovertemplate: str | None = None


@dataclass
class Bars:
    """Vertical bars of data-space ``width`` centred on ``x``."""

    x: NDArray[np.float64]
    height: NDArray[np.float64]
    width: float
    colors: list[str]
    name: str | None = None
    customdata: Sequence[Any] | None = None
    hovertemplate: str | None = None


@dataclass
class HitLayer:
    """Invisible hover targets at peak tips. Plotly only; ignored for static output."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    hover: list[str]
    size_px: float = 22.0


@dataclass
class RefLine:
    """A reference line across the panel, e.g. the precursor m/z or zero error."""

    orient: Literal["v", "h"]
    value: float
    color: str
    width: float
    dash: str = "solid"
    label: RichText | None = None
    label_color: str | None = None
    #: Plotly annotation ``name``, so tests and users can find it.
    name: str = "refline"


@dataclass
class Band:
    """A shaded interval, e.g. an isolation window."""

    orient: Literal["v", "h"]
    lo: float
    hi: float
    color: str
    alpha: float = 0.08


@dataclass
class LabelSet:
    """Labels to place without collisions. Positions are chosen at layout time.

    ``x``/``y`` are the data-space anchors (a peak tip, a dot centre). The
    placer tries the spot straight above (or below, for ``direction="down"``),
    then nearby spots joined back by a leader line, and drops a label only when
    nothing fits. Higher ``priority`` is placed first.
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    texts: list[RichText]
    colors: list[str]
    priority: NDArray[np.float64]
    size: float  #: pt
    direction: Literal["up", "down"] = "up"
    #: Degrees. 0 horizontal, 90 reads bottom-to-top.
    rotation: float = 0.0
    #: Distance from the anchor to where a label may start, pt (e.g. a dot's radius).
    anchor_offset: NDArray[np.float64] | None = None
    gap: float = 1.5
    #: Allow leader lines. Off means a label either fits straight above or is dropped.
    leaders: bool = True
    name: str = "label"
    #: Per-label font size, pt. NaN (or ``None`` for all) uses ``size``.
    sizes: NDArray[np.float64] | None = None
    #: Per-label rotation, degrees. NaN (or ``None`` for all) uses ``rotation``.
    rotations: NDArray[np.float64] | None = None


@dataclass
class AxText:
    """Text anchored to a panel in axes-fraction coordinates plus a pt offset.

    Used for furniture outside the data space: the sequence header, the halves'
    names in a mirror plot, a similarity score in a corner.
    """

    xf: float
    yf: float
    text: RichText
    size: float
    color: str
    dx: float = 0.0
    dy: float = 0.0
    ha: Literal["left", "center", "right"] = "center"
    va: Literal["bottom", "middle", "top"] = "middle"
    bold: bool = False
    name: str = "text"


@dataclass
class AxSegments:
    """Line segments in pt, each anchored at an axes-fraction point.

    ``segments`` holds ``(xf, yf, x0, y0, x1, y1)``: the anchor plus two
    endpoints in pt relative to it.
    """

    segments: list[tuple[float, float, float, float, float, float]]
    color: str
    width: float
    name: str = "segments"


Mark = Sticks | Line | Points | Bars | HitLayer | RefLine | Band | LabelSet | AxText | AxSegments


# ---------------------------------------------------------------------------
# Axes, panels, figures
# ---------------------------------------------------------------------------


@dataclass
class Axis:
    """One axis. Leave ``lo``/``hi`` as ``None`` to fit the data."""

    label: RichText | None = None
    lo: float | None = None
    hi: float | None = None
    #: Explicit tick positions; ``None`` picks round numbers.
    ticks: list[float] | None = None
    ticktext: list[str] | None = None
    #: No ticks beyond these values (e.g. stop at 100 % under label headroom).
    tick_min: float | None = None
    tick_max: float | None = None
    #: Label ticks with absolute values (mirror plots: the lower half reads positive).
    abs_ticklabels: bool = False
    grid: bool | None = None
    zeroline: bool = False
    visible: bool = True
    #: Fractional padding added to a data-fitted range.
    pad: float = 0.0
    #: Allow the upper (and for mirrors, lower) limit to grow to fit labels.
    headroom: bool = False
    #: Divide large tick values by a power of ten and name it in the title.
    scale_exponent: bool = False


@dataclass
class Colorbar:
    """A continuous colour legend for a sequential encoding."""

    lo: float
    hi: float
    scale: list[list[Any]]
    title: RichText


@dataclass
class Panel:
    """One set of axes and what is drawn in it."""

    marks: list[Mark]
    x: Axis
    y: Axis
    #: Secondary axis on the right that relabels ``y``: ``(title, factor)``;
    #: tick text is ``value * factor``.
    y_secondary: tuple[RichText, float] | None = None
    #: Space reserved above the axes for header furniture, pt.
    header_height: float = 0.0
    #: Relative share of the stack's free height, or a fixed height in pt.
    weight: float = 1.0
    fixed_height: float | None = None
    colorbar: Colorbar | None = None
    #: Share the x axis with the panel above; the upper panel's x tick labels are hidden.
    share_x: bool = False


@dataclass
class Cell:
    """A stack of panels sharing a column, with its own title, legend and letter.

    The legend lists every mark that has ``legend=True`` and a ``name``.
    """

    panels: list[Panel]
    title: RichText | None = None
    letter: str | None = None
    #: Preferred height/width ratio of the cell, and fixed extra height in mm.
    aspect: float = 0.62
    extra_height_mm: float = 0.0
    #: Size the figure height to the cell's fixed-height panels plus its title
    #: and legend band (``aspect`` and ``extra_height_mm`` are then ignored).
    fit_height: bool = False


@dataclass
class FigureSpec:
    """A complete, backend-neutral figure. See the module docstring."""

    cells: list[Cell]
    style: FigureStyle
    theme_mode: theme.ThemeMode
    width_mm: float
    height_mm: float
    ncols: int = 1
    #: Plotly ``update_layout`` overrides, applied last.
    layout_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def panels(self) -> list[Panel]:
        return [p for cell in self.cells for p in cell.panels]

    def resolve(self) -> ResolvedFigure:
        """Lay the figure out: margins, ticks, and label positions."""
        from ._layout import resolve_figure

        return resolve_figure(self)

    def render(self, backend: Literal["plotly", "matplotlib"] = "plotly") -> Any:
        """Draw with a backend; see :func:`render`."""
        return render(self, backend)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def check_backend(backend: str) -> str:
    """Validate a ``backend=`` argument and make sure its package is importable."""
    key = str(backend).lower()
    if key not in BACKENDS:
        raise SpxtacularError(f"backend must be one of {', '.join(BACKENDS)}; got {backend!r}")
    if key == "matplotlib" and importlib.util.find_spec("matplotlib") is None:
        raise SpxtacularError("backend='matplotlib' needs matplotlib: pip install 'spxtacular[matplotlib]'")
    if key == "plotly" and importlib.util.find_spec("plotly") is None:
        raise SpxtacularError("backend='plotly' needs plotly: pip install plotly")
    return key


def render(spec: FigureSpec, backend: Literal["plotly", "matplotlib"] = "plotly") -> Any:
    """Draw a :class:`FigureSpec` with plotly or matplotlib.

    Returns a ``plotly.graph_objects.Figure`` or a ``matplotlib.figure.Figure``.
    """
    key = check_backend(backend)
    if key == "spec":
        return spec
    resolved = spec.resolve()
    if key == "plotly":
        from ._backend_plotly import draw

        return draw(resolved)
    if spec.layout_kwargs:
        raise SpxtacularError(
            "layout keyword arguments are plotly layout settings and do not apply to "
            f"backend='matplotlib' (got {', '.join(sorted(spec.layout_kwargs))})"
        )
    from ._backend_mpl import draw as draw_mpl

    return draw_mpl(resolved)


def _preferred_height_mm(cell: Cell, style: FigureStyle, width_mm: float) -> float:
    """The height a cell would like at ``width_mm``."""
    if cell.fit_height:
        from ._layout import fitted_height

        return fitted_height(cell, style, width_mm * PT_PER_MM) / PT_PER_MM
    return cell.aspect * width_mm + cell.extra_height_mm


def new_spec(
    cell: Cell,
    *,
    style: StyleName | str | FigureStyle | None,
    backend: str,
    size: SizeLike,
    theme_mode: theme.ThemeMode | None,
    layout_kwargs: dict[str, Any] | None = None,
) -> FigureSpec:
    """Wrap one cell in a figure at the resolved style and size."""
    fig_style = resolve_style(style, backend)
    if cell.fit_height:
        width, _ = resolve_size(size, fig_style, aspect=0.0)
        fitted_mm = _preferred_height_mm(cell, fig_style, width)
        width, height = resolve_size(size, fig_style, aspect=0.0, extra_height_mm=fitted_mm)
    else:
        width, height = resolve_size(size, fig_style, aspect=cell.aspect, extra_height_mm=cell.extra_height_mm)
    return FigureSpec(
        cells=[cell],
        style=fig_style,
        theme_mode=theme.resolve_mode(theme_mode),
        width_mm=width,
        height_mm=height,
        layout_kwargs=dict(layout_kwargs or {}),
    )


def finish(spec: FigureSpec, backend: str) -> Any:
    """Return the spec for ``backend="spec"``, else render it."""
    key = check_backend(backend)
    if key == "spec":
        return spec
    return render(spec, cast(Literal["plotly", "matplotlib"], key))


def compose_figure(
    figures: Sequence[FigureSpec],
    *,
    ncols: int | None = None,
    labels: Sequence[str] | Literal["abc", "ABC"] | None = "abc",
    size: SizeLike = "double",
    style: StyleName | str | FigureStyle | None = None,
    backend: Backend = "matplotlib",
    theme_mode: theme.ThemeMode | None = None,
) -> Any:
    """Arrange several figures into one multi-panel figure with a/b/c labels.

    Build each part with ``backend="spec"``, then compose::

        a = spx.annotate_spectrum(s, frags, backend="spec")
        b = spx.mirror_plot(s, library, fragments=frags, backend="spec")
        fig = spx.compose_figure([a, b], ncols=2, backend="matplotlib")
        spx.save_figure(fig, "figure2.pdf")

    The parts are re-laid out at their new size, so labels are placed again
    for the space they actually get.

    Parameters
    ----------
    figures:
        Figure specs from any plotting function called with ``backend="spec"``.
    ncols:
        Columns in the grid. Defaults to 2 for two or more parts.
    labels:
        Panel letters. ``"abc"`` (default) or ``"ABC"``, an explicit sequence,
        or ``None`` for no letters.
    size:
        Overall size: a column name (default ``"double"``), a width in mm, or
        ``(width_mm, height_mm)``. The height defaults to the sum of the rows'
        preferred heights.
    style:
        Style for the composed figure. ``None`` keeps the first part's style.
    backend:
        ``"matplotlib"`` (default), ``"plotly"``, or ``"spec"``.
    theme_mode:
        ``"light"`` or ``"dark"``. ``None`` keeps the first part's mode.
    """
    if not figures:
        raise SpxtacularError("compose_figure needs at least one figure")
    for fig in figures:
        if not isinstance(fig, FigureSpec):
            raise SpxtacularError("compose_figure takes figure specs; call the plotting functions with backend='spec'")
    n = len(figures)
    cols = ncols if ncols is not None else (1 if n == 1 else 2)
    if cols < 1:
        raise SpxtacularError(f"ncols must be at least 1, got {cols}")
    rows = -(-n // cols)

    fig_style = figures[0].style if style is None else resolve_style(style, backend)
    mode = figures[0].theme_mode if theme_mode is None else theme.resolve_mode(theme_mode)

    if labels is None:
        letters: list[str | None] = [None] * n
    elif labels in ("abc", "ABC"):
        base = "abcdefghijklmnopqrstuvwxyz"
        letters = [(base if labels == "abc" else base.upper())[i % 26] for i in range(n)]
    else:
        letters = list(labels)[:n] + [None] * max(0, n - len(labels))

    cells: list[Cell] = []
    for fig, letter in zip(figures, letters, strict=True):
        for cell in fig.cells:
            cells.append(replace(cell, letter=letter))

    width, height = resolve_size(size, fig_style, aspect=0.5)
    if not (isinstance(size, tuple) and len(size) == 2 and size[1] is not None):
        cell_width = width / cols
        row_heights = []
        for r in range(rows):
            row = cells[r * cols : (r + 1) * cols]
            row_heights.append(max(_preferred_height_mm(c, fig_style, cell_width) for c in row))
        height = sum(row_heights)

    spec = FigureSpec(
        cells=cells,
        style=fig_style,
        theme_mode=mode,
        width_mm=width,
        height_mm=height,
        ncols=cols,
    )
    return finish(spec, backend)
