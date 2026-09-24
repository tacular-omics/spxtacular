"""
Figure styles: the size, type and line weights a figure is drawn at.

A style is independent of colour (that is :mod:`spxtacular.theme`) and of the
backend. The same :class:`FigureStyle` drives plotly and matplotlib, so a paper
figure has 7 pt tick labels and 0.6 pt axis lines whichever engine drew it.

Three presets ship:

``"screen"``
    The interactive default for plotly: a wide figure that fills a notebook
    cell, light gridlines, the Inter-first font stack.
``"paper"``
    Journal figures. Single-column width (85 mm), Arial/Helvetica, 7 pt text,
    thin dark axes, no gridlines, no title. The default for matplotlib.
``"talk"``
    Slides. A 16:9 figure, large type, heavier lines.

Sizes
-----
Every plotting function takes ``size=``. A string names a journal column width
from :data:`COLUMN_WIDTHS_MM` (``"single"`` 85 mm, ``"onehalf"`` 114 mm,
``"double"`` 175 mm); a pair is ``(width_mm, height_mm)``; a single number is a
width in mm. When no height is given the figure picks one that suits its
content, e.g. taller when a mass-error panel is added.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from .errors import SpxtacularError

MM_PER_PT: float = 25.4 / 72.0
PT_PER_MM: float = 72.0 / 25.4
#: CSS pixels per point: plotly sizes are in CSS px at 96 dpi.
PX_PER_PT: float = 96.0 / 72.0

#: Common journal column widths in millimetres.
COLUMN_WIDTHS_MM: dict[str, float] = {
    "single": 85.0,
    "onehalf": 114.0,
    "double": 175.0,
}

StyleName = Literal["paper", "screen", "talk"]
SizeLike = str | float | int | tuple[float, float] | tuple[float, None] | None

#: Arial and Helvetica first, the faces journals ask for; Liberation Sans is the
#: metric-compatible free substitute found on most Linux systems.
PAPER_FONTS: tuple[str, ...] = ("Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans")
SCREEN_FONTS: tuple[str, ...] = (
    "Inter",
    "-apple-system",
    "BlinkMacSystemFont",
    "Segoe UI",
    "Helvetica",
    "Arial",
    "Liberation Sans",
    "DejaVu Sans",
)


@dataclass(frozen=True)
class FigureStyle:
    """Everything about a figure's geometry and type that is not colour.

    All sizes are in points (1/72 inch) unless the name says otherwise.
    Create variants with :meth:`with_`::

        style = spx.get_style("paper").with_(font_size=8, label_size=6.5)
    """

    name: str
    #: Default figure width in mm, and the default height as a fraction of it.
    width_mm: float
    aspect: float
    fonts: tuple[str, ...]
    #: Tick labels and legend.
    font_size: float
    #: Axis titles.
    axis_title_size: float
    #: Figure title. Only drawn when ``show_title`` is on or a title is passed.
    title_size: float
    #: Ion labels and other direct annotations.
    label_size: float
    #: Panel letters (a, b, c) in composed figures.
    panel_letter_size: float
    #: Axis spine and tick line width.
    axis_width: float
    tick_length: float
    #: Stick width for matched / highlighted peaks, and for context peaks.
    stick_width: float
    stick_width_context: float
    #: Width of continuous traces (profile, chromatogram).
    line_width: float
    #: Leader lines from a displaced label back to its peak.
    leader_width: float
    #: Horizontal gridlines on the value axis.
    grid: bool
    #: Draw the default title ("Annotated spectrum") when none is passed.
    show_title: bool
    #: Colour ion labels with their series hue (when it has enough contrast).
    label_series_color: bool
    #: Dark axes and ink for print instead of the recessive screen greys.
    print_ink: bool
    #: Draw context (unmatched) peaks in the darker neutral grey.
    strong_context: bool
    #: Preferred spacing between ticks, in points, per axis.
    x_tick_spacing: float
    y_tick_spacing: float
    #: Raster resolution for PNG export.
    dpi: int
    #: Gap between a peak tip and its label.
    label_gap: float
    #: Horizontal ion labels (True) or rotated to read bottom-to-top.
    horizontal_labels: bool
    #: Plotly only: let the figure fill its container (interactive use). Figures
    #: with placed peak labels keep their design width, since label offsets are px.
    autosize: bool = False

    def with_(self, **changes: object) -> FigureStyle:
        """A copy with some fields changed."""
        return replace(self, **changes)  # type: ignore[arg-type]

    @property
    def font_family_css(self) -> str:
        """The font stack as a CSS ``font-family`` value, for plotly."""
        return ", ".join(f"'{f}'" if " " in f else f for f in self.fonts) + ", sans-serif"


_PRESETS: dict[str, FigureStyle] = {
    "paper": FigureStyle(
        name="paper",
        width_mm=COLUMN_WIDTHS_MM["single"],
        aspect=0.62,
        fonts=PAPER_FONTS,
        font_size=7.0,
        axis_title_size=7.5,
        title_size=8.0,
        label_size=6.5,
        panel_letter_size=9.0,
        axis_width=0.6,
        tick_length=2.5,
        stick_width=0.8,
        stick_width_context=0.5,
        line_width=0.8,
        leader_width=0.35,
        grid=False,
        show_title=False,
        label_series_color=True,
        print_ink=True,
        strong_context=True,
        x_tick_spacing=48.0,
        y_tick_spacing=30.0,
        dpi=600,
        label_gap=1.2,
        horizontal_labels=True,
    ),
    "screen": FigureStyle(
        name="screen",
        width_mm=238.0,  # 900 CSS px
        aspect=0.5,
        fonts=SCREEN_FONTS,
        font_size=8.25,  # 11 px
        axis_title_size=9.0,  # 12 px
        title_size=12.0,  # 16 px
        label_size=8.25,
        panel_letter_size=11.0,
        axis_width=0.75,
        tick_length=3.0,
        stick_width=1.2,
        stick_width_context=0.75,
        line_width=1.05,
        leader_width=0.5,
        grid=True,
        show_title=True,
        label_series_color=True,
        print_ink=False,
        strong_context=False,
        x_tick_spacing=70.0,
        y_tick_spacing=40.0,
        dpi=192,
        label_gap=2.0,
        horizontal_labels=True,
        autosize=True,
    ),
    "talk": FigureStyle(
        name="talk",
        width_mm=254.0,  # 10 in, 16:9
        aspect=0.5625,
        fonts=PAPER_FONTS,
        font_size=14.0,
        axis_title_size=16.0,
        title_size=20.0,
        label_size=13.0,
        panel_letter_size=20.0,
        axis_width=1.25,
        tick_length=5.0,
        stick_width=2.0,
        stick_width_context=1.1,
        line_width=2.0,
        leader_width=0.8,
        grid=False,
        show_title=True,
        label_series_color=True,
        print_ink=True,
        strong_context=True,
        x_tick_spacing=90.0,
        y_tick_spacing=55.0,
        dpi=200,
        label_gap=2.5,
        horizontal_labels=True,
    ),
}


def get_style(style: StyleName | str | FigureStyle) -> FigureStyle:
    """Look up a style preset by name, or pass a :class:`FigureStyle` through.

    Raises
    ------
    SpxtacularError
        For an unknown preset name.
    """
    if isinstance(style, FigureStyle):
        return style
    key = str(style).lower()
    if key not in _PRESETS:
        raise SpxtacularError(f"unknown figure style {style!r}; expected one of {', '.join(sorted(_PRESETS))}")
    return _PRESETS[key]


def resolve_style(style: StyleName | str | FigureStyle | None, backend: str) -> FigureStyle:
    """The style to draw with. ``None`` means ``"screen"`` for plotly, ``"paper"`` otherwise."""
    if style is None:
        return _PRESETS["screen" if backend == "plotly" else "paper"]
    return get_style(style)


def resolve_size(
    size: SizeLike,
    style: FigureStyle,
    *,
    aspect: float | None = None,
    extra_height_mm: float = 0.0,
) -> tuple[float, float]:
    """Figure ``(width_mm, height_mm)`` from a ``size=`` argument.

    ``aspect`` is the height/width ratio the figure would like for its main
    panel; ``extra_height_mm`` is added for fixed-height furniture such as a
    sequence header or an error strip, so those do not squash the spectrum.
    """
    height: float | None = None
    if size is None:
        width = style.width_mm
    elif isinstance(size, str):
        key = size.lower()
        if key not in COLUMN_WIDTHS_MM:
            raise SpxtacularError(
                f"unknown figure size {size!r}; expected one of {', '.join(COLUMN_WIDTHS_MM)} or (width_mm, height_mm)"
            )
        width = COLUMN_WIDTHS_MM[key]
    elif isinstance(size, int | float):
        width = float(size)
    elif isinstance(size, tuple) and len(size) == 2:
        width = float(size[0])
        height = None if size[1] is None else float(size[1])
    else:
        raise SpxtacularError(f"size must be a column name, a width in mm, or (width_mm, height_mm); got {size!r}")
    if width <= 0 or (height is not None and height <= 0):
        raise SpxtacularError(f"figure size must be positive, got {size!r}")
    if height is None:
        ratio = aspect if aspect is not None else style.aspect
        # Wide figures do not need to be proportionally tall: a double-column
        # spectrum at the single-column aspect wastes half a page.
        base = width * ratio
        if size is not None and width > 120:
            base = min(base, 120 * ratio + (width - 120) * ratio * 0.45)
        height = base + extra_height_mm
    return width, height
