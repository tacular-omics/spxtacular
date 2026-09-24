"""
Draw a resolved figure with matplotlib.

The figure is a bare :class:`matplotlib.figure.Figure` (no pyplot, so nothing
leaks into a global figure registry). Positions come from
:mod:`spxtacular._layout` in points; every axes is placed with ``add_axes`` at
its resolved rectangle and every label at its resolved offset.

Fonts
-----
Journals want Arial or Helvetica, embedded as real fonts. The figure uses the
first installed face of the style's font list (Liberation Sans stands in for
Arial on Linux) for text *and* for mathtext, so sub- and superscripts in ion
labels match the surrounding text. PDF output embeds TrueType (``fonttype
42``), SVG output keeps text as text. These settings are applied whenever the
figure draws or saves, via :class:`SpxFigure`, so they hold however the user
saves the figure.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np

from ._layout import ResolvedAxis, ResolvedFigure, ResolvedPanel
from .figspec import AxSegments, AxText, Band, Bars, Line, Points, RefLine, Sticks

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_DASH: dict[str, Any] = {
    "solid": "solid",
    "dash": (0, (4, 2)),
    "dot": (0, (1, 1.5)),
    "dashdot": (0, (4, 1.5, 1, 1.5)),
    "longdash": (0, (7, 2)),
}


@lru_cache(maxsize=8)
def installed_font(candidates: tuple[str, ...]) -> str:
    """The first of ``candidates`` matplotlib can find, else DejaVu Sans."""
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in names:
            return name
    return "DejaVu Sans"


def rc_for(fonts: tuple[str, ...]) -> dict[str, Any]:
    font = installed_font(fonts)
    return {
        "font.family": "sans-serif",
        "font.sans-serif": [font, "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": font,
        "mathtext.sf": font,
        "mathtext.it": f"{font}:italic",
        "mathtext.bf": f"{font}:bold",
        "mathtext.cal": font,
        "mathtext.default": "rm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": True,
    }


def _figure_class():
    from matplotlib import rc_context
    from matplotlib.figure import Figure

    class SpxFigure(Figure):
        """A matplotlib Figure that draws and saves with spxtacular's font settings."""

        spx_rc: ClassVar[dict[str, Any]] = {}
        spx_dpi: int = 300

        def draw(self, renderer):  # type: ignore[override]
            with rc_context(cast(Any, self.spx_rc)):
                return super().draw(renderer)

        def savefig(self, fname, *args, **kwargs):  # type: ignore[override]
            kwargs.setdefault("dpi", self.spx_dpi)
            kwargs.setdefault("facecolor", self.get_facecolor())
            with rc_context(cast(Any, self.spx_rc)):
                return super().savefig(fname, *args, **kwargs)

    return SpxFigure


_FIGURE_CLASS: Any = None


def figure_class():
    global _FIGURE_CLASS
    if _FIGURE_CLASS is None:
        _FIGURE_CLASS = _figure_class()
    return _FIGURE_CLASS


def draw(fig: ResolvedFigure) -> Any:
    from matplotlib import rc_context

    rc = rc_for(fig.style.fonts)
    with rc_context(cast(Any, rc)):
        out = _draw(fig)
    out.spx_rc = rc
    out.spx_dpi = fig.style.dpi
    return out


def _axis_setup(ax: Axes, rx: ResolvedAxis, ry: ResolvedAxis, fig: ResolvedFigure, rp: ResolvedPanel) -> None:
    style = fig.style
    ink = fig.ink
    ax.set_facecolor("none")
    ax.set_xlim(rx.lo, rx.hi)
    ax.set_ylim(ry.lo, ry.hi)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side, visible in (("bottom", rx.visible), ("left", ry.visible)):
        spine = ax.spines[side]
        spine.set_visible(visible)
        spine.set_linewidth(style.axis_width)
        spine.set_color(ink.axis)
        spine.set_capstyle("projecting")
    ax.set_xticks(rx.ticks, labels=rx.ticktext if rx.show_ticklabels else [""] * len(rx.ticks))
    ax.set_yticks(ry.ticks, labels=ry.ticktext if ry.show_ticklabels else [""] * len(ry.ticks))
    ax.minorticks_off()
    ax.tick_params(
        direction="out",
        length=style.tick_length,
        width=style.axis_width,
        colors=ink.tick,
        labelcolor=ink.ticklabel,
        labelsize=style.font_size,
        pad=style.font_size * 0.35,
    )
    if not rx.visible:
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    if not ry.visible:
        ax.tick_params(axis="y", left=False, labelleft=False)
    gap = style.font_size * 0.4
    if rx.title:
        ax.set_xlabel(rx.title.mathtext(), fontsize=style.axis_title_size, color=ink.axis_title, labelpad=gap)
    if ry.title:
        ax.set_ylabel(ry.title.mathtext(), fontsize=style.axis_title_size, color=ink.axis_title, labelpad=gap)
    if ry.grid:
        ax.yaxis.grid(True, color=ink.grid, linewidth=0.5)
        ax.set_axisbelow(True)
    if rx.grid:
        ax.xaxis.grid(True, color=ink.grid, linewidth=0.5)
        ax.set_axisbelow(True)
    if ry.zeroline:
        ax.axhline(0.0, color=ink.axis, linewidth=style.axis_width, zorder=2.5)
    del rp


def _draw(fig: ResolvedFigure) -> Any:
    from matplotlib import transforms
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D

    style = fig.style
    ink = fig.ink
    W, H = fig.width, fig.height
    cls = figure_class()
    out = cls(figsize=(W / 72.0, H / 72.0), dpi=100, facecolor=ink.surface)
    # Figure coordinates in points, bottom-left origin: the layout's native frame.
    pt = transforms.Affine2D().scale(1.0 / 72.0) + out.dpi_scale_trans

    for cell in fig.cells:
        handles: list[Any] = []
        for rp in cell.panels:
            left, b, w, h = rp.rect
            ax = out.add_axes((left / W, b / H, w / W, h / H))
            _axis_setup(ax, rp.x, rp.y, fig, rp)
            leader_segs: list[tuple[tuple[float, float], tuple[float, float]]] = []

            def data_to_pt(x: float, y: float, rp: ResolvedPanel = rp) -> tuple[float, float]:
                left, b, w, h = rp.rect
                return (
                    left + (x - rp.x.lo) / (rp.x.hi - rp.x.lo) * w,
                    b + (y - rp.y.lo) / (rp.y.hi - rp.y.lo) * h,
                )

            for k, m in enumerate(rp.panel.marks):
                if isinstance(m, Sticks):
                    x = np.asarray(m.x, dtype=np.float64)
                    y = np.asarray(m.y, dtype=np.float64)
                    if len(x):
                        segs = np.stack(
                            [np.stack([x, np.full(len(x), m.base)], axis=1), np.stack([x, y], axis=1)], axis=1
                        )
                        lc = LineCollection(
                            list(segs),
                            colors=m.color,
                            linewidths=m.width,
                            alpha=m.opacity,
                            linestyles=_DASH.get(m.dash, "solid"),
                            capstyle="butt",
                            zorder=2,
                        )
                        ax.add_collection(lc)
                elif isinstance(m, Line):
                    x = np.asarray(m.x, dtype=np.float64)
                    y = np.asarray(m.y, dtype=np.float64)
                    ax.plot(x, y, color=m.color, linewidth=m.width, linestyle=_DASH.get(m.dash, "solid"), zorder=2)
                    if m.fill:
                        ax.fill_between(x, 0, y, color=m.color, alpha=m.fill_alpha, linewidth=0, zorder=1.5)
                elif isinstance(m, Points):
                    sizes = np.asarray(m.sizes, dtype=np.float64)
                    ax.scatter(
                        np.asarray(m.x, dtype=np.float64),
                        np.asarray(m.y, dtype=np.float64),
                        s=sizes**2,
                        c=list(m.colors),
                        edgecolors=m.outline if m.outline else "none",
                        linewidths=m.outline_width if m.outline else 0,
                        alpha=m.opacity,
                        zorder=3,
                        clip_on=False,
                    )
                elif isinstance(m, Bars):
                    ax.bar(
                        np.asarray(m.x, dtype=np.float64),
                        np.asarray(m.height, dtype=np.float64),
                        width=m.width,
                        color=list(m.colors),
                        linewidth=0,
                        zorder=2,
                    )
                elif isinstance(m, RefLine):
                    ls = _DASH.get(m.dash, "solid")
                    if m.orient == "v":
                        ax.axvline(m.value, color=m.color, linewidth=m.width, linestyle=ls, zorder=1)
                        if m.label and k in rp.refline_labels:
                            dx, dy, ha = rp.refline_labels[k]
                            ax.annotate(
                                m.label.mathtext(),
                                xy=(m.value, 1.0),
                                xycoords=("data", "axes fraction"),
                                xytext=(dx, dy),
                                textcoords="offset points",
                                ha=ha,
                                va="top",
                                fontsize=style.label_size,
                                color=m.label_color or ink.muted,
                                annotation_clip=False,
                            )
                    else:
                        ax.axhline(m.value, color=m.color, linewidth=m.width, linestyle=ls, zorder=1)
                elif isinstance(m, Band):
                    if m.orient == "v":
                        ax.axvspan(m.lo, m.hi, color=m.color, alpha=m.alpha, linewidth=0, zorder=0)
                    else:
                        ax.axhspan(m.lo, m.hi, color=m.color, alpha=m.alpha, linewidth=0, zorder=0)
                elif isinstance(m, AxText):
                    ax.text(
                        m.xf,
                        m.yf,
                        m.text.mathtext(),
                        transform=transforms.offset_copy(ax.transAxes, fig=out, x=m.dx, y=m.dy, units="points"),
                        ha=m.ha,
                        va="center" if m.va == "middle" else m.va,
                        fontsize=m.size,
                        color=m.color,
                        fontweight="bold" if m.bold else "normal",
                        clip_on=False,
                    )
                elif isinstance(m, AxSegments):
                    segs_pt = []
                    for xf, yf, x0, y0, x1, y1 in m.segments:
                        ax_x = left + xf * w
                        ax_y = b + yf * h
                        segs_pt.append([(ax_x + x0, ax_y + y0), (ax_x + x1, ax_y + y1)])
                    out.add_artist(
                        LineCollection(segs_pt, colors=m.color, linewidths=m.width, transform=pt, capstyle="butt")
                    )
                if getattr(m, "legend", False) and getattr(m, "name", None):
                    pass

            for lab in rp.labels:
                ax.annotate(
                    lab.text.mathtext(),
                    xy=(lab.x, lab.y),
                    xytext=(lab.dx, lab.dy),
                    textcoords="offset points",
                    ha="center",
                    va=lab.va,
                    rotation=lab.rotation,
                    fontsize=lab.size,
                    color=lab.color,
                    annotation_clip=False,
                    zorder=4,
                )
                if lab.leader is not None:
                    ax_x, ax_y = data_to_pt(lab.x, lab.y)
                    x0, y0, x1, y1 = lab.leader
                    leader_segs.append(((ax_x + x0, ax_y + y0), (ax_x + x1, ax_y + y1)))
            if leader_segs:
                out.add_artist(
                    LineCollection(leader_segs, colors=ink.muted, linewidths=style.leader_width, transform=pt, zorder=3)
                )

            if rp.y2 is not None:
                ax2 = ax.twinx()
                ax2.set_ylim(rp.y.lo, rp.y.hi)
                ax2.set_facecolor("none")
                for side in ("top", "left", "bottom"):
                    ax2.spines[side].set_visible(False)
                ax2.spines["right"].set_linewidth(style.axis_width)
                ax2.spines["right"].set_color(ink.axis)
                ax2.set_yticks(rp.y2.ticks, labels=rp.y2.ticktext)
                ax2.minorticks_off()
                ax2.tick_params(
                    direction="out",
                    length=style.tick_length,
                    width=style.axis_width,
                    colors=ink.tick,
                    labelcolor=ink.ticklabel,
                    labelsize=style.font_size,
                    pad=style.font_size * 0.35,
                )
                if rp.y2.title:
                    ax2.set_ylabel(
                        rp.y2.title.mathtext(),
                        fontsize=style.axis_title_size,
                        color=ink.axis_title,
                        labelpad=style.font_size * 0.4,
                    )

            if rp.panel.colorbar is not None and cell.colorbar_rect is not None:
                from matplotlib.cm import ScalarMappable

                cb = rp.panel.colorbar
                cl, cb_b, cw, ch = cell.colorbar_rect
                cax = out.add_axes((cl / W, cb_b / H, cw / W, ch / H))
                stops = [(float(pos), str(col)) for pos, col in cb.scale]
                cmap = LinearSegmentedColormap.from_list("spx", stops)
                sm = ScalarMappable(norm=Normalize(cb.lo, cb.hi), cmap=cmap)
                bar = out.colorbar(sm, cax=cax)
                bar.outline.set_visible(False)  # type: ignore[union-attr]
                cax.tick_params(
                    length=style.tick_length,
                    width=style.axis_width,
                    colors=ink.tick,
                    labelcolor=ink.ticklabel,
                    labelsize=style.font_size,
                )
                bar.set_label(cb.title.mathtext(), fontsize=style.font_size, color=ink.axis_title)

        for item in cell.legend:
            if item.kind == "marker":
                handles.append(
                    Line2D([], [], linestyle="none", marker="o", markersize=style.font_size * 0.7, color=item.color)
                )
            else:
                handles.append(
                    Line2D(
                        [],
                        [],
                        color=item.color,
                        linewidth=max(item.width, style.stick_width) * 1.6,
                        linestyle=_DASH.get(item.dash, "solid"),
                        solid_capstyle="butt",
                    )
                )
        if handles and cell.legend_xy is not None:
            rx, by = cell.legend_xy
            out.legend(
                handles,
                [it.name for it in cell.legend],
                loc="lower right",
                bbox_to_anchor=(rx / W, by / H),
                bbox_transform=out.transFigure,
                ncol=len(handles),
                frameon=False,
                fontsize=style.font_size,
                handlelength=1.1,
                handletextpad=0.4,
                columnspacing=0.9,
                borderaxespad=0.0,
                borderpad=0.0,
                labelcolor=ink.axis_title,
            )
        if cell.title_xy is not None and cell.cell.title:
            tx, ty = cell.title_xy
            out.text(
                tx / W,
                ty / H,
                cell.cell.title.mathtext(),
                ha="left",
                va="top",
                fontsize=style.title_size,
                color=ink.title,
            )
        if cell.letter_xy is not None and cell.cell.letter:
            lx, ly = cell.letter_xy
            out.text(
                lx / W,
                ly / H,
                cell.cell.letter,
                ha="left",
                va="top",
                fontsize=style.panel_letter_size,
                fontweight="bold",
                color=ink.title,
            )
    return out
