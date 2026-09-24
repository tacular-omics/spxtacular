"""
Draw a resolved figure with plotly.

The layout is fully decided by :mod:`spxtacular._layout`; this module maps it
onto plotly: points become CSS pixels (96/72), panel rectangles become axis
domains with a zero margin, and placed labels become annotations shifted by a
fixed pixel offset, so they stay attached to their peaks when the reader zooms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from . import theme
from ._layout import ResolvedAxis, ResolvedFigure, ResolvedPanel
from .figspec import AxSegments, AxText, Band, Bars, HitLayer, LabelSet, Line, Points, RefLine, Sticks
from .style import PX_PER_PT

if TYPE_CHECKING:
    import plotly.graph_objects as go

# Dash patterns in multiples of the line width, the same as the matplotlib
# backend, so a dashed hairline has short dashes rather than plotly's fixed ones.
_DASH_UNITS: dict[str, tuple[float, ...]] = {
    "dash": (4, 2),
    "dot": (1, 1.5),
    "dashdot": (4, 1.5, 1, 1.5),
    "longdash": (7, 2),
}


def _dash(dash: str, width_px: float) -> str:
    units = _DASH_UNITS.get(dash)
    if units is None:
        return dash
    w = max(width_px, 1.0)
    return ",".join(f"{u * w:.2f}px" for u in units)


def px(pt: float) -> float:
    return round(pt * PX_PER_PT, 4)


def rgba(hex_color: str, alpha: float) -> str:
    """``#rrggbb`` -> ``rgba(r,g,b,a)``; other colour strings pass through."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def stick_xy(x: np.ndarray, y: np.ndarray, base: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Interleave ``(x, base, x, NaN)`` triples: one polyline draws every stick."""
    n = len(x)
    xs = np.empty(n * 3, dtype=np.float64)
    ys = np.empty(n * 3, dtype=np.float64)
    xs[0::3] = x
    xs[1::3] = x
    xs[2::3] = np.nan
    ys[0::3] = base
    ys[1::3] = y
    ys[2::3] = np.nan
    return xs, ys


def _axis_names(index: int) -> tuple[str, str, str, str]:
    """(layout x key, layout y key, trace x ref, trace y ref)."""
    suffix = "" if index == 1 else str(index)
    return f"xaxis{suffix}", f"yaxis{suffix}", f"x{suffix}", f"y{suffix}"


def _axis_layout(ax: ResolvedAxis, fig: ResolvedFigure, *, is_x: bool) -> dict[str, Any]:
    style = fig.style
    ink = fig.ink
    out: dict[str, Any] = {
        "range": [ax.lo, ax.hi],
        "autorange": False,
        "showline": ax.visible,
        "linecolor": ink.axis,
        "linewidth": px(style.axis_width),
        "ticks": "outside" if ax.visible else "",
        "ticklen": px(style.tick_length),
        "tickwidth": px(style.axis_width),
        "tickcolor": ink.tick,
        "tickfont": {"size": px(style.font_size), "color": ink.ticklabel, "family": style.font_family_css},
        "showticklabels": ax.show_ticklabels and ax.visible,
        "showgrid": ax.grid,
        "gridcolor": ink.grid,
        "gridwidth": px(0.5),
        "zeroline": ax.zeroline,
        "zerolinecolor": ink.axis,
        "zerolinewidth": px(style.axis_width),
        "automargin": False,
        "fixedrange": False,
        "visible": ax.visible,
        "title": {
            "text": ax.title.html() if ax.title else None,
            "font": {"size": px(style.axis_title_size), "color": ink.axis_title, "family": style.font_family_css},
            "standoff": px(style.font_size * 0.4),
        },
    }
    if ax.auto_ticks:
        out["tickmode"] = "auto"
        out["nticks"] = max(len(ax.ticks) + 1, 3)
    else:
        out["tickmode"] = "array"
        out["tickvals"] = ax.ticks
        out["ticktext"] = ax.ticktext
    if not is_x:
        out["rangemode"] = "normal"
    return out


def _trace_common(xref: str, yref: str) -> dict[str, Any]:
    return {"xaxis": xref, "yaxis": yref}


def _expand_custom(customdata: Any) -> list[Any]:
    """Repeat each value for both stick endpoints plus the gap point.

    The gap is NaN for numeric data so the array stays numeric (None would turn
    it into an object array that readers of ``fig.data`` cannot compare).
    """
    out: list[Any] = []
    for c in customdata:
        gap = float("nan") if isinstance(c, int | float) and not isinstance(c, bool) else None
        out += [c, c, gap]
    return out


def _panel_traces(rp: ResolvedPanel, fig: ResolvedFigure, legend_ref: str, seen: set[str]) -> list[go.Scatter | go.Bar]:
    import plotly.graph_objects as go

    _, _, xref, yref = _axis_names(rp.index)
    style = fig.style
    traces: list[Any] = []
    for m in rp.panel.marks:
        common = _trace_common(xref, yref)
        if isinstance(m, Sticks):
            xs, ys = stick_xy(np.asarray(m.x, dtype=np.float64), np.asarray(m.y, dtype=np.float64), m.base)
            show = bool(m.legend and m.name and m.name not in seen)
            if show:
                seen.add(str(m.name))
            kwargs: dict[str, Any] = {
                "x": xs,
                "y": ys,
                "mode": "lines",
                "name": m.name or "",
                "line": {"color": m.color, "width": px(m.width), "dash": _dash(m.dash, px(m.width))},
                "opacity": m.opacity,
                "showlegend": show,
                "legendgroup": m.name,
                "legend": legend_ref,
            }
            if m.customdata is not None:
                kwargs["customdata"] = _expand_custom(m.customdata)
                kwargs["hovertemplate"] = m.hovertemplate or "%{customdata}<extra></extra>"
            elif m.hovertemplate is not None:
                kwargs["hovertemplate"] = m.hovertemplate
            else:
                kwargs["hoverinfo"] = "skip"
            traces.append(go.Scatter(**kwargs, **common))
        elif isinstance(m, Line):
            show = bool(m.legend and m.name and m.name not in seen)
            if show:
                seen.add(str(m.name))
            kwargs = {
                "x": np.asarray(m.x, dtype=np.float64),
                "y": np.asarray(m.y, dtype=np.float64),
                "mode": "lines",
                "name": m.name or "",
                # simplify=False: plotly otherwise drops vertices, which shows as
                # corners on peak apexes in high-dpi exports.
                "line": {
                    "color": m.color,
                    "width": px(m.width),
                    "dash": _dash(m.dash, px(m.width)),
                    "simplify": False,
                },
                "showlegend": show,
                "legend": legend_ref,
            }
            if m.fill:
                kwargs["fill"] = "tozeroy"
                kwargs["fillcolor"] = rgba(m.color, m.fill_alpha)
            if m.customdata is not None:
                kwargs["customdata"] = list(m.customdata)
            if m.hovertemplate is not None:
                kwargs["hovertemplate"] = m.hovertemplate
            traces.append(go.Scatter(**kwargs, **common))
        elif isinstance(m, Points):
            show = bool(m.legend and m.name and m.name not in seen)
            if show:
                seen.add(str(m.name))
            marker: dict[str, Any] = {
                "size": [px(s) for s in np.asarray(m.sizes, dtype=np.float64)],
                "color": list(m.colors),
                "opacity": m.opacity,
            }
            if m.outline:
                marker["line"] = {"color": m.outline, "width": px(m.outline_width)}
            kwargs = {
                "x": np.asarray(m.x, dtype=np.float64),
                "y": np.asarray(m.y, dtype=np.float64),
                "mode": "markers",
                "marker": marker,
                "name": m.name or "",
                "showlegend": show,
                "legend": legend_ref,
            }
            if m.customdata is not None:
                kwargs["customdata"] = list(m.customdata)
            if m.hovertemplate is not None:
                kwargs["hovertemplate"] = m.hovertemplate
            traces.append(go.Scatter(**kwargs, **common))
        elif isinstance(m, Bars):
            kwargs = {
                "x": np.asarray(m.x, dtype=np.float64),
                "y": np.asarray(m.height, dtype=np.float64),
                "width": m.width,
                "marker": {"color": list(m.colors), "line": {"width": 0}},
                "name": m.name or "",
                "showlegend": False,
            }
            if m.customdata is not None:
                kwargs["customdata"] = list(m.customdata)
            if m.hovertemplate is not None:
                kwargs["hovertemplate"] = m.hovertemplate
            traces.append(go.Bar(**kwargs, **common))
        elif isinstance(m, HitLayer):
            traces.append(
                go.Scatter(
                    x=np.asarray(m.x, dtype=np.float64),
                    y=np.asarray(m.y, dtype=np.float64),
                    mode="markers",
                    marker={"size": m.size_px, "color": "rgba(0,0,0,0)"},
                    customdata=list(m.hover),
                    hovertemplate="%{customdata}<extra></extra>",
                    showlegend=False,
                    name="",
                    **common,
                )
            )
    if rp.panel.colorbar is not None:
        cb = rp.panel.colorbar
        # A colour scale needs a trace to hang on; this one draws nothing.
        traces.append(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={
                    "color": [cb.lo],
                    "colorscale": cb.scale,
                    "cmin": cb.lo,
                    "cmax": cb.hi,
                    "showscale": True,
                    "colorbar": {},  # positioned in draw()
                },
                showlegend=False,
                hoverinfo="skip",
                name="",
                **_trace_common(xref, yref),
            )
        )
    del style
    return traces


def _panel_decor(rp: ResolvedPanel, fig: ResolvedFigure) -> tuple[list[dict], list[dict]]:
    """Annotations and shapes for one panel."""
    _, _, xref, yref = _axis_names(rp.index)
    style = fig.style
    ink = fig.ink
    family = style.font_family_css
    annotations: list[dict] = []
    shapes: list[dict] = []
    for k, m in enumerate(rp.panel.marks):
        if isinstance(m, RefLine):
            dash = _dash(m.dash, px(m.width))
            if m.orient == "v":
                shapes.append(
                    {
                        "type": "line",
                        "xref": xref,
                        "yref": f"{yref} domain",
                        "x0": m.value,
                        "x1": m.value,
                        "y0": 0,
                        "y1": 1,
                        "line": {"color": m.color, "width": px(m.width), "dash": dash},
                        "layer": "below",
                        "name": m.name,
                    }
                )
                if m.label and k in rp.refline_labels:
                    dx, dy, ha = rp.refline_labels[k]
                    annotations.append(
                        {
                            "x": m.value,
                            "y": 1,
                            "xref": xref,
                            "yref": f"{yref} domain",
                            "text": m.label.html(),
                            "showarrow": False,
                            "xanchor": ha,
                            "yanchor": "top",
                            "xshift": px(dx),
                            "yshift": px(dy),
                            "borderpad": 0,
                            "font": {
                                "size": px(style.label_size),
                                "color": m.label_color or ink.muted,
                                "family": family,
                            },
                            "name": m.name,
                        }
                    )
            else:
                shapes.append(
                    {
                        "type": "line",
                        "xref": f"{xref} domain",
                        "yref": yref,
                        "x0": 0,
                        "x1": 1,
                        "y0": m.value,
                        "y1": m.value,
                        "line": {"color": m.color, "width": px(m.width), "dash": dash},
                        "layer": "below",
                        "name": m.name,
                    }
                )
        elif isinstance(m, Band):
            if m.orient == "v":
                shape = {"xref": xref, "yref": f"{yref} domain", "x0": m.lo, "x1": m.hi, "y0": 0, "y1": 1}
            else:
                shape = {"xref": f"{xref} domain", "yref": yref, "x0": 0, "x1": 1, "y0": m.lo, "y1": m.hi}
            shapes.append(
                {
                    "type": "rect",
                    **shape,
                    "fillcolor": rgba(m.color, m.alpha),
                    "line": {"width": 0},
                    "layer": "below",
                }
            )
        elif isinstance(m, AxText):
            text = m.text.html()
            if m.bold:
                text = f"<b>{text}</b>"
            annotations.append(
                {
                    "x": m.xf,
                    "y": m.yf,
                    "xref": f"{xref} domain",
                    "yref": f"{yref} domain",
                    "text": text,
                    "showarrow": False,
                    "xanchor": m.ha,
                    "yanchor": m.va,
                    "xshift": px(m.dx),
                    "yshift": px(m.dy),
                    "borderpad": 0,
                    "font": {"size": px(m.size), "color": m.color, "family": family},
                    "name": m.name,
                }
            )
        elif isinstance(m, AxSegments):
            for xf, yf, x0, y0, x1, y1 in m.segments:
                shapes.append(
                    {
                        "type": "line",
                        "xref": f"{xref} domain",
                        "yref": f"{yref} domain",
                        "xsizemode": "pixel",
                        "ysizemode": "pixel",
                        "xanchor": xf,
                        "yanchor": yf,
                        "x0": px(x0),
                        "x1": px(x1),
                        "y0": px(y0),
                        "y1": px(y1),
                        "line": {"color": m.color, "width": px(m.width)},
                        "name": m.name,
                    }
                )
        elif isinstance(m, LabelSet):
            pass  # placed labels below
    for lab in rp.labels:
        annotations.append(
            {
                "x": lab.x,
                "y": lab.y,
                "xref": xref,
                "yref": yref,
                "text": lab.text.html(),
                "showarrow": False,
                "xanchor": "center",
                "yanchor": lab.va,
                "xshift": px(lab.dx),
                "yshift": px(lab.dy),
                "textangle": -lab.rotation,
                "borderpad": 0,
                "font": {"size": px(lab.size), "color": lab.color, "family": family},
                "name": lab.name,
            }
        )
        if lab.leader is not None:
            x0, y0, x1, y1 = lab.leader
            shapes.append(
                {
                    "type": "line",
                    "xref": xref,
                    "yref": yref,
                    "xsizemode": "pixel",
                    "ysizemode": "pixel",
                    "xanchor": lab.x,
                    "yanchor": lab.y,
                    "x0": px(x0),
                    "x1": px(x1),
                    "y0": px(y0),
                    "y1": px(y1),
                    "line": {"color": ink.muted, "width": px(style.leader_width)},
                    "name": "leader",
                }
            )
    return annotations, shapes


def draw(fig: ResolvedFigure) -> go.Figure:
    import plotly.graph_objects as go

    style = fig.style
    ink = fig.ink
    W, H = fig.width, fig.height
    family = style.font_family_css

    layout: dict[str, Any] = {
        "template": theme.template(fig.spec.theme_mode),
        "paper_bgcolor": ink.surface,
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": family, "size": px(style.font_size), "color": ink.axis_title},
        "margin": {"l": 0, "r": 0, "t": 0, "b": 0, "pad": 0},
        "height": px(H),
        "hovermode": "closest",
        "bargap": 0,
    }
    if style.autosize:
        layout["autosize"] = True
        layout["meta"] = {"spx_width": px(W), "spx_style": style.name, "spx_dpi": style.dpi}
    else:
        layout["autosize"] = False
        layout["width"] = px(W)
        layout["meta"] = {"spx_width": px(W), "spx_style": style.name, "spx_dpi": style.dpi}

    traces: list[Any] = []
    annotations: list[dict] = []
    shapes: list[dict] = []
    n_panels = len(fig.panels)
    overlay = n_panels
    any_legend = False

    for c_i, cell in enumerate(fig.cells):
        legend_key = "legend" if c_i == 0 else f"legend{c_i + 1}"
        legend_ref = "legend" if c_i == 0 else f"legend{c_i + 1}"
        seen: set[str] = set()
        for rp in cell.panels:
            xk, yk, xref, yref = _axis_names(rp.index)
            left, b, w, h = rp.rect
            xa = _axis_layout(rp.x, fig, is_x=True)
            xa["domain"] = [max(0.0, left / W), min(1.0, (left + w) / W)]
            xa["anchor"] = yref
            if rp.shared_with is not None:
                xa["matches"] = _axis_names(rp.shared_with)[2]
            ya = _axis_layout(rp.y, fig, is_x=False)
            ya["domain"] = [max(0.0, b / H), min(1.0, (b + h) / H)]
            ya["anchor"] = xref
            layout[xk] = xa
            layout[yk] = ya
            if rp.y2 is not None:
                overlay += 1
                y2k = f"yaxis{overlay}"
                y2 = _axis_layout(rp.y2, fig, is_x=False)
                y2.update(
                    {
                        "overlaying": yref,
                        "side": "right",
                        "anchor": xref,
                        "tickmode": "array",
                        "tickvals": rp.y2.ticks,
                        "ticktext": rp.y2.ticktext,
                        "showgrid": False,
                        "matches": yref,
                    }
                )
                layout[y2k] = y2
            traces.extend(_panel_traces(rp, fig, legend_ref, seen))
            a, s = _panel_decor(rp, fig)
            annotations.extend(a)
            shapes.extend(s)
            if rp.panel.colorbar is not None and cell.colorbar_rect is not None:
                cl, cb_, cw, ch = cell.colorbar_rect
                cbar = rp.panel.colorbar
                traces[-1].marker.colorbar = {
                    "x": cl / W,
                    "xanchor": "left",
                    "xref": "paper",
                    "y": cb_ / H,
                    "yanchor": "bottom",
                    "yref": "paper",
                    "len": ch / H,
                    "lenmode": "fraction",
                    "thickness": px(cw),
                    "thicknessmode": "pixels",
                    "outlinewidth": 0,
                    "ticks": "outside",
                    "ticklen": px(style.tick_length),
                    "tickwidth": px(style.axis_width),
                    "tickcolor": ink.tick,
                    "tickfont": {"size": px(style.font_size), "color": ink.ticklabel, "family": family},
                    "title": {
                        "text": cbar.title.html(),
                        "side": "right",
                        "font": {"size": px(style.font_size), "color": ink.axis_title, "family": family},
                    },
                }
        if cell.legend and cell.legend_xy is not None:
            any_legend = True
            rx, by = cell.legend_xy
            layout[legend_key] = {
                "orientation": "h",
                "x": rx / W,
                "xanchor": "right",
                "y": by / H,
                "yanchor": "bottom",
                "xref": "paper",
                "yref": "paper",
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "font": {"size": px(style.font_size), "color": ink.axis_title, "family": family},
                "itemsizing": "trace",
                "itemwidth": 30,
                "tracegroupgap": 0,
                "entrywidthmode": "pixels",
            }
        # Title and letter.
        if cell.title_xy is not None and cell.cell.title:
            tx, ty = cell.title_xy
            if len(fig.cells) == 1:
                layout["title"] = {
                    "text": cell.cell.title.html(),
                    "x": tx / W,
                    "xanchor": "left",
                    "xref": "paper",
                    "y": ty / H,
                    "yanchor": "top",
                    "yref": "paper",
                    "pad": {"l": 0, "t": 0, "r": 0, "b": 0},
                    "font": {"size": px(style.title_size), "color": ink.title, "family": family},
                }
            else:
                annotations.append(
                    {
                        "x": tx / W,
                        "y": ty / H,
                        "xref": "paper",
                        "yref": "paper",
                        "xanchor": "left",
                        "yanchor": "top",
                        "text": cell.cell.title.html(),
                        "showarrow": False,
                        "borderpad": 0,
                        "font": {"size": px(style.title_size), "color": ink.title, "family": family},
                        "name": "title",
                    }
                )
        if cell.letter_xy is not None and cell.cell.letter:
            lx, ly = cell.letter_xy
            annotations.append(
                {
                    "x": lx / W,
                    "y": ly / H,
                    "xref": "paper",
                    "yref": "paper",
                    "xanchor": "left",
                    "yanchor": "top",
                    "text": f"<b>{cell.cell.letter}</b>",
                    "showarrow": False,
                    "borderpad": 0,
                    "font": {"size": px(style.panel_letter_size), "color": ink.title, "family": family},
                    "name": "panel_letter",
                }
            )

    layout["showlegend"] = any_legend
    layout["annotations"] = annotations
    layout["shapes"] = shapes
    out = go.Figure(data=traces, layout=layout)
    if fig.spec.layout_kwargs:
        out.update_layout(**fig.spec.layout_kwargs)
    return out
