"""
Visual theme for spxtacular plots: palettes, plotly templates, and colour lookup.

This is the single source of truth for colour. ``plot_table.py`` and
``visualization.py`` both read from here, so a palette change lands everywhere at
once rather than being "kept in sync" by comment.

Colour is assigned by the *job* it does, not by taste:

``ion type``
    Nominal categorical — which fragment series a peak belongs to. Eight fixed
    hues in a fixed order, assigned in sequence and never cycled; anything past
    the eighth folds into a neutral "other" rather than inventing a ninth hue.
``charge state``
    **Ordinal**, not categorical — 1+, 2+, 3+ have a natural order, so charge
    takes a single-hue ramp that runs light to dark. The reader sees the
    ordering in the colour. Unassigned peaks (``charge == -1``) are neutral
    grey: absence of identity, not another category.
``iso_score`` / ``ion mobility``
    Sequential magnitude — a single hue, light to dark, with a colourbar.
``unmatched``
    Recessive grey. Unmatched peaks are context, not subject.

Every palette here was checked with a colour-vision-deficiency validator in both
light and dark modes (protanopia and deuteranopia, Machado-Oliveira-Fernandes at
severity 1.0). Three light-mode categorical hues sit below 3:1 against the light
surface; matched peaks always carry visible ion labels, which is the required
relief. Do not substitute eyeballed hex values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import plotly.graph_objects as go

ThemeMode = Literal["light", "dark"]

#: Default mode used when a caller does not pass ``theme=``.
_DEFAULT_MODE: ThemeMode = "light"


# ---------------------------------------------------------------------------
# Surfaces and ink
# ---------------------------------------------------------------------------

_SURFACE: dict[ThemeMode, str] = {"light": "#fcfcfb", "dark": "#1a1a19"}
_TEXT_PRIMARY: dict[ThemeMode, str] = {"light": "#0b0b0b", "dark": "#ffffff"}
_TEXT_SECONDARY: dict[ThemeMode, str] = {"light": "#52514e", "dark": "#c3c2b7"}
_TEXT_MUTED: dict[ThemeMode, str] = {"light": "#8a8983", "dark": "#8f8e86"}
#: Gridlines sit one step off the surface — present when looked for, invisible otherwise.
_GRID: dict[ThemeMode, str] = {"light": "#ececea", "dark": "#2b2b29"}
_AXIS: dict[ThemeMode, str] = {"light": "#d8d7d3", "dark": "#3a3a37"}

#: Peaks with no annotation. Deliberately low-contrast: context, not subject.
_UNMATCHED: dict[ThemeMode, str] = {"light": "#c9c8c3", "dark": "#4a4a46"}
#: Deconvolution singletons (charge == -1) and any "other" category.
_NEUTRAL: dict[ThemeMode, str] = {"light": "#9a9993", "dark": "#6e6d67"}


# ---------------------------------------------------------------------------
# Categorical palette — fragment ion series
# ---------------------------------------------------------------------------

# Fixed hue order. The ordering is the CVD-safety mechanism, not cosmetics.
# b and y take the first two slots because they are by far the most common pair
# (CID/HCD), so the pair that co-occurs most often is the most separable.
_CATEGORICAL: dict[ThemeMode, list[str]] = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}

#: Ion series in fixed slot order. Anything not listed folds to the neutral "other".
_ION_SLOTS: tuple[str, ...] = ("b", "y", "a", "c", "x", "z", "p", "i")


# ---------------------------------------------------------------------------
# Ordinal ramp — charge state
# ---------------------------------------------------------------------------

# Single blue hue, monotone lightness. Light mode runs light->dark as charge
# rises; dark mode runs dark->light so the ramp stays legible on a dark surface.
# Both directions validated: monotone L, adjacent dL >= 0.06, end step clears 2:1.
_CHARGE_RAMP: dict[ThemeMode, list[str]] = {
    "light": ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"],
    "dark": ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"],
}

# ---------------------------------------------------------------------------
# Sequential ramp — iso_score, ion mobility
# ---------------------------------------------------------------------------

#: One hue, light to dark. Never a rainbow: a multi-hue ramp invents structure.
_SEQUENTIAL: dict[ThemeMode, list[list]] = {
    "light": [
        [0.0, "#cde2fb"],
        [0.25, "#9ec5f4"],
        [0.5, "#5598e7"],
        [0.75, "#2a78d6"],
        [1.0, "#104281"],
    ],
    "dark": [
        [0.0, "#184f95"],
        [0.25, "#256abf"],
        [0.5, "#3987e5"],
        [0.75, "#86b6ef"],
        [1.0, "#cde2fb"],
    ],
}

_FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


# ---------------------------------------------------------------------------
# Public lookups
# ---------------------------------------------------------------------------


def resolve_mode(theme: ThemeMode | None = None) -> ThemeMode:
    """Return the effective theme mode."""
    return theme if theme is not None else _DEFAULT_MODE


def set_plot_theme(mode: ThemeMode) -> None:
    """Set the default theme mode for every subsequent plot.

    Parameters
    ----------
    mode:
        ``"light"`` or ``"dark"``.
    """
    global _DEFAULT_MODE
    if mode not in ("light", "dark"):
        raise ValueError(f"theme must be 'light' or 'dark', got {mode!r}")
    _DEFAULT_MODE = mode


def surface(theme: ThemeMode | None = None) -> str:
    """Chart surface colour for the given mode."""
    return _SURFACE[resolve_mode(theme)]


def unmatched_color(theme: ThemeMode | None = None) -> str:
    """Colour for peaks carrying no annotation."""
    return _UNMATCHED[resolve_mode(theme)]


def neutral_color(theme: ThemeMode | None = None) -> str:
    """Colour for singletons and any category past the eighth slot."""
    return _NEUTRAL[resolve_mode(theme)]


def text_color(level: Literal["primary", "secondary", "muted"] = "secondary", theme: ThemeMode | None = None) -> str:
    """Ink colour. Labels never wear the series colour — identity comes from the mark."""
    mode = resolve_mode(theme)
    return {"primary": _TEXT_PRIMARY, "secondary": _TEXT_SECONDARY, "muted": _TEXT_MUTED}[level][mode]


def ion_color(ion_type: str, theme: ThemeMode | None = None) -> str:
    """Colour for a fragment ion series.

    Assigned from a fixed slot order, never cycled. Unrecognised series --
    including internal fragments, which have two-letter types like ``"by"`` --
    fold to the neutral colour rather than inventing a ninth hue.
    """
    mode = resolve_mode(theme)
    key = str(ion_type).lower()
    if key in _ION_SLOTS:
        return _CATEGORICAL[mode][_ION_SLOTS.index(key)]
    return _NEUTRAL[mode]


def sequential_scale(theme: ThemeMode | None = None) -> list[list]:
    """Plotly colourscale for continuous magnitude (iso_score, ion mobility)."""
    return _SEQUENTIAL[resolve_mode(theme)]


def charge_color(charge: int, theme: ThemeMode | None = None) -> str:
    """Colour for a charge state.

    Charge is ordinal, so this is a single-hue ramp rather than a categorical
    cycle: the reader sees 1+ < 2+ < 3+ in the lightness. Charges beyond the
    ramp clamp to its dark end instead of wrapping around to an earlier colour
    (the old cycle made z=1 and z=11 identical).

    ``charge <= 0`` -- singletons (-1) and decharged peaks (0) -- is neutral grey.
    """
    mode = resolve_mode(theme)
    if charge <= 0:
        return _NEUTRAL[mode]
    ramp = _CHARGE_RAMP[mode]
    return ramp[min(charge - 1, len(ramp) - 1)]


# ---------------------------------------------------------------------------
# Plotly template
# ---------------------------------------------------------------------------


def template(theme: ThemeMode | None = None) -> go.layout.Template:
    """Build the plotly template for the given mode.

    Chrome is deliberately recessive: no panel fill, solid hairline gridlines one
    step off the surface, no vertical grid (m/z position is read from the peak,
    not from a grid), and generous margins. The data is the only loud thing.
    """
    import plotly.graph_objects as go

    mode = resolve_mode(theme)
    surf = _SURFACE[mode]
    grid = _GRID[mode]
    axis = _AXIS[mode]

    axis_common = {
        "showgrid": False,
        "zeroline": False,
        "linecolor": axis,
        "linewidth": 1,
        "ticks": "outside",
        "ticklen": 4,
        "tickwidth": 1,
        "tickcolor": axis,
        "tickfont": {"size": 11, "color": _TEXT_MUTED[mode], "family": _FONT_FAMILY},
        "title": {"font": {"size": 12, "color": _TEXT_SECONDARY[mode], "family": _FONT_FAMILY}},
        "automargin": True,
    }

    return go.layout.Template(
        layout={
            "paper_bgcolor": surf,
            "plot_bgcolor": surf,
            "font": {"family": _FONT_FAMILY, "size": 12, "color": _TEXT_SECONDARY[mode]},
            "title": {
                "font": {"size": 16, "color": _TEXT_PRIMARY[mode], "family": _FONT_FAMILY},
                "x": 0.0,
                "xanchor": "left",
                "y": 0.97,
                "yanchor": "top",
                # x=0 anchors to the paper edge, so without this the first glyph
                # sits flush against the image border.
                "pad": {"l": 8, "t": 4},
            },
            "margin": {"l": 72, "r": 28, "t": 64, "b": 60},
            "xaxis": {**axis_common},
            # Horizontal grid only: it carries the intensity values the peaks are
            # measured against. A vertical grid would just add ink.
            "yaxis": {**axis_common, "showgrid": True, "gridcolor": grid, "gridwidth": 1},
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.0,
                "xanchor": "right",
                "x": 1.0,
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "font": {"size": 11, "color": _TEXT_SECONDARY[mode], "family": _FONT_FAMILY},
                "itemsizing": "constant",
            },
            "hoverlabel": {
                "bgcolor": surf,
                "bordercolor": axis,
                "font": {"family": _FONT_FAMILY, "size": 11, "color": _TEXT_PRIMARY[mode]},
                "align": "left",
            },
            "colorway": _CATEGORICAL[mode],
        }
    )


def apply(fig: go.Figure, theme: ThemeMode | None = None) -> go.Figure:
    """Apply the spxtacular template to a figure in place and return it."""
    fig.update_layout(template=template(theme))
    return fig
