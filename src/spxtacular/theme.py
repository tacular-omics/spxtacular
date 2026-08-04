"""
Visual theme for spxtacular plots: palettes, plotly templates, and colour lookup.

This is the single source of truth for colour. ``plot_table.py`` and
``visualization.py`` both read from here, so a palette change lands everywhere at
once rather than being "kept in sync" by comment.

Colour is assigned by the *job* it does, not by taste:

``ion type``
    Nominal categorical — which fragment series a peak belongs to. Eight fixed
    hues, assigned by the proteomics convention (b blue, y red, a green, c teal,
    x purple, z orange) and never cycled; anything else folds into a neutral
    "other" rather than inventing a ninth hue.
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

# The eight validated hues. Which ion series takes which is set by
# _ION_SLOT_INDEX below, following the proteomics convention rather than this
# order -- the order here is only the palette's own adjacent-pair guarantee.
_CATEGORICAL: dict[ThemeMode, list[str]] = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}

#: Ion series in priority order. Anything not listed folds to the neutral "other".
#: Used for the texture channel and to break ties when one peak matches several ions.
_ION_SLOTS: tuple[str, ...] = ("b", "y", "a", "c", "x", "z", "p", "i")

#: Which categorical slot each ion series takes.
#:
#: This is not the slot *order* -- it follows the long-standing proteomics
#: convention instead, so a spectrum from spxtacular is readable by anyone used to
#: Skyline, MetaDraw, IPSA or spectrum_utils:
#:
#:     b blue · y red · a green · c teal · x purple · z orange
#:
#: The hues are this palette's own validated steps, picked from the family each
#: tool uses rather than copied, so the convention is honoured without giving up
#: the colour-vision checks. b/y -- the pair that co-occurs in nearly every
#: spectrum -- separates at CVD ΔE 21.6 light / 19.2 dark, comfortably above the
#: ≥8 target and better than the blue/orange pairing it replaced.
#:
#: The one caveat is a-vs-y (green vs red), the classic confusion pair, at ΔE 7.2
#: in light mode: inside the 6-8 band that is legal *only* alongside secondary
#: encoding. Annotated spectra always carry direct mzPAF ion labels, which
#: supplies it; ``texture=True`` adds dash patterns if you need more.
_ION_SLOT_INDEX: dict[str, int] = {
    "b": 0,  # blue
    "y": 7,  # red
    "a": 5,  # green
    "c": 2,  # aqua, standing in for the conventional teal
    "x": 6,  # violet, standing in for the conventional purple
    "z": 1,  # orange
    "p": 4,  # magenta  (precursor)
    "i": 3,  # yellow   (immonium)
}


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
# Texture — the non-colour channel
# ---------------------------------------------------------------------------

#: Dash patterns per ion slot, for print, forced-colours, and readers who cannot
#: separate two hues. Opt-in (``texture=True``) rather than always on: at stick
#: density dashes add noise, and the direct ion labels already carry identity in
#: the normal case.
_ION_DASH: tuple[str, ...] = (
    "solid",
    "dash",
    "dot",
    "dashdot",
    "longdash",
    "longdashdot",
    "solid",
    "dash",
)


def ion_dash(ion_type: str) -> str:
    """Dash pattern for a fragment ion series (the texture channel)."""
    key = str(ion_type).lower()
    return _ION_DASH[_ION_SLOTS.index(key)] if key in _ION_SLOTS else "solid"


def set_palette(
    *,
    categorical: dict[ThemeMode, list[str]] | None = None,
    charge_ramp: dict[ThemeMode, list[str]] | None = None,
    sequential: dict[ThemeMode, list[list]] | None = None,
) -> None:
    """Replace a palette with your own, for brand colours.

    Each argument takes a ``{"light": [...], "dark": [...]}`` mapping and
    replaces that palette wholesale in both modes.

    The shipped palettes were checked with a colour-vision-deficiency validator
    (protanopia and deuteranopia) against both surfaces. **A substituted palette
    is not checked** -- validate your own hues before relying on them, or you
    lose the property the defaults were chosen for. Categorical hues want a
    fixed order with adjacent pairs kept apart; a charge ramp wants one hue with
    monotone lightness.

    Raises
    ------
    ValueError
        If a mapping is missing a mode, or a categorical palette has fewer
        entries than there are ion slots.
    """
    for name, value in (("categorical", categorical), ("charge_ramp", charge_ramp), ("sequential", sequential)):
        if value is None:
            continue
        missing = {"light", "dark"} - set(value)
        if missing:
            raise ValueError(f"{name} palette must define both modes; missing {sorted(missing)}")

    if categorical is not None:
        for mode, hues in categorical.items():
            if len(hues) < len(_ION_SLOTS):
                raise ValueError(
                    f"categorical palette for {mode!r} needs at least {len(_ION_SLOTS)} hues, got {len(hues)}"
                )
        _CATEGORICAL.update(categorical)
    if charge_ramp is not None:
        _CHARGE_RAMP.update(charge_ramp)
    if sequential is not None:
        _SEQUENTIAL.update(sequential)


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

    Follows the proteomics convention -- b blue, y red, a green, c teal,
    x purple, z orange -- using this palette's validated steps. Unrecognised
    series, including internal fragments with two-letter types like ``"by"``,
    fold to the neutral colour rather than inventing a ninth hue.
    """
    mode = resolve_mode(theme)
    key = str(ion_type).lower()
    slot = _ION_SLOT_INDEX.get(key)
    if slot is not None:
        return _CATEGORICAL[mode][slot]
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

    # A crosshair on the m/z axis. Sticks are ~1.5px wide, so without it the
    # reader has to hit a hairline to find out where the pointer is; the spike
    # answers "which m/z am I on" without any hit at all. Solid, one step off the
    # surface, so it reads as chrome rather than as data.
    x_axis = {
        **axis_common,
        "showspikes": True,
        "spikemode": "across",
        "spikesnap": "cursor",
        "spikethickness": 1,
        "spikedash": "solid",
        "spikecolor": axis,
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
            # Fill the container rather than a fixed default box, so figures sit
            # correctly in docs pages and notebooks.
            "autosize": True,
            # Hit target bigger than the mark: a 1.5px stick is a pinpoint. The
            # hit layer added by the renderer does the rest.
            "hovermode": "closest",
            "hoverdistance": 24,
            "spikedistance": -1,
            "xaxis": x_axis,
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
