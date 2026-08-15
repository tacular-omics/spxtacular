#!/usr/bin/env python3
"""Write ../../figures/graphical_abstract.png -- the paper's one graphic.

A graphical abstract rather than a data figure: the whole pipeline as one
left-to-right strip. Three panels -- the demo MS2 scan as read, its isotope
clusters coloured by assigned charge, and the neutral masses `decharge()`
returns -- joined by arrows carrying the method names, so the picture restates
the library's one-line story: spec.denoise().deconvolute().decharge().

Each panel is the library's OWN output, not a redraw of it: `plot_spectrum` is
a public spxtacular function and this script calls it exactly as a user would,
then composes the three Plotly figures onto one canvas. For a package whose
visualization is one of the features being claimed, a matplotlib
reconstruction would illustrate the claim rather than demonstrate it.

Sized at the ACS table-of-contents ratio (3.25 x 1.75 in) and rendered at
2600 x 1400 px, which is 800 dpi at TOC print size and ~410 dpi when placed at
full text width -- above the 300 dpi floor `just prose-check` enforces either
way. So the same file serves as the in-body figure and the submission's
graphical abstract without a second render that could drift.

Picked up by `just assets` on the filename pattern gen_*_figure.py.
"""
from __future__ import annotations

import plotly.graph_objects as go
import spxtacular as spx

from _assets import record
from _demo import DATA_FILES, PAPER, SCAN, demo_spectra

OUT = PAPER / "figures" / "graphical_abstract.png"

WIDTH, HEIGHT, SCALE = 1300, 700, 2.0

# Panel x-domains, with gaps between them wide enough for the arrow labels.
DOMAINS = [(0.00, 0.285), (0.36, 0.645), (0.72, 1.00)]

_AXIS_LINE = "#6b6a67"  # neutral baseline under the sticks; theme-agnostic
_TEXT = "#52514e"


def _panel(spec, **kwargs) -> go.Figure:
    """One panel, drawn by the library with every overlay silenced."""
    return spx.plot_spectrum(
        spec,
        title="",
        max_labels=0,
        show_scores=False,
        show_precursor=False,
        theme_mode="light",
        **kwargs,
    )


def main() -> int:
    raw, decon, neutral = demo_spectra()

    panels = [
        (_panel(raw, color=None), "centroid spectrum, as read"),
        (_panel(decon, color="charge"), "isotope clusters, by charge"),
        (_panel(neutral, color=None), "neutral masses"),
    ]

    combo = go.Figure()
    for i, (fig, _) in enumerate(panels):
        suffix = "" if i == 0 else str(i + 1)
        for trace in fig.data:
            trace.update(xaxis=f"x{suffix}", yaxis=f"y{suffix}")
            combo.add_trace(trace)

    axis_common = dict(
        showticklabels=False,
        ticks="",
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor=_AXIS_LINE,
        linewidth=1,
    )
    layout: dict = dict(
        template=panels[0][0].layout.template,
        width=WIDTH,
        height=HEIGHT,
        showlegend=False,
        margin=dict(t=86, b=72, l=14, r=14),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    for i in range(3):
        suffix = "" if i == 0 else str(i + 1)
        layout[f"xaxis{suffix}"] = dict(domain=list(DOMAINS[i]), anchor=f"y{suffix}", **axis_common)
        layout[f"yaxis{suffix}"] = dict(
            domain=[0.0, 1.0], anchor=f"x{suffix}", visible=False, rangemode="tozero"
        )
    combo.update_layout(**layout)

    annotations = [
        # The one-line story, across the top.
        dict(
            x=0.5, y=1.14, xref="paper", yref="paper", showarrow=False,
            text="<b>spxtacular</b>&nbsp;&nbsp;"
                 "<span style='font-family:monospace'>"
                 "spec.denoise().deconvolute().decharge()</span>",
            font=dict(size=21, color=_TEXT),
        ),
    ]
    # Captions under each panel.
    for (lo, hi), (_, label) in zip(DOMAINS, panels):
        annotations.append(dict(
            x=(lo + hi) / 2, y=-0.045, xref="paper", yref="paper", showarrow=False,
            yanchor="top", text=label, font=dict(size=15, color=_TEXT),
        ))
    # Arrows in the gaps, each carrying the methods that take you across it.
    for gap_center, methods in [
        (0.3225, ".denoise()<br>.deconvolute()"),
        (0.6825, ".decharge()"),
    ]:
        annotations.append(dict(
            x=gap_center, y=0.60, xref="paper", yref="paper", showarrow=False,
            text=f"<span style='font-family:monospace'>{methods}</span>",
            font=dict(size=13, color=_TEXT),
        ))
        annotations.append(dict(
            x=gap_center, y=0.45, xref="paper", yref="paper", showarrow=False,
            text="&#10230;",  # long rightwards arrow
            font=dict(size=26, color=_TEXT),
        ))
    combo.update_layout(annotations=annotations)

    OUT.parent.mkdir(exist_ok=True)
    spx.save_figure(combo, OUT, scale=SCALE)

    record(
        "fig.abstract",
        str(OUT.relative_to(PAPER)),
        kind="figure",
        inputs=DATA_FILES,
        desc=(
            f"Graphical abstract: MS2 scan {SCAN} of the example timsTOF run as "
            f"read, deconvoluted with charge colouring, and as neutral masses, "
            f"joined by the method calls between the stages"
        ),
    )
    print(f"wrote {OUT.relative_to(PAPER)}  ({len(raw.mz)} peaks in)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
