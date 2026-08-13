#!/usr/bin/env python3
"""Write ../../figures/pipeline.png -- the paper's one figure.

The figure is the library's OWN output, not a redraw of it: `mirror_plot` is a
public spxtacular function and this script calls it exactly as a user would,
then renders the Plotly result through `save_figure`. For a package whose
visualization is one of the features being claimed, a matplotlib reconstruction
would illustrate the claim rather than demonstrate it.

Picked up by `just assets` on the filename pattern gen_*_figure.py.
"""
from __future__ import annotations

import spxtacular as spx

from _assets import record
from _demo import DATA_FILES, PAPER, SCAN, demo_spectra

OUT = PAPER / "figures" / "pipeline.png"

# Rendered size. The figure is placed at full text width (160 mm = 6.30 in), so
# 1100 px x scale 2 = 2200 px lands at ~350 dpi, above the 300 dpi floor
# `just prose-check` enforces. Plotly has no dpi setting; pixels and the placed
# width are the only two numbers that matter.
WIDTH, HEIGHT, SCALE = 1100, 560, 2.0


def main() -> int:
    raw, decon, neutral = demo_spectra()

    # Light mode explicitly: the theme has a global default that a user's
    # environment could have flipped, and a figure whose background colour
    # depended on that would churn its bytes between machines.
    fig = spx.mirror_plot(
        raw,
        decon,
        title="",
        show_charges=True,
        show_scores=False,
        max_labels=25,
        theme_mode="light",
        width=WIDTH,
        height=HEIGHT,
    )

    # The caption carries the title in the manuscript, so drop the one the plot
    # supplies for interactive use. mirror_plot falls back to its default for any
    # falsy title, so this cannot be done through the argument.
    fig.update_layout(title=None, margin=dict(t=30, b=50, l=60, r=20))

    OUT.parent.mkdir(exist_ok=True)
    spx.save_figure(fig, OUT, scale=SCALE)

    record(
        "fig.pipeline",
        str(OUT.relative_to(PAPER)),
        kind="figure",
        inputs=DATA_FILES,
        desc=(
            f"Deconvoluted MS2 scan {SCAN} of the example timsTOF run, coloured "
            f"by assigned charge, mirrored against the neutral masses decharge() "
            f"produces from it"
        ),
    )
    print(f"wrote {OUT.relative_to(PAPER)}  ({len(raw.mz)} peaks in)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
