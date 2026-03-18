"""Run this script to plot example spectra.

Outputs HTML files to the plots/ folder (created if absent).
Pass --show to open each plot in the browser instead.

Usage:
    uv run python plot_example.py           # write plots/
    uv run python plot_example.py --show    # open in browser
"""
import sys
from pathlib import Path

import peptacular as pt

import spxtacular as spx
from tests.data import EXAMPLE_SPECTRUM

SHOW = "--show" in sys.argv

_out_arg = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--out"), None)
OUT_DIR = Path(_out_arg) if _out_arg else Path("plots")
OUT_DIR.mkdir(exist_ok=True)


def save_or_show(fig, name: str) -> None:
    if SHOW:
        fig.show()
    else:
        path = OUT_DIR / f"{name}.html"
        fig.write_html(path)
        print(f"wrote {path}")


# ── spectra ────────────────────────────────────────────────────────────────────
raw = EXAMPLE_SPECTRUM
decon = raw.deconvolute(charge_range=(1, 2), tolerance=500, tolerance_type="ppm")
decon_filtered = raw.deconvolute(charge_range=(1, 2), tolerance=500, tolerance_type="ppm", min_score=0.5)
neutral = decon.decharge()
neutral_filtered = decon_filtered.decharge()

# ── basic spectrum plots ───────────────────────────────────────────────────────
for spec, name in [
    (raw, "raw"),
    (decon, "deconvoluted"),
    (decon_filtered, "deconvoluted_filtered"),
    (neutral, "neutral_mass"),
    (neutral_filtered, "neutral_mass_filtered"),
]:
    save_or_show(spec.plot(title=name), name)

# ── mirror plots (raw below, deconvoluted above) ───────────────────────────────
save_or_show(
    spx.mirror_plot(raw, decon, title="Mirror – all clusters"),
    "mirror",
)
save_or_show(
    spx.mirror_plot(raw, decon_filtered, title="Mirror – min_score=0.5"),
    "mirror_filtered",
)

# ── annotated spectrum ─────────────────────────────────────────────────────────
# Use a peptide whose fragment m/z values land in the example spectrum range
# (~280–1900 Da).  PEPTIDE is a tryptic peptide with charge-2 precursor in
# that window; adjust as needed.
PEPTIDE = "FDSFGDLSSASAIMGNPK"

fragments = pt.fragment(
    PEPTIDE,
    ion_types=("b", "y"),
    charges=(1, 2),
    monoisotopic=True,
)

save_or_show(
    spx.annotate_spectrum(
        raw,
        fragments,
        mz_tol=5,
        mz_tol_type="da",
        title=f"Annotated – {PEPTIDE}",
    ),
    "annotated",
)
