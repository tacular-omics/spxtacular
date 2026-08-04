"""Run this script to plot example spectra.

Outputs HTML files to the plots/ folder (created if absent).
Pass --show to open each plot in the browser instead.

Usage:
    uv run python plot_example.py           # write plots/
    uv run python plot_example.py --show    # open in browser
"""
import sys
from pathlib import Path

import numpy as np
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
# The annotation figures need a spectrum the peptide genuinely explains.
#
# EXAMPLE_SPECTRUM is real data whose true identification we do not have, and
# PEPTIDE does not match it: at a realistic 10-20 ppm tolerance it matches
# *nothing*. Annotating it anyway needed a 5 Da window (~10,000 ppm at m/z 500),
# which produces 57 purely coincidental "matches" -- and a mass-error plot of
# coincidental matches just draws the tolerance window, teaching the reader
# something false.
#
# So the annotation-based figures below use a simulated MS2: this peptide's own
# fragments, displaced by a few ppm the way a real instrument would, over a noise
# floor. The annotations, the mass errors and the coverage ladder are then all
# genuine. The raw/deconvolution/mirror figures above still use the real spectrum,
# where no peptide is involved.
PEPTIDE = "FDSFGDLSSASAIMGNPK"

fragments = pt.fragment(
    PEPTIDE,
    ion_types=("b", "y"),
    charges=(1, 2),
    monoisotopic=True,
)


def _simulated_ms2(frags, ppm_error=5.0, n_noise=140, seed=0):
    """An MS2 that the peptide actually explains, with realistic ppm scatter."""
    rng = np.random.default_rng(seed)
    frag_mz = np.array([f.mz for f in frags], dtype=np.float64)
    # Keep a realistic subset: not every theoretical ion is observed.
    keep = rng.random(len(frag_mz)) < 0.75
    frag_mz = frag_mz[keep]
    # Per-peak mass error, normally distributed within a few ppm.
    observed = frag_mz * (1.0 + rng.normal(0.0, ppm_error / 3.0, frag_mz.size) / 1e6)
    signal = rng.lognormal(10.5, 0.9, observed.size)
    noise_mz = rng.uniform(observed.min() - 40, observed.max() + 40, n_noise)
    noise_i = rng.lognormal(7.5, 0.8, n_noise)
    mz = np.concatenate([observed, noise_mz])
    inten = np.concatenate([signal, noise_i])
    order = np.argsort(mz)
    return spx.Spectrum(mz=mz[order], intensity=inten[order])


annot_spec = _simulated_ms2(fragments)
annot_decon = annot_spec.deconvolute(charge_range=(1, 2), tolerance=20, tolerance_type="ppm")
# A tolerance a real search would use, on data the peptide really explains.
ANNOT_TOL, ANNOT_TOL_TYPE = 20, "ppm"

save_or_show(
    spx.annotate_spectrum(
        annot_spec,
        fragments,
        tolerance=ANNOT_TOL,
        tolerance_type=ANNOT_TOL_TYPE,
        title=f"Annotated – {PEPTIDE}",
    ),
    "annotated",
)

# ── sequence coverage ladder ───────────────────────────────────────────────────
# Where along the peptide the evidence sits, which the spectrum alone can't show.
save_or_show(
    spx.sequence_coverage_plot(
        annot_spec,
        PEPTIDE,
        fragments,
        tolerance=ANNOT_TOL,
        tolerance_type=ANNOT_TOL_TYPE,
    ),
    "sequence_coverage",
)

# ── dark mode ──────────────────────────────────────────────────────────────────
# The dark palette is its own set of steps for the dark surface, not an inversion.
save_or_show(
    spx.annotate_spectrum(
        annot_spec,
        fragments,
        tolerance=ANNOT_TOL,
        tolerance_type=ANNOT_TOL_TYPE,
        title=f"Annotated (dark) – {PEPTIDE}",
        theme_mode="dark",
    ),
    "annotated_dark",
)

# ── log intensity ──────────────────────────────────────────────────────────────
# Compresses dynamic range so low-abundance matched ions stay visible next to a
# dominant base peak.
save_or_show(
    spx.annotate_spectrum(
        annot_spec,
        fragments,
        tolerance=ANNOT_TOL,
        tolerance_type=ANNOT_TOL_TYPE,
        title=f"Annotated, log intensity – {PEPTIDE}",
        intensity_transform="log",
    ),
    "annotated_log",
)

# ── mass error plot ────────────────────────────────────────────────────────────
# Bubble chart of fragment mass error vs m/z; bubble size tracks peak intensity.
save_or_show(
    spx.mass_error_plot(
        annot_spec,
        fragments,
        tolerance=ANNOT_TOL,
        tolerance_type=ANNOT_TOL_TYPE,
        unit="ppm",
        title=f"Mass errors – {PEPTIDE}",
    ),
    "mass_errors",
)

# ── facet plot ─────────────────────────────────────────────────────────────────
# Annotated spectrum + mass errors + a mirror panel, on a shared m/z axis.
save_or_show(
    spx.facet_plot(
        annot_spec,
        fragments,
        mirror_spectrum=annot_decon,
        tolerance=ANNOT_TOL,
        tolerance_type=ANNOT_TOL_TYPE,
        title=f"Facet – {PEPTIDE}",
    ),
    "facet",
)
