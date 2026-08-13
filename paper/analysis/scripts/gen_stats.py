#!/usr/bin/env python3
"""Write stats.json at the manuscript root: every number the prose states.

Each value is measured by running spxtacular, never typed in. Two kinds appear
here:

  * facts about the library itself (version, public API size), read out of the
    installed package, so a release cannot leave the paper describing the
    previous one;
  * results of running the pipeline on the demo spectrum in _demo.py, which is
    the same run the figure and the SI table are drawn from.

Nothing here is timed. A benchmark number would differ on every machine and on
every re-run, and `just check-stats-deep` re-derives values and diffs them, so a
wall-clock measurement would fail the gate forever for no defect. Speed claims,
if the paper makes any, belong in a separate benchmark with its own error bars.

ONE of these per project: it writes a single file, so a second script would
clobber the first.
"""
from __future__ import annotations

import numpy as np
import spxtacular as spx

from _demo import CHARGE_RANGE, MIN_SCORE, SCAN, TOLERANCE, DATA_FILES, demo_spectra
from _stats import Stats


def main() -> int:
    raw, decon, neutral = demo_spectra()
    assigned = decon.charge > 0
    tic = float(raw.intensity.sum())

    st = Stats()

    # --- the library ------------------------------------------------------
    st.add("lib.version", spx.__version__,
           desc="spxtacular version the paper describes")

    st.add("lib.public_symbols", len(spx.__all__), fmt=",",
           desc="Names exported from the top-level spxtacular namespace",
           sign="+")

    # --- how the demo pipeline was run ------------------------------------
    st.add("demo.scan", SCAN, fmt=",",
           desc="MS2 scan number used for the figure, table and numbers",
           sign="+")

    st.add("demo.tolerance_ppm", TOLERANCE, fmt=".0f", unit="ppm",
           desc="Deconvolution matching tolerance",
           sign="+", between=(1, 100))

    st.add("demo.min_score", MIN_SCORE, fmt=".2f",
           desc="Minimum Bhattacharyya isotope score for a cluster to be kept",
           between=(0, 1))

    st.add("demo.max_charge_searched", CHARGE_RANGE[1], fmt=",",
           desc="Highest charge state the deconvolution searched",
           sign="+")

    # --- what it found ----------------------------------------------------
    st.add("demo.raw_peaks", int(len(raw.mz)), fmt=",",
           desc="Centroid peaks in the demo MS2 spectrum as read from the file",
           sign="+")

    st.add("demo.clusters", int(assigned.sum()), fmt=",",
           desc="Isotope clusters assigned a charge state",
           sign="+")

    st.add("demo.neutral_masses", int(len(neutral.mz)), fmt=",",
           desc="Neutral masses surviving the score filter, after decharge()",
           sign="+")

    st.add("demo.max_charge_found", int(decon.charge.max()), fmt=",",
           desc="Highest charge state actually assigned in the demo spectrum",
           sign="+")

    # Guarded as a fraction rather than a count: the prose calls it a good
    # profile match, and a re-run that dropped it below half would contradict
    # the sentence rather than merely change a digit.
    st.add("demo.mean_score", float(np.mean(decon.iso_score[assigned])), fmt=".2f",
           desc="Mean isotope profile score across the assigned clusters",
           between=(0.5, 1.0))

    st.add("demo.tic_share", float(decon.intensity[assigned].sum()) / tic,
           fmt=".1%",
           desc="Share of the spectrum's total ion current in assigned clusters",
           sign="+", between=(0, 1))

    st.add("demo.max_neutral_mass", float(neutral.mz.max()), fmt=",.0f", unit="Da",
           desc="Largest neutral mass recovered from the demo spectrum",
           sign="+", between=(100, 100000))

    st.add("demo.precursor_charge", int(raw.precursors[0].charge), fmt=",",
           desc="Charge of the precursor selected for the demo MS2 spectrum",
           sign="+")

    return st.write(inputs=DATA_FILES)


if __name__ == "__main__":
    raise SystemExit(main())
