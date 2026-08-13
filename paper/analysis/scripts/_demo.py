"""The one spectrum this paper's figure, table and numbers are all built from.

WHY IT IS SHARED. The figure, the Supporting Information table and every number
in the prose describe the SAME run of the SAME pipeline. Written three times
they would drift the first time a tolerance changed and only one script was
re-run; written once here, `just assets` cannot produce a figure that disagrees
with the sentence beside it.

WHAT THE DATA IS. `tests/data/example_dda.d` in the spxtacular repository this
manuscript sits inside: a real Bruker timsTOF DDA acquisition, committed to the
repository as the library's own test fixture. Using it rather than a simulated
envelope means the figure shows what the software does to instrument data, and
any reader can open the same file.

Every parameter below is fixed rather than tuned per consumer, so the pipeline
is reproducible from the paper alone.
"""
from __future__ import annotations

from pathlib import Path

import spxtacular as spx

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent                    # analysis/scripts/ -> analysis/ -> paper/
# The manuscript lives inside the library's repository, so the demo data is one
# level above it. Declared to record() as this same relative path.
DATA_REL = "../tests/data/example_dda.d"
DATA = PAPER / DATA_REL

# The Bruker .d directory is two files; assets.json hashes files, not directories.
DATA_FILES = [f"{DATA_REL}/analysis.tdf", f"{DATA_REL}/analysis.tdf_bin"]

# The MS2 spectrum used throughout. Chosen once, by number rather than by any
# "most intense" rule, so re-running never silently switches spectra.
SCAN = 2

# Deconvolution parameters, stated in the paper. One definition, so the prose,
# the figure and the table cannot disagree about how the numbers were produced.
CHARGE_RANGE = (1, 4)
TOLERANCE = 15.0
TOLERANCE_TYPE = "ppm"
MIN_SCORE = 0.5


def demo_spectra() -> tuple[spx.MsnSpectrum, spx.Spectrum, spx.Spectrum]:
    """Return (centroid, deconvoluted, neutral) for the demo MS2 spectrum.

    The three stages the paper describes, in the order it describes them:
    the vendor centroids as read; the same peaks with a charge state and an
    isotopic-profile score attached; and the high-confidence clusters collapsed
    to neutral masses.
    """
    with spx.Reader(str(DATA)) as reader:
        raw = next(s for s in reader.ms2 if s.scan_number == SCAN)

    decon = raw.deconvolute(
        charge_range=CHARGE_RANGE,
        tolerance=TOLERANCE,
        tolerance_type=TOLERANCE_TYPE,
        min_score=MIN_SCORE,
    )
    neutral = decon.filter(min_score=MIN_SCORE).decharge()
    return raw, decon, neutral
