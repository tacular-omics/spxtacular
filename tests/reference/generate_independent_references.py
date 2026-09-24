"""Generate ``independent_references.json`` from sources that do not use spxtacular.

Run from the repository root with::

    uv run --no-project --with pyteomics==5.0.1 --with scipy==1.18.1 --with numpy \
        python tests/reference/generate_independent_references.py

Produced 2026-09-23 with pyteomics 5.0.1, scipy 1.18.1 and numpy 2.x on Python 3.12.
Neither spxtacular nor peptacular is imported. The sources are:

* Isotope envelopes: pyteomics ``mass.isotopologues`` (explicit isotopologue
  enumeration over the NIST/IUPAC abundances in ``pyteomics.mass.nist_mass``),
  aggregated to nominal neutron offsets; and, for compositions too large to
  enumerate, a direct polynomial convolution of per-element isotope
  distributions (repeated squaring), which is independent of the BRAIN
  Newton-Girard recurrence spxtacular uses.
* Averagine: Senko, Beu & McLafferty (1995), J. Am. Soc. Mass Spectrom. 6, 229,
  C4.9384 H7.7583 N1.3577 O1.4773 S0.0417 per 111.1254 Da.
* Ion m/z: CODATA 2018 proton and electron masses and AME2020 atomic masses,
  written out below; pyteomics ``calculate_mass(..., charge=z)`` as a cross-check.
* Fragment ions: pyteomics ``fast_mass`` b/y ion m/z.
* Binomial survival: ``scipy.stats.binom.logsf``.
* Deconvolution: pyteomics peptide compositions and monoisotopic masses, with
  isotope intensities from the enumerated envelopes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from pyteomics import mass
from scipy.stats import binom

# CODATA 2018 (https://physics.nist.gov/cuu/Constants/) in unified atomic mass units.
CODATA_PROTON_MASS = 1.007276466621
CODATA_ELECTRON_MASS = 5.48579909065e-4
# AME2020 atomic masses (Wang et al., Chinese Phys. C 45, 030003).
AME_NA23 = 22.989769282
AME_N14 = 14.003074004
AME_H1 = 1.00782503224

N_PEAKS = 16

# IUPAC representative isotopic compositions (the same values pyteomics ships).
ABUNDANCES = {
    "C": [0.9893, 0.0107],
    "H": [0.999885, 0.000115],
    "N": [0.99636, 0.00364],
    "O": [0.99757, 0.00038, 0.00205],
    "P": [1.0],
    "S": [0.9499, 0.0075, 0.0425, 0.0, 0.0001],
}


def _truncate(values: np.ndarray) -> np.ndarray:
    return values[:N_PEAKS]


def _power(dist: np.ndarray, count: int) -> np.ndarray:
    result = np.array([1.0])
    base = dist.copy()
    while count:
        if count & 1:
            result = _truncate(np.convolve(result, base))
        base = _truncate(np.convolve(base, base))
        count >>= 1
    return result


def convolution_envelope(composition: dict[str, int]) -> list[float]:
    total = np.array([1.0])
    for element, count in composition.items():
        total = _truncate(np.convolve(total, _power(np.asarray(ABUNDANCES[element]), int(count))))
    out = np.zeros(N_PEAKS)
    out[: len(total)] = total
    return (out / out.sum()).tolist()


def pyteomics_envelope(composition: dict[str, int]) -> list[float]:
    formula = "".join(f"{element}{count}" for element, count in composition.items())
    mono = mass.calculate_mass(formula=formula)
    out = np.zeros(N_PEAKS)
    for isotopologue, abundance in mass.isotopologues(
        formula=formula, report_abundance=True, isotope_threshold=1e-7, overall_threshold=1e-14
    ):
        offset = round(mass.calculate_mass(composition=isotopologue) - mono)
        if offset < N_PEAKS:
            out[offset] += abundance
    return (out / out.sum()).tolist()


def peptide_composition(sequence: str) -> dict[str, int]:
    comp = mass.Composition(sequence=sequence)
    return {element: int(comp[element]) for element in "CHNOS" if comp[element]}


def isotope_cases() -> list[dict]:
    # Small enough for pyteomics to enumerate every isotopologue in seconds. The isotope
    # threshold only drops isotopes with zero natural abundance.
    small = {
        "water": {"H": 2, "O": 1},
        "glucose": {"C": 6, "H": 12, "O": 6},
        "cholesterol": {"C": 27, "H": 46, "O": 1},
        "ATP": {"C": 10, "H": 16, "N": 5, "O": 13, "P": 3},
    }
    large = {
        "MCM": peptide_composition("MCM"),
        "PEPTIDE": peptide_composition("PEPTIDE"),
        "angiotensin_II": peptide_composition("DRVYIHPF"),
        "PC_34_1": {"C": 42, "H": 82, "N": 1, "O": 8, "P": 1},
        "all_20_residues": peptide_composition("ACDEFGHIKLMNPQRSTVWY"),
        "ubiquitin_like": {"C": 378, "H": 629, "N": 105, "O": 118, "S": 1},
        "sulfur_rich": {"C": 200, "H": 300, "N": 50, "O": 60, "S": 20},
        "carbon_2000": {"C": 2000},
    }
    cases = []
    for name, comp in small.items():
        cases.append(
            {
                "name": name,
                "composition": comp,
                "pyteomics": pyteomics_envelope(comp),
                "convolution": convolution_envelope(comp),
            }
        )
    for name, comp in large.items():
        cases.append({"name": name, "composition": comp, "pyteomics": None, "convolution": convolution_envelope(comp)})
    return cases


def averagine_cases() -> list[dict]:
    senko = {"C": 4.9384, "H": 7.7583, "N": 1.3577, "O": 1.4773, "S": 0.0417}
    cases = []
    for target in (500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0):
        units = target / 111.1254
        comp = {element: round(rate * units) for element, rate in senko.items()}
        comp = {element: count for element, count in comp.items() if count}
        cases.append({"mass": target, "senko_composition": comp, "senko_envelope": convolution_envelope(comp)})
    return cases


def real_peptide_cases() -> list[dict]:
    sequences = ["SAMPLER", "LVNELTEFAK", "HLVDEPQNLIK", "YLYEIARRHPYFYAPELLFFAK", "DTHKSEIAHRFKDLGEEHFK"]
    cases = []
    for sequence in sequences:
        comp = peptide_composition(sequence)
        cases.append(
            {
                "sequence": sequence,
                "monoisotopic_mass": mass.calculate_mass(sequence=sequence),
                "composition": comp,
                "envelope": convolution_envelope(comp),
            }
        )
    return cases


def ion_mz_cases() -> list[dict]:
    na_cation = AME_NA23 - CODATA_ELECTRON_MASS
    nh4_cation = AME_N14 + 4 * AME_H1 - CODATA_ELECTRON_MASS
    carriers = {
        "protonated": CODATA_PROTON_MASS,
        "deprotonated": -CODATA_PROTON_MASS,
        "sodiated": na_cation,
        "ammoniated": nh4_cation,
    }
    cases = []
    for neutral in (0.0, 18.0105646863, 500.25, 1234.5678, 9999.9, 100000.0):
        for z in (1, 2, 3, 4, 5, 6, 10, 50):
            for model, carrier in carriers.items():
                mz = (neutral + z * carrier) / z
                if mz <= 0.0:
                    continue
                cases.append({"model": model, "neutral_mass": neutral, "charge": z, "mz": mz})
    return cases


def pyteomics_peptide_ions() -> list[dict]:
    cases = []
    for sequence in ("PEPTIDE", "SAMPLER"):
        for z in (1, 2, 3):
            cases.append(
                {
                    "sequence": sequence,
                    "charge": z,
                    "mz": mass.calculate_mass(sequence=sequence, ion_type="M", charge=z),
                    "neutral_mass": mass.calculate_mass(sequence=sequence),
                }
            )
    return cases


def fragment_cases() -> list[dict]:
    cases = []
    for sequence in ("PEPTIDE", "SAMPLER"):
        for z in (1, 2):
            b = [mass.fast_mass(sequence[:i], ion_type="b", charge=z) for i in range(1, len(sequence))]
            y = [mass.fast_mass(sequence[-i:], ion_type="y", charge=z) for i in range(1, len(sequence))]
            cases.append({"sequence": sequence, "charge": z, "b": b, "y": y})
    return cases


def binomial_cases() -> list[dict]:
    cases = []
    for k, n, p in [(1, 10, 0.1), (3, 10, 0.1), (5, 50, 0.02), (20, 200, 0.05), (10, 10, 0.5), (40, 1000, 0.001)]:
        cases.append({"k": k, "n": n, "p": p, "log10_sf": float(binom.logsf(k - 1, n, p) / math.log(10))})
    return cases


def main() -> None:
    payload = {
        "constants": {
            "codata2018_proton_mass": CODATA_PROTON_MASS,
            "codata2018_electron_mass": CODATA_ELECTRON_MASS,
            "ame2020_na23": AME_NA23,
            "ame2020_n14": AME_N14,
            "ame2020_h1": AME_H1,
            "pyteomics_proton_mass": mass.nist_mass["H+"][0][0],
        },
        "isotope_envelopes": isotope_cases(),
        "averagine": averagine_cases(),
        "real_peptides": real_peptide_cases(),
        "ion_mz": ion_mz_cases(),
        "pyteomics_peptide_ions": pyteomics_peptide_ions(),
        "fragments": fragment_cases(),
        "binomial": binomial_cases(),
    }
    out = Path(__file__).with_name("independent_references.json")
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
