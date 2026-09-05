"""Analytic references that do not use the production envelope generator."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from peptacular import IonType

from spxtacular import IsotopeModel, Spectrum, entropy_similarity, score

CASES = json.loads((Path(__file__).parent / "reference" / "carbon_envelopes.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_deconvolution_against_frozen_binomial_reference(case: dict) -> None:
    spectrum = Spectrum(
        mz=np.asarray(case["mz"]),
        intensity=np.asarray(case["intensity"]),
        im=np.asarray(case["im"]) if "im" in case else None,
    )
    decon = spectrum.deconvolute(
        isotope_model=IsotopeModel(atoms_per_da={}, fixed_composition={"C": case["carbon_count"]}),
        ionization_model=case["ionization_model"],
        charge_range=(1, 6),
        tolerance=5.0,
        min_intensity=case["min_intensity"],
        min_score=0.4,
    )
    assert decon.charge is not None
    assigned = decon.charge > 0
    np.testing.assert_array_equal(decon.charge[assigned], case["expected_charges"])
    assert np.count_nonzero(decon.charge == -1) == case["expected_singletons"]
    np.testing.assert_allclose(decon.decharge().mz, case["expected_masses"], rtol=0, atol=1e-4)
    assert decon.intensity.sum() == pytest.approx(spectrum.intensity.sum(), rel=1e-12)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([1.0, 1.0], [1.0, 1.0], 1.0),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([1.0, 0.0], [0.5, 0.5], 0.6887218755408672),
        ([0.9, 0.1], [0.1, 0.9], 0.4689955935892812),
    ],
)
def test_unweighted_entropy_reference_values(a: list[float], b: list[float], expected: float) -> None:
    # Values follow 1 - JSD(a,b)/ln(2) for these already aligned distributions.
    mz = np.array([100.0, 200.0])
    actual = entropy_similarity(Spectrum(mz=mz, intensity=np.array(a)), Spectrum(mz=mz, intensity=np.array(b)))
    assert actual == pytest.approx(expected, abs=1e-12)


def test_hyperscore_does_not_count_duplicate_peak_intensity_twice() -> None:
    spectrum = Spectrum(mz=np.array([100.0, 200.0]), intensity=np.array([10.0, 20.0]))
    fragments = {(IonType("b"), 1): [100.0, 200.0], (IonType("y"), 1): [100.0]}
    result = score(spectrum, fragments, tolerance=0.01)
    assert result["hyperscore"] == pytest.approx(math.log10(30.0 * math.factorial(2)))
