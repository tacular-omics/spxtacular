"""Check spxtacular against values computed without spxtacular or peptacular.

The fixture ``reference/independent_references.json`` is written by
``reference/generate_independent_references.py`` (pyteomics, scipy and hand-written
CODATA/AME constants). See that script's header for the sources and versions.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from peptacular import IonType

from spxtacular import Spectrum, score
from spxtacular.core import _centroid_peaks
from spxtacular.decon.greedy import NEUTRON_MASS
from spxtacular.ionization import IONIZATION_MODELS
from spxtacular.isotopes import PEPTIDE_ISOTOPE_MODEL, brain_isotopic_distribution
from spxtacular.matching import match_fragments
from spxtacular.scoring import _binom_log10_survival

REF = json.loads((Path(__file__).parent / "reference" / "independent_references.json").read_text())

# AME2020: m(13C) - m(12C), with m(12C) = 12 exactly by definition.
C13_SPACING = 13.00335483507 - 12.0
MASS_ATOL = 1e-6


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# Isotope envelopes and averagine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", REF["isotope_envelopes"], ids=[c["name"] for c in REF["isotope_envelopes"]])
def test_brain_matches_independent_isotope_envelope(case: dict) -> None:
    expected = np.asarray(case["convolution"])
    actual = brain_isotopic_distribution(case["composition"], max_isotopes=len(expected))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    if case["pyteomics"] is not None:
        # pyteomics enumerates isotopologues explicitly; its thresholds drop terms
        # below 1e-14, so the far tail is compared at that level.
        np.testing.assert_allclose(actual, case["pyteomics"], rtol=0, atol=1e-12)


@pytest.mark.parametrize("case", REF["averagine"], ids=[str(int(c["mass"])) for c in REF["averagine"]])
def test_peptide_model_is_close_to_senko_averagine(case: dict) -> None:
    # The peptide model uses its own atom rates, not Senko's, so the envelopes are
    # close rather than equal. At 5000 Da peaks 2 and 3 are a near tie and the two
    # models put the apex on different sides of it.
    senko = np.asarray(case["senko_envelope"])
    model = PEPTIDE_ISOTOPE_MODEL.distribution(case["mass"], len(senko))
    assert _cosine(model, senko) > 0.9999
    assert np.max(np.abs(model - senko)) < 0.01
    if case["mass"] != 5000.0:
        assert int(np.argmax(model)) == int(np.argmax(senko))
    else:
        assert abs(senko[2] - senko[3]) < 0.01


@pytest.mark.parametrize("case", REF["real_peptides"], ids=[c["sequence"] for c in REF["real_peptides"]])
def test_peptide_model_predicts_real_peptide_envelopes(case: dict) -> None:
    exact = np.asarray(case["envelope"])
    model = PEPTIDE_ISOTOPE_MODEL.distribution(case["monoisotopic_mass"], len(exact))
    assert _cosine(model, exact) > 0.998
    assert np.max(np.abs(model - exact)) < 0.03


# ---------------------------------------------------------------------------
# m/z <-> neutral mass
# ---------------------------------------------------------------------------


def test_proton_mass_matches_codata() -> None:
    carrier = IONIZATION_MODELS["protonated"].carrier_mass
    assert carrier == pytest.approx(REF["constants"]["codata2018_proton_mass"], abs=1e-9)


@pytest.mark.parametrize(
    "case",
    REF["ion_mz"],
    ids=[f"{c['model']}-{c['neutral_mass']}-z{c['charge']}" for c in REF["ion_mz"]],
)
def test_ion_mz_against_codata_and_ame(case: dict) -> None:
    model = IONIZATION_MODELS[case["model"]]
    z = case["charge"]
    assert model.ion_mz(case["neutral_mass"], z) == pytest.approx(case["mz"], abs=1e-7 / z + 1e-9)
    if case["neutral_mass"] > 0.0:
        # At neutral mass 0 the carrier constants differ by ~1e-10, so the inverse can
        # land a hair below zero and is rejected; that is not a physical input.
        assert model.neutral_mass(case["mz"], z) == pytest.approx(case["neutral_mass"], abs=1e-7)


@pytest.mark.parametrize("case", REF["pyteomics_peptide_ions"], ids=lambda c: f"{c['sequence']}-z{c['charge']}")
def test_ion_mz_against_pyteomics(case: dict) -> None:
    model = IONIZATION_MODELS["protonated"]
    assert model.ion_mz(case["neutral_mass"], case["charge"]) == pytest.approx(case["mz"], abs=1e-8)


# ---------------------------------------------------------------------------
# Deconvolution and decharging
# ---------------------------------------------------------------------------


def test_neutron_spacing_is_the_c13_mass_difference() -> None:
    assert pytest.approx(C13_SPACING, abs=1e-10) == NEUTRON_MASS


@pytest.mark.parametrize("charge", [1, 2, 3, 4])
@pytest.mark.parametrize("case", REF["real_peptides"], ids=[c["sequence"] for c in REF["real_peptides"]])
def test_deconvolution_recovers_exact_peptide_mass(case: dict, charge: int) -> None:
    mono = case["monoisotopic_mass"]
    envelope = np.asarray(case["envelope"])
    keep = np.flatnonzero(envelope >= 0.01 * envelope.max())
    proton = REF["constants"]["codata2018_proton_mass"]
    spectrum = Spectrum(
        mz=(mono + keep * C13_SPACING) / charge + proton,
        intensity=1e6 * envelope[keep],
        spectrum_type="centroid",
    )
    decon = spectrum.deconvolute(tolerance=5, charge_range=(1, 6))
    assert decon.charge is not None
    np.testing.assert_array_equal(decon.charge, [charge])
    np.testing.assert_allclose(decon.decharge().mz, [mono], rtol=0, atol=MASS_ATOL)
    assert decon.intensity.sum() == pytest.approx(spectrum.intensity.sum(), rel=1e-12)


# ---------------------------------------------------------------------------
# Fragment matching and scoring
# ---------------------------------------------------------------------------


def _pyteomics_fragment_dict(case: dict) -> dict:
    z = case["charge"]
    return {(IonType("b"), z): case["b"], (IonType("y"), z): case["y"]}


@pytest.mark.parametrize("case", REF["fragments"], ids=lambda c: f"{c['sequence']}-z{c['charge']}")
def test_exact_fragments_all_match(case: dict) -> None:
    mz = np.asarray(case["b"] + case["y"])
    spectrum = Spectrum(mz=mz, intensity=np.arange(1.0, mz.size + 1.0))
    matches = match_fragments(spectrum, _pyteomics_fragment_dict(case), tolerance=1e-9, tolerance_type="da")
    assert len(matches) == mz.size
    for match in matches:
        assert match.da_error == pytest.approx(0.0, abs=1e-9)
        assert match.peak_mz == pytest.approx(match.fragment.mz, abs=1e-9)


@pytest.mark.parametrize("unit", ["ppm", "da"])
@pytest.mark.parametrize("case", REF["fragments"], ids=lambda c: f"{c['sequence']}-z{c['charge']}")
def test_fragment_tolerance_edges(case: dict, unit: str) -> None:
    # A peak just inside the window matches and one just outside does not. ppm is
    # relative to the theoretical m/z.
    theoretical = np.asarray(case["b"] + case["y"])
    tolerance = 10.0 if unit == "ppm" else 0.02
    window = theoretical * tolerance * 1e-6 if unit == "ppm" else np.full_like(theoretical, tolerance)
    fragments = _pyteomics_fragment_dict(case)
    for sign in (1.0, -1.0):
        inside = Spectrum(mz=theoretical + sign * window * (1 - 1e-6), intensity=np.ones_like(theoretical))
        outside = Spectrum(mz=theoretical + sign * window * (1 + 1e-6), intensity=np.ones_like(theoretical))
        inside_matches = match_fragments(inside, fragments, tolerance=tolerance, tolerance_type=unit)
        assert sorted(m.peak_index for m in inside_matches) == list(range(theoretical.size))
        assert match_fragments(outside, fragments, tolerance=tolerance, tolerance_type=unit) == []
        for match in inside_matches:
            error = match.ppm_error if unit == "ppm" else match.da_error
            assert abs(error) <= tolerance
            assert math.copysign(1.0, error) == sign


@pytest.mark.parametrize("case", REF["binomial"], ids=lambda c: f"k{c['k']}-n{c['n']}-p{c['p']}")
def test_binomial_survival_against_scipy(case: dict) -> None:
    actual = _binom_log10_survival(case["k"], case["n"], case["p"])
    assert actual == pytest.approx(case["log10_sf"], rel=1e-10, abs=1e-12)


def test_hyperscore_on_pyteomics_fragments() -> None:
    case = next(c for c in REF["fragments"] if c["sequence"] == "PEPTIDE" and c["charge"] == 1)
    b, y = case["b"], case["y"]
    # Three b ions and two y ions present, plus two unmatched peaks.
    mz = np.asarray([b[0], b[2], b[4], y[1], y[3], 300.0, 400.0])
    intensity = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
    order = np.argsort(mz)
    spectrum = Spectrum(mz=mz[order], intensity=intensity[order])
    result = score(spectrum, _pyteomics_fragment_dict(case), tolerance=0.001, tolerance_type="da")
    expected = math.log10(150.0) + math.log10(math.factorial(3)) + math.log10(math.factorial(2))
    assert result["hyperscore"] == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# Centroiding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("center", "height", "sigma", "step", "phase"),
    [
        (500.0, 1e6, 0.01, 0.002, 0.0),
        (500.0, 1e6, 0.01, 0.002, 0.37),
        (1234.5678, 3.5e4, 0.004, 0.001, -0.42),
        (150.1234, 10.0, 0.02, 0.005, 0.25),
    ],
)
def test_centroid_of_sampled_gaussian_is_exact(
    center: float, height: float, sigma: float, step: float, phase: float
) -> None:
    # A three-point parabola in log space is exact for a Gaussian, so the fitted
    # centre and height should be recovered to round-off.
    mz = center + (np.arange(-40, 41) + phase) * step
    intensity = height * np.exp(-0.5 * ((mz - center) / sigma) ** 2)
    centers, heights, _ = _centroid_peaks(mz, intensity)
    assert centers.size == 1
    assert centers[0] == pytest.approx(center, abs=1e-9)
    assert heights[0] == pytest.approx(height, rel=1e-9)


def test_centroid_separates_two_gaussians() -> None:
    mz = np.arange(499.0, 502.0, 0.001)
    intensity = 1e5 * np.exp(-0.5 * ((mz - 500.0) / 0.005) ** 2) + 5e4 * np.exp(-0.5 * ((mz - 501.0) / 0.005) ** 2)
    profile = Spectrum(mz=mz, intensity=intensity, spectrum_type="profile")
    centroided = profile.centroid()
    np.testing.assert_allclose(centroided.mz, [500.0, 501.0], rtol=0, atol=1e-8)
    np.testing.assert_allclose(centroided.intensity, [1e5, 5e4], rtol=1e-8)
