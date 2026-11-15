"""Ascore site localization: exact synthetic cases, agreement with pyAscore, real data.

These are validation items 1, 2 and 4 of the localization design. Item 3, the Marx
synthetic phosphopeptide library, is a measurement run outside the suite.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from spxtacular import (
    ASCORE_CONFIDENT_CUT,
    ASCORE_PEAK_DEPTHS,
    LocalizationProvenance,
    LocalizationResult,
    MsnSpectrum,
    Spectrum,
    SpxtacularError,
    localize,
)

pt = pytest.importorskip("peptacular")

# PEPTIDESK carries one phosphorylation on either T4 or S8. The ProForma group names both
# candidates without committing to one.
AMBIGUOUS = "PEPT[Phospho#g1]IDES[#g1]K"
ON_T4 = "PEPT[Phospho]IDESK"
ON_S8 = "PEPTIDES[Phospho]K"
TOL = 0.05


def fragment_mz(peptide, ion_types=("b", "y"), charges=(1,)):
    """The 1+ b/y m/z of one peptide, full-length ions dropped as Ascore drops them."""
    out = []
    for fragment in pt.fragment(peptide, ion_types=ion_types, charges=charges):
        if fragment.position is not None and int(fragment.position) >= int(fragment.parent_sequence_length):
            continue
        out.append(fragment.mz)
    return sorted(out)


def spectrum_of(mz_values, intensity=1000.0, noise=0):
    """A centroided spectrum of these m/z, optionally with uniform noise peaks added."""
    mz = list(mz_values)
    intensities = [intensity] * len(mz)
    if noise:
        rng = np.random.default_rng(0)
        extra = sorted(rng.uniform(100.0, 1100.0, noise))
        mz += extra
        intensities += [intensity / 20.0] * len(extra)
    order = np.argsort(np.asarray(mz))
    return Spectrum(mz=np.asarray(mz)[order], intensity=np.asarray(intensities)[order])


# ---------------------------------------------------------------------------
# Validation item 1: synthetic and exact
# ---------------------------------------------------------------------------


def test_the_synthesised_site_wins():
    result = localize(spectrum_of(fragment_mz(ON_T4), noise=40), AMBIGUOUS, tolerance=TOL)
    assert result.best == "PEPT[Phospho#g1]IDESK"
    assert result.runner_up == "PEPTIDES[Phospho#g1]K"
    assert result.ascore is not None and result.ascore > ASCORE_CONFIDENT_CUT
    assert not result.is_ambiguous()


def test_the_other_site_wins_when_the_spectrum_says_so():
    result = localize(spectrum_of(fragment_mz(ON_S8), noise=40), AMBIGUOUS, tolerance=TOL)
    assert result.best == "PEPTIDES[Phospho#g1]K"
    assert result.ascore is not None and result.ascore > ASCORE_CONFIDENT_CUT


def test_ascore_is_zero_without_the_site_determining_ions():
    """Only the ions both candidates share: nothing distinguishes them."""
    shared = sorted(set(fragment_mz(ON_T4)) & set(fragment_mz(ON_S8)))
    assert shared, "the two candidates must share some ions for this to be a real test"
    result = localize(spectrum_of(shared), AMBIGUOUS, tolerance=TOL)
    assert result.ascore == 0.0
    assert result.is_ambiguous()
    assert [matched for matched, _ in result.site_determining] == [0, 0]


def test_ascore_does_not_fall_as_site_determining_ions_are_added():
    """Monotonicity: more evidence for the true site never lowers the score."""
    shared = sorted(set(fragment_mz(ON_T4)) & set(fragment_mz(ON_S8)))
    only_t4 = sorted(set(fragment_mz(ON_T4)) - set(fragment_mz(ON_S8)))
    scores: list[float] = []
    for take in range(len(only_t4) + 1):
        result = localize(spectrum_of(shared + only_t4[:take]), AMBIGUOUS, tolerance=TOL)
        assert result.ascore is not None
        scores.append(result.ascore)
    assert scores[0] == 0.0
    assert scores == sorted(scores), f"not monotone: {scores}"
    assert scores[-1] > scores[0]


def test_one_candidate_has_no_ascore():
    result = localize(spectrum_of(fragment_mz(ON_T4)), ON_T4, tolerance=TOL)
    assert result.isomers == (ON_T4,)
    assert result.best == ON_T4
    assert result.runner_up is None
    assert result.ascore is None
    assert not result.is_ambiguous()
    assert result.peak_depth in ASCORE_PEAK_DEPTHS


def test_empty_spectrum_scores_zero():
    result = localize(Spectrum(mz=np.array([]), intensity=np.array([])), AMBIGUOUS, tolerance=TOL)
    assert result.ascore == 0.0
    assert result.peptide_scores == (0.0, 0.0)


def test_precursor_only_ions_cannot_localize():
    """A precursor has the same mass whichever site carries the modification."""
    result = localize(spectrum_of(fragment_mz(ON_T4)), AMBIGUOUS, ion_types=("p",), tolerance=TOL)
    assert result.ascore == 0.0
    assert result.site_determining == ((0, 0), (0, 0))


def test_existing_localization_scores_are_not_read_as_evidence():
    """The incoming ``#g1(0.8)`` numbers are another tool's; only ours come back."""
    with_scores = "PEPT[Phospho#g1(0.99)]IDES[#g1(0.01)]K"
    spectrum = spectrum_of(fragment_mz(ON_S8), noise=40)
    result = localize(spectrum, with_scores, tolerance=TOL)
    assert result.best.startswith("PEPTIDES[Phospho")
    assert result.ascore is not None and result.ascore > ASCORE_CONFIDENT_CUT


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"peak_depths": ()}, "peak_depths is empty"),
        ({"peak_depths": (0, 1)}, "must all be >= 1"),
        ({"window": 0.0}, "window must be"),
        ({"window": float("inf")}, "window must be"),
        ({"max_isomers": 1}, "Cannot expand the candidate sites"),
    ],
)
def test_bad_arguments_raise(kwargs, match):
    with pytest.raises(SpxtacularError, match=match):
        localize(spectrum_of(fragment_mz(ON_T4)), AMBIGUOUS, tolerance=TOL, **kwargs)


def test_decharged_spectrum_raises():
    neutral = MsnSpectrum(
        mz=np.array([100.0, 200.0]),
        intensity=np.array([1.0, 2.0]),
        charge=np.array([0, 0], dtype=np.int32),
        ms_level=2,
    )
    with pytest.raises(SpxtacularError, match="decharged"):
        localize(neutral, AMBIGUOUS, tolerance=TOL)


def test_profile_spectrum_raises():
    profile = Spectrum(mz=np.array([100.0, 200.0]), intensity=np.array([1.0, 2.0]), spectrum_type="profile")
    with pytest.raises(SpxtacularError, match="centroid"):
        localize(profile, AMBIGUOUS, tolerance=TOL)


def test_ppm_tolerance_works_and_is_recorded():
    result = localize(spectrum_of(fragment_mz(ON_T4), noise=40), AMBIGUOUS, tolerance=20.0, tolerance_unit="ppm")
    assert result.best == "PEPT[Phospho#g1]IDESK"
    assert result.provenance.tolerance_unit == "ppm"
    assert result.provenance.tolerance == 20.0


def test_result_round_trips_through_dict():
    result = localize(spectrum_of(fragment_mz(ON_T4), noise=40), AMBIGUOUS, tolerance=TOL)
    payload = result.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    back = LocalizationResult.from_dict(payload)
    assert back == result
    assert LocalizationProvenance.from_dict(payload["provenance"]) == result.provenance


def test_custom_peak_depths_are_weighted_uniformly():
    result = localize(spectrum_of(fragment_mz(ON_T4), noise=40), AMBIGUOUS, tolerance=TOL, peak_depths=(3, 5))
    assert result.provenance.peak_depths == (3, 5)
    assert result.peak_depth in (3, 5)


# ---------------------------------------------------------------------------
# Validation item 2: agreement with pyAscore, the published implementation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("true_isomer", [ON_T4, ON_S8])
def test_agrees_with_pyascore(true_isomer):
    """Same spectrum, same tolerance, same depths, against pyAscore 1.0.1.

    pyAscore calls itself "a modified version of Beausoleil et al.", so this pins
    agreement rather than assuming it. It skips cleanly when pyascore is not installed.
    """
    pyascore = pytest.importorskip("pyascore")
    mz_values = fragment_mz(true_isomer)
    spectrum = spectrum_of(mz_values, noise=40)

    scorer = pyascore.PyAscore(100.0, max(ASCORE_PEAK_DEPTHS), "STY", 79.966331, TOL, "by")
    scorer.score(
        mz_arr=np.asarray(spectrum.mz, dtype=np.float64),
        int_arr=np.asarray(spectrum.intensity, dtype=np.float64),
        peptide="PEPTIDESK",
        n_of_mod=1,
        max_fragment_charge=1,
    )
    ours = localize(spectrum, AMBIGUOUS, tolerance=TOL)

    # Same winning residue: pyAscore writes the mass in brackets, we keep the ProForma.
    theirs_site = scorer.best_sequence.index("[")
    ours_site = ours.best.index("[")
    assert theirs_site == ours_site, f"{scorer.best_sequence} against {ours.best}"

    # Same peptide score for the winner, and the same AScore.
    assert ours.peptide_scores[ours.isomers.index(ours.best)] == pytest.approx(scorer.best_score, rel=1e-6)
    assert ours.ascore == pytest.approx(float(scorer.ascores[0]), rel=1e-5)


# ---------------------------------------------------------------------------
# Validation item 4: a real PSM, end to end
# ---------------------------------------------------------------------------

_FIXTURE = Path(__file__).resolve().parents[1] / "paper" / "abstract" / "data" / "stages.json"


@pytest.mark.skipif(not _FIXTURE.exists(), reason="the paper's PSM fixture is not in this tree")
def test_runs_on_a_real_psm():
    """mzspec:PXD000561:...:scan:17555, VLHPLEGAVVIIFK, the spectrum the paper figure uses.

    The peptide carries no modification, so an unknown-position phospho has one candidate
    per residue and nothing in the spectrum supports any of them. The call must still
    return, and it must not claim a confident site.
    """
    data = json.loads(_FIXTURE.read_text())
    spectrum = Spectrum(
        mz=np.asarray(data["raw"]["mz"], dtype=np.float64),
        intensity=np.asarray(data["raw"]["intensity"], dtype=np.float64),
    )
    peptide = data["peptide"]
    result = localize(spectrum, f"[Phospho]?{peptide}", tolerance=data["tolerance_ppm"], tolerance_unit="ppm")
    assert len(result.isomers) == len(peptide)
    assert result.best in result.isomers
    assert result.ascore is not None
    assert result.is_ambiguous(), f"a real unmodified PSM should not localize a phospho: {result.ascore}"
    assert len(result.peptide_scores) == len(result.isomers)
    assert len(result.site_determining) == len(result.isomers)
