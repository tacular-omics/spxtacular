"""Tests for BRAIN isotope envelopes and average-composition models."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from spxtacular import (
    DNA_ISOTOPE_MODEL,
    GLYCAN_ISOTOPE_MODEL,
    ISOTOPE_MODELS,
    LIPID_ISOTOPE_MODEL,
    PEPTIDE_ISOTOPE_MODEL,
    RNA_ISOTOPE_MODEL,
    IsotopeModel,
    IsotopeModelType,
    Spectrum,
    brain_isotopic_distribution,
    resolve_isotope_model,
)
from spxtacular.decon.greedy import NEUTRON_MASS, PROTON_MASS, _match_apex_cluster
from spxtacular.decon.scored import deconvolve_spectrum
from spxtacular.isotopes import MAX_ISOTOPE_PEAKS, NATURAL_ISOTOPE_ABUNDANCES


def _reference_convolution(composition: Mapping[str, int], max_isotopes: int) -> np.ndarray:
    distribution = np.array([1.0])
    for element, count in composition.items():
        pattern = NATURAL_ISOTOPE_ABUNDANCES[element]
        atom = np.zeros(max(offset for offset, _ in pattern) + 1)
        for offset, abundance in pattern:
            atom[offset] = abundance
        for _ in range(count):
            distribution = np.convolve(distribution, atom)[:max_isotopes]
    if len(distribution) < max_isotopes:
        distribution = np.pad(distribution, (0, max_isotopes - len(distribution)))
    return distribution / distribution.sum()


@pytest.mark.parametrize(
    "composition",
    [
        {"C": 6, "H": 12, "O": 6},
        {"C": 12, "H": 22, "N": 2, "O": 11},
        {"C": 8, "H": 18, "N": 1, "O": 6, "P": 1, "S": 1},
    ],
)
def test_brain_matches_direct_polynomial_convolution(composition: dict[str, int]) -> None:
    observed = brain_isotopic_distribution(composition, max_isotopes=16)
    expected = _reference_convolution(composition, max_isotopes=16)
    np.testing.assert_allclose(observed, expected, rtol=1e-13, atol=1e-15)


def test_custom_isotope_abundances() -> None:
    distribution = brain_isotopic_distribution(
        {"C": 2},
        max_isotopes=3,
        isotope_abundances={"C": {0: 0.5, 1: 0.5}},
    )
    np.testing.assert_allclose(distribution, [0.25, 0.5, 0.25])


def test_custom_average_composition_model() -> None:
    model = IsotopeModel(
        atoms_per_da={"C": 1 / 12},
        fixed_composition={"H": 2},
        isotope_abundances={"C": {0: 0.8, 1: 0.2}},
        name="carbon-rich",
    )
    assert model.estimate_composition(122.0) == {"C": 10, "H": 2}
    distribution = model.distribution(122.0, max_isotopes=11)
    assert distribution.argmax() == 2
    assert distribution.sum() == pytest.approx(1.0)


def test_custom_model_value_description_roundtrips() -> None:
    model = IsotopeModel(
        atoms_per_da={"C": 0.05, "O": 0.02},
        fixed_composition={"H": 2},
        isotope_abundances={"C": {0: 0.8, 1: 0.2}},
        name="carbon-rich",
    )

    assert IsotopeModel.from_dict(model.to_dict()) == model


def test_distribution_is_cached_at_one_dalton_resolution() -> None:
    low = PEPTIDE_ISOTOPE_MODEL.distribution(1000.1)
    same_bin = PEPTIDE_ISOTOPE_MODEL.distribution(1000.4)
    next_bin = PEPTIDE_ISOTOPE_MODEL.distribution(1000.6)
    np.testing.assert_array_equal(low, same_bin)
    assert not np.array_equal(low, next_bin)


def test_adaptive_distribution_exceeds_old_limit_and_honors_cap() -> None:
    adaptive = PEPTIDE_ISOTOPE_MODEL.adaptive_distribution(50_000.0)
    capped = PEPTIDE_ISOTOPE_MODEL.adaptive_distribution(50_000.0, max_isotopes=20)
    assert len(adaptive) > MAX_ISOTOPE_PEAKS
    assert int(adaptive.argmax()) == 31
    assert len(capped) == 20


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("peptide", PEPTIDE_ISOTOPE_MODEL),
        ("protein", PEPTIDE_ISOTOPE_MODEL),
        ("averagine", PEPTIDE_ISOTOPE_MODEL),
        ("glycan", GLYCAN_ISOTOPE_MODEL),
        ("averagose", GLYCAN_ISOTOPE_MODEL),
        ("lipid", LIPID_ISOTOPE_MODEL),
        ("dna", DNA_ISOTOPE_MODEL),
        ("rna", RNA_ISOTOPE_MODEL),
    ],
)
def test_model_names_and_aliases(name: str, expected: IsotopeModel) -> None:
    assert resolve_isotope_model(name) is expected


def test_every_enum_model_is_registered_and_normalized() -> None:
    assert set(ISOTOPE_MODELS) == set(IsotopeModelType)
    for model_type in IsotopeModelType:
        distribution = resolve_isotope_model(model_type).distribution(5000.0)
        assert len(distribution) == MAX_ISOTOPE_PEAKS
        assert np.all(distribution >= 0.0)
        assert distribution.sum() == pytest.approx(1.0)


def test_unknown_model_has_actionable_error() -> None:
    with pytest.raises(ValueError, match=r"unknown isotope model.*peptide.*glycan.*lipid.*dna.*rna"):
        resolve_isotope_model("small-molecule")


def test_custom_model_controls_automatic_precursor_isotopes() -> None:
    precursor_mz = 300.0
    spectrum = Spectrum(
        mz=np.array([precursor_mz, precursor_mz + NEUTRON_MASS]),
        intensity=np.array([100.0, 50.0]),
    )
    monoisotopic_model = IsotopeModel(atoms_per_da={"P": 1 / 30.97376199842})

    cleaned = spectrum.remove_precursor_peak(
        precursor_mz=precursor_mz,
        precursor_charge=1,
        tolerance=0.001,
        tolerance_type="da",
        isotopes="auto",
        isotope_model=monoisotopic_model,
    )

    np.testing.assert_allclose(cleaned.mz, [precursor_mz + NEUTRON_MASS])


def test_automatic_precursor_removal_covers_adaptive_high_mass_envelope() -> None:
    neutral_mass = 50_000.0
    charge = 10
    precursor_mz = (neutral_mass + charge * PROTON_MASS) / charge
    distribution = PEPTIDE_ISOTOPE_MODEL.adaptive_distribution(neutral_mass)
    relative = distribution / distribution.max()
    offsets = np.flatnonzero(relative >= 0.01)
    assert offsets[-1] >= MAX_ISOTOPE_PEAKS

    spectrum = Spectrum(
        mz=precursor_mz + offsets * NEUTRON_MASS / charge,
        intensity=np.ones(len(offsets)),
    )
    cleaned = spectrum.remove_precursor_peak(
        precursor_mz=precursor_mz,
        precursor_charge=charge,
        tolerance=0.001,
        tolerance_type="da",
        isotopes="auto",
        remove_charge_states=False,
    )

    assert len(cleaned.mz) == 0


def test_high_mass_envelope_recovers_monoisotopic_anchor() -> None:
    neutral_mass = 20_000.0
    charge = 10
    distribution = PEPTIDE_ISOTOPE_MODEL.adaptive_distribution(neutral_mass)
    apex = int(distribution.argmax())
    assert apex > 10  # Regression: the old backward search stopped at A+4.

    mono_mz = (neutral_mass + charge * PROTON_MASS) / charge
    relative = distribution / distribution.max()
    offsets = np.flatnonzero(relative >= 0.01).astype(np.float64)
    assert 0 not in offsets  # A+0 is too weak to observe at this mass.
    mz = mono_mz + offsets * 1.00335483507 / charge
    intensity = distribution[offsets.astype(int)] * 1e8

    out_mz, out_charge, out_intensity, out_score = deconvolve_spectrum(
        mz,
        intensity,
        charge_range=(charge, charge),
        tolerance=5.0,
        is_ppm=True,
        isotope_model="peptide",
    )

    assert len(out_mz) == 1
    assert out_mz[0] == pytest.approx(mono_mz)
    assert out_charge[0] == charge
    assert out_intensity[0] == pytest.approx(float(intensity.sum()))
    assert out_score[0] == pytest.approx(1.0)

    _, _, base_intensity, _ = deconvolve_spectrum(
        mz,
        intensity,
        charge_range=(charge, charge),
        tolerance=5.0,
        is_ppm=True,
        intensity_mode="base",
    )
    assert base_intensity[0] == 0.0


def test_near_apex_alignment_handles_observed_maximum_shift() -> None:
    """A noisy near-apex isotope must not introduce a one-isotope mass error."""
    neutral_mass = 50_000.0
    charge = 5
    distribution = PEPTIDE_ISOTOPE_MODEL.adaptive_distribution(neutral_mass)
    relative = distribution / distribution.max()
    offsets = np.flatnonzero(relative >= 0.01)
    predicted_apex = int(distribution.argmax())
    observed_apex = predicted_apex - 1
    mono_mz = (neutral_mass + charge * PROTON_MASS) / charge
    mz = mono_mz + offsets * NEUTRON_MASS / charge
    intensity = distribution[offsets] * 1e8

    observed_position = int(np.flatnonzero(offsets == observed_apex)[0])
    predicted_position = int(np.flatnonzero(offsets == predicted_apex)[0])
    intensity[observed_position] = intensity[predicted_position] * 1.02
    assert int(offsets[intensity.argmax()]) == observed_apex

    out_mz, out_charge, _, out_score = deconvolve_spectrum(
        mz,
        intensity,
        charge_range=(1, 10),
        tolerance=10.0,
        is_ppm=True,
        isotope_model="peptide",
    )

    assert len(out_mz) == 1
    assert out_mz[0] == pytest.approx(mono_mz)
    assert out_charge[0] == charge
    assert out_score[0] > 0.99


def test_inferred_monoisotopic_peak_keeps_apex_ion_mobility() -> None:
    neutral_mass = 20_000.0
    charge = 10
    distribution = PEPTIDE_ISOTOPE_MODEL.adaptive_distribution(neutral_mass)
    relative = distribution / distribution.max()
    offsets = np.flatnonzero(relative >= 0.01)
    mono_mz = (neutral_mass + charge * PROTON_MASS) / charge
    mz = mono_mz + offsets * NEUTRON_MASS / charge
    intensity = distribution[offsets] * 1e8
    im = 1.5 + np.linspace(-0.01, 0.01, len(mz))
    apex_position = int(np.argmax(intensity))

    deconvoluted = Spectrum(mz=mz, intensity=intensity, im=im).deconvolute(
        charge_range=(charge, charge),
        tolerance=5.0,
        tolerance_type="ppm",
    )

    assert deconvoluted.mz[0] == pytest.approx(mono_mz)
    assert deconvoluted.im is not None
    assert deconvoluted.im[0] == pytest.approx(im[apex_position])


def test_candidate_score_can_prefer_abundance_over_closest_mz() -> None:
    neutral_mass = 1000.0
    charge = 2
    distribution = PEPTIDE_ISOTOPE_MODEL.adaptive_distribution(neutral_mass)
    relative = distribution / distribution.max()
    assert int(distribution.argmax()) == 0

    mono_mz = (neutral_mass + charge * PROTON_MASS) / charge
    target_mz = mono_mz + NEUTRON_MASS / charge
    seed_intensity = 1e6
    expected_intensity = seed_intensity * relative[1]
    mz = np.array(
        [
            mono_mz,
            target_mz * (1.0 + 1e-6),
            target_mz * (1.0 + 4e-6),
        ]
    )
    intensity = np.array([seed_intensity, expected_intensity * 0.55, expected_intensity])

    out_mz, out_charge, out_intensity, _ = deconvolve_spectrum(
        mz,
        intensity,
        charge_range=(charge, charge),
        tolerance=10.0,
        is_ppm=True,
    )

    cluster = int(np.flatnonzero(out_charge == charge)[0])
    assert out_mz[cluster] == pytest.approx(mono_mz)
    assert out_intensity[cluster] == pytest.approx(seed_intensity + expected_intensity)
    assert out_intensity.sum() == pytest.approx(intensity.sum())


def test_candidate_score_uses_ion_mobility_to_reject_closer_interference() -> None:
    neutral_mass = 1000.0
    charge = 2
    distribution = PEPTIDE_ISOTOPE_MODEL.adaptive_distribution(neutral_mass)
    relative = distribution / distribution.max()
    mono_mz = (neutral_mass + charge * PROTON_MASS) / charge
    target_mz = mono_mz + NEUTRON_MASS / charge
    seed_intensity = 1e6
    expected_intensity = seed_intensity * relative[1]
    mz = np.array(
        [
            mono_mz,
            target_mz * (1.0 + 1e-6),
            target_mz * (1.0 + 4e-6),
        ]
    )
    intensity = np.array([seed_intensity, expected_intensity, expected_intensity])
    ion_mobility = np.array([1.0, 1.2, 1.01])

    deconvoluted = Spectrum(mz=mz, intensity=intensity, im=ion_mobility).deconvolute(
        charge_range=(charge, charge),
        tolerance=10.0,
        tolerance_type="ppm",
        im_tolerance=0.05,
        im_tolerance_type="absolute",
    )

    assert deconvoluted.charge is not None
    cluster = int(np.flatnonzero(deconvoluted.charge == charge)[0])
    assert deconvoluted.mz[cluster] == pytest.approx(mono_mz)
    assert deconvoluted.intensity[cluster] == pytest.approx(seed_intensity + expected_intensity)
    assert deconvoluted.intensity.sum() == pytest.approx(intensity.sum())


def test_zero_gap_default_stops_before_later_isotope() -> None:
    neutral_mass = 1000.0
    charge = 2
    distribution = PEPTIDE_ISOTOPE_MODEL.adaptive_distribution(neutral_mass)
    relative = distribution / distribution.max()
    assert int(relative.argmax()) == 0
    mono_mz = (neutral_mass + charge * PROTON_MASS) / charge
    mz = np.array([mono_mz, mono_mz + 2 * NEUTRON_MASS / charge])
    intensity = np.array([1e6, relative[2] * 1e6])
    ion_mobility = np.full(2, np.nan)
    used = np.zeros(2, dtype=np.bool_)

    no_gaps, *_ = _match_apex_cluster(
        mz,
        intensity,
        ion_mobility,
        used,
        0,
        charge,
        5.0,
        True,
        relative,
        0,
        0.01,
        2.0,
        0,
        False,
        0.05,
        True,
    )
    one_gap, *_ = _match_apex_cluster(
        mz,
        intensity,
        ion_mobility,
        used,
        0,
        charge,
        5.0,
        True,
        relative,
        0,
        0.01,
        2.0,
        1,
        False,
        0.05,
        True,
    )

    assert no_gaps == 1
    assert one_gap == 2


def test_fold_disagreement_stops_and_leaves_blocking_peaks_for_later_passes() -> None:
    neutral_mass = 1000.0
    charge = 2
    distribution = PEPTIDE_ISOTOPE_MODEL.adaptive_distribution(neutral_mass)
    relative = distribution / distribution.max()
    offsets = np.flatnonzero(relative >= 0.01)
    mono_mz = (neutral_mass + charge * PROTON_MASS) / charge
    mz = mono_mz + offsets * NEUTRON_MASS / charge
    intensity = distribution[offsets] * 1e6
    intensity[2] *= 3.0  # Outside the default twofold gate, but still below the apex.

    out_mz, out_charge, out_intensity, _ = deconvolve_spectrum(
        mz,
        intensity,
        charge_range=(charge, charge),
        tolerance=5.0,
        is_ppm=True,
    )

    first_cluster = int(np.argmin(np.abs(out_mz - mono_mz)))
    assert out_charge[first_cluster] == charge
    assert out_intensity[first_cluster] == pytest.approx(float(intensity[:2].sum()))
    assert len(out_mz) > 1
    assert float(out_intensity.sum()) == pytest.approx(float(intensity.sum()))
