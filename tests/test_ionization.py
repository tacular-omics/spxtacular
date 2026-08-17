"""Ionization-model conversion, selection, and provenance tests."""

from __future__ import annotations

import numpy as np
import peptacular as pt
import pytest

from spxtacular import (
    AMMONIATED,
    DEPROTONATED,
    PROTONATED,
    SODIATED,
    DeconvolutionProvenance,
    IonizationModel,
    IsotopeModel,
    MsnSpectrum,
    Spectrum,
    from_matchms,
    from_spectrl_token,
    resolve_ionization_model,
    to_matchms,
    to_spectrl_token,
)


@pytest.mark.parametrize("model", [PROTONATED, DEPROTONATED, SODIATED, AMMONIATED])
@pytest.mark.parametrize("charge", [1, 2, 5])
def test_builtin_models_roundtrip_neutral_mass(model: IonizationModel, charge: int) -> None:
    mz = model.ion_mz(1234.567, charge)
    assert model.neutral_mass(mz, charge) == pytest.approx(1234.567)


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("[M+H]+", PROTONATED),
        ("[M-H]-", DEPROTONATED),
        ("[M+Na]+", SODIATED),
        ("[M+NH4]+", AMMONIATED),
    ],
)
def test_adduct_aliases(alias: str, expected: IonizationModel) -> None:
    assert resolve_ionization_model(alias) is expected


def test_custom_carrier_mass() -> None:
    model = IonizationModel("potassiated", "positive", 38.963158, "K")
    assert model.neutral_mass(model.ion_mz(800.0, 3), 3) == pytest.approx(800.0)
    assert model.notation(3) == "[M+3K]3+"


def test_numeric_custom_negative_carrier() -> None:
    model = resolve_ionization_model(-2.5)
    assert str(model.polarity) == "negative"
    assert model.neutral_mass(model.ion_mz(500.0, 2), 2) == pytest.approx(500.0)


def _provenance(model: IonizationModel) -> DeconvolutionProvenance:
    return DeconvolutionProvenance(
        isotope_model="peptide",
        ionization_model=model,
        charge_range=(1, 5),
        tolerance=10.0,
        tolerance_type="ppm",
        intensity_mode="total",
        min_intensity=0.0,
        min_score=0.0,
    )


@pytest.mark.parametrize("model", [PROTONATED, DEPROTONATED, SODIATED, AMMONIATED])
def test_decharge_uses_recorded_ionization_model(model: IonizationModel) -> None:
    neutral = 1000.0
    charge = 2
    spec = Spectrum(
        mz=np.array([model.ion_mz(neutral, charge)]),
        intensity=np.array([100.0]),
        charge=np.array([charge]),
        spectrum_type="deconvoluted",
        deconvolution=_provenance(model),
    )
    decharged = spec.decharge()
    np.testing.assert_allclose(decharged.mz, [neutral])
    np.testing.assert_array_equal(decharged.charge, [0])


def test_negative_polarity_defaults_to_deprotonation_without_provenance() -> None:
    neutral = 750.0
    mz = DEPROTONATED.ion_mz(neutral, 2)
    spec = MsnSpectrum(
        mz=np.array([mz]),
        intensity=np.array([100.0]),
        charge=np.array([2]),
        spectrum_type="deconvoluted",
        polarity="negative",
    )
    assert spec.decharge().mz[0] == pytest.approx(neutral)


def test_explicit_model_must_match_spectrum_polarity() -> None:
    spec = MsnSpectrum(
        mz=np.array([500.0]),
        intensity=np.array([100.0]),
        charge=np.array([1]),
        spectrum_type="deconvoluted",
        polarity="negative",
    )
    with pytest.raises(ValueError, match="spectrum polarity"):
        spec.decharge(ionization_model="[M+H]+")


def test_negative_precursor_removal_uses_deprotonated_charge_targets() -> None:
    neutral = 900.0
    charge_1 = DEPROTONATED.ion_mz(neutral, 1)
    charge_2 = DEPROTONATED.ion_mz(neutral, 2)
    spec = MsnSpectrum(
        mz=np.array([charge_2, charge_1, 700.0]),
        intensity=np.array([100.0, 80.0, 10.0]),
        spectrum_type="centroid",
        polarity="negative",
    )
    result = spec.remove_precursor_peak(
        precursor_mz=float(charge_2),
        precursor_charge=2,
        isotopes=0,
    )
    np.testing.assert_allclose(result.mz, [700.0])


def test_deconvolution_records_models_and_parameters() -> None:
    step = pt.C13_NEUTRON_MASS / 2
    mz = np.array([500.0, 500.0 + step, 500.0 + 2 * step])
    intensity = np.array([1000.0, 700.0, 300.0])
    result = Spectrum(mz=mz, intensity=intensity).deconvolute(
        charge_range=(2, 2),
        tolerance=10.0,
        min_isotope_abundance=0.02,
        max_isotope_fold_error=3.0,
        max_isotope_gaps=1,
        max_isotopes=12,
        im_tolerance=0.03,
        im_tolerance_type="absolute",
        ionization_model="[M+Na]+",
    )
    assert result.deconvolution is not None
    assert result.deconvolution.isotope_model == "peptide"
    assert result.deconvolution.isotope_model_definition is not None
    assert result.deconvolution.isotope_model_definition.name == "peptide"
    assert result.deconvolution.ionization_model is SODIATED
    assert result.deconvolution.charge_range == (2, 2)
    assert result.deconvolution.min_isotope_abundance == 0.02
    assert result.deconvolution.max_isotope_fold_error == 3.0
    assert result.deconvolution.max_isotope_gaps == 1
    assert result.deconvolution.max_isotopes == 12
    assert result.deconvolution.im_tolerance == 0.03
    assert result.deconvolution.im_tolerance_type == "absolute"


def test_provenance_roundtrips_native_persistence(tmp_path) -> None:
    spec = Spectrum(
        mz=np.array([500.0]),
        intensity=np.array([100.0]),
        charge=np.array([1]),
        spectrum_type="deconvoluted",
        deconvolution=_provenance(DEPROTONATED),
    )
    path = tmp_path / "negative.npz"
    spec.save(path)
    restored = Spectrum.load(path)
    assert restored.deconvolution == spec.deconvolution


def test_custom_isotope_model_roundtrips_in_provenance(tmp_path) -> None:
    model = IsotopeModel(
        name="carbon-rich",
        atoms_per_da={"C": 0.05, "O": 0.02},
        isotope_abundances={"C": {0: 0.8, 1: 0.2}},
    )
    raw = Spectrum(
        mz=np.array([500.0, 500.0 + pt.C13_NEUTRON_MASS]),
        intensity=np.array([1000.0, 400.0]),
    )
    result = raw.deconvolute(charge_range=(1, 1), isotope_model=model)
    path = tmp_path / "custom-isotope-model.npz"

    result.save(path)
    restored_spectra = [
        Spectrum.load(path),
        from_matchms(to_matchms(result)),
        from_spectrl_token(to_spectrl_token(result, lossless=True)),
    ]

    for restored in restored_spectra:
        assert restored.deconvolution is not None
        assert restored.deconvolution.isotope_model_definition == model
        assert restored.deconvolution == result.deconvolution


def test_schema_one_provenance_remains_readable() -> None:
    value = _provenance(PROTONATED).to_dict()
    value["schema_version"] = 1
    value["isotope_model"] = "peptide"
    value.pop("isotope_model_definition")

    restored = DeconvolutionProvenance.from_dict(value)

    assert restored.isotope_model == "peptide"
    assert restored.isotope_model_definition is None


def _sodiated_spectrum() -> Spectrum:
    return Spectrum(
        mz=np.array([500.0]),
        intensity=np.array([100.0]),
        charge=np.array([1]),
        spectrum_type="deconvoluted",
        deconvolution=_provenance(SODIATED),
    )


def test_provenance_roundtrips_matchms_payload() -> None:
    spec = _sodiated_spectrum()
    restored = from_matchms(to_matchms(spec))
    assert restored.deconvolution == spec.deconvolution


def test_provenance_roundtrips_spectrl_user_parameter() -> None:
    spec = _sodiated_spectrum()
    restored = from_spectrl_token(to_spectrl_token(spec, lossless=True))
    assert restored.deconvolution == spec.deconvolution


@pytest.mark.parametrize("bad_charge", [0, -1, 1.5])
def test_models_reject_non_positive_integer_charge(bad_charge: float) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        PROTONATED.ion_mz(1000.0, bad_charge)
