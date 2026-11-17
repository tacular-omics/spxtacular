"""Activation method -> peptacular fragmentation preset, and ``localize(ion_types="auto")``."""

from __future__ import annotations

from typing import Any

import numpy as np
import peptacular as pt
import pytest

from spxtacular import ActivationType, MsnSpectrum, Spectrum, fragmentation_preset_name, localize
from spxtacular.enums import _ACTIVATION_PRESETS
from spxtacular.errors import SpxtacularError
from spxtacular.spectrl_bridge import _ACTIVATION_ACCESSIONS

AMBIGUOUS = "PEPT[Phospho#g1]IDES[#g1]K"


def _spectrum(ion_types: tuple[Any, ...], activation: str | None) -> Spectrum:
    """A spectrum made only of the given series of PEPT[Phospho]IDESK."""
    frags = pt.fragment("PEPT[Phospho]IDESK", ion_types=ion_types, charges=(1,))
    mz = np.array(sorted(f.mz for f in frags))
    intensity = np.full(len(mz), 1000.0)
    if activation is None:
        return Spectrum(mz=mz, intensity=intensity)
    return MsnSpectrum(mz=mz, intensity=intensity, activation_type=activation)


# --------------------------------------------------------------------------- EAD


def test_ead_member_and_accession() -> None:
    assert ActivationType.EAD.value == "EAD"
    assert ActivationType("ead") is ActivationType.EAD
    # PSI-MS "electron activated dissociation", psi-ms.obo 4.1.261; is_a ECD.
    assert _ACTIVATION_ACCESSIONS[ActivationType.EAD] == "MS:1003294"
    assert ActivationType.from_accession("MS:1003294") == ActivationType.EAD


# ------------------------------------------------------------------- the mapping


def test_every_member_is_decided() -> None:
    """A new ActivationType must be given a preset or listed here deliberately."""
    undecided = set(ActivationType) - set(_ACTIVATION_PRESETS)
    assert undecided == {ActivationType.NETD, ActivationType.PD}


def test_every_preset_name_exists_in_peptacular() -> None:
    """The cross-package drift guard: peptacular owns the preset names."""
    presets = pt.FRAGMENTATION_PRESETS
    assert presets, "peptacular has no FRAGMENTATION_PRESETS"
    for member, name in _ACTIVATION_PRESETS.items():
        assert name in presets, f"{member!r} -> {name!r} is not a peptacular preset"


@pytest.mark.parametrize(
    ("activation", "expected"),
    [
        (ActivationType.CID, "cid"),
        (ActivationType.HCD, "hcd"),
        (ActivationType.ETD, "etd"),
        (ActivationType.ECD, "ecd"),
        (ActivationType.EAD, "ead"),
        (ActivationType.ETHCD, "ethcd"),
        (ActivationType.ETCID, "ethcd"),
        (ActivationType.UVPD, "uvpd"),
        (ActivationType.PQD, "cid"),
        (ActivationType.SID, "cid"),
        (ActivationType.IRMPD, "cid"),
        (ActivationType.BIRD, "cid"),
        (ActivationType.SORI, "cid"),
        (ActivationType.PASEF, "hcd"),
        ("hcd", "hcd"),
        ("  EThcD ", "ethcd"),
        ("MS:1000598", "etd"),
        ("MS:1002481", "hcd"),
    ],
)
def test_preset_names(activation: ActivationType | str, expected: str) -> None:
    assert fragmentation_preset_name(activation) == expected


@pytest.mark.parametrize(
    "activation",
    [None, ActivationType.NETD, ActivationType.PD, "NETD", "PD", "Vendor magic mode", "MS:9999999", ""],
)
def test_no_preset(activation: ActivationType | str | None) -> None:
    assert fragmentation_preset_name(activation) is None


# ----------------------------------------------------- localize(ion_types="auto")


def test_default_ion_types_are_unchanged_by_the_feature() -> None:
    """A caller who passes no ion_types gets b and y, activation method or not."""
    etd = _spectrum(("c", "z."), "ETD")
    result = localize(etd, AMBIGUOUS, tolerance=0.05)
    assert result.provenance.ion_types == ("b", "y")


def test_auto_follows_the_activation_method() -> None:
    etd = _spectrum(("c", "z."), "ETD")
    auto = localize(etd, AMBIGUOUS, ion_types="auto", tolerance=0.05)
    default = localize(etd, AMBIGUOUS, tolerance=0.05)
    assert auto.provenance.ion_types == ("c", "z.")
    assert auto.ascore is not None and default.ascore is not None
    # The point of the feature: b/y ions score nothing on a c/z. spectrum.
    assert auto.ascore > default.ascore
    assert default.ascore == 0.0


@pytest.mark.parametrize(("activation", "expected"), [("HCD", ("b", "y")), ("EAD", ("a", "b", "c", "x", "y", "z."))])
def test_auto_series_per_method(activation: str, expected: tuple[str, ...]) -> None:
    spec = _spectrum(("b", "y"), activation)
    assert localize(spec, AMBIGUOUS, ion_types="auto", tolerance=0.05).provenance.ion_types == expected


def test_auto_is_case_insensitive() -> None:
    spec = _spectrum(("b", "y"), "HCD")
    assert localize(spec, AMBIGUOUS, ion_types=" AUTO ", tolerance=0.05).provenance.ion_types == ("b", "y")


def test_auto_refuses_without_an_activation_method() -> None:
    plain = _spectrum(("b", "y"), None)
    with pytest.raises(SpxtacularError, match="needs an activation method"):
        localize(plain, AMBIGUOUS, ion_types="auto", tolerance=0.05)


@pytest.mark.parametrize("activation", ["NETD", "PD", "Vendor magic mode"])
def test_auto_refuses_a_method_with_no_preset(activation: str) -> None:
    spec = _spectrum(("b", "y"), activation)
    with pytest.raises(SpxtacularError, match="fragmentation preset"):
        localize(spec, AMBIGUOUS, ion_types="auto", tolerance=0.05)


def test_a_string_that_is_not_auto_is_refused() -> None:
    """A bare string used to be read character by character; now it is named."""
    spec = _spectrum(("b", "y"), "HCD")
    with pytest.raises(SpxtacularError, match='sequence of ion types or "auto"'):
        localize(spec, AMBIGUOUS, ion_types="by", tolerance=0.05)
