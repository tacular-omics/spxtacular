"""Tests for the field-level PSI-MS enums (ActivationType, IMType, Analyzer,
Polarity) and the single-source-of-truth accession mappings that back them.

These do not require the optional ``spectrl`` extra: the accession dicts live in
``spectrl_bridge`` but that module imports cleanly without ``spectrl`` installed.
"""

import re

import numpy as np
import pytest

from spxtacular import (
    ActivationType,
    Analyzer,
    IMType,
    MsnSpectrum,
    Polarity,
    Precursor,
    SpxtacularError,
    ToleranceType,
)
from spxtacular.spectrl_bridge import (
    _ACTIVATION_ACCESSIONS,
    _ACTIVATION_NAMES,
    _ANALYZER_ACCESSIONS,
    _IM_TYPE_ACCESSIONS,
    _POLARITY_FROM_ACCESSION,
)

_ACCESSION_RE = re.compile(r"^MS:\d{7}$")


def _spec(**kwargs) -> MsnSpectrum:
    return MsnSpectrum(
        mz=np.array([100.0, 200.0], dtype=np.float64),
        intensity=np.array([10.0, 20.0], dtype=np.float64),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# StrEnum semantics — members are strings equal to their values
# ---------------------------------------------------------------------------


def test_members_are_strings_equal_to_values() -> None:
    assert ActivationType.HCD == "HCD"
    assert IMType.OOK0 == "ook0"
    assert Analyzer.TOF == "tof"
    assert Polarity.POSITIVE == "positive"
    # StrEnum members ARE str instances, so they slot in anywhere a str is expected.
    assert isinstance(ActivationType.HCD, str)


# ---------------------------------------------------------------------------
# Fields accept both enum members and raw strings (open vocabulary)
# ---------------------------------------------------------------------------


def test_fields_accept_enum_members() -> None:
    spec = _spec(
        activation_type=ActivationType.ETHCD,
        im_type=IMType.DRIFT_TIME_MS,
        analyzer=Analyzer.ORBITRAP,
        polarity=Polarity.NEGATIVE,
    )
    assert spec.activation_type == "EThcD"
    assert spec.im_type == "drift_time_ms"
    assert spec.analyzer == "orbitrap"
    assert spec.polarity == "negative"


def test_fields_canonicalise_known_strings() -> None:
    # Known names (any case), PSI-MS accessions and aliases become members.
    spec = _spec(activation_type="MS:1002481", analyzer="TOF", im_type="1/k0", polarity="Positive")
    assert spec.activation_type is ActivationType.HCD
    assert spec.analyzer is Analyzer.TOF
    assert spec.im_type is IMType.OOK0
    assert spec.polarity is Polarity.POSITIVE


def test_open_vocabulary_fields_keep_unknown_strings() -> None:
    # activation_type and analyzer are open: vendor terms pass through.
    spec = _spec(activation_type="MS:1999999", analyzer="FTMS")
    assert spec.activation_type == "MS:1999999"
    assert spec.analyzer == "FTMS"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"im_type": "banana"},
        {"polarity": "up"},
        {"polarity": 1},
        {"activation_type": ""},
        {"activation_type": 3},
        {"analyzer": "  "},
    ],
)
def test_invalid_enum_like_fields_raise(kwargs) -> None:
    with pytest.raises(SpxtacularError):
        _spec(**kwargs)


def test_precursor_im_type_is_validated() -> None:
    assert Precursor(precursor_mz=1.0, im_type="OOK0").im_type is IMType.OOK0
    with pytest.raises(SpxtacularError, match=r"Precursor\.im_type"):
        Precursor(precursor_mz=1.0, im_type="banana")
    with pytest.raises(SpxtacularError):
        Precursor(precursor_mz=1.0, im_type=5)


def test_enum_coercion_raises_spxtacular_error() -> None:
    assert ToleranceType("PPM") is ToleranceType.PPM
    with pytest.raises(SpxtacularError, match="expected one of 'da', 'ppm'"):
        ToleranceType("ppb")


# ---------------------------------------------------------------------------
# Drift guard — every enum member has a matching accession, no stale keys.
# This is the regression that the single-source-of-truth restructure buys us.
# ---------------------------------------------------------------------------


def test_every_activation_member_has_accession() -> None:
    for member in ActivationType:
        assert member in _ACTIVATION_ACCESSIONS, f"{member!r} missing from _ACTIVATION_ACCESSIONS"
    # and every accession key is a real ActivationType member (no orphan keys)
    for key in _ACTIVATION_ACCESSIONS:
        assert key in tuple(ActivationType)


def test_every_analyzer_member_has_accession() -> None:
    assert set(_ANALYZER_ACCESSIONS) == set(Analyzer)


def test_analyzer_accessions_map_to_members() -> None:
    for member, accession in _ANALYZER_ACCESSIONS.items():
        assert Analyzer.from_accession(accession) is member
        assert _spec(analyzer=accession).analyzer is member
    assert _spec(analyzer="MS:1000484").analyzer is Analyzer.ORBITRAP
    # an accession without a member stays as given
    assert Analyzer.from_accession("MS:9999999") == "MS:9999999"


def test_every_im_type_member_resolves() -> None:
    for member in IMType:
        assert member in _IM_TYPE_ACCESSIONS, f"{member!r} missing from _IM_TYPE_ACCESSIONS"


def test_polarity_covers_both_members() -> None:
    assert set(_POLARITY_FROM_ACCESSION.values()) == set(Polarity)


# ---------------------------------------------------------------------------
# Accession values are well-formed and the activation inverse map round-trips.
# ---------------------------------------------------------------------------


def test_accession_values_well_formed() -> None:
    for acc in (*_ACTIVATION_ACCESSIONS.values(), *_ANALYZER_ACCESSIONS.values(), *_IM_TYPE_ACCESSIONS.values()):
        assert _ACCESSION_RE.match(acc), f"malformed accession {acc!r}"


def test_activation_inverse_map_round_trips() -> None:
    # HCD and PASEF both encode to MS:1002481 ("higher energy beam-type CID"),
    # so that accession cannot round-trip to both. It decodes to HCD, which is
    # what the term means to every other mzML consumer; PASEF is encode-only.
    shared = {ActivationType.HCD, ActivationType.PASEF}
    for member, acc in _ACTIVATION_ACCESSIONS.items():
        assert isinstance(_ACTIVATION_NAMES[acc], ActivationType)
        if member in shared:
            assert _ACTIVATION_NAMES[acc] == ActivationType.HCD
        else:
            assert _ACTIVATION_NAMES[acc] == member
