"""Invariant tests: properties that must hold for every input, not just hand-picked examples.

1. Peak-list round trips: any MsnSpectrum written to MGF / MS2 / MSP reads back with the
   same peaks, precursor m/z, signed charge (so polarity) and neutral mass.
2. Ionization grid: every preset and ProForma charge carrier, at charges 1-4 of its
   polarity, recovers the neutral mass of a synthetic isotope envelope through
   deconvolute() + decharge().
3. Isotope envelopes from 500 Da to 200 kDa match an independent convolution reference:
   same apex, sum to one.
4. Every fragment ion type tacular declares gets a theme colour in the annotation table
   and plot, and forward / backward types land on the matching side of the coverage ladder.
"""

from __future__ import annotations

import re
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import peptacular as pt
import pytest
from hypothesis import given
from hypothesis import strategies as st
from paftacular import PaftacularError
from tacular import ELEMENT_LOOKUP, FRAGMENT_ION_LOOKUP, IonType, IonTypeProperty
from tacular.constants import ELECTRON_MASS, PROTON_MASS

from spxtacular import (
    DEPROTONATED,
    IONIZATION_MODELS,
    ISOTOPE_MODELS,
    PROTONATED,
    IonizationModel,
    IsotopeModelType,
    MgfReader,
    Ms2Reader,
    MsnSpectrum,
    MspReader,
    Precursor,
    Spectrum,
    annotate_spectrum,
    build_annot_plot_table,
    resolve_ionization_model,
    sequence_coverage_plot,
    theme,
    write_mgf,
    write_ms2,
    write_msp,
)
from spxtacular.decon.greedy import NEUTRON_MASS
from spxtacular.errors import SpxtacularError
from spxtacular.isotopes import NATURAL_ISOTOPE_ABUNDANCES
from spxtacular.utils import signed_precursor_charge

# ---------------------------------------------------------------------------
# 1. Peak-list round trips
# ---------------------------------------------------------------------------

MZ = st.floats(min_value=50.0, max_value=5000.0, allow_nan=False, allow_infinity=False)
INTENSITY = st.floats(min_value=0.0, max_value=1e10, allow_nan=False, allow_infinity=False)
CHARGE_MAGNITUDE = st.integers(min_value=1, max_value=8)

# Titles a real file might carry: separators, quotes, '=', ':', '/', unicode, inner runs of
# spaces. No line breaks (a title is one header line) and no leading/trailing whitespace
# (every reader strips the value, as the formats intend).
_TITLE_CHARS = st.characters(
    codec="utf-8",
    exclude_categories=("Cc", "Cs", "Zl", "Zp"),
    exclude_characters="\x85",
)
TITLE = (
    st.one_of(
        st.text(_TITLE_CHARS, min_size=1, max_size=40),
        st.sampled_from(
            [
                'run 1  "fraction=3"',
                "controllerType=0 controllerNumber=1 scan=42",
                "F1:2478",
                "# not a comment",
                "a=b=c",
                "Name: PEPTIDE",
                "Ähnlich \u2013 質量",
                "tab\tseparated",
            ]
        ),
    )
    .map(str.strip)
    .filter(bool)
)


@st.composite
def precursors(draw) -> Precursor:
    magnitude = draw(st.one_of(st.none(), CHARGE_MAGNITUDE))
    sign = draw(st.sampled_from([1, -1]))
    im = draw(st.one_of(st.none(), st.floats(min_value=0.5, max_value=1.8)))
    return Precursor(
        precursor_mz=draw(st.floats(min_value=100.0, max_value=4000.0, allow_nan=False)),
        intensity=draw(st.one_of(st.just(0.0), INTENSITY)),
        charge=None if magnitude is None else sign * magnitude,
        im=im,
        im_type=None if im is None else "ook0",
        is_monoisotopic=draw(st.one_of(st.none(), st.booleans())),
    )


@st.composite
def msn_spectra(draw, *, peak_charges: bool = False) -> MsnSpectrum:
    n = draw(st.integers(min_value=0, max_value=25))
    mz = np.asarray(draw(st.lists(MZ, min_size=n, max_size=n)), dtype=np.float64)
    intensity = np.asarray(draw(st.lists(INTENSITY, min_size=n, max_size=n)), dtype=np.float64)
    im = None
    if draw(st.booleans()):
        im = np.asarray(draw(st.lists(st.floats(0.5, 1.8), min_size=n, max_size=n)), dtype=np.float64)
    charge = None
    if peak_charges and draw(st.booleans()):
        charge = np.asarray(draw(st.lists(st.sampled_from([-1, 1, 2, 3, 4]), min_size=n, max_size=n)), dtype=np.int32)
    precs = draw(st.one_of(st.none(), st.lists(precursors(), min_size=1, max_size=3)))
    first_charge = precs[0].charge if precs else None
    # Polarity never contradicts the sign of a known precursor charge.
    if first_charge is None:
        polarity = draw(st.sampled_from([None, "positive", "negative"]))
    elif first_charge < 0:
        polarity = draw(st.sampled_from([None, "negative"]))
    else:
        polarity = draw(st.sampled_from([None, "positive", "negative"]))
    return MsnSpectrum(
        mz=mz,
        intensity=intensity,
        charge=charge,
        im=im,
        spectrum_type="centroid" if charge is None else None,
        ms_level=2,
        scan_number=draw(st.one_of(st.none(), st.integers(min_value=1, max_value=10**7))),
        native_id=draw(st.one_of(st.none(), TITLE)),
        rt=draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=1e4, allow_nan=False))),
        polarity=polarity,
        precursors=precs,
    )


def _round_trip(writer: Callable[..., Path], reader_cls: type, spectra: list[MsnSpectrum]) -> list[MsnSpectrum]:
    with tempfile.TemporaryDirectory() as tmp:
        path = writer(spectra, Path(tmp) / "out.txt")
        with reader_cls(path) as reader:
            return list(reader)


def _expected_signed_charge(spec: MsnSpectrum) -> int | None:
    prec = spec.precursors[0] if spec.precursors else None
    return signed_precursor_charge(prec.charge if prec else None, spec.polarity)


def _neutral_mass(mz: float, signed_charge: int) -> float:
    """Neutral mass under the default model of the charge's sign ([M+H]+ / [M-H]-)."""
    model = DEPROTONATED if signed_charge < 0 else PROTONATED
    return float(model.neutral_mass(mz, abs(signed_charge)))


def _assert_peaks_and_precursor(original: MsnSpectrum, read: MsnSpectrum) -> int | None:
    """Peaks exact; first precursor m/z exact; the signed charge and neutral mass survive."""
    np.testing.assert_array_equal(read.mz, original.mz)
    np.testing.assert_array_equal(read.intensity, original.intensity)
    if not original.precursors:
        return None
    assert read.precursors is not None and len(read.precursors) == 1  # formats hold one precursor
    prec, got = original.precursors[0], read.precursors[0]
    assert got.precursor_mz == prec.precursor_mz
    expected = _expected_signed_charge(original)
    got_signed = signed_precursor_charge(got.charge, read.polarity)
    assert got_signed == expected
    if expected is not None:
        assert read.polarity == ("negative" if expected < 0 else "positive")
        assert got_signed is not None
        assert _neutral_mass(got.precursor_mz, got_signed) == pytest.approx(
            _neutral_mass(prec.precursor_mz, expected), rel=1e-12
        )
    return expected


@given(st.lists(msn_spectra(peak_charges=True), min_size=1, max_size=3))
def test_mgf_round_trip_invariants(spectra: list[MsnSpectrum]) -> None:
    read = _round_trip(write_mgf, MgfReader, spectra)
    assert len(read) == len(spectra)
    for original, got in zip(spectra, read, strict=True):
        _assert_peaks_and_precursor(original, got)
        if original.charge is not None and len(original.mz):
            np.testing.assert_array_equal(got.charge, original.charge)
        expected_title = original.native_id
        if expected_title is None and original.scan_number is not None:
            expected_title = f"scan={original.scan_number}"
        assert got.native_id == expected_title
        assert got.scan_number == original.scan_number
        if original.rt is None:
            assert got.rt is None
        else:
            assert got.rt == original.rt
        if original.precursors and original.precursors[0].intensity != 0.0:
            assert got.precursors is not None
            assert got.precursors[0].intensity == original.precursors[0].intensity


_Z_LINE = re.compile(r"^Z\t(\S+)\t(\S+)$", re.MULTILINE)


@given(st.lists(msn_spectra(), min_size=1, max_size=3))
def test_ms2_round_trip_invariants(spectra: list[MsnSpectrum]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = write_ms2(spectra, Path(tmp) / "out.ms2")
        text = path.read_text(encoding="utf-8")
        with Ms2Reader(path) as reader:
            read = list(reader)
    z_lines = iter(_Z_LINE.findall(text))
    assert len(read) == len(spectra)
    for index, (original, got) in enumerate(zip(spectra, read, strict=True)):
        expected = _assert_peaks_and_precursor(original, got)
        scan = original.scan_number if original.scan_number is not None else index + 1
        assert got.scan_number == scan
        assert got.native_id == (original.native_id if original.native_id is not None else f"scan={scan}")
        if original.rt is None:
            assert got.rt is None
        else:
            assert got.rt == pytest.approx(original.rt, rel=1e-12, abs=1e-9)
        if expected is not None:
            # The Z line holds the singly protonated [M+H]+ mass whatever the polarity.
            z_text, mass_text = next(z_lines)
            assert int(z_text) == expected
            assert original.precursors is not None
            # Closed form: M = mz*|z| - z*m(H+), so [M+H]+ = mz*|z| - z*m(H+) + m(H+).
            mz = original.precursors[0].precursor_mz
            mh = mz * abs(expected) - expected * PROTON_MASS + PROTON_MASS
            assert float(mass_text) == pytest.approx(mh, rel=1e-12)
    assert next(z_lines, None) is None


@given(st.lists(msn_spectra(), min_size=1, max_size=3))
def test_msp_round_trip_invariants(spectra: list[MsnSpectrum]) -> None:
    read = _round_trip(write_msp, MspReader, spectra)
    assert len(read) == len(spectra)
    for original, got in zip(spectra, read, strict=True):
        _assert_peaks_and_precursor(original, got)
        expected_name = original.native_id
        if expected_name is None and original.scan_number is not None:
            expected_name = f"scan={original.scan_number}"
        assert got.native_id == expected_name
        if original.polarity is not None:
            assert got.polarity == original.polarity  # MSP has its own Ion_mode line
        if original.rt is None:
            assert got.rt is None
        else:
            assert got.rt == original.rt


# ---------------------------------------------------------------------------
# 2. Ionization grid: deconvolute + decharge recovers the neutral mass
# ---------------------------------------------------------------------------

# ProForma 2.1 charge carriers (section 11.5), resolved through peptacular's parser.
_PROFORMA_CARRIERS = (
    "H:z+1",  # proton (returns the protonated preset)
    "H-1:z-1",  # proton loss
    "Na:z+1",
    "Na:z+1^2",  # one carrier with an occurrence count
    "Na:z+1,Na:z+1",  # the same carrier listed twice
    "K:z+1",
    "Li:z+1",
    "Cs:z+1",
    "Ag:z+1",
    "NH4:z+1",
    "Cl:z-1",
    "Br:z-1",
    "H:z-1",  # hydride attachment, [M+H]-
)

# Electron loss ([M]+.) and electron attachment ([M]-.): no element formula expresses
# these, so they are models built directly. The carrier mass is the per-charge ion
# mass delta, so electron loss is -m(e) at positive polarity.
_ELECTRON_MODELS = (
    IonizationModel("electron loss", "positive", -ELECTRON_MASS, carrier="e"),
    IonizationModel("electron attachment", "negative", ELECTRON_MASS, carrier="e"),
)

_GRID_MODELS: dict[str, IonizationModel] = {
    **{f"preset:{name}": model for name, model in IONIZATION_MODELS.items()},
    **{f"proforma:{text}": resolve_ionization_model(text) for text in _PROFORMA_CARRIERS},
    **{model.name: model for model in _ELECTRON_MODELS},
    "custom:+38.96": resolve_ionization_model(38.963158),
    "custom:-34.97": resolve_ionization_model(-34.969402 + ELECTRON_MASS),
}


# The envelopes are noise-free and exact, so recovery is limited only by float error.
_RECOVERY_TOLERANCE_DA = 1e-4


def _mass(symbol: str) -> float:
    return ELEMENT_LOOKUP.get_mass(symbol, monoisotopic=True)


# Independent carrier masses: the signed ion-mass change per unit charge, from tacular's
# monoisotopic element masses. A cation is the atom less one electron; an anion the atom
# plus one electron; losing a proton removes the cation mass.
_EXPECTED_CARRIER: dict[str, tuple[str, Callable[[], float]]] = {
    "preset:protonated": ("positive", lambda: _mass("H") - ELECTRON_MASS),
    "preset:deprotonated": ("negative", lambda: -(_mass("H") - ELECTRON_MASS)),
    "preset:sodiated": ("positive", lambda: _mass("Na") - ELECTRON_MASS),
    "preset:ammoniated": ("positive", lambda: _mass("N") + 4 * _mass("H") - ELECTRON_MASS),
    "proforma:H:z+1": ("positive", lambda: _mass("H") - ELECTRON_MASS),
    "proforma:H-1:z-1": ("negative", lambda: -(_mass("H") - ELECTRON_MASS)),
    "proforma:H:z-1": ("negative", lambda: _mass("H") + ELECTRON_MASS),
    "proforma:Na:z+1": ("positive", lambda: _mass("Na") - ELECTRON_MASS),
    "proforma:Na:z+1^2": ("positive", lambda: _mass("Na") - ELECTRON_MASS),
    "proforma:NH4:z+1": ("positive", lambda: _mass("N") + 4 * _mass("H") - ELECTRON_MASS),
    "proforma:K:z+1": ("positive", lambda: _mass("K") - ELECTRON_MASS),
    "proforma:Li:z+1": ("positive", lambda: _mass("Li") - ELECTRON_MASS),
    "proforma:Cl:z-1": ("negative", lambda: _mass("Cl") + ELECTRON_MASS),
    "proforma:Br:z-1": ("negative", lambda: _mass("Br") + ELECTRON_MASS),
    "electron loss": ("positive", lambda: -ELECTRON_MASS),
    "electron attachment": ("negative", lambda: ELECTRON_MASS),
}


@pytest.mark.parametrize("key", list(_EXPECTED_CARRIER))
def test_carrier_mass_matches_element_masses(key: str) -> None:
    polarity, expected_mass = _EXPECTED_CARRIER[key]
    model = _GRID_MODELS[key]
    assert model.polarity == polarity
    # 1e-7 Da: the CODATA proton mass differs from m(H) - m(e) by the 13.6 eV binding
    # energy of hydrogen (1.5e-8 Da); a wrong element, charge or electron sign is >= 5e-4 Da.
    assert model.carrier_mass == pytest.approx(expected_mass(), abs=1e-7)


def _envelope_spectrum(neutral_mass: float, charge: int, model: IonizationModel) -> MsnSpectrum:
    """Centroided isotope envelope of ``neutral_mass`` at ``charge`` with ``model``'s carrier."""
    dist = ISOTOPE_MODELS[IsotopeModelType.PEPTIDE].distribution(neutral_mass)
    keep = np.flatnonzero(dist / dist.max() >= 0.02)
    mz = (neutral_mass + keep * NEUTRON_MASS + charge * model.carrier_mass) / charge
    return MsnSpectrum(
        mz=mz,
        intensity=1e6 * dist[keep],
        spectrum_type="centroid",
        ms_level=1,
        polarity=model.polarity,
    )


@pytest.mark.parametrize("charge", [1, 2, 3, 4])
@pytest.mark.parametrize("model", list(_GRID_MODELS.values()), ids=list(_GRID_MODELS))
def test_deconvolute_decharge_recovers_neutral_mass(model: IonizationModel, charge: int) -> None:
    neutral = 987.654 + 611.3 * charge
    spec = _envelope_spectrum(neutral, charge, model)

    decon = spec.deconvolute(charge_range=(1, 4), tolerance=10, ionization_model=model)
    apex = int(np.argmax(decon.intensity))
    assert decon.charge is not None
    assert int(decon.charge[apex]) == charge  # charge magnitudes stay positive
    assert decon.deconvolution is not None
    assert decon.deconvolution.ionization_model == model

    neutral_spec = decon.decharge()
    found = float(neutral_spec.mz[int(np.argmax(neutral_spec.intensity))])
    assert found == pytest.approx(neutral, abs=_RECOVERY_TOLERANCE_DA)
    # The sign of the charge is the polarity the provenance records.
    assert neutral_spec.deconvolution is not None
    signed = signed_precursor_charge(charge, neutral_spec.deconvolution.ionization_model.polarity)
    assert signed == (charge if model.polarity == "positive" else -charge)


@pytest.mark.parametrize("charge", [1, 2, 3, 4])
@pytest.mark.parametrize("polarity", ["positive", "negative"])
def test_default_model_follows_scan_polarity(polarity: str, charge: int) -> None:
    model = PROTONATED if polarity == "positive" else DEPROTONATED
    neutral = 1500.25 * charge
    decon = _envelope_spectrum(neutral, charge, model).deconvolute(charge_range=(1, 4), tolerance=10)
    assert decon.deconvolution is not None
    assert decon.deconvolution.ionization_model is model
    found = float(decon.decharge().mz[int(np.argmax(decon.intensity))])
    assert found == pytest.approx(neutral, abs=_RECOVERY_TOLERANCE_DA)


@pytest.mark.parametrize("text", ["Na:z+1,H:z+1", "K:z+1,Na:z+1", "Na:z+1,Cl:z-1"])
def test_mixed_carriers_are_rejected_not_guessed(text: str) -> None:
    # An IonizationModel is one carrier per unit charge; a mixture has no single
    # per-charge mass, so it must raise rather than silently pick one carrier.
    with pytest.raises(SpxtacularError, match="mixed charge carriers"):
        resolve_ionization_model(text)


@pytest.mark.parametrize(("text", "preset"), [("H:z+1", "protonated"), ("H-1:z-1", "deprotonated")])
def test_proton_carriers_resolve_to_presets(text: str, preset: str) -> None:
    assert resolve_ionization_model(text) is IONIZATION_MODELS[preset]


@pytest.mark.parametrize("text", ["Na:z+1^2", "Na:z+1,Na:z+1"])
def test_repeated_sodium_carrier_is_the_sodiated_preset(text: str) -> None:
    assert resolve_ionization_model(text) is IONIZATION_MODELS["sodiated"]


# ---------------------------------------------------------------------------
# 3. Isotope envelopes from 500 Da to 200 kDa
# ---------------------------------------------------------------------------


def _reference_envelope(composition: dict[str, int], length: int) -> np.ndarray:
    """Isotope distribution by direct polynomial convolution (exponentiation by squaring).

    Independent of BRAIN's Newton-Girard recurrence; the window is long enough that
    truncation does not affect the first ``length`` peaks.
    """
    out = np.zeros(length)
    out[0] = 1.0
    for element, count in composition.items():
        pattern = np.zeros(length)
        for offset, abundance in NATURAL_ISOTOPE_ABUNDANCES[element]:
            if offset < length:
                pattern[offset] = abundance
        power = np.zeros(length)
        power[0] = 1.0
        base, k = pattern, count
        while k:
            if k & 1:
                power = np.convolve(power, base)[:length]
            base = np.convolve(base, base)[:length]
            k >>= 1
        out = np.convolve(out, power)[:length]
    return out


_ENVELOPE_MASSES = (500.0, 1500.0, 5000.0, 12000.0, 30000.0, 75000.0, 150000.0, 200000.0)


@pytest.mark.parametrize("mass", _ENVELOPE_MASSES)
@pytest.mark.parametrize("model_name", [str(name) for name in ISOTOPE_MODELS])
def test_envelope_apex_and_sum_match_reference(model_name: str, mass: float) -> None:
    model = next(m for name, m in ISOTOPE_MODELS.items() if str(name) == model_name)
    dist = model.distribution(mass)
    reference = _reference_envelope(model.estimate_composition(mass), len(dist) + 64)

    assert dist.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(dist >= 0.0)
    # The window covers the envelope: what it leaves out is negligible.
    assert reference[: len(dist)].sum() > 1.0 - 1e-6
    true_apex = int(np.argmax(reference))
    assert model.apex_index(mass) == true_apex
    assert int(np.argmax(dist)) == true_apex
    np.testing.assert_allclose(dist, reference[: len(dist)] / reference[: len(dist)].sum(), atol=1e-9)

    adaptive = model.adaptive_distribution(mass)
    assert adaptive.sum() == pytest.approx(1.0, abs=1e-9)
    assert int(np.argmax(adaptive)) == true_apex


def test_large_protein_apex_is_past_the_old_32_peak_window() -> None:
    model = ISOTOPE_MODELS[IsotopeModelType.PEPTIDE]
    apexes = [model.apex_index(mass) for mass in (60000.0, 100000.0, 200000.0)]
    assert all(apex > 31 for apex in apexes)
    assert apexes == sorted(apexes)


@given(st.floats(min_value=500.0, max_value=200000.0, allow_nan=False))
def test_apex_matches_reference_at_any_mass(mass: float) -> None:
    model = ISOTOPE_MODELS[IsotopeModelType.PEPTIDE]
    dist = model.distribution(mass)
    reference = _reference_envelope(model.estimate_composition(mass), len(dist) + 64)
    assert dist.sum() == pytest.approx(1.0, abs=1e-9)
    assert model.apex_index(mass) == int(np.argmax(reference))


# ---------------------------------------------------------------------------
# 4. Every tacular ion type: colour from the theme, ladder side from is_forward / is_backward
# ---------------------------------------------------------------------------

# V, T and I give the amino-acid-specific side-chain ions (d-valine, wa-threonine, ...) a site.
_LADDER_PEPTIDE = "PEVTIDEKVTIR"
_ION_INFO = {str(info.ion_type.value): info for info in FRAGMENT_ION_LOOKUP.values()}
_ION_TYPES = list(_ION_INFO)
_SIDE_CHAIN = IonTypeProperty.AA_SPECIFIC_FWD | IonTypeProperty.AA_SPECIFIC_BWD


# Backbone ladder types (and their hydrogen-shifted variants), immonium and precursor ions
# each have a series colour; only side-chain (d/v/w), internal and intact-neutral ions are
# drawn in the neutral colour.
_SERIES_COLOURED = {
    t
    for t, info in _ION_INFO.items()
    if ((info.is_forward or info.is_backward) and not info.properties & _SIDE_CHAIN)
    or info.ion_type in (IonType.IMMONIUM, IonType.PRECURSOR)
}


def _theme_colours(mode: theme.ThemeMode) -> set[str]:
    return set(theme._CATEGORICAL[mode]) | {theme.neutral_color(mode)}


def _fragments_of(ion_type: str) -> tuple[list, Spectrum]:
    frags = [
        f
        for f in pt.fragment(_LADDER_PEPTIDE, ion_types=(_ION_INFO[ion_type].ion_type,), charges=[1])
        if f.mz is not None and np.isfinite(f.mz)
    ]
    assert frags, f"peptacular produced no {ion_type} fragments for {_LADDER_PEPTIDE}"
    mz = np.unique(np.array([f.mz for f in frags], dtype=np.float64))
    return frags, Spectrum(mz=mz, intensity=np.linspace(1e4, 1e5, len(mz)))


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize("ion_type", _ION_TYPES)
def test_every_ion_type_is_coloured_by_theme_ion_color(ion_type: str, mode: theme.ThemeMode) -> None:
    frags, spec = _fragments_of(ion_type)
    table = build_annot_plot_table(spec, frags, tolerance=0.001, tolerance_unit="da", theme_mode=mode)
    matched = table[table["series"] != "unmatched"]
    assert len(matched) == len(spec.mz)  # every peak is one of the fragments
    for series, colour in zip(matched["series"], matched["color"], strict=True):
        assert colour == theme.ion_color(series, mode)
        assert colour in _theme_colours(mode)
        if ion_type in _SERIES_COLOURED:
            assert colour != theme.neutral_color(mode), f"{ion_type} fell back to the neutral colour"

    fs = annotate_spectrum(spec, frags, tolerance=0.001, tolerance_unit="da", theme_mode=mode, backend="spec")
    sticks = [m for cell in fs.cells for panel in cell.panels for m in panel.marks if type(m).__name__ == "Sticks"]
    assert sticks
    for mark in sticks:
        if mark.name != "unmatched":
            assert mark.color == theme.ion_color(mark.name, mode)


@pytest.mark.parametrize(
    "ion_type",
    [t for t in _ION_TYPES if _ION_INFO[t].is_forward or _ION_INFO[t].is_backward],
)
def test_terminal_ion_types_sit_on_their_ladder_side(ion_type: str) -> None:
    info = _ION_INFO[ion_type]
    above = bool(info.is_forward)
    n = len(_LADDER_PEPTIDE)
    frags, spec = _fragments_of(ion_type)
    fs = sequence_coverage_plot(
        spec, _LADDER_PEPTIDE, frags, tolerance=0.001, tolerance_unit="da", theme_mode="light", backend="spec"
    )
    marks = [m for cell in fs.cells for panel in cell.panels for m in panel.marks]
    residues = [m for m in marks if getattr(m, "name", None) == "residue"]
    ticks = [m for m in marks if getattr(m, "name", None) == "coverage_tick"]
    (dy,) = {m.dy for m in residues}
    dxs = sorted(m.dx for m in residues)
    assert ticks, f"{ion_type} fragments left no ticks on the coverage ladder"
    assert all(t.color in _theme_colours("light") for t in ticks)

    bond_at = {round((dxs[k - 1] + dxs[k]) / 2.0, 6): k for k in range(1, n)}
    stems = [seg for t in ticks for seg in t.segments if seg[2] == seg[4]]
    drawn = sorted(bond_at[round(seg[2], 6)] for seg in stems)
    positions = {f.position for f in frags if 0 < f.position < n}
    assert drawn == sorted(positions if above else {n - p for p in positions})
    assert all((seg[5] > dy) == above for seg in stems)


@pytest.mark.parametrize(
    "ion_type",
    [t for t in _ION_TYPES if not (_ION_INFO[t].is_forward or _ION_INFO[t].is_backward)],
)
def test_non_terminal_ion_types_leave_the_ladder_empty(ion_type: str) -> None:
    frags, spec = _fragments_of(ion_type)
    fs = sequence_coverage_plot(spec, _LADDER_PEPTIDE, frags, tolerance=0.001, tolerance_unit="da", backend="spec")
    ticks = [
        m
        for cell in fs.cells
        for panel in cell.panels
        for m in panel.marks
        if getattr(m, "name", None) == "coverage_tick"
    ]
    assert ticks == []


def test_uncharged_fragment_still_raises_in_the_annotation_table() -> None:
    # Only the charged intact neutral (n) gets a fallback label; mzPAF's other refusals stand.
    frags = pt.fragment(_LADDER_PEPTIDE, ion_types=("b",), charges=[0])
    spec = Spectrum(mz=np.unique([f.mz for f in frags]), intensity=np.ones(len(frags)))
    with pytest.raises(PaftacularError, match="uncharged"):
        build_annot_plot_table(spec, frags, tolerance=0.001, tolerance_unit="da")


@pytest.mark.parametrize(("charge", "label"), [(1, "n"), (2, "n^2"), (-2, "n^-2")])
def test_intact_neutral_fragment_label(charge: int, label: str) -> None:
    frags = pt.fragment(_LADDER_PEPTIDE, ion_types=("n",), charges=[charge])
    spec = Spectrum(mz=np.array([frags[0].mz]), intensity=np.array([1e5]))
    table = build_annot_plot_table(spec, frags, tolerance=0.001, tolerance_unit="da")
    assert list(table.loc[table["series"] != "unmatched", "label"]) == [label]
