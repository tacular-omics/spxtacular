"""An ion-mobility array term whose CV definition states its unit types the values.

**The expected values in this module come from outside this package.** They are read at import
time out of ``tests/data/psi-ms-im-array-terms.obo``, a byte-for-byte excerpt of ``psi-ms.obo``
(data-version 4.2.17) downloaded from
https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo on 2026-11-17. Which
terms are checked, how many units each has, and what those units are are the CV's statements,
not ours: the CV records a term's unit as ``relationship: has_units <accession>``, and the
definitions of the three inverse-reduced array terms also say the values are "reported in
volt-second per square centimeter".

What the CV does **not** say, and what this module therefore does not claim to test: what a
reader should assume when a unit is absent. ``has_units`` is declared in psi-ms.obo as a bare
Typedef with no def and no comment; ``mzML1.1.0.xsd`` calls ``unitAccession`` "An optional CV
accession number for the unit term associated with the value, if any"; and the semantic
validator's ``ms-mapping.xml`` constrains accessions only and never mentions units. The three
array terms below are covered because their own definitions state the unit **of the values**.
The precursor term MS:1002476 is not: it names one unit but its definition states none, so this
module pins that it stays generic rather than being resolved from the term.

The only bridge declared here is from a CV unit to the matching
:class:`~spxtacular.enums.IMType` member, because that vocabulary is this package's own. It is
keyed by the CV's unit name so a renamed unit fails loudly rather than silently.
"""

import pathlib
import re
import warnings

import pytest

pytest.importorskip("mzmlpy")

from spxtacular import IMType  # noqa: E402
from spxtacular.reader import MzmlReader  # noqa: E402

CV = pathlib.Path(__file__).parent / "data" / "psi-ms-im-array-terms.obo"
FIXTURE = pathlib.Path(__file__).parent / "data" / "bruker_im_one_spectrum.mzML"
ARRAY_TERM = 'accession="MS:1003006" name="mean inverse reduced ion mobility array"'
OOK0_UNIT = 'unitCvRef="MS" unitAccession="MS:1002814" unitName="volt-second per square centimeter"'
ARRAY_PARENT = "MS:1002893"  # ion mobility array

# The one CV term an installed mzmlpy need not surface as an ion-mobility array: released
# mzmlpy 0.11.0 omits MS:1003154 "deconvoluted ion mobility array" from its ION_MOBILITIES set
# (measured in the wheel-plus-PyPI environment), and its unreleased fix adds it. No array
# reaches this package for a term mzmlpy does not list, so there is nothing of ours to assert.
# Any other term that fails to arrive is a defect, not a version difference.
MZMLPY_MAY_NOT_SURFACE = {"MS:1003154"}

# The CV's unit name -> the IMType whose values are in that unit. The names are quoted from the
# CV; `test_cv_excerpt_is_intact` asserts every unit name in the excerpt is still one of these.
UNIT_NAME_TO_IM_TYPE = {
    "volt-second per square centimeter": IMType.OOK0,
    "millisecond": IMType.DRIFT_TIME_MS,
    "second": IMType.DRIFT_TIME_MS,
}


def _one(pattern: str, block: str, flags: int = re.M) -> str:
    """The first capture of ``pattern`` in ``block``, or a failure naming what was missing."""
    match = re.search(pattern, block, flags)
    assert match is not None, f"the CV excerpt has no {pattern!r}; re-extract it from psi-ms.obo"
    return match.group(1)


class Term:
    """One ``[Term]`` block of the CV excerpt."""

    def __init__(self, block: str) -> None:
        self.accession = _one(r"^id: (MS:\d+)$", block)
        self.name = _one(r"^name: (.+)$", block)
        self.definition = " ".join(_one(r'^def: "(.+?)" \[', block, re.M | re.S).split())
        self.units = dict(re.findall(r"^relationship: has_units (\S+) ! (.+)$", block, re.M))
        self.parents = set(re.findall(r"^is_a: (MS:\d+)", block, re.M))

    @property
    def is_array(self) -> bool:
        return ARRAY_PARENT in self.parents


# Parsed at import so each CV term can be its own test case. A missing or trimmed-away term
# breaks collection loudly rather than quietly shrinking what is checked.
CV_TERMS = {t.accession: t for t in (Term(b) for b in CV.read_text().split("[Term]")[1:])}
SINGLE_UNIT_ARRAYS = sorted(a for a, t in CV_TERMS.items() if t.is_array and len(t.units) == 1)
MULTI_UNIT_ARRAYS = sorted(a for a, t in CV_TERMS.items() if t.is_array and len(t.units) > 1)


def _read(path: pathlib.Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with MzmlReader(str(path)) as reader:
            return next(iter(reader.ms2))


def _variant(tmp_path: pathlib.Path, accession: str):
    """The real Bruker fixture with its array term swapped and its unit element removed.

    Returns the spectrum, or ``None`` when the installed mzmlpy does not surface an array with
    that term at all, in which case nothing of ours is reachable.
    """
    term = CV_TERMS[accession]
    text = FIXTURE.read_text()
    assert ARRAY_TERM in text and OOK0_UNIT in text
    text = text.replace(ARRAY_TERM, f'accession="{accession}" name="{term.name}"')
    text = text.replace(" " + OOK0_UNIT, "")
    assert OOK0_UNIT not in text, "the unit must be gone for this test to mean anything"
    path = tmp_path / f"{accession.replace(':', '_')}.mzML"
    path.write_text(text)
    spec = _read(path)
    if spec.im is None:
        assert accession in MZMLPY_MAY_NOT_SURFACE, (
            f"the installed mzmlpy did not surface an ion-mobility array for {accession}, which is not a known gap"
        )
        pytest.skip(f"the installed mzmlpy does not list {accession} as an ion-mobility array")
    return spec


def test_cv_excerpt_is_intact():
    """The excerpt is trimmed by hand, so prove it still holds what this module reads from it."""
    assert len(CV_TERMS) >= 10, f"the CV excerpt holds only {len(CV_TERMS)} terms"
    assert len(SINGLE_UNIT_ARRAYS) == 3, SINGLE_UNIT_ARRAYS
    assert len(MULTI_UNIT_ARRAYS) == 6, MULTI_UNIT_ARRAYS
    for term in CV_TERMS.values():
        for unit_accession, unit_name in term.units.items():
            assert unit_name in UNIT_NAME_TO_IM_TYPE, f"{term.accession} names an unmapped unit {unit_name!r}"
            assert unit_accession in ("MS:1002814", "UO:0000028", "UO:0000010")


def test_our_term_unit_table_is_what_the_cv_says():
    """Our table must hold exactly the CV's single-unit ion-mobility *array* terms.

    Scoped to the children of MS:1002893 because that is what the table is for: the CV's other
    single-unit ion-mobility terms are precursor values, and an absent unit on one of those is
    not resolved here (see the module docstring).
    """
    from spxtacular.reader import _MZML_IM_ARRAY_UNITS  # noqa: PLC0415

    expected = {a: next(iter(CV_TERMS[a].units)) for a in SINGLE_UNIT_ARRAYS}
    assert expected == _MZML_IM_ARRAY_UNITS


def test_multi_unit_terms_are_absent_from_our_table():
    """A term the CV gives two units cannot fix one, so the file must declare it."""
    from spxtacular.reader import _MZML_IM_ARRAY_UNITS  # noqa: PLC0415

    assert not (set(MULTI_UNIT_ARRAYS) & set(_MZML_IM_ARRAY_UNITS))


@pytest.mark.parametrize("accession", SINGLE_UNIT_ARRAYS)
def test_single_unit_array_term_without_a_declared_unit(tmp_path, accession):
    """One case per CV term: its definition states the unit, so an undeclared array has it."""
    term = CV_TERMS[accession]
    unit_name = next(iter(term.units.values()))
    assert f"reported in {unit_name}" in term.definition, (
        f"{accession} is in this table because its definition states the unit; it no longer does"
    )
    spec = _variant(tmp_path, accession)
    assert spec.im_type == UNIT_NAME_TO_IM_TYPE[unit_name], (
        f"{accession} ({term.name}) declares no unit; the CV gives it {unit_name!r}"
    )


@pytest.mark.parametrize("accession", MULTI_UNIT_ARRAYS)
def test_multi_unit_array_term_without_a_declared_unit_stays_generic(tmp_path, accession):
    """With two units in the CV and none in the file, the scale is unknown: stay generic."""
    term = CV_TERMS[accession]
    assert len(term.units) > 1
    spec = _variant(tmp_path, accession)
    assert spec.im_type == IMType.IM, f"{accession} ({term.name}) names {len(term.units)} units"


def test_precursor_drift_time_term_without_a_declared_unit_stays_generic(tmp_path):
    """A precursor drift time with no unit is not resolved from the term, and here is why.

    MS:1002476 does name one unit, millisecond, but unlike the array terms above its definition
    states no unit, and no source says what an absent unit means. This pins that stop as
    behaviour, so a later change cannot start assuming it silently.
    """
    term = CV_TERMS["MS:1002476"]
    assert term.units == {"UO:0000028": "millisecond"}, "the CV changed; re-read it before acting"
    assert "reported in" not in term.definition, "its definition states no unit"
    text = FIXTURE.read_text()
    # The selected ion's own peak intensity param: the drift time goes beside it, inside
    # <selectedIon>, which is where mzML puts a precursor's ion-mobility value.
    anchor = (
        'name="peak intensity" value="2614.0" unitCvRef="MS" unitAccession="MS:1000131" '
        'unitName="number of detector counts"/>'
    )
    assert anchor in text
    drift = f'<cvParam cvRef="MS" accession="{term.accession}" name="{term.name}" value="12.5"/>'
    path = tmp_path / "precursor_drift.mzML"
    path.write_text(text.replace(anchor, anchor + "\n" + drift, 1))
    (precursor,) = _read(path).precursors
    assert precursor.im_type == IMType.IM
    assert precursor.im == pytest.approx(12.5)
