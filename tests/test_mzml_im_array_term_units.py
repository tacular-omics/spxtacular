"""An ion-mobility term's own unit, taken from the PSI-MS CV, decides the type.

**The expected values in this module come from outside this package.** They are read at
test time out of ``tests/data/psi-ms-im-array-terms.obo``, a byte-for-byte excerpt of
``psi-ms.obo`` (data-version 4.2.17) downloaded from
https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo on 2026-11-17.
Which terms are checked, and what each one's unit is, are the CV's statements, not ours:
the CV records a term's unit as ``relationship: has_units <accession>``.

The one bridge this module declares is from a CV unit to the matching
:class:`~spxtacular.enums.IMType` member, because that vocabulary is this package's own.
It is keyed by the CV's unit name so a renamed unit fails loudly rather than silently.

The two tests that check the reader's own term table import it inside the test body, so the
three behaviour tests here still collect against a build that predates that table.
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

# The CV's unit name -> the IMType whose values are in that unit. The names are quoted
# from the CV; `_cv_terms` asserts each one is still the name the file gives.
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


@pytest.fixture(scope="module")
def cv_terms() -> dict[str, Term]:
    """Every term in the committed CV excerpt, parsed from the excerpt itself."""
    terms = {}
    for block in CV.read_text().split("[Term]")[1:]:
        term = Term(block)
        terms[term.accession] = term
    # The excerpt is trimmed by hand, so prove it still holds what the module reads from it
    # rather than silently testing nothing (an empty excerpt would pass every loop below).
    assert len(terms) >= 10, f"the CV excerpt holds only {len(terms)} terms"
    array_terms = [t for t in terms.values() if "MS:1002893" in t.parents]
    assert len(array_terms) >= 9, f"only {len(array_terms)} ion-mobility array terms in the excerpt"
    for term in terms.values():
        for unit_accession, unit_name in term.units.items():
            assert unit_name in UNIT_NAME_TO_IM_TYPE, f"{term.accession} names an unmapped unit {unit_name!r}"
            assert unit_accession in ("MS:1002814", "UO:0000028", "UO:0000010")
    return terms


def _single_unit_terms(cv_terms: dict[str, Term]) -> dict[str, Term]:
    """The terms the CV gives exactly one unit — the ones a file need not repeat."""
    return {acc: t for acc, t in cv_terms.items() if len(t.units) == 1}


def _read(path: pathlib.Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with MzmlReader(str(path)) as reader:
            return next(iter(reader.ms2))


def _variant(tmp_path: pathlib.Path, accession: str, name: str) -> pathlib.Path:
    """The real Bruker fixture with its array term swapped and its unit element removed."""
    text = FIXTURE.read_text()
    assert ARRAY_TERM in text and OOK0_UNIT in text
    text = text.replace(ARRAY_TERM, f'accession="{accession}" name="{name}"')
    text = text.replace(" " + OOK0_UNIT, "")
    assert OOK0_UNIT not in text, "the unit must be gone for this test to mean anything"
    path = tmp_path / f"{accession.replace(':', '_')}.mzML"
    path.write_text(text)
    return path


def test_our_term_unit_table_is_what_the_cv_says(cv_terms):
    """Our table must hold exactly the CV's single-unit terms, with the CV's unit."""
    from spxtacular.reader import _MZML_IM_ARRAY_UNITS  # noqa: PLC0415

    expected = {acc: next(iter(t.units)) for acc, t in _single_unit_terms(cv_terms).items()}
    # MS:1002815 is a precursor term the reader types without consulting a unit at all
    # (mzmlpy exposes it as `ook0`), so it is not in our table; every other single-unit
    # term must be, and nothing else may be.
    expected.pop("MS:1002815", None)
    assert expected == _MZML_IM_ARRAY_UNITS


def test_multi_unit_terms_are_absent_from_our_table(cv_terms):
    """A term the CV gives two units cannot fix one, so the file must declare it."""
    from spxtacular.reader import _MZML_IM_ARRAY_UNITS  # noqa: PLC0415

    multi = {acc for acc, t in cv_terms.items() if len(t.units) > 1}
    assert multi, "the excerpt should contain terms with more than one unit"
    assert not (multi & set(_MZML_IM_ARRAY_UNITS))


def test_single_unit_array_term_without_a_declared_unit(tmp_path, cv_terms):
    """An array whose term names one unit is typed by it, with no unit element in the file."""
    checked, unsupported = 0, []
    for accession, term in _single_unit_terms(cv_terms).items():
        if "MS:1002893" not in term.parents:
            continue  # not an array term; the precursor case is its own test below
        unit_name = next(iter(term.units.values()))
        spec = _read(_variant(tmp_path, accession, term.name))
        if spec.im is None:
            unsupported.append(accession)
            continue
        assert spec.im_type == UNIT_NAME_TO_IM_TYPE[unit_name], (
            f"{accession} ({term.name}) declares no unit; the CV gives it {unit_name!r}"
        )
        assert "reported in" in term.definition  # the definition states the unit too
        checked += 1
    # Every one of the three must be reachable: each is in this package's term table, so a
    # term the installed mzmlpy does not surface would leave an entry no test can reach.
    assert checked == 3, f"checked {checked} of the CV's 3 single-unit array terms; mzmlpy skipped {unsupported}"


def test_multi_unit_array_term_without_a_declared_unit_stays_generic(tmp_path, cv_terms):
    """With two units in the CV and none in the file, the scale is unknown: stay generic."""
    checked, unsupported = 0, []
    for accession, term in cv_terms.items():
        if "MS:1002893" not in term.parents or len(term.units) < 2:
            continue
        spec = _read(_variant(tmp_path, accession, term.name))
        if spec.im is None:
            # The installed mzmlpy does not count this term as an ion-mobility array, so no
            # array reaches this package and there is nothing of ours to assert. Released
            # mzmlpy 0.11.0 omits MS:1003154 from its ION_MOBILITIES set; a newer one has it.
            unsupported.append(accession)
            continue
        assert spec.im_type == IMType.IM, f"{accession} ({term.name}) names {len(term.units)} units"
        checked += 1
    # A floor rather than the full 6, because which terms mzmlpy surfaces is its decision and
    # moves between its releases. Without the floor this loop could assert nothing in silence.
    assert checked >= 4, f"checked only {checked} multi-unit array terms; mzmlpy skipped {unsupported}"
    assert len(unsupported) <= 2, f"mzmlpy surfaced too few ion-mobility array terms: skipped {unsupported}"


def test_precursor_drift_time_term_without_a_declared_unit(tmp_path, cv_terms):
    """MS:1002476 names millisecond and nothing else, so an undeclared value is in ms."""
    term = cv_terms["MS:1002476"]
    assert term.units == {"UO:0000028": "millisecond"}, "the CV changed; update the reader"
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
    assert precursor.im_type == UNIT_NAME_TO_IM_TYPE[term.units["UO:0000028"]]
    assert precursor.im == pytest.approx(12.5)
