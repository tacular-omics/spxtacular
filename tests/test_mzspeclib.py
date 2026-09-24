"""Tests for mzSpecLib reading/writing and mzPAF annotations in MSP/MGF output."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import paftacular as paf
import peptacular as pt
import pytest

from spxtacular import (
    ActivationType,
    Analyte,
    CvParam,
    IMType,
    Interpretation,
    LibraryEntry,
    MgfReader,
    MsnSpectrum,
    MspReader,
    Polarity,
    Precursor,
    SpectralLibrary,
    SpectrumType,
    SpxtacularError,
    match_fragments,
    read_mzspeclib,
    write_mgf,
    write_msp,
    write_mzspeclib,
)

DATA = Path(__file__).parent / "data" / "mzspeclib"
DIANN = DATA / "phl004_canonical_sall_pv_plasma.head.diann.mzSpecLib.txt"


def make_spectrum(**overrides: object) -> MsnSpectrum:
    values: dict[str, object] = {
        "mz": np.array([175.119, 263.087, 376.171, 505.214]),
        "intensity": np.array([100.0, 55.5, 20.25, 7.0]),
        "spectrum_type": SpectrumType.CENTROID,
        "ms_level": 2,
        "scan_number": 17,
        "rt": 1234.5,
        "polarity": Polarity.POSITIVE,
        "collision_energy": 27.0,
        "activation_type": ActivationType.HCD,
        "injection_time": 22.0,
        "total_ion_current": 182.75,
        "precursors": [
            Precursor(
                precursor_mz=400.2,
                intensity=1.5e6,
                charge=2,
                im=0.95,
                im_type=IMType.OOK0,
                is_monoisotopic=True,
            )
        ],
    }
    values.update(overrides)
    return MsnSpectrum(**values)  # type: ignore[arg-type]


def make_entry(**overrides: object) -> LibraryEntry:
    values: dict[str, object] = {
        "key": 1,
        "name": "PEPTIDE/2",
        "analytes": (
            Analyte(
                peptidoform="PEPT[Phospho]IDE/2", attributes=(CvParam("MS:1000885", "protein accession", "P12345"),)
            ),
        ),
        "interpretations": (Interpretation(score=0.99),),
        "peak_annotations": ["y1/0.3ppm", None, "b3/-1.1ppm,y2-H2O^2/0.5ppm", "?"],
        "attributes": (
            CvParam("MS:1003063", "universal spectrum identifier", "mzspec:PXD000001:run:scan:17"),
            CvParam("MS:1003275", "other attribute name", "Source", group=7),
            CvParam("MS:1003276", "other attribute value", "unit test", group=7),
        ),
    }
    spectrum = values.pop("spectrum", None) or make_spectrum()
    values.update(overrides)
    return LibraryEntry(spectrum, **values)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["lib.mzspeclib.txt", "lib.mzspeclib.json", "lib.mzspeclib.txt.gz", "lib.json.gz"])
def test_round_trip(tmp_path: Path, name: str) -> None:
    entries = [make_entry(), make_entry(key=5, name=None, peak_annotations=None, interpretations=())]
    path = write_mzspeclib(entries, tmp_path / name)
    library = read_mzspeclib(path)
    assert list(library) == entries
    if name.endswith(".gz"):
        assert path.read_bytes()[:2] == b"\x1f\x8b"


def test_text_and_json_read_back_equal(tmp_path: Path) -> None:
    library = SpectralLibrary(
        entries=[make_entry()],
        attributes=(CvParam("MS:1003188", "library name", "test"),),
        clusters={3: (CvParam("MS:1003268", "spectrum cluster member spectrum keys", (1,)),)},
    )
    text = read_mzspeclib(write_mzspeclib(library, tmp_path / "a.txt"))
    as_json = read_mzspeclib(write_mzspeclib(library, tmp_path / "a.json"))
    assert text == as_json == library


def test_fields_map_to_spectrum(tmp_path: Path) -> None:
    entry = read_mzspeclib(write_mzspeclib(make_entry(), tmp_path / "a.txt"))[0]
    spec = entry.spectrum
    assert spec.rt == 1234.5
    assert spec.polarity == Polarity.POSITIVE
    assert spec.activation_type == ActivationType.HCD
    assert spec.collision_energy == 27.0
    assert spec.scan_number == 17
    (prec,) = spec.precursors or []
    assert (prec.precursor_mz, prec.charge, prec.im, prec.im_type) == (400.2, 2, 0.95, IMType.OOK0)
    assert prec.is_monoisotopic is True
    assert entry.peptidoform == pt.parse("PEPT[Phospho]IDE/2")
    assert entry.charge == 2
    assert entry.score == 0.99
    assert entry.name == "PEPTIDE/2"


def test_text_output_uses_cv_terms(tmp_path: Path) -> None:
    text = write_mzspeclib([make_entry()], tmp_path / "a.txt").read_text()
    assert text.startswith("<mzSpecLib>\nMS:1003186|library format version=1.0\n<Spectrum=1>\n")
    assert "MS:1003208|experimental precursor monoisotopic m/z=400.2" in text
    assert "MS:1003270|proforma peptidoform ion notation=PEPT[Phospho]IDE/2" in text
    assert "[1]MS:1000894|retention time=1234.5\n[1]UO:0000000|unit=UO:0000010|second" in text
    assert "MS:1000044|dissociation method=MS:1000422|beam-type collision-induced dissociation" in text
    assert "MS:1003103|ion annotation format=MS:1003104|mzPAF peptide ion annotation format" in text
    # User groups are renumbered after the mapped ones.
    assert "[3]MS:1003275|other attribute name=Source" in text
    assert "376.171\t20.25\tb3/-1.1ppm,y2-H2O^2/0.5ppm" in text
    assert "263.087\t55.5\n" in text


def test_json_output_shape(tmp_path: Path) -> None:
    doc = json.loads(write_mzspeclib([make_entry()], tmp_path / "a.json").read_text())
    assert doc["format_version"] == "1.0"
    (spectrum,) = doc["spectra"]
    assert {"accession": "MS:1003237", "name": "library spectrum key", "value": 1} in spectrum["attributes"]
    assert spectrum["peak_annotations"][1] == []
    assert spectrum["peak_annotations"][2] == ["b3/-1.1ppm", "y2-H2O^2/0.5ppm"]
    assert spectrum["analytes"]["1"]["attributes"][0]["accession"] == "MS:1003270"


def test_minimal_spectrum_round_trips(tmp_path: Path) -> None:
    spec = MsnSpectrum(mz=np.array([100.0]), intensity=np.array([1.0]), spectrum_type=SpectrumType.CENTROID, ms_level=2)
    entry = LibraryEntry(spec, key=1)
    assert read_mzspeclib(write_mzspeclib(entry, tmp_path / "a.txt"))[0] == entry


def test_keys_filled_by_position(tmp_path: Path) -> None:
    spec = make_spectrum()
    library = read_mzspeclib(write_mzspeclib([LibraryEntry(spec), LibraryEntry(spec)], tmp_path / "a.txt"))
    assert [entry.key for entry in library] == [1, 2]


def test_selected_ion_mz_and_minutes(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text(
        "<mzSpecLib>\nMS:1003186|library format version=1.0\n<Spectrum=4>\n"
        "MS:1000744|selected ion m/z=500.5\nMS:1000041|charge state=3\n"
        "[1]MS:1000894|retention time=2.5\n[1]UO:0000000|unit=UO:0000031|minute\n"
        "MS:1000044|dissociation method=MS:1000598|electron transfer dissociation\n"
        "MS:1000044|dissociation method=MS:1002678|supplemental beam-type collision-induced dissociation\n"
        "<Peaks>\n100\t5\n200\t6\n"
    )
    (entry,) = read_mzspeclib(path)
    assert entry.key == 4
    assert entry.spectrum.rt == 150.0
    assert entry.spectrum.activation_type == ActivationType.ETHCD
    (prec,) = entry.spectrum.precursors or []
    assert (prec.precursor_mz, prec.charge, prec.is_monoisotopic) == (500.5, 3, None)
    assert entry.attributes == ()


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------


def test_annotations_are_paf_objects(tmp_path: Path) -> None:
    entry = read_mzspeclib(write_mzspeclib(make_entry(), tmp_path / "a.txt"))[0]
    assert entry.peak_annotations is not None
    assert entry.peak_annotations[1] == ()
    assert [a.serialize() for a in entry.peak_annotations[2]] == ["b3/-1.1ppm", "y2-H2O^2/0.5ppm"]
    assert all(isinstance(a, paf.PafAnnotation) for peak in entry.peak_annotations for a in peak)


def test_matched_fragments_as_annotations(tmp_path: Path) -> None:
    frags = list(pt.fragment("PEPTIDEK", ion_types=("b", "y"), charges=(1,)))
    spec = MsnSpectrum(
        mz=np.array(sorted(f.mz for f in frags[:4])),
        intensity=np.array([1.0, 2.0, 3.0, 4.0]),
        spectrum_type=SpectrumType.CENTROID,
        ms_level=2,
    )
    matches = match_fragments(spec, frags, tolerance=10, tolerance_type="ppm")
    entry = LibraryEntry.from_spectrum(spec, "PEPTIDEK/1", key=1, peak_annotations=matches)
    assert entry.peak_annotations is not None
    assert all(entry.peak_annotations)
    assert read_mzspeclib(write_mzspeclib(entry, tmp_path / "a.txt"))[0] == entry


def test_invalid_mzpaf_raises(tmp_path: Path) -> None:
    with pytest.raises(SpxtacularError, match="mzPAF"):
        make_entry(peak_annotations=["y1", "not an annotation!!", None, None])
    path = tmp_path / "a.txt"
    path.write_text("<mzSpecLib>\nMS:1003186|library format version=1.0\n<Spectrum=1>\n<Peaks>\n100\t5\t4498 4498\n")
    with pytest.raises(SpxtacularError, match=r"a\.txt:5"):
        read_mzspeclib(path)


def test_annotation_count_mismatch_raises() -> None:
    with pytest.raises(SpxtacularError, match="3 annotations for 4 peaks"):
        make_entry(peak_annotations=["y1", None, None])


# ---------------------------------------------------------------------------
# Real example and attribute sets
# ---------------------------------------------------------------------------


def test_reads_spec_example() -> None:
    library = read_mzspeclib(DIANN)
    assert len(library) == 9
    first = library[0]
    assert first.key == 1
    assert first.name == "AAAAAAAAAAAAAAAASAGGK2"
    assert first.peptidoform == pt.parse("AAAAAAAAAAAAAAAASAGGK/2")
    assert first.charge == 2
    (prec,) = first.spectrum.precursors or []
    assert prec.precursor_mz == 778.41296
    assert prec.im_type == IMType.DRIFT_TIME_MS
    assert len(first.spectrum.mz) == 20
    assert first.peak_annotations is not None
    assert first.peak_annotations[0][0].serialize() == "b6/0"
    assert CvParam("MS:1003207", "library creation software", "DIA-NN", "MS:1003253") in library.attributes
    assert all(entry.peptidoform is not None and entry.charge for entry in library)


def test_spec_example_round_trips(tmp_path: Path) -> None:
    library = read_mzspeclib(DIANN)
    assert read_mzspeclib(write_mzspeclib(library, tmp_path / "a.txt")) == library
    assert read_mzspeclib(write_mzspeclib(library, tmp_path / "a.json")) == library


def test_attribute_sets_resolve(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text(
        "<mzSpecLib>\nMS:1003186|library format version=1.0\n"
        "<AttributeSet Spectrum=all>\nMS:1000044|dissociation method=MS:1000133|collision-induced dissociation\n"
        "<AttributeSet Spectrum=hot>\n[1]MS:1000045|collision energy=35\n[1]UO:0000000|unit=UO:0000266|electronvolt\n"
        "<Spectrum=1>\nMS:1003212|library attribute set name=hot\n<Peaks>\n100\t5\n"
        "<Spectrum=2>\nMS:1000044|dissociation method=MS:1000422|beam-type collision-induced dissociation\n"
        "<Peaks>\n100\t5\n"
    )
    first, second = read_mzspeclib(path)
    assert first.spectrum.activation_type == ActivationType.CID
    assert first.spectrum.collision_energy == 35.0
    assert second.spectrum.activation_type == ActivationType.HCD
    assert second.spectrum.collision_energy is None


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("<Spectrum=1>\n", "not an mzSpecLib"),
        ("<mzSpecLib>\nMS:1003186|library format version=2.0\n", "unsupported mzSpecLib format version"),
        ("<mzSpecLib>\n<Spectrum=1>\n<Peaks>\n", "library format version"),
        ("<mzSpecLib>\nMS:1003186|library format version=1.0\n<Spectrum=1>\n<Peaks>\nabc\tdef\n", "numbers"),
        ("<mzSpecLib>\nMS:1003186|library format version=1.0\n<Spectrum=1>\nnot an attribute\n", "ACCESSION"),
        ("<mzSpecLib>\nMS:1003186|library format version=1.0\n<Peaks>\n", "outside a <Spectrum>"),
        (
            "<mzSpecLib>\nMS:1003186|library format version=1.0\n<Spectrum=1>\nMS:1003059|number of peaks=3\n"
            "<Peaks>\n100\t1\n",
            "declares 3 peaks",
        ),
        (
            "<mzSpecLib>\nMS:1003186|library format version=1.0\n<Spectrum=1>\n<Peaks>\n1\t1\n<Spectrum=1>\n<Peaks>\n",
            "duplicate",
        ),
        (
            "<mzSpecLib>\nMS:1003186|library format version=1.0\n<Spectrum=1>\n"
            "MS:1003212|library attribute set name=missing\n<Peaks>\n",
            "unknown attribute set",
        ),
        (
            "<mzSpecLib>\nMS:1003186|library format version=1.0\n<Spectrum=1>\n<Analyte=1>\n"
            "MS:1003270|proforma peptidoform ion notation=PEP[[/2\n<Peaks>\n",
            "ProForma",
        ),
        ('{"format_version": "1.0", "spectra": [{"mzs": [1.0], "intensities": []}]}', "equal length"),
        ('{"spectra": []}', "format_version"),
        ("{not json", "invalid JSON"),
    ],
)
def test_invalid_files_raise(tmp_path: Path, content: str, match: str) -> None:
    path = tmp_path / "bad.mzspeclib"
    path.write_text(content)
    with pytest.raises(SpxtacularError, match=match):
        read_mzspeclib(path)


def test_invalid_entries_raise(tmp_path: Path) -> None:
    two = make_spectrum(precursors=[Precursor(precursor_mz=1.0), Precursor(precursor_mz=2.0)])
    with pytest.raises(SpxtacularError, match="one precursor"):
        write_mzspeclib(LibraryEntry(two), tmp_path / "a.txt")
    with pytest.raises(SpxtacularError, match="duplicate library spectrum key"):
        write_mzspeclib([make_entry(), make_entry()], tmp_path / "a.txt")
    with pytest.raises(SpxtacularError, match="profile"):
        write_mzspeclib(LibraryEntry(make_spectrum(spectrum_type=SpectrumType.PROFILE)), tmp_path / "a.txt")
    with pytest.raises(SpxtacularError, match="duplicates a field"):
        write_mzspeclib(make_entry(attributes=(CvParam("MS:1000041", "charge state", 2),)), tmp_path / "a.txt")
    generic_im = make_spectrum(precursors=[Precursor(precursor_mz=1.0, im=3.0, im_type=IMType.IM)])
    with pytest.raises(SpxtacularError, match="im_type"):
        write_mzspeclib(LibraryEntry(generic_im), tmp_path / "a.txt")
    with pytest.raises(SpxtacularError, match="format"):
        write_mzspeclib([make_entry()], tmp_path / "a.txt", format="xml")  # type: ignore[arg-type]
    with pytest.raises(SpxtacularError, match="conflicts"):
        Analyte(peptidoform="PEPTIDE/2", charge=3)
    with pytest.raises(SpxtacularError, match="accession"):
        CvParam("bad accession", "x")


def test_gzip_detected_by_magic(tmp_path: Path) -> None:
    path = tmp_path / "noext"
    path.write_bytes(gzip.compress(DIANN.read_bytes()))
    assert len(read_mzspeclib(path)) == 9


# ---------------------------------------------------------------------------
# MSP / MGF annotations=
# ---------------------------------------------------------------------------


def test_peak_list_default_output_unchanged(tmp_path: Path) -> None:
    spec = make_spectrum()
    for writer, suffix in ((write_mgf, "mgf"), (write_msp, "msp")):
        plain = writer([spec], tmp_path / f"plain.{suffix}").read_text()
        explicit = writer([spec], tmp_path / f"none.{suffix}", annotations=None).read_text()
        assert plain == explicit
        assert '"' not in plain


@pytest.mark.parametrize(("writer", "reader", "suffix"), [(write_msp, MspReader, "msp"), (write_mgf, MgfReader, "mgf")])
def test_peak_list_annotations(tmp_path: Path, writer, reader, suffix: str) -> None:  # noqa: ANN001
    spec = make_spectrum()
    anns = [["y1/0.3ppm", None, [paf.parse("b3"), "y2"], ""]]
    path = writer([spec], tmp_path / f"a.{suffix}", annotations=anns)
    text = path.read_text()
    assert '\t"y1/0.3ppm"' in text or ' "y1/0.3ppm"' in text
    assert '"b3,y2"' in text
    assert text.count('"') == 4
    with reader(path) as r:
        (back,) = list(r)
    np.testing.assert_allclose(back.mz, spec.mz)
    np.testing.assert_allclose(back.intensity, spec.intensity)


def test_peak_list_annotations_errors(tmp_path: Path) -> None:
    spec = make_spectrum()
    with pytest.raises(SpxtacularError, match="fewer|more|spectra|entries"):
        write_msp([spec, spec], tmp_path / "a.msp", annotations=[[None] * 4])
    with pytest.raises(SpxtacularError, match="more|spectra|entries"):
        write_mgf([spec], tmp_path / "a.mgf", annotations=[[None] * 4, [None] * 4])
    with pytest.raises(SpxtacularError, match="annotations for 4 peaks"):
        write_msp([spec], tmp_path / "a.msp", annotations=[["y1"]])
    with pytest.raises(SpxtacularError, match="contains"):
        write_mgf([spec], tmp_path / "a.mgf", annotations=[['y1"', None, None, None]])
    with pytest.raises(SpxtacularError, match="string"):
        write_msp([spec], tmp_path / "a.msp", annotations="y1")


def test_peak_list_matched_fragments(tmp_path: Path) -> None:
    frags = list(pt.fragment("PEPTIDEK", ion_types=("b", "y"), charges=(1,)))
    spec = MsnSpectrum(
        mz=np.array(sorted(f.mz for f in frags[:3])),
        intensity=np.array([1.0, 2.0, 3.0]),
        spectrum_type=SpectrumType.CENTROID,
        ms_level=2,
    )
    matches = match_fragments(spec, frags, tolerance=10, tolerance_type="ppm")
    text = write_msp([spec], tmp_path / "a.msp", annotations=[matches]).read_text()
    assert text.count('"') == 6
