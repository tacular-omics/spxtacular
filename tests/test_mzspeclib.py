"""Tests for mzSpecLib reading/writing and mzPAF annotations in MSP/MGF output."""

from __future__ import annotations

import gzip
import json
import random
import time
import tracemalloc
from pathlib import Path
from typing import IO

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
    MzSpecLibReader,
    Precursor,
    SpectralLibrary,
    SpectrumType,
    SpxtacularError,
    match_fragments,
    mzspeclib,
    read_mzspeclib,
    write_mgf,
    write_msp,
    write_mzspeclib,
)

DATA = Path(__file__).parent / "data" / "mzspeclib"
DIANN = DATA / "phl004_canonical_sall_pv_plasma.head.diann.mzSpecLib.txt"
SPECTRAST_TEXT = DATA / "fetal_brain_tiny.mzSpecLib.txt.gz"
SPECTRAST_JSON = DATA / "fetal_brain_tiny.mzSpecLib.json.gz"


def make_spectrum(**overrides: object) -> MsnSpectrum:
    values: dict[str, object] = {
        "mz": np.array([175.119, 263.087, 376.171, 505.214]),
        "intensity": np.array([100.0, 55.5, 20.25, 7.0]),
        "spectrum_type": SpectrumType.CENTROID,
        "ms_level": 2,
        "scan_number": 17,
        "native_id": "controllerType=0 controllerNumber=1 scan=17",
        "rt": 1234.5,
        "polarity": "positive",
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
        "peak_annotations": ["y1/0.3ppm", None, "b3/-1.1ppm,y2-H2O^2/0.5ppm", "?^2"],
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
    assert spec.polarity == "positive"
    assert spec.activation_type == ActivationType.HCD
    assert spec.collision_energy == 27.0
    assert spec.scan_number == 17
    assert spec.native_id == "controllerType=0 controllerNumber=1 scan=17"
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
    # One comma-joined mzPAF string per peak, "?" when unannotated (mzspeclib-py's form).
    assert spectrum["peak_annotations"] == ["y1/0.3ppm", "?", "b3/-1.1ppm,y2-H2O^2/0.5ppm", "?^2"]
    assert {
        "accession": "MS:1000767",
        "name": "native spectrum identifier",
        "value": "controllerType=0 controllerNumber=1 scan=17",
    } in spectrum["attributes"]
    unannotated = json.loads(write_mzspeclib([make_entry(peak_annotations=None)], tmp_path / "b.json").read_text())
    assert unannotated["spectra"][0]["peak_annotations"] == ["?"] * 4
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


def test_bare_question_mark_is_unannotated(tmp_path: Path) -> None:
    """A bare "?" marks an unannotated peak in both forms; "?" with more is kept."""
    doc = {
        "format_version": "1.0",
        "attributes": [],
        "spectra": [
            {
                "attributes": [{"accession": "MS:1003237", "name": "library spectrum key", "value": 1}],
                "mzs": [100.0, 200.0, 300.0],
                "intensities": [1.0, 2.0, 3.0],
                "peak_annotations": ["?", ["b2", "?"], "?^2"],
            }
        ],
    }
    path = tmp_path / "a.json"
    path.write_text(json.dumps(doc))
    (entry,) = read_mzspeclib(path)
    assert entry.peak_annotations is not None
    assert [[a.serialize() for a in peak] for peak in entry.peak_annotations] == [[], ["b2"], ["?^2"]]
    text = tmp_path / "a.txt"
    text.write_text(
        "<mzSpecLib>\nMS:1003186|library format version=1.0\n<Spectrum=1>\n<Peaks>\n100\t1\t?\n200\t2\tb2\n"
    )
    (entry,) = read_mzspeclib(text)
    assert entry.peak_annotations is not None
    assert entry.peak_annotations[0] == ()


def test_matched_fragments_as_annotations(tmp_path: Path) -> None:
    frags = list(pt.fragment("PEPTIDEK", ion_types=("b", "y"), charges=(1,)))
    spec = MsnSpectrum(
        mz=np.array(sorted(f.mz for f in frags[:4])),
        intensity=np.array([1.0, 2.0, 3.0, 4.0]),
        spectrum_type=SpectrumType.CENTROID,
        ms_level=2,
    )
    matches = match_fragments(spec, frags, tolerance=10, tolerance_unit="ppm")
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


@pytest.mark.parametrize("source", [SPECTRAST_TEXT, SPECTRAST_JSON], ids=["text", "json"])
@pytest.mark.parametrize("suffix", ["txt", "json"])
def test_spectrast_example_round_trips(tmp_path: Path, source: Path, suffix: str) -> None:
    library = read_mzspeclib(source)
    assert len(library) == 21
    assert read_mzspeclib(write_mzspeclib(library, tmp_path / f"out.{suffix}")) == library


def test_spectrast_text_and_json_agree() -> None:
    # The upstream pair is not attribute-for-attribute identical (the JSON repeats
    # a collision-energy group), so compare what both forms carry.
    text, as_json = read_mzspeclib(SPECTRAST_TEXT), read_mzspeclib(SPECTRAST_JSON)
    assert text.attributes == as_json.attributes
    assert len(text) == len(as_json)
    for a, b in zip(text, as_json, strict=True):
        assert a.key == b.key
        assert a.spectrum == b.spectrum
        assert a.analytes == b.analytes
        assert a.interpretations == b.interpretations
        assert a.peak_annotations == b.peak_annotations
    assert all(entry.peak_annotations for entry in text)


def test_text_cluster_key_attribute(tmp_path: Path) -> None:
    """A <Cluster=N> section may repeat its key as MS:1003267; it must not be written twice to JSON."""
    path = tmp_path / "a.txt"
    path.write_text(
        "<mzSpecLib>\nMS:1003186|library format version=1.0\n<Cluster=1>\n"
        "MS:1003267|spectrum cluster key=1\nMS:1003268|spectrum cluster member spectrum keys=1\n"
    )
    library = read_mzspeclib(path)
    assert [a.accession for a in library.clusters[1]] == ["MS:1003268"]
    assert read_mzspeclib(write_mzspeclib(library, tmp_path / "a.json")) == library
    path.write_text(
        "<mzSpecLib>\nMS:1003186|library format version=1.0\n<Cluster=1>\nMS:1003267|spectrum cluster key=2\n"
    )
    with pytest.raises(SpxtacularError, match="does not match the section key"):
        read_mzspeclib(path)


def test_native_id_round_trips(tmp_path: Path) -> None:
    entry = make_entry()
    text = write_mzspeclib([entry], tmp_path / "a.txt").read_text()
    assert "MS:1000767|native spectrum identifier=controllerType=0 controllerNumber=1 scan=17\n" in text
    for name in ("a.txt", "a.json"):
        back = read_mzspeclib(write_mzspeclib([entry], tmp_path / name))[0]
        assert back.spectrum.native_id == "controllerType=0 controllerNumber=1 scan=17"
        assert back.attributes == entry.attributes


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
    with pytest.raises(SpxtacularError, match=match):
        list(MzSpecLibReader(path))


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
# Streaming (MzSpecLibReader)
# ---------------------------------------------------------------------------


def _library_files(tmp_path: Path) -> list[Path]:
    files = [DIANN, SPECTRAST_TEXT, SPECTRAST_JSON]
    library = read_mzspeclib(DIANN)
    files += [write_mzspeclib(library, tmp_path / name) for name in ("a.txt", "a.json", "a.txt.gz", "a.json.gz")]
    return files


def test_streamed_equals_whole_read(tmp_path: Path) -> None:
    for path in _library_files(tmp_path):
        whole = read_mzspeclib(path)
        with MzSpecLibReader(path) as reader:
            assert reader.attributes == whole.attributes
            assert list(reader) == whole.entries
            assert reader.clusters == whole.clusters
        assert reader.format == ("json" if "json" in path.name else "text")


def test_json_streams_across_chunk_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    whole = read_mzspeclib(SPECTRAST_JSON)
    monkeypatch.setattr(mzspeclib, "_JSON_CHUNK", 7)  # splits numbers, strings and keys
    assert list(MzSpecLibReader(SPECTRAST_JSON)) == whole.entries


def test_header_is_read_without_peaks(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text(
        "<mzSpecLib>\nMS:1003186|library format version=1.0\nMS:1003188|library name=demo\n"
        "<Spectrum=1>\n<Peaks>\nnot a peak\n"
    )
    reader = MzSpecLibReader(path)
    reader.open()
    assert reader.attributes == (CvParam("MS:1003188", "library name", "demo"),)
    with pytest.raises(SpxtacularError, match=r"a\.txt:6: .*numbers"):
        next(iter(reader))


def test_json_sets_after_spectra(tmp_path: Path) -> None:
    """Upstream JSON sorts its keys, so attribute sets can follow the spectra."""
    path = tmp_path / "a.json"
    spectrum = {
        "attributes": [
            {"accession": "MS:1003237", "name": "library spectrum key", "value": 1},
            {"accession": "MS:1003212", "name": "library attribute set name", "value": "hot"},
        ],
        "mzs": [100.0],
        "intensities": [5.0],
    }
    energy = {"accession": "MS:1000045", "name": "collision energy", "value": 35}
    path.write_text(
        json.dumps(
            {
                "attributes": [],
                "format_version": "1.0",
                "spectra": [spectrum],
                "spectrum_attribute_sets": {"hot": [energy]},
            }
        )
    )
    with MzSpecLibReader(path) as reader:
        (entry,) = reader
    assert entry.spectrum.collision_energy == 35.0


def test_errors_raise_at_the_bad_spectrum(tmp_path: Path) -> None:
    library = read_mzspeclib(DIANN)
    text = write_mzspeclib(library, tmp_path / "a.txt").read_text().replace("<Spectrum=3>\n", "<Spectrum=3>\nbad\n")
    (tmp_path / "a.txt").write_text(text)
    iterator = iter(MzSpecLibReader(tmp_path / "a.txt"))
    assert [entry.key for entry in (next(iterator), next(iterator))] == [1, 2]
    with pytest.raises(SpxtacularError, match="ACCESSION"):
        next(iterator)

    document = json.loads(write_mzspeclib(library, tmp_path / "a.json").read_text())
    document["spectra"][2]["mzs"] = []
    (tmp_path / "a.json").write_text(json.dumps(document))
    iterator = iter(MzSpecLibReader(tmp_path / "a.json"))
    assert [entry.key for entry in (next(iterator), next(iterator))] == [1, 2]
    with pytest.raises(SpxtacularError, match="spectrum 2: 'mzs'"):
        next(iterator)


@pytest.mark.parametrize("name", ["a.txt", "a.json.gz"])
def test_early_break_closes_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    path = write_mzspeclib(read_mzspeclib(DIANN), tmp_path / name)
    opened: list[IO[str]] = []
    real_open = mzspeclib._open_library_text

    def recording_open(source: Path) -> IO[str]:
        handle = real_open(source)
        opened.append(handle)
        return handle

    monkeypatch.setattr(mzspeclib, "_open_library_text", recording_open)
    with MzSpecLibReader(path) as reader:
        for _ in reader:
            break
        assert opened and all(handle.closed for handle in opened)
        held = iter(reader)
        next(held)
        assert not opened[-1].closed
    assert opened[-1].closed  # close() ends unfinished iterations


def test_clusters_before_iterating(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text(
        "<mzSpecLib>\nMS:1003186|library format version=1.0\n<Spectrum=1>\n<Peaks>\n100\t5\n"
        "<Cluster=1>\nMS:1003268|spectrum cluster member spectrum keys=1\n"
    )
    reader = MzSpecLibReader(path)
    assert reader.clusters == {1: (CvParam("MS:1003268", "spectrum cluster member spectrum keys", (1,)),)}


def test_reader_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        MzSpecLibReader(tmp_path / "missing.txt").open()
    path = tmp_path / "latin1.txt"
    path.write_bytes(b"<mzSpecLib>\nMS:1003186|library format version=1.0\nMS:1003188|library name=caf\xe9\n")
    with pytest.raises(SpxtacularError, match="not UTF-8"):
        MzSpecLibReader(path).open()


def test_seen_keys() -> None:
    seen = mzspeclib._SeenKeys()
    for key in (1, 2, 3, 10, 5, 4, 9, 7, 6, 8, 0, 11, 12):
        assert seen.add(key)
    assert (seen.starts, seen.ends, seen.others) == ([1, 10], [3, 12], {0, 4, 5, 6, 7, 8, 9})
    assert not any(seen.add(key) for key in range(13))
    assert seen.add(13) and seen.add(20) and not seen.add(20) and not seen.add(1)


def test_seen_keys_scattered_is_fast() -> None:
    """Out-of-order keys with gaps (a filtered library re-sorted by m/z) must not go quadratic."""
    keys = list(range(0, 400_000, 2))
    random.Random(0).shuffle(keys)
    seen = mzspeclib._SeenKeys()
    start = time.perf_counter()
    assert all(seen.add(key) for key in keys)
    assert time.perf_counter() - start < 2
    assert not any(seen.add(key) for key in keys[:1000])
    assert all(seen.add(key + 1) for key in keys[:1000])


def _json_version_library(version: str, pad: int = 1) -> str:
    """A JSON library whose top level holds ``"format_version": <version>`` as a bare number."""
    spectrum = {
        "attributes": [{"accession": "MS:1003237", "name": "library spectrum key", "value": 1}],
        "mzs": [100.0],
        "intensities": [5.0],
    }
    name = {"accession": "MS:1003188", "name": "library name", "value": "x" * pad}
    head = json.dumps({"attributes": [name]})[:-1] + ', "format_version": '
    return head + version + ', "spectra": ' + json.dumps([spectrum]) + "}"


def test_json_number_cut_at_chunk_boundary(tmp_path: Path) -> None:
    """``raw_decode`` of "...1." returns 1; the reader must read on and get 1.0."""
    probe = _json_version_library("1.0")
    pad = mzspeclib._JSON_CHUNK - (probe.index("1.0") + 2) + 1
    text = _json_version_library("1.0", pad)
    assert text[: mzspeclib._JSON_CHUNK].endswith("1.")
    path = tmp_path / "a.json"
    path.write_text(text)
    with MzSpecLibReader(path) as reader:
        assert [entry.key for entry in reader] == [1]
    assert read_mzspeclib(path).attributes == reader.attributes


@pytest.mark.parametrize("version", ["1.0", "1e0", "1.0e+0", "10e-1"])
def test_json_numbers_at_chunk_size_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    path = tmp_path / "a.json"
    path.write_text(_json_version_library(version))
    monkeypatch.setattr(mzspeclib, "_JSON_CHUNK", 1)
    with MzSpecLibReader(path) as reader:
        assert [entry.key for entry in reader] == [1]


def test_invalid_json_on_a_valid_file_is_an_internal_error(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    path.write_text(_json_version_library("1.0"))
    assert "internal error" in str(mzspeclib._invalid_json(path))


@pytest.mark.parametrize("prefix", ["", "library_"])
def test_json_header_stops_at_spectra_without_cluster_sets(tmp_path: Path, prefix: str) -> None:
    """Entries need no cluster sets, so ``open()`` does not scan the spectra for them."""
    path = tmp_path / "a.json"
    sets = "".join(f'"{prefix}{kind}_attribute_sets": {{}}, ' for kind in ("spectrum", "analyte", "interpretation"))
    path.write_text('{"format_version": "1.0", "attributes": [], ' + sets + '"spectra": [ not json')
    reader = MzSpecLibReader(path)
    reader.open()
    assert reader.attributes == ()
    with pytest.raises(SpxtacularError, match="invalid JSON"):
        list(reader)


def test_json_cluster_sets_after_spectra(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    members = {"accession": "MS:1003268", "name": "spectrum cluster member spectrum keys", "value": [1]}
    cluster = {
        "attributes": [
            {"accession": "MS:1003267", "name": "spectrum cluster key", "value": 1},
            {"accession": "MS:1003212", "name": "library attribute set name", "value": "shared"},
        ]
    }
    document = json.loads(_json_version_library('"1.0"'))
    document |= {
        "spectrum_attribute_sets": {},
        "analyte_attribute_sets": {},
        "interpretation_attribute_sets": {},
        "clusters": [cluster],
    }
    path.write_text(
        json.dumps(document)[:-1] + ', "cluster_attribute_sets": {"shared": [' + json.dumps(members) + "]}}"
    )
    with MzSpecLibReader(path) as reader:
        assert [entry.key for entry in reader] == [1]
        assert reader.clusters == {1: (CvParam("MS:1003268", "spectrum cluster member spectrum keys", (1,)),)}
    assert read_mzspeclib(path).clusters == reader.clusters


def test_json_duplicate_spectra_member_raises(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    text = _json_version_library("1.0")
    path.write_text(text[:-1] + ', "spectra": []}')
    with pytest.raises(SpxtacularError, match="more than one 'spectra'"):
        read_mzspeclib(path)


def _text_library(n_spectra: int) -> str:
    spectrum = (
        "<Spectrum={key}>\nMS:1003061|library spectrum name=PEPTIDEK/2\nMS:1003208|experimental precursor "
        "monoisotopic m/z=467.2\nMS:1000041|charge state=2\n<Analyte=1>\nMS:1003270|proforma peptidoform ion "
        "notation=PEPTIDEK/2\n<Peaks>\n147.1128\t100\ty1/0.1ppm\n244.1656\t50\n341.2183\t25\n"
    )
    header = "<mzSpecLib>\nMS:1003186|library format version=1.0\n"
    return header + "".join(spectrum.format(key=key) for key in range(1, n_spectra + 1))


@pytest.mark.slow  # streams 1100 spectra under tracemalloc, ~2 s each
@pytest.mark.parametrize("suffix", ["txt", "json"])
def test_streaming_memory_is_flat(tmp_path: Path, suffix: str) -> None:
    """Peak memory while streaming does not grow with the number of spectra (informal)."""
    peaks = []
    for n_spectra in (100, 1000):
        path = tmp_path / f"{n_spectra}.txt"
        path.write_text(_text_library(n_spectra))
        if suffix == "json":
            path = write_mzspeclib(MzSpecLibReader(path), tmp_path / f"{n_spectra}.json")
        tracemalloc.start()
        try:
            count = sum(1 for _ in MzSpecLibReader(path))
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
        assert count == n_spectra
    assert peaks[1] < 2 * peaks[0], peaks


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


@pytest.mark.parametrize(("writer", "reader", "suffix"), [(write_msp, MspReader, "msp"), (write_mgf, MgfReader, "mgf")])
def test_peak_list_annotation_with_equals_sign(tmp_path: Path, writer, reader, suffix: str) -> None:  # noqa: ANN001
    """An mzPAF SMILES holds "="; the MGF reader must still see an ion line, not a header."""
    spec = make_spectrum(native_id='title "with quotes" = x')
    anns = [["s{CC(=O)O}/0.1", None, "b3", "s{C=C}"]]
    path = writer([spec], tmp_path / f"a.{suffix}", annotations=anns)
    assert "s{CC(=O)O}/0.1" in path.read_text()
    with reader(path) as r:
        (back,) = list(r)
    np.testing.assert_allclose(back.mz, spec.mz)
    np.testing.assert_allclose(back.intensity, spec.intensity)
    if suffix == "mgf":
        assert back.native_id == 'title "with quotes" = x'


def test_peak_list_annotations_errors(tmp_path: Path) -> None:
    spec = make_spectrum()
    with pytest.raises(SpxtacularError, match=r"fewer|more|spectra|entries"):
        write_msp([spec, spec], tmp_path / "a.msp", annotations=[[None] * 4])
    with pytest.raises(SpxtacularError, match=r"more|spectra|entries"):
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
    matches = match_fragments(spec, frags, tolerance=10, tolerance_unit="ppm")
    text = write_msp([spec], tmp_path / "a.msp", annotations=[matches]).read_text()
    assert text.count('"') == 6
