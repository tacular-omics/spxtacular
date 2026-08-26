"""Tests for the MGF / MS2 peak-list readers and writers (src/spxtacular/peaklist.py)."""

from __future__ import annotations

import gzip

import numpy as np
import pytest

from spxtacular import (
    MgfReader,
    Ms2Reader,
    MsnSpectrum,
    Precursor,
    Reader,
    Spectrum,
    SpectrumType,
    write_mgf,
    write_ms2,
)
from spxtacular.enums import Polarity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_spectrum(
    *,
    scan: int = 1,
    mz: list[float] | None = None,
    intensity: list[float] | None = None,
    precursor_mz: float | None = 445.1234567,
    precursor_intensity: float = 0.0,
    charge: int | None = 2,
    rt: float | None = 601.5,
    native_id: str | None = None,
    polarity: Polarity | None = None,
) -> MsnSpectrum:
    """An MS2 spectrum with the metadata both peak-list formats can carry."""
    precursors = None
    if precursor_mz is not None:
        precursors = [
            Precursor(
                mz=precursor_mz,
                intensity=precursor_intensity,
                charge=charge,
                im=None,
                is_monoisotopic=None,
            )
        ]
    return MsnSpectrum(
        mz=np.array(mz if mz is not None else [110.0715, 200.123456789, 1004.5]),
        intensity=np.array(intensity if intensity is not None else [1234.5, 9.87e5, 42.0]),
        spectrum_type=SpectrumType.CENTROID,
        ms_level=2,
        scan_number=scan,
        native_id=native_id,
        rt=rt,
        polarity=polarity,
        precursors=precursors,
    )


def precursor(spec: MsnSpectrum) -> Precursor:
    """First precursor of a spectrum, asserting it has one (keeps the type checker happy)."""
    assert spec.precursors is not None
    return spec.precursors[0]


def write_text(path, text: str):
    """Write fixture content, dedenting the leading newline of a triple-quoted block."""
    path.write_text(text.lstrip("\n"), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_mgf_round_trip_preserves_peaks_and_metadata(tmp_path):
    spectra = [
        make_spectrum(scan=101, native_id="scan 101, run A"),
        make_spectrum(scan=102, precursor_mz=812.4001, charge=3, rt=612.25, mz=[300.5], intensity=[7.0]),
    ]
    path = write_mgf(spectra, tmp_path / "out.mgf")
    assert path == tmp_path / "out.mgf"

    read = list(MgfReader(path))
    assert len(read) == 2

    for original, restored in zip(spectra, read, strict=True):
        np.testing.assert_array_equal(restored.mz, original.mz)
        np.testing.assert_array_equal(restored.intensity, original.intensity)
        assert restored.ms_level == 2
        assert restored.spectrum_type == SpectrumType.CENTROID
        assert restored.scan_number == original.scan_number
        assert restored.rt == original.rt
        assert restored.precursors is not None
        assert precursor(restored).mz == precursor(original).mz
        assert precursor(restored).charge == precursor(original).charge

    # TITLE falls back to scan=N when the spectrum has no native_id.
    assert read[0].native_id == "scan 101, run A"
    assert read[1].native_id == "scan=102"


def test_mgf_round_trip_precursor_intensity(tmp_path):
    spec = make_spectrum(precursor_intensity=6.25e4)
    path = write_mgf([spec], tmp_path / "out.mgf")
    assert "PEPMASS=445.1234567 62500.0" in path.read_text()
    restored = next(iter(MgfReader(path)))
    assert precursor(restored).intensity == 6.25e4


def test_mgf_round_trip_per_peak_charges(tmp_path):
    spec = MsnSpectrum(
        mz=np.array([100.0, 200.0]),
        intensity=np.array([5.0, 6.0]),
        charge=np.array([1, 2], dtype=np.int32),
        spectrum_type=SpectrumType.CENTROID,
        ms_level=2,
    )
    path = write_mgf([spec], tmp_path / "out.mgf")
    assert "100.0 5.0 1+" in path.read_text()
    restored = next(iter(MgfReader(path)))
    np.testing.assert_array_equal(restored.charge, np.array([1, 2], dtype=np.int32))
    # An explicit CENTROID type is honoured — a charge array does not promote it.
    assert restored.spectrum_type == SpectrumType.CENTROID


def test_ms2_round_trip_preserves_peaks_and_metadata(tmp_path):
    spectra = [make_spectrum(scan=7), make_spectrum(scan=9, charge=1, mz=[500.0], intensity=[1.0])]
    path = write_ms2(spectra, tmp_path / "out.ms2")

    read = list(Ms2Reader(path))
    assert len(read) == 2
    for original, restored in zip(spectra, read, strict=True):
        np.testing.assert_array_equal(restored.mz, original.mz)
        np.testing.assert_array_equal(restored.intensity, original.intensity)
        assert restored.ms_level == 2
        assert restored.spectrum_type == SpectrumType.CENTROID
        assert restored.scan_number == original.scan_number
        assert restored.native_id == f"scan={original.scan_number}"
        # rt goes out as minutes, so it returns to within floating-point noise.
        assert restored.rt == pytest.approx(original.rt)
        assert precursor(restored).mz == precursor(original).mz
        assert precursor(restored).charge == precursor(original).charge


def test_ms2_round_trip_optional_info_values(tmp_path):
    spec = make_spectrum(precursor_intensity=1.5e5)
    spec.injection_time = 25.4
    spec.total_ion_current = 9.9e6
    spec.activation_type = "HCD"
    path = write_ms2([spec], tmp_path / "out.ms2")

    restored = next(iter(Ms2Reader(path)))
    assert restored.injection_time == 25.4
    assert restored.total_ion_current == 9.9e6
    assert restored.activation_type == "HCD"
    assert precursor(restored).intensity == 1.5e5


@pytest.mark.parametrize(
    ("writer", "reader_cls", "name"), [(write_mgf, MgfReader, "out.mgf.gz"), (write_ms2, Ms2Reader, "out.ms2.gz")]
)
def test_gzip_round_trip(tmp_path, writer, reader_cls, name):
    spectra = [make_spectrum(scan=3), make_spectrum(scan=4)]
    path = writer(spectra, tmp_path / name)

    # Really gzipped, not just named that way.
    assert path.read_bytes()[:2] == b"\x1f\x8b"
    with gzip.open(path, "rt") as fh:
        assert fh.read()

    read = list(reader_cls(path))
    assert len(read) == 2
    np.testing.assert_array_equal(read[0].mz, spectra[0].mz)
    assert read[1].scan_number == 4


def test_gzip_detected_by_magic_bytes_not_suffix(tmp_path):
    plain = write_mgf([make_spectrum()], tmp_path / "plain.mgf")
    disguised = tmp_path / "disguised.mgf"
    with gzip.open(disguised, "wt") as fh:
        fh.write(plain.read_text())

    assert len(list(MgfReader(disguised))) == 1


def test_single_spectrum_accepted_without_a_list(tmp_path):
    path = write_mgf(make_spectrum(), tmp_path / "one.mgf")
    assert len(list(MgfReader(path))) == 1


def test_plain_spectrum_writes_without_metadata(tmp_path):
    spec = Spectrum(mz=np.array([100.0]), intensity=np.array([1.0]))
    mgf = write_mgf([spec], tmp_path / "plain.mgf")
    restored = next(iter(MgfReader(mgf)))
    assert restored.precursors is None
    assert restored.scan_number is None
    np.testing.assert_array_equal(restored.mz, spec.mz)

    ms2 = write_ms2([spec], tmp_path / "plain.ms2")
    restored = next(iter(Ms2Reader(ms2)))
    # An S line needs both: the position stands in for the scan, 0.0 for the m/z.
    assert restored.scan_number == 1
    assert precursor(restored).mz == 0.0


# ---------------------------------------------------------------------------
# Lenient parsing
# ---------------------------------------------------------------------------


def test_mgf_lenient_parsing(tmp_path):
    path = write_text(
        tmp_path / "lenient.mgf",
        """
# a comment
; another one
! and another
/ and one more
SEARCH=MIS
COM=global header outside any block

BEGIN IONS
TITLE=weird title with = signs and spaces
PEPMASS=445.1234 8000.0
CHARGE=2+ and 3+
RTINMINUTES=10.5
SCANS=1024-1030
USER01=whatever the vendor felt like
NEWMOD=nonsense

110.0715 1234.5
200.5,900.0
END IONS
""",
    )
    (spec,) = list(MgfReader(path))

    assert spec.native_id == "weird title with = signs and spaces"
    assert precursor(spec).mz == 445.1234
    assert precursor(spec).intensity == 8000.0
    # Multi-charge values collapse to the first state.
    assert precursor(spec).charge == 2
    assert spec.rt == pytest.approx(630.0)  # RTINMINUTES converted to seconds
    assert spec.scan_number == 1024  # scan ranges collapse to the first scan
    np.testing.assert_array_equal(spec.mz, np.array([110.0715, 200.5]))
    np.testing.assert_array_equal(spec.intensity, np.array([1234.5, 900.0]))


def test_mgf_rtinseconds_wins_over_rtinminutes(tmp_path):
    path = write_text(
        tmp_path / "rt.mgf",
        """
BEGIN IONS
PEPMASS=400.0
RTINSECONDS=120.0
RTINMINUTES=99.0
100.0 1.0
END IONS
""",
    )
    (spec,) = list(MgfReader(path))
    assert spec.rt == 120.0


def test_mgf_comma_and_charge_column_variants(tmp_path):
    path = write_text(
        tmp_path / "variants.mgf",
        """
BEGIN IONS
PEPMASS=400.0
CHARGE=2+,3+
100.0 1.0 1+
200.0 2.0 +1
END IONS
""",
    )
    (spec,) = list(MgfReader(path))
    assert precursor(spec).charge == 2
    np.testing.assert_array_equal(spec.charge, np.array([1, 1], dtype=np.int32))


def test_ms2_lenient_parsing(tmp_path):
    path = write_text(
        tmp_path / "lenient.ms2",
        """
H	CreationDate	2020-01-01
H	Extractor	RawConverter
# comment line
S	1024	1024	445.1234
I	RTime	10.5
I	IonInjectionTime	13.5
I	TIC	123456.0
I	ActivationType	ETD
I	NumberOfPeaks
D	Charge	nonsense
Z	2	889.2394
Z	3	1333.35
110.0715 1234.5
200.5 900.0
""",
    )
    (spec,) = list(Ms2Reader(path))

    assert spec.scan_number == 1024
    assert precursor(spec).mz == 445.1234
    # Several Z lines: the first charge state wins, the rest do not crash the parse.
    assert precursor(spec).charge == 2
    assert spec.rt == pytest.approx(630.0)  # RTime is minutes
    assert spec.injection_time == 13.5
    assert spec.total_ion_current == 123456.0
    assert spec.activation_type == "ETD"
    np.testing.assert_array_equal(spec.mz, np.array([110.0715, 200.5]))


@pytest.mark.parametrize("value", ["F1:2478", "scan=2478", "controller:1 scan:2478"])
def test_mgf_scans_accepts_prefixed_identifiers(tmp_path, value):
    path = write_text(
        tmp_path / "prefixed-scan.mgf",
        f"BEGIN IONS\nSCANS={value}\n100 10\nEND IONS\n",
    )

    (spec,) = list(MgfReader(path))

    assert spec.scan_number == 2478


def test_ms2_rettime_alias(tmp_path):
    path = write_text(
        tmp_path / "rettime.ms2",
        """
S	1	1	400.0
I	RetTime	2.0
100.0 1.0
""",
    )
    (spec,) = list(Ms2Reader(path))
    assert spec.rt == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# Polarity / charge signs
# ---------------------------------------------------------------------------


def test_mgf_negative_charge_sign_sets_polarity(tmp_path):
    path = write_text(
        tmp_path / "neg.mgf",
        """
BEGIN IONS
PEPMASS=400.0
CHARGE=3-
100.0 1.0
END IONS
""",
    )
    (spec,) = list(MgfReader(path))
    assert precursor(spec).charge == -3
    assert spec.polarity == Polarity.NEGATIVE


def test_mgf_writes_negative_sign_from_polarity(tmp_path):
    spec = make_spectrum(charge=2, polarity=Polarity.NEGATIVE)
    path = write_mgf([spec], tmp_path / "neg.mgf")
    assert "CHARGE=2-" in path.read_text()

    restored = next(iter(MgfReader(path)))
    assert precursor(restored).charge == -2
    assert restored.polarity == Polarity.NEGATIVE


def test_mgf_writes_negative_sign_from_negative_charge(tmp_path):
    spec = make_spectrum(charge=-2)
    path = write_mgf([spec], tmp_path / "neg.mgf")
    assert "CHARGE=2-" in path.read_text()
    assert precursor(next(iter(MgfReader(path)))).charge == -2


def test_ms2_negative_charge_round_trip(tmp_path):
    spec = make_spectrum(charge=2, polarity=Polarity.NEGATIVE)
    path = write_ms2([spec], tmp_path / "neg.ms2")
    assert "\nZ\t-2\t" in path.read_text()

    restored = next(iter(Ms2Reader(path)))
    assert precursor(restored).charge == -2
    assert restored.polarity == Polarity.NEGATIVE


def test_positive_charge_has_positive_polarity(tmp_path):
    path = write_mgf([make_spectrum(charge=2)], tmp_path / "pos.mgf")
    assert "CHARGE=2+" in path.read_text()
    assert next(iter(MgfReader(path))).polarity == Polarity.POSITIVE


# ---------------------------------------------------------------------------
# Empty spectra
# ---------------------------------------------------------------------------


def test_mgf_empty_block_yields_empty_spectrum(tmp_path):
    path = write_text(
        tmp_path / "empty.mgf",
        """
BEGIN IONS
TITLE=nothing here
PEPMASS=400.0
END IONS
""",
    )
    (spec,) = list(MgfReader(path))
    assert len(spec) == 0
    assert spec.mz.dtype == np.float64
    assert spec.charge is None
    assert precursor(spec).mz == 400.0


def test_ms2_empty_scan_yields_empty_spectrum(tmp_path):
    path = write_text(
        tmp_path / "empty.ms2",
        """
S	1	1	400.0
Z	2	799.0
S	2	2	500.0
Z	2	999.0
100.0 1.0
""",
    )
    first, second = list(Ms2Reader(path))
    assert len(first) == 0
    assert precursor(first).mz == 400.0
    assert len(second) == 1


def test_empty_file_yields_no_spectra(tmp_path):
    mgf = write_text(tmp_path / "empty.mgf", "\n")
    ms2 = write_text(tmp_path / "empty.ms2", "\n")
    assert list(MgfReader(mgf)) == []
    assert list(Ms2Reader(ms2)) == []
    assert len(MgfReader(mgf)) == 0
    assert len(Ms2Reader(ms2)) == 0


def test_writing_an_empty_spectrum_round_trips(tmp_path):
    spec = make_spectrum(mz=[], intensity=[])
    mgf = write_mgf([spec], tmp_path / "e.mgf")
    ms2 = write_ms2([spec], tmp_path / "e.ms2")
    assert len(next(iter(MgfReader(mgf)))) == 0
    assert len(next(iter(Ms2Reader(ms2)))) == 0


# ---------------------------------------------------------------------------
# Malformed input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "line", "message"),
    [
        ("BEGIN IONS\nPEPMASS=400.0\n100.0 1.0\n", 1, "unterminated"),
        ("END IONS\n", 1, "without a matching"),
        ("BEGIN IONS\nPEPMASS=400.0\nBEGIN IONS\n", 3, "inside the spectrum block"),
        ("BEGIN IONS\nPEPMASS=400.0\n100.0 abc\nEND IONS\n", 3, "peak intensity"),
        ("BEGIN IONS\nPEPMASS=400.0\n100.0\nEND IONS\n", 3, "expected 'mz intensity'"),
        ("BEGIN IONS\nPEPMASS=not-a-number\n100.0 1.0\nEND IONS\n", 2, "PEPMASS m/z"),
        ("BEGIN IONS\nCHARGE=two\nPEPMASS=400.0\n100.0 1.0\nEND IONS\n", 2, "charge"),
        ("BEGIN IONS\nPEPMASS=400.0\n100.0 1.0 zz\nEND IONS\n", 3, "charge"),
    ],
)
def test_mgf_malformed_input_names_the_line(tmp_path, content, line, message):
    path = write_text(tmp_path / "bad.mgf", content)
    with pytest.raises(ValueError, match=message) as excinfo:
        list(MgfReader(path))
    assert f"bad.mgf:{line}:" in str(excinfo.value)


@pytest.mark.parametrize(
    ("content", "line", "message"),
    [
        ("100.0 1.0\n", 1, "before any 'S' scan line"),
        ("S\t1\t1\n", 1, "expected 'S <first_scan>"),
        ("S\t1\t1\tnope\n", 1, "precursor m/z"),
        ("S\t1\t1\t400.0\nZ\n", 2, "expected 'Z <charge> <mass>'"),
        ("S\t1\t1\t400.0\nZ\ttwo\t900.0\n", 2, "charge"),
        ("S\t1\t1\t400.0\n100.0 oops\n", 2, "peak intensity"),
        ("S\t1\t1\t400.0\nI\tRTime\tsoon\n", 2, "RTime"),
    ],
)
def test_ms2_malformed_input_names_the_line(tmp_path, content, line, message):
    path = write_text(tmp_path / "bad.ms2", content)
    with pytest.raises(ValueError, match=message) as excinfo:
        list(Ms2Reader(path))
    assert f"bad.ms2:{line}:" in str(excinfo.value)


def test_missing_file_raises_on_open(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such file"):
        MgfReader(tmp_path / "nope.mgf").open()
    with pytest.raises(FileNotFoundError):
        list(Ms2Reader(tmp_path / "nope.ms2"))


# ---------------------------------------------------------------------------
# Profile rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("writer", [write_mgf, write_ms2])
def test_profile_spectra_are_rejected(tmp_path, writer):
    profile = Spectrum(
        mz=np.array([100.0, 100.1]),
        intensity=np.array([1.0, 2.0]),
        spectrum_type=SpectrumType.PROFILE,
    )
    with pytest.raises(ValueError, match="peak lists hold centroid data"):
        writer([make_spectrum(), profile], tmp_path / "out.txt")


# ---------------------------------------------------------------------------
# Reader interface
# ---------------------------------------------------------------------------


def test_reader_interface_len_index_and_lookups(tmp_path):
    spectra = [make_spectrum(scan=1, native_id="first"), make_spectrum(scan=2, native_id="second")]
    path = write_mgf(spectra, tmp_path / "out.mgf")

    with MgfReader(path) as reader:
        assert len(reader) == 2
        assert reader[0].native_id == "first"
        assert reader["second"].scan_number == 2
        assert [s.scan_number for s in reader.ms2] == [1, 2]
        assert list(reader.ms1) == []  # peak lists carry no MS1
        assert reader.ms2[1].native_id == "second"
        assert next(iter(reader.ms2)).native_id == "first"

        with pytest.raises(IndexError, match="out of range"):
            reader[5]
        with pytest.raises(IndexError, match="negative indices"):
            reader[-1]
        with pytest.raises(KeyError, match="native_id"):
            reader["missing"]


def test_ms2_reader_len_counts_scans(tmp_path):
    path = write_ms2([make_spectrum(scan=1), make_spectrum(scan=2), make_spectrum(scan=3)], tmp_path / "out.ms2")
    with Ms2Reader(path) as reader:
        assert len(reader) == 3
        assert len(reader) == 3  # cached second call
        assert reader[2].scan_number == 3


def test_iteration_is_reentrant(tmp_path):
    path = write_mgf([make_spectrum(scan=1), make_spectrum(scan=2)], tmp_path / "out.mgf")
    reader = MgfReader(path)
    outer = [(a.scan_number, [b.scan_number for b in reader]) for a in reader]
    assert outer == [(1, [1, 2]), (2, [1, 2])]


# ---------------------------------------------------------------------------
# Reader auto-detect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "writer", "expected"),
    [
        ("run.mgf", write_mgf, MgfReader),
        ("run.mgf.gz", write_mgf, MgfReader),
        ("run.ms2", write_ms2, Ms2Reader),
        ("run.ms2.gz", write_ms2, Ms2Reader),
    ],
)
def test_reader_autodetects_peak_lists(tmp_path, name, writer, expected):
    path = writer([make_spectrum(scan=11)], tmp_path / name)
    with Reader(path) as reader:
        assert isinstance(reader._reader, expected)
        specs = list(reader.ms2)
        assert [s.scan_number for s in specs] == [11]
        assert list(reader.ms1) == []


def test_reader_autodetect_is_case_insensitive(tmp_path):
    path = write_mgf([make_spectrum()], tmp_path / "RUN.MGF")
    with Reader(path) as reader:
        assert isinstance(reader._reader, MgfReader)


def test_reader_rejects_unknown_extension(tmp_path):
    with pytest.raises(ValueError, match=r"\.mgf"):
        Reader(tmp_path / "run.txt")
