"""Tests for get_by_scan / get_by_native_id / get_by_sage_scannr on every reader."""

from __future__ import annotations

import gzip
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

import spxtacular.reader as _reader_module
import spxtacular.thermo as thermo_module
from spxtacular import MgfReader, Ms2Reader, MsnSpectrum, MspReader, Precursor, Reader, SpectrumType
from spxtacular.errors import SpxtacularError
from spxtacular.reader import DReader, MzmlReader

DATA_DIR = Path(__file__).parent / "data"


def write_text(path: Path, text: str) -> Path:
    path.write_text(text.lstrip("\n"), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Peak lists
# ---------------------------------------------------------------------------

MGF = """
SEARCH=MIS
BEGIN IONS
TITLE=run.19.19.2
PEPMASS=500.25
CHARGE=2+
SCANS=19
100.0 10.0
END IONS

BEGIN IONS
TITLE=run.20.20.3
PEPMASS=600.5
SCANS=20
200.0 20.0
201.0 21.0
END IONS
BEGIN IONS
TITLE=run.21.21.2
PEPMASS=700.75
SCANS=21
300.0 30.0
END IONS
"""


@pytest.fixture
def mgf_path(tmp_path: Path) -> Path:
    return write_text(tmp_path / "run.mgf", MGF)


def test_mgf_get_by_scan(mgf_path: Path) -> None:
    reader = MgfReader(mgf_path)
    spec = reader.get_by_scan(20)
    assert spec.scan_number == 20
    assert spec.native_id == "run.20.20.3"
    assert spec.mz.tolist() == [200.0, 201.0]
    assert reader.get_by_scan(21, ms_level=2).native_id == "run.21.21.2"
    assert reader.get_by_scan(19).mz.tolist() == [100.0]


def test_mgf_get_by_native_id(mgf_path: Path) -> None:
    reader = MgfReader(mgf_path)
    assert reader.get_by_native_id("run.21.21.2").scan_number == 21
    assert reader.get_by_native_id("run.19.19.2").mz.tolist() == [100.0]


def test_peak_list_missing_keys_raise_key_error(mgf_path: Path) -> None:
    reader = MgfReader(mgf_path)
    with pytest.raises(KeyError):
        reader.get_by_scan(99)
    with pytest.raises(KeyError):
        reader.get_by_scan(20, ms_level=1)
    with pytest.raises(KeyError):
        reader.get_by_native_id("nope")


def test_peak_list_lookup_matches_iteration(mgf_path: Path) -> None:
    reader = MgfReader(mgf_path)
    for spec in reader:
        assert spec.scan_number is not None and spec.native_id is not None
        by_scan = reader.get_by_scan(spec.scan_number)
        by_id = reader.get_by_native_id(spec.native_id)
        for other in (by_scan, by_id):
            np.testing.assert_array_equal(other.mz, spec.mz)
            assert other.precursors == spec.precursors


def test_peak_list_index_follows_file_changes(mgf_path: Path) -> None:
    reader = MgfReader(mgf_path)
    assert reader.get_by_scan(20).native_id == "run.20.20.3"
    write_text(mgf_path, MGF.replace("SCANS=20", "SCANS=42") + "\n")  # different size -> rebuilt
    with pytest.raises(KeyError):
        reader.get_by_scan(20)
    assert reader.get_by_scan(42).native_id == "run.20.20.3"


def test_peak_list_lookup_in_gzip_and_crlf(tmp_path: Path) -> None:
    path = tmp_path / "run.mgf.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="\r\n") as handle:
        handle.write(MGF.lstrip("\n"))
    reader = MgfReader(path)
    assert reader.get_by_scan(21).mz.tolist() == [300.0]
    assert reader.get_by_native_id("run.20.20.3").scan_number == 20


def test_peak_list_duplicates_raise(tmp_path: Path) -> None:
    path = write_text(tmp_path / "dup.mgf", MGF.replace("SCANS=21", "SCANS=20").replace("run.21.21.2", "run.19.19.2"))
    reader = MgfReader(path)
    with pytest.raises(SpxtacularError, match="shared by several spectra"):
        reader.get_by_scan(20)
    with pytest.raises(SpxtacularError, match="more than one spectrum"):
        reader.get_by_native_id("run.19.19.2")
    assert reader.get_by_scan(19).native_id == "run.19.19.2"
    assert reader.get_by_native_id("run.20.20.3").mz.tolist() == [200.0, 201.0]


def test_mgf_without_scans_points_to_native_id(tmp_path: Path) -> None:
    path = write_text(tmp_path / "noscans.mgf", "\n".join(line for line in MGF.splitlines() if "SCANS" not in line))
    reader = MgfReader(path)
    with pytest.raises(SpxtacularError, match="get_by_native_id"):
        reader.get_by_scan(19)
    assert reader.get_by_native_id("run.20.20.3").mz.tolist() == [200.0, 201.0]


def test_ms2_lookup_and_synthesised_native_id(tmp_path: Path) -> None:
    path = write_text(
        tmp_path / "run.ms2",
        """
H\tCreationDate\ttoday
S\t7\t7\t500.25
Z\t2\t999.49
100.0 10.0
S\t8\t8\t600.5
I\tNativeID\tcontrollerType=0 controllerNumber=1 scan=8
200.0 20.0
S\t9\t9\t700.75
300.0 30.0
""",
    )
    reader = Ms2Reader(path)
    assert reader.get_by_scan(9).mz.tolist() == [300.0]
    assert reader.get_by_scan(7).mz.tolist() == [100.0]
    assert reader.get_by_native_id("scan=7").scan_number == 7
    assert reader.get_by_native_id("controllerType=0 controllerNumber=1 scan=8").mz.tolist() == [200.0]
    # A Sage .pin scannr is the bare scan number.
    assert reader.get_by_sage_scannr("9").scan_number == 9
    assert reader.get_by_sage_scannr(8).scan_number == 8


def test_msp_has_no_scan_numbers(tmp_path: Path) -> None:
    path = write_text(
        tmp_path / "lib.msp",
        """
Name: PEPTIDE/2
PrecursorMZ: 400.2
Num Peaks: 1
100.0 10.0

Name: PEPTIDEK/2
PrecursorMZ: 450.2
Num Peaks: 2
100.0 10.0
200.0 20.0
""",
    )
    reader = MspReader(path)
    with pytest.raises(SpxtacularError, match="get_by_native_id"):
        reader.get_by_scan(1)
    assert reader.get_by_native_id("PEPTIDEK/2").mz.tolist() == [100.0, 200.0]
    assert reader.get_by_sage_scannr("PEPTIDE/2").mz.tolist() == [100.0]


def test_sage_scannr_as_mgf_title(mgf_path: Path) -> None:
    reader = MgfReader(mgf_path)
    assert reader.get_by_sage_scannr("run.20.20.3").scan_number == 20
    assert reader.get_by_sage_scannr("21").native_id == "run.21.21.2"  # .pin: bare scan number
    with pytest.raises(KeyError):
        reader.get_by_sage_scannr("not-a-title")


@pytest.mark.parametrize("bad", [1.5, None, True])
def test_bad_argument_types_raise(mgf_path: Path, bad: object) -> None:
    reader = MgfReader(mgf_path)
    with pytest.raises(SpxtacularError):
        reader.get_by_scan(bad)  # ty: ignore[invalid-argument-type]
    with pytest.raises(SpxtacularError):
        reader.get_by_sage_scannr(bad)  # ty: ignore[invalid-argument-type]


def test_bad_ms_level_raises(mgf_path: Path) -> None:
    with pytest.raises(SpxtacularError, match="ms_level"):
        MgfReader(mgf_path).get_by_scan(19, ms_level=0)


def test_unified_reader_delegates(mgf_path: Path) -> None:
    with Reader(mgf_path) as reader:
        assert reader.get_by_scan(20).native_id == "run.20.20.3"
        assert reader.get_by_native_id("run.19.19.2").scan_number == 19
        assert reader.get_by_sage_scannr("run.21.21.2").scan_number == 21
        with pytest.raises(SpxtacularError, match="Bruker"):
            reader.get_by_sage_scannr("1", precursor_offset=0)


# ---------------------------------------------------------------------------
# mzML
# ---------------------------------------------------------------------------

needs_mzml = pytest.mark.skipif(not _reader_module._HAS_MZMLPY, reason="mzmlpy is not installed")

EXAMPLE_MZML = DATA_DIR / "example.mzML"  # ids: scan=19 (MS1), scan=20 (MS2), scan=21 (MS1), SCIEX-style
BRUKER_MZML = DATA_DIR / "bruker_im_one_spectrum.mzML"


@needs_mzml
@pytest.mark.parametrize("path", [EXAMPLE_MZML, DATA_DIR / "example.mzML.gz"])
@pytest.mark.parametrize("opened", [True, False])
def test_mzml_get_by_scan(path: Path, opened: bool) -> None:
    reader = MzmlReader(path)
    if opened:
        reader.open()
    try:
        spec = reader.get_by_scan(20)
        assert spec.native_id == "scan=20"
        assert spec.ms_level == 2
        assert reader.get_by_scan(21, ms_level=1).native_id == "scan=21"
        with pytest.raises(KeyError):
            reader.get_by_scan(20, ms_level=1)
        with pytest.raises(KeyError):
            reader.get_by_scan(22)  # the SCIEX-style id carries cycle=22, not a scan number
    finally:
        reader.close()


@needs_mzml
def test_mzml_get_by_native_id() -> None:
    with MzmlReader(EXAMPLE_MZML) as reader:
        spec = reader.get_by_native_id("sample=1 period=1 cycle=22 experiment=1")
        assert spec.scan_number is None
        assert reader.get_by_native_id("scan=19").scan_number == 19
        with pytest.raises(KeyError):
            reader.get_by_native_id("scan=99")


@needs_mzml
@pytest.mark.filterwarnings("ignore:This spectrum has multiple scans")
def test_mzml_without_scan_numbers_points_to_native_id() -> None:
    native_id = "merged=1015 frame=1016 scanStart=655 scanEnd=679"
    with MzmlReader(BRUKER_MZML) as reader:
        with pytest.raises(SpxtacularError, match="get_by_native_id"):
            reader.get_by_scan(1016)
        assert reader.get_by_native_id(native_id).native_id == native_id
        assert reader.get_by_sage_scannr(native_id).native_id == native_id


@needs_mzml
def test_mzml_sage_scannr() -> None:
    with Reader(EXAMPLE_MZML) as reader:
        assert reader.get_by_sage_scannr("scan=20").scan_number == 20  # results.sage.tsv
        assert reader.get_by_sage_scannr("20").native_id == "scan=20"  # .pin
        assert reader.get_by_sage_scannr(19).native_id == "scan=19"


@needs_mzml
def test_mzml_duplicate_scan_numbers_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    import spxtacular.reader as reader_module

    monkeypatch.setattr(reader_module, "_mzml_spectrum_ids", lambda handle: ["scan=5", "index=5", "scan=6"])
    with MzmlReader(EXAMPLE_MZML) as reader:
        with pytest.raises(SpxtacularError, match="shared by several spectra"):
            reader.get_by_scan(5)


# ---------------------------------------------------------------------------
# Thermo
# ---------------------------------------------------------------------------

RAW_PATH = DATA_DIR / "Angiotensin_325-CID.raw"

try:
    thermo_module._require_fisher()
    _HAS_FISHER = True
except ImportError:
    if os.environ.get("SPXTACULAR_REQUIRE_THERMO") == "1":
        raise
    _HAS_FISHER = False


@pytest.mark.skipif(not _HAS_FISHER, reason="fisher-py or its .NET runtime is unavailable")
def test_thermo_lookups() -> None:
    with thermo_module.ThermoReader(RAW_PATH) as reader:
        spec = reader.get_by_scan(3)
        assert spec.scan_number == 3
        assert reader.get_by_scan(3, ms_level=2).scan_number == 3
        assert reader.get_by_native_id("controllerType=0 controllerNumber=1 scan=4").scan_number == 4
        assert reader.get_by_native_id("scan=5").scan_number == 5
        assert reader.get_by_sage_scannr("controllerType=0 controllerNumber=1 scan=6").scan_number == 6
        assert reader.get_by_sage_scannr("7").scan_number == 7
        with pytest.raises(KeyError):
            reader.get_by_scan(3, ms_level=1)
        with pytest.raises(KeyError):
            reader.get_by_scan(999)
        with pytest.raises(KeyError):
            reader.get_by_native_id("frame=3")


def test_thermo_native_id_forms_do_not_need_the_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """The id parsing happens before any scan is read."""
    seen: list[int] = []

    def fake_getitem(self: object, scan_number: int) -> MsnSpectrum:
        seen.append(scan_number)
        return MsnSpectrum(
            mz=np.array([1.0]),
            intensity=np.array([1.0]),
            spectrum_type=SpectrumType.CENTROID,
            scan_number=scan_number,
            ms_level=2,
            precursors=[Precursor(precursor_mz=1.0, intensity=0.0, charge=1, im=None)],
        )

    monkeypatch.setattr(thermo_module.ThermoScanLookup, "__getitem__", fake_getitem)
    reader = thermo_module.ThermoReader.__new__(thermo_module.ThermoReader)
    reader.get_by_native_id("controllerType=0 controllerNumber=1 scan=12")
    reader.get_by_native_id("scan=13")
    assert seen == [12, 13]
    with pytest.raises(KeyError):
        reader.get_by_native_id("controllerType=0 controllerNumber=2 scan=12")


# ---------------------------------------------------------------------------
# Bruker .d
# ---------------------------------------------------------------------------

needs_tdfpy = pytest.mark.skipif(not _reader_module._HAS_TDFPY, reason="tdfpy is not installed")

DDA_D = DATA_DIR / "example_dda.d"
DIA_D = DATA_DIR / "example_dia.d"
PRM_D = DATA_DIR / "example_prm.d"


@pytest.fixture(scope="module")
def dda() -> Iterator[DReader]:
    if not _reader_module._HAS_TDFPY:
        pytest.skip("tdfpy is not installed")
    with DReader(DDA_D) as reader:
        yield reader


@needs_tdfpy
def test_dreader_native_ids(dda: DReader) -> None:
    ms1 = next(iter(dda.ms1))
    ms2 = next(iter(dda.ms2))
    assert ms1.native_id == f"frame={ms1.scan_number}"
    assert ms2.native_id == f"precursor={ms2.scan_number}"
    assert dda.get_by_native_id(ms1.native_id).scan_number == ms1.scan_number
    by_id = dda.get_by_native_id(ms2.native_id)
    np.testing.assert_array_equal(by_id.mz, ms2.mz)


@needs_tdfpy
def test_dreader_dda_get_by_scan(dda: DReader) -> None:
    # Frame 1 is MS1 and precursor 1 exists: the number is ambiguous without ms_level.
    with pytest.raises(SpxtacularError, match="ms_level"):
        dda.get_by_scan(1)
    assert dda.get_by_scan(1, ms_level=1).ms_level == 1
    assert dda.get_by_scan(1, ms_level=2).native_id == "precursor=1"
    # Precursor 2 exists, frame 2 is not MS1: unambiguous.
    assert dda.get_by_scan(2).native_id == "precursor=2"
    with pytest.raises(KeyError):
        dda.get_by_scan(2, ms_level=1)
    with pytest.raises(KeyError):
        dda.get_by_scan(10**9)


@needs_tdfpy
def test_dreader_sage_scannr(dda: DReader) -> None:
    # Upstream Sage (timsrust 0.4) numbers spectra from 0: scannr N is precursor N + 1.
    assert dda.get_by_sage_scannr(0).native_id == "precursor=1"
    assert dda.get_by_sage_scannr("41").native_id == "precursor=42"
    assert dda.get_by_sage_scannr("41", precursor_offset=0).native_id == "precursor=41"
    with Reader(DDA_D) as reader:
        assert reader.get_by_sage_scannr("41", precursor_offset=0).native_id == "precursor=41"
        assert reader.get_by_sage_scannr("41").native_id == "precursor=42"
    with pytest.raises(SpxtacularError, match="bare integer"):
        dda.get_by_sage_scannr("scan=41")
    with pytest.raises(KeyError):
        dda.get_by_sage_scannr(10**9)


@needs_tdfpy
@pytest.mark.parametrize("native_id", ["nonsense", "precursor=abc", "1@w0", "1@t1", "frame=2"])
def test_dreader_unknown_native_ids_raise_key_error(dda: DReader, native_id: str) -> None:
    with pytest.raises(KeyError):
        dda.get_by_native_id(native_id)


@needs_tdfpy
def test_dreader_requires_open() -> None:
    reader = DReader(DDA_D)
    with pytest.raises(SpxtacularError, match="opened"):
        reader.get_by_scan(1, ms_level=1)
    with pytest.raises(SpxtacularError, match="opened"):
        reader.get_by_native_id("frame=1")


@needs_tdfpy
def test_dreader_dia_lookups() -> None:
    with DReader(DIA_D) as reader:
        window = next(iter(reader.ms2))
        assert window.native_id is not None and window.scan_number is not None
        with pytest.raises(SpxtacularError, match="get_by_native_id"):
            reader.get_by_scan(window.scan_number)
        by_id = reader.get_by_native_id(window.native_id)
        assert by_id.native_id == window.native_id
        np.testing.assert_array_equal(by_id.mz, window.mz)
        ms1 = next(iter(reader.ms1))
        assert reader.get_by_scan(ms1.scan_number or 0).native_id == ms1.native_id
        with pytest.raises(KeyError):
            reader.get_by_native_id("precursor=1")
        with pytest.raises(KeyError):
            reader.get_by_native_id(f"{window.scan_number}@w999")
        with pytest.raises(SpxtacularError, match="DDA runs only"):
            reader.get_by_sage_scannr(0)


@needs_tdfpy
def test_dreader_prm_lookups() -> None:
    with DReader(PRM_D) as reader:
        transition = next(iter(reader.ms2))
        assert transition.native_id is not None and transition.scan_number is not None
        assert reader.get_by_scan(transition.scan_number).native_id == transition.native_id
        assert reader.get_by_native_id(transition.native_id).scan_number == transition.scan_number
        with pytest.raises(KeyError):
            reader.get_by_native_id(f"{transition.scan_number}@w0")
        with pytest.raises(SpxtacularError, match="DDA runs only"):
            reader.get_by_sage_scannr(0)
