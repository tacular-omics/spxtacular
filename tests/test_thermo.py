"""Tests for ThermoReader (Thermo .raw via fisher-py).

The fixture ``Angiotensin_325-CID.raw`` is fisher-py's own test file (MIT
licensed): an Orbitrap Fusion Lumos run of 10 profile-mode FTMS CID MS2 scans
of angiotensin (precursor m/z 325, z=1, CE 35).

fisher-py needs a .NET runtime at *import* time, so everything touching the
backend is skipped — not failed — on machines without one.
"""

from __future__ import annotations

import os
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

import spxtacular.thermo as thermo_module
from spxtacular import MsnSpectrum, Reader, SpectrumType, ThermoReader
from spxtacular.enums import ActivationType, Analyzer, Polarity

RAW_PATH = Path(__file__).parent / "data" / "Angiotensin_325-CID.raw"

try:
    thermo_module._require_fisher()
    _HAS_FISHER = True
except ImportError:
    if os.environ.get("SPXTACULAR_REQUIRE_THERMO") == "1":
        raise
    _HAS_FISHER = False

needs_fisher = pytest.mark.skipif(not _HAS_FISHER, reason="fisher-py or its .NET runtime is unavailable")


@pytest.fixture(scope="module")
def raw_reader() -> Iterator[ThermoReader]:
    """Keep one vendor file handle open for read-only fixture assertions."""
    if not _HAS_FISHER:
        pytest.skip("fisher-py or its .NET runtime is unavailable")
    with ThermoReader(RAW_PATH) as reader:
        yield reader


@pytest.fixture(scope="module")
def raw_spectra(raw_reader: ThermoReader) -> list[MsnSpectrum]:
    """Read the ten fixture spectra once instead of once per assertion."""
    return list(raw_reader.ms2)


@pytest.fixture(scope="module")
def raw_spectrum(raw_reader: ThermoReader) -> MsnSpectrum:
    return raw_reader.ms2[1]


@pytest.fixture(scope="module")
def profile_spectrum() -> MsnSpectrum:
    if not _HAS_FISHER:
        pytest.skip("fisher-py or its .NET runtime is unavailable")
    with ThermoReader(RAW_PATH, prefer_vendor_centroid=False) as reader:
        return reader.ms2[1]


class _FakeScanEvent:
    def __init__(self, reaction) -> None:
        self.reaction = reaction

    def get_reaction(self, index: int):
        if index == 0:
            return self.reaction
        raise IndexError(index)


class _FakeRaw:
    """Small RawFileReader-shaped object for parser coverage without .NET."""

    def __init__(self) -> None:
        self.stats = types.SimpleNamespace(
            is_centroid_scan=False,
            start_time=1.25,
            tic=12345.0,
            low_mass=100.0,
            high_mass=1000.0,
        )
        self.scan_filter = types.SimpleNamespace(
            ms_order=types.SimpleNamespace(value=2),
            polarity=types.SimpleNamespace(name="Negative"),
            mass_analyzer=types.SimpleNamespace(name="MassAnalyzerFTMS"),
        )
        self.reaction = types.SimpleNamespace(
            activation_type=types.SimpleNamespace(name="CollisionInducedDissociation"),
            collision_energy=30.0,
            collision_energy_valid=True,
            precursor_mass=500.0,
            isolation_width=2.0,
            isolation_width_offset=0.25,
        )

    def get_scan_stats_for_scan_number(self, _scan_number: int):
        return self.stats

    def get_filter_for_scan_number(self, _scan_number: int):
        return self.scan_filter

    def get_trailer_extra_information(self, _scan_number: int):
        return types.SimpleNamespace(
            labels=["Ion Injection Time (ms):", "Orbitrap Resolution:", "Monoisotopic M/Z:", "Charge State:"],
            values=["7.5", "60000", "499.9", "2"],
        )

    def get_centroid_stream(self, _scan_number: int, _centroid_result: bool):
        return types.SimpleNamespace(masses=[150.0, 250.0], intensities=[1000.0, 500.0], charges=[0, 2])

    def get_segmented_scan_from_scan_number(self, _scan_number: int, _stats):
        return types.SimpleNamespace(positions=[149.9, 150.0, 150.1], intensities=[1.0, 10.0, 1.0])

    def get_scan_event_for_scan_number(self, _scan_number: int):
        return _FakeScanEvent(self.reaction)


def _reader_with_fake_raw(raw: _FakeRaw, *, prefer_vendor_centroid: bool) -> ThermoReader:
    reader = object.__new__(ThermoReader)
    reader.raw_path = RAW_PATH
    reader.prefer_vendor_centroid = prefer_vendor_centroid
    reader._raw = raw
    reader._instrument_model = "Orbitrap Fusion Lumos"
    return reader


# ---------------------------------------------------------------------------
# Import-failure translation (no backend needed)
# ---------------------------------------------------------------------------


def test_missing_fisher_py_raises_install_hint(monkeypatch):
    monkeypatch.setattr(thermo_module, "_fisher_modules", None)
    monkeypatch.setitem(sys.modules, "fisher_py", None)
    monkeypatch.setitem(sys.modules, "fisher_py.data", None)
    with pytest.raises(ImportError, match=r"spxtacular\[thermo\]"):
        ThermoReader(RAW_PATH)


def test_missing_dotnet_runtime_raises_dotnet_hint(monkeypatch):
    # fisher_py boots the .NET runtime at import; simulate that failing with
    # the RuntimeError pythonnet raises when no runtime is found.
    broken = types.ModuleType("fisher_py.data")

    def _boom(_name):
        raise RuntimeError("Failed to create a .NET runtime (coreclr)")

    broken.__getattr__ = _boom  # ty: ignore[invalid-assignment]
    monkeypatch.setattr(thermo_module, "_fisher_modules", None)
    monkeypatch.setitem(sys.modules, "fisher_py", types.ModuleType("fisher_py"))
    monkeypatch.setitem(sys.modules, "fisher_py.data", broken)
    with pytest.raises(ImportError, match=r"\.NET"):
        ThermoReader(RAW_PATH)


def test_parse_scan_without_fisher_runtime() -> None:
    reader = _reader_with_fake_raw(_FakeRaw(), prefer_vendor_centroid=True)

    spec = reader._parse_scan(7)

    assert spec.scan_number == 7
    assert spec.ms_level == 2
    assert spec.spectrum_type == SpectrumType.CENTROID
    assert spec.rt == pytest.approx(75.0)
    assert spec.polarity == Polarity.NEGATIVE
    assert spec.analyzer == Analyzer.ORBITRAP
    assert spec.activation_type == ActivationType.CID
    assert spec.collision_energy == 30.0
    assert spec.injection_time == 7.5
    assert spec.resolution == 60000.0
    assert spec.isolation_mz_range == pytest.approx((499.25, 501.25))
    np.testing.assert_array_equal(spec.mz, [150.0, 250.0])
    np.testing.assert_array_equal(spec.charge, [-1, 2])
    assert spec.precursors is not None
    assert spec.precursors[0].mz == 499.9
    assert spec.precursors[0].charge == 2


def test_profile_parse_path_without_fisher_runtime() -> None:
    raw = _FakeRaw()
    raw.scan_filter.ms_order.value = 1
    reader = _reader_with_fake_raw(raw, prefer_vendor_centroid=False)

    spec = reader._parse_scan(3)

    assert spec.ms_level == 1
    assert spec.spectrum_type == SpectrumType.PROFILE
    assert spec.charge is None
    assert spec.precursors is None
    np.testing.assert_array_equal(spec.mz, [149.9, 150.0, 150.1])


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------


@needs_fisher
def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        ThermoReader("no_such_file.raw")


@needs_fisher
def test_raw_directory_rejected_as_waters(tmp_path):
    waters_dir = tmp_path / "sample.raw"
    waters_dir.mkdir()
    with pytest.raises(ValueError, match="Waters"):
        ThermoReader(waters_dir)


@needs_fisher
def test_lookup_before_open_raises():
    reader = ThermoReader(RAW_PATH)
    with pytest.raises(RuntimeError, match="open"):
        next(iter(reader.ms2))


# ---------------------------------------------------------------------------
# Reading the fixture
# ---------------------------------------------------------------------------


@needs_fisher
def test_reader_autodetects_raw_suffix():
    reader = Reader(RAW_PATH)
    assert isinstance(reader._reader, ThermoReader)


@needs_fisher
def test_iterate_ms2(raw_spectra: list[MsnSpectrum]):
    assert len(raw_spectra) == 10
    for spec in raw_spectra:
        assert isinstance(spec, MsnSpectrum)
        assert spec.ms_level == 2
        assert spec.spectrum_type == SpectrumType.CENTROID
        assert len(spec) > 0
        assert np.all(spec.intensity >= 0)


@needs_fisher
def test_ms1_iteration_is_empty_for_ms2_only_file(raw_reader: ThermoReader):
    assert list(raw_reader.ms1) == []


@needs_fisher
def test_scan_metadata(raw_spectrum: MsnSpectrum):
    spec = raw_spectrum
    assert spec.scan_number == 1
    assert spec.native_id == "controllerType=0 controllerNumber=1 scan=1"
    assert spec.polarity == Polarity.POSITIVE
    assert spec.analyzer == Analyzer.ORBITRAP
    assert spec.mz_range == (150.0, 2000.0)
    assert spec.rt is not None and 0 < spec.rt < 60  # seconds, not minutes
    assert spec.total_ion_current == pytest.approx(37687076.0, rel=1e-3)
    assert spec.injection_time == pytest.approx(7.422, abs=1e-3)
    assert spec.resolution == 60000
    assert np.all(spec.mz >= 150.0)
    assert np.all(spec.mz <= 2000.0)


@needs_fisher
def test_precursor_metadata(raw_spectrum: MsnSpectrum):
    spec = raw_spectrum
    assert spec.activation_type == ActivationType.CID
    assert spec.collision_energy == 35.0
    assert spec.isolation_mz_range == (324.0, 326.0)
    assert spec.precursors is not None and len(spec.precursors) == 1
    prec = spec.precursors[0]
    assert prec.mz == 325.0
    assert prec.charge == 1
    # Trailer "Monoisotopic M/Z" is -1 (unset) in this file, so the isolation
    # target is used and not claimed to be monoisotopic.
    assert prec.is_monoisotopic is False
    assert prec.intensity > 0


@needs_fisher
def test_vendor_centroid_charges_use_spxtacular_conventions(raw_spectrum: MsnSpectrum):
    spec = raw_spectrum
    # Thermo label streams store 0 for "unknown charge"; spxtacular reserves 0
    # for decharged spectra, so unknowns must arrive as -1.
    assert spec.charge is not None
    assert spec.charge.dtype == np.int32
    assert np.all((spec.charge == -1) | (spec.charge > 0))
    assert not np.any(spec.charge == 0)


@needs_fisher
def test_profile_mode_returns_profile_trace(profile_spectrum: MsnSpectrum, raw_spectrum: MsnSpectrum):
    profile = profile_spectrum
    centroid = raw_spectrum
    assert profile.spectrum_type == SpectrumType.PROFILE
    assert profile.charge is None
    assert len(profile) > len(centroid)  # trace has many samples per peak
    # Same scan, same metadata either way.
    assert profile.rt == centroid.rt
    assert profile.precursors is not None and centroid.precursors is not None
    assert profile.precursors[0].mz == centroid.precursors[0].mz


@needs_fisher
def test_getitem_by_scan_number(raw_reader: ThermoReader):
    assert raw_reader[3].scan_number == 3
    assert raw_reader.ms2[10].scan_number == 10
    with pytest.raises(KeyError):
        raw_reader[999]
    with pytest.raises(KeyError):
        raw_reader.ms1[1]  # scan 1 is MS2


@needs_fisher
def test_spectrum_is_processable(raw_spectrum: MsnSpectrum):
    processed = raw_spectrum.filter(min_intensity=1000.0).normalize()
    assert len(processed) > 0
    assert processed.intensity.max() == pytest.approx(1.0)


@needs_fisher
def test_reopen_after_close():
    reader = ThermoReader(RAW_PATH)
    with reader:
        first = reader.ms2[1]
    with reader:
        again = reader.ms2[1]
    np.testing.assert_array_equal(first.mz, again.mz)
