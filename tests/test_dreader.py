import pathlib
import re
import sqlite3

import numpy as np
import pytest

tdfpy = pytest.importorskip("tdfpy")

from spxtacular.core import MsnSpectrum, SpectrumType  # noqa: E402
from spxtacular.enums import ActivationType, Analyzer  # noqa: E402
from spxtacular.reader import AcquisitionType, DReader, _detect_acquisition_type  # noqa: E402

DATA_DIR = pathlib.Path(__file__).parent / "data"
HELA_D = DATA_DIR / "example_dda.d"
PRM_D = DATA_DIR / "example_prm.d"
DIA_D = DATA_DIR / "example_dia.d"


@pytest.fixture(scope="module")
def ms1_spectrum():
    with DReader(str(HELA_D)) as r:
        return next(iter(r.ms1))


@pytest.fixture(scope="module")
def ms2_spectrum():
    with DReader(str(HELA_D)) as r:
        return next(iter(r.ms2))


# --- acquisition type ---


def test_dreader_detects_dda():
    assert DReader(str(HELA_D)).acquisition_type == AcquisitionType.DDA


def test_fractional_precursor_scan_preserves_backend_mobility():
    with tdfpy.DDA(str(HELA_D)) as reader:
        precursor = next(p for p in reader.precursors if p.scan_number != int(p.scan_number))
        converted = DReader._parse_dda_precursor(precursor)
        restored = MsnSpectrum.from_json(converted.to_json())
        assert isinstance(restored, MsnSpectrum)
        assert restored.scan_number == precursor.precursor_id
        assert restored.precursors is not None
        assert restored.precursors[0].im == precursor.ook0


def _fake_d_folder(tmp_path, msms_types):
    """A .d folder whose analysis.tdf declares only the given MsMsType values."""
    d_dir = tmp_path / "fake.d"
    d_dir.mkdir()
    with sqlite3.connect(str(d_dir / "analysis.tdf")) as conn:
        conn.execute("CREATE TABLE Frames (Id INTEGER PRIMARY KEY, MsMsType INTEGER)")
        conn.executemany(
            "INSERT INTO Frames (Id, MsMsType) VALUES (?, ?)",
            [(i + 1, t) for i, t in enumerate(msms_types)],
        )
    conn.close()
    return d_dir


def test_classic_msms_raises_instead_of_opening_as_dda(tmp_path):
    """MsMsType 2 has no tdfpy backend; DDA would crash or yield nothing."""
    d_dir = _fake_d_folder(tmp_path, [0, 2, 0, 2])
    with pytest.raises(ValueError, match="Unsupported acquisition type"):
        _detect_acquisition_type(d_dir)
    with pytest.raises(ValueError, match="MsMsType 2 "):
        DReader(str(d_dir))


def test_classic_msms_error_points_at_the_alternative(tmp_path):
    d_dir = _fake_d_folder(tmp_path, [0, 2])
    with pytest.raises(ValueError, match="MzmlReader"):
        _detect_acquisition_type(d_dir)


def test_pasef_frames_win_over_classic_msms_frames(tmp_path):
    """A run that also holds PASEF frames is still readable as DDA."""
    d_dir = _fake_d_folder(tmp_path, [0, 2, 8])
    assert _detect_acquisition_type(d_dir) == AcquisitionType.DDA


def test_ms1_only_run_is_still_unknown(tmp_path):
    d_dir = _fake_d_folder(tmp_path, [0, 0])
    assert _detect_acquisition_type(d_dir) == AcquisitionType.UNKNOWN


# --- MS1 ---


def test_ms1_is_msn_spectrum(ms1_spectrum):
    assert isinstance(ms1_spectrum, MsnSpectrum)


def test_ms1_level(ms1_spectrum):
    assert ms1_spectrum.ms_level == 1


def test_ms1_spectrum_type_is_centroid(ms1_spectrum):
    assert ms1_spectrum.spectrum_type == SpectrumType.CENTROID


def test_ms1_arrays_nonempty(ms1_spectrum):
    assert len(ms1_spectrum.mz) > 0


def test_ms1_mz_intensity_same_length(ms1_spectrum):
    assert len(ms1_spectrum.mz) == len(ms1_spectrum.intensity)


def test_ms1_intensity_positive(ms1_spectrum):
    assert np.all(ms1_spectrum.intensity > 0)


def test_ms1_has_ion_mobility(ms1_spectrum):
    assert ms1_spectrum.im is not None
    assert len(ms1_spectrum.im) == len(ms1_spectrum.mz)


def test_ms1_im_values_positive(ms1_spectrum):
    assert np.all(ms1_spectrum.im > 0)


def test_ms1_rt_is_float(ms1_spectrum):
    assert isinstance(ms1_spectrum.rt, float)
    assert ms1_spectrum.rt > 0


def test_ms1_analyzer_is_tof(ms1_spectrum):
    assert ms1_spectrum.analyzer == Analyzer.TOF


def test_ms1_polarity_is_positive(ms1_spectrum):
    assert ms1_spectrum.polarity == "positive"


def test_ms1_scan_number_set(ms1_spectrum):
    assert ms1_spectrum.scan_number is not None


def test_ms1_no_precursors(ms1_spectrum):
    assert ms1_spectrum.precursors is None


def test_ms1_no_charge_array(ms1_spectrum):
    assert ms1_spectrum.charge is None


# --- MS2 ---


def test_ms2_is_msn_spectrum(ms2_spectrum):
    assert isinstance(ms2_spectrum, MsnSpectrum)


def test_ms2_level(ms2_spectrum):
    assert ms2_spectrum.ms_level == 2


def test_ms2_spectrum_type_is_centroid(ms2_spectrum):
    assert ms2_spectrum.spectrum_type == SpectrumType.CENTROID


def test_ms2_arrays_nonempty(ms2_spectrum):
    assert len(ms2_spectrum.mz) > 0


def test_ms2_mz_intensity_same_length(ms2_spectrum):
    assert len(ms2_spectrum.mz) == len(ms2_spectrum.intensity)


def test_ms2_analyzer_is_tof(ms2_spectrum):
    assert ms2_spectrum.analyzer == Analyzer.TOF


def test_ms2_has_precursor(ms2_spectrum):
    assert ms2_spectrum.precursors is not None
    assert len(ms2_spectrum.precursors) == 1


def test_ms2_precursor_mz_positive(ms2_spectrum):
    assert ms2_spectrum.precursors[0].mz > 0


def test_ms2_precursor_charge_set(ms2_spectrum):
    assert ms2_spectrum.precursors[0].charge is not None
    assert ms2_spectrum.precursors[0].charge > 0


def test_ms2_precursor_has_ion_mobility(ms2_spectrum):
    assert ms2_spectrum.precursors[0].im is not None


def test_ms2_collision_energy_set(ms2_spectrum):
    assert ms2_spectrum.collision_energy is not None
    assert ms2_spectrum.collision_energy > 0


def test_ms2_activation_type_set(ms2_spectrum):
    assert ms2_spectrum.activation_type is not None


# --- lookup __getitem__ ---


def test_ms1_lookup_by_frame_id(ms1_spectrum):
    with DReader(str(HELA_D)) as r:
        spec = r.ms1[ms1_spectrum.scan_number]
    assert isinstance(spec, MsnSpectrum)
    np.testing.assert_array_equal(spec.mz, ms1_spectrum.mz)


def test_ms1_lookup_invalid_id_raises():
    with DReader(str(HELA_D)) as r:
        with pytest.raises(KeyError):
            r.ms1[999_999_999]


def test_ms2_lookup_by_precursor_id(ms2_spectrum):
    with DReader(str(HELA_D)) as r:
        spec = r.ms2[ms2_spectrum.scan_number]
    assert isinstance(spec, MsnSpectrum)
    np.testing.assert_array_equal(spec.mz, ms2_spectrum.mz)


def test_ms2_lookup_invalid_id_raises():
    with DReader(str(HELA_D)) as r:
        with pytest.raises(KeyError):
            r.ms2[999_999_999]


def test_lookup_outside_context_raises():
    r = DReader(str(HELA_D))
    with pytest.raises(RuntimeError):
        r.ms1[1]


# ---------------------------------------------------------------------------
# PRM
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prm_ms1_spectrum():
    with DReader(str(PRM_D)) as r:
        return next(iter(r.ms1))


@pytest.fixture(scope="module")
def prm_ms2_spectrum():
    with DReader(str(PRM_D)) as r:
        return next(iter(r.ms2))


def test_dreader_detects_prm():
    assert DReader(str(PRM_D)).acquisition_type == AcquisitionType.PRM


def test_prm_open_uses_prm_reader():
    with DReader(str(PRM_D)) as r:
        assert isinstance(r._reader, tdfpy.PRM)


def test_prm_ms1_is_msn_spectrum(prm_ms1_spectrum):
    assert isinstance(prm_ms1_spectrum, MsnSpectrum)
    assert prm_ms1_spectrum.ms_level == 1
    assert prm_ms1_spectrum.spectrum_type == SpectrumType.CENTROID
    assert prm_ms1_spectrum.analyzer == Analyzer.TOF
    assert prm_ms1_spectrum.polarity == "positive"
    assert prm_ms1_spectrum.scan_number is not None
    assert prm_ms1_spectrum.precursors is None


def test_prm_ms1_lookup_by_frame_id(prm_ms1_spectrum):
    with DReader(str(PRM_D)) as r:
        spec = r.ms1[prm_ms1_spectrum.scan_number]
    assert isinstance(spec, MsnSpectrum)
    assert spec.scan_number == prm_ms1_spectrum.scan_number


def test_prm_ms2_is_msn_spectrum(prm_ms2_spectrum):
    assert isinstance(prm_ms2_spectrum, MsnSpectrum)
    assert prm_ms2_spectrum.ms_level == 2
    assert prm_ms2_spectrum.spectrum_type == SpectrumType.CENTROID
    assert prm_ms2_spectrum.analyzer == Analyzer.TOF


def test_prm_ms2_native_id_format(prm_ms2_spectrum):
    assert prm_ms2_spectrum.native_id is not None
    assert re.fullmatch(r"\d+@t\d+", prm_ms2_spectrum.native_id)


def test_prm_ms2_has_precursor(prm_ms2_spectrum):
    assert prm_ms2_spectrum.precursors is not None
    assert len(prm_ms2_spectrum.precursors) == 1
    prec = prm_ms2_spectrum.precursors[0]
    assert prec.is_monoisotopic is True
    assert prec.mz > 0
    assert prec.charge is not None and prec.charge > 0
    assert prec.im is not None and prec.im > 0


def test_prm_ms2_isolation_window(prm_ms2_spectrum):
    assert prm_ms2_spectrum.isolation_mz_range is not None
    lo, hi = prm_ms2_spectrum.isolation_mz_range
    assert lo < hi
    assert prm_ms2_spectrum.isolation_im_range is not None


def test_prm_ms2_collision_energy(prm_ms2_spectrum):
    assert prm_ms2_spectrum.collision_energy is not None
    assert prm_ms2_spectrum.collision_energy > 0
    assert prm_ms2_spectrum.activation_type == ActivationType.PASEF


def test_prm_ms2_iteration_yields_multiple():
    with DReader(str(PRM_D)) as r:
        n = 0
        for _ in r.ms2:
            n += 1
            if n >= 3:
                break
        assert n >= 1


def test_prm_ms2_getitem_raises():
    with DReader(str(PRM_D)) as r:
        with pytest.raises(NotImplementedError):
            r.ms2[0]


# ---------------------------------------------------------------------------
# DIA
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dia_ms1_spectrum():
    with DReader(str(DIA_D)) as r:
        return next(iter(r.ms1))


@pytest.fixture(scope="module")
def dia_ms2_spectrum():
    with DReader(str(DIA_D)) as r:
        return next(iter(r.ms2))


def test_dreader_detects_dia():
    assert DReader(str(DIA_D)).acquisition_type == AcquisitionType.DIA


def test_dia_open_uses_dia_reader():
    with DReader(str(DIA_D)) as r:
        assert isinstance(r._reader, tdfpy.DIA)


def test_dia_ms1_is_msn_spectrum(dia_ms1_spectrum):
    assert isinstance(dia_ms1_spectrum, MsnSpectrum)
    assert dia_ms1_spectrum.ms_level == 1
    assert dia_ms1_spectrum.spectrum_type == SpectrumType.CENTROID
    assert dia_ms1_spectrum.analyzer == Analyzer.TOF
    assert dia_ms1_spectrum.polarity == "positive"
    assert dia_ms1_spectrum.scan_number is not None
    assert dia_ms1_spectrum.precursors is None


def test_dia_ms1_has_ion_mobility(dia_ms1_spectrum):
    assert dia_ms1_spectrum.im is not None
    assert len(dia_ms1_spectrum.im) == len(dia_ms1_spectrum.mz)


def test_dia_ms1_lookup_by_frame_id(dia_ms1_spectrum):
    with DReader(str(DIA_D)) as r:
        spec = r.ms1[dia_ms1_spectrum.scan_number]
    assert isinstance(spec, MsnSpectrum)
    assert spec.scan_number == dia_ms1_spectrum.scan_number


def test_dia_ms2_is_msn_spectrum(dia_ms2_spectrum):
    assert isinstance(dia_ms2_spectrum, MsnSpectrum)
    assert dia_ms2_spectrum.ms_level == 2
    assert dia_ms2_spectrum.spectrum_type == SpectrumType.CENTROID
    assert dia_ms2_spectrum.analyzer == Analyzer.TOF


def test_dia_ms2_native_id_format(dia_ms2_spectrum):
    # DReader._parse_dia_window builds native_id as "{frame_id}@w{window_index}"
    assert dia_ms2_spectrum.native_id is not None
    assert re.fullmatch(r"\d+@w\d+", dia_ms2_spectrum.native_id)


def test_dia_ms2_no_precursors(dia_ms2_spectrum):
    # DIA windows don't have individual precursor ions (they're isolation windows)
    assert dia_ms2_spectrum.precursors is None


def test_dia_ms2_isolation_window(dia_ms2_spectrum):
    assert dia_ms2_spectrum.isolation_mz_range is not None
    lo, hi = dia_ms2_spectrum.isolation_mz_range
    assert lo < hi
    assert dia_ms2_spectrum.isolation_im_range is not None


def test_dia_ms2_collision_energy(dia_ms2_spectrum):
    assert dia_ms2_spectrum.collision_energy is not None
    assert dia_ms2_spectrum.collision_energy > 0
    assert dia_ms2_spectrum.activation_type == ActivationType.PASEF


def test_dia_ms2_has_ion_mobility(dia_ms2_spectrum):
    assert dia_ms2_spectrum.im is not None
    assert len(dia_ms2_spectrum.im) == len(dia_ms2_spectrum.mz)


def test_dia_ms2_iteration_yields_multiple():
    with DReader(str(DIA_D)) as r:
        n = 0
        for _ in r.ms2:
            n += 1
            if n >= 3:
                break
        assert n >= 1


def test_dia_ms2_getitem_raises():
    with DReader(str(DIA_D)) as r:
        with pytest.raises(NotImplementedError):
            r.ms2[0]
