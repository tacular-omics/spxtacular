import pathlib

import numpy as np
import pytest

from spxtacular.core import MsnSpectrum, SpectrumType
from spxtacular.reader import AcquisitionType, DReader

DATA_DIR = pathlib.Path(__file__).parent
HELA_D = DATA_DIR / "200ngHeLaPASEF_1min.d"


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
    assert ms1_spectrum.analyzer == "TOF"


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
    assert ms2_spectrum.analyzer == "TOF"


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
