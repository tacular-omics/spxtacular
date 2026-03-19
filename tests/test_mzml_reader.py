import pathlib

import numpy as np
import pytest

from spxtacular.core import MsnSpectrum, SpectrumType
from spxtacular.reader import MzmlReader

DATA_DIR = pathlib.Path(__file__).parent
EXAMPLE_MZML = DATA_DIR / "example.mzML"


@pytest.fixture(scope="module")
def ms1_spectrum():
    with MzmlReader(str(EXAMPLE_MZML)) as r:
        return next(iter(r.ms1))


@pytest.fixture(scope="module")
def ms2_spectrum():
    with MzmlReader(str(EXAMPLE_MZML)) as r:
        return next(iter(r.ms2))


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


def test_ms1_mz_sorted(ms1_spectrum):
    assert np.all(ms1_spectrum.mz[:-1] <= ms1_spectrum.mz[1:])


def test_ms1_intensity_positive(ms1_spectrum):
    assert np.all(ms1_spectrum.intensity > 0)


def test_ms1_rt_is_float(ms1_spectrum):
    assert isinstance(ms1_spectrum.rt, float)
    assert ms1_spectrum.rt > 0


def test_ms1_mz_range_set(ms1_spectrum):
    assert ms1_spectrum.mz_range is not None
    lo, hi = ms1_spectrum.mz_range
    assert hi > lo > 0


def test_ms1_native_id_set(ms1_spectrum):
    assert ms1_spectrum.native_id is not None
    assert len(ms1_spectrum.native_id) > 0


def test_ms1_no_precursors(ms1_spectrum):
    assert ms1_spectrum.precursors is None


def test_ms1_no_charge_array(ms1_spectrum):
    assert ms1_spectrum.charge is None


def test_ms1_no_ion_mobility(ms1_spectrum):
    assert ms1_spectrum.im is None


# --- MS2 ---

def test_ms2_is_msn_spectrum(ms2_spectrum):
    assert isinstance(ms2_spectrum, MsnSpectrum)


def test_ms2_level(ms2_spectrum):
    assert ms2_spectrum.ms_level == 2


def test_ms2_spectrum_type(ms2_spectrum):
    assert ms2_spectrum.spectrum_type in (SpectrumType.CENTROID, SpectrumType.PROFILE)


def test_ms2_arrays_nonempty(ms2_spectrum):
    assert len(ms2_spectrum.mz) > 0


def test_ms2_mz_intensity_same_length(ms2_spectrum):
    assert len(ms2_spectrum.mz) == len(ms2_spectrum.intensity)


def test_ms2_rt_is_float(ms2_spectrum):
    assert isinstance(ms2_spectrum.rt, float)
    assert ms2_spectrum.rt > 0


def test_ms2_has_precursor(ms2_spectrum):
    assert ms2_spectrum.precursors is not None
    assert len(ms2_spectrum.precursors) > 0


def test_ms2_precursor_mz_positive(ms2_spectrum):
    assert ms2_spectrum.precursors[0].mz > 0


def test_ms2_precursor_charge_set(ms2_spectrum):
    assert ms2_spectrum.precursors[0].charge is not None
    assert ms2_spectrum.precursors[0].charge > 0


def test_ms2_collision_energy_set(ms2_spectrum):
    assert ms2_spectrum.collision_energy is not None
    assert ms2_spectrum.collision_energy > 0


def test_ms2_activation_type_set(ms2_spectrum):
    assert ms2_spectrum.activation_type is not None


# --- __getitem__ ---

def test_getitem_by_index_returns_msn_spectrum():
    r = MzmlReader(str(EXAMPLE_MZML))
    spec = r[0]
    assert isinstance(spec, MsnSpectrum)


def test_getitem_by_index_matches_iteration():
    r = MzmlReader(str(EXAMPLE_MZML))
    spec_iter = next(iter(r.ms1))
    spec_item = r[0]
    assert spec_item.native_id == spec_iter.native_id
    np.testing.assert_array_equal(spec_item.mz, spec_iter.mz)


def test_getitem_by_native_id():
    r = MzmlReader(str(EXAMPLE_MZML))
    spec = r["scan=19"]
    assert isinstance(spec, MsnSpectrum)
    assert spec.native_id == "scan=19"


def test_getitem_by_native_id_matches_index():
    r = MzmlReader(str(EXAMPLE_MZML))
    by_index = r[0]
    by_id = r[by_index.native_id]
    np.testing.assert_array_equal(by_index.mz, by_id.mz)


def test_getitem_invalid_id_raises():
    r = MzmlReader(str(EXAMPLE_MZML))
    with pytest.raises(KeyError):
        r["scan=999999"]
