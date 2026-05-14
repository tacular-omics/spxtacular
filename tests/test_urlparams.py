from urllib.parse import urlencode

import numpy as np
import pytest
from data import EXAMPLE_SPECTRUM

from spxtacular import (
    MsnSpectrum,
    Precursor,
    Spectrum,
    spectrum_from_query_params,
    spectrum_to_query_params,
    spectrum_to_query_string,
)


def _make_full_spectrum() -> Spectrum:
    mz = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float64)
    intensity = np.array([10.0, 50.0, 80.0, 30.0], dtype=np.float64)
    charge = np.array([1, 2, 1, 3], dtype=np.int32)
    im = np.array([0.8, 0.9, 1.0, 1.1], dtype=np.float64)
    return Spectrum(mz=mz, intensity=intensity, charge=charge, im=im)


def test_spectrum_roundtrip_mz_intensity():
    params = spectrum_to_query_params(EXAMPLE_SPECTRUM)
    decoded = spectrum_from_query_params(params)

    assert isinstance(decoded, Spectrum)
    assert not isinstance(decoded, MsnSpectrum)
    np.testing.assert_allclose(decoded.mz, EXAMPLE_SPECTRUM.mz, rtol=1e-5)
    np.testing.assert_allclose(decoded.intensity, EXAMPLE_SPECTRUM.intensity, rtol=1e-5)


def test_spectrum_roundtrip_with_charge_and_im():
    spec = _make_full_spectrum()
    decoded = spectrum_from_query_params(spectrum_to_query_params(spec))

    np.testing.assert_allclose(decoded.mz, spec.mz, rtol=1e-5)
    np.testing.assert_allclose(decoded.intensity, spec.intensity, rtol=1e-5)
    np.testing.assert_array_equal(decoded.charge, spec.charge)
    np.testing.assert_allclose(decoded.im, spec.im, rtol=1e-5)


def test_max_peaks_keeps_top_intensity_sorted_by_mz():
    spec = _make_full_spectrum()
    params = spectrum_to_query_params(spec, max_peaks=2)
    decoded = spectrum_from_query_params(params)

    assert len(decoded.mz) == 2
    # Top 2 by intensity were idx 1 (int=50) and idx 2 (int=80); m/z 200 and 300.
    np.testing.assert_allclose(decoded.mz, [200.0, 300.0], rtol=1e-5)
    # m/z monotonically non-decreasing.
    assert np.all(np.diff(decoded.mz) >= 0)


def test_max_peaks_larger_than_spectrum_is_noop():
    spec = _make_full_spectrum()
    decoded = spectrum_from_query_params(spectrum_to_query_params(spec, max_peaks=999))
    assert len(decoded.mz) == len(spec.mz)


def test_max_peaks_select_by_mz():
    spec = _make_full_spectrum()
    decoded = spectrum_from_query_params(spectrum_to_query_params(spec, max_peaks=2, select_by="mz"))
    assert len(decoded.mz) == 2
    np.testing.assert_allclose(decoded.mz, [300.0, 400.0], rtol=1e-5)


def test_max_peaks_negative_raises():
    with pytest.raises(ValueError):
        spectrum_to_query_params(EXAMPLE_SPECTRUM, max_peaks=-1)


def test_msn_spectrum_roundtrip_metadata():
    spec = MsnSpectrum(
        mz=np.array([150.0, 250.0], dtype=np.float64),
        intensity=np.array([1000.0, 2000.0], dtype=np.float64),
        scan_number=1234,
        ms_level=2,
        native_id="scan=1234",
        rt=42.5,
        injection_time=15.0,
        total_ion_current=98765.0,
        mz_range=(100.0, 1500.0),
        im_range=(0.5, 1.5),
        im_type="1/K0",
        polarity="positive",
        resolution=60000.0,
        analyzer="FTMS",
        collision_energy=27.0,
        activation_type="HCD",
        isolation_mz_range=(498.5, 501.5),
        precursors=[
            Precursor(mz=500.25, intensity=1e6, charge=2, im=0.95, iso_score=0.98, is_monoisotopic=True),
        ],
    )

    params = spectrum_to_query_params(spec)
    decoded = spectrum_from_query_params(params)

    assert isinstance(decoded, MsnSpectrum)
    assert decoded.scan_number == 1234
    assert decoded.ms_level == 2
    assert decoded.native_id == "scan=1234"
    assert decoded.rt == 42.5
    assert decoded.injection_time == 15.0
    assert decoded.total_ion_current == 98765.0
    assert decoded.mz_range == (100.0, 1500.0)
    assert decoded.im_range == (0.5, 1.5)
    assert decoded.im_type == "1/K0"
    assert decoded.polarity == "positive"
    assert decoded.resolution == 60000.0
    assert decoded.analyzer == "FTMS"
    assert decoded.collision_energy == 27.0
    assert decoded.activation_type == "HCD"
    assert decoded.isolation_mz_range == (498.5, 501.5)
    assert decoded.precursors is not None and len(decoded.precursors) == 1
    p = decoded.precursors[0]
    assert p.mz == 500.25
    assert p.intensity == 1e6
    assert p.charge == 2
    assert p.im == 0.95
    assert p.iso_score == 0.98
    assert p.is_monoisotopic is True


def test_msn_spectrum_omits_none_fields():
    spec = MsnSpectrum(
        mz=np.array([100.0], dtype=np.float64),
        intensity=np.array([1.0], dtype=np.float64),
        scan_number=7,
    )
    params = spectrum_to_query_params(spec)
    assert "scan_number" in params
    assert "rt" not in params
    assert "precursors" not in params
    assert "mz_range" not in params

    decoded = spectrum_from_query_params(params)
    assert isinstance(decoded, MsnSpectrum)
    assert decoded.scan_number == 7
    assert decoded.rt is None


def test_accepts_query_string_with_question_mark():
    spec = _make_full_spectrum()
    qs = "?" + spectrum_to_query_string(spec)
    decoded = spectrum_from_query_params(qs)
    np.testing.assert_allclose(decoded.mz, spec.mz, rtol=1e-5)


def test_accepts_query_string_without_question_mark():
    spec = _make_full_spectrum()
    decoded = spectrum_from_query_params(spectrum_to_query_string(spec))
    np.testing.assert_allclose(decoded.mz, spec.mz, rtol=1e-5)


def test_query_string_is_urlencode_compatible():
    spec = MsnSpectrum(
        mz=np.array([100.0], dtype=np.float64),
        intensity=np.array([1.0], dtype=np.float64),
        scan_number=1,
        precursors=[Precursor(mz=500.0, intensity=1.0, charge=2, im=None, iso_score=None, is_monoisotopic=None)],
    )
    params = spectrum_to_query_params(spec)
    qs = urlencode(params)
    # Round-trip via the raw string form.
    decoded = spectrum_from_query_params(qs)
    assert isinstance(decoded, MsnSpectrum)
    assert decoded.precursors is not None
    assert decoded.precursors[0].mz == 500.0


def test_version_mismatch_raises():
    params = spectrum_to_query_params(EXAMPLE_SPECTRUM)
    params["version"] = "99"
    with pytest.raises(ValueError, match="version"):
        spectrum_from_query_params(params)


def test_missing_spectrum_param_raises():
    with pytest.raises(ValueError, match="spectrum"):
        spectrum_from_query_params({"version": "1"})


def test_empty_spectrum_roundtrip():
    spec = Spectrum(
        mz=np.array([], dtype=np.float64),
        intensity=np.array([], dtype=np.float64),
    )
    decoded = spectrum_from_query_params(spectrum_to_query_params(spec))
    assert len(decoded.mz) == 0
    assert len(decoded.intensity) == 0


def test_method_form_roundtrip():
    spec = _make_full_spectrum()
    decoded = Spectrum.from_url_params(spec.to_url_params())
    np.testing.assert_allclose(decoded.mz, spec.mz, rtol=1e-5)
    np.testing.assert_array_equal(decoded.charge, spec.charge)


def test_msn_metadata_only_triggers_msn_class():
    # Even an MsnSpectrum with no MSn fields set should decode as base Spectrum.
    spec = MsnSpectrum(
        mz=np.array([100.0], dtype=np.float64),
        intensity=np.array([1.0], dtype=np.float64),
    )
    decoded = spectrum_from_query_params(spectrum_to_query_params(spec))
    assert type(decoded) is Spectrum
