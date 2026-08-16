"""Interoperability tests for spectrum_utils."""

from typing import Any

import numpy as np
import pytest

pytest.importorskip("spectrum_utils")

from spectrum_utils.spectrum import MsmsSpectrum  # noqa: E402

from spxtacular import (  # noqa: E402  # noqa: E402
    MsnSpectrum,
    Precursor,
    Spectrum,
    SpectrumType,
    from_spectrum_utils,
    to_spectrum_utils,
)


def _msn(**kwargs: Any) -> MsnSpectrum:
    values: dict[str, Any] = {
        "mz": np.array([300.0, 100.0, 200.0]),
        "intensity": np.array([30.0, 10.0, 20.0]),
        "spectrum_type": SpectrumType.CENTROID,
        "scan_number": 42,
        "native_id": "scan=42",
        "ms_level": 2,
        "rt": 125.5,
        "precursors": [Precursor(mz=500.25, intensity=0.0, charge=2, is_monoisotopic=None)],
    }
    values.update(kwargs)
    return MsnSpectrum(**values)


def test_to_spectrum_utils_maps_required_fields_and_sorts() -> None:
    converted = to_spectrum_utils(_msn())
    assert isinstance(converted, MsmsSpectrum)
    assert converted.identifier == "scan=42"
    assert converted.precursor_mz == pytest.approx(500.25)
    assert converted.precursor_charge == 2
    assert converted.retention_time == pytest.approx(125.5)
    np.testing.assert_array_equal(converted.mz, [100.0, 200.0, 300.0])
    np.testing.assert_array_equal(converted.intensity, [10.0, 20.0, 30.0])
    assert converted.intensity.dtype == np.float32


def test_spectrum_utils_round_trip_maps_supported_fields() -> None:
    restored = from_spectrum_utils(to_spectrum_utils(_msn()))
    assert isinstance(restored, MsnSpectrum)
    assert restored.spectrum_type == SpectrumType.CENTROID
    assert restored.native_id == "scan=42"
    assert restored.scan_number == 42
    assert restored.ms_level == 2
    assert restored.rt == pytest.approx(125.5)
    assert restored.precursors is not None
    assert restored.precursors[0].mz == pytest.approx(500.25)
    assert restored.precursors[0].charge == 2


def test_to_spectrum_utils_warns_about_rich_fields_and_extra_precursors() -> None:
    spectrum = _msn(
        charge=np.array([3, 1, 2]),
        im=np.array([1.3, 1.1, 1.2]),
        collision_energy=28.0,
        precursors=[
            Precursor(mz=500.25, intensity=8000.0, charge=2, is_monoisotopic=True),
            Precursor(mz=600.25, intensity=4000.0, charge=3, is_monoisotopic=False),
        ],
    )
    with pytest.warns(
        UserWarning,
        match="charge.*im.*collision_energy.*all precursors except index 1.*precursor intensity.*monoisotopic flag",
    ):
        converted = to_spectrum_utils(spectrum, precursor_index=1)
    assert converted.precursor_mz == pytest.approx(600.25)


def test_to_spectrum_utils_rejects_plain_profile_or_missing_precursor_data() -> None:
    plain = Spectrum(mz=np.array([100.0]), intensity=np.array([1.0]))
    with pytest.raises(TypeError, match="MsnSpectrum"):
        to_spectrum_utils(plain)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    with pytest.raises(ValueError, match="profile"):
        to_spectrum_utils(_msn(spectrum_type=SpectrumType.PROFILE))

    with pytest.raises(ValueError, match="at least one precursor"):
        to_spectrum_utils(_msn(precursors=None))

    with pytest.raises(ValueError, match="precursor charge"):
        to_spectrum_utils(_msn(precursors=[Precursor(mz=500.25, intensity=1.0, charge=None, is_monoisotopic=None)]))


def test_identifier_can_be_explicit_or_derived_from_scan_number() -> None:
    no_native = _msn(native_id=None)
    assert to_spectrum_utils(no_native).identifier == "scan=42"
    with pytest.warns(UserWarning, match="scan_number"):
        assert to_spectrum_utils(no_native, identifier="custom").identifier == "custom"
    with pytest.raises(ValueError, match="identifier"):
        to_spectrum_utils(_msn(native_id=None, scan_number=None))


def test_from_spectrum_utils_turns_nan_rt_into_none() -> None:
    source = MsmsSpectrum("id", 500.0, 2, np.array([100.0]), np.array([10.0]), np.nan)
    restored = from_spectrum_utils(source)
    assert restored.rt is None
