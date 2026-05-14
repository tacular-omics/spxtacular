import struct

import numpy as np
import pytest
from data import EXAMPLE_SPECTRUM

from spxtacular.compress import _encode_binary_payload
from spxtacular.core import Spectrum, SpectrumType


def test_spectrum_compression_roundtrip():
    spec = EXAMPLE_SPECTRUM

    compressed = spec.compress()
    spec_restored = Spectrum.from_compressed(compressed)

    np.testing.assert_allclose(spec_restored.mz, spec.mz)
    np.testing.assert_allclose(spec_restored.intensity, spec.intensity)


def test_spectrum_compression_no_charge_im():
    mz = np.array([100.0, 200.0], dtype=np.float64)
    intensity = np.array([1000.0, 5000.0], dtype=np.float64)

    spec = Spectrum(mz=mz, intensity=intensity)

    compressed = spec.compress()
    spec_restored = Spectrum.from_compressed(compressed)

    np.testing.assert_allclose(spec_restored.mz, spec.mz)
    np.testing.assert_allclose(spec_restored.intensity, spec.intensity)
    assert spec_restored.charge is None
    assert spec_restored.im is None


def test_spectrum_compression_charge_only():
    mz = np.array([100.0, 200.0], dtype=np.float64)
    intensity = np.array([1000.0, 5000.0], dtype=np.float64)
    charge = np.array([1, 2], dtype=np.int32)

    spec = Spectrum(mz=mz, intensity=intensity, charge=charge)

    compressed = spec.compress()
    spec_restored = Spectrum.from_compressed(compressed)

    np.testing.assert_allclose(spec_restored.mz, spec.mz)
    np.testing.assert_allclose(spec_restored.intensity, spec.intensity)
    np.testing.assert_array_equal(spec_restored.charge, spec.charge)
    assert spec_restored.im is None


def test_spectrum_compression_iso_score_roundtrip():
    mz = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    intensity = np.array([1000.0, 5000.0, 2500.0], dtype=np.float64)
    charge = np.array([1, 2, 1], dtype=np.int32)
    iso_score = np.array([0.95, 0.82, 0.71], dtype=np.float64)

    spec = Spectrum(mz=mz, intensity=intensity, charge=charge, iso_score=iso_score)
    spec_restored = Spectrum.from_compressed(spec.compress())

    np.testing.assert_allclose(spec_restored.mz, spec.mz)
    np.testing.assert_allclose(spec_restored.intensity, spec.intensity)
    np.testing.assert_array_equal(spec_restored.charge, spec.charge)
    assert spec_restored.iso_score is not None
    np.testing.assert_allclose(spec_restored.iso_score, iso_score, rtol=1e-5)


def test_binary_payload_backward_compatible_without_iso_score():
    """A payload without iso_score is byte-identical to the pre-iso_score wire
    format (mz + intensity + charge + im, always 4 chunks). The iso_score
    chunk is appended only when non-empty."""
    expected_4 = (
        struct.pack("!I", 4) + b"abcd" + struct.pack("!I", 4) + b"efgh" + struct.pack("!I", 0) + struct.pack("!I", 0)
    )
    assert _encode_binary_payload("abcd", "efgh", "", "", "") == expected_4
    assert _encode_binary_payload("abcd", "efgh", "", "") == expected_4

    # With iso_score, a 5th chunk is appended.
    with_iso = _encode_binary_payload("abcd", "efgh", "", "", "ij")
    assert with_iso == expected_4 + struct.pack("!I", 2) + b"ij"


def test_spectrum_compression_singleton_charge_roundtrip():
    """Regression test for the singleton compress bug.

    Deconvoluted spectra carry ``charge == -1`` for unassigned singleton peaks
    (per CLAUDE.md). The wire format reserves hex ``'f'`` as the singleton
    sentinel so these round-trip cleanly.
    """
    spec = Spectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype=np.float64),
        intensity=np.array([1e3, 2e3, 3e3], dtype=np.float64),
        charge=np.array([2, -1, 3], dtype=np.int32),
        spectrum_type=SpectrumType.DECONVOLUTED,
    )
    restored = Spectrum.from_compressed(spec.compress())
    np.testing.assert_array_equal(restored.charge, spec.charge)


def test_spectrum_compression_rejects_charge_above_14():
    """Charge 15 collides with the singleton sentinel and is rejected."""
    spec = Spectrum(
        mz=np.array([100.0], dtype=np.float64),
        intensity=np.array([1e3], dtype=np.float64),
        charge=np.array([15], dtype=np.int32),
        spectrum_type=SpectrumType.DECONVOLUTED,
    )
    with pytest.raises(ValueError, match="out of range"):
        spec.compress()
