from tkinter import E

import numpy as np
import pytest
from data import EXAMPLE_SPECTRUM

from spxtacular.core import Spectrum


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
    assert spec_restored.ion_mobility is None


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
    assert spec_restored.ion_mobility is None
