import numpy as np
import pytest

from spxtacular.core import Spectrum


def test_denoise_mad():
    # Create a spectrum with some noise and a signal
    # Signal: 100, 200
    # Noise: around 1-5
    np.random.seed(42)
    mz = np.linspace(100, 200, 100)
    noise = np.random.normal(0, 1, 100) + 2  # Mean 2, std 1
    intensity = noise.copy()

    # Add peaks
    intensity[25] = 100
    intensity[75] = 200

    spec = Spectrum(mz=mz, intensity=intensity)

    # Denoise
    # MAD should estimate noise level around 2 + 3*1.48*std
    # std is roughly 1. So threshold roughly 5-6.
    denoised_spec = spec.denoise(method="mad")

    assert len(denoised_spec.mz) < len(spec.mz)
    assert 25 in np.where(denoised_spec.intensity == 100)[0] or 100 in denoised_spec.intensity
    assert 200 in denoised_spec.intensity


def test_denoise_fixed_value():
    mz = np.array([100.0, 101.0, 102.0, 103.0])
    intensity = np.array([10.0, 50.0, 5.0, 100.0])
    spec = Spectrum(mz=mz, intensity=intensity)

    # Threshold 20
    denoised = spec.denoise(method=20.0)

    assert len(denoised.mz) == 2
    assert np.allclose(denoised.intensity, [50.0, 100.0])


if __name__ == "__main__":
    test_denoise_fixed_value()
    print("Fixed value test passed")
    test_denoise_mad()
    print("MAD test passed")
