import numpy as np
import pytest

from spxtacular.core import Spectrum


def test_deconvolution_integration():
    # Create a synthetic isotopic pattern for a peptide
    # Monoisotopic peak at 1000.0, charge 2 -> z=2, M=2000 approx.
    # Isotopes at 1000.5, 1001.0, 1001.5
    mz = np.array([1000.0, 1000.5, 1001.0, 1001.5, 500.0, 1200.0], dtype=np.float64)
    # Descending intensities roughly mocking an isotopic envelope
    intensity = np.array([10000.0, 8000.0, 4000.0, 1000.0, 500.0, 500.0], dtype=np.float64)

    spec = Spectrum(mz=mz, intensity=intensity)

    # Run deconvolution
    # This should identify the group at 1000.0 as charge 2
    decon_spec = spec.deconvolute(charge_range=(1, 3), tolerance=10, tolerance_type="ppm")

    assert decon_spec.charge is not None

    # Check indices corresponding to the isotope cluster
    # 1000.0 -> index 0
    # 1000.5 -> index 1
    # 1001.0 -> index 2
    # 1001.5 -> index 3

    print(f"Charges: {decon_spec.charge}")

    # We expect charge 2 for the cluster
    assert decon_spec.charge[0] == 2
    assert decon_spec.charge[1] == 2
    assert decon_spec.charge[2] == 2
    assert decon_spec.charge[3] == 2

    # Noise peaks might be 0 or None (represented as 0 in int array)
    # The peaks at 500.0 (index 4) and 1200.0 (index 5) are likely singletons, charge 0/None
    assert decon_spec.charge[4] == 0 or decon_spec.charge[4] is None
    assert decon_spec.charge[5] == 0 or decon_spec.charge[5] is None
