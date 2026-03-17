import numpy as np
import pytest

from spxtacular.core import Spectrum


def test_deconvolution_identifies_charge_2_cluster():
    """Greedy deconvolution should group the z=2 isotope envelope and mark singletons."""
    # Isotopic envelope at charge 2: spacing = NEUTRON_MASS/2 ≈ 0.5017 Da
    # 1000.0, 1000.5, 1001.0, 1001.5 are within 10 ppm of expected positions
    mz = np.array([1000.0, 1000.5, 1001.0, 1001.5, 500.0, 1200.0], dtype=np.float64)
    intensity = np.array([10000.0, 8000.0, 4000.0, 1000.0, 500.0, 500.0], dtype=np.float64)

    spec = Spectrum(mz=mz, intensity=intensity)
    decon = spec.deconvolute(charge_range=(1, 3), tolerance=10, tolerance_type="ppm")

    assert decon.charge is not None
    assert len(decon.mz) == 3  # one cluster representative + two singletons

    # Output is sorted by m/z: [500.0, 1000.0, 1200.0]
    assert decon.mz[0] == pytest.approx(500.0)
    assert decon.charge[0] == -1   # singleton

    assert decon.mz[1] == pytest.approx(1000.0)
    assert decon.charge[1] == 2    # charge-2 isotope cluster

    assert decon.mz[2] == pytest.approx(1200.0)
    assert decon.charge[2] == -1   # singleton


def test_deconvolution_then_decharge():
    """decharge() should convert the charge-2 cluster peak to its neutral mass."""
    mz = np.array([1000.0, 1000.5, 1001.0, 1001.5, 500.0, 1200.0], dtype=np.float64)
    intensity = np.array([10000.0, 8000.0, 4000.0, 1000.0, 500.0, 500.0], dtype=np.float64)

    spec = Spectrum(mz=mz, intensity=intensity)
    decon = spec.deconvolute(charge_range=(1, 3), tolerance=10, tolerance_type="ppm")
    decharged = decon.decharge()

    # Only the charge-2 peak survives decharge (singletons with charge=-1 are dropped)
    assert len(decharged.mz) == 1
    expected_neutral_mass = 1000.0 * 2 - 2 * 1.007276
    assert decharged.mz[0] == pytest.approx(expected_neutral_mass, abs=0.01)
