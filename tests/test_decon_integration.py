import numpy as np
import pytest

from spxtacular import PEPTIDE_ISOTOPE_MODEL
from spxtacular.core import Spectrum
from spxtacular.decon.greedy import NEUTRON_MASS, PROTON_MASS


def _charge_2_envelope() -> tuple[np.ndarray, np.ndarray, float]:
    neutral_mass = 2000.0
    charge = 2
    distribution = PEPTIDE_ISOTOPE_MODEL.adaptive_distribution(neutral_mass)
    relative = distribution / distribution.max()
    offsets = np.flatnonzero(relative >= 0.01)
    mono_mz = (neutral_mass + charge * PROTON_MASS) / charge
    mz = mono_mz + offsets * NEUTRON_MASS / charge
    intensity = distribution[offsets] * 1e6
    return mz, intensity, neutral_mass


def test_deconvolution_identifies_charge_2_cluster():
    """Greedy deconvolution should group the z=2 isotope envelope and mark singletons."""
    envelope_mz, envelope_intensity, neutral_mass = _charge_2_envelope()
    mz = np.concatenate((envelope_mz, [500.0, 1200.0]))
    intensity = np.concatenate((envelope_intensity, [500.0, 500.0]))

    spec = Spectrum(mz=mz, intensity=intensity)
    decon = spec.deconvolute(charge_range=(1, 3), tolerance=10, tolerance_type="ppm")

    assert decon.charge is not None
    assert len(decon.mz) == 3  # one cluster representative + two singletons

    # Output is sorted by m/z: singleton, inferred monoisotopic m/z, singleton.
    assert decon.mz[0] == pytest.approx(500.0)
    assert decon.charge[0] == -1  # singleton

    expected_mono_mz = (neutral_mass + 2 * PROTON_MASS) / 2
    assert decon.mz[1] == pytest.approx(expected_mono_mz)
    assert decon.charge[1] == 2  # charge-2 isotope cluster

    assert decon.mz[2] == pytest.approx(1200.0)
    assert decon.charge[2] == -1  # singleton


def test_deconvolution_then_decharge():
    """decharge() should convert the charge-2 cluster peak to its neutral mass."""
    envelope_mz, envelope_intensity, neutral_mass = _charge_2_envelope()
    mz = np.concatenate((envelope_mz, [500.0, 1200.0]))
    intensity = np.concatenate((envelope_intensity, [500.0, 500.0]))

    spec = Spectrum(mz=mz, intensity=intensity)
    decon = spec.deconvolute(charge_range=(1, 3), tolerance=10, tolerance_type="ppm")
    decharged = decon.decharge()

    # Only the charge-2 peak survives decharge (singletons with charge=-1 are dropped)
    assert len(decharged.mz) == 1
    assert decharged.mz[0] == pytest.approx(neutral_mass, abs=0.01)
