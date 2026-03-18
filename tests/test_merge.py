import numpy as np

from spxtacular.core import Spectrum


def test_merge_peaks_basic():
    mz = np.array([100.0, 100.01, 200.0, 300.0])
    intensity = np.array([100.0, 100.0, 50.0, 20.0])

    spec = Spectrum(mz=mz, intensity=intensity)
    merged = spec.merge(mz_tolerance=0.02, mz_tolerance_type="da")

    assert len(merged) == 3
    # 100.0 and 100.01 should merge
    assert np.isclose(merged.mz[0], 100.005)
    assert np.isclose(merged.intensity[0], 200.0)

    assert np.isclose(merged.mz[1], 200.0)
    assert np.isclose(merged.intensity[1], 50.0)


def test_merge_peaks_ppm():
    mz = np.array([100.0, 100.0005, 200.0])  # 5 ppm diff at 100 is 0.0005
    intensity = np.array([100.0, 100.0, 50.0])

    spec = Spectrum(mz=mz, intensity=intensity)
    merged = spec.merge(mz_tolerance=10, mz_tolerance_type="ppm")

    assert len(merged) == 2
    assert np.isclose(merged.mz[0], 100.00025)
    assert np.isclose(merged.intensity[0], 200.0)


def test_merge_peaks_ion_mobility():
    mz = np.array([100.0, 100.01, 200.0])
    intensity = np.array([100.0, 100.0, 50.0])
    im = np.array([1.0, 1.2, 2.0])

    spec = Spectrum(mz=mz, intensity=intensity, im=im)
    merged = spec.merge(mz_tolerance=0.02, mz_tolerance_type="da", im_tolerance=0.3, im_tolerance_type="absolute")

    assert len(merged) == 2
    assert merged.im is not None
    assert np.isclose(merged.im[0], 1.1)


def test_merge_peaks_inplace():
    mz = np.array([100.0, 100.01, 200.0])
    intensity = np.array([100.0, 100.0, 50.0])

    spec = Spectrum(mz=mz, intensity=intensity)
    spec.merge(mz_tolerance=0.02, mz_tolerance_type="da", inplace=True)
    assert len(spec) == 2
    assert np.isclose(spec.mz[0], 100.005)


def test_merge_peaks_charge_separation():
    # Peaks close in m/z but different charges should NOT merge
    mz = np.array([100.0, 100.01, 200.0])
    intensity = np.array([100.0, 100.0, 50.0])
    charge = np.array([1, 2, 1])

    spec = Spectrum(mz=mz, intensity=intensity, charge=charge, spectrum_type="deconvoluted")
    merged = spec.merge(mz_tolerance=0.02, mz_tolerance_type="da")

    # Expect 3 peaks because 100.0 (z=1) and 100.01 (z=2) shouldn't merge
    # m/z 100.0 (z=1) -> 100.0, charge 1
    # m/z 100.01 (z=2) -> 100.01, charge 2
    # m/z 200.0 (z=1) -> 200.0, charge 1

    assert len(merged) == 3
    # The output is sorted by mz, so we expect [100.0, 100.01, 200.0]
    expected_charges = np.array([1, 2, 1])
    assert np.array_equal(merged.charge, expected_charges)


def test_merge_peaks_charge_merge():
    # Peaks close in m/z AND same charge SHOULD merge
    mz = np.array([100.0, 100.01, 200.0])
    intensity = np.array([100.0, 100.0, 50.0])
    charge = np.array([2, 2, 1])

    spec = Spectrum(mz=mz, intensity=intensity, charge=charge, spectrum_type="deconvoluted")
    merged = spec.merge(mz_tolerance=0.02, mz_tolerance_type="da")

    # Expect 2 peaks, 100.0/100.01 merged
    assert len(merged) == 2
    assert merged.charge[0] == 2
    assert merged.charge[1] == 1


def test_merge_im_tolerance_relative():
    # Test Ion Mobility tolerance merging (relative mode)
    # mz all very close. im separates.
    # relative tolerance: 0.06
    # im: 1.0 (base). 1.05 (diff=0.05, within 0.06). 1.2 (diff=0.2, >0.06).
    mz = np.array([100.0, 100.01, 100.02, 200.0])
    intensity = np.array([100.0, 50.0, 25.0, 10.0])
    im = np.array([1.0, 1.05, 1.2, 2.0])

    spec = Spectrum(mz=mz, intensity=intensity, im=im)

    merged = spec.merge(mz_tolerance=0.1, mz_tolerance_type="da", im_tolerance=0.06, im_tolerance_type="relative")

    # Expect:
    # 100.0 and 100.01 merged
    # 100.02 separate
    # 200.0 separate

    # Use approximate comparisons because sorting on nearly identical m/z is tricky
    assert len(merged) == 3

    # Check values
    all_mz = np.sort(merged.mz)
    all_im = np.sort(merged.im)

    # Expected mzs:
    # 1. (100.0*100 + 100.01*50)/150 = 100.00333...
    # 2. 100.02
    # 3. 200.0
    expected_mzs = np.sort([100.003333333, 100.02, 200.0])
    assert np.allclose(all_mz, expected_mzs)

    # Expected ims:
    # 1. (1.0*100 + 1.05*50)/150 = 1.01666...
    # 2. 1.2
    # 3. 2.0
    expected_ims = np.sort([1.016666667, 1.2, 2.0])
    assert np.allclose(all_im, expected_ims)


def test_merge_im_tolerance_absolute():
    # Test Ion Mobility tolerance merging (absolute mode)
    # mz identical. im separates.
    # abs tol 0.05.
    mz = np.array([100.0, 100.0, 100.0])
    intensity = np.array([100.0, 50.0, 25.0])
    im = np.array([1.0, 1.04, 1.1])

    spec = Spectrum(mz=mz, intensity=intensity, im=im)
    merged = spec.merge(mz_tolerance=0.1, im_tolerance=0.05, im_tolerance_type="absolute")

    assert len(merged) == 2
    ims = np.sort(merged.im)
    expected_ims = np.sort([(100 * 1.0 + 50 * 1.04) / 150, 1.1])
    assert np.allclose(ims, expected_ims)


def test_merge_zero_intensity():
    # Test merging peaks with zero intensity
    mz = np.array([100.0, 100.01])
    intensity = np.array([0.0, 0.0])

    spec = Spectrum(mz=mz, intensity=intensity)
    merged = spec.merge(mz_tolerance=0.1, mz_tolerance_type="Da")

    assert len(merged) == 1
    assert merged.intensity[0] == 0.0
    assert np.isclose(merged.mz[0], 100.005)  # Simple mean since weights sum to zero
