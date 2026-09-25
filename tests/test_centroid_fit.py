"""Regression tests for the three-point apex fit in ``Spectrum.centroid``.

The log-parabola (Gaussian) fit is exact for well-sampled Gaussian peaks but is
ill-conditioned when a flank is tiny relative to the apex: ``log`` of a near-zero
flank dominates the curvature and the fitted height explodes (1e-30 flanks gave
~5e3x the apex, 1e-300 gave ~1e36x). Zero or negative flanks made the peak
disappear altogether.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from spxtacular import Spectrum
from spxtacular.core import _centroid_peaks

STEP = 0.01
MZ3 = np.array([100.0, 100.0 + STEP, 100.0 + 2 * STEP])


def _assert_sane(mz: np.ndarray, intensity: np.ndarray, centers: np.ndarray, heights: np.ndarray) -> None:
    assert np.all(np.isfinite(centers))
    assert np.all(np.isfinite(heights))
    top = np.nanmax(intensity)
    # Every fitted height is at least a local maximum and never more than twice
    # the largest raw sample (a Gaussian sampled at one point per FWHM gains 2x).
    assert np.all(heights <= 2.0 * top * (1 + 1e-12))
    assert np.all(heights > 0)
    assert np.all(centers >= mz[0]) and np.all(centers <= mz[-1])


@pytest.mark.parametrize("flank", [1e-30, 1e-300, 5e-324, 1e-12, 1e-6])
def test_near_zero_symmetric_flanks_stay_at_apex(flank: float) -> None:
    intensity = np.array([flank, 1.0, flank])
    centers, heights, _ = _centroid_peaks(MZ3, intensity)
    assert centers.size == 1
    assert centers[0] == pytest.approx(MZ3[1], abs=1e-12)
    assert heights[0] == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("tiny", [1e-30, 1e-300, 5e-324, 0.0, -1.0, -1e6])
@pytest.mark.parametrize("other", [0.9, 0.5, 0.999999])
@pytest.mark.parametrize("left_tiny", [True, False])
def test_one_sided_tiny_flank_does_not_explode(tiny: float, other: float, left_tiny: bool) -> None:
    intensity = np.array([tiny, 1.0, other] if left_tiny else [other, 1.0, tiny])
    centers, heights, _ = _centroid_peaks(MZ3, intensity)
    assert centers.size == 1
    _assert_sane(MZ3, intensity, centers, heights)
    # The centre moves toward the larger flank, never past it.
    if left_tiny:
        assert MZ3[1] <= centers[0] <= MZ3[2]
    else:
        assert MZ3[0] <= centers[0] <= MZ3[1]


@pytest.mark.parametrize("flank", [0.0, -3.0])
def test_zero_or_negative_flanks_keep_the_peak(flank: float) -> None:
    intensity = np.array([flank, 7.0, flank])
    spec = Spectrum(mz=MZ3, intensity=intensity, spectrum_type="profile").centroid()
    np.testing.assert_allclose(spec.mz, [MZ3[1]], atol=1e-12, rtol=0)
    np.testing.assert_allclose(spec.intensity, [7.0], rtol=1e-12)


def test_isolated_spikes_in_zero_padded_profile_are_kept() -> None:
    # Zero-padded profile data (common in vendor exports) with single-sample spikes.
    intensity = np.array([0.0, 0.0, 4.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0])
    mz = 500.0 + np.arange(intensity.size) * STEP
    centers, heights, _ = _centroid_peaks(mz, intensity)
    assert centers.size == 2
    _assert_sane(mz, intensity, centers, heights)
    assert centers[0] == pytest.approx(mz[2], abs=1e-12)
    assert heights[0] == pytest.approx(4.0)
    assert mz[5] <= centers[1] <= mz[6]


@pytest.mark.parametrize("flank", [1e-30, 0.0, 0.5])
@pytest.mark.parametrize("width", [2, 3])
def test_flat_top_with_tiny_flanks_uses_plateau(flank: float, width: int) -> None:
    intensity = np.array([flank] + [3.0] * width + [flank])
    mz = 200.0 + np.arange(intensity.size) * STEP
    centers, heights, _ = _centroid_peaks(mz, intensity)
    np.testing.assert_allclose(centers, [(mz[1] + mz[width]) / 2], atol=1e-12)
    np.testing.assert_array_equal(heights, [3.0])


finite_intensity = st.one_of(
    st.floats(-1e3, 1e6, allow_nan=False, allow_infinity=False),
    st.sampled_from([0.0, 1e-300, 5e-324, 1e-30, 1e-12]),
)


@given(
    intensity=st.lists(finite_intensity, min_size=3, max_size=30),
    gaps=st.lists(st.floats(0.2, 2.0), min_size=29, max_size=29),
)
def test_fitted_peaks_are_bounded_and_inside_their_window(intensity: list[float], gaps: list[float]) -> None:
    y = np.asarray(intensity, dtype=np.float64)
    # Neighbouring gaps within a 10x ratio, the regime in which the fit is used.
    mz = 300.0 + np.r_[0.0, np.cumsum(np.asarray(gaps[: y.size - 1]) * STEP)]
    centers, heights, _ = _centroid_peaks(mz, y)
    assert np.all(np.isfinite(centers)) and np.all(np.isfinite(heights))
    for c, h in zip(centers, heights, strict=True):
        # Each centroid sits in the span of some local maximum's neighbours, and
        # its height is bounded by that window's raw maximum.
        i = int(np.searchsorted(mz, c))
        lo, hi = max(i - 2, 0), min(i + 2, y.size)
        window = y[lo:hi]
        assert h <= 2.0 * window.max() * (1 + 1e-12)
        assert h >= 0.0
        assert mz[0] < c < mz[-1]


# ---------------------------------------------------------------------------
# Well-formed peaks must fit exactly as before the fix.
# ---------------------------------------------------------------------------


def _pre_fix_centroid_peaks(mz: np.ndarray, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frozen copy of the 0.9 fit (commit 0bf7d4e), for well-conditioned data only."""
    starts = np.r_[0, np.flatnonzero(intensity[1:] != intensity[:-1]) + 1]
    values = intensity[starts]
    runs = np.flatnonzero((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])) + 1
    lo = starts[runs]
    hi = starts[runs + 1] - 1
    keep = np.isfinite(values[runs]) & (values[runs] > 0)
    lo, hi = lo[keep], hi[keep]
    apex = (lo + hi) // 2
    centers = (mz[lo] + mz[hi]) / 2.0
    heights = intensity[apex].copy()
    sharp = lo == hi
    fit = sharp & (intensity[lo - 1] > 0) & (intensity[hi + 1] > 0)
    valid = ~sharp | fit
    idx = np.flatnonzero(fit)
    x = mz[apex[idx]]
    left = mz[lo[idx] - 1] - x
    right = mz[hi[idx] + 1] - x
    regular = (left < 0) & (right > 0) & (-left <= right * 10) & (right <= -left * 10)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_apex = np.log(heights[idx])
        ls = (np.log(intensity[lo[idx] - 1]) - log_apex) / left
        rs = (np.log(intensity[hi[idx] + 1]) - log_apex) / right
        q = (rs - ls) / (right - left)
        lin = ls - q * left
        off = -lin / (2 * q)
        h = np.exp(log_apex - lin * lin / (4 * q))
    centers[idx] = x + off
    heights[idx] = h
    valid[idx] = regular & (q < 0) & (off >= left) & (off <= right)
    valid &= np.isfinite(centers) & np.isfinite(heights)
    return centers[valid], heights[valid]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_normal_profile_matches_pre_fix_output(seed: int) -> None:
    rng = np.random.default_rng(seed)
    mz = np.arange(400.0, 410.0, 0.002)
    intensity = 50.0 + rng.uniform(0.0, 20.0, mz.size)  # positive noise floor
    for center in rng.uniform(401.0, 409.0, 12):
        sigma = rng.uniform(0.003, 0.01)
        intensity += rng.uniform(1e3, 1e6) * np.exp(-0.5 * ((mz - center) / sigma) ** 2)
    expected_c, expected_h = _pre_fix_centroid_peaks(mz, intensity)
    centers, heights, _ = _centroid_peaks(mz, intensity)
    np.testing.assert_array_equal(centers, expected_c)
    np.testing.assert_array_equal(heights, expected_h)
