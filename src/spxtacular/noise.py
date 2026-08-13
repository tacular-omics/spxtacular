from typing import Literal

import numpy as np


def _estimate_noise_mad(intensity_array: np.ndarray) -> float:
    """Estimate noise using Median Absolute Deviation."""
    median = np.median(intensity_array)
    mad = np.median(np.abs(intensity_array - median))
    # Noise threshold: median + k * MAD (k=3 to 5 is typical)
    return float(median + 3 * 1.4826 * mad)  # 1.4826 makes MAD consistent with std


def _estimate_noise_histogram(intensity_array: np.ndarray) -> float:
    """Estimate noise using histogram mode.

    The histogram is built over the low-intensity bulk rather than the full
    dynamic range. Real spectra span several orders of magnitude, so binning
    everything puts every noise peak in bin 0: the mode is then one bin wide,
    the "FWHM" degenerates to that bin width, and the estimate comes out
    roughly two orders of magnitude too high.
    """
    cutoff = np.median(intensity_array)
    low = intensity_array[intensity_array <= cutoff]
    if len(low) < 2:
        low = intensity_array

    hist, bin_edges = np.histogram(low, bins=100)
    noise_bin_idx = int(np.argmax(hist))
    noise_mode = (bin_edges[noise_bin_idx] + bin_edges[noise_bin_idx + 1]) / 2

    # Estimate std of noise from histogram width
    half_max = hist[noise_bin_idx] / 2
    left_idx = noise_bin_idx
    while left_idx > 0 and hist[left_idx] > half_max:
        left_idx -= 1
    right_idx = noise_bin_idx
    while right_idx < len(hist) - 1 and hist[right_idx] > half_max:
        right_idx += 1

    noise_std = (bin_edges[right_idx] - bin_edges[left_idx]) / 2.355  # FWHM to std
    return float(noise_mode + 3 * noise_std)


def _estimate_noise_baseline(intensity_array: np.ndarray) -> float:
    """Estimate noise using bottom quantile statistics."""
    bottom_25 = np.percentile(intensity_array, 25)
    noise_intensities = intensity_array[intensity_array <= bottom_25]
    noise_mean = np.mean(noise_intensities)
    noise_std = np.std(noise_intensities)
    return float(noise_mean + 3 * noise_std)


_ITERATIVE_MEDIAN_MIN_SAMPLES: int = 100
"""Stop iterating early when the surviving sample count drops below this.

Three rounds of median + 2·MAD clipping on a small array can shrink the
distribution to a handful of values, at which point further trimming is
unstable (the median/MAD estimates become dominated by sampling noise).
``100`` is empirically large enough to keep the final std meaningful.
"""


def _estimate_noise_iterative_median(intensity_array: np.ndarray) -> float:
    """Estimate noise using iterative median filtering.

    Three passes of ``median + 2 * 1.4826 * MAD`` clipping, stopping early
    when fewer than ``_ITERATIVE_MEDIAN_MIN_SAMPLES`` samples remain.
    """
    current = intensity_array.copy()
    for _ in range(3):
        median = np.median(current)
        mad = np.median(np.abs(current - median))
        threshold = median + 2 * 1.4826 * mad
        current = current[current <= threshold]
        if len(current) < _ITERATIVE_MEDIAN_MIN_SAMPLES:
            break
    return float(np.median(current) + 3 * np.std(current))


def estimate_noise_level(
    intensity_array: np.ndarray,
    method: Literal["mad", "percentile", "histogram", "baseline", "iterative_median"] | float | int = "mad",
) -> float:
    """
    Estimate noise level using various methods.

    Parameters:
    -----------
    intensity_array : np.ndarray
        Array of intensity values
    method : str or float or int
        Method to use: one of 'mad', 'percentile', 'histogram', 'baseline', 'iterative_median',
        or a numeric value (float/int) to be used directly as the noise level.

    Returns:
    --------
    float : Estimated noise level threshold

    Notes:
    ------
    ``'mad'``, ``'histogram'``, ``'baseline'`` and ``'iterative_median'`` all
    return a *threshold* of the form ``centre + 3 * spread``.  ``'percentile'``
    is different: it returns the raw 5th percentile of the intensities, which
    is a level rather than a threshold and is therefore substantially lower.

    An empty intensity array returns ``0.0`` for every method.

    A ``bool`` is rejected with ``ValueError`` rather than taken for the numeric
    threshold ``bool``'s ``int`` ancestry would otherwise make it.
    """
    # ``bool`` is a subclass of ``int``, so it would otherwise be taken for a
    # numeric threshold: ``estimate_noise_level(arr, True)`` is a mistake, not a
    # threshold of 1.0. Rejected before the empty-array shortcut so the answer
    # does not depend on the input length.
    if isinstance(method, bool):
        msg = f"Unknown method: {method}. A bool is not a noise threshold."
        raise ValueError(msg)

    # If a numeric value is provided, use it directly as the noise level.
    if isinstance(method, (int, float)):
        return float(method)

    if len(intensity_array) == 0:
        # Every estimator below reduces over the array; on an empty one they
        # variously return NaN with RuntimeWarnings or raise IndexError.
        return 0.0

    if method == "mad":
        return _estimate_noise_mad(intensity_array)
    elif method == "percentile":
        return float(np.percentile(intensity_array, 5))
    elif method == "histogram":
        return _estimate_noise_histogram(intensity_array)
    elif method == "baseline":
        return _estimate_noise_baseline(intensity_array)
    elif method == "iterative_median":
        return _estimate_noise_iterative_median(intensity_array)
    else:
        msg = f"Unknown method: {method}"
        raise ValueError(msg)
