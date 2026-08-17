"""
Isotope-cluster helper functions shared by the deconvolution algorithm.

Numba is used automatically when installed (``pip install numba``).
Falls back to pure NumPy when not available.
"""

from __future__ import annotations

import numpy as np
import peptacular as pt
from numpy.typing import NDArray

try:
    from numba import njit as _njit

    _HAS_NUMBA = True
except ImportError:

    def _njit(*args, **kwargs):
        def _wrap(f):
            return f

        return _wrap

    _HAS_NUMBA = False

NEUTRON_MASS: float = pt.C13_NEUTRON_MASS
PROTON_MASS: float = pt.PROTON_MASS


@_njit(cache=True)
def _tol_da(mz: float, tolerance: float, is_ppm: bool) -> float:
    if is_ppm:
        return mz * tolerance / 1e6
    return tolerance


@_njit(cache=True)
def _best_available_peak(
    mz: NDArray[np.float64],
    intensity: NDArray[np.float64],
    ion_mobility: NDArray[np.float64],
    available: NDArray[np.bool_],
    target_mz: float,
    tolerance: float,
    is_ppm: bool,
    expected_intensity: float,
    max_fold_error: float,
    seed_im: float,
    use_im: bool,
    im_tolerance: float,
    im_is_relative: bool,
) -> int:
    """Best available peak by mass, abundance, and optional mobility.

    Returns ``-1`` when the m/z window is empty and ``-2`` when peaks occur in
    the window but all fail the abundance or ion-mobility gates.
    """
    tol = _tol_da(target_mz, tolerance, is_ppm)
    left = int(np.searchsorted(mz, target_mz - tol, side="left"))
    right = int(np.searchsorted(mz, target_mz + tol, side="right"))
    best_idx = -1
    best_score = np.inf
    saw_available = False
    fold_scale = np.log(max_fold_error) if max_fold_error > 1.0 else 1.0
    im_window = abs(seed_im) * im_tolerance if im_is_relative else im_tolerance

    for i in range(left, right):
        if not available[i]:
            continue
        saw_available = True
        distance = abs(float(mz[i]) - target_mz)
        if distance > tol:
            continue

        observed_intensity = float(intensity[i])
        if expected_intensity <= 0.0 or observed_intensity <= 0.0:
            if observed_intensity != expected_intensity:
                continue
            abundance_error = 0.0
        else:
            log_ratio = abs(np.log(observed_intensity / expected_intensity))
            if max_fold_error == 1.0:
                if log_ratio > 1e-12:
                    continue
                abundance_error = 0.0
            else:
                abundance_error = log_ratio / fold_scale
                if abundance_error > 1.0:
                    continue

        im_error = 0.0
        if use_im and np.isfinite(seed_im):
            candidate_im = float(ion_mobility[i])
            if not np.isfinite(candidate_im):
                continue
            im_delta = abs(candidate_im - seed_im)
            if im_window <= 0.0:
                if im_delta > 0.0:
                    continue
            else:
                im_error = im_delta / im_window
                if im_error > 1.0:
                    continue

        mass_error = distance / tol if tol > 0.0 else 0.0
        score = mass_error * mass_error + abundance_error * abundance_error + im_error * im_error
        if score < best_score or (score == best_score and observed_intensity > float(intensity[best_idx])):
            best_idx = i
            best_score = score

    if best_idx >= 0:
        return best_idx
    return -2 if saw_available else -1


@_njit(cache=True)
def _has_isotope_neighbor(
    mz: NDArray[np.float64],
    used: NDArray[np.bool_],
    seed_idx: int,
    charge: int,
    tolerance: float,
    is_ppm: bool,
    max_steps: int,
) -> bool:
    """Whether an unused peak occurs at a reachable isotope position."""
    seed_mz = float(mz[seed_idx])
    for step_index in range(1, max_steps + 1):
        offset = step_index * NEUTRON_MASS / charge
        for target_mz in (seed_mz - offset, seed_mz + offset):
            tol = _tol_da(target_mz, tolerance, is_ppm)
            left = int(np.searchsorted(mz, target_mz - tol, side="left"))
            right = int(np.searchsorted(mz, target_mz + tol, side="right"))
            for i in range(left, right):
                if i != seed_idx and not used[i]:
                    return True
    return False


@_njit(cache=True)
def _match_apex_cluster(
    mz: NDArray[np.float64],
    intensity: NDArray[np.float64],
    ion_mobility: NDArray[np.float64],
    used: NDArray[np.bool_],
    seed_idx: int,
    charge: int,
    tolerance: float,
    is_ppm: bool,
    relative_abundance: NDArray[np.float64],
    apex_index: int,
    min_relative_abundance: float,
    max_fold_error: float,
    max_gaps: int,
    use_im: bool,
    im_tolerance: float,
    im_is_relative: bool,
) -> tuple[int, float, float, NDArray[np.intp]]:
    """Match an isotope envelope outwards from its observed apex.

    Expansion in each direction stops independently when the theoretical
    abundance falls below the threshold, the allowed number of missing peaks
    is exceeded, or the nearest peak disagrees with the expected intensity by
    more than ``max_fold_error``. The blocking peak is not consumed.
    """
    indices = np.full(len(relative_abundance), -1, dtype=np.intp)
    indices[apex_index] = seed_idx
    available = ~used.copy()
    available[seed_idx] = False

    seed_intensity = float(intensity[seed_idx])
    seed_im = float(ion_mobility[seed_idx]) if use_im else np.nan
    total_intensity = seed_intensity
    n_peaks = 1

    for direction in (-1, 1):
        gaps = 0
        isotope_index = apex_index + direction
        while 0 <= isotope_index < len(relative_abundance):
            relative = float(relative_abundance[isotope_index])
            if relative < min_relative_abundance:
                break

            target_mz = float(mz[seed_idx]) + (isotope_index - apex_index) * NEUTRON_MASS / charge
            expected_intensity = seed_intensity * relative
            match_idx = _best_available_peak(
                mz,
                intensity,
                ion_mobility,
                available,
                target_mz,
                tolerance,
                is_ppm,
                expected_intensity,
                max_fold_error,
                seed_im,
                use_im,
                im_tolerance,
                im_is_relative,
            )
            if match_idx == -2:
                break
            if match_idx < 0:
                gaps += 1
                if gaps > max_gaps:
                    break
                isotope_index += direction
                continue

            observed_intensity = float(intensity[match_idx])

            indices[isotope_index] = match_idx
            available[match_idx] = False
            total_intensity += observed_intensity
            n_peaks += 1
            gaps = 0
            isotope_index += direction

    base_idx = int(indices[0])
    base_intensity = float(intensity[base_idx]) if base_idx >= 0 else 0.0
    return n_peaks, total_intensity, base_intensity, indices
