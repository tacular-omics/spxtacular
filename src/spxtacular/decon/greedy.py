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
def _find_isotope_cluster(
    mz: NDArray[np.float64],
    intensity: NDArray[np.float64],
    used: NDArray[np.bool_],
    seed_idx: int,
    charge: int,
    tolerance: float,
    is_ppm: bool,
) -> tuple[int, float, float, NDArray[np.intp]]:
    """
    Greedily extend an isotope cluster forward from seed_idx.

    Returns (n_peaks, total_intensity, base_intensity, indices).
    - indices is length-10, padded with -1.
    - Does NOT modify *used*; uses a local copy internally.

    Each step is measured from the *expected* position ``mz[seed] + k * step``
    rather than from the previously matched peak, so a chain of peaks each
    just inside tolerance cannot ratchet the cluster away from the seed.
    """
    indices = np.full(10, -1, dtype=np.intp)
    indices[0] = seed_idx
    n_peaks = 1
    base_intensity = float(intensity[seed_idx])
    total_intensity = base_intensity
    step = NEUTRON_MASS / charge
    seed_mz = float(mz[seed_idx])

    available = ~used.copy()
    available[seed_idx] = False

    for k in range(1, 10):  # at most 10 peaks total
        next_mz = seed_mz + k * step
        tol = _tol_da(next_mz, tolerance, is_ppm)

        dists = np.abs(mz - next_mz)
        candidates = available & (dists <= tol)

        if not np.any(candidates):
            break

        best_idx = int(np.argmin(np.where(candidates, dists, np.inf)))

        indices[n_peaks] = best_idx
        total_intensity += float(intensity[best_idx])
        n_peaks += 1
        available[best_idx] = False

    return n_peaks, total_intensity, base_intensity, indices


@_njit(cache=True)
def _find_anchor_candidates(
    mz: NDArray[np.float64],
    used: NDArray[np.bool_],
    seed_idx: int,
    charge: int,
    tolerance: float,
    is_ppm: bool,
    max_back: int,
) -> tuple[NDArray[np.intp], int]:
    """
    Walk *backwards* from seed_idx looking for candidate monoisotopic peaks.

    The seed is the most intense unused peak, but for peptides above roughly
    1900 Da the most intense isotope peak is A+1 (or A+2 above ~3500 Da), so
    the seed is frequently *not* the monoisotopic peak.  This returns every
    peak reachable by stepping backwards in ``NEUTRON_MASS / charge``
    increments, so the caller can score a cluster anchored at each candidate
    and pick the alignment that actually fits the isotope template.

    Returns (anchors, n_anchors); ``anchors[0]`` is always seed_idx and the
    array is padded with -1.
    """
    anchors = np.full(max_back + 1, -1, dtype=np.intp)
    anchors[0] = seed_idx
    n_anchors = 1
    step = NEUTRON_MASS / charge
    seed_mz = float(mz[seed_idx])

    for k in range(1, max_back + 1):
        prev_mz = seed_mz - k * step
        if prev_mz <= 0.0:
            break
        tol = _tol_da(prev_mz, tolerance, is_ppm)

        dists = np.abs(mz - prev_mz)
        candidates = (~used) & (dists <= tol)

        if not np.any(candidates):
            break

        best_idx = int(np.argmin(np.where(candidates, dists, np.inf)))
        anchors[n_anchors] = best_idx
        n_anchors += 1

    return anchors, n_anchors
