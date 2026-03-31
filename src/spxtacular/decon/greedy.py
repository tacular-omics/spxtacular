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
    mz: NDArray[np.float32],
    intensity: NDArray[np.float32],
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
    """
    indices = np.full(10, -1, dtype=np.intp)
    indices[0] = seed_idx
    n_peaks = 1
    base_intensity = float(intensity[seed_idx])
    total_intensity = base_intensity
    step = NEUTRON_MASS / charge
    current_mz = float(mz[seed_idx])

    available = ~used.copy()
    available[seed_idx] = False

    for _ in range(9):  # at most 10 peaks total
        next_mz = current_mz + step
        tol = _tol_da(next_mz, tolerance, is_ppm)

        dists = np.abs(mz - next_mz)
        candidates = available & (dists <= tol)

        if not np.any(candidates):
            break

        best_idx = int(np.argmin(np.where(candidates, dists, np.inf)))

        indices[n_peaks] = best_idx
        total_intensity += float(intensity[best_idx])
        current_mz = float(mz[best_idx])
        n_peaks += 1
        available[best_idx] = False

    return n_peaks, total_intensity, base_intensity, indices
