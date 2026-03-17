"""
Greedy isotope-cluster deconvolution implemented with pure NumPy.

Each peak in the input spectrum is assigned to exactly one isotope cluster.
Clusters with more than one peak are assigned the charge state that produces
the longest contiguous chain; singletons (no neighbours found at any charge)
are marked charge = -1 (unassigned).

Public entry point for a single spectrum:

    mz_out, charges_out, intensity_out = deconvolve_spectrum(
        mz, intensity, charge_range=(1, 5), tolerance=10.0, is_ppm=True
    )
"""

import numpy as np
import peptacular as pt
from numpy.typing import NDArray

NEUTRON_MASS: float = pt.C13_NEUTRON_MASS
PROTON_MASS: float = pt.PROTON_MASS


def _tol_da(mz: float, tolerance: float, is_ppm: bool) -> float:
    if is_ppm:
        return mz * tolerance / 1e6
    return tolerance


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

    # Local mask — prevents double-picking within one cluster trial
    available = ~used.copy()
    available[seed_idx] = False

    for _ in range(9):  # at most 10 peaks total
        next_mz = current_mz + step
        tol = _tol_da(next_mz, tolerance, is_ppm)

        dists = np.abs(mz - next_mz)
        candidates = available & (dists <= tol)

        if not np.any(candidates):
            break

        masked_dists = np.where(candidates, dists, np.inf)
        best_idx = int(np.argmin(masked_dists))

        indices[n_peaks] = best_idx
        total_intensity += float(intensity[best_idx])
        current_mz = float(mz[best_idx])
        n_peaks += 1
        available[best_idx] = False

    return n_peaks, total_intensity, base_intensity, indices


def _deconvolve_single(
    mz: NDArray[np.float32],
    intensity: NDArray[np.float32],
    min_charge: int,
    max_charge: int,
    tolerance: float,
    is_ppm: bool,
    max_dpeaks: int,
) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.float32], NDArray[np.float32], int]:
    """
    Deconvolute a single spectrum using greedy isotope clustering.

    Returns (out_mz, out_charges, out_total_int, out_base_int, n_out).
    - out_mz      : monoisotopic m/z of each cluster
    - out_charges : charge state (-1 for singletons)
    - out_total_int: summed cluster intensity
    - out_base_int : intensity of the seed (monoisotopic) peak
    """
    n = len(mz)
    used = np.zeros(n, dtype=np.bool_)

    out_mz = np.zeros(max_dpeaks, dtype=np.float32)
    out_charges = np.full(max_dpeaks, -1, dtype=np.int32)
    out_total_int = np.zeros(max_dpeaks, dtype=np.float32)
    out_base_int = np.zeros(max_dpeaks, dtype=np.float32)
    n_out = 0

    while n_out < max_dpeaks:
        # Seed = most intense unused peak
        masked_intensity = np.where(~used, intensity, -np.inf)
        seed_idx = int(np.argmax(masked_intensity))

        if used[seed_idx]:
            break  # all peaks consumed

        best_charge = min_charge
        best_n = 0
        best_total = 0.0
        best_base = float(intensity[seed_idx])
        best_indices = np.full(10, -1, dtype=np.intp)

        for charge in range(min_charge, max_charge + 1):
            n_peaks, total_intensity, base_intensity, indices = _find_isotope_cluster(
                mz, intensity, used, seed_idx, charge, tolerance, is_ppm
            )
            if n_peaks > best_n or (n_peaks == best_n and total_intensity > best_total):
                best_n = n_peaks
                best_total = total_intensity
                best_base = base_intensity
                best_charge = charge
                best_indices[:] = indices

        # Mark cluster peaks as used
        for ki in range(best_n):
            used[best_indices[ki]] = True

        out_mz[n_out] = mz[seed_idx]
        # Singletons (no isotope neighbours found) → unassigned
        out_charges[n_out] = best_charge if best_n > 1 else -1
        out_total_int[n_out] = best_total
        out_base_int[n_out] = best_base
        n_out += 1

    return out_mz, out_charges, out_total_int, out_base_int, n_out


def deconvolve_spectrum(
    mz: NDArray[np.float64],
    intensity: NDArray[np.float64],
    charge_range: tuple[int, int],
    tolerance: float,
    is_ppm: bool,
    max_dpeaks: int = 2000,
    intensity_mode: str = "total",
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.float64]]:
    """
    Deconvolute a single spectrum using greedy isotope clustering.

    Args:
        mz: m/z array (float64).
        intensity: Intensity array (float64), same length as mz.
        charge_range: (min_charge, max_charge) inclusive.
        tolerance: Peak matching tolerance value.
        is_ppm: If True, tolerance is in ppm; otherwise Da.
        max_dpeaks: Maximum number of output peaks.
        intensity_mode: ``"total"`` (sum of cluster) or ``"base"`` (monoisotopic peak).

    Returns:
        Tuple of (mz, charges, intensity) arrays sorted by m/z.
        Singletons have charge == -1.
    """
    if len(mz) == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, np.empty(0, dtype=np.int32), empty

    min_charge, max_charge = charge_range
    mz32 = mz.astype(np.float32)
    int32 = intensity.astype(np.float32)

    out_mz, out_charges, out_total, out_base, n_out = _deconvolve_single(
        mz32, int32, min_charge, max_charge, float(tolerance), bool(is_ppm), max_dpeaks
    )

    if n_out == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, np.empty(0, dtype=np.int32), empty

    out_int = out_base[:n_out] if intensity_mode == "base" else out_total[:n_out]

    order = np.argsort(out_mz[:n_out])
    return (
        out_mz[:n_out][order].astype(np.float64),
        out_charges[:n_out][order],
        out_int[order].astype(np.float64),
    )
