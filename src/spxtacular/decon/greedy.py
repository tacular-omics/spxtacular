"""
Greedy isotope-cluster deconvolution implemented with pure NumPy.

Mirrors the logic of the Numba reference implementation but uses vectorised
array operations instead of compiled loops wherever possible.
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
) -> tuple[int, float, NDArray[np.intp]]:
    """
    Starting from seed_idx, greedily extend an isotope cluster forward.

    Returns (n_peaks, total_intensity, indices) where indices is a length-10
    array padded with -1.  Does NOT modify *used* — uses a local copy so
    multiple charge-state trials from the same caller are independent.
    """
    indices = np.full(10, -1, dtype=np.intp)
    indices[0] = seed_idx
    n_peaks = 1
    total_intensity = float(intensity[seed_idx])
    step = NEUTRON_MASS / charge
    current_mz = float(mz[seed_idx])

    # Local availability mask: avoids picking the same peak twice inside
    # one cluster while leaving the caller's *used* array untouched.
    available = ~used.copy()
    available[seed_idx] = False  # seed is already in the cluster

    for _ in range(9):  # at most 10 peaks total
        next_mz = current_mz + step
        tol = _tol_da(next_mz, tolerance, is_ppm)

        dists = np.abs(mz - next_mz)
        candidates = available & (dists <= tol)

        if not np.any(candidates):
            break  # no match — no skips allowed

        # pick the closest candidate
        masked_dists = np.where(candidates, dists, np.inf)
        best_idx = int(np.argmin(masked_dists))

        indices[n_peaks] = best_idx
        total_intensity += float(intensity[best_idx])
        current_mz = float(mz[best_idx])
        n_peaks += 1
        available[best_idx] = False

    return n_peaks, total_intensity, indices


def _decharge_single(
    mz: NDArray[np.float32],
    intensity: NDArray[np.float32],
    min_charge: int,
    max_charge: int,
    tolerance: float,
    is_ppm: bool,
    max_dpeaks: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], int]:
    """
    Deconvolute a single spectrum using greedy isotope clustering.

    Returns (out_mz, out_intensity, n_out).
    """
    n = len(mz)
    used = np.zeros(n, dtype=np.bool_)
    out_mz = np.zeros(max_dpeaks, dtype=np.float32)
    out_int = np.zeros(max_dpeaks, dtype=np.float32)
    n_out = 0

    while n_out < max_dpeaks:
        # find the most intense unused peak as seed
        masked_intensity = np.where(~used, intensity, -np.inf)
        seed_idx = int(np.argmax(masked_intensity))

        if used[seed_idx]:
            break  # all peaks consumed

        # try each charge state; prefer more peaks, break ties by intensity
        best_charge = min_charge
        best_n = 0
        best_intensity = 0.0
        best_indices = np.full(10, -1, dtype=np.intp)

        for charge in range(min_charge, max_charge + 1):
            n_peaks, total_intensity, indices = _find_isotope_cluster(
                mz, intensity, used, seed_idx, charge, tolerance, is_ppm
            )
            if n_peaks > best_n or (n_peaks == best_n and total_intensity > best_intensity):
                best_n = n_peaks
                best_intensity = total_intensity
                best_charge = charge
                best_indices[:] = indices

        # mark all cluster peaks as used
        for ki in range(best_n):
            used[best_indices[ki]] = True

        # monoisotopic peak is the seed; compute neutral mass
        mono_mz = float(mz[seed_idx])
        neutral_mass = mono_mz * best_charge - best_charge * PROTON_MASS

        out_mz[n_out] = neutral_mass
        out_int[n_out] = best_intensity
        n_out += 1

    return out_mz, out_int, n_out


def _decharge_matrix(
    spectrum_matrix: NDArray[np.float32],
    min_charge: int,
    max_charge: int,
    tolerance: float,
    is_ppm: bool,
    max_dpeaks: int,
) -> NDArray[np.float32]:
    n_spectra = spectrum_matrix.shape[0]
    out = np.zeros((n_spectra, max_dpeaks, 2), dtype=np.float32)

    for i in range(n_spectra):
        raw_mz = spectrum_matrix[i, :, 0]
        raw_int = spectrum_matrix[i, :, 1]

        valid = raw_mz > 0.0
        if not np.any(valid):
            continue

        mz = raw_mz[valid]
        intensity = raw_int[valid]

        out_mz, out_int, n_out = _decharge_single(
            mz, intensity, min_charge, max_charge, tolerance, is_ppm, max_dpeaks
        )

        if n_out == 0:
            continue

        # sort by neutral mass
        order = np.argsort(out_mz[:n_out])
        out[i, :n_out, 0] = out_mz[order]
        out[i, :n_out, 1] = out_int[order]

    return out


def decharge_spectrum_matrix(
    spectrum_matrix: NDArray[np.float32],
    charge_range: tuple[int, int],
    tolerance: float,
    is_ppm: bool,
    max_dpeaks: int,
) -> NDArray[np.float32]:
    """
    Deconvolute a spectrum matrix using a greedy isotope clustering approach.

    Args:
        spectrum_matrix: (num_spectra, max_peaks, 2) float32 array of (mz, intensity).
        charge_range: (min_charge, max_charge) to consider.
        tolerance: Peak matching tolerance.
        is_ppm: Whether tolerance is in ppm.
        max_dpeaks: Maximum deconvoluted peaks per spectrum.

    Returns:
        (num_spectra, max_dpeaks, 2) float32 array of (neutral_mass, summed_intensity).
    """
    if spectrum_matrix.ndim != 3 or spectrum_matrix.shape[2] != 2:
        raise ValueError("spectrum_matrix must have shape (n_spectra, max_peaks, 2)")

    min_charge, max_charge = charge_range
    return _decharge_matrix(
        spectrum_matrix.astype(np.float32),
        min_charge,
        max_charge,
        float(tolerance),
        bool(is_ppm),
        max_dpeaks,
    )
