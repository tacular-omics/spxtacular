"""Greedy peak-merge kernel behind :meth:`spxtacular.Spectrum.merge`.

Numba compiles the kernel when installed; otherwise the same loop runs as plain
Python, which is still faster than the previous per-seed NumPy slicing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .decon.greedy import _njit


@_njit(cache=True)
def _merge_kernel(
    mz: NDArray[np.float64],
    intensity: NDArray[np.float64],
    order: NDArray[np.intp],
    left: NDArray[np.intp],
    right: NDArray[np.intp],
    charge: NDArray[np.int32],
    has_charge: bool,
    im: NDArray[np.float64],
    has_im: bool,
    im_tolerance: float,
    im_relative: bool,
    score: NDArray[np.float64],
    has_score: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.intp], NDArray[np.float64], int]:
    """Cluster m/z-sorted peaks greedily, most intense seed first.

    ``left[i]:right[i]`` is seed ``i``'s m/z window in the sorted arrays. A peak
    joins the seed's cluster when it is unused, has the seed's charge (when
    charges exist) and lies within the ion-mobility tolerance (when IM exists).
    The seed always joins its own cluster, even with a NaN m/z or IM.

    Returns the merged m/z, intensity, IM and score, the seed index of each
    cluster (for its charge), and the number of clusters.
    """
    n = mz.shape[0]
    used = np.zeros(n, dtype=np.bool_)
    out_mz = np.empty(n, dtype=np.float64)
    out_int = np.empty(n, dtype=np.float64)
    out_im = np.empty(n, dtype=np.float64)
    out_seed = np.empty(n, dtype=np.intp)
    out_score = np.empty(n, dtype=np.float64)
    members = np.empty(n, dtype=np.intp)
    count = 0
    for k in range(n):
        idx = order[k]
        if used[idx]:
            continue
        lo = left[idx]
        hi = right[idx]
        n_members = 0
        if lo <= idx < hi:
            seed_charge = charge[idx] if has_charge else 0
            seed_im = im[idx] if has_im else 0.0
            im_delta = seed_im * im_tolerance if im_relative else im_tolerance
            for j in range(lo, hi):
                if used[j]:
                    continue
                if j != idx:
                    if has_charge and charge[j] != seed_charge:
                        continue
                    if has_im and not (abs(im[j] - seed_im) <= im_delta):
                        continue
                members[n_members] = j
                n_members += 1
        else:
            # Only a NaN m/z gets here: its searchsorted window never
            # contains its own index, so it stays a peak of its own.
            members[0] = idx
            n_members = 1

        total = 0.0
        for m in range(n_members):
            total += intensity[members[m]]
        sum_mz = 0.0
        sum_im = 0.0
        best = -np.inf
        best_nan = False
        for m in range(n_members):
            j = members[m]
            w = intensity[j] if total > 0.0 else 1.0
            sum_mz += mz[j] * w
            if has_im:
                sum_im += im[j] * w
            if has_score:
                s = score[j]
                if np.isnan(s):
                    best_nan = True
                elif s > best:
                    best = s
            used[j] = True
        norm = total if total > 0.0 else float(n_members)
        out_mz[count] = sum_mz / norm
        out_int[count] = total
        out_im[count] = sum_im / norm if has_im else 0.0
        out_score[count] = np.nan if best_nan else best
        out_seed[count] = idx
        count += 1
    return out_mz, out_int, out_im, out_seed, out_score, count


def merge_peaks(
    mz: NDArray[np.float64],
    intensity: NDArray[np.float64],
    charge: NDArray[np.int32] | None,
    im: NDArray[np.float64] | None,
    iso_score: NDArray[np.float64] | None,
    *,
    mz_tolerance: float,
    is_ppm: bool,
    im_tolerance: float,
    im_relative: bool,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int32] | None,
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
]:
    """Merge peaks within an m/z (and IM) tolerance; the result is sorted by m/z.

    Arrays may come in any order. Returns ``(mz, intensity, charge, im, iso_score)``,
    each ``None`` output matching a ``None`` input.
    """
    sort_idx = np.argsort(mz)
    s_mz = np.ascontiguousarray(mz[sort_idx], dtype=np.float64)
    s_int = np.ascontiguousarray(intensity[sort_idx], dtype=np.float64)
    s_charge = np.ascontiguousarray(charge[sort_idx], dtype=np.int32) if charge is not None else np.zeros(1, np.int32)
    s_im = np.ascontiguousarray(im[sort_idx], dtype=np.float64) if im is not None else np.zeros(1, np.float64)
    s_score = (
        np.ascontiguousarray(iso_score[sort_idx], dtype=np.float64)
        if iso_score is not None
        else np.zeros(1, np.float64)
    )

    # Greedy order: most intense first (same argsort as the previous implementation,
    # so ties resolve identically).
    order = np.ascontiguousarray(np.argsort(s_int)[::-1]).astype(np.intp)
    delta = s_mz * mz_tolerance / 1e6 if is_ppm else np.full(s_mz.shape, mz_tolerance)
    left = np.searchsorted(s_mz, s_mz - delta, side="left").astype(np.intp)
    right = np.searchsorted(s_mz, s_mz + delta, side="right").astype(np.intp)

    out_mz, out_int, out_im, out_seed, out_score, count = _merge_kernel(
        s_mz,
        s_int,
        order,
        left,
        right,
        s_charge,
        charge is not None,
        s_im,
        im is not None,
        float(im_tolerance),
        im_relative,
        s_score,
        iso_score is not None,
    )
    new_mz = out_mz[:count]
    final = np.argsort(new_mz)
    new_mz = new_mz[final]
    new_int = out_int[:count][final]
    new_im = out_im[:count][final] if im is not None else None
    new_charge = s_charge[out_seed[:count][final]].astype(np.int32) if charge is not None else None
    new_score = out_score[:count][final] if iso_score is not None else None
    return new_mz, new_int, new_charge, new_im, new_score
