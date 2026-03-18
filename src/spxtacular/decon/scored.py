"""
Greedy isotope-cluster deconvolution with isotopic profile scoring.

Same interface as greedy.py, plus ``min_intensity`` for S/N filtering.
The best charge state is chosen by isotopic pattern score (Bhattacharyya
coefficient penalised for missed detectable peaks) rather than longest
chain length.

Public entry point::

    mz_out, charges_out, intensity_out, scores_out = deconvolve_spectrum(
        mz, intensity, charge_range=(1, 5), tolerance=10.0, is_ppm=True,
        min_intensity=500.0,
    )
"""

from __future__ import annotations

import numpy as np
import peptacular as pt
from numpy.typing import NDArray

from .greedy import PROTON_MASS, _find_isotope_cluster

try:
    from numba import njit as _njit
except ImportError:

    def _njit(*args, **kwargs):
        def _wrap(f):
            return f

        return _wrap

# ---------------------------------------------------------------------------
# Isotope template table (built once, looked up by neutral mass)
# ---------------------------------------------------------------------------

_MAX_ISO: int = 10
_MASS_STEP: int = 50
_MAX_MASS: int = 5000

_TEMPLATE_MASSES: NDArray[np.float64] | None = None
_TEMPLATE_DISTS: NDArray[np.float64] | None = None  # shape (T, _MAX_ISO)


def _build_templates() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    masses = np.arange(_MASS_STEP, _MAX_MASS + _MASS_STEP, _MASS_STEP, dtype=np.float64)
    T = len(masses)
    dists = np.zeros((T, _MAX_ISO), dtype=np.float64)
    for i, mass in enumerate(masses):
        pattern = pt.estimate_isotopic_distribution(
            float(mass),
            max_isotopes=_MAX_ISO,
            min_abundance_threshold=0.0,
            use_neutron_count=True,
        )
        abundances = np.array([iso.abundance for iso in pattern[:_MAX_ISO]], dtype=np.float64)
        s = abundances.sum()
        if s > 0.0:
            dists[i, : len(abundances)] = abundances / s
    return masses, dists


def _get_templates() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    global _TEMPLATE_MASSES, _TEMPLATE_DISTS
    if _TEMPLATE_MASSES is None:
        _TEMPLATE_MASSES, _TEMPLATE_DISTS = _build_templates()
    assert _TEMPLATE_MASSES is not None and _TEMPLATE_DISTS is not None
    return _TEMPLATE_MASSES, _TEMPLATE_DISTS


def _lookup_template(neutral_mass: float) -> NDArray[np.float64]:
    """Return the normalised isotope distribution closest to neutral_mass."""
    masses, dists = _get_templates()
    idx = int(np.searchsorted(masses, neutral_mass))
    if idx >= len(masses):
        idx = len(masses) - 1
    elif idx > 0 and abs(masses[idx - 1] - neutral_mass) < abs(masses[idx] - neutral_mass):
        idx -= 1
    return dists[idx]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@_njit(cache=True)
def _score_cluster(
    obs: NDArray[np.float64],
    template: NDArray[np.float64],
    min_intensity: float,
) -> float:
    """Isotopic pattern score: bhattacharyya × (1 − missed_penalty).

    Parameters
    ----------
    obs:
        Observed cluster intensities, shape (k,).
    template:
        Normalised isotope distribution, shape (_MAX_ISO,).
    min_intensity:
        Absolute intensity floor.  Theoretical peaks scaled below this value
        are treated as undetectable and not penalised when absent.
    """
    k = len(obs)
    if k == 0:
        return 0.0

    max_obs = float(obs.max())
    max_theo = float(template.max())
    if max_theo <= 0.0 or max_obs <= 0.0:
        return 0.0

    # Scale template to observed maximum
    scaled_theo = template * (max_obs / max_theo)

    # Pad observed to full template length (zeros beyond what we collected)
    obs_padded = np.zeros(_MAX_ISO, dtype=np.float64)
    obs_padded[:k] = obs

    detectable = scaled_theo >= min_intensity
    include = detectable | (obs_padded > 0.0)

    obs_f = obs_padded * include
    theo_f = scaled_theo * include

    sum_obs = float(obs_f.sum())
    sum_theo = float(theo_f.sum())
    if sum_obs <= 0.0 or sum_theo <= 0.0:
        return 0.0

    obs_n = obs_f / sum_obs
    theo_n = theo_f / sum_theo

    bhatt = float(np.sqrt(obs_n * theo_n).sum())

    missed = float(theo_f[detectable & (obs_padded == 0.0)].sum())
    total_det = float(theo_f[detectable].sum())
    missed_penalty = missed / total_det if total_det > 0.0 else 0.0

    return bhatt * (1.0 - missed_penalty)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def deconvolve_spectrum(
    mz: NDArray[np.float64],
    intensity: NDArray[np.float64],
    charge_range: tuple[int, int],
    tolerance: float,
    is_ppm: bool,
    max_dpeaks: int = 2000,
    intensity_mode: str = "total",
    min_intensity: float = 0.0,
    min_score: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.float64], NDArray[np.float64]]:
    """Greedy isotope deconvolution with isotopic profile scoring.

    Parameters
    ----------
    mz:
        m/z array (float64), sorted ascending.
    intensity:
        Intensity array (float64), same length as mz.
    charge_range:
        (min_charge, max_charge) inclusive.
    tolerance:
        Peak matching tolerance value.
    is_ppm:
        If True, tolerance is in ppm; otherwise Da.
    max_dpeaks:
        Maximum number of output peaks.
    intensity_mode:
        ``"total"`` (sum of cluster) or ``"base"`` (monoisotopic peak only).
    min_intensity:
        Absolute intensity threshold for detectability.  Theoretical isotope
        peaks scaled below this value are not penalised when absent.
        Set to ``0.0`` (default) to disable S/N filtering.
    min_score:
        Minimum isotopic profile score (0–1) for a cluster to be assigned a
        charge state.  Clusters whose best score falls below this threshold
        are recorded as singletons (charge == -1).  Set to ``0.0`` (default)
        to accept all clusters.

    Returns
    -------
    Tuple of (mz, charges, intensity, scores) arrays sorted by m/z.
    Singletons have charge == -1 and score == 0.0.
    """
    if len(mz) == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, np.empty(0, dtype=np.int32), empty, empty

    min_charge, max_charge = charge_range
    mz32 = mz.astype(np.float32)
    int32 = intensity.astype(np.float32)

    n = len(mz)
    used = np.zeros(n, dtype=np.bool_)

    out_mz = np.zeros(max_dpeaks, dtype=np.float64)
    out_charges = np.full(max_dpeaks, -1, dtype=np.int32)
    out_total_int = np.zeros(max_dpeaks, dtype=np.float64)
    out_base_int = np.zeros(max_dpeaks, dtype=np.float64)
    out_scores = np.zeros(max_dpeaks, dtype=np.float64)
    n_out = 0

    while n_out < max_dpeaks:
        masked_intensity = np.where(~used, int32, -np.inf)
        seed_idx = int(np.argmax(masked_intensity))
        if used[seed_idx]:
            break

        best_score = -np.inf
        best_charge = min_charge
        best_indices = np.full(10, -1, dtype=np.intp)
        best_n = 1
        best_total = float(int32[seed_idx])
        best_base = float(int32[seed_idx])

        for charge in range(min_charge, max_charge + 1):
            n_peaks, total_intensity, base_intensity, indices = _find_isotope_cluster(
                mz32, int32, used, seed_idx, charge, tolerance, is_ppm
            )
            cluster_idx = indices[:n_peaks]
            obs = int32[cluster_idx].astype(np.float64)
            neutral_mass = (float(mz32[seed_idx]) - PROTON_MASS) * charge
            template = _lookup_template(neutral_mass)
            score = _score_cluster(obs, template, min_intensity)
            if score > best_score or (score == best_score and n_peaks > best_n):
                best_score = score
                best_charge = charge
                best_indices[:] = indices
                best_n = n_peaks
                best_total = total_intensity
                best_base = base_intensity

        if best_n > 1 and best_score >= min_score:
            for ki in range(best_n):
                used[best_indices[ki]] = True
        else:
            used[seed_idx] = True  # reject: only consume the seed, leave cluster peaks free

        accepted = best_n > 1 and best_score >= min_score
        out_mz[n_out] = float(mz32[seed_idx])
        out_charges[n_out] = best_charge if accepted else -1
        out_scores[n_out] = best_score if accepted else 0.0
        out_total_int[n_out] = best_total
        out_base_int[n_out] = best_base
        n_out += 1

    if n_out == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, np.empty(0, dtype=np.int32), empty, empty

    out_int = out_base_int[:n_out] if intensity_mode == "base" else out_total_int[:n_out]
    order = np.argsort(out_mz[:n_out])
    return (
        out_mz[:n_out][order],
        out_charges[:n_out][order],
        out_int[order],
        out_scores[:n_out][order],
    )
