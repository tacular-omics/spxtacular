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

import warnings

import numpy as np
import peptacular as pt
from numpy.typing import NDArray

from .greedy import PROTON_MASS, _find_anchor_candidates, _find_isotope_cluster

try:
    from numba import njit as _njit
except ImportError:

    def _njit(*args, **kwargs):
        def _wrap(f):
            return f

        return _wrap


#: Accepted ``intensity_mode`` values for :func:`deconvolve_spectrum`.
_INTENSITY_MODES: tuple[str, ...] = ("total", "base")

# ---------------------------------------------------------------------------
# Isotope template table (built once, looked up by neutral mass)
# ---------------------------------------------------------------------------

_MAX_ISO: int = 10
_MASS_STEP: int = 50
_MAX_MASS: int = 20000

#: How many neutron steps to search *below* the seed for the monoisotopic peak.
#: The seed is the most intense peak, which drifts to A+1 above ~1900 Da and to
#: A+2 above ~3500 Da; 4 covers analytes well beyond the template ceiling.
_MAX_BACK: int = 4

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
    """Return the normalised isotope distribution closest to neutral_mass.

    The template masses are the regular grid ``_MASS_STEP, 2*_MASS_STEP, ...``, so
    the nearest one is arithmetic rather than a search. This is called once per
    (seed, charge, anchor) combination -- tens of thousands of times for a single
    spectrum -- so the searchsorted it replaces was measurable.
    """
    _, dists = _get_templates()
    # +0.5 then truncate is round-half-up, matching the tie-break of the
    # searchsorted form this replaced (numpy's round() is half-to-even).
    idx = int(neutral_mass / _MASS_STEP + 0.5) - 1
    return dists[min(max(idx, 0), dists.shape[0] - 1)]


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
    if k < 2:
        # A single peak is not evidence of a charge state.  Scoring it against
        # a one-element template would give a perfect 1.0 (the vector is
        # trivially identical to itself after normalisation), which would beat
        # every genuine multi-peak cluster and destroy it.  Clusters of one are
        # rejected downstream anyway, so score them as no evidence at all.
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
    carrier_mass: float = PROTON_MASS,
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
    carrier_mass:
        Signed ion-mass delta per unit charge. Positive proton mass preserves
        the historical ``[M+H]+`` behavior; negative proton mass represents
        ``[M-H]-``. Used for neutral-mass isotope-template selection.

    Returns
    -------
    Tuple of (mz, charges, intensity, scores) arrays sorted by m/z.
    Singletons have charge == -1 and score == 0.0.

    Raises
    ------
    ValueError
        If ``intensity_mode`` is neither ``"total"`` nor ``"base"``.
    """
    # Normalise the mode once, up front, the way match_fragments normalises its
    # string-ish inputs: the selection below is a plain ``== "base"`` test, so
    # every other spelling used to fall through to "total" silently -- "Base"
    # returned cluster sums under the name of monoisotopic intensities, and a
    # typo like "mono" did the same. Kept in plain python, outside the numba-JIT'd
    # cluster loops, and ahead of the empty-input shortcut so the check does not
    # depend on how many peaks came in.
    mode = str(intensity_mode).lower()
    if mode not in _INTENSITY_MODES:
        raise ValueError(f"intensity_mode must be one of {_INTENSITY_MODES}, got {intensity_mode!r}")

    if len(mz) == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, np.empty(0, dtype=np.int32), empty, empty

    carrier_mass = float(carrier_mass)
    if not np.isfinite(carrier_mass):
        raise ValueError(f"carrier_mass must be finite, got {carrier_mass!r}")

    min_charge, max_charge = charge_range
    if min_charge < 1 or max_charge < 1:
        raise ValueError(f"charge_range must contain positive charges, got {charge_range}")
    if min_charge > max_charge:
        raise ValueError(f"charge_range must be (min, max) with min <= max, got {charge_range}")

    mz64 = np.ascontiguousarray(mz, dtype=np.float64)
    int64 = np.ascontiguousarray(intensity, dtype=np.float64)

    n = len(mz)
    used = np.zeros(n, dtype=np.bool_)

    out_mz = np.zeros(max_dpeaks, dtype=np.float64)
    out_charges = np.full(max_dpeaks, -1, dtype=np.int32)
    out_total_int = np.zeros(max_dpeaks, dtype=np.float64)
    out_base_int = np.zeros(max_dpeaks, dtype=np.float64)
    out_scores = np.zeros(max_dpeaks, dtype=np.float64)
    n_out = 0

    # Seeds are taken most-intense-first. Scanning for the maximum on each pass
    # makes that O(n) per seed and O(n^2) overall; peaks only ever become used,
    # never un-used, so one descending sort plus a cursor gives the same sequence
    # in O(n log n). Stable sort on the negated intensity reproduces argmax's
    # first-index tie-break exactly.
    seed_order = np.argsort(-int64, kind="stable")
    cursor = 0
    # Reused across seeds rather than reallocated: np.full on a ten-element array
    # is dominated by call overhead, and this runs once per output peak.
    best_indices = np.full(10, -1, dtype=np.intp)

    while n_out < max_dpeaks:
        while cursor < n and used[seed_order[cursor]]:
            cursor += 1
        if cursor >= n:
            break
        seed_idx = int(seed_order[cursor])

        best_score = -np.inf
        best_charge = min_charge
        best_indices.fill(-1)
        best_n = 1
        best_anchor = seed_idx
        best_total = float(int64[seed_idx])
        best_base = float(int64[seed_idx])

        for charge in range(min_charge, max_charge + 1):
            # The seed is the most intense peak, which is only the monoisotopic
            # peak for smaller analytes.  Try anchoring the cluster at the seed
            # and at each peak reachable by stepping backwards, then let the
            # isotope template decide which alignment is real.
            anchors, n_anchors = _find_anchor_candidates(mz64, used, seed_idx, charge, tolerance, is_ppm, _MAX_BACK)
            for ai in range(n_anchors):
                anchor = int(anchors[ai])
                n_peaks, total_intensity, base_intensity, indices = _find_isotope_cluster(
                    mz64, int64, used, anchor, charge, tolerance, is_ppm
                )
                cluster_idx = indices[:n_peaks]
                # A cluster that does not reach back to the seed describes a
                # different feature.  Accepting it would leave the seed unused
                # and re-seed on it forever, so skip it.
                #
                # Checked with a Python loop rather than np.any(...): the cluster
                # holds at most ten entries, and at that size numpy's per-call
                # overhead costs more than the comparison it performs. The loop
                # runs once per (seed, charge, anchor) combination, so it is hot.
                found = False
                for ci in range(n_peaks):
                    if indices[ci] == seed_idx:
                        found = True
                        break
                if not found:
                    continue
                obs = int64[cluster_idx]
                neutral_mass = float(mz64[anchor]) * charge - carrier_mass * charge
                if neutral_mass < 0.0:
                    continue
                template = _lookup_template(neutral_mass)
                score = _score_cluster(obs, template, min_intensity)
                if score > best_score or (score == best_score and n_peaks > best_n):
                    best_score = score
                    best_charge = charge
                    best_indices[:] = indices
                    best_n = n_peaks
                    best_anchor = anchor
                    best_total = total_intensity
                    best_base = base_intensity

        accepted = best_n > 1 and best_score >= min_score

        if accepted:
            for ki in range(best_n):
                used[best_indices[ki]] = True
        # Always consume the seed: on rejection the rest of the tried cluster
        # stays free and is re-seeded later, and on acceptance this guarantees
        # forward progress even if the seed somehow fell outside the cluster.
        used[seed_idx] = True

        out_mz[n_out] = float(mz64[best_anchor]) if accepted else float(mz64[seed_idx])
        out_charges[n_out] = best_charge if accepted else -1
        out_scores[n_out] = best_score if accepted else 0.0
        if accepted:
            out_total_int[n_out] = best_total
            out_base_int[n_out] = best_base
        else:
            # Rejected: only the seed is consumed here; the rest of the tried
            # cluster stays free and is re-seeded later, so record the seed's own
            # intensity (not best_total) to avoid double-counting it in "total" mode.
            seed_int = float(int64[seed_idx])
            out_total_int[n_out] = seed_int
            out_base_int[n_out] = seed_int
        n_out += 1

    if n_out >= max_dpeaks and not np.all(used):
        warnings.warn(
            f"Deconvolution stopped at max_dpeaks={max_dpeaks} with "
            f"{int((~used).sum())} input peaks still unprocessed; raise max_dpeaks "
            "to deconvolute the whole spectrum.",
            UserWarning,
            stacklevel=2,
        )

    if n_out == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, np.empty(0, dtype=np.int32), empty, empty

    out_int = out_base_int[:n_out] if mode == "base" else out_total_int[:n_out]
    order = np.argsort(out_mz[:n_out])
    return (
        out_mz[:n_out][order],
        out_charges[:n_out][order],
        out_int[order],
        out_scores[:n_out][order],
    )
