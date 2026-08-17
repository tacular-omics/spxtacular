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
from numpy.typing import NDArray

from ..isotopes import IsotopeModelLike, resolve_isotope_model
from .greedy import NEUTRON_MASS, PROTON_MASS, _has_isotope_neighbor, _match_apex_cluster

try:
    from numba import njit as _njit
except ImportError:

    def _njit(*args, **kwargs):
        def _wrap(f):
            return f

        return _wrap


#: Accepted ``intensity_mode`` values for :func:`deconvolve_spectrum`.
_INTENSITY_MODES: tuple[str, ...] = ("total", "base")

# At high neutral masses, several adjacent isotope peaks can have nearly the
# same predicted abundance. Small measurement fluctuations may therefore move
# the observed maximum away from the model's strict argmax. Treat contiguous
# positions this close to the predicted maximum as plausible seed alignments,
# then let the complete-envelope score choose between them.
_MIN_SEED_ALIGNMENT_ABUNDANCE: float = 0.9

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@_njit(cache=True)
def _score_cluster(
    obs: NDArray[np.float64],
    template: NDArray[np.float64],
    min_intensity: float,
    min_relative_abundance: float,
) -> float:
    """Isotopic pattern score: bhattacharyya × (1 − missed_penalty).

    Parameters
    ----------
    obs:
        Observed intensities aligned to theoretical isotope indices. Missing
        or rejected peaks are zero.
    template:
        Normalised isotope distribution, same shape as ``obs``.
    min_intensity:
        Absolute intensity floor.  Theoretical peaks scaled below this value
        are treated as undetectable and not penalised when absent.
    """
    if np.count_nonzero(obs) < 2:
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

    relative = template / max_theo
    detectable = (scaled_theo >= min_intensity) & (relative >= min_relative_abundance)
    include = detectable | (obs > 0.0)

    obs_f = obs * include
    theo_f = scaled_theo * include

    sum_obs = float(obs_f.sum())
    sum_theo = float(theo_f.sum())
    if sum_obs <= 0.0 or sum_theo <= 0.0:
        return 0.0

    obs_n = obs_f / sum_obs
    theo_n = theo_f / sum_theo

    bhatt = float(np.sqrt(obs_n * theo_n).sum())

    missed = float(theo_f[detectable & (obs == 0.0)].sum())
    total_det = float(theo_f[detectable].sum())
    missed_penalty = missed / total_det if total_det > 0.0 else 0.0

    return bhatt * (1.0 - missed_penalty)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _deconvolve_spectrum(
    mz: NDArray[np.float64],
    intensity: NDArray[np.float64],
    charge_range: tuple[int, int],
    tolerance: float,
    is_ppm: bool,
    max_dpeaks: int = 2000,
    intensity_mode: str = "total",
    min_intensity: float = 0.0,
    min_score: float = 0.0,
    isotope_model: IsotopeModelLike = "peptide",
    min_isotope_abundance: float = 0.01,
    max_isotope_fold_error: float = 2.0,
    max_isotope_gaps: int = 0,
    max_isotopes: int | None = None,
    ion_mobility: NDArray[np.float64] | None = None,
    im_tolerance: float = 0.05,
    im_tolerance_type: str = "relative",
    carrier_mass: float = PROTON_MASS,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int32],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.intp],
]:
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
        ``"total"`` (sum of matched peaks) or ``"base"`` (observed A+0,
        or zero when the monoisotopic peak is absent).
    min_intensity:
        Absolute intensity threshold for detectability.  Theoretical isotope
        peaks scaled below this value are not penalised when absent.
        Set to ``0.0`` (default) to disable S/N filtering.
    min_score:
        Minimum isotopic profile score (0–1) for a cluster to be assigned a
        charge state.  Clusters whose best score falls below this threshold
        are recorded as singletons (charge == -1).  Set to ``0.0`` (default)
        to accept all clusters.
    isotope_model:
        Average-composition model used to score isotope envelopes. Pass a
        preset name, :class:`~spxtacular.isotopes.IsotopeModelType`, or a
        custom :class:`~spxtacular.isotopes.IsotopeModel`.
    min_isotope_abundance:
        Minimum theoretical abundance relative to the envelope apex. Expansion
        stops when the next isotope falls below this value.
    max_isotope_fold_error:
        Maximum allowed ratio between observed and expected intensity. A peak
        outside ``[1 / value, value]`` stops expansion in that direction.
    max_isotope_gaps:
        Missing expected peaks allowed before stopping one direction. Defaults
        to zero for contiguous centroid envelopes.
    max_isotopes:
        Optional hard envelope-length limit. ``None`` chooses the length
        adaptively from the predicted distribution.
    ion_mobility:
        Optional per-peak ion-mobility values. When supplied, candidate peaks
        are gated and scored against the seed peak's mobility.
    im_tolerance:
        Maximum candidate-to-seed mobility difference.
    im_tolerance_type:
        ``"relative"`` scales ``im_tolerance`` by the seed mobility;
        ``"absolute"`` uses it directly.
    carrier_mass:
        Signed ion-mass delta per unit charge, used to estimate neutral mass
        for isotope-model selection.

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

    resolved_model = resolve_isotope_model(isotope_model)
    carrier_mass = float(carrier_mass)
    if not np.isfinite(carrier_mass):
        raise ValueError(f"carrier_mass must be finite, got {carrier_mass!r}")
    if not np.isfinite(min_isotope_abundance) or not 0.0 < min_isotope_abundance <= 1.0:
        raise ValueError(f"min_isotope_abundance must be in (0, 1], got {min_isotope_abundance!r}")
    if not np.isfinite(max_isotope_fold_error) or max_isotope_fold_error < 1.0:
        raise ValueError(f"max_isotope_fold_error must be finite and at least 1, got {max_isotope_fold_error!r}")
    if isinstance(max_isotope_gaps, bool) or not isinstance(max_isotope_gaps, int) or max_isotope_gaps < 0:
        raise ValueError(f"max_isotope_gaps must be a non-negative integer, got {max_isotope_gaps!r}")
    if max_isotopes is not None and (
        isinstance(max_isotopes, bool) or not isinstance(max_isotopes, int) or max_isotopes < 1
    ):
        raise ValueError(f"max_isotopes must be a positive integer or None, got {max_isotopes!r}")
    if not np.isfinite(im_tolerance) or im_tolerance < 0.0:
        raise ValueError(f"im_tolerance must be finite and non-negative, got {im_tolerance!r}")
    resolved_im_tolerance_type = str(im_tolerance_type).lower()
    if resolved_im_tolerance_type not in ("relative", "absolute"):
        raise ValueError(f"im_tolerance_type must be 'relative' or 'absolute', got {im_tolerance_type!r}")
    if ion_mobility is not None and len(ion_mobility) != len(mz):
        raise ValueError(f"ion_mobility must have the same length as mz, got {len(ion_mobility)} and {len(mz)}")

    if len(mz) == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, np.empty(0, dtype=np.int32), empty, empty, np.empty(0, dtype=np.intp)

    min_charge, max_charge = charge_range
    if min_charge < 1 or max_charge < 1:
        raise ValueError(f"charge_range must contain positive charges, got {charge_range}")
    if min_charge > max_charge:
        raise ValueError(f"charge_range must be (min, max) with min <= max, got {charge_range}")

    mz64 = np.ascontiguousarray(mz, dtype=np.float64)
    int64 = np.ascontiguousarray(intensity, dtype=np.float64)
    use_im = ion_mobility is not None
    im64 = (
        np.ascontiguousarray(ion_mobility, dtype=np.float64)
        if ion_mobility is not None
        else np.full(len(mz64), np.nan, dtype=np.float64)
    )
    source_index_map = np.arange(len(mz64), dtype=np.intp)
    if len(mz64) > 1 and np.any(mz64[1:] < mz64[:-1]):
        mz_order = np.argsort(mz64, kind="stable")
        mz64 = np.ascontiguousarray(mz64[mz_order])
        int64 = np.ascontiguousarray(int64[mz_order])
        im64 = np.ascontiguousarray(im64[mz_order])
        source_index_map = source_index_map[mz_order]

    n = len(mz)
    used = np.zeros(n, dtype=np.bool_)

    out_mz = np.zeros(max_dpeaks, dtype=np.float64)
    out_charges = np.full(max_dpeaks, -1, dtype=np.int32)
    out_total_int = np.zeros(max_dpeaks, dtype=np.float64)
    out_base_int = np.zeros(max_dpeaks, dtype=np.float64)
    out_scores = np.zeros(max_dpeaks, dtype=np.float64)
    out_source_indices = np.zeros(max_dpeaks, dtype=np.intp)
    n_out = 0

    # Seeds are taken most-intense-first. Scanning for the maximum on each pass
    # makes that O(n) per seed and O(n^2) overall; peaks only ever become used,
    # never un-used, so one descending sort plus a cursor gives the same sequence
    # in O(n log n). Stable sort on the negated intensity reproduces argmax's
    # first-index tie-break exactly.
    seed_order = np.argsort(-int64, kind="stable")
    cursor = 0

    while n_out < max_dpeaks:
        while cursor < n and used[seed_order[cursor]]:
            cursor += 1
        if cursor >= n:
            break
        seed_idx = int(seed_order[cursor])

        best_score = -np.inf
        best_charge = min_charge
        best_indices: NDArray[np.intp] | None = None
        best_n = 1
        best_mono_mz = float(mz64[seed_idx])
        best_total = float(int64[seed_idx])
        best_base = float(int64[seed_idx])

        for charge in range(min_charge, max_charge + 1):
            if not _has_isotope_neighbor(
                mz64,
                used,
                seed_idx,
                charge,
                tolerance,
                is_ppm,
                max_isotope_gaps + 1,
            ):
                continue
            # First estimate treats the seed as A+0. Aligning the predicted
            # apex with the seed then removes its neutron offset. Repeating this
            # once or twice resolves the small circular dependency between
            # neutral mass and the envelope apex.
            seed_mass = float(mz64[seed_idx]) * charge - carrier_mass * charge
            if seed_mass <= 0.0:
                continue
            template = resolved_model.adaptive_distribution(
                seed_mass,
                min_relative_abundance=min_isotope_abundance,
                max_isotopes=max_isotopes,
            )
            apex = int(np.argmax(template))
            for _ in range(2):
                neutral_mass = max(0.0, seed_mass - apex * NEUTRON_MASS)
                updated_template = resolved_model.adaptive_distribution(
                    neutral_mass,
                    min_relative_abundance=min_isotope_abundance,
                    max_isotopes=max_isotopes,
                )
                updated_apex = int(np.argmax(updated_template))
                template = updated_template
                if updated_apex == apex:
                    break
                apex = updated_apex
            peak_abundance = float(template[apex])
            near_apex = template >= peak_abundance * _MIN_SEED_ALIGNMENT_ABUNDANCE
            left_apex = apex
            while left_apex > 0 and near_apex[left_apex - 1]:
                left_apex -= 1
            right_apex = apex
            while right_apex + 1 < len(template) and near_apex[right_apex + 1]:
                right_apex += 1

            # Restrict custom or unusual distributions to the contiguous
            # near-maximum region containing the strict apex so a distant,
            # similarly intense mode is not treated as a nearby alignment.
            candidate_indices = range(left_apex, right_apex + 1)

            for seed_isotope_index in candidate_indices:
                seed_isotope_index = int(seed_isotope_index)
                relative = np.ascontiguousarray(
                    template / float(template[seed_isotope_index]),
                    dtype=np.float64,
                )
                n_peaks, total_intensity, base_intensity, indices = _match_apex_cluster(
                    mz64,
                    int64,
                    im64,
                    used,
                    seed_idx,
                    charge,
                    tolerance,
                    is_ppm,
                    relative,
                    seed_isotope_index,
                    min_isotope_abundance,
                    max_isotope_fold_error,
                    max_isotope_gaps,
                    use_im,
                    im_tolerance,
                    resolved_im_tolerance_type == "relative",
                )
                obs = np.zeros(len(template), dtype=np.float64)
                matched = indices >= 0
                obs[matched] = int64[indices[matched]]
                score = _score_cluster(obs, template, min_intensity, min_isotope_abundance)
                if score > best_score or (score == best_score and n_peaks > best_n):
                    best_score = score
                    best_charge = charge
                    best_indices = indices
                    best_n = n_peaks
                    best_mono_mz = float(mz64[seed_idx]) - seed_isotope_index * NEUTRON_MASS / charge
                    best_total = total_intensity
                    best_base = base_intensity

        accepted = best_indices is not None and best_n > 1 and best_score >= min_score

        if accepted and best_indices is not None:
            for matched_idx in best_indices:
                if matched_idx >= 0:
                    used[matched_idx] = True
        # Always consume the seed: on rejection the rest of the tried cluster
        # stays free and is re-seeded later, and on acceptance this guarantees
        # forward progress even if the seed somehow fell outside the cluster.
        used[seed_idx] = True

        out_mz[n_out] = best_mono_mz if accepted else float(mz64[seed_idx])
        out_charges[n_out] = best_charge if accepted else -1
        out_scores[n_out] = best_score if accepted else 0.0
        out_source_indices[n_out] = source_index_map[seed_idx]
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
        return empty, np.empty(0, dtype=np.int32), empty, empty, np.empty(0, dtype=np.intp)

    out_int = out_base_int[:n_out] if mode == "base" else out_total_int[:n_out]
    order = np.argsort(out_mz[:n_out])
    return (
        out_mz[:n_out][order],
        out_charges[:n_out][order],
        out_int[order],
        out_scores[:n_out][order],
        out_source_indices[:n_out][order],
    )


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
    isotope_model: IsotopeModelLike = "peptide",
    min_isotope_abundance: float = 0.01,
    max_isotope_fold_error: float = 2.0,
    max_isotope_gaps: int = 0,
    max_isotopes: int | None = None,
    ion_mobility: NDArray[np.float64] | None = None,
    im_tolerance: float = 0.05,
    im_tolerance_type: str = "relative",
    carrier_mass: float = PROTON_MASS,
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.float64], NDArray[np.float64]]:
    """Greedy apex-first isotope deconvolution.

    Every charge is evaluated before the winning candidate consumes peaks.
    See :meth:`spxtacular.Spectrum.deconvolute` for parameter details.
    """
    result = _deconvolve_spectrum(
        mz=mz,
        intensity=intensity,
        charge_range=charge_range,
        tolerance=tolerance,
        is_ppm=is_ppm,
        max_dpeaks=max_dpeaks,
        intensity_mode=intensity_mode,
        min_intensity=min_intensity,
        min_score=min_score,
        isotope_model=isotope_model,
        min_isotope_abundance=min_isotope_abundance,
        max_isotope_fold_error=max_isotope_fold_error,
        max_isotope_gaps=max_isotope_gaps,
        max_isotopes=max_isotopes,
        ion_mobility=ion_mobility,
        im_tolerance=im_tolerance,
        im_tolerance_type=im_tolerance_type,
        carrier_mass=carrier_mass,
    )
    return result[0], result[1], result[2], result[3]


def _deconvolve_spectrum_with_sources(
    mz: NDArray[np.float64],
    intensity: NDArray[np.float64],
    charge_range: tuple[int, int],
    tolerance: float,
    is_ppm: bool,
    max_dpeaks: int = 2000,
    intensity_mode: str = "total",
    min_intensity: float = 0.0,
    min_score: float = 0.0,
    isotope_model: IsotopeModelLike = "peptide",
    min_isotope_abundance: float = 0.01,
    max_isotope_fold_error: float = 2.0,
    max_isotope_gaps: int = 0,
    max_isotopes: int | None = None,
    ion_mobility: NDArray[np.float64] | None = None,
    im_tolerance: float = 0.05,
    im_tolerance_type: str = "relative",
    carrier_mass: float = PROTON_MASS,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int32],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.intp],
]:
    """Internal variant that also returns each output cluster's apex source index."""
    return _deconvolve_spectrum(
        mz=mz,
        intensity=intensity,
        charge_range=charge_range,
        tolerance=tolerance,
        is_ppm=is_ppm,
        max_dpeaks=max_dpeaks,
        intensity_mode=intensity_mode,
        min_intensity=min_intensity,
        min_score=min_score,
        isotope_model=isotope_model,
        min_isotope_abundance=min_isotope_abundance,
        max_isotope_fold_error=max_isotope_fold_error,
        max_isotope_gaps=max_isotope_gaps,
        max_isotopes=max_isotopes,
        ion_mobility=ion_mobility,
        im_tolerance=im_tolerance,
        im_tolerance_type=im_tolerance_type,
        carrier_mass=carrier_mass,
    )
