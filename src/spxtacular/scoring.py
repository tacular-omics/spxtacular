"""
Fragment ion scoring.

Single public entry point: :func:`score`.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any, cast

import numpy as np

from .core import Spectrum
from .enums import (
    DEFAULT_FRAGMENT_TOLERANCE,
    DEFAULT_FRAGMENT_TOLERANCE_TYPE,
    PeakSelection,
    PeakSelectionLike,
    ToleranceLike,
    ToleranceType,
)
from .matching import FragmentInput, MatchedFragment, match_fragments

# Floor for the per-peak random-match probability used by ``_probability_score``.
# A tolerance of exactly 0 makes p == 0, whose survival function is -inf and whose
# negated score is +inf; clamping keeps every reported score finite (and large).
_MIN_MATCH_PROBABILITY = 1e-12

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _unique_peak_indices(matches: list[MatchedFragment]) -> list[int]:
    seen: set[int] = set()
    return [m.peak_index for m in matches if not (m.peak_index in seen or seen.add(m.peak_index))]


def _unique_series_positions(
    matches: list[MatchedFragment],
) -> dict[str, set]:
    """Map each ion series to the set of unique positions matched.

    Neutral-loss and isotope variants share ``ion_type`` + ``position`` and
    collapse to one entry, preventing inflation of the hyperscore factorial.
    """
    sp: dict[str, set] = defaultdict(set)
    for m in matches:
        sp[str(m.fragment.ion_type)].add(m.fragment.position)
    return sp


def _count_unique_ions(fragments: FragmentInput) -> int:
    """Unique ``(ion_type, position)`` pairs — collapses loss/isotope variants.

    Both input shapes are counted the same way.  In the dict form the key is
    ``(ion_type, charge_state)`` and the position is the 1-based index within the
    mass list (exactly how :func:`~spxtacular.matching.match_fragments` assigns
    it), so the same ion seen at several charge states collapses to one entry —
    matching what the ``Sequence[Fragment]`` branch reports for the same physical
    fragment set.
    """
    if not isinstance(fragments, dict):
        return len({(str(f.ion_type), f.position) for f in fragments})
    d: Any = fragments
    seen: set[tuple[str, int]] = set()
    for key, masses in d.items():
        ion_type = cast(tuple, key)[0]  # key is (ion_type, charge_state)
        n_masses = len(cast(list, masses))
        seen.update((str(ion_type), pos) for pos in range(1, n_masses + 1))
    return len(seen)


def _log10_factorial(n: int) -> float:
    return math.lgamma(n + 1) / math.log(10)


# ``math.lgamma`` as a ufunc: bit-identical to the scalar function, but the loop
# runs in C instead of building intermediate Python lists.
_lgamma_ufunc = np.frompyfunc(math.lgamma, 1, 1)

# ``_LOG_FACTORIAL[m] == math.lgamma(m + 1) == log(m!)``. Grown on demand and
# shared between calls: scoring many candidate peptides against one spectrum
# otherwise recomputes the same O(n_peaks) lgamma values every single time.
# Capped so a single freak spectrum cannot pin an unbounded array for the life of
# the process (1e6 entries is 8 MB, and well above any real centroid peak count).
_LOG_FACTORIAL_MAX_CACHED = 1_000_000
_LOG_FACTORIAL: np.ndarray = np.zeros(1, dtype=np.float64)


def _lgamma_range(lo: int, hi: int) -> np.ndarray:
    """``[math.lgamma(m + 1) for m in range(lo, hi)]`` as a float64 array."""
    return np.asarray(_lgamma_ufunc(np.arange(lo + 1, hi + 1, dtype=np.float64)), dtype=np.float64)


def _log_factorial_table(n: int) -> np.ndarray:
    """Return ``lf`` with ``lf[m] == math.lgamma(m + 1)`` for all ``0 <= m <= n``."""
    global _LOG_FACTORIAL
    have = _LOG_FACTORIAL.size
    if have > n:
        return _LOG_FACTORIAL
    grown = np.empty(n + 1, dtype=np.float64)
    grown[:have] = _LOG_FACTORIAL
    grown[have:] = _lgamma_range(have, n + 1)
    if n < _LOG_FACTORIAL_MAX_CACHED:
        _LOG_FACTORIAL = grown
    return grown


def _binom_log10_survival(k: int, n: int, p: float) -> float:
    """log10 P(X >= k) for X ~ Binomial(n, p), log-space computation."""
    if k <= 0:
        return 0.0
    if k > n or p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return 0.0

    log_p = math.log(p)
    log_1mp = math.log(1.0 - p)
    i = np.arange(k, n + 1, dtype=np.float64)
    # log C(n, j) == lgamma(n + 1) - lgamma(j + 1) - lgamma(n - j + 1); both lgamma
    # terms are lookups into one shared table rather than two Python-level loops.
    lf = _log_factorial_table(n)
    j = np.arange(k, n + 1)
    log_c = lf[n] - (lf[j] + lf[n - j])
    log_terms = log_c + i * log_p + (n - i) * log_1mp
    max_t = float(log_terms.max())
    log_prob = max_t + math.log(float(np.exp(log_terms - max_t).sum()))
    return log_prob / math.log(10)


# ---------------------------------------------------------------------------
# Individual scorers (private)
# ---------------------------------------------------------------------------


def _series_key(fragment) -> str:
    """Ion-series name for a fragment, e.g. ``"b"`` / ``"y"`` / ``"c"``."""
    return str(fragment.ion_type)


def _searched_series(fragments: FragmentInput) -> set[str]:
    """Ion series present in the theoretical fragment set.

    These are the series the search *asked about*, which is what the hyperscore
    product runs over -- a series that was searched and returned nothing is
    evidence against the match, and can only be counted as such if we know it was
    looked for.
    """
    if not isinstance(fragments, dict):
        return {str(f.ion_type) for f in fragments}
    d: Any = fragments
    return {str(cast(tuple, key)[0]) for key in d}


def _hyperscore(
    spectrum: Spectrum,
    matches: list[MatchedFragment],
    searched_series: Iterable[str] | None = None,
) -> float:
    """X!Tandem hyperscore, generalised over ion series.

    ``log10( prod_s sum(I_s) ) + sum_s log10(n_s!)``, the product running over the
    ion series that were *searched* (``searched_series``), where ``sum(I_s)`` is
    the summed intensity of the peaks matched by series ``s`` and ``n_s`` the
    number of distinct ions matched from it.

    For the usual b/y search this is **exactly** the X!Tandem hyperscore
    ``log10(sum(I_b) * sum(I_y) * n_b! * n_y!)`` — verified to ~1e-15 — so scores
    are comparable with X!Tandem, Comet and MSFragger. Unlike those, it is not
    limited to b/y: an ETD search over c/z gets the same treatment.

    The product, rather than a sum over all matched peaks, is what makes the score
    discriminating. A searched series with no signal at all collapses the whole
    product to zero, so a PSM supported only by b ions cannot look as good as one
    corroborated from both directions. Summing instead lets those through: on a
    target-decoy trial, 17% of decoys scored above zero under a sum where the
    product correctly rejected them (separation, Cohen's d: 5.8 product vs 4.6 sum).

    .. warning::
       The intensity term consumes **raw** intensities, so the score is
       intensity-scale dependent: it shifts by ``log10(s)`` if the whole spectrum
       is multiplied by ``s``, and it can go *negative* on a normalised spectrum
       (e.g. TIC-normalised, where the matched sums are < 1). Only compare
       hyperscores computed on identically scaled spectra.
    """
    if not matches:
        return 0.0

    series_positions = _unique_series_positions(matches)
    expected = set(searched_series) if searched_series else set(series_positions)
    if not expected:
        return 0.0

    # Sum intensity per series, counting each peak once within a series even if
    # several of that series' ions hit it.
    series_intensity: dict[str, float] = {}
    seen: dict[str, set[int]] = {}
    for m in matches:
        s = _series_key(m.fragment)
        if m.peak_index in seen.setdefault(s, set()):
            continue
        seen[s].add(m.peak_index)
        series_intensity[s] = series_intensity.get(s, 0.0) + float(spectrum.intensity[m.peak_index])

    total = 0.0
    for s in sorted(expected):
        intensity_s = series_intensity.get(s, 0.0)
        if intensity_s <= 0.0:
            # A searched series with no signal collapses the product.
            return 0.0
        total += math.log10(intensity_s)
        total += _log10_factorial(len(series_positions.get(s, ())))
    return float(total)


def _probability_score(
    spectrum: Spectrum,
    matches: list[MatchedFragment],
    n_unique: int,
    tolerance: float,
    tolerance_type: ToleranceLike,
) -> float:
    n_exp = len(spectrum.mz)
    k = len(_unique_peak_indices(matches))
    if k == 0 or n_exp == 0 or n_unique == 0:
        return 0.0
    mz_range = float(spectrum.mz[-1] - spectrum.mz[0])
    if mz_range <= 0.0:
        return 0.0
    if ToleranceType(str(tolerance_type).lower()) is ToleranceType.PPM:
        tol_da = tolerance * float(np.median(spectrum.mz)) / 1e6
    else:
        tol_da = float(tolerance)
    # Floor p: a zero tolerance would otherwise give p == 0 -> log10 survival of
    # -inf -> a score of +inf, which poisons every downstream comparison.
    p = min(1.0, max(_MIN_MATCH_PROBABILITY, 2.0 * tol_da * n_unique / mz_range))
    return float(-_binom_log10_survival(k, n_exp, p))


def _total_matched_intensity(
    spectrum: Spectrum,
    matches: list[MatchedFragment],
) -> float:
    if not matches:
        return 0.0
    return float(np.sum(spectrum.intensity[_unique_peak_indices(matches)]))


def _matched_fraction(
    matches: list[MatchedFragment],
    n_unique: int,
) -> float:
    if n_unique == 0:
        return 0.0
    return len(_unique_peak_indices(matches)) / n_unique


def _intensity_fraction(
    spectrum: Spectrum,
    matches: list[MatchedFragment],
) -> float:
    total = float(spectrum.intensity.sum())
    if total == 0.0 or not matches:
        return 0.0
    return _total_matched_intensity(spectrum, matches) / total


def _mean_ppm_error(
    matches: list[MatchedFragment],
) -> float:
    if not matches:
        return 0.0
    return float(np.mean([abs(m.ppm_error) for m in matches]))


def _fragment_identity(fragment) -> tuple[str, Any, Any]:
    """Key identifying a theoretical ion across the fragment list and the matches."""
    return (str(fragment.ion_type), fragment.position, getattr(fragment, "charge", None))


def _spectral_angle_predicted(
    spectrum: Spectrum,
    matches: list[MatchedFragment],
    fragments: FragmentInput,
    predicted: Sequence[float],
) -> float:
    """The literature spectral angle against a predicted intensity vector.

    ``1 - 2 * arccos(cos) / pi`` over the cosine between observed and predicted
    intensities, both taken over the full theoretical ion set with unmatched ions
    contributing an observed intensity of zero. This is the metric reported by
    Prosit, Spectronaut and the like, so values are comparable with them.
    """
    if isinstance(fragments, dict):
        raise TypeError(
            "predicted_intensities requires the Sequence[Fragment] form of `fragments`, "
            "so each predicted value can be paired with its ion"
        )
    frag_list = list(fragments)
    if len(predicted) != len(frag_list):
        raise ValueError(f"predicted_intensities has {len(predicted)} entries but there are {len(frag_list)} fragments")

    # Collapse to unique ions; a fragment appearing twice keeps its first prediction.
    order: dict[tuple, int] = {}
    pred_vec: list[float] = []
    for frag, value in zip(frag_list, predicted, strict=True):
        key = _fragment_identity(frag)
        if key not in order:
            order[key] = len(pred_vec)
            pred_vec.append(float(value))

    obs_vec = np.zeros(len(pred_vec), dtype=np.float64)
    seen_peaks: set[int] = set()
    for m in matches:
        key = _fragment_identity(m.fragment)
        slot = order.get(key)
        if slot is None or m.peak_index in seen_peaks:
            continue
        seen_peaks.add(m.peak_index)
        obs_vec[slot] += float(spectrum.intensity[m.peak_index])

    pred = np.asarray(pred_vec, dtype=np.float64)
    obs_norm = float(np.linalg.norm(obs_vec))
    pred_norm = float(np.linalg.norm(pred))
    if obs_norm == 0.0 or pred_norm == 0.0:
        return 0.0

    cos = float(np.clip(float(obs_vec @ pred) / (obs_norm * pred_norm), -1.0, 1.0))
    if math.isnan(cos):
        return 0.0
    return float(1.0 - 2.0 * math.acos(cos) / math.pi)


def _spectral_angle(
    spectrum: Spectrum,
    matches: list[MatchedFragment],
    n_unique: int,
) -> float:
    """Normalised angle between the matched intensities and a flat reference.

    This is the **fallback** used when no predicted intensities are given. Pass
    ``predicted_intensities`` to :func:`score` to get the real spectral angle
    instead — see :func:`_spectral_angle_predicted`.

    .. warning::
       Despite the name this is **not** the spectral angle / spectral contrast
       angle of the literature (Toprak et al.; used by Prosit, Spectronaut, …).
       That metric needs a *predicted* intensity vector, and none is available
       when the caller supplies only m/z values. What is actually computed is
       the cosine between the observed matched-intensity vector (length
       ``n_unique``, zero-padded for unmatched theoretical ions) and an implicit
       **ones-vector**, mapped through ``1 - acos(cos)/(pi/2)``.

       The reference being flat means this measures *intensity uniformity across
       the matched ions* x *coverage*, not similarity to a predicted spectrum. A
       perfect, complete match with realistic intensities ``[100, 50, 10, 1]``
       scores ``0.509``, while a flat ``[7, 7, 7, 7]`` scores ``1.0``. Treat it as
       a coverage/evenness feature, and do not compare it to published spectral
       angles.

    Returns ``0.0`` when there is nothing to score (no matches, no theoretical
    ions, all-zero intensities) or when the intensities are not finite.
    """
    if not matches or n_unique == 0:
        return 0.0
    unique_idx = _unique_peak_indices(matches)
    matched = np.asarray(spectrum.intensity[unique_idx], dtype=np.float64)
    # The vector must have exactly n_unique entries for the Cauchy-Schwarz bound
    # (`sum(v) <= ||v|| * sqrt(len(v))`) to hold. With peak_selection="all" a single
    # theoretical ion can claim several peaks, so more unique peaks than theoretical
    # ions are possible; keeping the n_unique most intense ones bounds the ratio
    # instead of letting the raw cosine exceed 1 and get clamped to a perfect score.
    obs = np.zeros(n_unique, dtype=np.float64)
    if matched.size > n_unique:
        matched = np.sort(matched)[::-1][:n_unique]
    obs[: matched.size] = matched
    obs_norm = float(np.linalg.norm(obs))
    if obs_norm == 0.0:
        return 0.0
    # np.clip propagates NaN, unlike max(-1.0, min(1.0, nan)) which yields 1.0 —
    # i.e. a NaN intensity used to be reported as a perfect score.
    dot = float(np.clip(float(obs.sum()) / (obs_norm * math.sqrt(n_unique)), -1.0, 1.0))
    if math.isnan(dot):
        return 0.0
    return float(1.0 - math.acos(dot) / (math.pi / 2))


def _longest_run(matches: list[MatchedFragment]) -> int:
    """Longest ladder of consecutive positions within a single ion series.

    Terminal ions (``b``, ``y``, ``c``, ``z``, …) carry an ``int`` position and
    ladder on it directly.  Internal ions carry a ``(start, end)`` tuple; they are
    grouped by ion type **and** every element but the last (i.e. the start), and
    the ladder runs over the last element (the end), so an internal series that
    grows one residue at a time from a common start still counts as a run.
    Previously any tuple position was dropped outright, which reported ``0`` for
    spectra whose only matches were internal ions.

    Fragments with no position (``None`` — precursor / immonium ions) have no
    ladder to be part of and are ignored.
    """
    if not matches:
        return 0
    # key: (ion_type, tuple-position prefix); () for plain int positions
    series_positions: dict[tuple[str, tuple[int, ...]], list[int]] = defaultdict(list)
    for m in matches:
        pos = m.fragment.position
        if isinstance(pos, int):
            prefix: tuple[int, ...] = ()
            coord = pos
        elif isinstance(pos, tuple) and pos and all(isinstance(p, int) for p in pos):
            prefix, coord = tuple(pos[:-1]), pos[-1]
        else:
            continue
        series_positions[(str(m.fragment.ion_type), prefix)].append(coord)
    best = 0
    for positions in series_positions.values():
        sorted_pos = sorted(set(positions))
        run = 1
        for a, b in zip(sorted_pos, sorted_pos[1:], strict=False):
            if b == a + 1:
                run += 1
                best = max(best, run)
            else:
                run = 1
        best = max(best, run)
    return best


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score(
    spectrum: Spectrum,
    fragments: FragmentInput,
    tolerance: float = DEFAULT_FRAGMENT_TOLERANCE,
    tolerance_type: ToleranceLike = DEFAULT_FRAGMENT_TOLERANCE_TYPE,
    peak_selection: PeakSelectionLike = PeakSelection.CLOSEST,
    predicted_intensities: Sequence[float] | None = None,
) -> dict[str, float]:
    """Match fragments against a spectrum and return all scores.

    Internally calls :func:`~spxtacular.matching.match_fragments` and
    computes ``n_theoretical`` as the number of unique ``(ion_type, position)``
    pairs, so neutral-loss and isotope variants of the same fragment do not
    inflate the scores.

    ``hyperscore`` is the X!Tandem hyperscore, generalised so the product runs over
    whichever ion series were searched rather than only b/y; for a b/y search it is
    numerically identical to X!Tandem. It consumes raw intensities and is therefore
    intensity-scale dependent — see :func:`_hyperscore`.

    ``spectral_angle`` is the literature spectral angle when
    ``predicted_intensities`` is supplied. Without one there is nothing to compare
    against, and the value falls back to a coverage/evenness measure against a flat
    reference, which is *not* comparable to published spectral angles — see
    :func:`_spectral_angle`.

    Parameters
    ----------
    spectrum:
        Experimental centroid spectrum.
    fragments:
        Theoretical fragment ions from peptacular, or the
        ``dict[tuple[IonType, int], list[float]]`` returned by
        :meth:`~peptacular.ProFormaAnnotation.fast_fragment`.
    tolerance:
        Matching tolerance.
    tolerance_type:
        ``"da"`` or ``"ppm"`` (case-insensitive; anything else raises
        ``ValueError``).
    peak_selection:
        How to resolve multiple peaks within tolerance per fragment:
        ``"closest"`` (default), ``"largest"``, or ``"all"``.

    Returns
    -------
    dict with keys:
    ``hyperscore``, ``probability_score``, ``total_matched_intensity``,
    ``matched_fraction``, ``intensity_fraction``, ``mean_ppm_error``,
    ``spectral_angle``, ``longest_run``.
    """
    # Normalise once so ``"PPM"``/``"Da"`` cannot mean one thing to the matcher and
    # another to _probability_score, and so typos raise here rather than silently
    # falling back to Da.
    tol_type = ToleranceType(str(tolerance_type).lower())
    matches = match_fragments(spectrum, fragments, tolerance, tol_type, peak_selection)
    n_unique = _count_unique_ions(fragments)

    return {
        "hyperscore": _hyperscore(spectrum, matches, _searched_series(fragments)),
        "probability_score": _probability_score(spectrum, matches, n_unique, tolerance, tol_type),
        "total_matched_intensity": _total_matched_intensity(spectrum, matches),
        "matched_fraction": _matched_fraction(matches, n_unique),
        "intensity_fraction": _intensity_fraction(spectrum, matches),
        "mean_ppm_error": _mean_ppm_error(matches),
        "spectral_angle": (
            _spectral_angle_predicted(spectrum, matches, fragments, predicted_intensities)
            if predicted_intensities is not None
            else _spectral_angle(spectrum, matches, n_unique)
        ),
        "longest_run": float(_longest_run(matches)),
    }
