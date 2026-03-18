"""
Fragment ion scoring.

Single public entry point: :func:`score`.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

import numpy as np
from peptacular.annotation.frag import Fragment

from .core import Spectrum
from .matching import match_fragments

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _unique_peak_indices(matches: list[tuple[int, Fragment]]) -> list[int]:
    seen: set[int] = set()
    return [i for i, _ in matches if not (i in seen or seen.add(i))]


def _unique_series_positions(
    matches: list[tuple[int, Fragment]],
) -> dict[str, set]:
    """Map each ion series to the set of unique positions matched.

    Neutral-loss and isotope variants share ``ion_type`` + ``position`` and
    collapse to one entry, preventing inflation of the hyperscore factorial.
    """
    sp: dict[str, set] = defaultdict(set)
    for _, frag in matches:
        sp[str(frag.ion_type)].add(frag.position)
    return sp


def _count_unique_ions(fragments: Sequence[Fragment]) -> int:
    """Unique ``(ion_type, position)`` pairs — collapses loss/isotope variants."""
    return len({(str(f.ion_type), f.position) for f in fragments})


def _log10_factorial(n: int) -> float:
    return math.lgamma(n + 1) / math.log(10)


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
    log_c = math.lgamma(n + 1) - (
        np.array([math.lgamma(x + 1) for x in range(k, n + 1)])
        + np.array([math.lgamma(n - x + 1) for x in range(k, n + 1)])
    )
    log_terms = log_c + i * log_p + (n - i) * log_1mp
    max_t = float(log_terms.max())
    log_prob = max_t + math.log(float(np.exp(log_terms - max_t).sum()))
    return log_prob / math.log(10)


# ---------------------------------------------------------------------------
# Individual scorers (private)
# ---------------------------------------------------------------------------


def _hyperscore(
    spectrum: Spectrum,
    matches: list[tuple[int, Fragment]],
) -> float:
    if not matches:
        return 0.0
    unique_idx = _unique_peak_indices(matches)
    dot = float(np.sum(spectrum.intensity[unique_idx]))
    if dot <= 0.0:
        return 0.0
    series_counts = {s: len(pos) for s, pos in _unique_series_positions(matches).items()}
    return float(math.log10(dot) + sum(_log10_factorial(n) for n in series_counts.values()))


def _probability_score(
    spectrum: Spectrum,
    matches: list[tuple[int, Fragment]],
    n_unique: int,
    mz_tol: float,
    mz_tol_type: Literal["Da", "ppm"],
) -> float:
    n_exp = len(spectrum.mz)
    k = len(_unique_peak_indices(matches))
    if k == 0 or n_exp == 0 or n_unique == 0:
        return 0.0
    mz_range = float(spectrum.mz[-1] - spectrum.mz[0])
    if mz_range <= 0.0:
        return 0.0
    if mz_tol_type == "ppm":
        tol_da = mz_tol * float(np.median(spectrum.mz)) / 1e6
    else:
        tol_da = float(mz_tol)
    p = min(1.0, 2.0 * tol_da * n_unique / mz_range)
    return float(-_binom_log10_survival(k, n_exp, p))


def _total_matched_intensity(
    spectrum: Spectrum,
    matches: list[tuple[int, Fragment]],
) -> float:
    if not matches:
        return 0.0
    return float(np.sum(spectrum.intensity[_unique_peak_indices(matches)]))


def _matched_fraction(
    matches: list[tuple[int, Fragment]],
    n_unique: int,
) -> float:
    if n_unique == 0:
        return 0.0
    return len(_unique_peak_indices(matches)) / n_unique


def _intensity_fraction(
    spectrum: Spectrum,
    matches: list[tuple[int, Fragment]],
) -> float:
    total = float(spectrum.intensity.sum())
    if total == 0.0 or not matches:
        return 0.0
    return _total_matched_intensity(spectrum, matches) / total


def _mean_ppm_error(
    spectrum: Spectrum,
    matches: list[tuple[int, Fragment]],
) -> float:
    if not matches:
        return 0.0
    errors = [
        abs(float(spectrum.mz[idx]) - float(frag.mz)) / float(frag.mz) * 1e6
        for idx, frag in matches
    ]
    return float(np.mean(errors))


def _spectral_angle(
    spectrum: Spectrum,
    matches: list[tuple[int, Fragment]],
    n_unique: int,
) -> float:
    if not matches or n_unique == 0:
        return 0.0
    unique_idx = _unique_peak_indices(matches)
    obs = spectrum.intensity[unique_idx]
    obs_norm = float(np.linalg.norm(obs))
    if obs_norm == 0.0:
        return 0.0
    dot = float(obs.sum()) / (obs_norm * math.sqrt(n_unique))
    dot = max(-1.0, min(1.0, dot))
    return float(1.0 - math.acos(dot) / (math.pi / 2))


def _longest_run(matches: list[tuple[int, Fragment]]) -> int:
    if not matches:
        return 0
    series_positions: dict[str, list[int]] = defaultdict(list)
    for _, frag in matches:
        pos = frag.position
        if isinstance(pos, int):
            series_positions[str(frag.ion_type)].append(pos)
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
    fragments: Sequence[Fragment],
    mz_tol: float = 0.02,
    mz_tol_type: Literal["Da", "ppm"] = "ppm",
    peak_selection: Literal["closest", "largest", "all"] = "closest",
) -> dict[str, float]:
    """Match fragments against a spectrum and return all scores.

    Internally calls :func:`~spxtacular.visualization.match_fragments` and
    computes ``n_theoretical`` as the number of unique ``(ion_type, position)``
    pairs, so neutral-loss and isotope variants of the same fragment do not
    inflate the scores.

    Parameters
    ----------
    spectrum:
        Experimental centroid spectrum.
    fragments:
        Theoretical fragment ions from peptacular.
    mz_tol:
        Matching tolerance.
    mz_tol_type:
        ``"Da"`` or ``"ppm"``.
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
    matches = match_fragments(spectrum, fragments, mz_tol, mz_tol_type, peak_selection)
    n_unique = _count_unique_ions(fragments)

    return {
        "hyperscore": _hyperscore(spectrum, matches),
        "probability_score": _probability_score(spectrum, matches, n_unique, mz_tol, mz_tol_type),
        "total_matched_intensity": _total_matched_intensity(spectrum, matches),
        "matched_fraction": _matched_fraction(matches, n_unique),
        "intensity_fraction": _intensity_fraction(spectrum, matches),
        "mean_ppm_error": _mean_ppm_error(spectrum, matches),
        "spectral_angle": _spectral_angle(spectrum, matches, n_unique),
        "longest_run": float(_longest_run(matches)),
    }
