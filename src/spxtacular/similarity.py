"""
Spectrum-to-spectrum similarity.

:mod:`~spxtacular.scoring` answers "how well does this *peptide* explain this
spectrum". These functions answer "how alike are these two *spectra*", which is
what spectral library search, replicate comparison and clustering are built on.

    cosine(query, reference, tolerance=20)                  # 0-1
    modified_cosine(query, reference, 500.25, 508.28)       # tolerates a mass shift
    entropy_similarity(query, reference)                    # unweighted entropy

All three align peaks within a tolerance, **one-to-one**: a peak may back at most
one match. Greedy alignment by descending contribution is the convention here
(and what GNPS uses) -- taking every pair within tolerance instead would let one
intense peak match several and inflate the score above 1.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .core import Spectrum
from .enums import DEFAULT_FRAGMENT_TOLERANCE, DEFAULT_FRAGMENT_TOLERANCE_TYPE, ToleranceLike, ToleranceType

IntensityTransform = Literal["sqrt", "linear", "log"]


def _prepared(
    spectrum: Spectrum,
    transform: IntensityTransform,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """m/z ascending plus transformed, L2-normalised intensities."""
    mz = np.asarray(spectrum.mz, dtype=np.float64)
    inten = np.asarray(spectrum.intensity, dtype=np.float64)

    if mz.size > 1 and bool(np.any(mz[1:] < mz[:-1])):
        order = np.argsort(mz, kind="stable")
        mz, inten = mz[order], inten[order]

    inten = np.clip(inten, 0.0, None)
    if transform == "sqrt":
        # The convention for spectral matching: square-rooting compresses the
        # dynamic range so a single dominant peak cannot carry the whole score.
        inten = np.sqrt(inten)
    elif transform == "log":
        inten = np.log1p(inten)
    elif transform != "linear":
        raise ValueError(f"transform must be 'sqrt', 'linear' or 'log', got {transform!r}")

    norm = float(np.linalg.norm(inten))
    if norm > 0:
        inten = inten / norm
    return mz, inten


def _tolerance_da(target: NDArray[np.float64], tolerance: float, tol_type: ToleranceType) -> NDArray[np.float64]:
    if tol_type is ToleranceType.PPM:
        return target * tolerance / 1e6
    return np.full_like(target, tolerance)


def _greedy_align(
    mz_a: NDArray[np.float64],
    int_a: NDArray[np.float64],
    mz_b: NDArray[np.float64],
    int_b: NDArray[np.float64],
    tolerance: float,
    tol_type: ToleranceType,
    shifts: tuple[float, ...] = (0.0,),
) -> list[tuple[int, int]]:
    """Pair peaks one-to-one, strongest contribution first.

    ``shifts`` lists the m/z offsets to consider; ``(0.0,)`` is a plain cosine,
    and adding a precursor mass difference is what makes the modified cosine
    tolerate a modification.
    """
    pairs: list[tuple[float, int, int]] = []
    for shift in shifts:
        target = mz_a + shift
        tol = _tolerance_da(np.abs(target), tolerance, tol_type)
        lo = np.searchsorted(mz_b, target - tol, side="left")
        hi = np.searchsorted(mz_b, target + tol, side="right")
        for i in range(mz_a.size):
            for j in range(int(lo[i]), int(hi[i])):
                pairs.append((int_a[i] * int_b[j], i, j))

    pairs.sort(key=lambda t: t[0], reverse=True)

    used_a: set[int] = set()
    used_b: set[int] = set()
    matched: list[tuple[int, int]] = []
    for _, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched.append((i, j))
    return matched


def cosine(
    query: Spectrum,
    reference: Spectrum,
    tolerance: float = DEFAULT_FRAGMENT_TOLERANCE,
    tolerance_type: ToleranceLike = DEFAULT_FRAGMENT_TOLERANCE_TYPE,
    transform: IntensityTransform = "sqrt",
) -> float:
    """Cosine similarity between two spectra, 0 to 1.

    The standard spectral dot product: intensities are square-rooted by default,
    each spectrum is normalised to unit length, and peaks are matched one-to-one
    within ``tolerance``. The score is the sum of the matched products, so it is
    1.0 only when every peak pairs up with the same relative intensity, and
    unaffected by an overall scaling of either spectrum.

    Parameters
    ----------
    query, reference:
        Spectra to compare. Any m/z order is accepted.
    tolerance, tolerance_type:
        Peak matching window, ``"da"`` (default) or ``"ppm"``.
    transform:
        ``"sqrt"`` (default, the matching convention), ``"linear"`` or ``"log"``.

    Returns
    -------
    float in ``[0, 1]``. Returns ``0.0`` if either spectrum is empty or has no
    positive intensity.
    """
    tol_type = ToleranceType(str(tolerance_type).lower())
    mz_a, int_a = _prepared(query, transform)
    mz_b, int_b = _prepared(reference, transform)
    if mz_a.size == 0 or mz_b.size == 0 or int_a.sum() == 0 or int_b.sum() == 0:
        return 0.0

    total = sum(int_a[i] * int_b[j] for i, j in _greedy_align(mz_a, int_a, mz_b, int_b, tolerance, tol_type))
    # Both vectors are unit-length, so Cauchy-Schwarz bounds this at 1; clamp
    # only to absorb floating-point drift.
    return float(min(1.0, max(0.0, total)))


def modified_cosine(
    query: Spectrum,
    reference: Spectrum,
    query_precursor_mz: float,
    reference_precursor_mz: float,
    tolerance: float = DEFAULT_FRAGMENT_TOLERANCE,
    tolerance_type: ToleranceLike = DEFAULT_FRAGMENT_TOLERANCE_TYPE,
    transform: IntensityTransform = "sqrt",
) -> float:
    """Cosine that also matches peaks displaced by the precursor mass difference.

    Two spectra of the same molecule differing by one modification share many
    fragments, but every fragment containing the modified site is shifted by the
    modification's mass. A plain cosine sees those as mismatches and scores the
    pair as unrelated; allowing the precursor difference as a second alignment
    offset recovers them. This is the GNPS molecular-networking metric.

    Parameters
    ----------
    query, reference:
        Spectra to compare.
    query_precursor_mz, reference_precursor_mz:
        Precursor m/z of each. Their difference is the extra offset considered.
    tolerance, tolerance_type, transform:
        As for :func:`cosine`.

    Returns
    -------
    float in ``[0, 1]``. Equals :func:`cosine` when the precursors match.
    """
    tol_type = ToleranceType(str(tolerance_type).lower())
    mz_a, int_a = _prepared(query, transform)
    mz_b, int_b = _prepared(reference, transform)
    if mz_a.size == 0 or mz_b.size == 0 or int_a.sum() == 0 or int_b.sum() == 0:
        return 0.0

    shift = float(reference_precursor_mz) - float(query_precursor_mz)
    shifts = (0.0,) if shift == 0.0 else (0.0, shift)

    total = sum(int_a[i] * int_b[j] for i, j in _greedy_align(mz_a, int_a, mz_b, int_b, tolerance, tol_type, shifts))
    return float(min(1.0, max(0.0, total)))


def _spectral_entropy(intensity: NDArray[np.float64]) -> float:
    """Shannon entropy of an intensity vector normalised to a probability."""
    total = float(intensity.sum())
    if total <= 0:
        return 0.0
    p = intensity / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def entropy_similarity(
    query: Spectrum,
    reference: Spectrum,
    tolerance: float = DEFAULT_FRAGMENT_TOLERANCE,
    tolerance_type: ToleranceLike = DEFAULT_FRAGMENT_TOLERANCE_TYPE,
) -> float:
    """Unweighted entropy similarity between two spectra, from 0 to 1.

    Compare the entropy of the merged distribution with the original entropies.
    Intensities are clipped to nonnegative values and probability-normalized.
    Peaks are greedily aligned one-to-one. No entropy-dependent weighting,
    noise removal, precursor removal, or additional centroiding is applied.
    Preprocess inputs consistently before comparing scores.

    Returns
    -------
    float in ``[0, 1]``; ``0.0`` if either spectrum is empty.
    """
    tol_type = ToleranceType(str(tolerance_type).lower())
    # Entropy is defined on the raw intensity distribution, so no sqrt here.
    mz_a, int_a = _prepared(query, "linear")
    mz_b, int_b = _prepared(reference, "linear")
    if mz_a.size == 0 or mz_b.size == 0 or int_a.sum() == 0 or int_b.sum() == 0:
        return 0.0

    a = int_a / int_a.sum()
    b = int_b / int_b.sum()

    matched = _greedy_align(mz_a, a, mz_b, b, tolerance, tol_type)
    pair_a = {i for i, _ in matched}
    pair_b = {j for _, j in matched}

    # Merged distribution: matched peaks add, unmatched carry their own weight.
    merged = [a[i] + b[j] for i, j in matched]
    merged += [a[i] for i in range(a.size) if i not in pair_a]
    merged += [b[j] for j in range(b.size) if j not in pair_b]
    merged_arr = np.asarray(merged, dtype=np.float64)

    s_a, s_b = _spectral_entropy(a), _spectral_entropy(b)
    s_ab = _spectral_entropy(merged_arr / 2.0)

    # 1 - (2*S_ab - S_a - S_b) / ln(4); the normaliser puts the result in [0, 1].
    similarity = 1.0 - (2.0 * s_ab - s_a - s_b) / math.log(4)
    return float(min(1.0, max(0.0, similarity)))
