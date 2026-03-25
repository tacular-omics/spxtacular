"""
Fragment-to-peak matching.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from peptacular.annotation.frag import Fragment

from .core import Spectrum


def match_fragments(
    spectrum: Spectrum,
    fragments: Sequence[Fragment],
    tolerance: float = 0.02,
    tolerance_type: Literal["Da", "ppm"] = "Da",
    peak_selection: Literal["closest", "largest", "all"] = "closest",
) -> list[tuple[int, Fragment]]:
    """Match a list of Fragment objects to spectrum peaks.

    Multiple fragments may match the same peak.

    Parameters
    ----------
    spectrum:
        Spectrum to search.  Must be sorted by m/z (standard for centroid data).
    fragments:
        Fragment objects from peptacular, each with a ``.mz`` property.
    tolerance:
        Tolerance value.
    tolerance_type:
        ``"Da"`` for absolute or ``"ppm"`` for parts-per-million.
    peak_selection:
        How to resolve multiple peaks within tolerance for a single fragment:

        - ``"closest"`` — keep the peak with the smallest m/z error (default).
        - ``"largest"`` — keep the peak with the highest intensity.
        - ``"all"``     — keep every peak within tolerance.

    Returns
    -------
    list of ``(peak_index, fragment)`` pairs sorted by peak index.

    Notes
    -----
    When ``spectrum.charge`` is present (deconvoluted spectrum), a candidate
    peak is only accepted when its charge state equals ``fragment.charge_state``.
    Singletons (``charge == -1``) and peaks of wrong charge are excluded.
    """
    mz = spectrum.mz
    intensity = spectrum.intensity
    charge = spectrum.charge  # None for raw/centroid spectra
    results: list[tuple[int, Fragment]] = []

    def _charge_ok(peak_idx: int) -> bool:
        """Return True if peak charge matches fragment charge (or charge unknown)."""
        if charge is None:
            return True
        peak_charge = int(charge[peak_idx])
        return peak_charge == frag.charge_state

    for frag in fragments:
        frag_mz = frag.mz
        idx = int(np.searchsorted(mz, frag_mz))

        # For "closest" the two nearest candidates are always idx-1 and idx by
        # definition of binary search.  For the other modes we must scan the
        # full tolerance window.
        if peak_selection == "closest":
            candidates = []
            for i in (idx - 1, idx):
                if 0 <= i < len(mz) and _charge_ok(i):
                    delta = abs(mz[i] - frag_mz)
                    err = delta / frag_mz * 1e6 if tolerance_type == "ppm" else delta
                    if err <= tolerance:
                        candidates.append((i, delta))
            if candidates:
                best = min(candidates, key=lambda c: c[1])
                results.append((best[0], frag))
        else:
            candidates = []
            for i in range(idx - 1, -1, -1):
                delta = abs(mz[i] - frag_mz)
                err = delta / frag_mz * 1e6 if tolerance_type == "ppm" else delta
                if err > tolerance:
                    break
                if _charge_ok(i):
                    candidates.append((i, delta))
            for i in range(idx, len(mz)):
                delta = abs(mz[i] - frag_mz)
                err = delta / frag_mz * 1e6 if tolerance_type == "ppm" else delta
                if err > tolerance:
                    break
                if _charge_ok(i):
                    candidates.append((i, delta))

            if not candidates:
                continue

            if peak_selection == "largest":
                best_i = max(candidates, key=lambda c: float(intensity[c[0]]))[0]
                results.append((best_i, frag))
            else:  # "all"
                results.extend((i, frag) for i, _ in candidates)

    results.sort(key=lambda t: t[0])
    return results
