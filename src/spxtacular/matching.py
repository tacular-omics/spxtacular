"""
Fragment-to-peak matching.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from peptacular import IonType
from peptacular.annotation.frag import Fragment

from .core import Spectrum
from .enums import PeakSelection, PeakSelectionLike, ToleranceLike, ToleranceType

FragmentInput = Sequence[Fragment] | dict[tuple[IonType, int], list[float]]


@dataclass(frozen=True)
class MatchedFragment:
    """A confirmed fragment-to-peak match, carrying both the fragment and peak metadata."""

    fragment: Fragment
    peak_index: int
    peak_mz: float
    peak_intensity: float
    intensity_pct: float  # peak_intensity / total_spectrum_intensity * 100
    ppm_error: float  # signed: (peak_mz - theoretical_mz) / theoretical_mz * 1e6
    da_error: float  # signed: peak_mz - theoretical_mz


def match_fragments(
    spectrum: Spectrum,
    fragments: FragmentInput,
    tolerance: float = 0.02,
    tolerance_type: ToleranceLike = ToleranceType.PPM,
    peak_selection: PeakSelectionLike = PeakSelection.CLOSEST,
    is_monoisotopic: bool = True,
) -> list[MatchedFragment]:
    """Match a list of Fragment objects (or a fragment-masses dict) to spectrum peaks.

    Multiple fragments may match the same peak.

    Parameters
    ----------
    spectrum:
        Spectrum to search.  Must be sorted by m/z (standard for centroid data).
    fragments:
        Fragment objects from peptacular (each with a ``.mz`` property), **or** the
        ``dict[tuple[IonType, int], list[float]]`` returned by
        :meth:`~peptacular.ProFormaAnnotation.fragment_masses`.
    tolerance:
        Tolerance value.
    tolerance_type:
        ``"Da"`` for absolute or ``"ppm"`` for parts-per-million.
    peak_selection:
        How to resolve multiple peaks within tolerance for a single fragment:

        - ``"closest"`` — keep the peak with the smallest m/z error (default).
        - ``"largest"`` — keep the peak with the highest intensity.
        - ``"all"``     — keep every peak within tolerance.
    is_monoisotopic:
        Passed to the :class:`~peptacular.annotation.frag.Fragment` constructor
        when building fragments from a dict input.  Has no effect when
        ``fragments`` is already a ``Sequence[Fragment]``.

    Returns
    -------
    list of :class:`MatchedFragment` sorted by ``peak_index``.

    Notes
    -----
    When ``spectrum.charge`` is present (deconvoluted spectrum), a candidate
    peak is only accepted when its charge state equals ``fragment.charge_state``.
    Singletons (``charge == -1``) and peaks of wrong charge are excluded.
    """
    mz = spectrum.mz
    intensity = spectrum.intensity
    charge = spectrum.charge  # None for raw/centroid spectra
    total_intensity = float(intensity.sum())
    results: list[MatchedFragment] = []

    def _charge_ok(peak_idx: int, frag_charge: int) -> bool:
        if charge is None:
            return True
        return int(charge[peak_idx]) == frag_charge

    def _build_matched(peak_idx: int, frag: Fragment) -> MatchedFragment:
        p_mz = float(mz[peak_idx])
        p_int = float(intensity[peak_idx])
        theoretical_mz = frag.mz
        da_err = p_mz - theoretical_mz
        ppm_err = da_err / theoretical_mz * 1e6
        pct = p_int / total_intensity * 100.0 if total_intensity > 0.0 else 0.0
        return MatchedFragment(
            fragment=frag,
            peak_index=peak_idx,
            peak_mz=p_mz,
            peak_intensity=p_int,
            intensity_pct=pct,
            ppm_error=ppm_err,
            da_error=da_err,
        )

    def _search(frag_mz: float, frag_charge: int) -> list[tuple[int, float]]:
        """Return (peak_idx, abs_delta) candidates within tolerance."""
        idx = int(np.searchsorted(mz, frag_mz))
        candidates: list[tuple[int, float]] = []

        if peak_selection == "closest":
            for i in (idx - 1, idx):
                if 0 <= i < len(mz) and _charge_ok(i, frag_charge):
                    delta = abs(float(mz[i]) - frag_mz)
                    err = delta / frag_mz * 1e6 if tolerance_type == "ppm" else delta
                    if err <= tolerance:
                        candidates.append((i, delta))
        else:
            for i in range(idx - 1, -1, -1):
                delta = abs(float(mz[i]) - frag_mz)
                err = delta / frag_mz * 1e6 if tolerance_type == "ppm" else delta
                if err > tolerance:
                    break
                if _charge_ok(i, frag_charge):
                    candidates.append((i, delta))
            for i in range(idx, len(mz)):
                delta = abs(float(mz[i]) - frag_mz)
                err = delta / frag_mz * 1e6 if tolerance_type == "ppm" else delta
                if err > tolerance:
                    break
                if _charge_ok(i, frag_charge):
                    candidates.append((i, delta))

        return candidates

    def _emit(candidates: list[tuple[int, float]], frag: Fragment) -> None:
        if not candidates:
            return
        if peak_selection == "closest":
            best_i = min(candidates, key=lambda c: c[1])[0]
            results.append(_build_matched(best_i, frag))
        elif peak_selection == "largest":
            best_i = max(candidates, key=lambda c: float(intensity[c[0]]))[0]
            results.append(_build_matched(best_i, frag))
        else:  # "all"
            for i, _ in candidates:
                results.append(_build_matched(i, frag))

    if isinstance(fragments, dict):
        frag_dict = cast(dict[tuple[IonType, int], list[float]], fragments)
        for (ion_type, charge_state), masses in frag_dict.items():
            for pos, mz_val in enumerate(masses, start=1):
                candidates = _search(mz_val, charge_state)
                if candidates:
                    frag = Fragment(
                        ion_type=ion_type,
                        position=pos,
                        mass=mz_val * charge_state,
                        monoisotopic=is_monoisotopic,
                        charge_state=charge_state,
                    )
                    _emit(candidates, frag)
    else:
        for frag in fragments:
            candidates = _search(frag.mz, frag.charge_state)
            _emit(candidates, frag)

    results.sort(key=lambda m: m.peak_index)
    return results
