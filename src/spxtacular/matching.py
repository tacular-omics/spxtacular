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
from .enums import (
    DEFAULT_FRAGMENT_TOLERANCE,
    DEFAULT_FRAGMENT_TOLERANCE_TYPE,
    PeakSelection,
    PeakSelectionLike,
    ToleranceLike,
)
from .utils import da_to_ppm

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
    tolerance: float = DEFAULT_FRAGMENT_TOLERANCE,
    tolerance_type: ToleranceLike = DEFAULT_FRAGMENT_TOLERANCE_TYPE,
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
        :meth:`~peptacular.ProFormaAnnotation.fast_fragment`.
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
    Matching adapts to the spectrum's processing state so the same call
    works for centroid, deconvoluted, and decharged spectra:

    * **Centroid / profile** (``spectrum.charge is None``) — match by m/z
      with no charge constraint.
    * **Deconvoluted** (``spectrum.charge`` has values > 0 or -1) — match by
      m/z; require the peak's assigned charge to equal
      ``fragment.charge_state``. Singletons (``charge == -1``, unknown
      charge) are treated as a wildcard and may still match by m/z.
    * **Decharged** (every peak's ``charge`` is 0, i.e. neutral masses) —
      match the peak's neutral mass against ``fragment.neutral_mass``;
      ``charge_state`` is no longer a constraint, so multi-charge
      fragments collapse onto the same neutral target. The reported
      ``peak_mz`` is the stored neutral mass and the ppm/Da errors are
      against ``fragment.neutral_mass`` rather than ``fragment.mz``.
    """
    mz = spectrum.mz
    intensity = spectrum.intensity
    charge = spectrum.charge  # None for raw/centroid spectra
    total_intensity = float(intensity.sum())
    results: list[MatchedFragment] = []

    # Detect spectrum state once. Decharged spectra have every (non-dropped)
    # peak's charge set to 0; deconvoluted spectra carry per-peak charges in
    # {-1, 1, 2, ...}. A `charge` array of all -1 is treated as deconvoluted
    # (all singletons) — every peak is then a wildcard and falls back to m/z.
    is_decharged = spectrum.is_decharged

    def _target(frag: Fragment) -> float:
        return float(frag.neutral_mass) if is_decharged else float(frag.mz)

    def _charge_ok(peak_idx: int, frag_charge: int) -> bool:
        if charge is None or is_decharged:
            return True
        pc = int(charge[peak_idx])
        return pc == -1 or pc == frag_charge  # -1 = unknown, treat as wildcard

    def _ppm_err(delta: float, target_mz: float) -> float:
        return da_to_ppm(delta, target_mz) if target_mz != 0.0 else 0.0

    def _build_matched(peak_idx: int, frag: Fragment) -> MatchedFragment:
        p_mz = float(mz[peak_idx])
        p_int = float(intensity[peak_idx])
        theoretical_mz = _target(frag)
        da_err = p_mz - theoretical_mz
        ppm_err = _ppm_err(da_err, theoretical_mz)
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

    def _search(target_mz: float, frag_charge: int) -> list[tuple[int, float]]:
        """Return (peak_idx, abs_delta) candidates within tolerance."""
        idx = int(np.searchsorted(mz, target_mz))
        candidates: list[tuple[int, float]] = []

        if peak_selection == "closest":
            for i in (idx - 1, idx):
                if 0 <= i < len(mz) and _charge_ok(i, frag_charge):
                    delta = abs(float(mz[i]) - target_mz)
                    err = _ppm_err(delta, target_mz) if tolerance_type == "ppm" else delta
                    if err <= tolerance:
                        candidates.append((i, delta))
        else:
            for i in range(idx - 1, -1, -1):
                delta = abs(float(mz[i]) - target_mz)
                err = _ppm_err(delta, target_mz) if tolerance_type == "ppm" else delta
                if err > tolerance:
                    break
                if _charge_ok(i, frag_charge):
                    candidates.append((i, delta))
            for i in range(idx, len(mz)):
                delta = abs(float(mz[i]) - target_mz)
                err = _ppm_err(delta, target_mz) if tolerance_type == "ppm" else delta
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

    def _make_frag(ion_type: IonType, pos: int, charge_state: int, mz_val: float) -> Fragment:
        return Fragment(
            ion_type=ion_type,
            position=pos,
            mass=mz_val * charge_state,
            monoisotopic=is_monoisotopic,
            charge_state=charge_state,
        )

    if isinstance(fragments, dict):
        frag_dict = cast(dict[tuple[IonType, int], list[float]], fragments)
        for (ion_type, charge_state), masses in frag_dict.items():
            for pos, mz_val in enumerate(masses, start=1):
                # Only decharged spectra need the neutral-mass target, which requires
                # building the Fragment up front; otherwise defer construction until
                # a match is actually found (mz_val is already the search target).
                if is_decharged:
                    frag: Fragment | None = _make_frag(ion_type, pos, charge_state, mz_val)
                    target = _target(frag)
                else:
                    frag = None
                    target = mz_val
                candidates = _search(target, charge_state)
                if candidates:
                    if frag is None:
                        frag = _make_frag(ion_type, pos, charge_state, mz_val)
                    _emit(candidates, frag)
    else:
        for frag in fragments:
            candidates = _search(_target(frag), frag.charge_state)
            _emit(candidates, frag)

    results.sort(key=lambda m: m.peak_index)
    return results
