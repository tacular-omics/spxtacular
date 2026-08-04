"""
Fragment-to-peak matching.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
from numpy.typing import NDArray
from peptacular import IonType
from peptacular.annotation.frag import Fragment

from .core import Spectrum
from .enums import (
    DEFAULT_FRAGMENT_TOLERANCE,
    DEFAULT_FRAGMENT_TOLERANCE_TYPE,
    PeakSelection,
    PeakSelectionLike,
    ToleranceLike,
    ToleranceType,
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
        Spectrum to search. Any m/z order is accepted -- an unsorted spectrum is
        sorted internally and the reported ``peak_index`` still refers to the
        array you passed in. (timsTOF frames arrive ordered by ion mobility
        scan, so they are not globally m/z-sorted.)
    fragments:
        Fragment objects from peptacular (each with a ``.mz`` property), **or** the
        ``dict[tuple[IonType, int], list[float]]`` returned by
        :meth:`~peptacular.ProFormaAnnotation.fast_fragment`.
    tolerance:
        Tolerance value.
    tolerance_type:
        ``"da"`` for absolute or ``"ppm"`` for parts-per-million.  Matched
        case-insensitively; anything else raises ``ValueError``.
    peak_selection:
        How to resolve multiple peaks within tolerance for a single fragment
        (matched case-insensitively; anything else raises ``ValueError``):

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

    Raises
    ------
    ValueError
        If ``tolerance_type`` / ``peak_selection`` is not a recognised value, or if
        a dict key carries ``charge_state == 0`` (an m/z cannot be converted to a
        mass).

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
    # Normalise the string-ish inputs once: comparing raw strings further down
    # made ``"PPM"`` silently fall back to Da (a 10^6x too wide window) and any
    # ``peak_selection`` typo silently behave like ``"all"``.
    tol_type = ToleranceType(str(tolerance_type).lower())
    selection = PeakSelection(str(peak_selection).lower())

    mz = spectrum.mz
    intensity = spectrum.intensity
    charge = spectrum.charge  # None for raw/centroid spectra
    total_intensity = float(intensity.sum())
    results: list[MatchedFragment] = []

    # Every lookup below goes through np.searchsorted, which returns meaningless
    # positions on unsorted input -- silently missing or wrong matches rather than
    # an error. Unsorted input is not exotic: a timsTOF frame is ordered by ion
    # mobility scan and only sorted by m/z *within* each scan, so roughly half the
    # steps in a DReader MS1 frame descend. Sort a working copy and map the
    # reported peak indices back, so `peak_index` still refers to the caller's
    # array whatever order it arrived in.
    unsort: NDArray[np.intp] | None = None
    if mz.size > 1 and bool(np.any(mz[1:] < mz[:-1])):
        unsort = np.argsort(mz, kind="stable")
        mz = mz[unsort]
        intensity = intensity[unsort]
        if charge is not None:
            charge = charge[unsort]

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

    def _err(delta: float, target_mz: float) -> float:
        """``delta`` (always in Da) expressed in the active tolerance unit."""
        return _ppm_err(delta, target_mz) if tol_type is ToleranceType.PPM else delta

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

        if selection is PeakSelection.CLOSEST:
            # Walk outward from the insertion point on both sides. |delta| grows
            # monotonically away from `idx`, so the first charge-compatible peak on
            # each side is the nearest one there. Only the tolerance may stop a walk:
            # breaking on a charge mismatch would hide a compatible peak that sits
            # just beyond an incompatible immediate neighbour.
            for start, step in ((idx - 1, -1), (idx, 1)):
                i = start
                while 0 <= i < len(mz):
                    delta = abs(float(mz[i]) - target_mz)
                    if _err(delta, target_mz) > tolerance:
                        break
                    if _charge_ok(i, frag_charge):
                        candidates.append((i, delta))
                        break
                    i += step
        else:
            for i in range(idx - 1, -1, -1):
                delta = abs(float(mz[i]) - target_mz)
                if _err(delta, target_mz) > tolerance:
                    break
                if _charge_ok(i, frag_charge):
                    candidates.append((i, delta))
            for i in range(idx, len(mz)):
                delta = abs(float(mz[i]) - target_mz)
                if _err(delta, target_mz) > tolerance:
                    break
                if _charge_ok(i, frag_charge):
                    candidates.append((i, delta))

        return candidates

    def _emit(candidates: list[tuple[int, float]], frag: Fragment) -> None:
        if not candidates:
            return
        if selection is PeakSelection.CLOSEST:
            best_i = min(candidates, key=lambda c: c[1])[0]
            results.append(_build_matched(best_i, frag))
        elif selection is PeakSelection.LARGEST:
            best_i = max(candidates, key=lambda c: float(intensity[c[0]]))[0]
            results.append(_build_matched(best_i, frag))
        else:  # "all"
            for i, _ in candidates:
                results.append(_build_matched(i, frag))

    def _make_frag(ion_type: IonType, pos: int, charge_state: int, mz_val: float) -> Fragment:
        # Fragment.mz is ``mass / abs(charge_state)``, so the round trip from an
        # m/z back to a mass has to use the magnitude of the charge: a negative
        # charge_state would otherwise flip the sign of every derived m/z.
        if charge_state == 0:
            raise ValueError(
                f"fragment dict key ({ion_type!r}, 0) has charge_state == 0; "
                "an m/z cannot be converted to a fragment mass without a charge"
            )
        return Fragment(
            ion_type=ion_type,
            position=pos,
            mass=mz_val * abs(charge_state),
            monoisotopic=is_monoisotopic,
            charge_state=charge_state,
        )

    if isinstance(fragments, dict):
        frag_dict = cast(dict[tuple[IonType, int], list[float]], fragments)
        for (ion_type, charge_state), masses in frag_dict.items():
            # Validate up front so the error does not depend on whether a peak
            # happened to match (Fragment construction is deferred below).
            if charge_state == 0:
                raise ValueError(
                    f"fragment dict key ({ion_type!r}, 0) has charge_state == 0; "
                    "an m/z cannot be converted to a fragment mass without a charge"
                )
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

    if unsort is not None:
        # Indices currently refer to the sorted working copy; translate them back
        # to positions in the caller's array.
        results = [replace(m, peak_index=int(unsort[m.peak_index])) for m in results]

    results.sort(key=lambda m: m.peak_index)
    return results
