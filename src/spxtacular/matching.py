"""
Fragment-to-peak matching.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray
from peptacular import Fragment, IonType

from .core import Spectrum
from .enums import (
    DEFAULT_FRAGMENT_TOLERANCE,
    DEFAULT_FRAGMENT_TOLERANCE_TYPE,
    PeakSelection,
    PeakSelectionLike,
    ToleranceLike,
    ToleranceType,
)
from .errors import SpxtacularError
from .utils import da_to_ppm

if TYPE_CHECKING:
    from paftacular import PafAnnotation

FragmentInput = Sequence[Fragment] | dict[tuple[IonType, int], list[float]]


@dataclass(frozen=True, slots=True, kw_only=True)
class MatchedFragment:
    """A confirmed fragment-to-peak match, carrying both the fragment and peak metadata.

    Attributes
    ----------
    fragment
        The matched peptacular :class:`~peptacular.Fragment`.
    peak_index
        Index of the matched peak in the spectrum passed to :func:`match_fragments`.
    peak_mz
        Observed m/z (or neutral mass, for a decharged spectrum).
    peak_intensity
        Observed intensity.
    intensity_pct
        ``peak_intensity`` as a percentage of the spectrum's total intensity.
    ppm_error
        Signed error ``(peak_mz - theoretical) / |theoretical| * 1e6``.
    da_error
        Signed error ``peak_mz - theoretical``.
    """

    fragment: Fragment
    peak_index: int
    peak_mz: float
    peak_intensity: float
    intensity_pct: float
    ppm_error: float
    da_error: float
    _annotation: PafAnnotation | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def annotation(self) -> PafAnnotation:
        """The match as an mzPAF :class:`~paftacular.PafAnnotation` (built once, then cached).

        It carries the fragment's own sequence and the ppm mass error of this match.
        ``annotation.serialize()`` gives the mzPAF string.
        """
        cached = self._annotation
        if cached is None:
            from paftacular import to_mzpaf

            cached = to_mzpaf(self.fragment, mass_error=self.ppm_error, include_sequence=True)
            object.__setattr__(self, "_annotation", cached)
        return cached


def match_fragments(
    spectrum: Spectrum,
    fragments: FragmentInput,
    *,
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
    TypeError
        If ``fragments`` is a string such as ``"PEPTIDE/2"``; build the fragments
        with peptacular first.

    Notes
    -----
    Matching adapts to the spectrum's processing state so the same call
    works for centroid, deconvoluted, and decharged spectra:

    * **Centroid / profile** (``spectrum.charge is None``) — match by m/z
      with no charge constraint.
    * **Deconvoluted** (``spectrum.charge`` has values > 0 or -1) — match by
      m/z; require the peak's assigned charge magnitude to equal the fragment
      charge magnitude. Singletons (``charge == -1``, unknown charge) are
      treated as a wildcard and may still match by m/z.
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
    # A str is a Sequence, so "PEPTIDE/2" would otherwise fail deep inside with
    # "'str' object has no attribute 'mz'".
    if isinstance(fragments, str | bytes):
        raise TypeError(
            f"fragments must be a sequence of peptacular Fragment objects or a fast_fragment dict, "
            f"not {type(fragments).__name__} {fragments!r}. Build them first, e.g. "
            'peptacular.fragment("PEPTIDE", ion_types=("b", "y")).'
        )
    tol_type = ToleranceType(str(tolerance_type).lower())
    selection = PeakSelection(str(peak_selection).lower())

    mz = spectrum.mz
    intensity = spectrum.intensity
    charge = spectrum.charge  # None for raw/centroid spectra
    total_intensity = float(intensity.sum())

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

    def _make_frag(ion_type: IonType, pos: int, charge_state: int, mz_val: float) -> Fragment:
        # Fragment.mz is ``mass / abs(charge_state)``, so the round trip from an
        # m/z back to a mass has to use the magnitude of the charge: a negative
        # charge_state would otherwise flip the sign of every derived m/z.
        return Fragment(
            ion_type=ion_type,
            position=pos,
            mass=mz_val * abs(charge_state),
            monoisotopic=is_monoisotopic,
            charge_state=charge_state,
        )

    # Flatten the input into parallel target / charge arrays. Fragments from a
    # dict are only built for targets that match, unless the spectrum is
    # decharged (the neutral-mass target needs the Fragment).
    frag_list: list[Fragment | None]
    dict_keys: list[tuple[IonType, int, int, float]] | None = None
    if isinstance(fragments, dict):
        frag_dict = cast(dict[tuple[IonType, int], list[float]], fragments)
        dict_keys = []
        for (ion_type, charge_state), masses in frag_dict.items():
            # Validate up front so the error does not depend on whether a peak
            # happened to match.
            if charge_state == 0:
                raise SpxtacularError(
                    f"fragment dict key ({ion_type!r}, 0) has charge_state == 0; "
                    "an m/z cannot be converted to a fragment mass without a charge"
                )
            dict_keys.extend((ion_type, charge_state, pos, float(v)) for pos, v in enumerate(masses, start=1))
        if is_decharged:
            frag_list = [_make_frag(it, pos, cs, v) for it, cs, pos, v in dict_keys]
            targets = np.array([cast(Fragment, f).neutral_mass for f in frag_list], dtype=np.float64)
        else:
            frag_list = [None] * len(dict_keys)
            targets = np.array([k[3] for k in dict_keys], dtype=np.float64)
        frag_charges = np.array([k[1] for k in dict_keys], dtype=np.int64)
    else:
        given = list(cast(Iterable[Fragment], fragments))
        attr = "neutral_mass" if is_decharged else "mz"
        targets = np.array([getattr(f, attr) for f in given], dtype=np.float64)
        frag_charges = np.array([f.charge_state for f in given], dtype=np.int64)
        frag_list = list(given)

    frag_idx, peak_idx, _ = _search(
        mz,
        targets,
        frag_charges,
        None if is_decharged else charge,
        intensity,
        tolerance=float(tolerance),
        ppm=tol_type is ToleranceType.PPM,
        selection=selection,
    )
    if frag_idx.size == 0:
        return []

    # Signed errors, and the order the caller sees: by peak index in the caller's
    # array, then by fragment input order.
    signed_da = mz[peak_idx] - targets[frag_idx]
    out_peak = unsort[peak_idx] if unsort is not None else peak_idx
    order = np.lexsort((frag_idx, out_peak))

    results: list[MatchedFragment] = []
    for k in order.tolist():
        fi = int(frag_idx[k])
        pi = int(peak_idx[k])
        frag = frag_list[fi]
        if frag is None:
            assert dict_keys is not None
            it, cs, pos, v = dict_keys[fi]
            frag = _make_frag(it, pos, cs, v)
            frag_list[fi] = frag
        target = float(targets[fi])
        da_err = float(signed_da[k])
        p_int = float(intensity[pi])
        results.append(
            MatchedFragment(
                fragment=frag,
                peak_index=int(out_peak[k]),
                peak_mz=float(mz[pi]),
                peak_intensity=p_int,
                intensity_pct=p_int / total_intensity * 100.0 if total_intensity > 0.0 else 0.0,
                ppm_error=da_to_ppm(da_err, target) if target != 0.0 else 0.0,
                da_error=da_err,
            )
        )
    return results


def _search(
    mz: NDArray[np.float64],
    targets: NDArray[np.float64],
    frag_charges: NDArray[np.int64],
    charge: NDArray[np.int32] | None,
    intensity: NDArray[np.float64],
    *,
    tolerance: float,
    ppm: bool,
    selection: PeakSelection,
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]]:
    """Vectorised window search over an m/z-sorted peak array.

    Returns parallel ``(fragment_index, peak_index, abs_delta)`` arrays of the
    selected matches. ``charge`` is ``None`` when charge is no constraint.

    A peak is in tolerance when its error (``|delta|`` in Da, or
    ``|delta| / |target| * 1e6`` in ppm; 0 for a zero target) is at most
    ``tolerance``. Ties resolve as the previous per-fragment walk did: the peak
    below the target wins over the one above, and among equal m/z values the
    one nearest the insertion point wins.
    """
    empty = (np.empty(0, np.intp), np.empty(0, np.intp), np.empty(0, np.float64))
    n_frag = targets.size
    if n_frag == 0 or mz.size == 0:
        return empty

    abs_t = np.abs(targets)
    if ppm:
        with np.errstate(divide="ignore", invalid="ignore"):
            half = np.where(abs_t != 0.0, tolerance * abs_t / 1e6, np.inf)
    else:
        half = np.full(n_frag, tolerance)
    # Pad the window a little and apply the exact test below, so rounding in
    # the bound never drops a peak the exact test would keep.
    pad = half * 1e-9 + 1e-12
    lo = np.searchsorted(mz, targets - half - pad, side="left")
    hi = np.searchsorted(mz, targets + half + pad, side="right")
    counts = hi - lo
    total = int(counts.sum())
    if total == 0:
        return empty

    fi = np.repeat(np.arange(n_frag, dtype=np.intp), counts)
    starts = np.repeat(lo - np.concatenate(([0], np.cumsum(counts)[:-1])), counts)
    pi = (np.arange(total, dtype=np.intp) + starts).astype(np.intp)

    t = targets[fi]
    delta = np.abs(mz[pi] - t)
    if ppm:
        with np.errstate(divide="ignore", invalid="ignore"):
            err = np.where(t != 0.0, delta / np.abs(t) * 1e6, 0.0)
    else:
        err = delta
    keep = err <= tolerance
    if charge is not None:
        pc = charge[pi].astype(np.int64)
        keep &= (pc == -1) | (np.abs(pc) == np.abs(frag_charges[fi]))
    if not keep.all():
        fi, pi, delta, t = fi[keep], pi[keep], delta[keep], t[keep]
    if fi.size == 0:
        return empty
    if selection is PeakSelection.ALL:
        return fi, pi, delta

    # Side of the target (0 = below, searched first) and nearness to the
    # insertion point within a side, reproducing the walk's tie-breaking.
    below = mz[pi] < t
    side = np.where(below, 0, 1)
    nearness = np.where(below, -pi, pi)
    primary = delta if selection is PeakSelection.CLOSEST else -intensity[pi]
    order = np.lexsort((nearness, side, primary, fi))
    fi_sorted = fi[order]
    first = np.ones(fi_sorted.size, dtype=bool)
    first[1:] = fi_sorted[1:] != fi_sorted[:-1]
    chosen = order[first]
    return fi[chosen], pi[chosen], delta[chosen]
