"""
Spectacular: A peptacular companion for mass spectrometry data
Core data structures for spectra
"""

import warnings
from dataclasses import dataclass, fields, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, Self

if TYPE_CHECKING:
    import pandas as pd
    import plotly.graph_objects as go

    from .matching import FragmentInput, MatchedFragment

import numpy as np
import peptacular as pt
from numpy.typing import NDArray

from .decon.scored import deconvolve_spectrum as _deconvolve
from .enums import (
    DEFAULT_FRAGMENT_TOLERANCE,
    DEFAULT_FRAGMENT_TOLERANCE_TYPE,
    ActivationTypeLike,
    AnalyzerLike,
    IMTypeLike,
    PeakSelection,
    PeakSelectionLike,
    PolarityLike,
    ToleranceLike,
    ToleranceType,
)
from .noise import estimate_noise_level

# ============================================================================
# Core Data Structures
# ============================================================================


def _centroid_peaks(
    mz: NDArray[np.float64],
    intensity: NDArray[np.float64],
    im: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """Centroid peaks using numpy-optimized vectorized Gaussian fitting."""
    if len(intensity) < 4:
        empty_im = np.empty(0, dtype=np.float64) if im is not None else None
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), empty_im

    # Match pymzml: start at index 2
    i_prev = intensity[1:-2]
    i_curr = intensity[2:-1]
    i_next = intensity[3:]

    mz_prev = mz[1:-2]
    mz_curr = mz[2:-1]
    mz_next = mz[3:]

    # Match pymzml peak detection exactly
    is_peak = (i_prev > 0) & (i_prev < i_curr) & (i_curr > i_next) & (i_next > 0)

    # Filter out peaks with irregular spacing
    dx1 = mz_curr - mz_prev
    dx2 = mz_next - mz_curr
    valid_spacing = ~((dx1 > dx2 * 10) | (dx1 * 10 < dx2))
    is_peak = is_peak & valid_spacing

    # Extract valid peaks
    x1 = mz_prev[is_peak]
    y1 = i_prev[is_peak]
    x2 = mz_curr[is_peak]
    y2 = i_curr[is_peak]
    x3 = mz_next[is_peak]
    y3 = i_next[is_peak]

    if len(y1) == 0:
        empty_im = np.empty(0, dtype=np.float64) if im is not None else None
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), empty_im

    # Handle y3 == y1 case
    y3_adjusted = np.where(y3 == y1, y3 + 0.01 * y1, y3)

    # Vectorized Gaussian fit
    with np.errstate(divide="ignore", invalid="ignore"):
        double_log = np.log(y2 / y1) / np.log(y3_adjusted / y1)
        numerator = double_log * (x1 * x1 - x3 * x3) - x1 * x1 + x2 * x2
        denominator = 2 * (x2 - x1) - 2 * double_log * (x3 - x1)
        mue = numerator / denominator

        c_squared_num = x2 * x2 - x1 * x1 - 2 * x2 * mue + 2 * x1 * mue
        c_squared_denom = 2 * np.log(y1 / y2)
        c_squared = c_squared_num / c_squared_denom

        a = y1 * np.exp((x1 - mue) * (x1 - mue) / (2 * c_squared))

    # Filter only invalid numerical results
    valid = np.isfinite(mue) & np.isfinite(a)

    # Handle ion mobility if present - use apex value
    im_result = None
    if im is not None:
        im_apex = im[2:-1][is_peak][valid]
        im_result = im_apex

    return mue[valid], a[valid], im_result


def _is_ppm(tolerance_type: ToleranceLike) -> bool:
    """Resolve a tolerance type to a ppm/Da flag, rejecting unknown values.

    Comparing ``tolerance_type == "ppm"`` directly is case-sensitive and falls
    through to Da for anything else, so ``"PPM"`` silently yields a window a
    million times too wide.  Coercing through the enum raises instead.
    """
    return ToleranceType(str(tolerance_type).lower()) == ToleranceType.PPM


@dataclass(frozen=True, slots=True)
class Peak:
    """Single peak in a spectrum."""

    mz: float
    intensity: float
    charge: int | None = None
    im: float | None = None
    iso_score: float | None = None

    def __repr__(self) -> str:
        parts = [f"mz={self.mz:.4f}", f"int={self.intensity:.2e}"]
        if self.charge is not None:
            parts.append(f"z={self.charge}")
        if self.im is not None:
            parts.append(f"im={self.im:.3f}")
        if self.iso_score is not None:
            parts.append(f"score={self.iso_score:.3f}")
        return f"Peak({', '.join(parts)})"


class SpectrumType(StrEnum):
    CENTROID = "centroid"
    PROFILE = "profile"
    DECONVOLUTED = "deconvoluted"


@dataclass(slots=True, eq=False)
class Spectrum:
    """Mass spectrum with optional charge and ion mobility dimensions."""

    mz: NDArray[np.float64]  # Shape: (n,)
    intensity: NDArray[np.float64]  # Shape: (n,)
    charge: NDArray[np.int32] | None = None  # Shape: (n,)
    im: NDArray[np.float64] | None = None  # Shape: (n,)
    iso_score: NDArray[np.float64] | None = None  # Shape: (n,) — isotope profile scores from scored deconvolution
    spectrum_type: SpectrumType | str | None = None
    denoised: str | None = None
    normalized: str | None = None

    def __post_init__(self):
        """Coerce array dtypes and validate shapes."""
        self.mz = np.asarray(self.mz, dtype=np.float64)
        self.intensity = np.asarray(self.intensity, dtype=np.float64)
        if self.charge is not None:
            self.charge = np.asarray(self.charge, dtype=np.int32)
        if self.im is not None:
            self.im = np.asarray(self.im, dtype=np.float64)
        if self.iso_score is not None:
            self.iso_score = np.asarray(self.iso_score, dtype=np.float64)
        self._validate()
        # A charge array on its own implies deconvolution only when the caller
        # did not say otherwise — an explicit spectrum_type is always honoured,
        # so centroid data carrying instrument-assigned charges is not silently
        # promoted (which would unlock decharge() on non-deconvoluted m/z).
        if self.charge is not None and self.spectrum_type is None:
            self.spectrum_type = SpectrumType.DECONVOLUTED

    def _validate(self) -> None:
        """Check that every optional array matches the m/z length."""
        n = len(self.mz)
        if len(self.intensity) != n:
            raise ValueError("mz and intensity must have same length")
        if self.charge is not None and len(self.charge) != n:
            raise ValueError("charge array must match mz length")
        if self.im is not None and len(self.im) != n:
            raise ValueError("im array must match mz length")
        if self.iso_score is not None and len(self.iso_score) != n:
            raise ValueError("score array must match mz length")

    def __eq__(self, other: object) -> bool:
        """Value equality; array fields compare element-wise.

        Defined explicitly because the dataclass-generated ``__eq__`` compares
        field tuples, which calls ``bool()`` on a numpy array and raises
        "truth value of an array is ambiguous" — breaking ``==``, ``in``,
        ``list.remove`` and any ``assert spec == expected``.
        """
        if other.__class__ is not self.__class__:
            return NotImplemented
        for f in fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                if a is None or b is None:
                    if a is not b:
                        return False
                elif not np.array_equal(a, b):
                    return False
            elif a != b:
                return False
        return True

    __hash__ = None  # type: ignore[assignment]  # mutable, and __eq__ is value-based

    # -------------------------------------------------------------------------
    # Peak Access
    # -------------------------------------------------------------------------

    @property
    def is_decharged(self) -> bool:
        """Whether every (non-dropped) peak has already been decharged (charge == 0)."""
        return self.charge is not None and len(self.charge) > 0 and bool(np.all(self.charge == 0))

    @property
    def peaks(self) -> list[Peak]:
        """Convert to list of Peak objects."""
        return [
            Peak(
                mz=float(self.mz[i]),
                intensity=float(self.intensity[i]),
                charge=int(self.charge[i]) if self.charge is not None else None,
                im=float(self.im[i]) if self.im is not None else None,
                iso_score=float(self.iso_score[i]) if self.iso_score is not None else None,
            )
            for i in range(len(self.mz))
        ]

    def top_peaks(
        self,
        n: int,
        by: Literal["intensity", "mz", "charge", "im", "score"] = "intensity",
        reverse: bool = True,
    ) -> list[Peak]:
        """Get top N peaks sorted by specified attribute."""
        if by == "intensity":
            sort_key = self._argsort_intensity
        elif by == "mz":
            sort_key = self._argsort_mz
        elif by == "charge":
            sort_key = self._argsort_charge
        elif by == "im":
            sort_key = self._argsort_im
        elif by == "score":
            sort_key = self._argsort_score
        else:
            raise ValueError(f"Unknown sort key: {by!r}")

        n = max(n, 0)
        if not reverse:
            indices = sort_key[:n]
        else:
            indices = sort_key[-n:][::-1] if n > 0 else sort_key[:0]

        return [
            Peak(
                mz=float(self.mz[i]),
                intensity=float(self.intensity[i]),
                charge=int(self.charge[i]) if self.charge is not None else None,
                im=float(self.im[i]) if self.im is not None else None,
                iso_score=float(self.iso_score[i]) if self.iso_score is not None else None,
            )
            for i in indices
        ]

    # -------------------------------------------------------------------------
    # Sort indices
    # -------------------------------------------------------------------------

    @property
    def _argsort_mz(self) -> NDArray[np.int64]:
        return np.argsort(self.mz)

    @property
    def _argsort_intensity(self) -> NDArray[np.int64]:
        return np.argsort(self.intensity)

    @property
    def _argsort_charge(self) -> NDArray[np.int64]:
        if self.charge is None:
            raise ValueError("Spectrum has no charge information")
        return np.argsort(self.charge)

    @property
    def _argsort_im(self) -> NDArray[np.int64]:
        if self.im is None:
            raise ValueError("Spectrum has no ion mobility information")
        return np.argsort(self.im)

    @property
    def _argsort_score(self) -> NDArray[np.int64]:
        if self.iso_score is None:
            raise ValueError("Spectrum has no score information")
        return np.argsort(self.iso_score)

    # -------------------------------------------------------------------------
    # Peak Finding
    # -------------------------------------------------------------------------

    def has_peak(
        self,
        target_mz: float,
        tolerance: float = 0.01,
        tolerance_type: ToleranceLike = ToleranceType.DA,
        target_charge: int | None = None,
        target_im: float | None = None,
        im_tol: float = 0.01,
    ) -> bool:
        """Check if spectrum contains a peak matching criteria."""
        matches = self._find_matching_peaks(target_mz, tolerance, tolerance_type, target_charge, target_im, im_tol)
        return len(matches) > 0

    def get_peak(
        self,
        target_mz: float,
        tolerance: float = 0.01,
        tolerance_type: ToleranceLike = ToleranceType.DA,
        target_charge: int | None = None,
        target_im: float | None = None,
        im_tol: float = 0.01,
        collision: Literal["largest", "closest"] = "largest",
    ) -> Peak | None:
        """Get single peak matching criteria."""
        matches = self._find_matching_peaks(target_mz, tolerance, tolerance_type, target_charge, target_im, im_tol)

        if len(matches) == 0:
            return None

        if collision == "largest":
            idx = matches[np.argmax(self.intensity[matches])]
        else:  # closest
            mz_diffs = np.abs(self.mz[matches] - target_mz)
            idx = matches[np.argmin(mz_diffs)]

        return Peak(
            mz=float(self.mz[idx]),
            intensity=float(self.intensity[idx]),
            charge=int(self.charge[idx]) if self.charge is not None else None,
            im=float(self.im[idx]) if self.im is not None else None,
            iso_score=float(self.iso_score[idx]) if self.iso_score is not None else None,
        )

    def get_peaks(
        self,
        target_mz: float,
        tolerance: float = 0.01,
        tolerance_type: ToleranceLike = ToleranceType.DA,
        target_charge: int | None = None,
        target_im: float | None = None,
        im_tol: float = 0.01,
    ) -> list[Peak]:
        """Get all peaks matching criteria."""
        matches = self._find_matching_peaks(target_mz, tolerance, tolerance_type, target_charge, target_im, im_tol)

        return [
            Peak(
                mz=float(self.mz[i]),
                intensity=float(self.intensity[i]),
                charge=int(self.charge[i]) if self.charge is not None else None,
                im=float(self.im[i]) if self.im is not None else None,
                iso_score=float(self.iso_score[i]) if self.iso_score is not None else None,
            )
            for i in matches
        ]

    def _find_matching_peaks(
        self,
        target_mz: float,
        tolerance: float,
        tolerance_type: ToleranceLike,
        target_charge: int | None,
        target_im: float | None,
        im_tol: float,
    ) -> NDArray[np.int64]:
        """Find indices of peaks matching criteria."""
        # m/z tolerance
        if _is_ppm(tolerance_type):
            tol_da = target_mz * tolerance / 1e6
        else:
            tol_da = tolerance

        mask = np.abs(self.mz - target_mz) <= tol_da

        # Charge filter
        if target_charge is not None and self.charge is not None:
            mask &= self.charge == target_charge

        # Ion mobility filter
        if target_im is not None and self.im is not None:
            mask &= np.abs(self.im - target_im) <= im_tol

        return np.where(mask)[0]

    # -------------------------------------------------------------------------
    # Filtering & Processing
    # -------------------------------------------------------------------------

    def filter(
        self,
        min_mz: float | None = None,
        max_mz: float | None = None,
        min_intensity: float | None = None,
        max_intensity: float | None = None,
        min_charge: int | None = None,
        max_charge: int | None = None,
        min_im: float | None = None,
        max_im: float | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        top_n: int | None = None,
        inplace: bool = False,
    ) -> Self:
        """Filter spectrum by various criteria.

        Raises ``ValueError`` if a criterion is given for a dimension this
        spectrum does not carry — silently ignoring it would return every peak
        and read as "nothing was filtered out".
        """
        for name, value, array in (
            ("charge", min_charge, self.charge),
            ("charge", max_charge, self.charge),
            ("im", min_im, self.im),
            ("im", max_im, self.im),
            ("iso_score", min_score, self.iso_score),
            ("iso_score", max_score, self.iso_score),
        ):
            if value is not None and array is None:
                raise ValueError(f"Cannot filter on {name}: this spectrum has no {name} array")

        mask = np.ones(len(self.mz), dtype=bool)

        if min_mz is not None:
            mask &= self.mz >= min_mz
        if max_mz is not None:
            mask &= self.mz <= max_mz
        if min_intensity is not None:
            mask &= self.intensity >= min_intensity
        if max_intensity is not None:
            mask &= self.intensity <= max_intensity
        if min_charge is not None and self.charge is not None:
            mask &= self.charge >= min_charge
        if max_charge is not None and self.charge is not None:
            mask &= self.charge <= max_charge
        if min_im is not None and self.im is not None:
            mask &= self.im >= min_im
        if max_im is not None and self.im is not None:
            mask &= self.im <= max_im
        if min_score is not None and self.iso_score is not None:
            mask &= self.iso_score >= min_score
        if max_score is not None and self.iso_score is not None:
            mask &= self.iso_score <= max_score

        # Apply top_n after other filters
        if top_n is not None:
            valid_indices = np.where(mask)[0]
            intensities = self.intensity[valid_indices]
            order = np.argsort(intensities)
            order = order[-top_n:] if top_n > 0 else order[:0]
            top_indices = valid_indices[order]
            mask = np.zeros(len(self.mz), dtype=bool)
            mask[top_indices] = True

        return self._apply_mask(mask, inplace=inplace)

    def normalize(self, method: Literal["max", "tic", "median"] = "max", inplace: bool = False) -> Self:
        """Normalize intensities."""

        # if already normalized, raise error
        if self.normalized is not None:
            warnings.warn(
                f"Spectrum is already normalized with method '{self.normalized}'",
                UserWarning,
                stacklevel=2,
            )
            return self if inplace else self.copy()

        if len(self.intensity) == 0:
            return self if inplace else self.copy()

        if method == "max":
            norm_factor = self.intensity.max()
        elif method == "tic":
            norm_factor = self.intensity.sum()
        else:  # median
            norm_factor = np.median(self.intensity)

        if norm_factor == 0:
            warnings.warn(
                "Cannot normalize a spectrum with all-zero intensity; returning unchanged",
                UserWarning,
                stacklevel=2,
            )
            return self if inplace else self.copy()

        if not np.isfinite(norm_factor):
            warnings.warn(
                f"Cannot normalize: {method!r} normalisation factor is {norm_factor} "
                "(intensity contains NaN or inf); returning unchanged",
                UserWarning,
                stacklevel=2,
            )
            return self if inplace else self.copy()

        return self.update(intensity=self.intensity / norm_factor, normalized=method, inplace=inplace)

    def denoise(
        self,
        method: Literal["mad", "percentile", "histogram", "baseline", "iterative_median"] | float | int = "mad",
        inplace: bool = False,
    ) -> Self:
        """Remove low-intensity noise peaks."""

        # if already denoised, raise error
        if self.denoised is not None:
            warnings.warn(
                f"Spectrum is already denoised with method '{self.denoised}'",
                UserWarning,
                stacklevel=2,
            )
            return self if inplace else self.copy()

        if len(self.intensity) == 0:
            # Estimating a noise level from no peaks yields NaN and a pair of
            # numpy RuntimeWarnings; the answer is trivially "nothing to remove".
            return self.update(denoised=str(method), inplace=inplace)

        threshold = estimate_noise_level(self.intensity, method=method)
        return self.filter(min_intensity=threshold, inplace=inplace).update(denoised=str(method), inplace=inplace)

    def merge(
        self,
        mz_tolerance: float = 0.01,
        mz_tolerance_type: ToleranceLike = ToleranceType.DA,
        im_tolerance: float = 0.05,
        im_tolerance_type: Literal["relative", "absolute"] = "relative",
        inplace: bool = False,
    ) -> Self:
        """Merge nearby peaks within a given m/z (and optionally ion-mobility) tolerance.

        Peaks are processed in order of decreasing intensity. For each peak,
        neighbours within the tolerance window are identified. The merged peak
        carries the intensity-weighted average m/z (and ion mobility if
        present) and the summed intensity. Charge arrays are preserved — only
        peaks with matching charge are merged together.

        Parameters
        ----------
        mz_tolerance:
            m/z tolerance for merging. Default ``0.01``.
        mz_tolerance_type:
            ``"da"`` or ``"ppm"``. Default ``"da"``.
        im_tolerance:
            Ion-mobility tolerance for merging (only used when ``self.im`` is
            present). Default ``0.05``.
        im_tolerance_type:
            ``"relative"`` (fraction of current peak's IM) or ``"absolute"``
            (in IM units). Default ``"relative"``.
        inplace:
            Whether to modify the spectrum in place. Default ``False``.

        Returns
        -------
        Self
            The merged spectrum.
        """
        # Ensure arrays are sorted by m/z for efficient searching
        sort_idx = np.argsort(self.mz)
        mz = self.mz[sort_idx]
        intensity = self.intensity[sort_idx]
        im = self.im[sort_idx] if self.im is not None else None
        charge = self.charge[sort_idx] if self.charge is not None else None
        iso_score = self.iso_score[sort_idx] if self.iso_score is not None else None

        # Sort by intensity descending for greedy clustering order
        # We need the original indices relative to the SORTED arrays
        intensity_order = np.argsort(intensity)[::-1]

        used_mask = np.zeros(len(mz), dtype=bool)

        new_mz_list = []
        new_intensity_list = []
        new_im_list = []
        new_charge_list = []
        new_score_list = []

        if mz_tolerance_type not in ("ppm", "da"):
            raise ValueError("mz_tolerance_type must be 'ppm' or 'da'")

        im_tol_type = im_tolerance_type.lower()
        if im_tol_type not in ("relative", "absolute"):
            raise ValueError("im_tolerance_type must be 'relative' or 'absolute'")

        is_ppm = mz_tolerance_type == "ppm"
        if not is_ppm:
            # Constant tolerance
            mz_tol_abs = mz_tolerance

        for idx in intensity_order:
            if used_mask[idx]:
                continue

            current_mz = mz[idx]
            current_charge = charge[idx] if charge is not None else None

            # Calculate tolerance
            if is_ppm:
                delta = current_mz * mz_tolerance / 1e6
            else:
                delta = mz_tol_abs

            # Find range
            min_mz = current_mz - delta
            max_mz = current_mz + delta

            # Binary search in sorted mz array
            left_idx = np.searchsorted(mz, min_mz, side="left")
            right_idx = np.searchsorted(mz, max_mz, side="right")

            # Identify candidates in window
            window_indices = np.arange(left_idx, right_idx)

            # Filter out already used
            # Note: idx is guaranteed to be in window_indices and unused
            valid_indices = window_indices[~used_mask[window_indices]]

            # Additional Charge filtering if charges are present
            if charge is not None and current_charge is not None:
                # We can only merge if charges match the charge of the primary peak
                # (which is current_charge)
                charge_match_mask = charge[valid_indices] == current_charge
                valid_indices = valid_indices[charge_match_mask]

            # Additional Ion Mobility filtering if IMs are present
            if im is not None:
                current_im = im[idx]
                candidate_ims = im[valid_indices]

                if im_tol_type == "relative":
                    im_delta = current_im * im_tolerance
                else:
                    # absolute
                    im_delta = im_tolerance

                im_mask = np.abs(candidate_ims - current_im) <= im_delta
                valid_indices = valid_indices[im_mask]

            if len(valid_indices) == 0:
                continue

            # Check if valid_indices contains any peaks
            window_mz = mz[valid_indices]
            window_int = intensity[valid_indices]

            total_intensity = np.sum(window_int)
            if total_intensity > 0:
                avg_mz = np.average(window_mz, weights=window_int)
            else:
                avg_mz = np.mean(window_mz)

            new_mz_list.append(avg_mz)
            new_intensity_list.append(total_intensity)

            if charge is not None:
                new_charge_list.append(current_charge)

            if iso_score is not None:
                # Keep the strongest isotopic evidence in the merged group; the
                # merged peak is the same feature, so the best-fitting cluster's
                # score is the one that still describes it.
                new_score_list.append(float(np.max(iso_score[valid_indices])))

            if im is not None:
                window_im = im[valid_indices]
                if total_intensity > 0:
                    avg_im = np.average(window_im, weights=window_int)
                else:
                    avg_im = np.mean(window_im)
                new_im_list.append(avg_im)

            # Mark as used
            used_mask[valid_indices] = True

        # Convert back to arrays
        new_mz = np.array(new_mz_list, dtype=np.float64)
        new_intensity = np.array(new_intensity_list, dtype=np.float64)
        new_im = np.array(new_im_list, dtype=np.float64) if im is not None else None
        new_charge = np.array(new_charge_list, dtype=np.int32) if charge is not None else None
        new_score = np.array(new_score_list, dtype=np.float64) if iso_score is not None else None

        # Sort result by m/z
        final_sort = np.argsort(new_mz)
        new_mz = new_mz[final_sort]
        new_intensity = new_intensity[final_sort]
        if new_im is not None:
            new_im = new_im[final_sort]
        if new_charge is not None:
            new_charge = new_charge[final_sort]
        if new_score is not None:
            new_score = new_score[final_sort]

        if inplace:
            self.mz = new_mz
            self.intensity = new_intensity
            self.im = new_im
            self.charge = new_charge
            self.iso_score = new_score
            return self

        return replace(
            self,
            mz=new_mz,
            intensity=new_intensity,
            im=new_im,
            charge=new_charge,
            iso_score=new_score,
        )

    def centroid(self, inplace: bool = False) -> Self:
        """
        Centroid profile peaks using Gaussian fitting.

        Converts profile mode spectra to centroid mode by detecting local maxima
        and fitting Gaussian peaks to determine precise peak centers.
        Ion mobility data is preserved if present.
        """
        if self.spectrum_type == SpectrumType.CENTROID:
            warnings.warn(
                "Spectrum is already centroided",
                UserWarning,
                stacklevel=2,
            )
            return self if inplace else self.copy()

        mz_cent, int_cent, im_cent = _centroid_peaks(self.mz, self.intensity, self.im)

        return self.update(
            mz=mz_cent,
            intensity=int_cent,
            spectrum_type=SpectrumType.CENTROID,
            charge=None,
            im=im_cent,
            # Centroiding changes the peak count, so any per-peak scores from a
            # previous deconvolution no longer line up and must be dropped.
            iso_score=None,
            inplace=inplace,
        )

    def _apply_mask(self, mask: NDArray[np.bool_], inplace: bool = False) -> Self:
        if inplace:
            self.mz = self.mz[mask]
            self.intensity = self.intensity[mask]
            if self.charge is not None:
                self.charge = self.charge[mask]
            if self.im is not None:
                self.im = self.im[mask]
            if self.iso_score is not None:
                self.iso_score = self.iso_score[mask]
            return self

        return replace(
            self,
            mz=self.mz[mask],
            intensity=self.intensity[mask],
            charge=self.charge[mask] if self.charge is not None else None,
            im=self.im[mask] if self.im is not None else None,
            iso_score=self.iso_score[mask] if self.iso_score is not None else None,
        )

    def sort(
        self,
        by: Literal["mz", "intensity", "charge", "im", "score"] = "mz",
        reverse: bool = False,
        inplace: bool = False,
    ) -> Self:
        """Return a spectrum with peaks sorted by the given attribute."""
        if by == "mz":
            order = self._argsort_mz
        elif by == "intensity":
            order = self._argsort_intensity
        elif by == "charge":
            order = self._argsort_charge
        elif by == "im":
            order = self._argsort_im
        elif by == "score":
            order = self._argsort_score
        else:
            raise ValueError(f"Unknown sort key: {by!r}")

        if reverse:
            order = order[::-1]

        return self._apply_index(order, inplace=inplace)

    def copy(self) -> Self:
        """Return a deep copy of this spectrum with all arrays copied."""
        return replace(
            self,
            mz=self.mz.copy(),
            intensity=self.intensity.copy(),
            charge=self.charge.copy() if self.charge is not None else None,
            im=self.im.copy() if self.im is not None else None,
            iso_score=self.iso_score.copy() if self.iso_score is not None else None,
        )

    @classmethod
    def combine(cls, spectra: list["Spectrum"]) -> "Spectrum":
        """Concatenate peaks from multiple spectra into a single new Spectrum.

        Peaks are sorted by m/z ascending. Optional per-peak arrays (charge,
        im, score) are included only if **all** spectra carry that array;
        otherwise the field is dropped (set to None). Scalar metadata
        (spectrum_type, normalized, denoised) is preserved when all spectra
        share the same value, otherwise set to None.

        MsnSpectrum instances are accepted as input but the return type is
        always the base Spectrum — per-scan MSn metadata is not combinable.

        Parameters
        ----------
        spectra:
            List of Spectrum (or MsnSpectrum) objects to combine.

        Returns
        -------
        Spectrum
            A new Spectrum containing all peaks, sorted by m/z.

        Raises
        ------
        ValueError
            If spectra is empty.
        """
        if not spectra:
            raise ValueError("combine() requires at least one Spectrum")

        combined_mz = np.concatenate([s.mz for s in spectra])
        combined_intensity = np.concatenate([s.intensity for s in spectra])

        combined_charge: NDArray[np.int32] | None
        combined_im: NDArray[np.float64] | None
        combined_score: NDArray[np.float64] | None

        if all(s.charge is not None for s in spectra):
            combined_charge = np.concatenate([s.charge for s in spectra])  # type: ignore[misc]
        else:
            combined_charge = None

        if all(s.im is not None for s in spectra):
            combined_im = np.concatenate([s.im for s in spectra])  # type: ignore[misc]
        else:
            combined_im = None

        if all(s.iso_score is not None for s in spectra):
            combined_score = np.concatenate([s.iso_score for s in spectra])  # type: ignore[misc]
        else:
            combined_score = None

        sort_idx = np.argsort(combined_mz, kind="stable")
        combined_mz = combined_mz[sort_idx]
        combined_intensity = combined_intensity[sort_idx]
        if combined_charge is not None:
            combined_charge = combined_charge[sort_idx]
        if combined_im is not None:
            combined_im = combined_im[sort_idx]
        if combined_score is not None:
            combined_score = combined_score[sort_idx]

        types = {s.spectrum_type for s in spectra}
        spectrum_type: SpectrumType | str | None = types.pop() if len(types) == 1 else None

        normalized_vals = {s.normalized for s in spectra}
        normalized: str | None = normalized_vals.pop() if len(normalized_vals) == 1 else None

        denoised_vals = {s.denoised for s in spectra}
        denoised: str | None = denoised_vals.pop() if len(denoised_vals) == 1 else None

        return Spectrum(
            mz=combined_mz,
            intensity=combined_intensity,
            charge=combined_charge,
            im=combined_im,
            iso_score=combined_score,
            spectrum_type=spectrum_type,
            normalized=normalized,
            denoised=denoised,
        )

    def _apply_index(self, idx: NDArray[np.intp], inplace: bool = False) -> Self:
        if inplace:
            self.mz = self.mz[idx]
            self.intensity = self.intensity[idx]
            if self.charge is not None:
                self.charge = self.charge[idx]
            if self.im is not None:
                self.im = self.im[idx]
            if self.iso_score is not None:
                self.iso_score = self.iso_score[idx]
            return self

        return replace(
            self,
            mz=self.mz[idx],
            intensity=self.intensity[idx],
            charge=self.charge[idx] if self.charge is not None else None,
            im=self.im[idx] if self.im is not None else None,
            iso_score=self.iso_score[idx] if self.iso_score is not None else None,
        )

    def update(self, inplace: bool = False, **kwargs) -> Self:
        """Create new spectrum with updated fields.

        The returned spectrum never shares an array buffer with this one: any
        array field the caller did not replace is copied.  Without that, methods
        documented as "returning a new Spectrum" hand back views, and writing
        into the result silently mutates the original.

        The inplace path re-validates afterwards, so a partial update that
        leaves arrays at mismatched lengths raises instead of leaving the
        object quietly inconsistent.
        """
        if inplace:
            for k, v in kwargs.items():
                setattr(self, k, v)
            self._validate()
            return self

        for name in ("mz", "intensity", "charge", "im", "iso_score"):
            if name not in kwargs:
                current = getattr(self, name)
                if current is not None:
                    kwargs[name] = current.copy()

        return replace(self, **kwargs)

    # -------------------------------------------------------------------------
    # Plotting (requires plotly)
    # -------------------------------------------------------------------------

    def plot(
        self,
        title: str | None = None,
        *,
        color: "Literal['charge', 'im'] | None" = "charge",
        show_scores: bool = True,
        show_charges: bool | None = None,
        **layout_kwargs,
    ) -> "go.Figure":
        """Plot spectrum as a stick plot (requires plotly).

        Parameters
        ----------
        title:
            Plot title. Defaults to the spectrum type.
        color:
            Coloring mode — ``"charge"``, ``"im"``, or ``None``.
            See :func:`~spxtacular.plot_spectrum` for details.
        show_scores:
            Annotate peaks with isotope profile scores when score data is present.
        show_charges:
            Deprecated. Use ``color="charge"`` or ``color=None`` instead.
        **layout_kwargs:
            Forwarded to ``fig.update_layout``.
        """
        from .visualization import plot_spectrum

        return plot_spectrum(
            self,
            title=title,
            color=color,
            show_scores=show_scores,
            show_charges=show_charges,
            **layout_kwargs,
        )

    def plot_table(
        self,
        show_charges: bool = True,
        show_scores: bool = True,
    ) -> "pd.DataFrame":
        """Return an editable plot table (one row per peak) for this spectrum.

        The returned :class:`pandas.DataFrame` contains every data field
        (``mz``, ``intensity``, ``charge``, ``score``, ``im``) plus visual
        properties (``color``, ``linewidth``, ``opacity``, ``series``,
        ``label``, ``label_size``, ``label_font``, ``label_color``,
        ``label_yshift``, ``label_xanchor``, ``hover``).

        Modify the DataFrame freely, then pass it to
        :func:`spxtacular.plot_from_table` to produce a plotly Figure.

        Parameters
        ----------
        show_charges:
            Colour peaks by charge state when charge data is present.
        show_scores:
            Label peaks with their isotope profile score (score > 0 only).

        Returns
        -------
        pd.DataFrame
        """
        from .plot_table import build_plot_table

        return build_plot_table(self, show_charges=show_charges, show_scores=show_scores)

    def annot_plot_table(
        self,
        fragments: "FragmentInput",
        tolerance: float = DEFAULT_FRAGMENT_TOLERANCE,
        tolerance_type: ToleranceLike = DEFAULT_FRAGMENT_TOLERANCE_TYPE,
        peak_selection: PeakSelectionLike = PeakSelection.CLOSEST,
        include_sequence: bool = False,
    ) -> "pd.DataFrame":
        """Return an editable annotated plot table for this spectrum.

        Like :meth:`plot_table` but matched peaks are coloured by ion series
        and labelled with their fragment identifier.

        Parameters
        ----------
        fragments:
            Fragment objects from peptacular to match against peaks.
        tolerance:
            Matching tolerance.
        tolerance_type:
            ``"Da"`` or ``"ppm"``.
        peak_selection:
            ``"closest"``, ``"largest"``, or ``"all"``.
        include_sequence:
            Embed residue sequence in labels (e.g. ``b3{PEP}``).

        Returns
        -------
        pd.DataFrame
        """
        from .plot_table import build_annot_plot_table

        return build_annot_plot_table(self, fragments, tolerance, tolerance_type, peak_selection, include_sequence)

    def annotate(
        self,
        fragments: "FragmentInput",
        tolerance: float = DEFAULT_FRAGMENT_TOLERANCE,
        tolerance_type: ToleranceLike = DEFAULT_FRAGMENT_TOLERANCE_TYPE,
        title: str | None = None,
        peak_selection: PeakSelectionLike = PeakSelection.CLOSEST,
        include_sequence: bool = False,
        **layout_kwargs,
    ) -> "go.Figure":
        """Plot this spectrum with matched fragment ion annotations.

        Matched peaks are coloured by ion series (b=blue, y=red, …) and
        labelled; unmatched peaks are drawn in grey.

        Parameters
        ----------
        fragments:
            Fragment objects from peptacular to match against peaks.
        tolerance:
            Matching tolerance.
        tolerance_type:
            ``"Da"`` or ``"ppm"``.
        title:
            Plot title.
        peak_selection:
            Which peak to annotate per fragment — ``"closest"``, ``"largest"``,
            or ``"all"``.
        include_sequence:
            Embed the residue sequence in each label (e.g. ``b3{PEP}``).
        **layout_kwargs:
            Forwarded to ``fig.update_layout``.

        Returns
        -------
        plotly ``Figure``.
        """
        from .visualization import annotate_spectrum

        return annotate_spectrum(
            self,
            fragments,
            tolerance=tolerance,
            tolerance_type=tolerance_type,
            title=title,
            peak_selection=peak_selection,
            include_sequence=include_sequence,
            **layout_kwargs,
        )

    def deconvolute(
        self,
        tolerance: float = 50,
        tolerance_type: ToleranceLike = ToleranceType.PPM,
        charge_range: tuple[int, int] = (1, 3),
        intensity: Literal["base", "total"] = "total",
        max_dpeaks: int = 2000,
        inplace: bool = False,
        min_intensity: float | Literal["min"] = "min",
        min_score: float = 0.0,
    ) -> Self:
        min_charge, max_charge = charge_range
        if min_charge < 1 or max_charge < min_charge:
            raise ValueError(f"charge_range must be a (min, max) tuple with 1 <= min <= max; got {charge_range!r}")
        if self.spectrum_type == SpectrumType.DECONVOLUTED:
            warnings.warn(
                "Spectrum is already deconvoluted",
                UserWarning,
                stacklevel=2,
            )
            return self if inplace else self.copy()

        is_ppm = _is_ppm(tolerance_type)
        if min_intensity == "min":
            # Guard the reduction: an empty spectrum has no min. _deconvolve handles
            # the empty case and returns empty arrays, giving an empty DECONVOLUTED spectrum.
            resolved_min_intensity = float(self.intensity.min()) if len(self.intensity) else 0.0
        else:
            resolved_min_intensity = float(min_intensity)

        new_mz, new_charge, new_intensity, new_score = _deconvolve(
            mz=self.mz,
            intensity=self.intensity,
            charge_range=charge_range,
            tolerance=tolerance,
            is_ppm=is_ppm,
            max_dpeaks=max_dpeaks,
            intensity_mode=intensity,
            min_intensity=resolved_min_intensity,
            min_score=min_score,
        )

        # Carry ion mobility through. Every output m/z is an exact copy of the
        # m/z of the peak the cluster was anchored on, so the anchor's IM can be
        # recovered by exact lookup. Dropping it would destroy the IM dimension
        # for timsTOF data, which is the main reason DReader exists.
        new_im = None
        if self.im is not None and len(new_mz) > 0:
            order = np.argsort(self.mz)
            pos = np.searchsorted(self.mz[order], new_mz)
            pos = np.clip(pos, 0, len(order) - 1)
            new_im = self.im[order[pos]]

        return self.update(
            mz=new_mz,
            intensity=new_intensity,
            charge=new_charge,
            im=new_im,
            iso_score=new_score,
            spectrum_type=SpectrumType.DECONVOLUTED,
            inplace=inplace,
        )

    def decharge(self, inplace: bool = False) -> Self:
        """
        Decharge spectrum by converting m/z to neutral mass using charge information.

        Peaks with charge == -1 are dropped (charge unknown).

        Raises:
            ValueError: if the spectrum has not been deconvoluted yet. Call
                ``deconvolute()`` first so the charge states are known.

        Returns a new Spectrum with m/z values as neutral masses, sorted ascending.
        """
        if self.charge is None:
            raise ValueError("Cannot decharge a spectrum with no charge array; call deconvolute() first.")
        if self.spectrum_type != SpectrumType.DECONVOLUTED:
            raise ValueError("Cannot decharge a non-deconvoluted spectrum; call deconvolute() first.")

        if self.is_decharged:
            warnings.warn(
                "Spectrum is already decharged",
                UserWarning,
                stacklevel=2,
            )
            return self if inplace else self.copy()

        proton = pt.PROTON_MASS

        # charge > 0, not != -1: a charge of 0 means "already decharged", and
        # multiplying by it would collapse the peak to a neutral mass of 0.0.
        known = self.charge > 0
        known_mz = self.mz[known]
        known_charge = self.charge[known]
        known_int = self.intensity[known]
        known_im = self.im[known] if self.im is not None else None
        known_score = self.iso_score[known] if self.iso_score is not None else None

        neutral_mz = (known_mz * known_charge) - (known_charge * proton)

        order = np.argsort(neutral_mz)

        return self.update(
            mz=neutral_mz[order],
            intensity=known_int[order],
            charge=np.zeros_like(known_charge[order], dtype=np.int32),
            im=known_im[order] if known_im is not None else None,
            iso_score=known_score[order] if known_score is not None else None,
            inplace=inplace,
        )

    def __str__(self) -> str:
        return (
            f"Spectrum(n_peaks={len(self.mz)}, type={self.spectrum_type}, "
            f"denoised={self.denoised}, normalized={self.normalized})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    # -------------------------------------------------------------------------
    # Serialization (spectrl token format — see spectrl_bridge.py)
    # -------------------------------------------------------------------------

    def to_spectrl_token(self, *, lossless: bool = False, max_len: int | None = None) -> str:
        """Encode this spectrum as a ``spectrl1.…`` URL-safe token (requires
        ``spxtacular[spectrl]``).

        See :func:`spxtacular.spectrl_bridge.to_spectrl_token`.
        """
        from .spectrl_bridge import to_spectrl_token

        return to_spectrl_token(self, lossless=lossless, max_len=max_len)

    @classmethod
    def from_spectrl_token(cls, token: str) -> "Spectrum":
        """Decode a ``spectrl1.…`` token into a :class:`Spectrum` /
        :class:`MsnSpectrum` (requires ``spxtacular[spectrl]``).

        See :func:`spxtacular.spectrl_bridge.from_spectrl_token`.
        """
        from .spectrl_bridge import from_spectrl_token

        return from_spectrl_token(token)

    def to_spectrl_url(
        self,
        base: str | None = None,
        *,
        mode: str = "fragment",
        param: str = "d",
        lossless: bool = False,
        max_len: int | None = None,
    ) -> str:
        """Encode this spectrum into a shareable URL or ``data:`` URI (requires
        ``spxtacular[spectrl]``).

        See :func:`spxtacular.spectrl_bridge.to_spectrl_url`.
        """
        from .spectrl_bridge import to_spectrl_url

        return to_spectrl_url(self, base, mode=mode, param=param, lossless=lossless, max_len=max_len)

    @classmethod
    def from_spectrl_url(cls, url: str) -> "Spectrum":
        """Decode a spectrum from a URL fragment, query string, or ``data:`` URI
        carrying a ``spectrl1.…`` token (requires ``spxtacular[spectrl]``).

        See :func:`spxtacular.spectrl_bridge.from_spectrl_url`.
        """
        from .spectrl_bridge import from_spectrl_url

        return from_spectrl_url(url)

    @classmethod
    def from_usi(
        cls,
        usi: str,
        backend: str = "aggregator",
        timeout: float = 30,
    ) -> "Spectrum":
        """Load a spectrum from a public repository via Universal Spectrum Identifier.

        Uses the PROXI protocol to fetch spectra from aggregated proteomics
        repositories (PRIDE, MassIVE, PeptideAtlas, jPOST).

        Parameters
        ----------
        usi:
            Universal Spectrum Identifier, e.g.
            ``"mzspec:PXD000561:Adult_Frontalcortex_bRP_Elite_85_f09:scan:17555"``.
        backend:
            PROXI backend: ``"aggregator"`` (default), ``"pride"``,
            ``"massive"``, ``"peptideatlas"``, ``"jpost"``, or a full URL.
        timeout:
            HTTP request timeout in seconds.

        Returns
        -------
        Spectrum or MsnSpectrum
            :class:`MsnSpectrum` if precursor info is available, else
            :class:`Spectrum`.
        """
        from .usi import fetch_usi

        return fetch_usi(usi, backend=backend, timeout=timeout)

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _meta_dict(self) -> dict:
        """Build the JSON-serialisable metadata dict for :meth:`save`.

        Subclasses extend this to persist additional fields.
        """
        st = self.spectrum_type
        return {
            "spectrum_type": st if isinstance(st, str) else (st.value if st is not None else None),
            "denoised": self.denoised,
            "normalized": self.normalized,
        }

    @classmethod
    def _meta_kwargs(cls, meta: dict) -> dict:
        """Convert a loaded ``meta`` dict to constructor kwargs.

        Subclasses extend this to populate additional fields.
        """
        return {
            "spectrum_type": meta.get("spectrum_type"),
            "denoised": meta.get("denoised"),
            "normalized": meta.get("normalized"),
        }

    def save(self, path: str | Path) -> None:
        """Save spectrum to a ``.npz`` file.

        Arrays (``mz``, ``intensity``, and any optional ``charge``, ``im``,
        ``iso_score``) are stored natively; all scalar metadata is stored as a
        JSON string under the ``meta`` key.  The file extension ``.npz`` is
        appended automatically if absent.
        """
        import json

        def _json_default(obj):
            # Reader-produced metadata often holds numpy scalars. np.float64
            # subclasses float so json handles it, but np.int32 does not and
            # would abort the save with a bare TypeError.
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        arrays: dict = {
            "mz": self.mz,
            "intensity": self.intensity,
            "meta": np.array(json.dumps(self._meta_dict(), default=_json_default), dtype=object),
        }
        if self.charge is not None:
            arrays["charge"] = self.charge
        if self.im is not None:
            arrays["im"] = self.im
        if self.iso_score is not None:
            arrays["iso_score"] = self.iso_score
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a spectrum from a ``.npz`` file written by :meth:`save`."""
        import json

        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        # Back-compat: pre-unified Spectrum.save() used the key "score".
        if "iso_score" in data:
            iso_score = data["iso_score"]
        elif "score" in data:
            iso_score = data["score"]
        else:
            iso_score = None
        return cls(
            mz=data["mz"],
            intensity=data["intensity"],
            charge=data["charge"] if "charge" in data else None,
            im=data["im"] if "im" in data else None,
            iso_score=iso_score,
            **cls._meta_kwargs(meta),
        )

    # -------------------------------------------------------------------------
    # Fragment matching
    # -------------------------------------------------------------------------

    def match_fragments(
        self,
        fragments: "FragmentInput",
        tolerance: float = DEFAULT_FRAGMENT_TOLERANCE,
        tolerance_type: ToleranceLike = DEFAULT_FRAGMENT_TOLERANCE_TYPE,
        peak_selection: PeakSelectionLike = PeakSelection.CLOSEST,
        is_monoisotopic: bool = True,
    ) -> "list[MatchedFragment]":
        """Match fragment ions against this spectrum's peaks.

        Thin wrapper around :func:`~spxtacular.matching.match_fragments`.
        Returns a list of :class:`~spxtacular.matching.MatchedFragment` objects
        sorted by ascending ``peak_index``.
        """
        from .matching import match_fragments as _match

        return _match(
            self,
            fragments,
            tolerance=tolerance,
            tolerance_type=tolerance_type,
            peak_selection=peak_selection,
            is_monoisotopic=is_monoisotopic,
        )

    def score(
        self,
        fragments: "FragmentInput",
        tolerance: float = DEFAULT_FRAGMENT_TOLERANCE,
        tolerance_type: ToleranceLike = DEFAULT_FRAGMENT_TOLERANCE_TYPE,
        peak_selection: PeakSelectionLike = PeakSelection.CLOSEST,
    ) -> "dict[str, float]":
        """Match fragments and return all PSM scores.

        Thin wrapper around :func:`~spxtacular.scoring.score`.
        """
        from .scoring import score as _score

        return _score(
            self, fragments, tolerance=tolerance, tolerance_type=tolerance_type, peak_selection=peak_selection
        )

    # -------------------------------------------------------------------------
    # Precursor Peak Removal
    # -------------------------------------------------------------------------

    def remove_precursor_peak(
        self,
        precursor_mz: float | None = None,
        precursor_charge: int | None = None,
        tolerance: float = DEFAULT_FRAGMENT_TOLERANCE,
        tolerance_type: ToleranceLike = DEFAULT_FRAGMENT_TOLERANCE_TYPE,
        isotopes: int | Literal["auto"] = "auto",
        isotope_threshold: float = 0.01,
        remove_charge_states: bool = True,
        inplace: bool = False,
    ) -> Self:
        """Remove precursor peak(s), their isotope envelope, and charge states.

        When called on an :class:`MsnSpectrum` without explicit ``precursor_mz``,
        the method auto-detects precursor information from
        :attr:`MsnSpectrum.precursors` and removes peaks for **all** precursors.

        The method adapts its behaviour to the spectrum state:

        * **Centroid** — removes all charge states (1 … ``precursor_charge``)
          and their isotope envelopes.
        * **Deconvoluted** — isotope peaks have already been collapsed; only
          the monoisotopic peak at the precursor charge is targeted
          (charge-aware matching).
        * **Decharged** — m/z values are neutral masses; the precursor neutral
          mass is targeted directly.
        * **Profile** — raises ``ValueError`` (centroid first).

        Parameters
        ----------
        precursor_mz:
            Precursor m/z to remove.  If ``None``, auto-detected from
            ``self.precursors`` (requires :class:`MsnSpectrum`).
        precursor_charge:
            Precursor charge state.  If ``None``, auto-detected alongside
            ``precursor_mz``.  Required for multi-charge-state removal and
            automatic isotope detection.
        tolerance:
            Tolerance for matching precursor peaks.
        tolerance_type:
            ``"Da"`` or ``"ppm"``.
        isotopes:
            Number of isotope peaks to remove.  ``"auto"`` (default) uses
            :func:`peptacular.estimate_isotopic_distribution` to determine
            the number of significant isotopes.  Pass an ``int`` to override
            (0 = monoisotopic only).
        isotope_threshold:
            Minimum relative abundance for an isotope to be considered
            significant when ``isotopes="auto"``.  Default 0.01 (1 %).
        remove_charge_states:
            If ``True`` (default) and the precursor charge is known, remove
            peaks at **all** charge states from 1 to ``precursor_charge``.
        inplace:
            Whether to modify the spectrum in place.

        Returns
        -------
        Self
            Spectrum with precursor (and isotope / charge-state) peaks removed.

        Raises
        ------
        ValueError
            If the spectrum is profile mode, or if ``precursor_mz`` is ``None``
            and no precursor information is available.
        """
        PROTON: float = pt.PROTON_MASS
        NEUTRON: float = pt.C13_NEUTRON_MASS

        # -- guard: profile spectra -------------------------------------------
        if self.spectrum_type == SpectrumType.PROFILE:
            raise ValueError("remove_precursor_peak() requires centroid or deconvoluted data; call .centroid() first")

        # -- resolve precursor list -------------------------------------------
        precursors: list[tuple[float, int | None]]  # (mz, charge)

        if precursor_mz is not None:
            precursors = [(precursor_mz, precursor_charge)]
        elif isinstance(self, MsnSpectrum) and self.precursors:
            precursors = [(p.mz, p.charge) for p in self.precursors]
        else:
            raise ValueError("precursor_mz is required when the spectrum has no precursor information")

        # -- detect spectrum state --------------------------------------------
        is_decharged = self.is_decharged

        # -- collect all m/z targets to remove --------------------------------
        targets: list[float] = []
        # For deconvoluted spectra we also need charge-specific masks
        charge_targets: list[int | None] = []

        for prec_mz, prec_z in precursors:
            if is_decharged:
                # m/z values are neutral masses; compute precursor neutral mass
                z = prec_z if prec_z is not None and prec_z > 0 else 1
                neutral = (prec_mz * z) - (z * PROTON)
                targets.append(neutral)
                charge_targets.append(None)

            elif self.spectrum_type == SpectrumType.DECONVOLUTED:
                # Monoisotopic peaks only; match at precursor charge
                targets.append(prec_mz)
                charge_targets.append(prec_z)

            else:
                # Centroid: remove all charge states and isotope envelopes
                z = prec_z if prec_z is not None and prec_z > 0 else None
                neutral = (prec_mz * (z or 1)) - ((z or 1) * PROTON)

                # Determine isotope offsets
                if isotopes == "auto":
                    if z is not None:
                        dist = pt.estimate_isotopic_distribution(
                            neutral,
                            min_abundance_threshold=isotope_threshold,
                            use_neutron_count=True,
                        )
                        offsets = [iso.neutron_count for iso in dist]
                    else:
                        # No charge → can't compute neutral mass reliably
                        offsets = [0]
                else:
                    offsets = list(range(isotopes + 1))

                # Determine charge states to iterate
                if remove_charge_states and z is not None:
                    charges = list(range(1, z + 1))
                else:
                    charges = [z or 1]

                for cz in charges:
                    mz_at_cz = (neutral + cz * PROTON) / cz
                    for offset in offsets:
                        targets.append(mz_at_cz + offset * NEUTRON / cz)
                        charge_targets.append(None)  # no charge filter for centroid

        # -- build removal mask -----------------------------------------------
        mask = np.ones(len(self.mz), dtype=bool)
        for target, target_charge in zip(targets, charge_targets, strict=True):
            if tolerance_type == "ppm":
                tol_da = target * tolerance / 1e6
            else:
                tol_da = tolerance

            mz_match = np.abs(self.mz - target) <= tol_da

            if target_charge is not None and self.charge is not None:
                mz_match &= self.charge == target_charge

            mask &= ~mz_match

        return self._apply_mask(mask, inplace=inplace)

    # -------------------------------------------------------------------------
    # Intensity Scaling
    # -------------------------------------------------------------------------

    def scale_intensity(
        self,
        method: Literal["root", "log", "rank"] = "root",
        degree: int = 2,
        base: float = 2.0,
        inplace: bool = False,
    ) -> Self:
        """Apply intensity scaling transformations.

        Unlike :meth:`normalize` (which divides by a reference value), scaling
        applies non-linear transforms that compress the dynamic range of
        intensities.

        Parameters
        ----------
        method:
            ``"root"`` — nth-root transform (default: square root).
            ``"log"``  — log-base transform (log(intensity + 1)).
            ``"rank"`` — replace intensities with their rank (1 = lowest).
        degree:
            Root degree for ``"root"`` method (default 2 = sqrt).
        base:
            Logarithm base for ``"log"`` method (default 2).
        inplace:
            Whether to modify the spectrum in place.

        Returns
        -------
        Self
            Spectrum with scaled intensities.
        """
        if method == "root":
            if degree == 0:
                raise ValueError("degree must be non-zero for the 'root' scaling method")
            scaled = np.power(self.intensity, 1.0 / degree)
        elif method == "log":
            scaled = np.log1p(self.intensity) / np.log(base)
        elif method == "rank":
            # argsort of argsort gives ranks (0-based); add 1 for 1-based
            order = np.argsort(np.argsort(self.intensity))
            scaled = (order + 1).astype(np.float64)
        else:
            raise ValueError(f"Unknown scaling method: {method!r}")

        # Scaling changes the intensity distribution, so any prior normalisation
        # no longer holds. Clearing the flag lets normalize() run again instead
        # of warning and silently returning unnormalised data.
        return self.update(intensity=scaled, normalized=None, inplace=inplace)

    # -------------------------------------------------------------------------
    # Peak Rounding
    # -------------------------------------------------------------------------

    def round_mz(
        self,
        decimals: int = 0,
        combine: Literal["sum", "max"] = "sum",
        inplace: bool = False,
    ) -> Self:
        """Round m/z values and combine peaks with identical m/z.

        Parameters
        ----------
        decimals:
            Number of decimal places to round m/z to.
        combine:
            How to combine intensities of merged peaks:
            ``"sum"`` adds them, ``"max"`` keeps the maximum.
        inplace:
            Whether to modify the spectrum in place.

        Returns
        -------
        Self
            Spectrum with rounded m/z values and combined peaks.
        """
        rounded_mz = np.round(self.mz, decimals)
        unique_mz, inverse = np.unique(rounded_mz, return_inverse=True)

        if combine == "sum":
            new_intensity = np.zeros(len(unique_mz), dtype=np.float64)
            np.add.at(new_intensity, inverse, self.intensity)
        elif combine == "max":
            # -inf rather than 0 so that all-negative intensities survive.
            new_intensity = np.full(len(unique_mz), -np.inf, dtype=np.float64)
            np.maximum.at(new_intensity, inverse, self.intensity)
        else:
            raise ValueError(f"Unknown combine method: {combine!r}")

        return self.update(
            mz=unique_mz,
            intensity=new_intensity,
            charge=None,
            im=None,
            iso_score=None,
            # Rounding merges peaks and drops the charge/score arrays, so the
            # result is no longer deconvoluted. Leaving the flag set wedges the
            # spectrum: decharge() refuses it and deconvolute() no-ops on it.
            spectrum_type=(
                SpectrumType.CENTROID if self.spectrum_type == SpectrumType.DECONVOLUTED else self.spectrum_type
            ),
            inplace=inplace,
        )

    # -------------------------------------------------------------------------
    # Mass Error Plot (convenience)
    # -------------------------------------------------------------------------

    def mass_error_plot(
        self,
        fragments: "FragmentInput",
        tolerance: float = DEFAULT_FRAGMENT_TOLERANCE,
        tolerance_type: ToleranceLike = DEFAULT_FRAGMENT_TOLERANCE_TYPE,
        peak_selection: PeakSelectionLike = PeakSelection.CLOSEST,
        unit: Literal["ppm", "da"] = "ppm",
        title: str | None = None,
        **layout_kwargs,
    ) -> "go.Figure":
        """Plot mass errors as a bubble chart (requires plotly).

        Parameters
        ----------
        fragments:
            Fragment objects from peptacular to match against peaks.
        tolerance:
            Matching tolerance.
        tolerance_type:
            ``"Da"`` or ``"ppm"``.
        peak_selection:
            ``"closest"``, ``"largest"``, or ``"all"``.
        unit:
            Error unit to display: ``"ppm"`` or ``"da"``.
        title:
            Plot title.
        **layout_kwargs:
            Forwarded to ``fig.update_layout``.
        """
        from .visualization import mass_error_plot

        return mass_error_plot(
            self,
            fragments,
            tolerance=tolerance,
            tolerance_type=tolerance_type,
            peak_selection=peak_selection,
            unit=unit,
            title=title,
            **layout_kwargs,
        )

    def facet_plot(
        self,
        fragments: "FragmentInput | None" = None,
        mirror_spectrum: "Spectrum | None" = None,
        title: str | None = None,
        tolerance: float = DEFAULT_FRAGMENT_TOLERANCE,
        tolerance_type: ToleranceLike = DEFAULT_FRAGMENT_TOLERANCE_TYPE,
        peak_selection: PeakSelectionLike = PeakSelection.CLOSEST,
        include_sequence: bool = False,
        **layout_kwargs,
    ) -> "go.Figure":
        """Multi-panel facet plot: spectrum + mass errors + optional mirror.

        Parameters
        ----------
        fragments:
            Fragment objects for annotation and mass error panels.
        mirror_spectrum:
            Optional second spectrum shown as a mirror below.
        title:
            Plot title.
        tolerance:
            Matching tolerance.
        tolerance_type:
            ``"Da"`` or ``"ppm"``.
        peak_selection:
            ``"closest"``, ``"largest"``, or ``"all"``.
        include_sequence:
            Embed the residue sequence in annotation labels.
        **layout_kwargs:
            Forwarded to ``fig.update_layout``.
        """
        from .visualization import facet_plot

        return facet_plot(
            self,
            fragments=fragments,
            mirror_spectrum=mirror_spectrum,
            title=title,
            tolerance=tolerance,
            tolerance_type=tolerance_type,
            peak_selection=peak_selection,
            include_sequence=include_sequence,
            **layout_kwargs,
        )

    def __len__(self) -> int:
        return len(self.mz)


@dataclass(frozen=True, slots=True, kw_only=True)
class Precursor(Peak):
    """Represents a target ion for MS2 fragmentation."""

    is_monoisotopic: bool | None


@dataclass(slots=True, kw_only=True, eq=False)
class MsnSpectrum(Spectrum):
    """
    Base class for all MSn spectra (MS1, MS2, MS3, etc.).
    Contains fields common to all MS levels.
    """

    # -------------------------------------------------------------------------
    # Scan Identification
    # -------------------------------------------------------------------------
    scan_number: int | None = None  # Native scan number from instrument
    ms_level: int | None = None  # 1 for MS1, 2 for MS2, etc.
    native_id: str | None = None  # e.g., "scan=1234" or instrument-specific format
    im_type: IMTypeLike | None = None  # e.g., "ook0", "drift_time_ms", etc.

    # -------------------------------------------------------------------------
    # Timing & Chromatography
    # -------------------------------------------------------------------------
    rt: float | None = None  # Retention time (seconds recommended, but document units)
    injection_time: float | None = None  # Ion injection/accumulation time (ms)
    total_ion_current: float | None = None  # Total ion current for the scan

    # -------------------------------------------------------------------------
    # m/z & Ion Mobility Windows (NOT ISOLATION WINDOWS, represent the full)
    # -------------------------------------------------------------------------
    mz_range: tuple[float, float] | None = None  # Scan window (min_mz, max_mz)
    im_range: tuple[float, float] | None = None  # Ion mobility window (for timsTOF)

    # -------------------------------------------------------------------------
    # Instrument Settings
    # -------------------------------------------------------------------------
    polarity: PolarityLike | None = None

    # -------------------------------------------------------------------------
    # Optional Metadata
    # -------------------------------------------------------------------------
    resolution: float | None = None  # Resolution
    analyzer: AnalyzerLike | None = None  # e.g., "orbitrap", "tof"; vendor shorthands ("FTMS") pass through
    ramp_time: float | None = None  # Ramp time for ion mobility (ms)
    collision_energy: float | None = None  # Collision energy for MS2 spectra
    activation_type: ActivationTypeLike | None = None  # e.g., "HCD", "CID", "ETD"
    precursors: list[Precursor] | None = None  # For MS2/MSn, list of precursor peaks

    isolation_mz_range: tuple[float, float] | None = None  # Isolation window (min_mz, max_mz) for MS2
    isolation_im_range: tuple[float, float] | None = None  # Isolation window for ion mobility (if applicable)

    def __str__(self) -> str:
        rt_str = f"{self.rt:.2f}s" if self.rt is not None else "None"
        return (
            f"MsnSpectrum(scan={self.scan_number}, ms_level={self.ms_level}, "
            f"rt={rt_str}, polarity={self.polarity}, n_peaks={len(self.mz)})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    _MSN_SCALAR_META_FIELDS: ClassVar[tuple[str, ...]] = (
        "scan_number",
        "ms_level",
        "native_id",
        "rt",
        "injection_time",
        "total_ion_current",
        "im_type",
        "polarity",
        "resolution",
        "analyzer",
        "ramp_time",
        "collision_energy",
        "activation_type",
    )
    _MSN_TUPLE_META_FIELDS: ClassVar[tuple[str, ...]] = (
        "mz_range",
        "im_range",
        "isolation_mz_range",
        "isolation_im_range",
    )

    def _meta_dict(self) -> dict:
        # Explicit base-class call (not super()) because @dataclass(slots=True)
        # on Spectrum and MsnSpectrum breaks zero-arg super() on Python <3.13:
        # https://github.com/python/cpython/issues/90562
        meta = Spectrum._meta_dict(self)
        for field in self._MSN_SCALAR_META_FIELDS:
            meta[field] = getattr(self, field)
        for field in self._MSN_TUPLE_META_FIELDS:
            val = getattr(self, field)
            meta[field] = list(val) if val is not None else None
        meta["precursors"] = (
            [
                {
                    "mz": p.mz,
                    "intensity": p.intensity,
                    "charge": p.charge,
                    "im": p.im,
                    "iso_score": p.iso_score,
                    "is_monoisotopic": p.is_monoisotopic,
                }
                for p in self.precursors
            ]
            if self.precursors is not None
            else None
        )
        return meta

    @classmethod
    def _meta_kwargs(cls, meta: dict) -> dict:
        # Explicit base-class call (not super()) — see _meta_dict note above.
        kwargs = Spectrum._meta_kwargs(meta)
        for field in cls._MSN_SCALAR_META_FIELDS:
            kwargs[field] = meta.get(field)
        for field in cls._MSN_TUPLE_META_FIELDS:
            val = meta.get(field)
            kwargs[field] = tuple(val) if val is not None else None
        kwargs["precursors"] = (
            [Precursor(**p) for p in meta["precursors"]] if meta.get("precursors") is not None else None
        )
        return kwargs
