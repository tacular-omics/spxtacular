"""
Spectacular: A peptacular companion for mass spectrometry data
Core data structures for spectra
"""

import warnings
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

if TYPE_CHECKING:
    import pandas as pd
    import plotly.graph_objects as go

    from .matching import FragmentInput, MatchedFragment

import numpy as np
from numpy.typing import NDArray

from .compress import compress_spectra, decompress_spectra
from .decon.scored import deconvolve_spectrum as _deconvolve
from .enums import PeakSelection, PeakSelectionLike, ToleranceLike, ToleranceType
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


@dataclass(slots=True)
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
        """Validate array shapes."""
        n = len(self.mz)
        if len(self.intensity) != n:
            raise ValueError("mz and intensity must have same length")
        if self.charge is not None and len(self.charge) != n:
            raise ValueError("charge array must match mz length")
        if self.im is not None and len(self.im) != n:
            raise ValueError("im array must match mz length")
        if self.iso_score is not None and len(self.iso_score) != n:
            raise ValueError("score array must match mz length")
        if self.charge is not None and self.spectrum_type != SpectrumType.DECONVOLUTED:
            object.__setattr__(self, "spectrum_type", SpectrumType.DECONVOLUTED)

    # -------------------------------------------------------------------------
    # Peak Access
    # -------------------------------------------------------------------------

    @property
    def peaks(self) -> list[Peak]:
        """Convert to list of Peak objects."""
        return [
            Peak(
                mz=float(self.mz[i]),
                intensity=float(self.intensity[i]),
                charge=self.charge[i] if self.charge is not None else None,
                im=self.im[i] if self.im is not None else None,
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

        indices = sort_key[:n] if not reverse else sort_key[-n:][::-1]

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
    # Cached Sort Indices
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
            mz=self.mz[idx],
            intensity=self.intensity[idx],
            charge=self.charge[idx] if self.charge is not None else None,
            im=self.im[idx] if self.im is not None else None,
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
                mz=self.mz[i],
                intensity=self.intensity[i],
                charge=self.charge[i] if self.charge is not None else None,
                im=self.im[i] if self.im is not None else None,
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
        if tolerance_type == "ppm":
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
        """Filter spectrum by various criteria."""
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
            top_indices = valid_indices[np.argsort(intensities)[-top_n:]]
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
            return self

        if method == "max":
            norm_factor = self.intensity.max()
        elif method == "tic":
            norm_factor = self.intensity.sum()
        else:  # median
            norm_factor = np.median(self.intensity)

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
            return self

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
        """
        Merge nearby peaks within a given tolerance.

        Peaks are processed in order of decreasing intensity. For each peak,
        neighbors within the tolerance window are identified. The merged peak
        will have the weighted average m/z (and ion mobility if present) and
        sum of intensities.

        Parameters
        ----------
        tolerance : float, optional
            m/z tolerance for merging, by default 0.01
        tolerance_type : Literal["ppm", "da"], optional
            Type of tolerance, by default "da"
        inplace : bool, optional
            Whether to modify the spectrum in place, by default False

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

        # Sort by intensity descending for greedy clustering order
        # We need the original indices relative to the SORTED arrays
        intensity_order = np.argsort(intensity)[::-1]

        used_mask = np.zeros(len(mz), dtype=bool)

        new_mz_list = []
        new_intensity_list = []
        new_im_list = []
        new_charge_list = []

        if mz_tolerance_type not in ("ppm", "da"):
            raise ValueError("mz_tolerance_type must be 'ppm' or 'da'")

        if im_tolerance_type.lower() not in ("relative", "absolute"):
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

                if im_tolerance_type == "relative":
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

        # Sort result by m/z
        final_sort = np.argsort(new_mz)
        new_mz = new_mz[final_sort]
        new_intensity = new_intensity[final_sort]
        if new_im is not None:
            new_im = new_im[final_sort]
        if new_charge is not None:
            new_charge = new_charge[final_sort]

        if inplace:
            self.mz = new_mz
            self.intensity = new_intensity
            self.im = new_im
            self.charge = new_charge
            self.iso_score = None
            return self

        return replace(
            self,
            mz=new_mz,
            intensity=new_intensity,
            im=new_im,
            charge=new_charge,
            iso_score=None,
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
            return self

        mz_cent, int_cent, im_cent = _centroid_peaks(self.mz, self.intensity, self.im)

        return self.update(
            mz=mz_cent,
            intensity=int_cent,
            spectrum_type=SpectrumType.CENTROID,
            charge=None,
            im=im_cent,
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
        """Create new spectrum with updated fields."""
        if inplace:
            for k, v in kwargs.items():
                setattr(self, k, v)
            return self

        return replace(self, **kwargs)

    # -------------------------------------------------------------------------
    # Plotting (requires plotly)
    # -------------------------------------------------------------------------

    def plot(
        self,
        title: str | None = None,
        show_charges: bool = True,
        show_scores: bool = True,
        **layout_kwargs,
    ) -> "go.Figure":
        """Plot spectrum as a stick plot (requires plotly).

        Parameters
        ----------
        title:
            Plot title. Defaults to the spectrum type.
        show_charges:
            Colour sticks by charge state when charge data is present.
        show_scores:
            Annotate peaks with isotope profile scores when score data is present.
        **layout_kwargs:
            Forwarded to ``fig.update_layout``.
        """
        from .visualization import plot_spectrum

        return plot_spectrum(
            self,
            title=title,
            show_charges=show_charges,
            show_scores=show_scores,
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
        tolerance: float = 0.02,
        tolerance_type: ToleranceLike = ToleranceType.DA,
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
        tolerance: float = 0.02,
        tolerance_type: ToleranceLike = ToleranceType.DA,
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
        if self.spectrum_type == SpectrumType.DECONVOLUTED:
            warnings.warn(
                "Spectrum is already deconvoluted, returning original spectrum",
                UserWarning,
                stacklevel=2,
            )
            return self

        is_ppm = tolerance_type == "ppm"
        resolved_min_intensity: float = float(self.intensity.min()) if min_intensity == "min" else min_intensity

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

        return self.update(
            mz=new_mz,
            intensity=new_intensity,
            charge=new_charge,
            im=None,
            iso_score=new_score,
            spectrum_type=SpectrumType.DECONVOLUTED,
            inplace=inplace,
        )

    def decharge(self, inplace: bool = False) -> Self:
        """
        Decharge spectrum by converting m/z to neutral mass using charge information.

        Peaks with charge == -1 are dropped (charge unknown).
        If the spectrum is not yet deconvoluted, deconvolute() is called first with default parameters.

        Returns a new Spectrum with m/z values as neutral masses, sorted ascending.
        """
        if self.spectrum_type != SpectrumType.DECONVOLUTED or self.charge is None:
            return self.deconvolute(inplace=inplace).decharge(inplace=inplace)

        proton = 1.007276

        known = self.charge != -1
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
    # Compression
    # -------------------------------------------------------------------------

    def compress(
        self,
        url_safe: bool = False,
        mz_precision: int | None = None,
        intensity_precision: int | None = None,
        im_precision: int | None = None,
        compression: str = "gzip",
    ) -> str:
        """
        Compress spectrum data to a string.
        """
        return compress_spectra(
            self,
            url_safe=url_safe,
            mz_precision=mz_precision,
            intensity_precision=intensity_precision,
            im_precision=im_precision,
            compression=compression,
        )

    @classmethod
    def from_compressed(cls, compressed_str: str) -> "Spectrum":
        """
        Create a Spectrum from a compressed string.
        """
        return decompress_spectra(compressed_str)

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save spectrum to a ``.npz`` file.

        Arrays (``mz``, ``intensity``, and any optional ``charge``, ``im``,
        ``score``) are stored natively; all scalar metadata is stored as a
        JSON string under the ``meta`` key.  The file extension ``.npz`` is
        appended automatically if absent.
        """
        import json

        st = self.spectrum_type
        meta = {
            "spectrum_type": st if isinstance(st, str) else (st.value if st is not None else None),
            "denoised": self.denoised,
            "normalized": self.normalized,
        }
        arrays: dict = {
            "mz": self.mz,
            "intensity": self.intensity,
            "meta": np.array(json.dumps(meta), dtype=object),
        }
        if self.charge is not None:
            arrays["charge"] = self.charge
        if self.im is not None:
            arrays["im"] = self.im
        if self.iso_score is not None:
            arrays["score"] = self.iso_score
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Spectrum":
        """Load a :class:`Spectrum` from a ``.npz`` file written by :meth:`save`."""
        import json

        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        return cls(
            mz=data["mz"],
            intensity=data["intensity"],
            charge=data["charge"] if "charge" in data else None,
            im=data["im"] if "im" in data else None,
            iso_score=data["score"] if "score" in data else None,
            spectrum_type=meta["spectrum_type"],
            denoised=meta["denoised"],
            normalized=meta["normalized"],
        )

    # -------------------------------------------------------------------------
    # Fragment matching
    # -------------------------------------------------------------------------

    def match_fragments(
        self,
        fragments: "FragmentInput",
        tolerance: float = 0.02,
        tolerance_type: ToleranceLike = ToleranceType.DA,
        peak_selection: PeakSelectionLike = PeakSelection.CLOSEST,
    ) -> "list[MatchedFragment]":
        """Match fragment ions against this spectrum's peaks.

        Thin wrapper around :func:`~spxtacular.matching.match_fragments`.
        Returns a list of :class:`~spxtacular.matching.MatchedFragment` objects
        sorted by ascending ``peak_index``.
        """
        from .matching import match_fragments as _match

        return _match(
            self, fragments, tolerance=tolerance, tolerance_type=tolerance_type, peak_selection=peak_selection
        )

    def score(
        self,
        fragments: "FragmentInput",
        tolerance: float = 0.02,
        tolerance_type: ToleranceLike = ToleranceType.PPM,
        peak_selection: PeakSelectionLike = PeakSelection.CLOSEST,
    ) -> "dict[str, float]":
        """Match fragments and return all PSM scores.

        Thin wrapper around :func:`~spxtacular.scoring.score`.
        """
        from .scoring import score as _score

        return _score(
            self, fragments, tolerance=tolerance, tolerance_type=tolerance_type, peak_selection=peak_selection
        )

    def __len__(self) -> int:
        return len(self.mz)


@dataclass(frozen=True, slots=True, kw_only=True)
class Precursor(Peak):
    """Represents a target ion for MS2 fragmentation."""

    is_monoisotopic: bool | None


@dataclass(slots=True, kw_only=True)
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
    im_type: str | None = None  # e.g., "1/K0", "drift_time_ms", etc.

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
    polarity: Literal["positive", "negative"] | None = None

    # -------------------------------------------------------------------------
    # Optional Metadata
    # -------------------------------------------------------------------------
    resolution: float | None = None  # Resolution
    analyzer: str | None = None  # e.g., "FTMS", "ITMS", "TOFMS"
    ramp_time: float | None = None  # Ramp time for ion mobility (ms)
    collision_energy: float | None = None  # Collision energy for MS2 spectra
    activation_type: str | None = None  # e.g., "HCD", "CID", "ETD"
    precursors: list[Precursor] | None = None  # For MS2/MSn, list of precursor peaks

    isolation_mz_range: tuple[float, float] | None = None  # Isolation window (min_mz, max_mz) for MS2
    isolation_im_range: tuple[float, float] | None = None  # Isolation window for ion mobility (if applicable)

    def __str__(self) -> str:
        return (
            f"MsnSpectrum(scan={self.scan_number}, ms_level={self.ms_level}, "
            f"rt={self.rt:.2f}s, polarity={self.polarity}, n_peaks={len(self.mz)})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def save(self, path: str | Path) -> None:
        """Save spectrum to a ``.npz`` file (includes all MSn metadata)."""
        import json

        st = self.spectrum_type
        meta = {
            "spectrum_type": st if isinstance(st, str) else (st.value if st is not None else None),
            "denoised": self.denoised,
            "normalized": self.normalized,
            "scan_number": self.scan_number,
            "ms_level": self.ms_level,
            "native_id": self.native_id,
            "rt": self.rt,
            "injection_time": self.injection_time,
            "total_ion_current": self.total_ion_current,
            "mz_range": list(self.mz_range) if self.mz_range is not None else None,
            "im_range": list(self.im_range) if self.im_range is not None else None,
            "im_type": self.im_type,
            "polarity": self.polarity,
            "resolution": self.resolution,
            "analyzer": self.analyzer,
            "ramp_time": self.ramp_time,
            "collision_energy": self.collision_energy,
            "activation_type": self.activation_type,
            "isolation_mz_range": list(self.isolation_mz_range) if self.isolation_mz_range is not None else None,
            "isolation_im_range": list(self.isolation_im_range) if self.isolation_im_range is not None else None,
            "precursors": [
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
            else None,
        }
        arrays: dict = {
            "mz": self.mz,
            "intensity": self.intensity,
            "meta": np.array(json.dumps(meta), dtype=object),
        }
        if self.charge is not None:
            arrays["charge"] = self.charge
        if self.im is not None:
            arrays["im"] = self.im
        if self.iso_score is not None:
            arrays["iso_score"] = self.iso_score
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "MsnSpectrum":
        """Load an :class:`MsnSpectrum` from a ``.npz`` file written by :meth:`save`."""
        import json

        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        precursors = None
        if meta.get("precursors") is not None:
            precursors = [Precursor(**p) for p in meta["precursors"]]
        mz_range = tuple(meta["mz_range"]) if meta.get("mz_range") is not None else None
        im_range = tuple(meta["im_range"]) if meta.get("im_range") is not None else None
        return cls(
            mz=data["mz"],
            intensity=data["intensity"],
            charge=data["charge"] if "charge" in data else None,
            im=data["im"] if "im" in data else None,
            iso_score=data["iso_score"] if "iso_score" in data else None,
            spectrum_type=meta.get("spectrum_type"),
            denoised=meta.get("denoised"),
            normalized=meta.get("normalized"),
            scan_number=meta.get("scan_number"),
            ms_level=meta.get("ms_level"),
            native_id=meta.get("native_id"),
            rt=meta.get("rt"),
            injection_time=meta.get("injection_time"),
            total_ion_current=meta.get("total_ion_current"),
            mz_range=mz_range,
            im_range=im_range,
            im_type=meta.get("im_type"),
            polarity=meta.get("polarity"),
            resolution=meta.get("resolution"),
            analyzer=meta.get("analyzer"),
            ramp_time=meta.get("ramp_time"),
            collision_energy=meta.get("collision_energy"),
            activation_type=meta.get("activation_type"),
            precursors=precursors,
            isolation_mz_range=tuple(meta["isolation_mz_range"])
            if meta.get("isolation_mz_range") is not None
            else None,
            isolation_im_range=tuple(meta["isolation_im_range"])
            if meta.get("isolation_im_range") is not None
            else None,
        )
