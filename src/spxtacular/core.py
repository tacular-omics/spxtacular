"""
Spectacular: A peptacular companion for mass spectrometry data
Core data structures for spectra
"""

import warnings
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray

from .compress import compress_spectra, decompress_spectra
from .decon.greedy import deconvolve_spectrum
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

    def __repr__(self) -> str:
        parts = [f"mz={self.mz:.4f}", f"int={self.intensity:.2e}"]
        if self.charge is not None:
            parts.append(f"z={self.charge}")
        if self.im is not None:
            parts.append(f"im={self.im:.3f}")
        return f"Peak({', '.join(parts)})"


class SpectrumType(StrEnum):
    CENTROID: str = "centroid"
    PROFILE: str = "profile"
    DECONVOLUTED: str = "deconvoluted"


@dataclass(slots=True)
class Spectrum:
    """Mass spectrum with optional charge and ion mobility dimensions."""

    mz: NDArray[np.float64]  # Shape: (n,)
    intensity: NDArray[np.float64]  # Shape: (n,)
    charge: NDArray[np.int32] | None = None  # Shape: (n,)
    im: NDArray[np.float64] | None = None  # Shape: (n,)
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
            )
            for i in range(len(self.mz))
        ]

    def top_peaks(
        self,
        n: int,
        by: Literal["intensity", "mz", "charge", "im"] = "intensity",
        reverse: bool = True,
    ) -> list[Peak]:
        """Get top N peaks sorted by specified attribute."""
        sort_key = {
            "intensity": self._argsort_intensity,
            "mz": self._argsort_mz,
            "charge": self._argsort_charge,
            "im": self._argsort_im,
        }[by]

        indices = sort_key[:n] if not reverse else sort_key[-n:][::-1]

        return [
            Peak(
                mz=float(self.mz[i]),
                intensity=float(self.intensity[i]),
                charge=int(self.charge[i]) if self.charge is not None else None,
                im=float(self.im[i]) if self.im is not None else None,
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

    # -------------------------------------------------------------------------
    # Peak Finding
    # -------------------------------------------------------------------------

    def has_peak(
        self,
        target_mz: float,
        mz_tol: float = 0.01,
        mz_tol_type: Literal["Da", "ppm"] = "Da",
        target_charge: int | None = None,
        target_im: float | None = None,
        im_tol: float = 0.01,
    ) -> bool:
        """Check if spectrum contains a peak matching criteria."""
        matches = self._find_matching_peaks(
            target_mz, mz_tol, mz_tol_type, target_charge, target_im, im_tol
        )
        return len(matches) > 0

    def get_peak(
        self,
        target_mz: float,
        mz_tol: float = 0.01,
        mz_tol_type: Literal["Da", "ppm"] = "Da",
        target_charge: int | None = None,
        target_im: float | None = None,
        im_tol: float = 0.01,
        collision: Literal["largest", "closest"] = "largest",
    ) -> Peak | None:
        """Get single peak matching criteria."""
        matches = self._find_matching_peaks(
            target_mz, mz_tol, mz_tol_type, target_charge, target_im, im_tol
        )

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
        mz_tol: float = 0.01,
        mz_tol_type: Literal["Da", "ppm"] = "Da",
        target_charge: int | None = None,
        target_im: float | None = None,
        im_tol: float = 0.01,
    ) -> list[Peak]:
        """Get all peaks matching criteria."""
        matches = self._find_matching_peaks(
            target_mz, mz_tol, mz_tol_type, target_charge, target_im, im_tol
        )

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
        mz_tol: float,
        mz_tol_type: Literal["Da", "ppm"],
        target_charge: int | None,
        target_im: float | None,
        im_tol: float,
    ) -> NDArray[np.int64]:
        """Find indices of peaks matching criteria."""
        # m/z tolerance
        if mz_tol_type == "ppm":
            tol_da = target_mz * mz_tol / 1e6
        else:
            tol_da = mz_tol

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

        # Apply top_n after other filters
        if top_n is not None:
            valid_indices = np.where(mask)[0]
            intensities = self.intensity[valid_indices]
            top_indices = valid_indices[np.argsort(intensities)[-top_n:]]
            mask = np.zeros(len(self.mz), dtype=bool)
            mask[top_indices] = True

        return self._apply_mask(mask, inplace=inplace)

    def normalize(
        self, method: Literal["max", "tic", "median"] = "max", inplace: bool = False
    ) -> Self:
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

        return self.update(
            intensity=self.intensity / norm_factor, normalized=method, inplace=inplace
        )

    def denoise(
        self,
        method: Literal[
            "mad", "percentile", "histogram", "baseline", "iterative_median"
        ]
        | float
        | int = "mad",
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
        return self.filter(min_intensity=threshold, inplace=inplace).update(
            denoised=str(method), inplace=inplace
        )

    def merge(
        self,
        mz_tolerance: float = 0.01,
        mz_tolerance_type: Literal["ppm", "da"] = "da",
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

        if mz_tolerance_type.lower() not in ("ppm", "da"):
            raise ValueError("mz_tolerance_type must be 'ppm' or 'da'")

        if im_tolerance_type.lower() not in ("relative", "absolute"):
            raise ValueError("im_tolerance_type must be 'relative' or 'absolute'")

        is_ppm = mz_tolerance_type.lower() == "ppm"
        if not is_ppm:
            # Constant tolerance
            mz_tol_abs = mz_tolerance

        for idx in intensity_order:
            if used_mask[idx]:
                continue

            current_mz = mz[idx]
            current_int = intensity[idx]
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
        new_charge = (
            np.array(new_charge_list, dtype=np.int32) if charge is not None else None
        )

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
            return self

        return replace(
            self,
            mz=new_mz,
            intensity=new_intensity,
            im=new_im,
            charge=new_charge,
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
            return self

        return replace(
            self,
            mz=self.mz[mask],
            intensity=self.intensity[mask],
            charge=self.charge[mask] if self.charge is not None else None,
            im=self.im[mask] if self.im is not None else None,
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

    def plot(self, **kwargs):
        """Plot spectrum."""
        # TODO: Implement plotting with plotly
        raise NotImplementedError("Plotting not yet implemented")

    def deconvolute(
        self,
        tolerance: float = 50,
        tolerance_type: Literal["ppm", "da"] = "ppm",
        charge_range: tuple[int, int] = (1, 3),
        intensity: Literal["base", "total"] = "total",
        max_dpeaks: int = 2000,
        inplace: bool = False,
    ) -> Self:
        if self.spectrum_type == SpectrumType.DECONVOLUTED:
            warnings.warn(
                "Spectrum is already deconvoluted, returning original spectrum",
                UserWarning,
                stacklevel=2,
            )
            return self

        is_ppm = tolerance_type == "ppm"
        new_mz, new_charge, new_intensity = deconvolve_spectrum(
            mz=self.mz,
            intensity=self.intensity,
            charge_range=charge_range,
            tolerance=tolerance,
            is_ppm=is_ppm,
            max_dpeaks=max_dpeaks,
            intensity_mode=intensity,
        )

        return self.update(
            mz=new_mz,
            intensity=new_intensity,
            charge=new_charge,
            im=None,
            spectrum_type=SpectrumType.DECONVOLUTED,
            inplace=inplace,
        )

    def decharge(self, inplace: bool = False) -> Self:
        """
        Decharge spectrum by converting m/z to neutral mass using charge information.

        Peaks with charge == 0 are dropped (charge unknown).
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

        neutral_mz = (known_mz * known_charge) - (known_charge * proton)

        order = np.argsort(neutral_mz)

        return self.update(
            mz=neutral_mz[order],
            intensity=known_int[order],
            charge=np.zeros_like(
                known_charge[order], dtype=np.int32
            ),  # set charge to 0 (unknown)
            im=known_im[order] if known_im is not None else None,
            inplace=inplace,
        )

    def __str__(self) -> str:
        return f"Spectrum(n_peaks={len(self.mz)}, type={self.spectrum_type}, denoised={self.denoised}, normalized={self.normalized})"

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

    def __len__(self) -> int:
        return len(self.mz)


@dataclass(frozen=True, slots=True, kw_only=True)
class TargetIon(Peak):
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

    # -------------------------------------------------------------------------
    # Timing & Chromatography
    # -------------------------------------------------------------------------
    rt: float | None = None  # Retention time (seconds recommended, but document units)
    injection_time: float | None = None  # Ion injection/accumulation time (ms)

    # -------------------------------------------------------------------------
    # m/z & Ion Mobility Windows
    # -------------------------------------------------------------------------
    mz_range: tuple[float, float] | None = None  # Scan window (min_mz, max_mz)
    im_range: tuple[float, float] | None = None  # Ion mobility window (for timsTOF)
    im_type: str | None = None  # e.g., "1/K0", "drift_time_ms", etc.

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
    precursors: list[TargetIon] | None = None  # For MS2/MSn, list of precursor peaks

    def __str__(self) -> str:
        return f"MsnSpectrum(scan={self.scan_number}, ms_level={self.ms_level}, rt={self.rt:.2f}s, polarity={self.polarity}, n_peaks={len(self.mz)})"

    def __repr__(self) -> str:
        return self.__str__()
