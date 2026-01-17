"""
Spectacular: A peptacular companion for mass spectrometry data
Core data structures for spectra
"""

from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
import peptacular as pt
from numpy.typing import NDArray

from .compress import compress_spectra, decompress_spectra
from .noise import estimate_noise_level

# ============================================================================
# Core Data Structures
# ============================================================================


@dataclass(frozen=True, slots=True)
class Peak:
    """Single peak in a spectrum."""

    mz: float
    intensity: float
    charge: int | None = None
    ion_mobility: float | None = None

    def __repr__(self) -> str:
        parts = [f"mz={self.mz:.4f}", f"int={self.intensity:.2e}"]
        if self.charge is not None:
            parts.append(f"z={self.charge}")
        if self.ion_mobility is not None:
            parts.append(f"im={self.ion_mobility:.3f}")
        return f"Peak({', '.join(parts)})"


@dataclass(frozen=True, slots=True)
class Spectrum:
    """Mass spectrum with optional charge and ion mobility dimensions."""

    mz: NDArray[np.float64]  # Shape: (n,)
    intensity: NDArray[np.float64]  # Shape: (n,)
    charge: NDArray[np.int32] | None = None  # Shape: (n,)
    ion_mobility: NDArray[np.float64] | None = None  # Shape: (n,)

    # Metadata
    scan_number: int | None = None
    retention_time: float | None = None
    ms_level: int = 1
    precursor_mz: float | None = None
    precursor_charge: int | None = None

    def __post_init__(self):
        """Validate array shapes."""
        n = len(self.mz)
        if len(self.intensity) != n:
            raise ValueError("mz and intensity must have same length")
        if self.charge is not None and len(self.charge) != n:
            raise ValueError("charge array must match mz length")
        if self.ion_mobility is not None and len(self.ion_mobility) != n:
            raise ValueError("ion_mobility array must match mz length")

    # -------------------------------------------------------------------------
    # Peak Access
    # -------------------------------------------------------------------------

    @property
    def peaks(self) -> list[Peak]:
        """Convert to list of Peak objects."""
        return [
            Peak(
                mz=self.mz[i],
                intensity=self.intensity[i],
                charge=self.charge[i] if self.charge is not None else None,
                ion_mobility=self.ion_mobility[i] if self.ion_mobility is not None else None,
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
                mz=self.mz[i],
                intensity=self.intensity[i],
                charge=self.charge[i] if self.charge is not None else None,
                ion_mobility=self.ion_mobility[i] if self.ion_mobility is not None else None,
            )
            for i in indices
        ]

    # -------------------------------------------------------------------------
    # Cached Sort Indices
    # -------------------------------------------------------------------------

    @property
    def _argsort_mz(self) -> NDArray[np.int64]:
        """Cached argsort by m/z."""
        return np.argsort(self.mz)

    @property
    def _argsort_intensity(self) -> NDArray[np.int64]:
        """Cached argsort by intensity."""
        return np.argsort(self.intensity)

    @property
    def _argsort_charge(self) -> NDArray[np.int64]:
        """Cached argsort by charge (if available)."""
        if self.charge is None:
            raise ValueError("Spectrum has no charge information")
        return np.argsort(self.charge)

    @property
    def _argsort_im(self) -> NDArray[np.int64]:
        """Cached argsort by ion mobility (if available)."""
        if self.ion_mobility is None:
            raise ValueError("Spectrum has no ion mobility information")
        return np.argsort(self.ion_mobility)

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
        matches = self._find_matching_peaks(target_mz, mz_tol, mz_tol_type, target_charge, target_im, im_tol)
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
        matches = self._find_matching_peaks(target_mz, mz_tol, mz_tol_type, target_charge, target_im, im_tol)

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
            ion_mobility=self.ion_mobility[idx] if self.ion_mobility is not None else None,
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
        matches = self._find_matching_peaks(target_mz, mz_tol, mz_tol_type, target_charge, target_im, im_tol)

        return [
            Peak(
                mz=self.mz[i],
                intensity=self.intensity[i],
                charge=self.charge[i] if self.charge is not None else None,
                ion_mobility=self.ion_mobility[i] if self.ion_mobility is not None else None,
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
        if target_im is not None and self.ion_mobility is not None:
            mask &= np.abs(self.ion_mobility - target_im) <= im_tol

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
        if min_im is not None and self.ion_mobility is not None:
            mask &= self.ion_mobility >= min_im
        if max_im is not None and self.ion_mobility is not None:
            mask &= self.ion_mobility <= max_im

        # Apply top_n after other filters
        if top_n is not None:
            valid_indices = np.where(mask)[0]
            intensities = self.intensity[valid_indices]
            top_indices = valid_indices[np.argsort(intensities)[-top_n:]]
            mask = np.zeros(len(self.mz), dtype=bool)
            mask[top_indices] = True

        return self._apply_mask(mask)

    def normalize(self, method: Literal["max", "tic", "median"] = "max") -> Self:
        """Normalize intensities."""
        if method == "max":
            norm_factor = self.intensity.max()
        elif method == "tic":
            norm_factor = self.intensity.sum()
        else:  # median
            norm_factor = np.median(self.intensity)

        return self.update(intensity=self.intensity / norm_factor)

    def denoise(
        self,
        method: Literal["mad", "percentile", "histogram", "baseline", "iterative_median"] | float | int = "mad",
    ) -> Self:
        """Remove low-intensity noise peaks."""
        threshold = estimate_noise_level(self.intensity, method=method)
        return self.filter(min_intensity=threshold)

    def _apply_mask(self, mask: NDArray[np.bool_]) -> Self:
        return self.__class__(
            mz=self.mz[mask],
            intensity=self.intensity[mask],
            charge=self.charge[mask] if self.charge is not None else None,
            ion_mobility=self.ion_mobility[mask] if self.ion_mobility is not None else None,
            scan_number=self.scan_number,
            retention_time=self.retention_time,
            ms_level=self.ms_level,
            precursor_mz=self.precursor_mz,
            precursor_charge=self.precursor_charge,
        )

    def update(self, **kwargs) -> Self:
        """Create new spectrum with updated fields."""
        from dataclasses import replace

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
        max_left_decrease: float = 0.6,
        max_right_decrease: float = 0.9,
        isotope_mass: float = pt.C13_NEUTRON_MASS,
        isotope_lookup: pt.IsotopeLookup | None = None,
        inplace: bool = False,
    ) -> Self:
        """
        Deconvolute spectrum to find isotopic envelopes and determine charge states.

        If inplace is True, updates the current spectrum's charge array with identified charges.
        Currently returns the spectrum (updated or not).
        """
        from .decon.deconvolution import deconvolute

        # Call deconvolution logic
        dpeaks = deconvolute(
            mz=self.mz,
            intensity=self.intensity,
            tolerance=tolerance,
            tolerance_type=tolerance_type,
            charge_range=charge_range,
            max_left_decrease=max_left_decrease,
            max_right_decrease=max_right_decrease,
            isotope_mass=isotope_mass,
            isotope_lookup=isotope_lookup,
        )

        # dpeaks is list[DeconvolutedPeak].
        # Each DeconvolutedPeak corresponds to a group of peaks in the original spectrum.
        # We want to assign the charge to the peaks in the spectrum.
        # Note: A peak could theoretically belong to multiple envelopes in complex scenarios,
        # but the greedy algorithm assigns it to one.

        new_charges = np.zeros_like(self.mz, dtype=np.int32)

        # Map back to indices. DeconvolutedPeak has .peaks which are SpectrumPeak objects.
        # But we need indices to update the array.
        # Creating a map from SpectrumPeak object id to index is possible but tricky if objects are recreated.
        # The decon logic recreated SpectrumPeak objects from input arrays.
        # However, we passed mz/intensity arrays in parallel.
        # The deconvolution algorithm processed them in order and construct_graph indexed them.
        # But DeconvolutedPeak just stores the objects.

        # We can perform a fuzzy match or exact match on mz/intensity to find the index.
        # Or better: Update deconvolute to return indices!
        # But for now without modifying decon return type deeper:

        # Let's map (mz, intensity) -> index in original spectrum.
        # Assuming unique mz/intensity pairs or stable enough.
        # Actually, mz should be unique enough.

        # But wait, deconvolute uses 'sorted_peaks' indices internally but returns objects.
        # If we can assume exact float match:
        mz_to_idx = {mz: i for i, mz in enumerate(self.mz)}

        for dp in dpeaks:
            if dp.charge is not None:
                for p in dp.peaks:
                    if p.mz in mz_to_idx:
                        idx = mz_to_idx[p.mz]
                        # Set charge. If it was 0 (default in new_charges), set it.
                        # Ideally we overwrite.
                        new_charges[idx] = dp.charge

        if inplace:
            return self.update(charge=new_charges)
        else:
            return self.update(charge=new_charges)

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
