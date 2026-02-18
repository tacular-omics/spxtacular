"""
Spectacular: A peptacular companion for mass spectrometry data
Core data structures for spectra
"""
from pandas.tests.arrays.boolean.test_arithmetic import data

from dataclasses import dataclass
from enum import StrEnum
from tkinter import N
from typing import Literal, Self

import numpy as np
import peptacular as pt
from numpy.linalg import norm
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
        if self.charge is not None and self.spectrum_type is None:
            object.__setattr__(self, "spectrum_type", SpectrumType.DECONVOLUTED)
        # if charges are present but spectrum_type is not deconvoluted raise error
        if self.charge is not None and self.spectrum_type != SpectrumType.DECONVOLUTED:
            raise ValueError("Spectrum with charge information must have spectrum_type=DECONVOLUTED")

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
                mz=self.mz[i],
                intensity=self.intensity[i],
                charge=self.charge[i] if self.charge is not None else None,
                im=self.im[i] if self.im is not None else None,
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
        matches = self._find_matching_peaks(target_mz, mz_tol, mz_tol_type, target_charge, target_im, im_tol)

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

    def normalize(self, method: Literal["max", "tic", "median"] = "max", inplace: bool = False) -> Self:
        """Normalize intensities."""

        # if already normalized, raise error
        if self.normalized is not None:
            raise ValueError(f"Spectrum is already normalized with method '{self.normalized}'")

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
            raise ValueError(f"Spectrum is already denoised with method '{self.denoised}'")

        threshold = estimate_noise_level(self.intensity, method=method)
        return self.filter(min_intensity=threshold, inplace=inplace).update(denoised=str(method), inplace=inplace)

    def _apply_mask(self, mask: NDArray[np.bool_], inplace: bool = False) -> Self:
        if inplace:
            self.mz = self.mz[mask]
            self.intensity = self.intensity[mask]
            if self.charge is not None:
                self.charge = self.charge[mask]
            if self.im is not None:
                self.im = self.im[mask]
            return self

        from dataclasses import replace

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

        Returns a new Spectrum object with identified charges and spectrum_type=DECONVOLUTED.
        """
        from .decon.deconvolution import deconvolute

        # if already deconvoluted, raise error
        if self.spectrum_type == SpectrumType.DECONVOLUTED:
            raise ValueError("Spectrum is already deconvoluted")

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
        new_charges = np.zeros_like(self.mz, dtype=np.int32)
        mz_to_idx = {mz: i for i, mz in enumerate(self.mz)}

        for dp in dpeaks:
            if dp.charge is not None:
                for p in dp.peaks:
                    if p.mz in mz_to_idx:
                        idx = mz_to_idx[p.mz]
                        # Set charge. If it was 0 (default in new_charges), set it.
                        # Ideally we overwrite.
                        new_charges[idx] = dp.charge

        return self.update(charge=new_charges, spectrum_type=SpectrumType.DECONVOLUTED, inplace=inplace)

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


