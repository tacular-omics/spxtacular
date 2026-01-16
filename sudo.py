"""
Spectacular: A peptacular companion for mass spectrometry data
Handles spectra I/O, processing, and visualization
"""

from dataclasses import dataclass
from typing import Literal, Protocol, Self
from pathlib import Path
from collections.abc import Iterator, Sequence
import numpy as np
from numpy.typing import NDArray

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
    
    mz: NDArray[np.float64]           # Shape: (n,)
    intensity: NDArray[np.float64]    # Shape: (n,)
    charge: NDArray[np.int32] | None = None       # Shape: (n,)
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
        matches = self._find_matching_peaks(
            target_mz, mz_tol, mz_tol_type, target_charge, target_im, im_tol
        )
        
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
    
    def denoise(self, threshold: float = 0.01) -> Self:
        """Remove low-intensity noise peaks."""
        return self.filter(min_intensity=threshold * self.intensity.max())
    
    def centroid(self, window: int = 5) -> Self:
        """Simple centroiding (peak picking)."""
        # Find local maxima
        peaks = []
        for i in range(window, len(self.mz) - window):
            if self.intensity[i] == max(self.intensity[i-window:i+window+1]):
                peaks.append(i)
        
        mask = np.zeros(len(self.mz), dtype=bool)
        mask[peaks] = True
        return self._apply_mask(mask)
    
    def _apply_mask(self, mask: NDArray[np.bool_]) -> Self:
        """Apply boolean mask to create filtered spectrum."""
        return Spectrum(
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
    
    def plot(
        self,
        title: str | None = None,
        show_charges: bool = True,
        **layout_kwargs,
    ):
        """Plot spectrum using plotly."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("plotly required for plotting: pip install plotly")
        
        fig = go.Figure()
        
        # Stem plot for peaks
        for i in range(len(self.mz)):
            fig.add_trace(go.Scatter(
                x=[self.mz[i], self.mz[i]],
                y=[0, self.intensity[i]],
                mode='lines',
                line=dict(color='blue', width=1),
                showlegend=False,
                hovertemplate=f"m/z: {self.mz[i]:.4f}<br>Int: {self.intensity[i]:.2e}<extra></extra>",
            ))
        
        fig.update_layout(
            title=title or f"Spectrum (Scan {self.scan_number})",
            xaxis_title="m/z",
            yaxis_title="Intensity",
            **layout_kwargs,
        )
        
        return fig

    def plot_df(        
        peptide: "Annotation",  # From peptacular
        ion_types: Sequence[str] = ("b", "y"),
        charges: Sequence[int] = (1, 2),
        losses: Sequence[str] | None = None,
        isotopes: int = 0,
        mz_tol: float = 0.02,
        mz_tol_type: Literal["Da", "ppm"] = "Da",) -> df:
        # this returns a df with all the data needed for plotting
    
    def annotate(
        self,
        peptide: "Annotation",  # From peptacular
        ion_types: Sequence[str] = ("b", "y"),
        charges: Sequence[int] = (1, 2),
        losses: Sequence[str] | None = None,
        isotopes: int = 0,
        mz_tol: float = 0.02,
        mz_tol_type: Literal["Da", "ppm"] = "Da",
        **plot_kwargs,
    ):
        """Annotate spectrum with theoretical fragments from peptacular."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("plotly required for annotation")
        
        fig = self.plot(**plot_kwargs)
        
        # Generate theoretical fragments
        annotations = []
        for ion_type in ion_types:
            for charge in charges:
                try:
                    frag = peptide.frag(ion_type=ion_type, charge=charge)
                    
                    # Check if experimental peak exists
                    peak = self.get_peak(frag.mz, mz_tol, mz_tol_type)
                    if peak:
                        annotations.append({
                            "x": peak.mz,
                            "y": peak.intensity,
                            "text": f"{ion_type}{charge}+",
                            "showarrow": True,
                        })
                except Exception:
                    continue
        
        fig.update_layout(annotations=annotations)
        return fig

    def annotate_from_df(df):
        # this annotates from a dataframe generated from plot_df
        pass


# ============================================================================
# Reader Protocol & Implementations
# ============================================================================

class SpectrumReader(Protocol):
    """Protocol for spectrum file readers."""
    
    def __len__(self) -> int: ...
    def __getitem__(self, index: int | slice) -> Spectrum | list[Spectrum]: ...
    def __iter__(self) -> Iterator[Spectrum]: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *args) -> None: ...
    
    @property
    def ms_levels(self) -> set[int]: ...
    
    def filter_ms_level(self, level: int) -> Iterator[Spectrum]: ...


class MzMLReader:
    """Reader for mzML files."""
    
    def __init__(self, filepath: str | Path):
        try:
            import pymzml
        except ImportError:
            raise ImportError("pymzml required: pip install pymzml")
        
        self.filepath = Path(filepath)
        self._run = pymzml.run.Reader(str(filepath))
    
    def __len__(self) -> int:
        return len(list(self._run))
    
    def __getitem__(self, index: int) -> Spectrum:
        for i, spec in enumerate(self._run):
            if i == index:
                return self._parse_spectrum(spec)
        raise IndexError(f"Spectrum index {index} out of range")
    
    def __iter__(self) -> Iterator[Spectrum]:
        for spec in self._run:
            yield self._parse_spectrum(spec)
    
    def __enter__(self) -> Self:
        return self
    
    def __exit__(self, *args) -> None:
        pass
    
    def _parse_spectrum(self, spec) -> Spectrum:
        """Parse pymzml spectrum to Spectrum object."""
        mz = np.array(spec.mz, dtype=np.float64)
        intensity = np.array(spec.i, dtype=np.float64)
        
        return Spectrum(
            mz=mz,
            intensity=intensity,
            scan_number=spec.ID,
            ms_level=spec.ms_level,
        )


class TDFReader:
    """Reader for Bruker TDF files."""
    
    def __init__(self, dirpath: str | Path):
        try:
            import tdfpy
        except ImportError:
            raise ImportError("tdfpy required for TDF files")
        
        self.dirpath = Path(dirpath)
        # Implementation would go here
        raise NotImplementedError("TDF reader coming soon!")


# ============================================================================
# Public API
# ============================================================================

def reader(
    filepath: str | Path,
    format: Literal["auto", "mzml", "tdf"] | None = None,
) -> SpectrumReader:
    """Create a spectrum reader for the given file."""
    filepath = Path(filepath)
    
    # Auto-detect format
    if format is None or format == "auto":
        if filepath.suffix.lower() == ".mzml":
            format = "mzml"
        elif filepath.is_dir() and (filepath / "analysis.tdf").exists():
            format = "tdf"
        else:
            raise ValueError(f"Could not auto-detect format for {filepath}")
    
    if format == "mzml":
        return MzMLReader(filepath)
    elif format == "tdf":
        return TDFReader(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    # Read mzML file
    with reader("data.mzml") as r:
        spec = r[0]  # First spectrum
        
        # Filter and process
        filtered = spec.filter(min_mz=200, max_mz=2000, min_intensity=1e4)
        normalized = filtered.normalize()
        denoised = normalized.denoise(threshold=0.01)
        
        # Find peaks
        peak = spec.get_peak(target_mz=524.265, mz_tol=0.01)
        peaks = spec.get_peaks(target_mz=524.265, mz_tol=50, mz_tol_type="ppm")
        
        # Top peaks
        top10 = spec.top_peaks(10, by="intensity")
        
        # Plot
        fig = spec.plot(title="My Spectrum")
        fig.show()
    
    # Read TDF file
    with reader("data.d") as r:
        spec = r[0]
        
        # With peptacular integration
        from peptacular import Annotation
        
        peptide = Annotation.from_string("PEPTIDE")
        fig = spec.annotate(
            peptide,
            ion_types=["b", "y"],
            charges=[1, 2, 3],
            mz_tol=20,
            mz_tol_type="ppm",
        )
        fig.show()