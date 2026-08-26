from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, NamedTuple, Self

import numpy as np

from .core import MsnSpectrum, Precursor, SpectrumType
from .enums import ActivationType, Analyzer, IMType, Polarity
from .peaklist import MgfReader, Ms2Reader, MspReader, PeakListLookup
from .thermo import ThermoReader, ThermoScanLookup

# The optional backends load native libraries, so a broken install can raise
# OSError rather than ImportError. Either way the backend is simply unavailable —
# it must not take down ``import spxtacular`` for users who never touch it.
try:
    import mzmlpy as mzp

    _HAS_MZMLPY = True
except (ImportError, OSError):
    mzp = None  # type: ignore[assignment] # ty: ignore[invalid-assignment]
    _HAS_MZMLPY = False

try:
    import tdfpy

    _HAS_TDFPY = True
except (ImportError, OSError):
    tdfpy = None  # type: ignore[assignment] # ty: ignore[invalid-assignment]
    _HAS_TDFPY = False

# tdfpy's smoothing branch (post-1.2.0) reshaped Frame.centroid() — the old
# kwargs (mz_tolerance, …, noise_filter) became `centroid=MergePeaksCentroider(…)`
# and `noise=…`. Detect which API is available and adapt below.
try:
    from tdfpy import MergePeaksCentroider as _MergePeaksCentroider

    _HAS_NEW_CENTROID_API = True
except (ImportError, OSError):
    _MergePeaksCentroider = None  # type: ignore[assignment] # ty: ignore[invalid-assignment]
    _HAS_NEW_CENTROID_API = False

"""

Unified reader API for different mass-spectrometry file formats.
Supports DDA, DIA, and PRM data from Bruker timsTOF (.d) and mzML, Thermo
.raw files (see thermo.py), plus the MGF and MS2 peak-list formats (see
peaklist.py).
"""


class AcquisitionType(StrEnum):
    DDA = "DDA"
    DIA = "DIA"
    PRM = "PRM"
    UNKNOWN = "UNKNOWN"


# Bruker ``Frames.MsMsType`` values that identify the acquisition scheme.
_MSMS_TYPE_ACQUISITION: dict[int, AcquisitionType] = {
    8: AcquisitionType.DDA,
    9: AcquisitionType.DIA,
    10: AcquisitionType.PRM,
}

# MsMsType values that name a real acquisition scheme tdfpy has no reader for.
# Falling through to AcquisitionType.UNKNOWN would open these with the DDA
# backend, which has no precursor table to walk: it either raises from inside
# tdfpy or yields nothing at all.
_MSMS_TYPE_UNSUPPORTED: dict[int, str] = {
    2: "classic (non-PASEF) MS/MS",
}


def _detect_acquisition_type(analysis_dir: str | Path) -> AcquisitionType:
    """Determine a Bruker ``.d`` folder's acquisition scheme from ``analysis.tdf``.

    Equivalent to ``tdfpy.get_acquisition_type`` but owns its sqlite connection
    so it is closed deterministically — ``sqlite3``'s context manager only ends
    the transaction, it does not close the handle, so the tdfpy helper leaks one
    connection per :class:`DReader`.

    Raises
    ------
    FileNotFoundError
        If the folder holds no ``analysis.tdf``.
    ValueError
        If the run's only MS/MS frames are of a scheme spxtacular cannot read
        (see ``_MSMS_TYPE_UNSUPPORTED``).
    """
    import sqlite3
    from contextlib import closing

    tdf_path = Path(analysis_dir) / "analysis.tdf"
    if not tdf_path.exists():
        raise FileNotFoundError(f"analysis.tdf not found at {tdf_path}")

    with closing(sqlite3.connect(str(tdf_path))) as conn, closing(conn.cursor()) as cur:
        msms_types = {row[0] for row in cur.execute("SELECT DISTINCT MsMsType FROM Frames")}

    for msms_type, acquisition_type in _MSMS_TYPE_ACQUISITION.items():
        if msms_type in msms_types:
            return acquisition_type

    unsupported = sorted(t for t in msms_types if t in _MSMS_TYPE_UNSUPPORTED)
    if unsupported:
        described = ", ".join(f"{t} ({_MSMS_TYPE_UNSUPPORTED[t]})" for t in unsupported)
        raise ValueError(
            f"Unsupported acquisition type in {Path(analysis_dir)}: the run's MS/MS frames are "
            f"MsMsType {described}. DReader supports PASEF DDA (8), DIA (9) and PRM (10). "
            "Convert the run to mzML (e.g. with msconvert) and open it with MzmlReader instead."
        )

    return AcquisitionType.UNKNOWN


@dataclass
class CentroidConfig:
    """Parameters forwarded to tdfpy's ``frame.centroid()`` for Bruker .d files.

    Only relevant for ``DReader``; ignored by ``MzmlReader``.
    """

    mz_tolerance: float = 8.0
    mz_tolerance_type: Literal["ppm", "da"] = "ppm"
    im_tolerance: float = 0.1
    im_tolerance_type: Literal["relative", "absolute"] = "relative"
    min_peaks: int = 3
    noise_filter: Literal["mad", "percentile", "histogram", "baseline", "iterative_median"] | float | None = None


# ---------------------------------------------------------------------------
# DReader lookup objects
# ---------------------------------------------------------------------------


class DReaderMs1Lookup:
    """Iterable + index-accessible MS1 spectra from a DReader.

    Iteration yields all MS1 spectra. Index access (``lookup[frame_id]``)
    fetches a single spectrum by tdfpy ``frame_id``.
    """

    def __init__(self, dreader: DReader) -> None:
        self._dr = dreader

    def _open_reader(self) -> Any:
        if self._dr._reader is None:
            raise RuntimeError("DReader must be opened before use (call open() or use as a context manager)")
        return self._dr._reader

    def __iter__(self) -> Iterator[MsnSpectrum]:
        reader = self._open_reader()
        mz_range = reader.metadata.mz_acq_range
        im_range = reader.metadata.one_over_k0_acq_range
        for frame in reader.ms1:
            yield self._dr._parse_ms1_frame(frame, mz_range, im_range)

    def __getitem__(self, frame_id: int) -> MsnSpectrum:
        """Fetch a single MS1 spectrum by tdfpy frame_id."""
        reader = self._open_reader()
        mz_range = reader.metadata.mz_acq_range
        im_range = reader.metadata.one_over_k0_acq_range
        frame = reader.ms1[frame_id]  # raises KeyError if not found
        return self._dr._parse_ms1_frame(frame, mz_range, im_range)


class DReaderMs2Lookup:
    """Iterable + index-accessible MS2 spectra from a DReader.

    Iteration yields all MS2 spectra (DDA precursors, DIA windows, or PRM
    transitions depending on acquisition type). Index access
    (``lookup[precursor_id]``) fetches a single spectrum by tdfpy
    ``precursor_id`` (DDA only).
    """

    def __init__(self, dreader: DReader) -> None:
        self._dr = dreader

    def _open_reader(self) -> Any:
        if self._dr._reader is None:
            raise RuntimeError("DReader must be opened before use (call open() or use as a context manager)")
        return self._dr._reader

    def __iter__(self) -> Iterator[MsnSpectrum]:
        reader = self._open_reader()
        match self._dr.acquisition_type:
            # UNKNOWN is opened with the DDA backend (see DReader.open), so it
            # must be iterated as DDA too rather than rejected here.
            case AcquisitionType.DDA | AcquisitionType.UNKNOWN:
                for precursor in reader.precursors:
                    yield DReader._parse_dda_precursor(precursor)
            case AcquisitionType.DIA:
                for window in reader.windows:
                    yield self._dr._parse_dia_window(window)
            case AcquisitionType.PRM:
                for transition in reader.transitions:
                    yield self._dr._parse_prm_transition(transition)
            case _:
                raise ValueError(f"Unsupported acquisition type: {self._dr.acquisition_type}")

    def __getitem__(self, precursor_id: int) -> MsnSpectrum:
        """Fetch a single MS2 spectrum by tdfpy precursor_id (DDA only)."""
        reader = self._open_reader()
        match self._dr.acquisition_type:
            # UNKNOWN is opened with the DDA backend (see DReader.open).
            case AcquisitionType.DDA | AcquisitionType.UNKNOWN:
                precursor = reader.precursors[precursor_id]  # KeyError if not found
                return DReader._parse_dda_precursor(precursor)
            case AcquisitionType.DIA:
                raise NotImplementedError(
                    "DIA MS2 lookup by ID is not supported: DIA windows map to multiple frames. "
                    "Iterate reader.ms2 instead."
                )
            case AcquisitionType.PRM:
                raise NotImplementedError(
                    "PRM MS2 lookup by ID is not supported: PRM transitions are keyed by "
                    "(frame_id, target_id). Iterate reader.ms2 instead, or access via the "
                    "underlying tdfpy reader.targets / reader.transitions lookups."
                )
            case _:
                raise ValueError(f"Unsupported acquisition type: {self._dr.acquisition_type}")


# ---------------------------------------------------------------------------
# DReader
# ---------------------------------------------------------------------------


class DReader:
    def __init__(self, analysis_dir: str | Path, centroid_config: CentroidConfig | None = None) -> None:
        if not _HAS_TDFPY:
            raise ImportError(
                "DReader requires the 'tdfpy' package, which is not installed. "
                "Install it with: pip install spxtacular[bruker]"
            )
        import tdfpy as tdf

        self.analysis_dir = analysis_dir
        self._tdf = tdf
        self._centroid_config: CentroidConfig = centroid_config or CentroidConfig()
        self.acquisition_type: AcquisitionType = _detect_acquisition_type(analysis_dir)
        self._reader = None

    def open(self) -> None:
        """Open the underlying tdfpy reader. Call :meth:`close` when done, or use as a context manager."""
        if self._reader is not None:
            self.close()
        match self.acquisition_type:
            case AcquisitionType.DDA | AcquisitionType.UNKNOWN:
                reader = self._tdf.DDA(str(self.analysis_dir))
            case AcquisitionType.DIA:
                reader = self._tdf.DIA(str(self.analysis_dir))
            case AcquisitionType.PRM:
                reader = self._tdf.PRM(str(self.analysis_dir))
            case _:
                raise ValueError(f"Unsupported acquisition type: {self.acquisition_type}")
        # Only publish the handle once it is genuinely open, so a failing
        # __enter__ doesn't leave a half-open reader behind.
        reader.__enter__()
        self._reader = reader

    def close(self) -> None:
        """Close the underlying tdfpy reader."""
        if self._reader is not None:
            self._reader.__exit__(None, None, None)
            self._reader = None

    def __enter__(self) -> DReader:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Conversion helpers (shared by iteration and __getitem__)
    # ------------------------------------------------------------------

    def _centroid(self, obj: Any) -> np.ndarray:
        """Centroid against either the old (<=1.2.0) or new (smoothing-branch) tdfpy API."""
        cfg = self._centroid_config
        if _HAS_NEW_CENTROID_API:
            assert _MergePeaksCentroider is not None
            return obj.centroid(
                centroid=_MergePeaksCentroider(
                    mz_tolerance=cfg.mz_tolerance,
                    mz_tolerance_type=cfg.mz_tolerance_type,
                    im_tolerance=cfg.im_tolerance,
                    im_tolerance_type=cfg.im_tolerance_type,
                    min_peaks=cfg.min_peaks,
                ),
                noise=cfg.noise_filter,
            )
        return obj.centroid(
            mz_tolerance=cfg.mz_tolerance,
            mz_tolerance_type=cfg.mz_tolerance_type,
            im_tolerance=cfg.im_tolerance,
            im_tolerance_type=cfg.im_tolerance_type,
            min_peaks=cfg.min_peaks,
            noise_filter=cfg.noise_filter,
        )

    def _parse_ms1_frame(
        self,
        frame: Any,
        mz_range: tuple[float, float] | None,
        im_range: tuple[float, float] | None,
    ) -> MsnSpectrum:
        centroided_peaks = self._centroid(frame)
        match frame.polarity:
            case "positive":
                polarity = Polarity.POSITIVE
            case "negative":
                polarity = Polarity.NEGATIVE
            case _:
                polarity = None
        return MsnSpectrum(
            mz=centroided_peaks[:, 0],
            intensity=centroided_peaks[:, 1],
            charge=None,
            im=centroided_peaks[:, 2],
            spectrum_type=SpectrumType.CENTROID,
            denoised=None,
            normalized=None,
            scan_number=frame.frame_id,
            ms_level=1,
            native_id=None,
            rt=frame.time,
            injection_time=frame.accumulation_time,
            total_ion_current=None,
            mz_range=mz_range,
            im_range=im_range,
            polarity=polarity,
            resolution=None,
            analyzer=Analyzer.TOF,
            ramp_time=frame.ramp_time,
            precursors=None,
            im_type=IMType.OOK0,
        )

    @staticmethod
    def _parse_dda_precursor(precursor: tdfpy.Precursor) -> MsnSpectrum:
        peaks = precursor.peaks
        match precursor.polarity:
            case "positive":
                polarity = Polarity.POSITIVE
            case "negative":
                polarity = Polarity.NEGATIVE
            case _:
                polarity = None
        target_mz = precursor.monoisotopic_mz
        is_monoisotopic = True
        if target_mz is None:
            target_mz = precursor.largest_peak_mz
            is_monoisotopic = False
        prec = Precursor(
            mz=target_mz,
            intensity=precursor.intensity,
            charge=precursor.charge,
            im=precursor.ook0,
            is_monoisotopic=is_monoisotopic,
        )

        return MsnSpectrum(
            mz=peaks[:, 0],
            intensity=peaks[:, 1],
            charge=None,
            im=None,
            spectrum_type=SpectrumType.CENTROID,
            denoised=None,
            normalized=None,
            scan_number=precursor.precursor_id,
            ms_level=2,
            native_id=None,
            rt=precursor.rt,
            injection_time=None,
            total_ion_current=None,
            mz_range=None,
            im_range=None,
            polarity=polarity,
            resolution=None,
            analyzer=Analyzer.TOF,
            ramp_time=None,
            precursors=[prec],
            im_type=IMType.OOK0,
            isolation_im_range=precursor.ook0_range,
            isolation_mz_range=precursor.mz_range,
            collision_energy=precursor.collision_energy,
            activation_type=ActivationType.PASEF,
        )

    def _parse_dia_window(self, window: tdfpy.DiaWindow) -> MsnSpectrum:
        peaks = self._centroid(window)
        match window.polarity:
            case "positive":
                polarity = Polarity.POSITIVE
            case "negative":
                polarity = Polarity.NEGATIVE
            case _:
                polarity = None
        native_id = f"{window.frame_id}@w{window.window_index}"
        return MsnSpectrum(
            mz=peaks[:, 0],
            intensity=peaks[:, 1],
            charge=None,
            im=peaks[:, 2],
            spectrum_type=SpectrumType.CENTROID,
            denoised=None,
            normalized=None,
            scan_number=window.frame_id,
            ms_level=2,
            native_id=native_id,
            rt=window.rt,
            injection_time=None,
            total_ion_current=None,
            mz_range=None,
            im_range=None,
            polarity=polarity,
            resolution=None,
            analyzer=Analyzer.TOF,
            collision_energy=window.collision_energy,
            activation_type=ActivationType.PASEF,
            ramp_time=None,
            precursors=None,
            isolation_mz_range=window.mz_range,
            isolation_im_range=window.ook0_range,
            im_type=IMType.OOK0,
        )

    def _parse_prm_transition(self, transition: tdfpy.PrmTransition) -> MsnSpectrum:
        peaks = self._centroid(transition)
        match transition.polarity:
            case "positive":
                polarity = Polarity.POSITIVE
            case "negative":
                polarity = Polarity.NEGATIVE
            case _:
                polarity = None
        target = transition.target
        # PRM targets are user-defined and have no measured precursor intensity;
        # use the sum of the centroided MS2 peak intensities as a proxy.
        precursor_intensity = float(peaks[:, 1].sum()) if len(peaks) else 0.0
        prec = Precursor(
            mz=target.monoisotopic_mz,
            intensity=precursor_intensity,
            charge=target.charge,
            im=target.one_over_k0,
            is_monoisotopic=True,
        )
        native_id = f"{transition.frame_id}@t{target.target_id}"
        return MsnSpectrum(
            mz=peaks[:, 0],
            intensity=peaks[:, 1],
            charge=None,
            im=peaks[:, 2] if peaks.shape[1] > 2 else None,
            spectrum_type=SpectrumType.CENTROID,
            denoised=None,
            normalized=None,
            scan_number=transition.frame_id,
            ms_level=2,
            native_id=native_id,
            rt=transition.rt,
            injection_time=None,
            total_ion_current=None,
            mz_range=None,
            im_range=None,
            polarity=polarity,
            resolution=None,
            analyzer=Analyzer.TOF,
            collision_energy=transition.collision_energy,
            activation_type=ActivationType.PASEF,
            ramp_time=None,
            precursors=[prec],
            isolation_mz_range=transition.mz_range,
            isolation_im_range=transition.ook0_range,
            im_type=IMType.OOK0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def ms1(self) -> DReaderMs1Lookup:
        """MS1 spectra — supports iteration and frame_id-based access."""
        return DReaderMs1Lookup(self)

    @property
    def ms2(self) -> DReaderMs2Lookup:
        """MS2 spectra — supports iteration and precursor_id-based access (DDA only).

        For DIA, iterate to access window-level MS2 spectra. For PRM, iterate to
        access transition-level MS2 spectra; index access raises NotImplementedError
        because PRM transitions are keyed by ``(frame_id, target_id)``.
        """
        return DReaderMs2Lookup(self)


# ---------------------------------------------------------------------------
# MzmlReader lookup object
# ---------------------------------------------------------------------------


class MzmlSpectraLookup:
    """Iterable + index-accessible spectra from an mzML file.

    Iteration yields spectra filtered to ``ms_level`` (if given).
    Index access (``lookup[int]`` or ``lookup[str]``) fetches by overall
    spectrum index or native ID — no level filtering applied on random access.

    Uses the parent :class:`MzmlReader`'s open handle when available (fast path);
    falls back to opening the file per-operation otherwise (backward-compatible).
    """

    def __init__(self, reader: MzmlReader, ms_level: int | None = None) -> None:
        self._reader = reader
        self._ms_level = ms_level

    def __iter__(self) -> Iterator[MsnSpectrum]:
        handle = self._reader._mzml_handle
        if handle is not None:
            # Resolved once per walk: it is a property of the file, not the scan.
            decon = _deconvolution_refs(handle)
            for spec in handle.spectra:
                if self._ms_level is not None and spec.ms_level != self._ms_level:
                    continue
                yield MzmlReader._parse_spectrum(spec, decon)
        else:
            with self._reader._new_handle() as r:
                decon = _deconvolution_refs(r)
                for spec in r.spectra:
                    if self._ms_level is not None and spec.ms_level != self._ms_level:
                        continue
                    yield MzmlReader._parse_spectrum(spec, decon)

    def __getitem__(self, key: int | str) -> MsnSpectrum:
        """Fetch a single spectrum by 0-based index or native ID string."""
        handle = self._reader._mzml_handle
        if handle is not None:
            spec = handle.spectra[key]
            decon = _deconvolution_refs(handle)
        else:
            with self._reader._new_handle() as r:
                spec = r.spectra[key]
                decon = _deconvolution_refs(r)
        return MzmlReader._parse_spectrum(spec, decon)


# ---------------------------------------------------------------------------
# MzmlReader
# ---------------------------------------------------------------------------


# mzML has no per-spectrum "these are neutral masses" flag; the closest standard
# signal that a spectrum's charges were resolved is the charge-deconvolution
# data-transformation term. In practice it lives under
# <dataProcessing><processingMethod>, which a spectrum points at through its
# dataProcessingRef — so both places are checked (see _deconvolution_refs).
_DECONVOLUTION_ACCESSIONS: frozenset[str] = frozenset({"MS:1000034"})  # charge deconvolution


class _DeconvolutionRefs(NamedTuple):
    """Which ``dataProcessing`` entries of a file declare charge deconvolution.

    Attributes
    ----------
    ids:
        ``dataProcessing`` ids whose processing methods carry a deconvolution
        term. A spectrum whose ``dataProcessingRef`` is one of these had its
        charges resolved.
    unreferenced:
        Whether a spectrum that names no ``dataProcessingRef`` inherits one.
    """

    ids: frozenset[str] = frozenset()
    unreferenced: bool = False


_NO_DECONVOLUTION = _DeconvolutionRefs()
"""A file that declares no charge deconvolution anywhere."""


def _deconvolution_refs(handle: Any) -> _DeconvolutionRefs:
    """Find the file-level ``dataProcessing`` entries that declare deconvolution.

    A spectrum without an explicit ``dataProcessingRef`` inherits
    ``spectrumList/@defaultDataProcessingRef``, which mzmlpy does not expose. It
    can still be resolved when the file declares exactly one ``dataProcessing``,
    since that entry is then necessarily the default; with several, the
    unreferenced spectra are left alone rather than guessed at.
    """
    processes = getattr(handle, "data_processes", None)
    if not processes:
        return _NO_DECONVOLUTION

    ids = frozenset(
        dp_id
        for dp_id, dp in processes.items()
        if any(_DECONVOLUTION_ACCESSIONS & method.accessions for method in dp.processing_methods)
    )
    return _DeconvolutionRefs(ids=ids, unreferenced=len(processes) == 1 and bool(ids))


class MzmlReader:
    """Read spectra from an mzML or gzipped mzML file.

    Parameters
    ----------
    mzml_path:
        Path to the mzML file.
    gzip_mode:
        How mzmlpy opens gzipped input. ``"extract"`` preserves the historical
        behavior and gives fast random access after an up-front extraction.
        ``"stream"`` starts immediately and is well suited to sequential reads.
        ``"indexed"`` builds a random-access gzip index and requires rapidgzip.
    in_memory:
        Whether mzmlpy should keep its XML index in memory.
    extract_dir:
        Optional directory for files produced by ``gzip_mode="extract"``.
    """

    def __init__(
        self,
        mzml_path: str | Path,
        *,
        gzip_mode: Literal["extract", "indexed", "stream"] = "extract",
        in_memory: bool = True,
        extract_dir: str | Path | None = None,
    ) -> None:
        if not _HAS_MZMLPY:
            raise ImportError(
                "MzmlReader requires the 'mzmlpy' package, which is not installed. "
                "Install it with: pip install spxtacular[mzml]"
            )
        self.mzml_path = mzml_path
        self.gzip_mode = gzip_mode
        self.in_memory = in_memory
        self.extract_dir = extract_dir
        self._mzml_handle = None

    def _new_handle(self) -> Any:
        """Create an mzmlpy handle with this reader's public I/O options."""
        return mzp.Mzml(
            self.mzml_path,
            gzip_mode=self.gzip_mode,
            in_memory=self.in_memory,
            extract_dir=self.extract_dir,
        )

    @staticmethod
    def _parse_spectrum(spec: mzp.Spectrum, decon: _DeconvolutionRefs = _NO_DECONVOLUTION) -> MsnSpectrum:
        """Convert a raw mzmlpy Spectrum into an MsnSpectrum.

        ``decon`` carries the file's deconvolution ``dataProcessing`` ids (see
        :func:`_deconvolution_refs`); the default treats the file as declaring
        none, so a spectrum is only deconvoluted if it says so itself.
        """
        mz_array = spec.mz
        if mz_array is None:
            raise ValueError(f"Spectrum {spec} has no m/z array")
        mz_array = mz_array.astype(np.float64)

        int_array = spec.intensity
        if int_array is None:
            raise ValueError(f"Spectrum {spec} has no intensity array")
        int_array = int_array.astype(np.float64)

        if len(mz_array) != len(int_array):
            raise ValueError(f"Spectrum {spec} has m/z and intensity arrays of different lengths")

        charge_array = spec.charge
        if charge_array is not None:
            charge_array = charge_array.astype(np.int32)
            if len(charge_array) != len(mz_array):
                raise ValueError(f"Spectrum {spec} has charge array of different length than m/z array")

        im_array: np.ndarray | None = None
        im_types = list(spec.im_types)
        if len(im_types) == 1:
            darr = spec.get_binary_array(im_types[0])
            if darr is None:
                raise RuntimeError(f"Spectrum {spec} has ion mobility array type {im_types[0]} but it is None")
            im_array = darr.data.astype(np.float64)
            if len(im_array) != len(mz_array):
                raise ValueError(f"Spectrum {spec} has ion mobility array of different length than m/z array")
        elif len(im_types) > 1:
            warnings.warn(
                f"Spectrum {spec} has multiple ion mobility arrays; only the first is used: {im_types[0]}",
                stacklevel=3,
            )
            for im_type in im_types:
                darr = spec.get_binary_array(im_type)
                if darr is None:
                    raise RuntimeError(
                        f"Spectrum {spec}: multiple IM arrays, first is not None. Array types: {im_types}"
                    )
                candidate = darr.data.astype(np.float64)
                if len(candidate) == len(mz_array):
                    im_array = candidate
                    break
            if im_array is None:
                warnings.warn(
                    f"Spectrum {spec}: no ion mobility array length matches m/z array. Array types: {im_types}",
                    stacklevel=3,
                )

        match spec.spectrum_type:
            case "centroid":
                spectrum_type = SpectrumType.CENTROID
            case "profile":
                spectrum_type = SpectrumType.PROFILE
            case _:
                raise ValueError(f"Spectrum {spec} has unrecognized spectrum type: {spec.spectrum_type}")

        # A charge array alone does NOT mean the spectrum is deconvoluted: mzML
        # charge arrays are usually per-peak charge *annotations* on ordinary
        # centroid data, and mzML's 0 ("unknown charge") collides with
        # spxtacular's 0 ("already decharged"). Require an explicit
        # deconvolution CV term before overriding centroid/profile. The term is
        # rarely written on the spectrum itself — it standardly sits on the
        # <processingMethod> the spectrum's dataProcessingRef points at — so the
        # referenced processing is consulted too.
        if charge_array is not None:
            processing_ref = spec.data_processing_ref
            declared_by_processing = processing_ref in decon.ids if processing_ref is not None else decon.unreferenced
            if bool(_DECONVOLUTION_ACCESSIONS & spec.accessions) or declared_by_processing:
                spectrum_type = SpectrumType.DECONVOLUTED

        mz_range = None
        if spec.lower_mz is not None and spec.upper_mz is not None:
            mz_range = (spec.lower_mz, spec.upper_mz)

        precursors: list[Precursor] = []
        collision_energies: list[float] = []
        activation_types: list[ActivationType | str] = []
        isolation_ranges: list[tuple[float, float]] = []

        for precursor in spec.precursors:
            ions = precursor.selected_ions
            if len(ions) == 0:
                warnings.warn(
                    f"Spectrum {spec} has precursor with no selected ions. Precursor: {precursor}",
                    stacklevel=3,
                )
                continue
            if len(ions) > 1:
                warnings.warn(
                    f"Spectrum {spec} has multiple selected ions; using first. Precursor: {precursor}",
                    stacklevel=3,
                )
            ion = ions[0]
            mz = ion.selected_ion_mz
            if mz is None:
                warnings.warn(
                    f"Spectrum {spec} precursor selected ion missing m/z. Precursor: {precursor}",
                    stacklevel=3,
                )
                continue
            intensity = ion.peak_intensity
            if intensity is None:
                warnings.warn(
                    f"Spectrum {spec} precursor missing intensity. Precursor: {precursor}",
                    stacklevel=3,
                )
                continue
            precursors.append(
                Precursor(mz=mz, intensity=intensity, charge=ion.charge_state, im=ion.ir_im, is_monoisotopic=None)
            )
            activation = precursor.activation
            if activation is not None:
                if activation.ce is not None:
                    collision_energies.append(activation.ce)
                if activation.activation_type is not None:
                    # mzmlpy yields the raw PSI-MS accession (as a vendor enum);
                    # normalise to spxtacular's canonical ActivationType member.
                    activation_types.append(ActivationType.from_accession(str(activation.activation_type)))
            if precursor.isolation_window is not None:
                has_target_mz = precursor.isolation_window.target_mz is not None
                has_lower = precursor.isolation_window.lower_offset is not None
                has_upper = precursor.isolation_window.upper_offset is not None
                if has_target_mz and has_lower and has_upper:
                    isolation_ranges.append(
                        (
                            precursor.isolation_window.target_mz - precursor.isolation_window.lower_offset,  # type: ignore
                            precursor.isolation_window.target_mz + precursor.isolation_window.upper_offset,  # type: ignore
                        )
                    )
        if len(set(collision_energies)) > 1:
            warnings.warn(f"Spectrum {spec} has multiple collision energies: {set(collision_energies)}", stacklevel=3)
        if len(set(activation_types)) > 1:
            warnings.warn(f"Spectrum {spec} has multiple activation types: {set(activation_types)}", stacklevel=3)
        if len(set(isolation_ranges)) > 1:
            warnings.warn(
                f"Spectrum {spec} has multiple isolation window ranges: {set(isolation_ranges)}", stacklevel=3
            )

        return MsnSpectrum(
            mz=mz_array,
            intensity=int_array,
            charge=charge_array,
            im=im_array,
            spectrum_type=spectrum_type,
            denoised=None,
            normalized=None,
            scan_number=spec.index,
            ms_level=spec.ms_level,
            native_id=spec.id,
            rt=spec.scan_start_time.total_seconds() if spec.scan_start_time is not None else None,
            total_ion_current=spec.TIC,
            mz_range=mz_range,
            im_range=None,
            polarity=Polarity(spec.polarity) if spec.polarity is not None else None,
            resolution=None,
            analyzer=None,
            collision_energy=collision_energies[0] if collision_energies else None,
            activation_type=activation_types[0] if activation_types else None,
            ramp_time=None,
            precursors=precursors if precursors else None,
            isolation_mz_range=isolation_ranges[0] if isolation_ranges else None,
        )

    @property
    def ms1(self) -> MzmlSpectraLookup:
        """MS1 spectra — supports iteration and index/native-ID-based access."""
        return MzmlSpectraLookup(self, ms_level=1)

    @property
    def ms2(self) -> MzmlSpectraLookup:
        """MS2 spectra — supports iteration and index/native-ID-based access."""
        return MzmlSpectraLookup(self, ms_level=2)

    def __getitem__(self, key: int | str) -> MsnSpectrum:
        """Fetch a single spectrum by 0-based index or native ID string.

        Examples::

            reader[0]           # first spectrum by overall index
            reader["scan=19"]   # by full native ID
        """
        return MzmlSpectraLookup(self)[key]

    def open(self) -> None:
        """Open a persistent mzmlpy reader. Call :meth:`close` when done, or use as a context manager."""
        if self._mzml_handle is not None:
            self.close()
        handle = self._new_handle()
        # Only publish the handle once it is genuinely open, so a failing
        # __enter__ doesn't leave a half-open reader behind.
        handle.__enter__()
        self._mzml_handle = handle

    def close(self) -> None:
        """Close the persistent mzmlpy reader."""
        if self._mzml_handle is not None:
            self._mzml_handle.__exit__(None, None, None)
            self._mzml_handle = None

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Unified Reader
# ---------------------------------------------------------------------------


class Reader:
    """Format-agnostic reader — detects the format from the path.

    Usage is identical regardless of the underlying format::

        with Reader("data.mzML") as r:
            for spec in r.ms1:
                ...

        with Reader("data.d") as r:
            ms2 = r.ms2[42]

        with Reader("data.mgf") as r:
            for spec in r.ms2:
                ...

    Parameters
    ----------
    path:
        Path to a Bruker ``.d`` directory, an ``.mzML`` file, a Thermo
        ``.raw`` file, or an ``.mgf`` / ``.ms2`` / ``.msp`` peak list. Every
        text format may be gzipped (``.mzML.gz``, ``.mgf.gz``, ``.ms2.gz``,
        ``.msp.gz``). Extension matching is case-insensitive.
    centroid_config:
        Optional Bruker centroiding settings.
    mzml_gzip_mode:
        Gzip strategy forwarded to :class:`MzmlReader`. Use ``"stream"`` for
        low-latency sequential access to large gzipped mzML files.
    mzml_in_memory:
        Whether mzmlpy should keep its XML index in memory.
    mzml_extract_dir:
        Optional directory for mzmlpy's extracted gzip content.

    Raises
    ------
    ValueError
        If the path extension is not recognised.

    Notes
    -----
    ``.mgf`` and ``.ms2`` hold fragmentation spectra only, so ``.ms1`` on those
    inputs is a valid but always empty walk.
    """

    def __init__(
        self,
        path: str | Path,
        centroid_config: CentroidConfig | None = None,
        *,
        mzml_gzip_mode: Literal["extract", "indexed", "stream"] = "extract",
        mzml_in_memory: bool = True,
        mzml_extract_dir: str | Path | None = None,
    ) -> None:
        p = Path(path)
        suffixes = [s.lower() for s in p.suffixes]
        if suffixes and suffixes[-1] == ".gz":
            suffixes = suffixes[:-1]
        suffix = suffixes[-1] if suffixes else ""
        if suffix == ".d":
            self._reader: DReader | MzmlReader | ThermoReader | MgfReader | Ms2Reader | MspReader = DReader(
                p, centroid_config=centroid_config
            )
        elif suffix == ".mzml":
            self._reader = MzmlReader(
                p,
                gzip_mode=mzml_gzip_mode,
                in_memory=mzml_in_memory,
                extract_dir=mzml_extract_dir,
            )
        elif suffix == ".raw":
            self._reader = ThermoReader(p)
        elif suffix == ".mgf":
            self._reader = MgfReader(p)
        elif suffix == ".ms2":
            self._reader = Ms2Reader(p)
        elif suffix == ".msp":
            self._reader = MspReader(p)
        else:
            raise ValueError(
                f"Unsupported format {p.suffix!r}. Expected '.d', '.mzML', '.raw', '.mgf', '.ms2', or '.msp' "
                "(the text formats optionally gzipped)."
            )

    @property
    def ms1(self) -> DReaderMs1Lookup | MzmlSpectraLookup | ThermoScanLookup | PeakListLookup:
        """MS1 spectra — supports iteration and index-based access."""
        return self._reader.ms1

    @property
    def ms2(self) -> DReaderMs2Lookup | MzmlSpectraLookup | ThermoScanLookup | PeakListLookup:
        """MS2 spectra — supports iteration and index-based access."""
        return self._reader.ms2

    def open(self) -> None:
        """Open the underlying reader."""
        self._reader.open()

    def close(self) -> None:
        """Close the underlying reader."""
        self._reader.close()

    def __enter__(self) -> Reader:
        self._reader.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._reader.close()
