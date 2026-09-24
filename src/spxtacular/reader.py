"""Unified reader API for different mass-spectrometry file formats.

Supports DDA, DIA, and PRM data from Bruker timsTOF (.d) and mzML, Thermo
.raw files (see thermo.py), plus the MGF, MS2, and MSP peak-list formats (see
peaklist.py).
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, Self, runtime_checkable

import numpy as np

from ._scan_lookup import (
    IdIndex,
    build_id_index,
    by_sage_scannr,
    check_ms_level,
    check_scan_number,
    native_id_scan_number,
)
from .core import MsnSpectrum, Precursor, SpectrumType
from .enums import ActivationType, Analyzer, IMType, Polarity
from .errors import SpxtacularError
from .peaklist import MgfReader, Ms2Reader, MspReader
from .thermo import ThermoReader

if TYPE_CHECKING:
    from mzmlpy import Spectrum as MzmlSpectrum
    from tdfpy import DiaWindow, PrmTransition
    from tdfpy import Precursor as TdfPrecursor

# The optional backends load native libraries, so a broken install can raise
# OSError rather than ImportError. Either way the backend is simply unavailable —
# it must not take down ``import spxtacular`` for users who never touch it.
try:
    import mzmlpy as mzp

    _HAS_MZMLPY = True
except (ImportError, OSError):
    mzp = None
    _HAS_MZMLPY = False

try:
    import tdfpy

    _HAS_TDFPY = True
except (ImportError, OSError):
    tdfpy = None
    _HAS_TDFPY = False

if _HAS_TDFPY:
    assert tdfpy is not None
    AcquisitionType = tdfpy.AcquisitionType
else:  # pragma: no cover - exercised only without the [bruker] extra

    class AcquisitionType(StrEnum):  # type: ignore[no-redef]
        """Acquisition scheme of a Bruker timsTOF run (mirror of ``tdfpy.AcquisitionType``)."""

        DDA = "DDA"
        DIA = "DIA"
        PRM = "PRM"
        UNKNOWN = "unknown"


@runtime_checkable
class SpectrumLookup(Protocol):
    """What every reader's ``ms1`` / ``ms2`` view provides.

    A lookup is iterable (a fresh walk each time) and indexable by the key its
    format uses (Bruker frame or precursor id, mzML index or native id, Thermo
    scan number, peak-list position). It is not an iterator and has no ``len()``.
    """

    def __iter__(self) -> Iterator[MsnSpectrum]: ...

    def __getitem__(self, key: Any, /) -> MsnSpectrum: ...


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

    Mirrors ``tdfpy.get_acquisition_type`` (and returns its enum) but also
    rejects MS/MS schemes spxtacular cannot read, with a message naming the scheme.

    Raises
    ------
    FileNotFoundError
        If the folder holds no ``analysis.tdf``.
    SpxtacularError
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
        raise SpxtacularError(
            f"Unsupported acquisition type in {Path(analysis_dir)}: the run's MS/MS frames are "
            f"MsMsType {described}. DReader supports PASEF DDA (8), DIA (9) and PRM (10). "
            "Convert the run to mzML (e.g. with msconvert) and open it with MzmlReader instead."
        )

    return AcquisitionType.UNKNOWN


@dataclass(kw_only=True)
class CentroidConfig:
    """Parameters forwarded to tdfpy's ``Frame.centroid()`` for Bruker .d files.

    Applies to MS1 frames, DIA windows and PRM transitions. DDA (PASEF) MS2
    spectra ignore it: tdfpy returns them already merged across their PASEF
    windows and mobility-collapsed. Only relevant for ``DReader``; ignored by
    every other reader.
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
            raise SpxtacularError("DReader must be opened before use (call open() or use as a context manager)")
        return self._dr._reader

    def __iter__(self) -> Iterator[MsnSpectrum]:
        reader = self._open_reader()
        mz_range = reader.metadata.mz_acq_range
        im_range = reader.metadata.ook0_acq_range
        for frame in reader.ms1:
            yield self._dr._parse_ms1_frame(frame, mz_range, im_range)

    def __getitem__(self, frame_id: int) -> MsnSpectrum:
        """Fetch a single MS1 spectrum by tdfpy frame_id."""
        reader = self._open_reader()
        mz_range = reader.metadata.mz_acq_range
        im_range = reader.metadata.ook0_acq_range
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
            raise SpxtacularError("DReader must be opened before use (call open() or use as a context manager)")
        return self._dr._reader

    def __iter__(self) -> Iterator[MsnSpectrum]:
        reader = self._open_reader()
        match self._dr.acquisition_type:
            # UNKNOWN is opened with the DDA backend (see DReader.open), so it
            # must be iterated as DDA too rather than rejected here.
            case AcquisitionType.DDA | AcquisitionType.UNKNOWN:
                # Decodes each PASEF frame once for all the precursors it holds.
                assert tdfpy is not None
                for precursor, peaks in tdfpy.iter_precursor_spectra(reader.precursors):
                    yield DReader._parse_dda_precursor(precursor, peaks)
            case AcquisitionType.DIA:
                for window in reader.windows:
                    yield self._dr._parse_dia_window(window)
            case AcquisitionType.PRM:
                for transition in reader.transitions:
                    yield self._dr._parse_prm_transition(transition)
            case _:
                raise SpxtacularError(f"Unsupported acquisition type: {self._dr.acquisition_type}")

    def __getitem__(self, precursor_id: int) -> MsnSpectrum:
        """Fetch a single MS2 spectrum by tdfpy precursor_id (DDA only)."""
        reader = self._open_reader()
        match self._dr.acquisition_type:
            # UNKNOWN is opened with the DDA backend (see DReader.open).
            case AcquisitionType.DDA | AcquisitionType.UNKNOWN:
                precursor = reader.precursors[precursor_id]  # KeyError if not found
                return DReader._parse_dda_precursor(precursor, precursor.merged_peaks())
            case AcquisitionType.DIA:
                raise NotImplementedError(
                    "DIA MS2 lookup by ID is not supported: DIA windows map to multiple frames. "
                    "Use reader.get_by_native_id('<frame>@w<window_index>') or iterate reader.ms2."
                )
            case AcquisitionType.PRM:
                raise NotImplementedError(
                    "PRM MS2 lookup by ID is not supported: PRM transitions are keyed by "
                    "(frame_id, target_id). Use reader.get_by_native_id('<frame>@t<target_id>') "
                    "or iterate reader.ms2."
                )
            case _:
                raise SpxtacularError(f"Unsupported acquisition type: {self._dr.acquisition_type}")


# ---------------------------------------------------------------------------
# DReader
# ---------------------------------------------------------------------------


def _tdf_polarity(value: str | None) -> Polarity | None:
    """Map tdfpy's ``Polarity`` literal (``"positive"`` / ``"negative"``) to spxtacular's enum."""
    match value:
        case "positive":
            return Polarity.POSITIVE
        case "negative":
            return Polarity.NEGATIVE
        case _:
            return None


_DREADER_FRAME_ID_RE = re.compile(r"frame=(\d+)")
_DREADER_PRECURSOR_ID_RE = re.compile(r"precursor=(\d+)")
_DREADER_FRAME_ITEM_ID_RE = re.compile(r"(\d+)@([wt])(\d+)")


class DReader:
    """Reader for Bruker timsTOF ``.d`` directories.

    The optional ``tdfpy`` backend is loaded lazily. Open the reader before
    accessing ``ms1`` or ``ms2``, preferably by using it as a context manager.
    """

    def __init__(self, analysis_dir: str | Path, *, centroid_config: CentroidConfig | None = None) -> None:
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
        self._by_frame: dict[int, list[Any]] | None = None

    def open(self) -> None:
        """Open the underlying tdfpy reader. Call :meth:`close` when done, or use as a context manager."""
        self._by_frame = None
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
                raise SpxtacularError(f"Unsupported acquisition type: {self.acquisition_type}")
        # Only publish the handle once it is genuinely open, so a failing
        # __enter__ doesn't leave a half-open reader behind.
        reader.__enter__()
        self._reader = reader

    def close(self) -> None:
        """Close the underlying tdfpy reader."""
        self._by_frame = None
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
        """Centroid a tdfpy frame, DIA window or PRM transition with this reader's config."""
        assert tdfpy is not None
        cfg = self._centroid_config
        return obj.centroid(
            centroid=tdfpy.MergePeaksCentroider(
                mz_tolerance=cfg.mz_tolerance,
                mz_tolerance_type=cfg.mz_tolerance_type,
                im_tolerance=cfg.im_tolerance,
                im_tolerance_type=cfg.im_tolerance_type,
                min_peaks=cfg.min_peaks,
            ),
            noise=cfg.noise_filter,
        )

    def _parse_ms1_frame(
        self,
        frame: Any,
        mz_range: tuple[float, float] | None,
        im_range: tuple[float, float] | None,
    ) -> MsnSpectrum:
        centroided_peaks = self._centroid(frame)
        return MsnSpectrum(
            mz=centroided_peaks[:, 0],
            intensity=centroided_peaks[:, 1],
            im=centroided_peaks[:, 2],
            spectrum_type=SpectrumType.CENTROID,
            scan_number=frame.frame_id,
            ms_level=1,
            native_id=f"frame={frame.frame_id}",
            rt=frame.rt,
            injection_time=frame.accumulation_time,
            total_ion_current=frame.total_ion_current,
            mz_range=mz_range,
            im_range=im_range,
            polarity=_tdf_polarity(frame.polarity),
            analyzer=Analyzer.TOF,
            ramp_time=frame.ramp_time,
            im_type=IMType.OOK0,
        )

    @staticmethod
    def _parse_dda_precursor(precursor: TdfPrecursor, peaks: np.ndarray) -> MsnSpectrum:
        """Build an MS2 spectrum from a PASEF precursor and its merged ``(N, 2)`` peaks."""
        prec = Precursor(
            precursor_mz=precursor.precursor_mz,
            intensity=precursor.intensity,
            charge=precursor.charge,
            im=precursor.ook0,
            im_type=IMType.OOK0,
            is_monoisotopic=precursor.monoisotopic_mz is not None,
        )
        return MsnSpectrum(
            mz=peaks[:, 0],
            intensity=peaks[:, 1],
            spectrum_type=SpectrumType.CENTROID,
            scan_number=precursor.precursor_id,
            ms_level=2,
            native_id=f"precursor={precursor.precursor_id}",
            rt=precursor.rt,
            polarity=_tdf_polarity(precursor.polarity),
            analyzer=Analyzer.TOF,
            precursors=[prec],
            im_type=IMType.OOK0,
            isolation_ook0_range=precursor.ook0_range,
            isolation_mz_range=precursor.isolation_mz_range,
            collision_energy=precursor.collision_energy,
            activation_type=ActivationType.PASEF,
        )

    def _parse_dia_window(self, window: DiaWindow) -> MsnSpectrum:
        peaks = self._centroid(window)
        return MsnSpectrum(
            mz=peaks[:, 0],
            intensity=peaks[:, 1],
            im=peaks[:, 2],
            spectrum_type=SpectrumType.CENTROID,
            scan_number=window.frame_id,
            ms_level=2,
            native_id=f"{window.frame_id}@w{window.window_index}",
            rt=window.rt,
            polarity=_tdf_polarity(window.polarity),
            analyzer=Analyzer.TOF,
            collision_energy=window.collision_energy,
            activation_type=ActivationType.PASEF,
            isolation_mz_range=window.isolation_mz_range,
            isolation_ook0_range=window.ook0_range,
            im_type=IMType.OOK0,
        )

    def _parse_prm_transition(self, transition: PrmTransition) -> MsnSpectrum:
        peaks = self._centroid(transition)
        target = transition.target
        # PRM targets are user-defined and have no measured precursor intensity;
        # use the sum of the centroided MS2 peak intensities as a proxy.
        precursor_intensity = float(peaks[:, 1].sum()) if len(peaks) else 0.0
        prec = Precursor(
            precursor_mz=target.precursor_mz,
            intensity=precursor_intensity,
            charge=target.charge,
            im=target.ook0,
            im_type=IMType.OOK0 if target.ook0 is not None else None,
            is_monoisotopic=True,
        )
        return MsnSpectrum(
            mz=peaks[:, 0],
            intensity=peaks[:, 1],
            im=peaks[:, 2] if peaks.shape[1] > 2 else None,
            spectrum_type=SpectrumType.CENTROID,
            scan_number=transition.frame_id,
            ms_level=2,
            native_id=f"{transition.frame_id}@t{target.target_id}",
            rt=transition.rt,
            polarity=_tdf_polarity(transition.polarity),
            analyzer=Analyzer.TOF,
            collision_energy=transition.collision_energy,
            activation_type=ActivationType.PASEF,
            precursors=[prec],
            isolation_mz_range=transition.isolation_mz_range,
            isolation_ook0_range=transition.ook0_range,
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

    # ------------------------------------------------------------------
    # Lookup by scan number / native id
    # ------------------------------------------------------------------

    def _open_reader(self) -> Any:
        if self._reader is None:
            raise SpxtacularError("DReader must be opened before use (call open() or use as a context manager)")
        return self._reader

    def _is_dda(self) -> bool:
        # UNKNOWN is opened with the DDA backend (see open()).
        return self.acquisition_type in (AcquisitionType.DDA, AcquisitionType.UNKNOWN)

    def _ms2_by_frame(self) -> dict[int, list[Any]]:
        """DIA windows or PRM transitions grouped by frame id, built on first use and cached."""
        if self._by_frame is None:
            reader = self._open_reader()
            items = reader.windows if self.acquisition_type == AcquisitionType.DIA else reader.transitions
            grouped: dict[int, list[Any]] = {}
            for item in items:
                grouped.setdefault(item.frame_id, []).append(item)
            self._by_frame = grouped
        return self._by_frame

    def _ms1_spectrum(self, frame_id: int) -> MsnSpectrum:
        reader = self._open_reader()
        frame = reader.ms1[frame_id]  # KeyError if not an MS1 frame
        return self._parse_ms1_frame(frame, reader.metadata.mz_acq_range, reader.metadata.ook0_acq_range)

    def _dda_spectrum(self, precursor_id: int) -> MsnSpectrum:
        precursor = self._open_reader().precursors[precursor_id]  # KeyError if not found
        return self._parse_dda_precursor(precursor, precursor.merged_peaks())

    def _frame_item_spectrum(self, item: Any) -> MsnSpectrum:
        if self.acquisition_type == AcquisitionType.DIA:
            return self._parse_dia_window(item)
        return self._parse_prm_transition(item)

    def get_by_scan(self, scan_number: int, *, ms_level: int | None = None) -> MsnSpectrum:
        """Fetch a spectrum by its scan number (see ``MsnSpectrum.scan_number``).

        The scan number is the MS1 ``frame_id``, the DDA ``precursor_id``, or the
        DIA / PRM MS2 ``frame_id``. MS1 frame ids and DDA precursor ids are
        separate number spaces that overlap, so pass ``ms_level`` for DDA data.

        Parameters
        ----------
        scan_number:
            Frame id (MS1, DIA, PRM) or precursor id (DDA MS2).
        ms_level:
            ``1`` or ``2`` to choose the number space; ``None`` accepts either
            when only one of them has the number.

        Raises
        ------
        KeyError
            If no spectrum of the requested level has this number.
        SpxtacularError
            If ``ms_level`` is ``None`` and both an MS1 frame and a DDA precursor
            have this number, if a DIA or PRM frame holds several spectra (one
            per window or target; use :meth:`get_by_native_id`), or if the
            reader is not open.
        """
        scan_number = check_scan_number(scan_number)
        check_ms_level(ms_level)
        reader = self._open_reader()
        is_ms1 = ms_level in (None, 1) and scan_number in reader.ms1
        items: list[Any] = []
        is_ms2 = False
        if ms_level in (None, 2):
            if self._is_dda():
                is_ms2 = scan_number in reader.precursors
            else:
                items = self._ms2_by_frame().get(scan_number, [])
                is_ms2 = bool(items)
        if is_ms1 and is_ms2:
            raise SpxtacularError(
                f"{self.analysis_dir}: {scan_number} is both an MS1 frame id and a precursor id; "
                "pass ms_level=1 or ms_level=2"
            )
        if is_ms1:
            return self._ms1_spectrum(scan_number)
        if is_ms2:
            if self._is_dda():
                return self._dda_spectrum(scan_number)
            if len(items) > 1:
                native_ids = [self._frame_item_native_id(item) for item in items]
                raise SpxtacularError(
                    f"{self.analysis_dir}: frame {scan_number} holds {len(items)} MS2 spectra "
                    f"({', '.join(native_ids[:4])}{', …' if len(items) > 4 else ''}); use get_by_native_id()"
                )
            return self._frame_item_spectrum(items[0])
        level = f"MS{ms_level} " if ms_level is not None else ""
        raise KeyError(f"no {level}spectrum with scan number {scan_number} in {self.analysis_dir}")

    def _frame_item_native_id(self, item: Any) -> str:
        if self.acquisition_type == AcquisitionType.DIA:
            return f"{item.frame_id}@w{item.window_index}"
        return f"{item.frame_id}@t{item.target.target_id}"

    def get_by_native_id(self, native_id: str) -> MsnSpectrum:
        """Fetch a spectrum by the ``native_id`` this reader gives it.

        ============================  ===============================
        ``native_id``                 spectrum
        ============================  ===============================
        ``"frame=F"``                 MS1 frame ``F``
        ``"precursor=P"``             DDA precursor ``P``
        ``"F@wI"``                    DIA window row ``I`` of frame ``F``
        ``"F@tT"``                    PRM target ``T`` in frame ``F``
        ============================  ===============================

        Raises
        ------
        KeyError
            If the id is not one of these forms or names no spectrum in the run.
        SpxtacularError
            If the reader is not open.
        """
        if not isinstance(native_id, str):
            raise SpxtacularError(f"native_id must be a str, got {type(native_id).__name__} {native_id!r}")
        self._open_reader()
        if match := _DREADER_FRAME_ID_RE.fullmatch(native_id):
            return self._ms1_spectrum(int(match.group(1)))
        if match := _DREADER_PRECURSOR_ID_RE.fullmatch(native_id):
            if not self._is_dda():
                raise KeyError(f"{native_id!r}: precursor ids exist only in DDA runs, not {self.acquisition_type}")
            return self._dda_spectrum(int(match.group(1)))
        if match := _DREADER_FRAME_ITEM_ID_RE.fullmatch(native_id):
            frame_id, kind, key = int(match.group(1)), match.group(2), int(match.group(3))
            wanted = AcquisitionType.DIA if kind == "w" else AcquisitionType.PRM
            if self.acquisition_type != wanted:
                raise KeyError(f"{native_id!r} is a {wanted} id; this run is {self.acquisition_type}")
            for item in self._ms2_by_frame().get(frame_id, []):
                item_key = item.window_index if kind == "w" else item.target.target_id
                if item_key == key:
                    return self._frame_item_spectrum(item)
            raise KeyError(f"no spectrum with native id {native_id!r} in {self.analysis_dir}")
        raise KeyError(
            f"{native_id!r} is not a DReader native id (expected 'frame=F', 'precursor=P', 'F@wI' or 'F@tT')"
        )

    def get_by_sage_scannr(self, scannr: str | int, *, precursor_offset: int = 1) -> MsnSpectrum:
        """Fetch the DDA MS2 spectrum a Sage ``scannr`` refers to.

        For a Bruker ``.d`` run Sage writes timsrust's spectrum index as a bare
        integer. Upstream Sage (checked against Sage 0.15.0-beta.1, which uses
        timsrust 0.4.2: 2000 of 2000 DDA PSMs matched) numbers spectra from 0
        in precursor order, so ``scannr`` N is precursor ``N + 1``: the default.
        Sage builds on timsrust 0.6 or later write the precursor id itself;
        pass ``precursor_offset=0`` for those.

        Parameters
        ----------
        scannr:
            The ``scannr`` column value, as an int or a string of digits.
        precursor_offset:
            Added to ``scannr`` to get the Bruker precursor id.

        Raises
        ------
        KeyError
            If the run has no such precursor.
        SpxtacularError
            If ``scannr`` is not an integer, or the run is DIA or PRM (Sage
            numbers those by timsrust's expanded window list, which spxtacular
            does not reproduce).
        """
        if not self._is_dda():
            raise SpxtacularError(
                f"Sage scannr lookup supports DDA runs only; this run is {self.acquisition_type}. "
                "Sage numbers DIA/PRM spectra by timsrust's expanded window list, which spxtacular does not reproduce."
            )
        if isinstance(scannr, bool) or not isinstance(scannr, (str, int)):
            raise SpxtacularError(f"scannr must be a str or int, got {type(scannr).__name__} {scannr!r}")
        text = str(scannr).strip()
        if not text.isdigit():
            raise SpxtacularError(f"Sage scannr for a Bruker .d run is a bare integer, got {scannr!r}")
        if isinstance(precursor_offset, bool) or not isinstance(precursor_offset, int):
            raise SpxtacularError(f"precursor_offset must be an int, got {precursor_offset!r}")
        return self._dda_spectrum(int(text) + precursor_offset)


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

    def __init__(self, reader: MzmlReader, *, ms_level: int | None = None) -> None:
        self._reader = reader
        self._ms_level = ms_level

    def __iter__(self) -> Iterator[MsnSpectrum]:
        with _mzml_errors(self._reader.mzml_path):
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
        with _mzml_errors(self._reader.mzml_path):
            handle = self._reader._mzml_handle
            if handle is not None:
                spec = handle.spectra[key]
                decon = _deconvolution_refs(handle)
            else:
                with self._reader._new_handle() as r:
                    spec = r.spectra[key]
                    decon = _deconvolution_refs(r)
            return MzmlReader._parse_spectrum(spec, decon)


@contextmanager
def _mzml_errors(path: object) -> Iterator[None]:
    """Re-raise mzmlpy's errors about a malformed file as :class:`SpxtacularError`.

    The original is chained. A missing record stays a ``KeyError`` (mzmlpy's
    ``MzmlRecordNotFoundError``), so ``except KeyError`` lookups keep working.
    """
    try:
        yield
    except SpxtacularError:
        raise
    except Exception as error:
        if mzp is not None and isinstance(error, mzp.MzmlError) and not isinstance(error, KeyError):
            raise SpxtacularError(f"{path}: {error}") from error
        raise


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


# Ion-mobility unit accessions -> the IMType of the values, and the factor that
# converts them to that type's unit. The unit decides, not the array accession:
# PSI-MS allows several units on most ion-mobility array terms (MS:1002816 "mean
# ion mobility array" can hold 1/K0 or a drift time in ms or s).
_MZML_IM_UNITS: dict[str, tuple[IMType, float]] = {
    "MS:1002814": (IMType.OOK0, 1.0),  # volt-second per square centimeter
    "UO:0000028": (IMType.DRIFT_TIME_MS, 1.0),  # millisecond
    "UO:0000010": (IMType.DRIFT_TIME_MS, 1000.0),  # second
}


def _mzml_im_type(unit_accession: str | None) -> tuple[IMType, float]:
    """``(im_type, scale)`` for an ion-mobility value with this unit.

    A missing or unrecognised unit gives the generic :attr:`IMType.IM`, unscaled,
    rather than a guess.
    """
    if unit_accession is None:
        return IMType.IM, 1.0
    return _MZML_IM_UNITS.get(unit_accession, (IMType.IM, 1.0))


def _mzml_im_array(darr: Any, accession: object) -> tuple[np.ndarray, IMType]:
    """Decode an ion-mobility binary array, typed and scaled by its declared unit."""
    param = darr.get_cv_param(str(accession))
    im_type, scale = _mzml_im_type(param.unit_accession if param is not None else None)
    data = darr.data.astype(np.float64)
    if scale != 1.0:
        data = data * scale
    return data, im_type


def _mzml_scan_number(spec: MzmlSpectrum) -> int | None:
    """The instrument scan number from the native id, or ``None`` when it is not unique.

    See :func:`spxtacular._scan_lookup.native_id_scan_number` for which ids count.
    Nothing falls back to the 0-based list index; ``native_id`` always keeps the full id.
    """
    native_id = spec.id
    if not isinstance(native_id, str):
        return None
    return native_id_scan_number(native_id)


def _mzml_spectrum_ids(handle: Any) -> list[str]:
    """Every spectrum id of an open mzmlpy handle, in file order, from mzmlpy's id index (nothing is decoded)."""
    return list(handle.spectra.ids)


class MzmlReader:
    """Read spectra from an mzML or gzipped mzML file.

    The optional ``mzmlpy`` backend is loaded lazily. Iteration works without
    an explicit context manager. A context manager keeps one handle open and
    is more efficient for repeated access.

    Parameters
    ----------
    mzml_path:
        Path to the mzML file.
    gzip_mode:
        How mzmlpy opens gzipped input. ``"auto"`` reuses an embedded index or
        complete rapidgzip sidecars, and otherwise streams. ``"stream"`` starts
        immediately and is well suited to sequential reads. ``"indexed"``
        builds a random-access gzip index and requires rapidgzip.
    in_memory:
        Whether mzmlpy should keep its XML index in memory.
    """

    def __init__(
        self,
        mzml_path: str | Path,
        *,
        gzip_mode: Literal["auto", "indexed", "stream"] = "auto",
        in_memory: bool = False,
    ) -> None:
        if not _HAS_MZMLPY:
            raise ImportError(
                "MzmlReader requires the 'mzmlpy' package, which is not installed. "
                "Install it with: pip install spxtacular[mzml]"
            )
        self.mzml_path = mzml_path
        self.gzip_mode = gzip_mode
        self.in_memory = in_memory
        self._mzml_handle = None
        self._last_access_strategy: str | None = None
        self._index: IdIndex[str] | None = None

    def _new_handle(self) -> Any:
        """Create an mzmlpy handle with this reader's public I/O options."""
        assert mzp is not None
        handle = mzp.Mzml(
            self.mzml_path,
            gzip_mode=self.gzip_mode,
            in_memory=self.in_memory,
        )
        strategy = getattr(handle, "access_strategy", None)
        self._last_access_strategy = str(strategy) if strategy is not None else None
        return handle

    @property
    def access_strategy(self) -> str | None:
        """Concrete mzMLPy storage strategy selected by the latest open operation."""
        return self._last_access_strategy

    @staticmethod
    def _parse_spectrum(spec: MzmlSpectrum, decon: _DeconvolutionRefs = _NO_DECONVOLUTION) -> MsnSpectrum:
        """Convert a raw mzmlpy Spectrum into an MsnSpectrum.

        ``decon`` carries the file's deconvolution ``dataProcessing`` ids (see
        :func:`_deconvolution_refs`); the default treats the file as declaring
        none, so a spectrum is only deconvoluted if it says so itself.
        """
        mz_array = spec.mz
        if mz_array is None:
            raise SpxtacularError(f"Spectrum {spec} has no m/z array")
        mz_array = mz_array.astype(np.float64)

        int_array = spec.intensity
        if int_array is None:
            raise SpxtacularError(f"Spectrum {spec} has no intensity array")
        int_array = int_array.astype(np.float64)

        if len(mz_array) != len(int_array):
            raise SpxtacularError(f"Spectrum {spec} has m/z and intensity arrays of different lengths")

        charge_array = spec.charge_array
        if charge_array is not None:
            charge_array = charge_array.astype(np.int32)
            if len(charge_array) != len(mz_array):
                raise SpxtacularError(f"Spectrum {spec} has charge array of different length than m/z array")

        im_array: np.ndarray | None = None
        im_type: IMType | None = None
        im_types = sorted(spec.im_types)
        if len(im_types) == 1:
            darr = spec.get_binary_array(im_types[0])
            if darr is None:
                raise SpxtacularError(f"Spectrum {spec} has ion mobility array type {im_types[0]} but it is None")
            im_array, im_type = _mzml_im_array(darr, im_types[0])
            if len(im_array) != len(mz_array):
                raise SpxtacularError(f"Spectrum {spec} has ion mobility array of different length than m/z array")
        elif len(im_types) > 1:
            for candidate_type in im_types:
                darr = spec.get_binary_array(candidate_type)
                if darr is None:
                    raise SpxtacularError(
                        f"Spectrum {spec}: ion mobility array {candidate_type} is listed but missing. "
                        f"Array types: {im_types}"
                    )
                candidate, candidate_im_type = _mzml_im_array(darr, candidate_type)
                if len(candidate) == len(mz_array):
                    im_array = candidate
                    im_type = candidate_im_type
                    warnings.warn(
                        f"Spectrum {spec} has multiple ion mobility arrays {im_types}; using {candidate_type}, "
                        "the first whose length matches the m/z array",
                        stacklevel=3,
                    )
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
                raise SpxtacularError(f"Spectrum {spec} has unrecognized spectrum type: {spec.spectrum_type}")

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
            mz = ion.mz
            if mz is None:
                warnings.warn(
                    f"Spectrum {spec} precursor selected ion missing m/z. Precursor: {precursor}",
                    stacklevel=3,
                )
                continue
            # "peak intensity" (MS:1000042) is optional on a selectedIon. Keep the
            # precursor and read an absent value as 0.0, as the MGF reader does.
            intensity = ion.intensity if ion.intensity is not None else 0.0
            # mzmlpy keeps 1/K0 (MS:1002815) and drift time (MS:1002476) apart;
            # Precursor.im carries whichever the file has, tagged by im_type.
            prec_im = ion.ook0
            prec_im_type: IMType | None = IMType.OOK0 if prec_im is not None else None
            if prec_im is None and ion.drift_time is not None:
                # MS:1002476 "ion mobility drift time" is in ms or s; the unit decides.
                drift_param = ion.get_cv_param("MS:1002476")
                prec_im_type, scale = _mzml_im_type(drift_param.unit_accession if drift_param is not None else None)
                prec_im = ion.drift_time * scale
            precursors.append(
                Precursor(
                    precursor_mz=mz,
                    intensity=intensity,
                    charge=ion.charge,
                    im=prec_im,
                    im_type=prec_im_type,
                )
            )
            activation = precursor.activation
            if activation is not None:
                if activation.collision_energy is not None:
                    collision_energies.append(activation.collision_energy)
                if activation.activation_type is not None:
                    # mzmlpy yields the raw PSI-MS accession (as a vendor enum);
                    # normalise to spxtacular's canonical ActivationType member.
                    activation_types.append(ActivationType.from_accession(str(activation.activation_type)))
            if precursor.isolation_window is not None:
                isolation_range = precursor.isolation_window.isolation_mz_range
                if isolation_range is not None:
                    isolation_ranges.append(isolation_range)
        if len(set(collision_energies)) > 1:
            warnings.warn(f"Spectrum {spec} has multiple collision energies: {set(collision_energies)}", stacklevel=3)
        if len(set(activation_types)) > 1:
            warnings.warn(f"Spectrum {spec} has multiple activation types: {set(activation_types)}", stacklevel=3)
        if len(set(isolation_ranges)) > 1:
            warnings.warn(
                f"Spectrum {spec} has multiple isolation window ranges: {set(isolation_ranges)}", stacklevel=3
            )

        polarity = spec.polarity
        return MsnSpectrum(
            mz=mz_array,
            intensity=int_array,
            charge=charge_array,
            im=im_array,
            im_type=im_type,
            spectrum_type=spectrum_type,
            scan_number=_mzml_scan_number(spec),
            ms_level=spec.ms_level,
            native_id=spec.id,
            rt=spec.rt,
            injection_time=spec.ion_injection_time,
            total_ion_current=spec.total_ion_current,
            mz_range=spec.mz_range,
            polarity=Polarity(polarity) if polarity is not None else None,
            collision_energy=collision_energies[0] if collision_energies else None,
            activation_type=activation_types[0] if activation_types else None,
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

    # ------------------------------------------------------------------
    # Lookup by scan number / native id
    # ------------------------------------------------------------------

    def _id_index(self) -> IdIndex[str]:
        """Scan-number index of the file's native ids, built on first use and cached."""
        if self._index is None:
            with _mzml_errors(self.mzml_path):
                handle = self._mzml_handle
                if handle is not None:
                    ids = _mzml_spectrum_ids(handle)
                else:
                    with self._new_handle() as r:
                        ids = _mzml_spectrum_ids(r)
            self._index = build_id_index((native_id_scan_number(i), i, i) for i in ids)
        return self._index

    def get_by_native_id(self, native_id: str) -> MsnSpectrum:
        """Fetch a spectrum by its full native id (the mzML ``spectrum/@id``).

        Parameters
        ----------
        native_id:
            The exact id, e.g. ``"controllerType=0 controllerNumber=1 scan=19"``.

        Raises
        ------
        KeyError
            If no spectrum has this id.
        SpxtacularError
            If several spectra share it (invalid mzML).
        """
        if not isinstance(native_id, str):
            raise SpxtacularError(f"native_id must be a str, got {type(native_id).__name__} {native_id!r}")
        if native_id in self._id_index().duplicate_ids:
            raise SpxtacularError(f"{self.mzml_path}: native id {native_id!r} is used by more than one spectrum")
        return MzmlSpectraLookup(self)[native_id]

    def get_by_scan(self, scan_number: int, *, ms_level: int | None = None) -> MsnSpectrum:
        """Fetch a spectrum by its scan number (see ``MsnSpectrum.scan_number``).

        The scan number comes from the native id: ``scan=N`` (and Thermo
        ``controllerType=0 controllerNumber=1 scan=N``), ``index=N`` or
        ``spectrum=N``. It is not the 0-based position in the file; use
        ``reader[i]`` for that.

        Parameters
        ----------
        scan_number:
            The scan number.
        ms_level:
            If given, the spectrum must be of this MS level.

        Raises
        ------
        KeyError
            If no spectrum has this scan number, or it is not of ``ms_level``.
        SpxtacularError
            If the file's ids carry no scan numbers at all (Bruker ``frame=…``,
            Waters ``function=…``, SCIEX ``cycle=…``: use :meth:`get_by_native_id`),
            or several spectra share this scan number.
        """
        scan_number = check_scan_number(scan_number)
        check_ms_level(ms_level)
        index = self._id_index()
        native_id = index.scans.get(scan_number)
        if native_id is None:
            if scan_number in index.duplicate_scans:
                raise SpxtacularError(
                    f"{self.mzml_path}: scan number {scan_number} is shared by several spectra; "
                    "use get_by_native_id() instead"
                )
            if not index.scans and not index.duplicate_scans:
                raise SpxtacularError(
                    f"{self.mzml_path}: the native ids in this file carry no scan numbers "
                    "(e.g. Bruker 'frame=…', Waters 'function=…', SCIEX 'cycle=…'); use get_by_native_id() instead"
                )
            raise KeyError(f"no spectrum with scan number {scan_number} in {self.mzml_path}")
        spectrum = MzmlSpectraLookup(self)[native_id]
        if ms_level is not None and spectrum.ms_level != ms_level:
            raise KeyError(f"scan {scan_number} is MS{spectrum.ms_level}, not MS{ms_level}")
        return spectrum

    def get_by_sage_scannr(self, scannr: str | int) -> MsnSpectrum:
        """Fetch the spectrum a Sage ``scannr`` refers to.

        ``results.sage.tsv`` holds the mzML native id verbatim; Sage's ``.pin``
        output holds only the number from ``scan=N``. The value is tried as a
        native id and, if it is a bare integer, as a scan number too.

        Raises
        ------
        KeyError
            If nothing matches.
        SpxtacularError
            If a bare integer is one spectrum's native id and another's scan
            number, or as :meth:`get_by_native_id` and :meth:`get_by_scan`.
        """
        return by_sage_scannr(self, scannr)

    def open(self) -> None:
        """Open a persistent mzmlpy reader. Call :meth:`close` when done, or use as a context manager."""
        self._index = None
        if self._mzml_handle is not None:
            self.close()
        with _mzml_errors(self.mzml_path):
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


_READER_SUFFIXES = frozenset({".d", ".mzml", ".raw", ".mgf", ".ms2", ".msp"})


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
        Gzip strategy forwarded to :class:`MzmlReader`. ``"auto"`` is the
        default and selects the best valid random-access representation.
    mzml_in_memory:
        Whether mzmlpy should keep its XML index in memory.

    Raises
    ------
    SpxtacularError
        If the path extension is not recognised.
    FileNotFoundError
        If the path does not exist.

    Notes
    -----
    ``.mgf``, ``.ms2``, and ``.msp`` hold fragmentation spectra only, so
    ``.ms1`` on those inputs is a valid but always empty walk.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        centroid_config: CentroidConfig | None = None,
        mzml_gzip_mode: Literal["auto", "indexed", "stream"] = "auto",
        mzml_in_memory: bool = False,
    ) -> None:
        p = Path(path)
        suffixes = [s.lower() for s in p.suffixes]
        if suffixes and suffixes[-1] == ".gz":
            suffixes = suffixes[:-1]
        suffix = suffixes[-1] if suffixes else ""
        if suffix in _READER_SUFFIXES and not p.exists():
            raise FileNotFoundError(f"No such file or directory: {str(p)!r}")
        if suffix == ".d":
            self._reader: DReader | MzmlReader | ThermoReader | MgfReader | Ms2Reader | MspReader = DReader(
                p, centroid_config=centroid_config
            )
        elif suffix == ".mzml":
            self._reader = MzmlReader(
                p,
                gzip_mode=mzml_gzip_mode,
                in_memory=mzml_in_memory,
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
            raise SpxtacularError(
                f"Unsupported format {p.suffix!r}. Expected '.d', '.mzML', '.raw', '.mgf', '.ms2', or '.msp' "
                "(the text formats optionally gzipped)."
            )

    @property
    def ms1(self) -> SpectrumLookup:
        """MS1 spectra — supports iteration and index-based access."""
        return self._reader.ms1

    @property
    def ms2(self) -> SpectrumLookup:
        """MS2 spectra — supports iteration and index-based access."""
        return self._reader.ms2

    @property
    def access_strategy(self) -> str | None:
        """Concrete mzML access strategy, or ``None`` for non-mzML readers."""
        return self._reader.access_strategy if isinstance(self._reader, MzmlReader) else None

    def get_by_scan(self, scan_number: int, *, ms_level: int | None = None) -> MsnSpectrum:
        """Fetch a spectrum by scan number; see the format reader's ``get_by_scan`` for the rules.

        Raises
        ------
        KeyError
            If no spectrum has this scan number (at ``ms_level``, if given).
        SpxtacularError
            If the number does not identify one spectrum: the file carries no
            scan numbers (MSP, Bruker/Waters/SCIEX mzML), several spectra share
            it, or (Bruker) it is ambiguous without ``ms_level``.
        """
        return self._reader.get_by_scan(scan_number, ms_level=ms_level)

    def get_by_native_id(self, native_id: str) -> MsnSpectrum:
        """Fetch a spectrum by its ``native_id``; see the format reader's ``get_by_native_id``.

        Raises
        ------
        KeyError
            If no spectrum has this native id.
        SpxtacularError
            If several spectra share it.
        """
        return self._reader.get_by_native_id(native_id)

    def get_by_sage_scannr(self, scannr: str | int, *, precursor_offset: int | None = None) -> MsnSpectrum:
        """Fetch the spectrum a Sage ``scannr`` refers to.

        Parameters
        ----------
        scannr:
            The ``scannr`` column of ``results.sage.tsv`` or a ``.pin`` file.
        precursor_offset:
            Bruker ``.d`` only; see :meth:`DReader.get_by_sage_scannr` (default 1).

        Raises
        ------
        KeyError
            If nothing matches.
        SpxtacularError
            If ``precursor_offset`` is given for a format other than Bruker ``.d``,
            or as the format reader's method.
        """
        if isinstance(self._reader, DReader):
            if precursor_offset is None:
                return self._reader.get_by_sage_scannr(scannr)
            return self._reader.get_by_sage_scannr(scannr, precursor_offset=precursor_offset)
        if precursor_offset is not None:
            raise SpxtacularError("precursor_offset applies to Bruker .d runs only")
        return self._reader.get_by_sage_scannr(scannr)

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
