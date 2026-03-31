import warnings
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import mzmlpy as mzp
import numpy as np
import tdfpy

from .core import MsnSpectrum, Precursor, SpectrumType

"""

Unified reader API for different mass-spectrometry file formats.
Supports DDA and DIA data from Bruker timsTOF (.d) and mzML.
"""


class AcquisitionType(StrEnum):
    DDA = "DDA"
    DIA = "DIA"
    PRM = "PRM"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# DReader lookup objects
# ---------------------------------------------------------------------------


class DReaderMs1Lookup:
    """Iterable + index-accessible MS1 spectra from a DReader.

    Iteration yields all MS1 spectra. Index access (``lookup[frame_id]``)
    fetches a single spectrum by tdfpy ``frame_id``.
    """

    def __init__(self, dreader: "DReader") -> None:
        self._dr = dreader

    def _require_open(self) -> None:
        if self._dr._reader is None:
            raise RuntimeError("DReader must be opened before use (call open() or use as a context manager)")

    def __iter__(self) -> Iterator[MsnSpectrum]:
        self._require_open()
        reader = self._dr._reader
        assert reader is not None
        mz_range = reader.metadata.mz_acq_range
        im_range = reader.metadata.one_over_k0_acq_range
        for frame in reader.ms1:
            yield DReader._parse_ms1_frame(frame, mz_range, im_range)

    def __getitem__(self, frame_id: int) -> MsnSpectrum:
        """Fetch a single MS1 spectrum by tdfpy frame_id."""
        self._require_open()
        reader = self._dr._reader
        assert reader is not None
        mz_range = reader.metadata.mz_acq_range
        im_range = reader.metadata.one_over_k0_acq_range
        frame = reader.ms1[frame_id]  # raises KeyError if not found
        return DReader._parse_ms1_frame(frame, mz_range, im_range)


class DReaderMs2Lookup:
    """Iterable + index-accessible MS2 spectra from a DReader.

    Iteration yields all MS2 spectra. Index access (``lookup[precursor_id]``)
    fetches a single spectrum by tdfpy ``precursor_id`` (DDA only).
    """

    def __init__(self, dreader: "DReader") -> None:
        self._dr = dreader

    def _require_open(self) -> None:
        if self._dr._reader is None:
            raise RuntimeError("DReader must be opened before use (call open() or use as a context manager)")

    def __iter__(self) -> Iterator[MsnSpectrum]:
        self._require_open()
        reader = self._dr._reader
        assert reader is not None
        match self._dr.acquisition_type:
            case AcquisitionType.DDA:
                for precursor in reader.precursors:  # type: ignore
                    yield DReader._parse_dda_precursor(precursor)
            case AcquisitionType.DIA:
                for window in reader.windows:  # type: ignore
                    yield DReader._parse_dia_window(window)
            case _:
                raise ValueError(f"Unsupported acquisition type: {self._dr.acquisition_type}")

    def __getitem__(self, precursor_id: int) -> MsnSpectrum:
        """Fetch a single MS2 spectrum by tdfpy precursor_id (DDA only)."""
        self._require_open()
        reader = self._dr._reader
        assert reader is not None
        match self._dr.acquisition_type:
            case AcquisitionType.DDA:
                precursor = reader.precursors[precursor_id]  # type: ignore  # KeyError if not found
                return DReader._parse_dda_precursor(precursor)
            case AcquisitionType.DIA:
                raise NotImplementedError(
                    "DIA MS2 lookup by ID is not supported: DIA windows map to multiple frames. "
                    "Iterate reader.ms2 instead."
                )
            case _:
                raise ValueError(f"Unsupported acquisition type: {self._dr.acquisition_type}")


# ---------------------------------------------------------------------------
# DReader
# ---------------------------------------------------------------------------


class DReader:
    def __init__(self, analysis_dir: str | Path) -> None:
        import tdfpy as tdf

        self.analysis_dir = analysis_dir
        self._tdf = tdf
        _aqui = tdf.get_acquisition_type(str(analysis_dir))
        self.acquisition_type: AcquisitionType
        match _aqui:
            case "DDA":
                self.acquisition_type = AcquisitionType.DDA
            case "DIA":
                self.acquisition_type = AcquisitionType.DIA
            case "PRM":
                self.acquisition_type = AcquisitionType.PRM
            case _:
                self.acquisition_type = AcquisitionType.UNKNOWN
        self._reader = None

    def open(self) -> None:
        """Open the underlying tdfpy reader. Call :meth:`close` when done, or use as a context manager."""
        match self.acquisition_type:
            case AcquisitionType.DDA | AcquisitionType.PRM | AcquisitionType.UNKNOWN:
                self._reader = self._tdf.DDA(str(self.analysis_dir))
            case AcquisitionType.DIA:
                self._reader = self._tdf.DIA(str(self.analysis_dir))
            case _:
                raise ValueError(f"Unsupported acquisition type: {self.acquisition_type}")
        self._reader.__enter__()

    def close(self) -> None:
        """Close the underlying tdfpy reader."""
        if self._reader:
            self._reader.__exit__(None, None, None)

    def __enter__(self) -> "DReader":
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

    @staticmethod
    def _parse_ms1_frame(
        frame: Any,
        mz_range: tuple[float, float] | None,
        im_range: tuple[float, float] | None,
    ) -> MsnSpectrum:
        centroided_peaks = frame.centroid()
        match frame.polarity:
            case "positive":
                polarity = "positive"
            case "negative":
                polarity = "negative"
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
            analyzer="TOF",
            ramp_time=frame.ramp_time,
            precursors=None,
            im_type="ook0",
        )

    @staticmethod
    def _parse_dda_precursor(precursor: tdfpy.Precursor) -> MsnSpectrum:
        peaks = precursor.peaks
        match precursor.polarity:
            case "positive":
                polarity = "positive"
            case "negative":
                polarity = "negative"
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
            total_ion_current=None,  # TODO:
            mz_range=None,
            im_range=None,
            polarity=polarity,
            resolution=None,
            analyzer="TOF",
            ramp_time=None,
            precursors=[prec],
            im_type="ook0",
            isolation_im_range=precursor.ook0_range,
            isolation_mz_range=precursor.mz_range,
            collision_energy=precursor.collision_energy,
            activation_type="MS:1002481",
        )

    @staticmethod
    def _parse_dia_window(window: tdfpy.DiaWindow) -> MsnSpectrum:
        peaks = window.centroid()
        match window.polarity:
            case "positive":
                polarity = "positive"
            case "negative":
                polarity = "negative"
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
            analyzer="TOF",
            collision_energy=window.collision_energy,
            activation_type="MS:1002481",
            ramp_time=None,
            precursors=None,
            isolation_mz_range=window.mz_range,
            isolation_im_range=window.ook0_range,
            im_type="ook0",
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
        """MS2 spectra — supports iteration and precursor_id-based access (DDA only)."""
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

    def __init__(self, reader: "MzmlReader", ms_level: int | None = None) -> None:
        self._reader = reader
        self._ms_level = ms_level

    def __iter__(self) -> Iterator[MsnSpectrum]:
        handle = self._reader._mzml_handle
        if handle is not None:
            for spec in handle.spectra:
                if self._ms_level is not None and spec.ms_level != self._ms_level:
                    continue
                yield MzmlReader._parse_spectrum(spec)
        else:
            with mzp.Mzml(self._reader.mzml_path) as r:
                for spec in r.spectra:
                    if self._ms_level is not None and spec.ms_level != self._ms_level:
                        continue
                    yield MzmlReader._parse_spectrum(spec)

    def __getitem__(self, key: int | str) -> MsnSpectrum:
        """Fetch a single spectrum by 0-based index or native ID string."""
        handle = self._reader._mzml_handle
        if handle is not None:
            spec = handle.spectra[key]
        else:
            with mzp.Mzml(self._reader.mzml_path) as r:
                spec = r.spectra[key]
        return MzmlReader._parse_spectrum(spec)


# ---------------------------------------------------------------------------
# MzmlReader
# ---------------------------------------------------------------------------


class MzmlReader:
    def __init__(self, mzml_path: str | Path):
        self.mzml_path = mzml_path
        self._mzml_handle = None

    @staticmethod
    def _parse_spectrum(spec: mzp.Spectrum) -> MsnSpectrum:
        """Convert a raw mzmlpy Spectrum into an MsnSpectrum."""
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
                im_array = darr.data.astype(np.float64)
                if len(im_array) != len(mz_array):
                    im_array = None
                    continue
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

        if charge_array is not None:
            spectrum_type = SpectrumType.DECONVOLUTED

        mz_range = None
        if spec.lower_mz is not None and spec.upper_mz is not None:
            mz_range = (spec.lower_mz, spec.upper_mz)

        precursors: list[Precursor] = []
        collision_energies: list[float] = []
        activation_types: list[str] = []
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
                    activation_types.append(activation.activation_type)
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
            polarity=spec.polarity,
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
        self._mzml_handle = mzp.Mzml(self.mzml_path)
        self._mzml_handle.__enter__()

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
    """Format-agnostic reader — detects .d (Bruker timsTOF) or .mzML from the path.

    Usage is identical regardless of the underlying format::

        with Reader("data.mzML") as r:
            for spec in r.ms1:
                ...

        with Reader("data.d") as r:
            ms2 = r.ms2[42]

    Parameters
    ----------
    path:
        Path to a Bruker ``.d`` directory or an ``.mzML`` file.

    Raises
    ------
    ValueError
        If the path extension is not recognised.
    """

    def __init__(self, path: str | Path) -> None:
        p = Path(path)
        if p.suffix == ".d":
            self._reader: DReader | MzmlReader = DReader(p)
        elif p.suffix.lower() == ".mzml":
            self._reader = MzmlReader(p)
        else:
            raise ValueError(f"Unsupported format {p.suffix!r}. Expected '.d' or '.mzml'.")

    @property
    def ms1(self) -> DReaderMs1Lookup | MzmlSpectraLookup:
        """MS1 spectra — supports iteration and index-based access."""
        return self._reader.ms1

    @property
    def ms2(self) -> DReaderMs2Lookup | MzmlSpectraLookup:
        """MS2 spectra — supports iteration and index-based access."""
        return self._reader.ms2

    def open(self) -> None:
        """Open the underlying reader."""
        self._reader.open()

    def close(self) -> None:
        """Close the underlying reader."""
        self._reader.close()

    def __enter__(self) -> "Reader":
        self._reader.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._reader.close()
