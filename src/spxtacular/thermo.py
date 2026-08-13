"""Thermo ``.raw`` reading via `fisher-py <https://github.com/ethz-institute-of-microbiology/fisher_py>`_.

``fisher-py`` wraps Thermo's official RawFileReader .NET assemblies through
``pythonnet``, so reading ``.raw`` files needs a .NET runtime on the machine
(https://dotnet.microsoft.com/download). Unlike the other optional backends,
importing ``fisher_py`` *boots that runtime* and raises ``RuntimeError`` — not
``ImportError`` — when none is found. It is therefore imported lazily inside
:class:`ThermoReader` rather than at module import time, so ``import
spxtacular`` never pays for (or crashes on) the runtime.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import Any, NamedTuple

import numpy as np

from .core import MsnSpectrum, Precursor, SpectrumType
from .enums import ActivationType, Analyzer, AnalyzerLike, Polarity

# ---------------------------------------------------------------------------
# Lazy fisher-py loading
# ---------------------------------------------------------------------------


class _FisherModules(NamedTuple):
    raw_file_reader_adapter: Any
    device: Any


_fisher_modules: _FisherModules | None = None


def _require_fisher() -> _FisherModules:
    """Import fisher-py on first use, translating its failure modes to ImportError."""
    global _fisher_modules
    if _fisher_modules is None:
        try:
            from fisher_py.data import Device
            from fisher_py.raw_file_reader import RawFileReaderAdapter
        except ImportError as exc:
            raise ImportError(
                "ThermoReader requires the 'fisher-py' package, which is not installed. "
                "Install it with: pip install spxtacular[thermo]"
            ) from exc
        except (OSError, RuntimeError) as exc:
            raise ImportError(
                "fisher-py is installed but could not start its .NET runtime. Thermo .raw "
                "reading uses Thermo's RawFileReader, a .NET library: install the .NET 8 "
                "runtime (https://dotnet.microsoft.com/download) and make sure `dotnet` is "
                f"on PATH or DOTNET_ROOT points at it. Original error: {exc}"
            ) from exc
        _fisher_modules = _FisherModules(RawFileReaderAdapter, Device)
    return _fisher_modules


# ---------------------------------------------------------------------------
# Vendor-enum mapping tables (keyed by enum *name* so no fisher-py import is
# needed at module level)
# ---------------------------------------------------------------------------

# fisher_py.data.filter_enums.ActivationType member name → spxtacular member.
# Unlisted names (ProtonTransferReaction, ModeA…) pass through as raw strings —
# activation_type is an open vocabulary.
_ACTIVATION_BY_NAME: dict[str, ActivationType] = {
    "CollisionInducedDissociation": ActivationType.CID,
    "HigherEnergyCollisionalDissociation": ActivationType.HCD,
    "ElectronTransferDissociation": ActivationType.ETD,
    "ElectronCaptureDissociation": ActivationType.ECD,
    "NegativeElectronTransferDissociation": ActivationType.NETD,
    "UltraVioletPhotoDissociation": ActivationType.UVPD,
    "MultiPhotonDissociation": ActivationType.PD,
    "PQD": ActivationType.PQD,
}

# ETD followed by a supplemental-activation reaction is EThcD / ETciD.
_COMBINED_ACTIVATION: dict[frozenset[ActivationType], ActivationType] = {
    frozenset({ActivationType.ETD, ActivationType.HCD}): ActivationType.ETHCD,
    frozenset({ActivationType.ETD, ActivationType.CID}): ActivationType.ETCID,
}

# fisher_py MassAnalyzerType member name → spxtacular member. FTMS means
# "Fourier transform" and is resolved against the instrument model in
# _map_analyzer (Orbitrap for everything except the LTQ FT family, which is an
# FT-ICR). ITMS does not say linear vs 3D trap, so the generic member is used.
_ANALYZER_BY_NAME: dict[str, Analyzer] = {
    "MassAnalyzerITMS": Analyzer.ION_TRAP,
    "MassAnalyzerTQMS": Analyzer.QUADRUPOLE,
    "MassAnalyzerSQMS": Analyzer.QUADRUPOLE,
    "MassAnalyzerTOFMS": Analyzer.TOF,
    "MassAnalyzerSector": Analyzer.MAGNETIC_SECTOR,
}


def _map_analyzer(analyzer_name: str, instrument_model: str) -> AnalyzerLike | None:
    if analyzer_name == "MassAnalyzerFTMS":
        return Analyzer.FT_ICR if "LTQ FT" in instrument_model.upper() else Analyzer.ORBITRAP
    if analyzer_name == "Any":
        return None
    return _ANALYZER_BY_NAME.get(analyzer_name, analyzer_name)


# ---------------------------------------------------------------------------
# Trailer-extra helpers
# ---------------------------------------------------------------------------


def _trailer_dict(raw: Any, scan_number: int) -> dict[str, str]:
    """Per-scan 'trailer extra' records as a {label: value} dict (both stripped)."""
    trailer = raw.get_trailer_extra_information(scan_number)
    return {
        str(label).rstrip(":").strip(): str(value).strip()
        for label, value in zip(trailer.labels, trailer.values, strict=True)
    }


def _trailer_float(trailer: dict[str, str], *labels: str) -> float | None:
    """First parseable, strictly-positive float among ``labels`` (Thermo writes
    -1 / 0 for "not set")."""
    for label in labels:
        value = trailer.get(label)
        if not value:
            continue
        try:
            parsed = float(value)
        except ValueError:
            continue
        if parsed > 0:
            return parsed
    return None


# ---------------------------------------------------------------------------
# Lookup object
# ---------------------------------------------------------------------------


class ThermoScanLookup:
    """Iterable + index-accessible spectra from a Thermo .raw file.

    Iteration yields spectra filtered to ``ms_level`` (if given), in scan
    order. Index access (``lookup[scan_number]``) fetches a single spectrum by
    its native 1-based scan number and raises ``KeyError`` when the scan does
    not exist or is not of the lookup's MS level.
    """

    def __init__(self, reader: ThermoReader, ms_level: int | None = None) -> None:
        self._reader = reader
        self._ms_level = ms_level

    def __iter__(self) -> Iterator[MsnSpectrum]:
        raw = self._reader._open_raw()
        header = raw.run_header_ex
        for scan_number in range(header.first_spectrum, header.last_spectrum + 1):
            if self._ms_level is not None:
                order = int(raw.get_filter_for_scan_number(scan_number).ms_order.value)
                if order != self._ms_level:
                    continue
            yield self._reader._parse_scan(scan_number)

    def __getitem__(self, scan_number: int) -> MsnSpectrum:
        """Fetch a single spectrum by native (1-based) scan number."""
        raw = self._reader._open_raw()
        header = raw.run_header_ex
        if not header.first_spectrum <= scan_number <= header.last_spectrum:
            raise KeyError(f"scan {scan_number} not in file (scans {header.first_spectrum}-{header.last_spectrum})")
        if self._ms_level is not None:
            order = int(raw.get_filter_for_scan_number(scan_number).ms_order.value)
            if order != self._ms_level:
                raise KeyError(f"scan {scan_number} is MS{order}, not MS{self._ms_level}")
        return self._reader._parse_scan(scan_number)


# ---------------------------------------------------------------------------
# ThermoReader
# ---------------------------------------------------------------------------


class ThermoReader:
    """Reads Thermo ``.raw`` files via fisher-py / Thermo RawFileReader.

    Must be opened before use — either with :meth:`open` / :meth:`close` or
    (preferred) as a context manager.

    Parameters
    ----------
    raw_path:
        Path to a Thermo ``.raw`` file.
    prefer_vendor_centroid:
        For profile-mode scans that carry Thermo's own centroid ("label")
        stream — FTMS scans do — yield those centroids (``CENTROID``, with the
        vendor's per-peak charge annotations) instead of the profile trace.
        Pass ``False`` to always get the data as acquired: ``PROFILE`` for
        profile-mode scans. Scans acquired in centroid mode (typical for ion
        trap detectors) are unaffected. Default ``True``.
    """

    def __init__(self, raw_path: str | Path, prefer_vendor_centroid: bool = True) -> None:
        _require_fisher()
        path = Path(raw_path)
        if path.is_dir():
            raise ValueError(
                f"{path} is a directory. Thermo .raw is a single file; a .raw *directory* is "
                "the Waters format, which spxtacular does not support — convert it to mzML "
                "(e.g. with msconvert) and use MzmlReader instead."
            )
        if not path.exists():
            raise FileNotFoundError(f"Thermo .raw file not found: {path}")
        self.raw_path = path
        self.prefer_vendor_centroid = prefer_vendor_centroid
        self._raw: Any = None
        self._instrument_model: str = ""

    def open(self) -> None:
        """Open the underlying RawFileReader handle. Call :meth:`close` when done, or use as a context manager."""
        if self._raw is not None:
            self.close()
        fisher = _require_fisher()
        raw = fisher.raw_file_reader_adapter.file_factory(str(self.raw_path))
        try:
            raw.select_instrument(fisher.device.MS, 1)
            self._instrument_model = str(raw.get_instrument_data().model)
        except BaseException:
            raw.dispose()
            raise
        self._raw = raw

    def close(self) -> None:
        """Release the underlying RawFileReader handle."""
        if self._raw is not None:
            self._raw.dispose()
            self._raw = None

    def __enter__(self) -> ThermoReader:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def _open_raw(self) -> Any:
        if self._raw is None:
            raise RuntimeError("ThermoReader must be opened before use (call open() or use as a context manager)")
        return self._raw

    # ------------------------------------------------------------------
    # Scan conversion
    # ------------------------------------------------------------------

    def _read_peaks(
        self, raw: Any, scan_number: int, stats: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, SpectrumType]:
        """Peak arrays + spectrum type for one scan.

        Prefers the vendor centroid ("label") stream when configured to; falls
        back to the segmented scan, which is the profile trace for profile-mode
        scans and the peak list itself for centroid-mode scans.
        """
        if self.prefer_vendor_centroid and not stats.is_centroid_scan:
            centroid_stream = raw.get_centroid_stream(scan_number, False)
            masses = centroid_stream.masses
            if masses is not None and len(masses) > 0:
                mz = np.asarray(masses, dtype=np.float64)
                intensity = np.asarray(centroid_stream.intensities, dtype=np.float64)
                # Thermo's label stream uses 0 for "charge unknown"; spxtacular
                # uses -1 for unassigned (0 is reserved for decharged spectra).
                charge = np.asarray(centroid_stream.charges, dtype=np.int32)
                charge[charge == 0] = -1
                return mz, intensity, charge, SpectrumType.CENTROID

        segmented = raw.get_segmented_scan_from_scan_number(scan_number, stats)
        mz = np.asarray(segmented.positions, dtype=np.float64)
        intensity = np.asarray(segmented.intensities, dtype=np.float64)
        spectrum_type = SpectrumType.CENTROID if stats.is_centroid_scan else SpectrumType.PROFILE
        return mz, intensity, None, spectrum_type

    @staticmethod
    def _activation(scan_event: Any, ms_level: int) -> tuple[ActivationType | str | None, float | None]:
        """Activation type + collision energy of the current stage's reaction.

        The reaction at index ``ms_level - 2`` is the one that produced this
        scan; a following reaction flagged ``multiple_activation`` is the
        supplemental activation of a combined scheme (EThcD / ETciD).
        """
        reaction = scan_event.get_reaction(ms_level - 2)
        primary = _ACTIVATION_BY_NAME.get(reaction.activation_type.name, reaction.activation_type.name)
        collision_energy = float(reaction.collision_energy) if reaction.collision_energy_valid else None

        activation: ActivationType | str | None = primary
        if isinstance(primary, ActivationType):
            try:
                supplemental_reaction = scan_event.get_reaction(ms_level - 1)
            except Exception:
                supplemental_reaction = None
            if supplemental_reaction is not None and supplemental_reaction.multiple_activation:
                supplemental = _ACTIVATION_BY_NAME.get(supplemental_reaction.activation_type.name)
                if supplemental is not None:
                    activation = _COMBINED_ACTIVATION.get(frozenset({primary, supplemental}), primary)
        return activation, collision_energy

    def _parse_scan(self, scan_number: int) -> MsnSpectrum:
        raw = self._open_raw()
        stats = raw.get_scan_stats_for_scan_number(scan_number)
        scan_filter = raw.get_filter_for_scan_number(scan_number)
        trailer = _trailer_dict(raw, scan_number)

        order = int(scan_filter.ms_order.value)
        ms_level = order if order >= 1 else None

        mz, intensity, charge, spectrum_type = self._read_peaks(raw, scan_number, stats)

        match scan_filter.polarity.name:
            case "Positive":
                polarity = Polarity.POSITIVE
            case "Negative":
                polarity = Polarity.NEGATIVE
            case _:
                polarity = None

        precursors: list[Precursor] | None = None
        collision_energy: float | None = None
        activation: ActivationType | str | None = None
        isolation_mz_range: tuple[float, float] | None = None
        if ms_level is not None and ms_level >= 2:
            # A malformed scan event should degrade to "no precursor metadata",
            # not lose the peaks; RawFileReader errors surface as .NET
            # exceptions of no more specific Python type.
            try:
                scan_event = raw.get_scan_event_for_scan_number(scan_number)
                reaction = scan_event.get_reaction(ms_level - 2)
                activation, collision_energy = self._activation(scan_event, ms_level)

                monoisotopic_mz = _trailer_float(trailer, "Monoisotopic M/Z")
                target_mz = monoisotopic_mz if monoisotopic_mz is not None else float(reaction.precursor_mass)
                charge_state = _trailer_float(trailer, "Charge State")
                width = float(reaction.isolation_width)
                if width > 0:
                    center = float(reaction.precursor_mass) + float(reaction.isolation_width_offset)
                    isolation_mz_range = (center - width / 2, center + width / 2)
                # The scan itself records no precursor intensity (converters
                # look it up in the parent MS1); like the PRM path in DReader,
                # the summed product-ion intensity stands in as a proxy.
                precursors = [
                    Precursor(
                        mz=target_mz,
                        intensity=float(intensity.sum()) if len(intensity) else 0.0,
                        charge=int(charge_state) if charge_state is not None else None,
                        im=None,
                        is_monoisotopic=monoisotopic_mz is not None,
                    )
                ]
            except Exception:
                precursors = None

        return MsnSpectrum(
            mz=mz,
            intensity=intensity,
            charge=charge,
            im=None,
            spectrum_type=spectrum_type,
            denoised=None,
            normalized=None,
            scan_number=scan_number,
            ms_level=ms_level,
            native_id=f"controllerType=0 controllerNumber=1 scan={scan_number}",
            rt=float(stats.start_time) * 60.0,  # RawFileReader reports minutes
            injection_time=_trailer_float(trailer, "Ion Injection Time (ms)"),
            total_ion_current=float(stats.tic),
            mz_range=(float(stats.low_mass), float(stats.high_mass)),
            im_range=None,
            polarity=polarity,
            resolution=_trailer_float(trailer, "Orbitrap Resolution", "FT Resolution"),
            analyzer=_map_analyzer(scan_filter.mass_analyzer.name, self._instrument_model),
            collision_energy=collision_energy,
            activation_type=activation,
            ramp_time=None,
            precursors=precursors,
            isolation_mz_range=isolation_mz_range,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def ms1(self) -> ThermoScanLookup:
        """MS1 spectra — supports iteration and scan-number access."""
        return ThermoScanLookup(self, ms_level=1)

    @property
    def ms2(self) -> ThermoScanLookup:
        """MS2 spectra — supports iteration and scan-number access."""
        return ThermoScanLookup(self, ms_level=2)

    def __getitem__(self, scan_number: int) -> MsnSpectrum:
        """Fetch a single spectrum of any MS level by native (1-based) scan number."""
        return ThermoScanLookup(self)[scan_number]
