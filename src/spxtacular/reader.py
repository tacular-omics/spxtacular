import warnings
from collections.abc import Generator
from enum import StrEnum
from typing import Self, cast

import mzmlpy as mzp
import numpy as np

from .core import MsnSpectrum, SpectrumType, TargetIon

"""

Unified reader API for different mass-spectrometry file formats.
Supports DDA and DIA data from Bruker timsTOF (.d) and mzML.
"""


class AcquisitionType(StrEnum):
    DDA = "DDA"
    DIA = "DIA"
    PRM = "PRM"
    UNKNOWN = "UNKNOWN"


class DReader:
    def __init__(self, analysis_dir: str):
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

    def __enter__(self):
        match self.acquisition_type:
            case AcquisitionType.DDA | AcquisitionType.PRM | AcquisitionType.UNKNOWN:
                self._reader = self._tdf.DDA(self.analysis_dir)
            case AcquisitionType.DIA:
                self._reader = self._tdf.DIA(self.analysis_dir)
            case _:
                raise ValueError(
                    f"Unsupported acquisition type: {self.acquisition_type}"
                )

        self._reader.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._reader:
            self._reader.__exit__(exc_type, exc_val, exc_tb)

    @property
    def ms1(self) -> Generator[MsnSpectrum, None, None]:
        if self._reader is None:
            raise RuntimeError("DReader must be used as a context manager")

        match self.acquisition_type:
            case (
                AcquisitionType.DDA
                | AcquisitionType.DIA
                | AcquisitionType.PRM
                | AcquisitionType.UNKNOWN
            ):
                reader = self._reader
                mz_range = reader.metadata.mz_acq_range
                im_range = reader.metadata.one_over_k0_acq_range
                for ms1_spec in reader.ms1:
                    centroided_peaks = ms1_spec.centroid()

                    match ms1_spec.polarity:
                        case "positive":
                            polarity = "positive"
                        case "negative":
                            polarity = "negative"
                        case _:
                            polarity = None

                    yield MsnSpectrum(
                        mz=centroided_peaks[:, 0],
                        intensity=centroided_peaks[:, 1],
                        charge=None,
                        im=centroided_peaks[:, 2],
                        spectrum_type=SpectrumType.CENTROID,
                        denoised=None,
                        normalized=None,
                        scan_number=ms1_spec.frame_id,
                        ms_level=1,
                        native_id=None,
                        rt=ms1_spec.time,
                        injection_time=ms1_spec.accumulation_time,
                        mz_range=mz_range,
                        im_range=im_range,
                        polarity=polarity,
                        resolution=None,
                        analyzer="TOF",
                        collision_energy=None,
                        activation_type=None,
                        ramp_time=ms1_spec.ramp_time,
                        precursors=None,
                    )
            case _:
                raise ValueError(
                    f"Unsupported acquisition type: {self.acquisition_type}"
                )

    @property
    def ms2(self) -> Generator[MsnSpectrum, None, None]:
        """Yields MS2 spectra."""
        if self._reader is None:
            raise RuntimeError("DReader must be used as a context manager")

        match self.acquisition_type:
            case AcquisitionType.DDA:
                reader = cast(self._tdf.DDA, self._reader)
                for ms2_spec in reader.precursors:
                    peaks = ms2_spec.peaks

                    # polarity
                    match ms2_spec.polarity:
                        case "positive":
                            polarity = "positive"
                        case "negative":
                            polarity = "negative"
                        case _:
                            polarity = None

                    target_mz = ms2_spec.monoisotopic_mz
                    is_monoisotopic = True
                    if target_mz is None:
                        target_mz = ms2_spec.largest_peak_mz
                        is_monoisotopic = False

                    prec = TargetIon(
                        mz=target_mz,
                        intensity=ms2_spec.intensity,
                        charge=ms2_spec.charge,
                        im=ms2_spec.ook0,
                        is_monoisotopic=is_monoisotopic,
                    )

                    yield MsnSpectrum(
                        mz=peaks[:, 0],
                        intensity=peaks[:, 1],
                        charge=None,
                        im=None,
                        spectrum_type=SpectrumType.CENTROID,  # TimsData spectra are already centroided
                        denoised=None,
                        normalized=None,
                        scan_number=ms2_spec.precursor_id,
                        ms_level=2,
                        native_id=None,
                        rt=ms2_spec.rt,
                        injection_time=None,
                        mz_range=ms2_spec.mz_range,
                        im_range=ms2_spec.ook0_range,
                        polarity=polarity,
                        resolution=None,
                        analyzer="TOF",
                        collision_energy=ms2_spec.collision_energy,
                        activation_type="MS:1002481",
                        ramp_time=None,
                        precursors=[prec],
                    )

            case AcquisitionType.DIA:
                reader = cast(self._tdf.DIA, self._reader)
                for ms2_spec in reader.windows:
                    peaks = ms2_spec.centroid()

                    # polarity
                    match ms2_spec.polarity:
                        case "positive":
                            polarity = "positive"
                        case "negative":
                            polarity = "negative"
                        case _:
                            polarity = None

                    native_id = f"{ms2_spec.frame_id}@w{ms2_spec.window_index}"

                    yield MsnSpectrum(
                        mz=peaks[:, 0],
                        intensity=peaks[:, 1],
                        charge=None,
                        im=peaks[:, 2],
                        spectrum_type=SpectrumType.CENTROID,  # TimsData spectra are already centroided
                        denoised=None,
                        normalized=None,
                        scan_number=ms2_spec.frame_id,
                        ms_level=2,
                        native_id=native_id,
                        rt=ms2_spec.rt,
                        injection_time=None,
                        mz_range=ms2_spec.mz_range,
                        im_range=None,  # DIA spectra don't have ion mobility info
                        polarity=polarity,
                        resolution=None,
                        analyzer="TOF",
                        collision_energy=ms2_spec.collision_energy,  # DIA spectra don't have collision energy info
                        activation_type="MS:1002481",
                        ramp_time=None,
                        precursors=None,  # DIA spectra don't have defined precursors
                    )
            case _:
                raise ValueError(
                    f"Unsupported acquisition type: {self.acquisition_type}"
                )


class MzmlReader:
    def __init__(self, mzml_path: str):
        self.mzml_path = mzml_path

    @property
    def ms1(self) -> Generator[MsnSpectrum, None, None]:
        """Yields MS1 spectra."""
        with mzp.Mzml(self.mzml_path) as reader:
            for spec in reader.spectra:
                if spec.ms_level != 1:
                    continue

                mz_array = spec.mz
                if mz_array is None:
                    raise ValueError(f"Spectrum {spec} has no m/z array")
                mz_array = mz_array.astype(np.float64)

                int_array = spec.intensity
                if int_array is None:
                    raise ValueError(f"Spectrum {spec} has no intensity array")
                int_array = int_array.astype(np.float64)

                if len(mz_array) != len(int_array):
                    raise ValueError(
                        f"Spectrum {spec} has m/z and intensity arrays of different lengths"
                    )

                charge_array = spec.charge
                if charge_array is not None:
                    charge_array = charge_array.astype(np.int32)
                    if len(charge_array) != len(mz_array):
                        raise ValueError(
                            f"Spectrum {spec} has charge array of different length than m/z array"
                        )

                im_types = list(spec.im_types)
                im_array: np.ndarray | None = None
                if len(im_types) == 1:
                    darr = spec.get_binary_array(im_types[0])
                    if darr is None:
                        raise RuntimeError(
                            f"Spectrum {spec} has ion mobility array type {im_types[0]} but it is None"
                        )
                    im_array = darr.data.astype(np.float64)
                    if len(im_array) != len(mz_array):
                        raise ValueError(
                            f"Spectrum {spec} has ion mobility array of different length than m/z array"
                        )
                elif len(im_types) > 1:
                    warnings.warn(
                        f"Spectrum {spec} has multiple ion mobility arrays; only the first is used: {im_types[0]}",
                        stacklevel=2,
                    )
                    for im_type in im_types:
                        darr = spec.get_binary_array(im_type)
                        if darr is None:
                            raise RuntimeError(
                                f"Spectrum {spec}: multiple IM arrays, first is not None. "
                                f"Array types: {im_types}"
                            )
                        im_array = darr.data.astype(np.float64)
                        if len(im_array) != len(mz_array):
                            im_array = None
                            continue
                    if im_array is None:
                        warnings.warn(
                            f"Spectrum {spec}: no ion mobility array length matches m/z array. Array types: {im_types}",
                            stacklevel=2,
                        )

                match spec.spectrum_type:
                    case "centroid":
                        spectrum_type = SpectrumType.CENTROID
                    case "profile":
                        spectrum_type = SpectrumType.PROFILE
                    case _:
                        raise ValueError(
                            f"Spectrum {spec} has unrecognized spectrum type: {spec.spectrum_type}"
                        )

                if charge_array is not None:
                    spectrum_type = SpectrumType.DECONVOLUTED

                mz_range = None
                if spec.lower_mz is not None and spec.upper_mz is not None:
                    mz_range = (spec.lower_mz, spec.upper_mz)

                yield MsnSpectrum(
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
                    rt=spec.scan_start_time.total_seconds()
                    if spec.scan_start_time is not None
                    else None,
                    mz_range=mz_range,
                    im_range=None,
                    polarity=spec.polarity,
                    resolution=None,  # Could extract from XML if needed
                    analyzer=None,  # Could extract from XML if needed
                    collision_energy=None,  # None
                    activation_type=None,  # None
                    ramp_time=None,
                    precursors=None,  # None
                )

    @property
    def ms2(self) -> Generator[MsnSpectrum, None, None]:
        """Yields MS2 spectra."""

        with mzp.Mzml(self.mzml_path) as reader:
            for spec in reader.spectra:
                if spec.ms_level != 2:
                    continue

                # Similar processing as MS1, but also extract precursor info
                mz_array = spec.mz
                if mz_array is None:
                    raise ValueError(f"Spectrum {spec} has no m/z array")
                mz_array = mz_array.astype(np.float64)
                int_array = spec.intensity
                if int_array is None:
                    raise ValueError(f"Spectrum {spec} has no intensity array")
                int_array = int_array.astype(np.float64)
                if len(mz_array) != len(int_array):
                    raise ValueError(
                        f"Spectrum {spec} has m/z and intensity arrays of different lengths"
                    )
                charge_array = spec.charge
                if charge_array is not None:
                    charge_array = charge_array.astype(np.int32)
                    if len(charge_array) != len(mz_array):
                        raise ValueError(
                            f"Spectrum {spec} has charge array of different length than m/z array"
                        )
                im_types = list(spec.im_types)
                im_array: np.ndarray | None = None
                if len(im_types) == 1:
                    darr = spec.get_binary_array(im_types[0])
                    if darr is None:
                        raise RuntimeError(
                            f"Spectrum {spec} has ion mobility array type {im_types[0]} but it is None"
                        )
                    im_array = darr.data.astype(np.float64)
                    if len(im_array) != len(mz_array):
                        raise ValueError(
                            f"Spectrum {spec} has ion mobility array of different length than m/z array"
                        )
                elif len(im_types) > 1:
                    warnings.warn(
                        f"Spectrum {spec} has multiple ion mobility arrays; only the first is used: {im_types[0]}",
                        stacklevel=2,
                    )
                    for im_type in im_types:
                        darr = spec.get_binary_array(im_type)
                        if darr is None:
                            raise RuntimeError(
                                f"Spectrum {spec}: multiple IM arrays, first is not None. "
                                f"Array types: {im_types}"
                            )
                        im_array = darr.data.astype(np.float64)
                        if len(im_array) != len(mz_array):
                            im_array = None
                            continue
                    if im_array is None:
                        warnings.warn(
                            f"Spectrum {spec}: no ion mobility array length matches m/z array. Array types: {im_types}",
                            stacklevel=2,
                        )

                match spec.spectrum_type:
                    case "centroid":
                        spectrum_type = SpectrumType.CENTROID
                    case "profile":
                        spectrum_type = SpectrumType.CENTROID
                    case _:
                        raise ValueError(
                            f"Spectrum {spec} has unrecognized spectrum type: {spec.spectrum_type}"
                        )
                if charge_array is not None:
                    spectrum_type = SpectrumType.DECONVOLUTED

                mz_range = None
                if spec.lower_mz is not None and spec.upper_mz is not None:
                    mz_range = (spec.lower_mz, spec.upper_mz)

                precursors: list[TargetIon] = []
                collision_energies = []
                activation_types = []

                for precursor in spec.precursors:
                    ions = precursor.selected_ions
                    if len(ions) == 0:
                        warnings.warn(
                            f"Spectrum {spec} has precursor with no selected ions (unexpected). Precursor: {precursor}",
                            stacklevel=2,
                        )
                        continue
                    if len(ions) > 1:
                        warnings.warn(
                            f"Spectrum {spec} has multiple selected ions; using first. Precursor: {precursor}",
                            stacklevel=2,
                        )
                    ion = ions[0]

                    mz = ion.selected_ion_mz
                    if mz is None:
                        warnings.warn(
                            f"Spectrum {spec} precursor selected ion missing m/z (unexpected). Precursor: {precursor}",
                            stacklevel=2,
                        )
                        continue
                    intensity = ion.peak_intensity
                    if intensity is None:
                        warnings.warn(
                            f"Spectrum {spec} precursor missing intensity (unexpected). Precursor: {precursor}",
                            stacklevel=2,
                        )
                        continue
                    charge = ion.charge_state
                    im = ion.ir_im

                    precursors.append(
                        TargetIon(
                            mz=mz,
                            intensity=intensity,
                            charge=charge,
                            im=im,
                            is_monoisotopic=None,
                        )
                    )

                    activation = precursor.activation
                    if activation is not None:
                        collision_energy = activation.ce
                        if collision_energy is not None:
                            collision_energies.append(collision_energy)
                        activation_type = activation.activation_type
                        if activation_type is not None:
                            activation_types.append(activation_type)
                if len(set(collision_energies)) > 1:
                    warnings.warn(
                        f"Spectrum {spec} has multiple collision energies (unexpected): {set(collision_energies)}",
                        stacklevel=2,
                    )
                if len(set(activation_types)) > 1:
                    warnings.warn(
                        f"Spectrum {spec} has multiple activation types (unexpected): {set(activation_types)}",
                        stacklevel=2,
                    )

                ce = collision_energies[0] if len(collision_energies) > 0 else None
                activation_type = (
                    activation_types[0] if len(activation_types) > 0 else None
                )

                yield MsnSpectrum(
                    mz=mz_array,
                    intensity=int_array,
                    charge=charge_array,
                    im=im_array,
                    spectrum_type=spectrum_type,  # TimsData spectra are already centroided
                    denoised=None,
                    normalized=None,
                    scan_number=spec.index,
                    ms_level=spec.ms_level,
                    native_id=spec.id,
                    rt=spec.scan_start_time.total_seconds()
                    if spec.scan_start_time is not None
                    else None,
                    mz_range=mz_range,
                    im_range=None,
                    polarity=spec.polarity,
                    resolution=None,
                    analyzer=None,  # Could extract from XML if needed
                    collision_energy=ce,
                    activation_type=activation_type,
                    ramp_time=None,
                    precursors=precursors,
                )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
