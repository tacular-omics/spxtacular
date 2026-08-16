"""Adapters for the optional :mod:`matchms` and :mod:`spectrum_utils` ecosystems.

Neither dependency is imported until its adapter is called.  This keeps
``import spxtacular`` lightweight and lets downstream projects use the core
package without installing either interoperability stack.

``matchms`` has a flexible metadata mapping, so a namespaced JSON payload is
included by default to make a spxtacular -> matchms -> spxtacular round-trip as
faithful as possible.  ``spectrum_utils.MsmsSpectrum`` intentionally has a much
narrower model; that conversion is therefore explicitly lossy.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from collections.abc import Mapping
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from .core import MsnSpectrum, Precursor, Spectrum, SpectrumType
from .enums import Polarity

__all__ = ["from_matchms", "from_spectrum_utils", "to_matchms", "to_spectrum_utils"]

_MATCHMS_PAYLOAD_KEY = "spxtacular_metadata"
_MATCHMS_PAYLOAD_VERSION = 1


class _SpectrumUtilsLike(Protocol):
    identifier: str
    precursor_mz: float
    precursor_charge: int
    mz: NDArray[np.float64]
    intensity: NDArray[np.float64]
    retention_time: float


def _matchms_spectrum_class() -> type:
    try:
        from matchms import Spectrum as MatchmsSpectrum
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "matchms interoperability requires the optional dependency; "
            "install it with `pip install 'spxtacular[matchms]'`"
        ) from exc
    return MatchmsSpectrum


def _spectrum_utils_class() -> type:
    try:
        from spectrum_utils.spectrum import MsmsSpectrum
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "spectrum_utils interoperability requires the optional dependency; "
            "install it with `pip install 'spxtacular[spectrum-utils]'`"
        ) from exc
    return MsmsSpectrum


def _json_default(obj: object) -> object:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return cast(Any, obj).tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _sorted_order(mz: NDArray[np.float64]) -> NDArray[np.intp]:
    """Return a stable m/z order, as required by both target libraries."""
    return np.argsort(mz, kind="stable").astype(np.intp, copy=False)


def _matchms_payload(spectrum: Spectrum, order: NDArray[np.intp]) -> str:
    payload = {
        "schema_version": _MATCHMS_PAYLOAD_VERSION,
        "kind": "MsnSpectrum" if isinstance(spectrum, MsnSpectrum) else "Spectrum",
        "meta": spectrum._meta_dict(),
        # Keep the original m/z values so per-peak extension arrays can still
        # be aligned after a matchms filter removes a subset of the peaks.
        "peak_mz": spectrum.mz[order],
        "charge": spectrum.charge[order] if spectrum.charge is not None else None,
        "im": spectrum.im[order] if spectrum.im is not None else None,
        "iso_score": spectrum.iso_score[order] if spectrum.iso_score is not None else None,
    }
    return json.dumps(payload, default=_json_default, separators=(",", ":"))


def _standard_matchms_metadata(spectrum: Spectrum) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if spectrum.spectrum_type is not None:
        metadata["spectrum_type"] = str(spectrum.spectrum_type)
    if spectrum.denoised is not None:
        metadata["denoised"] = spectrum.denoised
    if spectrum.normalized is not None:
        metadata["normalized"] = spectrum.normalized

    if not isinstance(spectrum, MsnSpectrum):
        return metadata

    scalar_fields = (
        "scan_number",
        "ms_level",
        "retention_time",
        "injection_time",
        "total_ion_current",
        "collision_energy",
        "activation_type",
    )
    values = (
        spectrum.scan_number,
        spectrum.ms_level,
        spectrum.rt,
        spectrum.injection_time,
        spectrum.total_ion_current,
        spectrum.collision_energy,
        spectrum.activation_type,
    )
    metadata.update({key: value for key, value in zip(scalar_fields, values, strict=True) if value is not None})

    if spectrum.native_id is not None:
        metadata["id"] = spectrum.native_id
    if spectrum.polarity is not None:
        metadata["ionmode"] = str(spectrum.polarity)
    if spectrum.analyzer is not None:
        metadata["instrument_type"] = str(spectrum.analyzer)

    if spectrum.precursors:
        precursor = spectrum.precursors[0]
        metadata["precursor_mz"] = float(precursor.mz)
        metadata["precursor_intensity"] = float(precursor.intensity)
        if precursor.charge is not None:
            metadata["charge"] = int(precursor.charge)
    return metadata


def to_matchms(
    spectrum: Spectrum,
    *,
    extra_metadata: Mapping[str, object] | None = None,
    include_spxtacular_metadata: bool = True,
) -> Any:
    """Convert a spxtacular spectrum to :class:`matchms.Spectrum`.

    Peaks are stable-sorted by m/z because matchms requires ordered input.  The
    conventional matchms fields are populated for ecosystem tools, while the
    ``spxtacular_metadata`` JSON field preserves richer metadata and per-peak
    arrays for return conversion.  ``extra_metadata`` is useful for chemical
    metadata such as ``smiles`` or ``inchikey`` that spxtacular does not model.
    Spxtacular-derived values take precedence when keys overlap.
    """
    MatchmsSpectrum = _matchms_spectrum_class()
    order = _sorted_order(spectrum.mz)
    metadata = dict(extra_metadata or {})
    metadata.update(_standard_matchms_metadata(spectrum))
    if include_spxtacular_metadata:
        metadata[_MATCHMS_PAYLOAD_KEY] = _matchms_payload(spectrum, order)
    return MatchmsSpectrum(
        mz=np.asarray(spectrum.mz[order], dtype=np.float64),
        intensities=np.asarray(spectrum.intensity[order], dtype=np.float64),
        metadata=metadata,
        # The adapter has already selected canonical keys and must not let
        # matchms rewrite the namespaced round-trip payload.
        metadata_harmonization=False,
    )


def _matchms_metadata(spectrum: object) -> dict[str, object]:
    metadata_dict = getattr(spectrum, "metadata_dict", None)
    if callable(metadata_dict):
        return dict(cast(Any, metadata_dict)())
    metadata = getattr(spectrum, "metadata", {})
    return dict(cast(Any, metadata)) if metadata is not None else {}


def _metadata_first(metadata: Mapping[str, object], *keys: str) -> object | None:
    for key in keys:
        value = metadata.get(key)
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            return value
    return None


def _optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().rstrip("+-")
        if not value:
            return None
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return None


def _optional_float(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    result = str(value)
    return result if result else None


def _precursor_mz(metadata: Mapping[str, object]) -> float | None:
    value = _metadata_first(metadata, "precursor_mz", "pepmass")
    if isinstance(value, (list, tuple, np.ndarray)):
        value = cast(Any, value)[0] if len(value) else None
    return _optional_float(value)


def _payload_peak_indices(current_mz: NDArray[np.float64], original_mz: NDArray[np.float64]) -> NDArray[np.intp] | None:
    """Align a matchms-filtered peak subset with stored extension arrays."""
    if np.array_equal(current_mz, original_mz):
        return np.arange(len(current_mz), dtype=np.intp)
    # If duplicate m/z values existed and a filter retained only some of them,
    # m/z alone cannot identify which extension row survived. Drop rather than
    # silently attach the first duplicate's charge/mobility/score.
    if len(np.unique(original_mz)) != len(original_mz):
        return None
    indices: list[int] = []
    original_i = 0
    for value in current_mz:
        while original_i < len(original_mz) and not np.isclose(original_mz[original_i], value, rtol=0.0, atol=1e-12):
            original_i += 1
        if original_i == len(original_mz):
            return None
        indices.append(original_i)
        original_i += 1
    return np.asarray(indices, dtype=np.intp)


def _from_matchms_payload(mz: NDArray[np.float64], intensity: NDArray[np.float64], payload_text: str) -> Spectrum:
    try:
        payload = json.loads(payload_text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("matchms spectrum has invalid spxtacular_metadata JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("matchms spxtacular_metadata JSON must contain an object")
    if payload.get("schema_version") != _MATCHMS_PAYLOAD_VERSION:
        raise ValueError(f"unsupported spxtacular_metadata schema version: {payload.get('schema_version')!r}")

    original_mz = np.asarray(payload.get("peak_mz", []), dtype=np.float64)
    indices = _payload_peak_indices(mz, original_mz)
    extension_arrays: dict[str, NDArray[Any] | None] = {"charge": None, "im": None, "iso_score": None}
    if indices is None:
        if any(payload.get(key) is not None for key in extension_arrays):
            warnings.warn(
                "matchms changed peak m/z values, so spxtacular per-peak charge, ion mobility, and isotope scores "
                "could not be realigned and were dropped",
                UserWarning,
                stacklevel=3,
            )
    else:
        dtypes = {"charge": np.int32, "im": np.float64, "iso_score": np.float64}
        for key, dtype in dtypes.items():
            values = payload.get(key)
            if values is not None:
                stored = np.asarray(values, dtype=dtype)
                if len(stored) != len(original_mz):
                    raise ValueError(f"spxtacular_metadata {key!r} array does not match its stored peak array")
                extension_arrays[key] = stored[indices]

    meta = payload.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("spxtacular_metadata is missing its metadata object")
    charge = cast(NDArray[np.int32] | None, extension_arrays["charge"])
    im = cast(NDArray[np.float64] | None, extension_arrays["im"])
    iso_score = cast(NDArray[np.float64] | None, extension_arrays["iso_score"])
    if payload.get("kind") == "MsnSpectrum":
        return MsnSpectrum(
            mz=mz,
            intensity=intensity,
            charge=charge,
            im=im,
            iso_score=iso_score,
            **MsnSpectrum._meta_kwargs(meta),
        )
    return Spectrum(
        mz=mz,
        intensity=intensity,
        charge=charge,
        im=im,
        iso_score=iso_score,
        **Spectrum._meta_kwargs(meta),
    )


def from_matchms(spectrum: object, *, prefer_spxtacular_metadata: bool = True) -> Spectrum:
    """Convert a :class:`matchms.Spectrum` to spxtacular.

    A payload produced by :func:`to_matchms` is used when available.  Peak
    extension arrays remain aligned when ordinary matchms filters retain a
    subset of peaks.  Foreign matchms spectra are mapped from their conventional
    metadata fields and become :class:`MsnSpectrum` when MSn metadata exists.
    """
    peaks = getattr(spectrum, "peaks", None)
    if peaks is None or not hasattr(peaks, "mz") or not hasattr(peaks, "intensities"):
        raise TypeError("expected a matchms Spectrum with peaks.mz and peaks.intensities")
    mz = np.asarray(cast(Any, peaks).mz, dtype=np.float64)
    intensity = np.asarray(cast(Any, peaks).intensities, dtype=np.float64)
    metadata = _matchms_metadata(spectrum)

    payload = metadata.get(_MATCHMS_PAYLOAD_KEY)
    if prefer_spxtacular_metadata and payload is not None:
        if not isinstance(payload, str):
            raise ValueError("matchms spxtacular_metadata must be a JSON string")
        return _from_matchms_payload(mz, intensity, payload)

    precursor_mz = _precursor_mz(metadata)
    ms_level = _optional_int(metadata.get("ms_level"))
    msn_keys = {
        "id",
        "title",
        "scan_number",
        "ms_level",
        "retention_time",
        "scan_start_time",
        "precursor_mz",
        "pepmass",
        "ionmode",
        "collision_energy",
    }
    is_msn = precursor_mz is not None or any(metadata.get(key) is not None for key in msn_keys)
    if not is_msn:
        return Spectrum(
            mz=mz,
            intensity=intensity,
            spectrum_type=_optional_str(metadata.get("spectrum_type")),
            denoised=_optional_str(metadata.get("denoised")),
            normalized=_optional_str(metadata.get("normalized")),
        )

    precursor = None
    if precursor_mz is not None:
        precursor = [
            Precursor(
                mz=precursor_mz,
                intensity=_optional_float(metadata.get("precursor_intensity")) or 0.0,
                charge=_optional_int(metadata.get("charge")),
                is_monoisotopic=None,
            )
        ]
    native_id = _metadata_first(metadata, "id", "spectrum_id", "title")
    polarity_value = _metadata_first(metadata, "ionmode", "ion_mode")
    try:
        polarity = Polarity(str(polarity_value).lower()) if polarity_value is not None else None
    except ValueError:
        polarity = None
    return MsnSpectrum(
        mz=mz,
        intensity=intensity,
        spectrum_type=_optional_str(metadata.get("spectrum_type")) or SpectrumType.CENTROID,
        denoised=_optional_str(metadata.get("denoised")),
        normalized=_optional_str(metadata.get("normalized")),
        native_id=str(native_id) if native_id is not None else None,
        scan_number=_optional_int(metadata.get("scan_number")),
        ms_level=ms_level if ms_level is not None else (2 if precursor is not None else None),
        rt=_optional_float(_metadata_first(metadata, "retention_time", "scan_start_time")),
        polarity=polarity,
        analyzer=_optional_str(_metadata_first(metadata, "instrument_type", "analyzer")),
        collision_energy=_optional_float(metadata.get("collision_energy")),
        activation_type=_optional_str(metadata.get("activation_type")),
        precursors=precursor,
    )


def _spectrum_utils_identifier(spectrum: MsnSpectrum, identifier: str | None) -> str:
    if identifier is not None:
        return identifier
    if spectrum.native_id is not None:
        return spectrum.native_id
    if spectrum.scan_number is not None:
        return f"scan={spectrum.scan_number}"
    raise ValueError("spectrum_utils conversion requires an identifier, native_id, or scan_number")


def _warn_spectrum_utils_loss(spectrum: MsnSpectrum, precursor_index: int, identifier: str) -> None:
    dropped: list[str] = []
    if spectrum.spectrum_type not in (None, SpectrumType.CENTROID, "centroid"):
        dropped.append("spectrum_type")
    for field in ("charge", "im", "iso_score", "denoised", "normalized"):
        if getattr(spectrum, field) is not None:
            dropped.append(field)
    for field in (
        "injection_time",
        "total_ion_current",
        "im_type",
        "polarity",
        "resolution",
        "analyzer",
        "ramp_time",
        "collision_energy",
        "activation_type",
        "mz_range",
        "im_range",
        "isolation_mz_range",
        "isolation_im_range",
    ):
        if getattr(spectrum, field) is not None:
            dropped.append(field)
    if spectrum.ms_level not in (None, 2):
        dropped.append("ms_level")
    if spectrum.scan_number is not None:
        scan_pattern = rf"(?:^|\s)scan={spectrum.scan_number}(?:\s|$)"
        if re.search(scan_pattern, identifier) is None:
            dropped.append("scan_number")
    if spectrum.precursors is not None and len(spectrum.precursors) > 1:
        dropped.append(f"all precursors except index {precursor_index}")
    if spectrum.precursors:
        precursor = spectrum.precursors[precursor_index]
        if precursor.intensity != 0.0:
            dropped.append("precursor intensity")
        if precursor.im is not None:
            dropped.append("precursor ion mobility")
        if precursor.iso_score is not None:
            dropped.append("precursor isotope score")
        if precursor.is_monoisotopic is not None:
            dropped.append("precursor monoisotopic flag")
    if dropped:
        warnings.warn(
            "spectrum_utils cannot represent and will drop: " + ", ".join(dropped),
            UserWarning,
            stacklevel=3,
        )


def to_spectrum_utils(
    spectrum: MsnSpectrum,
    *,
    precursor_index: int = 0,
    identifier: str | None = None,
    warn_on_loss: bool = True,
) -> Any:
    """Convert an MSn spectrum to ``spectrum_utils.MsmsSpectrum``.

    ``spectrum_utils`` supports one precursor and converts intensities to
    float32.  Missing required precursor information raises instead of being
    invented; fields outside its model are reported with ``UserWarning``.
    """
    if not isinstance(spectrum, MsnSpectrum):
        raise TypeError("spectrum_utils conversion requires an MsnSpectrum")
    if spectrum.spectrum_type == SpectrumType.PROFILE or spectrum.spectrum_type == "profile":
        raise ValueError("spectrum_utils.MsmsSpectrum requires centroided fragment peaks, not profile data")
    if not spectrum.precursors:
        raise ValueError("spectrum_utils conversion requires at least one precursor")
    try:
        precursor = spectrum.precursors[precursor_index]
    except IndexError:
        message = f"precursor_index {precursor_index} is out of range for {len(spectrum.precursors)} precursors"
        raise IndexError(message) from None
    if precursor.charge is None:
        raise ValueError("spectrum_utils conversion requires a precursor charge")
    if not -128 <= precursor.charge <= 127:
        raise ValueError("spectrum_utils precursor charge must fit in a signed 8-bit integer")
    resolved_identifier = _spectrum_utils_identifier(spectrum, identifier)
    if warn_on_loss:
        _warn_spectrum_utils_loss(spectrum, precursor_index, resolved_identifier)

    MsmsSpectrum = _spectrum_utils_class()
    order = _sorted_order(spectrum.mz)
    return MsmsSpectrum(
        resolved_identifier,
        float(precursor.mz),
        int(precursor.charge),
        np.asarray(spectrum.mz[order], dtype=np.float64),
        np.asarray(spectrum.intensity[order], dtype=np.float64),
        float(spectrum.rt) if spectrum.rt is not None else np.nan,
    )


def from_spectrum_utils(spectrum: object, *, warn_on_loss: bool = True) -> MsnSpectrum:
    """Convert ``spectrum_utils.MsmsSpectrum`` to :class:`MsnSpectrum`.

    Existing ProForma annotations cannot be represented by spxtacular's core
    spectrum object and trigger a warning before being dropped.
    """
    required = ("identifier", "precursor_mz", "precursor_charge", "mz", "intensity")
    missing = [name for name in required if not hasattr(spectrum, name)]
    if missing:
        raise TypeError("expected a spectrum_utils.MsmsSpectrum; missing " + ", ".join(missing))
    source = cast(_SpectrumUtilsLike, spectrum)
    if warn_on_loss and (
        getattr(spectrum, "proforma", None) is not None or getattr(spectrum, "annotation", None) is not None
    ):
        warnings.warn(
            "spectrum_utils ProForma annotations are not represented by MsnSpectrum and were dropped",
            UserWarning,
            stacklevel=2,
        )
    rt = _optional_float(source.retention_time)
    identifier = str(source.identifier) if source.identifier else None
    scan_match = re.search(r"(?:^|\s)scan=(\d+)(?:\s|$)", identifier or "")
    return MsnSpectrum(
        mz=np.asarray(source.mz, dtype=np.float64),
        intensity=np.asarray(source.intensity, dtype=np.float64),
        spectrum_type=SpectrumType.CENTROID,
        native_id=identifier,
        scan_number=int(scan_match.group(1)) if scan_match is not None else None,
        ms_level=2,
        rt=rt,
        precursors=[
            Precursor(
                mz=float(source.precursor_mz),
                intensity=0.0,
                charge=int(source.precursor_charge),
                is_monoisotopic=None,
            )
        ],
    )
