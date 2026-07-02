"""Bridge between spxtacular's :class:`~spxtacular.core.Spectrum` and
`spectrl <https://github.com/tacular-omics/spectrl>`_'s
:class:`InlineSpectrum` / token format.

`spectrl` is a sibling library that encodes a full mass spectrum into a single
compact, URL-safe token. Its data model is faithful to mzML (typed
``SpectrlCvParam`` lists with PSI-MS accessions, a single CBOR document,
MS-Numpress peak compression, SHA-256 integrity hash) and is well-suited for sharing
spectra outside spxtacular — embedded in URLs, QR codes, notebooks, papers.

This module provides the four-way conversion::

    Spectrum / MsnSpectrum  →  spectrl.InlineSpectrum   →  token (str)
    Spectrum / MsnSpectrum  ←  spectrl.DecodedSpectrum  ←  token (str)

Install ``spxtacular[spectrl]`` to enable. Without the extra, importing this
module raises :class:`ImportError`.

Per-peak ``iso_score`` is carried through spectrl's ``extra_arrays`` slot
under the key ``"iso_score"`` (encoded as a non-standard mzML binary array,
``MS:1000786``). spxtacular-specific scalar fields without an mzML
counterpart — ``denoised``/``normalized`` provenance strings,
``scan_number``, ``resolution``, ``analyzer``, ``ramp_time``, ``im_range``,
``isolation_im_range`` — are carried losslessly as namespaced free-text
``user_params`` (``spxtacular:`` prefix), so the round-trip is faithful.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .core import MsnSpectrum, Precursor, Spectrum, SpectrumType

if TYPE_CHECKING:
    from spectrl import DecodedSpectrum, InlineSpectrum

try:
    import spectrl as _spectrl  # noqa: F401

    _HAS_SPECTRL = True
except ImportError:
    _HAS_SPECTRL = False


# ---------------------------------------------------------------------------
# PSI-MS accession constants used by the bridge
# ---------------------------------------------------------------------------

# Spectrum-level
_MS_LEVEL = "MS:1000511"
_CENTROID = "MS:1000127"
_PROFILE = "MS:1000128"
_POSITIVE = "MS:1000130"
_NEGATIVE = "MS:1000129"
_TIC = "MS:1000285"

# Scan-level
_SCAN_START_TIME = "MS:1000016"
_UNIT_SECOND = "UO:0000010"
_SCAN_WINDOW_LOWER = "MS:1000500"
_SCAN_WINDOW_UPPER = "MS:1000501"

# Precursor / isolation / activation
_SELECTED_ION_MZ = "MS:1000744"
_CHARGE_STATE = "MS:1000041"
_PEAK_INTENSITY = "MS:1000042"
_ISOL_TARGET_MZ = "MS:1000827"
_ISOL_LOWER_OFFSET = "MS:1000828"
_ISOL_UPPER_OFFSET = "MS:1000829"
_COLLISION_ENERGY = "MS:1000045"

# Common activation accessions; unrecognised strings pass through unchanged so
# the round trip stays faithful even for activation types we don't know about.
_ACTIVATION_ACCESSIONS: dict[str, str] = {
    "HCD": "MS:1000422",
    "CID": "MS:1000133",
    "ETD": "MS:1000598",
    "ECD": "MS:1000250",
    "PASEF": "MS:1002481",
    "MS:1002481": "MS:1002481",  # already an accession (BD-PASEF default in DReader)
}

# Inverse mapping for decoding
_ACTIVATION_NAMES: dict[str, str] = {v: k for k, v in _ACTIVATION_ACCESSIONS.items() if not k.startswith("MS:")}

# Ion-mobility array accessions (must be drawn from spectrl.ION_MOBILITY_ARRAY_TAILS,
# i.e. mzmlpy.constants.ION_MOBILITIES, otherwise spectrl will not recognise them on decode).
_IM_TYPE_ACCESSIONS: dict[str, str] = {
    "ook0": "MS:1003008",      # raw inverse reduced ion mobility (1/K0)
    "1/k0": "MS:1003008",
    "im": "MS:1002893",        # generic ion mobility
    "drift_time": "MS:1003153",
    "drift_time_ms": "MS:1003153",
    "ccs": "MS:1003007",       # raw ion mobility (CCS)
}

# Reverse lookup for decoding (covers raw/mean/deconvoluted variants).
_IM_TYPE_FROM_ACCESSION: dict[str, str] = {
    "MS:1003008": "ook0",   # raw inverse reduced ion mobility (1/K0)
    "MS:1003006": "ook0",   # mean inverse reduced ion mobility
    "MS:1003155": "ook0",   # deconvoluted inverse reduced ion mobility
    "MS:1002816": "ook0",   # mean ion mobility (fall back to ook0)
    "MS:1002893": "im",
    "MS:1003153": "drift_time_ms",
    "MS:1002477": "drift_time_ms",
    "MS:1003156": "drift_time_ms",
    "MS:1003007": "ccs",
}

_POLARITY_FROM_ACCESSION: dict[str, str] = {
    _POSITIVE: "positive",
    _NEGATIVE: "negative",
}

# ---------------------------------------------------------------------------
# Free-text user-param names for spxtacular scalar fields without an mzML
# CV counterpart. Carried losslessly through spectrl's ``user_params`` slot
# (added in the CBOR token format). Namespaced to avoid clashing with any
# real mzML userParam a producer may have emitted.
# ---------------------------------------------------------------------------

_UP_PREFIX = "spxtacular:"
_UP_DENOISED = _UP_PREFIX + "denoised"
_UP_NORMALIZED = _UP_PREFIX + "normalized"
_UP_SCAN_NUMBER = _UP_PREFIX + "scan_number"
_UP_RESOLUTION = _UP_PREFIX + "resolution"
_UP_ANALYZER = _UP_PREFIX + "analyzer"
_UP_RAMP_TIME = _UP_PREFIX + "ramp_time"
_UP_IM_RANGE_LO = _UP_PREFIX + "im_range_lower"
_UP_IM_RANGE_HI = _UP_PREFIX + "im_range_upper"
_UP_ISOL_IM_RANGE_LO = _UP_PREFIX + "isolation_im_range_lower"
_UP_ISOL_IM_RANGE_HI = _UP_PREFIX + "isolation_im_range_upper"
_UP_IM_TYPE = _UP_PREFIX + "im_type"

# user-param names that only an MsnSpectrum can carry (force MSn on decode)
_MSN_USER_PARAMS: frozenset[str] = frozenset(
    {
        _UP_SCAN_NUMBER,
        _UP_RESOLUTION,
        _UP_ANALYZER,
        _UP_RAMP_TIME,
        _UP_IM_RANGE_LO,
        _UP_IM_RANGE_HI,
        _UP_ISOL_IM_RANGE_LO,
        _UP_ISOL_IM_RANGE_HI,
        _UP_IM_TYPE,
    }
)


# ---------------------------------------------------------------------------
# Module-level guard
# ---------------------------------------------------------------------------


def _require_spectrl() -> None:
    if not _HAS_SPECTRL:
        raise ImportError(
            "spectrl is required for this operation. Install with: pip install spxtacular[spectrl]"
        )


# ---------------------------------------------------------------------------
# spxtacular  →  spectrl
# ---------------------------------------------------------------------------


def _cv(accession: str, value=None, unit_accession: str | None = None):
    """Build a SpectrlCvParam without keeping the import at module top-level."""
    from spectrl.model import SpectrlCvParam

    return SpectrlCvParam(accession=accession, value=value, unit_accession=unit_accession)


def _up(name: str, value=None, type_: str | None = None):
    """Build a SpectrlUserParam (free-text, no CV accession)."""
    from spectrl.model import SpectrlUserParam

    return SpectrlUserParam(name=name, value=value, type=type_)


def to_inline_spectrum(spec: Spectrum) -> InlineSpectrum:
    """Convert a spxtacular :class:`~spxtacular.core.Spectrum` (or
    :class:`~spxtacular.core.MsnSpectrum`) to an
    :class:`spectrl.InlineSpectrum` ready for :func:`spectrl.encode_spectrum`.

    Carries: ``mz``, ``intensity``, ``charge`` (cast to float64), ``im`` (with
    its ``im_type`` accession), ``iso_score`` (via ``extra_arrays["iso_score"]``,
    encoded as a non-standard mzML binary array ``MS:1000786``), and — for
    ``MsnSpectrum`` — ``native_id``, ``ms_level``, ``polarity``, ``rt`` (seconds),
    ``mz_range``, ``precursors`` (each with isolation window, selected ion,
    activation params), and ``total_ion_current``.

    spxtacular scalar fields without an mzML CV counterpart —
    ``denoised``/``normalized`` provenance strings, ``scan_number``,
    ``resolution``, ``analyzer``, ``ramp_time``, ``im_range``,
    ``isolation_im_range`` — are carried losslessly as namespaced free-text
    ``user_params`` (see the ``spxtacular:`` prefixed names above).
    """
    _require_spectrl()
    from spectrl.model import (
        InlineSpectrum,
        SpectrlActivation,
        SpectrlIsolationWindow,
        SpectrlPrecursor,
        SpectrlScan,
        SpectrlScanWindow,
        SpectrlSelectedIon,
    )

    n = len(spec.mz)
    params = []

    # Spectrum type
    st = spec.spectrum_type
    if st == SpectrumType.CENTROID or st == "centroid":
        params.append(_cv(_CENTROID))
    elif st == SpectrumType.PROFILE or st == "profile":
        params.append(_cv(_PROFILE))

    msn_spec: MsnSpectrum | None = spec if isinstance(spec, MsnSpectrum) else None

    if msn_spec is not None:
        if msn_spec.ms_level is not None:
            params.append(_cv(_MS_LEVEL, value=int(msn_spec.ms_level)))
        if msn_spec.polarity == "positive":
            params.append(_cv(_POSITIVE))
        elif msn_spec.polarity == "negative":
            params.append(_cv(_NEGATIVE))
        if msn_spec.total_ion_current is not None:
            params.append(_cv(_TIC, value=float(msn_spec.total_ion_current)))

    # Scan(s) — one entry holding RT and the m/z scan window
    scans: list[SpectrlScan] = []
    if msn_spec is not None:
        scan_params = []
        if msn_spec.rt is not None:
            scan_params.append(_cv(_SCAN_START_TIME, value=float(msn_spec.rt), unit_accession=_UNIT_SECOND))
        windows: list[SpectrlScanWindow] = []
        if msn_spec.mz_range is not None:
            lo, hi = msn_spec.mz_range
            windows.append(
                SpectrlScanWindow(
                    params=[
                        _cv(_SCAN_WINDOW_LOWER, value=float(lo)),
                        _cv(_SCAN_WINDOW_UPPER, value=float(hi)),
                    ]
                )
            )
        if scan_params or windows:
            scans.append(SpectrlScan(params=scan_params, windows=windows))

    # Ion mobility type accession — computed up front so precursor ion mobility
    # (below) can be tagged with the same accession as the spectrum-level array.
    ion_mobility_type: str | None = None
    if spec.im is not None and msn_spec is not None and msn_spec.im_type is not None:
        ion_mobility_type = _IM_TYPE_ACCESSIONS.get(msn_spec.im_type.lower(), "MS:1002893")
    elif spec.im is not None:
        ion_mobility_type = "MS:1002893"  # generic ion mobility when type unknown

    # Precursors
    precursors: list[SpectrlPrecursor] = []
    if msn_spec is not None and msn_spec.precursors:
        for prec in msn_spec.precursors:
            iw: SpectrlIsolationWindow | None = None
            if msn_spec.isolation_mz_range is not None:
                lo, hi = msn_spec.isolation_mz_range
                center = (lo + hi) / 2.0
                iw = SpectrlIsolationWindow(
                    params=[
                        _cv(_ISOL_TARGET_MZ, value=float(center)),
                        _cv(_ISOL_LOWER_OFFSET, value=float(center - lo)),
                        _cv(_ISOL_UPPER_OFFSET, value=float(hi - center)),
                    ]
                )

            ion_params = [_cv(_SELECTED_ION_MZ, value=float(prec.mz))]
            if prec.charge is not None:
                ion_params.append(_cv(_CHARGE_STATE, value=int(prec.charge)))
            if prec.intensity is not None and prec.intensity != 0.0:
                ion_params.append(_cv(_PEAK_INTENSITY, value=float(prec.intensity)))
            if prec.im is not None:
                ion_params.append(_cv(ion_mobility_type or "MS:1002893", value=float(prec.im)))
            selected_ion = SpectrlSelectedIon(params=ion_params)

            activation: SpectrlActivation | None = None
            act_params = []
            if msn_spec.collision_energy is not None:
                act_params.append(_cv(_COLLISION_ENERGY, value=float(msn_spec.collision_energy)))
            if msn_spec.activation_type is not None:
                acc = _ACTIVATION_ACCESSIONS.get(msn_spec.activation_type, msn_spec.activation_type)
                act_params.append(_cv(acc))
            if act_params:
                activation = SpectrlActivation(params=act_params)

            precursors.append(
                SpectrlPrecursor(
                    isolation_window=iw,
                    selected_ions=[selected_ion],
                    activation=activation,
                )
            )

    # charge needs to be float64 for spectrl's array model
    charge_arr: np.ndarray | None = None
    if spec.charge is not None:
        charge_arr = spec.charge.astype(np.float64)

    # iso_score travels as a non-standard mzML binary array (MS:1000786)
    # via spectrl's extra_arrays slot.
    extra_arrays: dict[str, np.ndarray] = {}
    if spec.iso_score is not None:
        extra_arrays["iso_score"] = spec.iso_score.astype(np.float64)

    # spxtacular scalar fields with no CV term travel as namespaced user_params.
    user_params = []
    if spec.denoised is not None:
        user_params.append(_up(_UP_DENOISED, spec.denoised, "xsd:string"))
    if spec.normalized is not None:
        user_params.append(_up(_UP_NORMALIZED, spec.normalized, "xsd:string"))
    if msn_spec is not None:
        if msn_spec.scan_number is not None:
            user_params.append(_up(_UP_SCAN_NUMBER, int(msn_spec.scan_number), "xsd:int"))
        if msn_spec.resolution is not None:
            user_params.append(_up(_UP_RESOLUTION, float(msn_spec.resolution), "xsd:float"))
        if msn_spec.analyzer is not None:
            user_params.append(_up(_UP_ANALYZER, msn_spec.analyzer, "xsd:string"))
        if msn_spec.ramp_time is not None:
            user_params.append(_up(_UP_RAMP_TIME, float(msn_spec.ramp_time), "xsd:float"))
        if msn_spec.im_range is not None:
            lo, hi = msn_spec.im_range
            user_params.append(_up(_UP_IM_RANGE_LO, float(lo), "xsd:float"))
            user_params.append(_up(_UP_IM_RANGE_HI, float(hi), "xsd:float"))
        if msn_spec.isolation_im_range is not None:
            lo, hi = msn_spec.isolation_im_range
            user_params.append(_up(_UP_ISOL_IM_RANGE_LO, float(lo), "xsd:float"))
            user_params.append(_up(_UP_ISOL_IM_RANGE_HI, float(hi), "xsd:float"))
        if msn_spec.im_type is not None:
            # The CV accession (ion_mobility_type) only carries a coarse, spectrl-recognised
            # IM category; stash the exact string here so unrecognized types round-trip too.
            user_params.append(_up(_UP_IM_TYPE, msn_spec.im_type, "xsd:string"))

    return InlineSpectrum(
        default_array_length=n,
        mz=spec.mz.astype(np.float64) if spec.mz is not None else None,
        intensity=spec.intensity.astype(np.float64) if spec.intensity is not None else None,
        charge=charge_arr,
        ion_mobility=spec.im.astype(np.float64) if spec.im is not None else None,
        ion_mobility_type=ion_mobility_type,
        id=msn_spec.native_id if msn_spec is not None else None,
        params=params,
        scans=scans,
        precursors=precursors,
        extra_arrays=extra_arrays,
        user_params=user_params,
    )


def to_spectrl_token(spec: Spectrum, *, lossless: bool = False, max_len: int | None = None) -> str:
    """Encode a spxtacular spectrum directly to a ``spectrl1.…`` token.

    Convenience wrapper over :func:`to_inline_spectrum` +
    :func:`spectrl.encode_spectrum`. See :func:`spectrl.encode_spectrum` for
    the meaning of ``lossless`` and ``max_len``.
    """
    _require_spectrl()
    from spectrl import encode_spectrum

    return encode_spectrum(to_inline_spectrum(spec), lossless=lossless, max_len=max_len)


# ---------------------------------------------------------------------------
# spectrl  →  spxtacular
# ---------------------------------------------------------------------------


def _find_param(params: list, accession: str):
    """Return the first SpectrlCvParam in ``params`` matching ``accession``, or None."""
    for p in params:
        if p.accession == accession:
            return p
    return None


def _range_from_user_params(
    up: dict[str, str | float | int | None], lo_key: str, hi_key: str
) -> tuple[float, float] | None:
    """Rebuild a ``(lower, upper)`` tuple from two user-param values, or None."""
    lo, hi = up.get(lo_key), up.get(hi_key)
    if lo is None or hi is None:
        return None
    return (float(lo), float(hi))


def from_decoded_spectrum(decoded: DecodedSpectrum) -> Spectrum:
    """Convert a :class:`spectrl.DecodedSpectrum` back to a spxtacular
    :class:`~spxtacular.core.Spectrum` (or :class:`~spxtacular.core.MsnSpectrum`
    when MSn metadata is present).

    Returns a plain :class:`Spectrum` when no MSn metadata is found in the
    spectrl token; otherwise an :class:`MsnSpectrum` populated with
    ``native_id``, ``ms_level``, ``polarity``, ``rt``, ``mz_range``,
    ``total_ion_current``, ``precursors`` (with ``charge``, ``intensity``,
    ``im``, and ``is_monoisotopic=None``), ``isolation_mz_range``,
    ``collision_energy``, ``activation_type``, and ``im_type``.

    spxtacular scalar fields carried as namespaced ``user_params`` —
    ``denoised``/``normalized`` (also restored on a plain :class:`Spectrum`),
    ``scan_number``, ``resolution``, ``analyzer``, ``ramp_time``, ``im_range``,
    ``isolation_im_range`` — are restored too; the MSn-only ones force an
    :class:`MsnSpectrum` even when no other MSn metadata is present.
    """
    _require_spectrl()

    mz = np.asarray(decoded.mz, dtype=np.float64) if decoded.mz is not None else np.empty(0, dtype=np.float64)
    intensity = (
        np.asarray(decoded.intensity, dtype=np.float64)
        if decoded.intensity is not None
        else np.empty(0, dtype=np.float64)
    )
    charge = np.asarray(decoded.charge, dtype=np.int32) if decoded.charge is not None else None
    im = np.asarray(decoded.ion_mobility, dtype=np.float64) if decoded.ion_mobility is not None else None

    # iso_score is carried as a non-standard mzML binary array under
    # extra_arrays["iso_score"] — see to_inline_spectrum.
    iso_score_arr = decoded.extra_arrays.get("iso_score") if decoded.extra_arrays else None
    iso_score = np.asarray(iso_score_arr, dtype=np.float64) if iso_score_arr is not None else None

    # Spectrum type
    spectrum_type: SpectrumType | None = None
    if _find_param(decoded.params, _CENTROID):
        spectrum_type = SpectrumType.CENTROID
    elif _find_param(decoded.params, _PROFILE):
        spectrum_type = SpectrumType.PROFILE

    # spxtacular scalar fields carried as namespaced user_params.
    up: dict[str, str | float | int | None] = (
        {p.name: p.value for p in decoded.user_params} if decoded.user_params else {}
    )
    denoised_v = up.get(_UP_DENOISED)
    denoised = str(denoised_v) if denoised_v is not None else None
    normalized_v = up.get(_UP_NORMALIZED)
    normalized = str(normalized_v) if normalized_v is not None else None

    # Detect whether any MSn metadata is present
    ms_level_p = _find_param(decoded.params, _MS_LEVEL)
    has_msn = (
        decoded.id is not None
        or ms_level_p is not None
        or decoded.scans
        or decoded.precursors
        or decoded.ion_mobility_type is not None
        or _find_param(decoded.params, _POSITIVE) is not None
        or _find_param(decoded.params, _NEGATIVE) is not None
        or any(name in up for name in _MSN_USER_PARAMS)
    )

    if not has_msn:
        return Spectrum(
            mz=mz,
            intensity=intensity,
            charge=charge,
            im=im,
            iso_score=iso_score,
            spectrum_type=spectrum_type,
            denoised=denoised,
            normalized=normalized,
        )

    # MSn metadata
    ms_level: int | None = int(ms_level_p.value) if ms_level_p is not None and ms_level_p.value is not None else None

    polarity: str | None = None
    if _find_param(decoded.params, _POSITIVE):
        polarity = "positive"
    elif _find_param(decoded.params, _NEGATIVE):
        polarity = "negative"

    tic_p = _find_param(decoded.params, _TIC)
    total_ion_current = float(tic_p.value) if tic_p is not None and tic_p.value is not None else None

    # First scan (if any) holds rt + mz_range
    rt: float | None = None
    mz_range: tuple[float, float] | None = None
    if decoded.scans:
        scan = decoded.scans[0]
        rt_p = _find_param(scan.params, _SCAN_START_TIME)
        if rt_p is not None and rt_p.value is not None:
            rt = float(rt_p.value)
        for window in scan.windows:
            lo_p = _find_param(window.params, _SCAN_WINDOW_LOWER)
            hi_p = _find_param(window.params, _SCAN_WINDOW_UPPER)
            if lo_p is not None and hi_p is not None:
                mz_range = (float(lo_p.value), float(hi_p.value))
                break

    # Precursors
    precursors: list[Precursor] | None = None
    isolation_mz_range: tuple[float, float] | None = None
    collision_energy: float | None = None
    activation_type: str | None = None
    if decoded.precursors:
        precursors = []
        for sp in decoded.precursors:
            for ion in sp.selected_ions:
                mz_p = _find_param(ion.params, _SELECTED_ION_MZ)
                if mz_p is None or mz_p.value is None:
                    continue
                charge_p = _find_param(ion.params, _CHARGE_STATE)
                intensity_p = _find_param(ion.params, _PEAK_INTENSITY)
                prec_intensity = (
                    float(intensity_p.value) if intensity_p is not None and intensity_p.value is not None else 0.0
                )
                prec_charge = (
                    int(charge_p.value) if charge_p is not None and charge_p.value is not None else None
                )
                prec_im: float | None = None
                for p in ion.params:
                    if p.accession in _IM_TYPE_FROM_ACCESSION and p.value is not None:
                        prec_im = float(p.value)
                        break
                precursors.append(
                    Precursor(
                        mz=float(mz_p.value),
                        intensity=prec_intensity,
                        charge=prec_charge,
                        im=prec_im,
                        is_monoisotopic=None,
                    )
                )
            if sp.isolation_window is not None and isolation_mz_range is None:
                target_p = _find_param(sp.isolation_window.params, _ISOL_TARGET_MZ)
                lo_p = _find_param(sp.isolation_window.params, _ISOL_LOWER_OFFSET)
                hi_p = _find_param(sp.isolation_window.params, _ISOL_UPPER_OFFSET)
                if target_p is not None and lo_p is not None and hi_p is not None:
                    center = float(target_p.value)
                    isolation_mz_range = (center - float(lo_p.value), center + float(hi_p.value))
            if sp.activation is not None:
                ce_p = _find_param(sp.activation.params, _COLLISION_ENERGY)
                if ce_p is not None and ce_p.value is not None and collision_energy is None:
                    collision_energy = float(ce_p.value)
                if activation_type is None:
                    for p in sp.activation.params:
                        if p.accession != _COLLISION_ENERGY:
                            activation_type = _ACTIVATION_NAMES.get(p.accession, p.accession)
                            break

    im_type_v = up.get(_UP_IM_TYPE)
    im_type = (
        str(im_type_v)
        if im_type_v is not None
        else (_IM_TYPE_FROM_ACCESSION.get(decoded.ion_mobility_type) if decoded.ion_mobility_type else None)
    )

    # Remaining spxtacular scalar fields from user_params.
    scan_number_v = up.get(_UP_SCAN_NUMBER)
    scan_number = int(scan_number_v) if scan_number_v is not None else None
    resolution_v = up.get(_UP_RESOLUTION)
    resolution = float(resolution_v) if resolution_v is not None else None
    analyzer_v = up.get(_UP_ANALYZER)
    analyzer = str(analyzer_v) if analyzer_v is not None else None
    ramp_time_v = up.get(_UP_RAMP_TIME)
    ramp_time = float(ramp_time_v) if ramp_time_v is not None else None
    im_range = _range_from_user_params(up, _UP_IM_RANGE_LO, _UP_IM_RANGE_HI)
    isolation_im_range = _range_from_user_params(up, _UP_ISOL_IM_RANGE_LO, _UP_ISOL_IM_RANGE_HI)

    return MsnSpectrum(
        mz=mz,
        intensity=intensity,
        charge=charge,
        im=im,
        iso_score=iso_score,
        spectrum_type=spectrum_type,
        native_id=decoded.id,
        ms_level=ms_level,
        rt=rt,
        polarity=polarity,
        total_ion_current=total_ion_current,
        mz_range=mz_range,
        precursors=precursors if precursors else None,
        isolation_mz_range=isolation_mz_range,
        collision_energy=collision_energy,
        activation_type=activation_type,
        im_type=im_type,
        denoised=denoised,
        normalized=normalized,
        scan_number=scan_number,
        resolution=resolution,
        analyzer=analyzer,
        ramp_time=ramp_time,
        im_range=im_range,
        isolation_im_range=isolation_im_range,
    )


def from_spectrl_token(token: str) -> Spectrum:
    """Decode a ``spectrl1.…`` token into a spxtacular
    :class:`~spxtacular.core.Spectrum` (or :class:`~spxtacular.core.MsnSpectrum`
    when MSn metadata is present).
    """
    _require_spectrl()
    from spectrl import decode_token

    return from_decoded_spectrum(decode_token(token))


# ---------------------------------------------------------------------------
# URL sharing helpers
# ---------------------------------------------------------------------------


def to_spectrl_url(
    spec: Spectrum,
    base: str | None = None,
    *,
    mode: str = "fragment",
    param: str = "d",
    lossless: bool = False,
    max_len: int | None = None,
) -> str:
    """Encode a spxtacular spectrum into a shareable URL (or ``data:`` URI).

    Convenience wrapper over :func:`to_spectrl_token` + spectrl's URL binding
    helpers. ``mode`` selects the binding:

    - ``"fragment"`` (default): ``base#spectrl1.…`` — the token rides in the URL
      fragment, which is never sent to the server (no length limits, no logs).
    - ``"query"``: ``base?<param>=spectrl1.…`` — token as a query parameter.
    - ``"data"``: a ``data:application/vnd.spectrl;v=1,…`` URI (``base`` ignored).

    ``base`` is required for ``"fragment"`` and ``"query"``. See
    :func:`spectrl.encode_spectrum` for ``lossless`` / ``max_len``.
    """
    _require_spectrl()
    from spectrl import to_data_uri, to_fragment, to_query

    # Validate before the (potentially expensive) encode so bad args fail fast.
    if mode not in ("fragment", "query", "data"):
        raise ValueError(f"unknown mode {mode!r}; expected 'fragment', 'query', or 'data'")
    if mode != "data" and base is None:
        raise ValueError(f"base URL is required for mode={mode!r}")

    token = to_spectrl_token(spec, lossless=lossless, max_len=max_len)
    if mode == "data":
        return to_data_uri(token)
    assert base is not None  # guaranteed by the validation above
    if mode == "fragment":
        return to_fragment(token, base)
    return to_query(token, base, param=param)


def from_spectrl_url(url: str) -> Spectrum:
    """Extract a ``spectrl1.…`` token from a URL fragment, query string, or
    ``data:`` URI and decode it into a spxtacular
    :class:`~spxtacular.core.Spectrum` / :class:`~spxtacular.core.MsnSpectrum`.
    """
    _require_spectrl()
    from spectrl import extract_token

    return from_spectrl_token(extract_token(url))
