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

Install ``spxtacular[spectrl]`` to enable. The module always imports — the
missing extra is reported at call time instead, so ``spxtacular`` stays
importable without it; every public function here raises
:class:`ImportError` when invoked and ``spectrl`` is not installed.

Per-peak ``iso_score`` is carried through spectrl's ``extra_arrays`` slot
under the key ``"iso_score"`` (encoded as a non-standard mzML binary array,
``MS:1000786``). spxtacular-specific scalar fields without an mzML
counterpart — ``denoised``/``normalized`` provenance strings,
``scan_number``, ``resolution``, ``analyzer``, ``ramp_time``, ``im_range``,
``isolation_im_range``, and each precursor's ``is_monoisotopic`` — are carried
losslessly as namespaced free-text ``user_params`` (``spxtacular:`` prefix), so
the round-trip is faithful.

Note that :func:`to_spectrl_token` / :func:`to_spectrl_url` are **lossy by
default** (MS-Numpress peak compression); pass ``lossless=True`` for a bit-exact
round-trip of the peak arrays.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import numpy as np

from .core import MsnSpectrum, Precursor, Spectrum, SpectrumType
from .enums import ActivationType, Analyzer, IMType, Polarity
from .ionization import DeconvolutionProvenance

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
_UNIT_MILLISECOND = "UO:0000028"
# spxtacular always writes scan start time in seconds, but minutes is the more
# common convention in mzML-derived tokens, so decode has to honour the unit.
_UNIT_MINUTE = "UO:0000031"
_INJECTION_TIME = "MS:1000927"  # ion injection time
# Per the official PSI-MS ontology MS:1000500 is the scan window *upper* limit
# and MS:1000501 the *lower* limit (the names read counter-intuitively).
_SCAN_WINDOW_LOWER = "MS:1000501"
_SCAN_WINDOW_UPPER = "MS:1000500"

# Precursor / isolation / activation
_SELECTED_ION_MZ = "MS:1000744"
_CHARGE_STATE = "MS:1000041"
_PEAK_INTENSITY = "MS:1000042"
_ISOL_TARGET_MZ = "MS:1000827"
_ISOL_LOWER_OFFSET = "MS:1000828"
_ISOL_UPPER_OFFSET = "MS:1000829"
_COLLISION_ENERGY = "MS:1000045"

# Scalar ion-selection ion-mobility terms. These are the correct params for a
# *selected ion* — unlike the MS:10030xx family, which are binary *array* terms
# describing a per-peak column and are meaningless on a scalar precursor.
_SELECTED_ION_OOK0 = "MS:1002815"  # inverse reduced ion mobility
_SELECTED_ION_DRIFT_TIME = "MS:1002476"  # ion mobility drift time

# Shape of a PSI-MS CV accession (e.g. "MS:1002481"). Used to tell an already-
# valid accession apart from free-text so the former can be passed to spectrl.
_MS_ACCESSION_RE = re.compile(r"^MS:\d{7}$")

# Common activation accessions, drawn from the PSI-MS "dissociation method"
# branch (MS:1000044). Unrecognised strings are NOT valid CV accessions
# (spectrl's accession_tail() requires an "MS:NNNNN"-shaped string), so they are
# never passed to spectrl as one; instead they're carried losslessly via the
# _UP_ACTIVATION_TYPE user_param below.
#
# "PASEF" has no PSI-MS term of its own — it's a Bruker timsTOF acquisition
# scheme (ion-mobility-gated precursor selection), not a dissociation
# mechanism. Bruker PASEF spectra are fragmented via higher energy beam-type
# CID, so it's mapped to that accession here; this matches DReader, which
# already writes MS:1002481 directly for PASEF spectra (see reader.py).
# Keyed by :class:`spxtacular.enums.ActivationType` members (the enum is the
# single source of truth for the acronyms) so the two can't drift apart. Since
# StrEnum members are strings, ``.get(activation_type)`` works whether the field
# holds an enum member or an equivalent raw string.
_ACTIVATION_ACCESSIONS: dict[str, str] = {
    ActivationType.CID: "MS:1000133",  # collision-induced dissociation
    ActivationType.HCD: "MS:1000422",  # beam-type collision-induced dissociation
    ActivationType.ETD: "MS:1000598",  # electron transfer dissociation
    ActivationType.ECD: "MS:1000250",  # electron capture dissociation
    ActivationType.ETHCD: "MS:1002631",  # electron-transfer/higher-energy collision dissociation
    ActivationType.ETCID: "MS:1003182",  # electron-transfer/collision-induced dissociation
    ActivationType.NETD: "MS:1003247",  # negative electron transfer dissociation
    ActivationType.UVPD: "MS:1003246",  # ultraviolet photodissociation
    ActivationType.PD: "MS:1000435",  # photodissociation
    ActivationType.PQD: "MS:1000599",  # pulsed q dissociation
    ActivationType.SID: "MS:1000136",  # surface-induced dissociation
    ActivationType.IRMPD: "MS:1000262",  # infrared multiphoton dissociation
    ActivationType.BIRD: "MS:1000242",  # blackbody infrared radiative dissociation
    ActivationType.SORI: "MS:1000282",  # sustained off-resonance irradiation
    ActivationType.PASEF: "MS:1002481",  # higher energy beam-type CID (Bruker PASEF); see note above
}

# Case-insensitive encode lookup so ``activation_type="hcd"`` still finds its
# accession (the field is an open vocabulary, so producers write any casing).
_ACTIVATION_ACCESSIONS_LOWER: dict[str, str] = {k.lower(): v for k, v in _ACTIVATION_ACCESSIONS.items()}

# Inverse mapping for decoding (accession -> ActivationType member). PASEF is
# encode-only: MS:1002481 is the generic "higher energy beam-type CID" term used
# by Thermo HCD data too, so decoding it as PASEF would mislabel non-Bruker
# spectra. It decodes to the dissociation method it actually names.
_ACTIVATION_NAMES: dict[str, str] = {v: k for k, v in _ACTIVATION_ACCESSIONS.items() if k != ActivationType.PASEF}
_ACTIVATION_NAMES["MS:1002481"] = ActivationType.HCD

# Mass-analyzer accessions, from the PSI-MS "mass analyzer type" (MS:1000443)
# branch and keyed by :class:`spxtacular.enums.Analyzer` members. Analyzer is a
# mzML instrument-configuration concept with no per-spectrum/scan slot in
# spectrl's single-spectrum model, so the bridge carries the exact string
# losslessly via the _UP_ANALYZER user_param below; this dict is the canonical
# enum -> accession reference for consumers that need the CV term.
_ANALYZER_ACCESSIONS: dict[str, str] = {
    Analyzer.ORBITRAP: "MS:1000484",
    Analyzer.FT_ICR: "MS:1000079",  # fourier transform ion cyclotron resonance
    Analyzer.TOF: "MS:1000084",  # time-of-flight
    Analyzer.QUADRUPOLE: "MS:1000081",
    Analyzer.ION_TRAP: "MS:1000264",
    Analyzer.LINEAR_ION_TRAP: "MS:1000291",
    Analyzer.QUADRUPOLE_ION_TRAP: "MS:1000082",
    Analyzer.MAGNETIC_SECTOR: "MS:1000080",
    Analyzer.ELECTROSTATIC_ENERGY_ANALYZER: "MS:1000254",
}

# Ion-mobility array accessions (must be drawn from spectrl.ION_MOBILITY_ARRAY_TAILS,
# i.e. mzmlpy.constants.ION_MOBILITIES, otherwise spectrl will not recognise them on decode).
# Keyed by :class:`spxtacular.enums.IMType` members where one exists; the extra
# lowercase aliases ("1/k0", "drift_time") stay accepted as raw strings.
# PSI-MS has no collision-cross-section array term, so ``ccs`` is encoded with
# the generic ion-mobility array accession and its exact type is preserved by the
# _UP_IM_TYPE user_param below.
_IM_TYPE_ACCESSIONS: dict[str, str] = {
    IMType.OOK0: "MS:1003008",  # raw inverse reduced ion mobility (1/K0)
    "1/k0": "MS:1003008",
    IMType.IM: "MS:1002893",  # generic ion mobility
    "drift_time": "MS:1003153",
    IMType.DRIFT_TIME_MS: "MS:1003153",
    IMType.CCS: "MS:1002893",  # generic ion mobility; exact type kept in _UP_IM_TYPE
}

# Reverse lookup for decoding (covers raw/mean/deconvoluted variants). Note that
# MS:1003007 is the *generic* raw ion mobility array, not CCS — a foreign token
# carrying it says nothing about the units beyond "ion mobility".
_IM_TYPE_FROM_ACCESSION: dict[str, str] = {
    "MS:1003008": IMType.OOK0,  # raw inverse reduced ion mobility (1/K0)
    "MS:1003006": IMType.OOK0,  # mean inverse reduced ion mobility
    "MS:1003155": IMType.OOK0,  # deconvoluted inverse reduced ion mobility
    "MS:1002816": IMType.OOK0,  # mean ion mobility (fall back to ook0)
    "MS:1002893": IMType.IM,
    "MS:1003007": IMType.IM,  # raw ion mobility (generic)
    "MS:1003154": IMType.IM,  # deconvoluted ion mobility (generic)
    "MS:1003153": IMType.DRIFT_TIME_MS,
    "MS:1002477": IMType.DRIFT_TIME_MS,
    "MS:1003156": IMType.DRIFT_TIME_MS,
}

# Ion-mobility accessions accepted on a *selected ion* when decoding: the correct
# scalar terms plus the binary-array terms older spxtacular releases wrote there.
_PRECURSOR_IM_ACCESSIONS: frozenset[str] = frozenset(
    {_SELECTED_ION_OOK0, _SELECTED_ION_DRIFT_TIME} | set(_IM_TYPE_FROM_ACCESSION)
)

_POLARITY_FROM_ACCESSION: dict[str, Polarity] = {
    _POSITIVE: Polarity.POSITIVE,
    _NEGATIVE: Polarity.NEGATIVE,
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
_UP_DECONVOLUTION = _UP_PREFIX + "deconvolution"
_UP_SCAN_NUMBER = _UP_PREFIX + "scan_number"
_UP_RESOLUTION = _UP_PREFIX + "resolution"
_UP_ANALYZER = _UP_PREFIX + "analyzer"
_UP_RAMP_TIME = _UP_PREFIX + "ramp_time"
_UP_IM_RANGE_LO = _UP_PREFIX + "im_range_lower"
_UP_IM_RANGE_HI = _UP_PREFIX + "im_range_upper"
_UP_ISOL_IM_RANGE_LO = _UP_PREFIX + "isolation_im_range_lower"
_UP_ISOL_IM_RANGE_HI = _UP_PREFIX + "isolation_im_range_upper"
_UP_IM_TYPE = _UP_PREFIX + "im_type"
_UP_ACTIVATION_TYPE = _UP_PREFIX + "activation_type"
# Per-precursor flag, suffixed with the precursor's index (``…is_monoisotopic.0``)
# since spectrl's user_params slot is spectrum-level.
_UP_PREC_MONOISOTOPIC = _UP_PREFIX + "precursor_is_monoisotopic"


def _up_prec_monoisotopic(index: int) -> str:
    """Name of the ``is_monoisotopic`` user_param for precursor ``index``."""
    return f"{_UP_PREC_MONOISOTOPIC}.{index}"


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
        _UP_ACTIVATION_TYPE,
    }
)


# ---------------------------------------------------------------------------
# Module-level guard
# ---------------------------------------------------------------------------


def _require_spectrl() -> None:
    if not _HAS_SPECTRL:
        raise ImportError("spectrl is required for this operation. Install with: pip install spxtacular[spectrl]")


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
    ``mz_range``, ``injection_time`` (``MS:1000927``), ``precursors`` (selected
    ion params, plus the spectrum-level isolation window / activation attached to
    the *first* precursor as mzML prescribes), and ``total_ion_current``.

    spxtacular scalar fields without an mzML CV counterpart —
    ``denoised``/``normalized`` provenance strings, ``scan_number``,
    ``resolution``, ``analyzer``, ``ramp_time``, ``im_range``,
    ``isolation_im_range``, and each precursor's ``is_monoisotopic`` — are
    carried losslessly as namespaced free-text ``user_params`` (see the
    ``spxtacular:`` prefixed names above).
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
        # Open vocabulary: normalise casing so "POSITIVE" isn't silently dropped.
        polarity_key = str(msn_spec.polarity).lower() if msn_spec.polarity is not None else None
        if polarity_key == Polarity.POSITIVE:
            params.append(_cv(_POSITIVE))
        elif polarity_key == Polarity.NEGATIVE:
            params.append(_cv(_NEGATIVE))
        if msn_spec.total_ion_current is not None:
            params.append(_cv(_TIC, value=float(msn_spec.total_ion_current)))

    # Scan(s) — one entry holding RT and the m/z scan window
    scans: list[SpectrlScan] = []
    if msn_spec is not None:
        scan_params = []
        if msn_spec.rt is not None:
            scan_params.append(_cv(_SCAN_START_TIME, value=float(msn_spec.rt), unit_accession=_UNIT_SECOND))
        if msn_spec.injection_time is not None:
            scan_params.append(
                _cv(_INJECTION_TIME, value=float(msn_spec.injection_time), unit_accession=_UNIT_MILLISECOND)
            )
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

    # Ion mobility array accession for the per-peak column.
    ion_mobility_type: str | None = None
    if spec.im is not None and msn_spec is not None and msn_spec.im_type is not None:
        ion_mobility_type = _IM_TYPE_ACCESSIONS.get(msn_spec.im_type.lower(), "MS:1002893")
    elif spec.im is not None:
        ion_mobility_type = "MS:1002893"  # generic ion mobility when type unknown

    # Scalar ion-selection accession for precursor ion mobility. Drift-time types
    # get MS:1002476; everything else (including an unknown type) gets the
    # inverse-reduced-mobility term, which is what both readers measure.
    im_type_key = msn_spec.im_type.lower() if msn_spec is not None and msn_spec.im_type is not None else None
    precursor_im_accession = (
        _SELECTED_ION_DRIFT_TIME if im_type_key in ("drift_time", IMType.DRIFT_TIME_MS) else _SELECTED_ION_OOK0
    )

    # Precursors
    precursors: list[SpectrlPrecursor] = []
    if msn_spec is not None and msn_spec.precursors:
        for prec_index, prec in enumerate(msn_spec.precursors):
            # The isolation window, collision energy and activation are stored
            # per-spectrum by spxtacular but are per-precursor in mzML. Replicating
            # them onto every precursor would assert something false about a
            # multi-precursor spectrum, so only the first precursor carries them.
            is_first = prec_index == 0
            iw: SpectrlIsolationWindow | None = None
            if is_first and msn_spec.isolation_mz_range is not None:
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
                ion_params.append(_cv(precursor_im_accession, value=float(prec.im)))
            selected_ion = SpectrlSelectedIon(params=ion_params)

            activation: SpectrlActivation | None = None
            act_params = []
            if is_first and msn_spec.collision_energy is not None:
                act_params.append(_cv(_COLLISION_ENERGY, value=float(msn_spec.collision_energy)))
            if is_first and msn_spec.activation_type is not None:
                # Emit a standard dissociation-method CV param when we can: either the
                # value is a known acronym (mapped to its accession) or it is already an
                # ``MS:NNNNNNN``-shaped accession (as both readers produce). Only truly
                # free-text vendor strings fall through to the _UP_ACTIVATION_TYPE
                # user_param below, since spectrl's accession_tail() would reject them.
                acc = _ACTIVATION_ACCESSIONS_LOWER.get(msn_spec.activation_type.lower())
                if acc is None and _MS_ACCESSION_RE.match(msn_spec.activation_type):
                    acc = msn_spec.activation_type
                if acc is not None:
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
        charge_arr = spec.charge.astype(np.float64, copy=False)

    # iso_score travels as a non-standard mzML binary array (MS:1000786)
    # via spectrl's extra_arrays slot.
    extra_arrays: dict[str, np.ndarray] = {}
    if spec.iso_score is not None:
        extra_arrays["iso_score"] = spec.iso_score.astype(np.float64, copy=False)

    # spxtacular scalar fields with no CV term travel as namespaced user_params.
    user_params = []
    if spec.denoised is not None:
        user_params.append(_up(_UP_DENOISED, spec.denoised, "xsd:string"))
    if spec.normalized is not None:
        user_params.append(_up(_UP_NORMALIZED, spec.normalized, "xsd:string"))
    if spec.deconvolution is not None:
        provenance_json = json.dumps(spec.deconvolution.to_dict(), separators=(",", ":"))
        user_params.append(_up(_UP_DECONVOLUTION, provenance_json, "xsd:string"))
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
        if msn_spec.activation_type is not None:
            # The CV accession (added to act_params above) only covers activation types in
            # _ACTIVATION_ACCESSIONS; stash the exact string here so unrecognized types
            # round-trip too instead of being dropped or crashing the encode.
            user_params.append(_up(_UP_ACTIVATION_TYPE, msn_spec.activation_type, "xsd:string"))
        if msn_spec.precursors:
            # mzML has no term for "this selected ion is the monoisotopic peak"
            # (as opposed to the most intense one), so it rides along per index.
            for prec_index, prec in enumerate(msn_spec.precursors):
                if prec.is_monoisotopic is not None:
                    user_params.append(_up(_up_prec_monoisotopic(prec_index), int(prec.is_monoisotopic), "xsd:boolean"))

    return InlineSpectrum(
        default_array_length=n,
        mz=spec.mz.astype(np.float64, copy=False) if spec.mz is not None else None,
        intensity=spec.intensity.astype(np.float64, copy=False) if spec.intensity is not None else None,
        charge=charge_arr,
        ion_mobility=spec.im.astype(np.float64, copy=False) if spec.im is not None else None,
        ion_mobility_type=ion_mobility_type,
        id=msn_spec.native_id if msn_spec is not None else None,
        params=params,
        scans=scans,
        precursors=precursors,
        extra_arrays=extra_arrays,
        user_params=user_params,
    )


def to_spectrl_token(spec: Spectrum, *, lossless: bool = False, max_len: int | None = None) -> str:
    """Encode a spxtacular spectrum directly to a ``spectrl2.…`` token.

    Convenience wrapper over :func:`to_inline_spectrum` +
    :func:`spectrl.encode_spectrum`. See :func:`spectrl.encode_spectrum` for
    the meaning of ``lossless`` and ``max_len``.

    .. warning::

       The **default encoding is lossy**: peak arrays go through MS-Numpress,
       which round-trips ``mz`` to roughly ``3.1e-8`` relative error,
       ``intensity`` to roughly ``1.3e-4``, and ``im`` to roughly ``6.7e-6``.
       That is well inside instrument precision for sharing and plotting, but it
       is not bit-exact. Pass ``lossless=True`` for a bit-exact round-trip (at
       the cost of a longer token).
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
    ``native_id``, ``ms_level``, ``polarity``, ``rt``, ``injection_time``,
    ``mz_range``, ``total_ion_current``, ``precursors`` (with ``charge``,
    ``intensity``, ``im``, and ``is_monoisotopic``), ``isolation_mz_range``,
    ``collision_energy``, ``activation_type``, and ``im_type``.

    spxtacular scalar fields carried as namespaced ``user_params`` —
    ``denoised``/``normalized`` (also restored on a plain :class:`Spectrum`),
    ``scan_number``, ``resolution``, ``analyzer``, ``ramp_time``, ``im_range``,
    ``isolation_im_range``, per-precursor ``is_monoisotopic`` — are restored
    too; the MSn-only ones force an :class:`MsnSpectrum` even when no other MSn
    metadata is present.
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
    deconvolution_v = up.get(_UP_DECONVOLUTION)
    deconvolution = None
    if deconvolution_v is not None:
        try:
            deconvolution_data = json.loads(str(deconvolution_v))
            if not isinstance(deconvolution_data, dict):
                raise TypeError("expected a JSON object")
            deconvolution = DeconvolutionProvenance.from_dict(deconvolution_data)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("invalid spxtacular deconvolution provenance in spectrl token") from exc

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
            deconvolution=deconvolution,
        )

    # MSn metadata
    ms_level: int | None = int(ms_level_p.value) if ms_level_p is not None and ms_level_p.value is not None else None

    polarity: Polarity | None = None
    for accession, name in _POLARITY_FROM_ACCESSION.items():
        if _find_param(decoded.params, accession):
            polarity = name
            break

    tic_p = _find_param(decoded.params, _TIC)
    total_ion_current = float(tic_p.value) if tic_p is not None and tic_p.value is not None else None

    # First scan (if any) holds rt + injection_time + mz_range
    rt: float | None = None
    injection_time: float | None = None
    mz_range: tuple[float, float] | None = None
    if decoded.scans:
        scan = decoded.scans[0]
        rt_p = _find_param(scan.params, _SCAN_START_TIME)
        if rt_p is not None and rt_p.value is not None:
            # MsnSpectrum.rt is seconds. Our own encode tags the param with
            # _UNIT_SECOND, but a foreign token may carry minutes instead.
            rt = float(rt_p.value) * (60.0 if rt_p.unit_accession == _UNIT_MINUTE else 1.0)
        it_p = _find_param(scan.params, _INJECTION_TIME)
        if it_p is not None and it_p.value is not None:
            injection_time = float(it_p.value)
        for window in scan.windows:
            lo_p = _find_param(window.params, _SCAN_WINDOW_LOWER)
            hi_p = _find_param(window.params, _SCAN_WINDOW_UPPER)
            # A foreign token may carry the term with no value at all.
            if lo_p is not None and lo_p.value is not None and hi_p is not None and hi_p.value is not None:
                mz_range = (float(lo_p.value), float(hi_p.value))
                break

    # Precursors
    precursors: list[Precursor] | None = None
    isolation_mz_range: tuple[float, float] | None = None
    collision_energy: float | None = None
    activation_type_v = up.get(_UP_ACTIVATION_TYPE)
    activation_type: str | None = str(activation_type_v) if activation_type_v is not None else None
    if decoded.precursors:
        precursors = []
        for prec_index, sp in enumerate(decoded.precursors):
            # Keyed by precursor index, mirroring the encode side. Keying by the
            # number of Precursors built so far would drift on a foreign token
            # whose precursor carries several selected ions, or one whose
            # selected ion is skipped below for a missing m/z.
            mono_v = up.get(_up_prec_monoisotopic(prec_index))
            is_monoisotopic = bool(int(mono_v)) if mono_v is not None else None
            for ion in sp.selected_ions:
                mz_p = _find_param(ion.params, _SELECTED_ION_MZ)
                if mz_p is None or mz_p.value is None:
                    continue
                charge_p = _find_param(ion.params, _CHARGE_STATE)
                intensity_p = _find_param(ion.params, _PEAK_INTENSITY)
                prec_intensity = (
                    float(intensity_p.value) if intensity_p is not None and intensity_p.value is not None else 0.0
                )
                prec_charge = int(charge_p.value) if charge_p is not None and charge_p.value is not None else None
                prec_im: float | None = None
                for p in ion.params:
                    if p.accession in _PRECURSOR_IM_ACCESSIONS and p.value is not None:
                        prec_im = float(p.value)
                        break
                precursors.append(
                    Precursor(
                        mz=float(mz_p.value),
                        intensity=prec_intensity,
                        charge=prec_charge,
                        im=prec_im,
                        is_monoisotopic=is_monoisotopic,
                    )
                )
            if sp.isolation_window is not None and isolation_mz_range is None:
                target_p = _find_param(sp.isolation_window.params, _ISOL_TARGET_MZ)
                lo_p = _find_param(sp.isolation_window.params, _ISOL_LOWER_OFFSET)
                hi_p = _find_param(sp.isolation_window.params, _ISOL_UPPER_OFFSET)
                if (
                    target_p is not None
                    and target_p.value is not None
                    and lo_p is not None
                    and lo_p.value is not None
                    and hi_p is not None
                    and hi_p.value is not None
                ):
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
        injection_time=injection_time,
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
        deconvolution=deconvolution,
        scan_number=scan_number,
        resolution=resolution,
        analyzer=analyzer,
        ramp_time=ramp_time,
        im_range=im_range,
        isolation_im_range=isolation_im_range,
    )


def from_spectrl_token(token: str) -> Spectrum:
    """Decode a ``spectrl2.…`` token into a spxtacular
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

    - ``"fragment"`` (default): ``base#spectrl2.…`` — the token rides in the URL
      fragment, which is never sent to the server (no length limits, no logs).
    - ``"query"``: ``base?<param>=spectrl2.…`` — token as a query parameter.
    - ``"data"``: a ``data:application/vnd.spectrl;v=2,…`` URI (``base`` ignored).

    ``base`` is required for ``"fragment"`` and ``"query"``. See
    :func:`spectrl.encode_spectrum` for ``lossless`` / ``max_len``.

    .. warning::

       As with :func:`to_spectrl_token`, the **default encoding is lossy**:
       ``mz`` round-trips to roughly ``3.1e-8`` relative error, ``intensity`` to
       roughly ``1.3e-4``, and ``im`` to roughly ``6.7e-6``. Pass
       ``lossless=True`` for a bit-exact round-trip.
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
    """Extract a ``spectrl2.…`` token from a URL fragment, query string, or
    ``data:`` URI and decode it into a spxtacular
    :class:`~spxtacular.core.Spectrum` / :class:`~spxtacular.core.MsnSpectrum`.
    """
    _require_spectrl()
    from spectrl import extract_token

    return from_spectrl_token(extract_token(url))
