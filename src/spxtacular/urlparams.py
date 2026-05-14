"""Encode/decode :class:`~spxtacular.core.Spectrum` to/from URL query parameters.

The peak arrays (``mz``, ``intensity``, ``charge``, ``im``) are routed through
:func:`~spxtacular.compress.compress_spectra` with ``url_safe=True`` and emitted
under the ``spectrum`` key. All :class:`~spxtacular.core.MsnSpectrum` scalar
metadata is emitted as separate, human-readable, plain query parameters. Fields
that are ``None`` are omitted to keep URLs short.

The wire format is versioned via the ``version`` parameter (currently ``"1"``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Literal
from urllib.parse import parse_qsl, urlencode

import numpy as np

from .compress import compress_spectra, decompress_spectra
from .core import MsnSpectrum, Precursor, Spectrum, SpectrumType

SCHEMA_VERSION = "1"

_MSN_SCALAR_STR_KEYS = (
    "native_id",
    "im_type",
    "polarity",
    "analyzer",
    "activation_type",
)
_MSN_SCALAR_INT_KEYS = ("scan_number", "ms_level")
_MSN_SCALAR_FLOAT_KEYS = (
    "rt",
    "injection_time",
    "total_ion_current",
    "resolution",
    "ramp_time",
    "collision_energy",
)
_MSN_TUPLE_KEYS = ("mz_range", "im_range", "isolation_mz_range", "isolation_im_range")
_MSN_ALL_KEYS = (
    *_MSN_SCALAR_STR_KEYS,
    *_MSN_SCALAR_INT_KEYS,
    *_MSN_SCALAR_FLOAT_KEYS,
    *_MSN_TUPLE_KEYS,
    "precursors",
)
_BASE_FLAG_KEYS = ("spectrum_type", "denoised", "normalized")


def spectrum_to_query_params(
    spectrum: Spectrum,
    *,
    max_peaks: int | None = None,
    select_by: Literal["intensity", "mz"] = "intensity",
    mz_precision: int | None = None,
    intensity_precision: int | None = None,
    im_precision: int | None = None,
    compression: Literal["gzip", "zlib", "brotli"] = "gzip",
) -> dict[str, str]:
    """Encode a spectrum as a dict of URL query parameters.

    Parameters
    ----------
    spectrum:
        The spectrum to encode. ``MsnSpectrum`` metadata is included as
        separate plain parameters when set.
    max_peaks:
        If set, retain at most this many peaks (top-N by ``select_by``).
        Trimming preserves the per-peak ``charge``, ``im``, and ``iso_score``
        arrays. Peaks are re-sorted by ``mz`` after trimming.
    select_by:
        Which attribute to use when picking top peaks: ``"intensity"`` (default)
        or ``"mz"``.
    mz_precision, intensity_precision, im_precision:
        Rounding precision passed to :func:`compress_spectra`.
    compression:
        Compression method passed to :func:`compress_spectra`.

    Returns
    -------
    dict[str, str]
        Query-param dict. Use :func:`urllib.parse.urlencode` (or
        :func:`spectrum_to_query_string`) to render to a URL string.
    """
    if max_peaks is not None and max_peaks < 0:
        raise ValueError("max_peaks must be non-negative")

    if max_peaks is not None and len(spectrum.mz) > max_peaks:
        if select_by == "intensity":
            sort_idx = spectrum._argsort_intensity
        elif select_by == "mz":
            sort_idx = spectrum._argsort_mz
        else:
            raise ValueError(f"select_by must be 'intensity' or 'mz', got {select_by!r}")
        keep = sort_idx[-max_peaks:]
        keep = keep[np.argsort(spectrum.mz[keep])]
        spectrum = spectrum._apply_index(keep)

    params: dict[str, str] = {"version": SCHEMA_VERSION}
    params["spectrum"] = compress_spectra(
        spectrum,
        url_safe=True,
        mz_precision=mz_precision,
        intensity_precision=intensity_precision,
        im_precision=im_precision,
        compression=compression,
    )

    for key in _BASE_FLAG_KEYS:
        val = getattr(spectrum, key, None)
        if val is None:
            continue
        if isinstance(val, SpectrumType):
            params[key] = val.value
        else:
            params[key] = str(val)

    if isinstance(spectrum, MsnSpectrum):
        for key in _MSN_SCALAR_STR_KEYS:
            val = getattr(spectrum, key)
            if val is not None:
                params[key] = str(val)
        for key in _MSN_SCALAR_INT_KEYS:
            val = getattr(spectrum, key)
            if val is not None:
                params[key] = str(int(val))
        for key in _MSN_SCALAR_FLOAT_KEYS:
            val = getattr(spectrum, key)
            if val is not None:
                params[key] = repr(float(val))
        for key in _MSN_TUPLE_KEYS:
            val = getattr(spectrum, key)
            if val is not None:
                lo, hi = val
                params[key] = f"{repr(float(lo))},{repr(float(hi))}"
        if spectrum.precursors:
            params["precursors"] = json.dumps(
                [_precursor_to_dict(p) for p in spectrum.precursors],
                separators=(",", ":"),
            )

    return params


def spectrum_to_query_string(spectrum: Spectrum, **kwargs) -> str:
    """Encode a spectrum as a URL query string (no leading ``?``)."""
    return urlencode(spectrum_to_query_params(spectrum, **kwargs))


def spectrum_from_query_params(params: Mapping[str, str] | str) -> Spectrum:
    """Decode a spectrum from URL query parameters.

    Accepts either a mapping (e.g. ``request.args`` from a web framework) or a
    raw query string (with or without a leading ``?``). Returns an
    :class:`MsnSpectrum` if any MSn-specific metadata is present, otherwise a
    :class:`Spectrum`.
    """
    if isinstance(params, str):
        raw = params.lstrip("?")
        parsed = dict(parse_qsl(raw, keep_blank_values=True))
    else:
        parsed = dict(params)

    version = parsed.get("version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported urlparams version: {version!r} (expected {SCHEMA_VERSION!r})")

    compressed = parsed.get("spectrum")
    if not compressed:
        raise ValueError("Missing 'spectrum' parameter")

    base = decompress_spectra(compressed)

    extra: dict = {}
    for key in _BASE_FLAG_KEYS:
        if key in parsed:
            extra[key] = parsed[key]

    is_msn = any(key in parsed for key in _MSN_ALL_KEYS)
    if not is_msn:
        return Spectrum(
            mz=base.mz,
            intensity=base.intensity,
            charge=base.charge,
            im=base.im,
            iso_score=base.iso_score,
            **extra,
        )

    msn_kwargs: dict = {}
    for key in _MSN_SCALAR_STR_KEYS:
        if key in parsed:
            msn_kwargs[key] = parsed[key]
    for key in _MSN_SCALAR_INT_KEYS:
        if key in parsed:
            msn_kwargs[key] = int(parsed[key])
    for key in _MSN_SCALAR_FLOAT_KEYS:
        if key in parsed:
            msn_kwargs[key] = float(parsed[key])
    for key in _MSN_TUPLE_KEYS:
        if key in parsed:
            lo_s, hi_s = parsed[key].split(",", 1)
            msn_kwargs[key] = (float(lo_s), float(hi_s))
    if "precursors" in parsed:
        msn_kwargs["precursors"] = [_precursor_from_dict(d) for d in json.loads(parsed["precursors"])]

    return MsnSpectrum(
        mz=base.mz,
        intensity=base.intensity,
        charge=base.charge,
        im=base.im,
        iso_score=base.iso_score,
        **extra,
        **msn_kwargs,
    )


def _precursor_to_dict(p: Precursor) -> dict:
    return {
        "mz": p.mz,
        "intensity": p.intensity,
        "charge": p.charge,
        "im": p.im,
        "iso_score": p.iso_score,
        "is_monoisotopic": p.is_monoisotopic,
    }


def _precursor_from_dict(d: dict) -> Precursor:
    return Precursor(
        mz=d["mz"],
        intensity=d["intensity"],
        charge=d.get("charge"),
        im=d.get("im"),
        iso_score=d.get("iso_score"),
        is_monoisotopic=d.get("is_monoisotopic"),
    )
