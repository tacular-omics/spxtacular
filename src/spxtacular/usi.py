"""
Load spectra from public repositories via Universal Spectrum Identifier (USI).

Uses the PROXI (PROteomics eXpression Interface) REST API to fetch spectra
from aggregated repositories (PRIDE, MassIVE, PeptideAtlas, jPOST).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from .core import MsnSpectrum, Spectrum

_PROXI_BACKENDS: dict[str, str] = {
    "aggregator": "https://proteomecentral.proteomexchange.org/api/proxi/v0.1/spectra",
    "pride": "https://www.ebi.ac.uk/pride/proxi/archive/v0.1/spectra",
    "massive": "https://massive.ucsd.edu/proxi/v0.1/spectra",
    "peptideatlas": "https://peptideatlas.org/api/proxi/v0.1/spectra",
    "jpost": "https://repository.jpostdb.org/proxi/spectra",
}

# The four index flags defined by the USI specification, keyed by their
# lowercase form so parsing can be case-insensitive while still normalising to
# the canonical spelling ("nativeId" is camel-case in the spec).
_INDEX_FLAGS: dict[str, str] = {
    "scan": "scan",
    "index": "index",
    "nativeid": "nativeId",
    "trace": "trace",
}

# Precursor m/z accessions in decreasing order of trustworthiness. The selected
# ion m/z is the measured precursor; the isolation window target is only the
# centre of the quadrupole window and can be off by half the window width, so it
# must never win over a real selected-ion value that is also present.
_PRECURSOR_MZ_ACCESSIONS: tuple[str, ...] = (
    "MS:1000744",  # selected ion m/z
    "MS:1002234",  # selected precursor m/z
    "MS:1000827",  # isolation window target m/z
)

_CHARGE_ACCESSION = "MS:1000041"  # charge state
_PRECURSOR_INTENSITY_ACCESSION = "MS:1000042"  # peak intensity
_MS_LEVEL_ACCESSION = "MS:1000511"  # ms level


class ParsedUsi(NamedTuple):
    """The components of a Universal Spectrum Identifier.

    ``mzspec:<collection>:<run>:<index_flag>:<index>[:<interpretation>]``
    """

    collection: str
    run: str
    index_flag: str
    index: str
    interpretation: str | None = None

    @property
    def scan_number(self) -> int | None:
        """The integer spectrum identifier, for ``scan`` / ``index`` USIs."""
        if self.index_flag in ("scan", "index"):
            return int(self.index)
        return None

    @property
    def native_id(self) -> str:
        """An mzML-style native ID for the referenced spectrum."""
        if self.index_flag in ("nativeId", "trace"):
            return self.index
        return f"{self.index_flag}={self.index}"


def parse_usi(usi: str) -> ParsedUsi:
    """Parse and validate a Universal Spectrum Identifier.

    Parameters
    ----------
    usi:
        Universal Spectrum Identifier, e.g.
        ``"mzspec:PXD000561:Adult_Frontalcortex_bRP_Elite_85_f09:scan:17555"``.

    Returns
    -------
    ParsedUsi
        The ``collection``, ``run``, ``index_flag``, ``index`` and optional
        ``interpretation`` components.

    Raises
    ------
    ValueError
        If the USI is not of the form
        ``mzspec:<collection>:<run>:<scan|index|nativeId|trace>:<n>[:<interpretation>]``.
    """
    if not isinstance(usi, str) or not usi.strip():
        raise ValueError("USI must be a non-empty string")

    parts = usi.strip().split(":", 5)
    if len(parts) < 5:
        raise ValueError(
            f"Invalid USI: {usi!r}. Expected "
            "'mzspec:<collection>:<run>:<scan|index|nativeId|trace>:<n>[:<interpretation>]'"
        )

    prefix, collection, run, index_flag, index = parts[:5]
    interpretation = parts[5] if len(parts) > 5 and parts[5] else None

    if prefix != "mzspec":
        raise ValueError(f"Invalid USI: {usi!r}. Must start with the 'mzspec' prefix, got {prefix!r}")
    if not collection:
        raise ValueError(f"Invalid USI: {usi!r}. Collection identifier is empty")
    if not run:
        raise ValueError(f"Invalid USI: {usi!r}. Run identifier is empty")

    canonical_flag = _INDEX_FLAGS.get(index_flag.lower())
    if canonical_flag is None:
        raise ValueError(
            f"Invalid USI: {usi!r}. Unknown index flag {index_flag!r}; "
            f"expected one of {', '.join(_INDEX_FLAGS.values())}"
        )
    if not index:
        raise ValueError(f"Invalid USI: {usi!r}. Index value is empty")
    if canonical_flag in ("scan", "index"):
        try:
            value = int(index)
        except ValueError as e:
            raise ValueError(f"Invalid USI: {usi!r}. {canonical_flag} index must be an integer, got {index!r}") from e
        if value < 0:
            raise ValueError(f"Invalid USI: {usi!r}. {canonical_flag} index must be non-negative, got {index!r}")

    return ParsedUsi(
        collection=collection,
        run=run,
        index_flag=canonical_flag,
        index=index,
        interpretation=interpretation,
    )


def _as_float(value: Any, accession: str, usi: str) -> float:
    """Coerce a PROXI attribute value to float, raising with USI context."""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"PROXI attribute {accession} has non-numeric value {value!r} for USI: {usi}") from None


def _as_int(value: Any, accession: str, usi: str) -> int:
    """Coerce a PROXI attribute value to int, raising with USI context."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        raise ValueError(f"PROXI attribute {accession} has non-integer value {value!r} for USI: {usi}") from None


def _parse_proxi_response(
    data: Any,
    usi: str,
) -> dict[str, Any]:
    """Extract m/z, intensity, and precursor info from a PROXI response."""
    if isinstance(data, dict):
        raise ValueError(
            f"Unexpected PROXI response for USI: {usi}. Expected a list of spectra, got a JSON object with keys: "
            f"{', '.join(sorted(map(str, data)))}"
        )
    if not isinstance(data, list):
        raise ValueError(
            f"Unexpected PROXI response for USI: {usi}. Expected a list of spectra, got {type(data).__name__}"
        )
    if not data:
        raise ValueError(f"Empty PROXI response for USI: {usi}")

    spectrum = data[0]
    if not isinstance(spectrum, dict):
        raise ValueError(
            f"Unexpected PROXI response for USI: {usi}. Expected a spectrum object, got {type(spectrum).__name__}"
        )

    # Explicit None checks: an empty-but-valid spectrum has ``"mzs": []``, which
    # is falsy but is *not* missing data.
    mzs = spectrum.get("mzs")
    if mzs is None:
        mzs = spectrum.get("m/z array")
    intensities = spectrum.get("intensities")
    if intensities is None:
        intensities = spectrum.get("intensity array")

    if mzs is None or intensities is None:
        raise ValueError(f"PROXI response missing m/z or intensity data for USI: {usi}")
    if len(mzs) != len(intensities):
        raise ValueError(
            f"PROXI response has {len(mzs)} m/z values but {len(intensities)} intensity values for USI: {usi}"
        )

    result: dict[str, Any] = {
        "mz": np.array(mzs, dtype=np.float64),
        "intensity": np.array(intensities, dtype=np.float64),
    }

    # Extract precursor attributes. Candidates are collected first so a fixed
    # precedence can be applied — the order the server happens to list them in
    # must not decide which m/z wins.
    mz_candidates: dict[str, float] = {}
    attributes = spectrum.get("attributes") or []
    for attr in attributes:
        if not isinstance(attr, dict):
            continue
        accession = attr.get("accession", "")
        value = attr.get("value")
        if value is None:
            continue

        if accession in _PRECURSOR_MZ_ACCESSIONS:
            mz_candidates.setdefault(accession, _as_float(value, accession, usi))
        elif accession == _CHARGE_ACCESSION and "precursor_charge" not in result:
            result["precursor_charge"] = _as_int(value, accession, usi)
        elif accession == _PRECURSOR_INTENSITY_ACCESSION and "precursor_intensity" not in result:
            result["precursor_intensity"] = _as_float(value, accession, usi)
        elif accession == _MS_LEVEL_ACCESSION and "ms_level" not in result:
            result["ms_level"] = _as_int(value, accession, usi)

    for accession in _PRECURSOR_MZ_ACCESSIONS:
        if accession in mz_candidates:
            result["precursor_mz"] = mz_candidates[accession]
            break

    return result


def fetch_usi(
    usi: str,
    backend: str = "aggregator",
    timeout: float = 30,
) -> Spectrum | MsnSpectrum:
    """Fetch a spectrum from a public repository via USI.

    Uses the PROXI protocol to retrieve spectra from aggregated proteomics
    repositories.

    Parameters
    ----------
    usi:
        Universal Spectrum Identifier, e.g.
        ``"mzspec:PXD000561:Adult_Frontalcortex_bRP_Elite_85_f09:scan:17555"``.
        Validated locally by :func:`parse_usi` before any request is made.
    backend:
        PROXI backend to query. Options: ``"aggregator"`` (default),
        ``"pride"``, ``"massive"``, ``"peptideatlas"``, ``"jpost"``.
        A full ``http://`` / ``https://`` URL can also be provided directly.
    timeout:
        HTTP request timeout in seconds.

    Returns
    -------
    Spectrum or MsnSpectrum
        :class:`~spxtacular.core.MsnSpectrum` if precursor information is
        available, otherwise :class:`~spxtacular.core.Spectrum`.

    Raises
    ------
    ValueError
        If the USI is invalid, the server returns an error, or the response
        is missing required data.
    """
    from .core import MsnSpectrum, Precursor, Spectrum

    parsed = parse_usi(usi)

    # Resolve backend URL
    if backend in _PROXI_BACKENDS:
        base_url = _PROXI_BACKENDS[backend]
    elif backend.startswith(("http://", "https://")):
        base_url = backend
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Available: {', '.join(_PROXI_BACKENDS)}")

    encoded_usi = urllib.parse.quote_plus(usi)
    url = f"{base_url}?resultType=full&usi={encoded_usi}"

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            data = json.loads(raw)
    except urllib.error.HTTPError as e:
        raise ValueError(f"HTTP {e.code} error fetching USI: {usi}. {e.reason}") from e
    except urllib.error.URLError as e:
        raise ValueError(f"Network error fetching USI: {usi}. {e.reason}") from e
    except (TimeoutError, OSError) as e:
        raise ValueError(f"Timeout fetching USI: {usi}. {e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response for USI: {usi}. {e}") from e

    response_data = _parse_proxi_response(data, usi)
    mz = response_data["mz"]
    intensity = response_data["intensity"]
    prec_mz = response_data.get("precursor_mz")
    prec_charge = response_data.get("precursor_charge")
    prec_intensity = response_data.get("precursor_intensity")
    ms_level = response_data.get("ms_level")

    if prec_mz is not None:
        precursor = Precursor(
            mz=prec_mz,
            # PROXI responses rarely carry a precursor intensity; 0.0 marks it
            # as unmeasured rather than genuinely zero.
            intensity=prec_intensity if prec_intensity is not None else 0.0,
            charge=prec_charge,
            is_monoisotopic=None,
        )
        return MsnSpectrum(
            mz=mz,
            intensity=intensity,
            precursors=[precursor],
            # Only assume MS2 when the server didn't say and a precursor exists.
            ms_level=ms_level if ms_level is not None else 2,
            scan_number=parsed.scan_number,
            native_id=parsed.native_id,
        )

    return Spectrum(mz=mz, intensity=intensity)
