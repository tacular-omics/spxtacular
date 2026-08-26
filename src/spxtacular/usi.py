"""
Load spectra from public repositories via Universal Spectrum Identifier (USI).

Uses the PROXI (PROteomics eXpression Interface) REST API to fetch spectra
from aggregated repositories (PRIDE, MassIVE, PeptideAtlas, jPOST).
"""

from __future__ import annotations

import http.client
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
_CENTROID_ACCESSION = "MS:1000127"  # centroid spectrum
_PROFILE_ACCESSION = "MS:1000128"  # profile spectrum
_SPECTRUM_REPRESENTATION_ACCESSION = "MS:1000525"  # spectrum representation
_POSITIVE_POLARITY_ACCESSION = "MS:1000130"  # positive scan
_NEGATIVE_POLARITY_ACCESSION = "MS:1000129"  # negative scan
_SCAN_POLARITY_ACCESSION = "MS:1000465"  # scan polarity


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


def _peak_arrays(entry: Any) -> tuple[Any, Any] | None:
    """The ``(m/z, intensity)`` pair of a PROXI entry, or ``None`` if it has neither.

    Explicit ``None`` checks: an empty-but-valid spectrum has ``"mzs": []``,
    which is falsy but is *not* missing data.
    """
    if not isinstance(entry, dict):
        return None

    mzs = entry.get("mzs")
    if mzs is None:
        mzs = entry.get("m/z array")
    intensities = entry.get("intensities")
    if intensities is None:
        intensities = entry.get("intensity array")

    if mzs is None or intensities is None:
        return None
    return mzs, intensities


def _parse_proxi_response(
    data: Any,
    usi: str,
) -> dict[str, Any]:
    """Extract peaks and spectrum metadata from a PROXI response."""
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

    # Not necessarily ``data[0]``: an aggregator that queries several repositories
    # can lead with a stub or error entry for the ones that missed and carry the
    # real spectrum further down the list. Take the first entry that actually has
    # peak arrays, and only complain if none of them does.
    found: tuple[dict[str, Any], Any, Any] | None = None
    for entry in data:
        arrays = _peak_arrays(entry)
        if arrays is not None:
            found = (entry, arrays[0], arrays[1])
            break

    if found is None:
        first = data[0]
        if not isinstance(first, dict):
            raise ValueError(
                f"Unexpected PROXI response for USI: {usi}. Expected a spectrum object, got {type(first).__name__}"
            )
        raise ValueError(f"PROXI response missing m/z or intensity data for USI: {usi}")

    spectrum, mzs, intensities = found
    if len(mzs) != len(intensities):
        raise ValueError(
            f"PROXI response has {len(mzs)} m/z values but {len(intensities)} intensity values for USI: {usi}"
        )

    result: dict[str, Any] = {
        "mz": np.array(mzs, dtype=np.float64),
        "intensity": np.array(intensities, dtype=np.float64),
    }

    # Extract spectrum and precursor attributes. Candidates are collected first so a fixed
    # precedence can be applied — the order the server happens to list them in
    # must not decide which m/z wins.
    mz_candidates: dict[str, float] = {}
    explicit_spectrum_types: set[str] = set()
    generic_spectrum_types: set[str] = set()
    explicit_polarities: set[str] = set()
    generic_polarities: set[str] = set()
    attributes = spectrum.get("attributes") or []
    for attr in attributes:
        if not isinstance(attr, dict):
            continue
        accession = attr.get("accession", "")
        value = attr.get("value")

        if accession == _CENTROID_ACCESSION:
            explicit_spectrum_types.add("centroid")
        elif accession == _PROFILE_ACCESSION:
            explicit_spectrum_types.add("profile")
        elif accession == _SPECTRUM_REPRESENTATION_ACCESSION:
            representation = str(value or attr.get("name", "")).lower()
            if "centroid" in representation:
                generic_spectrum_types.add("centroid")
            elif "profile" in representation:
                generic_spectrum_types.add("profile")

        if accession == _POSITIVE_POLARITY_ACCESSION:
            explicit_polarities.add("positive")
        elif accession == _NEGATIVE_POLARITY_ACCESSION:
            explicit_polarities.add("negative")
        elif accession == _SCAN_POLARITY_ACCESSION:
            polarity = str(value or attr.get("name", "")).lower()
            if "positive" in polarity:
                generic_polarities.add("positive")
            elif "negative" in polarity:
                generic_polarities.add("negative")

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

    spectrum_types = explicit_spectrum_types or generic_spectrum_types
    if len(spectrum_types) > 1:
        raise ValueError(f"PROXI response has conflicting spectrum representations for USI: {usi}")
    if spectrum_types:
        result["spectrum_type"] = next(iter(spectrum_types))
    else:
        # Some PROXI providers omit the representation CV term. A long run of
        # tightly spaced samples is characteristic of profile data; otherwise
        # use the conventional centroid representation. Centralising this
        # fallback keeps browser and native clients consistent.
        ordered_mz = np.sort(result["mz"])
        positive_diffs = np.diff(ordered_mz)
        positive_diffs = positive_diffs[positive_diffs > 0]
        is_profile = len(positive_diffs) >= 200 and float(np.mean(positive_diffs < 0.05)) >= 0.8
        result["spectrum_type"] = "profile" if is_profile else "centroid"

    polarities = explicit_polarities or generic_polarities
    if len(polarities) > 1:
        raise ValueError(f"PROXI response has conflicting scan polarities for USI: {usi}")
    if polarities:
        result["polarity"] = next(iter(polarities))

    return result


def spectrum_from_proxi_response(data: Any, usi: str) -> Spectrum | MsnSpectrum:
    """Create a spectrum from a decoded PROXI JSON response.

    This pure parser is useful when an application performs the HTTP request
    itself (for example, asynchronously in a browser). Spectrum representation
    and scan polarity CV terms are preserved when present.
    """
    from .core import MsnSpectrum, Precursor, Spectrum
    from .enums import Polarity

    parsed_usi = parse_usi(usi)
    response_data = _parse_proxi_response(data, usi)
    mz = response_data["mz"]
    intensity = response_data["intensity"]
    prec_mz = response_data.get("precursor_mz")
    prec_charge = response_data.get("precursor_charge")
    prec_intensity = response_data.get("precursor_intensity")
    ms_level = response_data.get("ms_level")
    spectrum_type = response_data.get("spectrum_type")
    polarity_value = response_data.get("polarity")
    polarity = Polarity(polarity_value) if polarity_value is not None else None

    precursor = None
    if prec_mz is not None:
        precursor = Precursor(
            mz=prec_mz,
            # PROXI responses rarely carry a precursor intensity; 0.0 marks it
            # as unmeasured rather than genuinely zero.
            intensity=prec_intensity if prec_intensity is not None else 0.0,
            charge=prec_charge,
            is_monoisotopic=None,
        )

    # MsnSpectrum owns scan-level metadata. Construct one whenever PROXI
    # supplied any such metadata, even if the response has no precursor.
    if precursor is not None or ms_level is not None or polarity is not None:
        return MsnSpectrum(
            mz=mz,
            intensity=intensity,
            spectrum_type=spectrum_type,
            precursors=[precursor] if precursor is not None else None,
            # Only assume MS2 when the server omitted the level but supplied a precursor.
            ms_level=ms_level if ms_level is not None else (2 if precursor is not None else None),
            scan_number=parsed_usi.scan_number,
            native_id=parsed_usi.native_id,
            polarity=polarity,
        )

    return Spectrum(mz=mz, intensity=intensity, spectrum_type=spectrum_type)


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
        :class:`~spxtacular.core.MsnSpectrum` when scan-level metadata or
        precursor information is available, otherwise
        :class:`~spxtacular.core.Spectrum`.

    Raises
    ------
    ValueError
        If the USI is invalid, the server returns an error, or the response
        is missing required data.
    """
    parse_usi(usi)

    # Resolve backend URL
    if backend in _PROXI_BACKENDS:
        base_url = _PROXI_BACKENDS[backend]
    elif backend.startswith(("http://", "https://")):
        base_url = backend
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Available: {', '.join(_PROXI_BACKENDS)}")

    encoded_usi = urllib.parse.quote_plus(usi)
    # A custom backend URL may already carry a query string (an API key, say), in
    # which case the parameters have to be appended rather than started.
    separator = "&" if "?" in base_url else "?"
    url = f"{base_url}{separator}resultType=full&usi={encoded_usi}"

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            data = json.loads(raw)
    except urllib.error.HTTPError as e:
        raise ValueError(f"HTTP {e.code} error fetching USI: {usi}. {e.reason}") from e
    except urllib.error.URLError as e:
        raise ValueError(f"Network error fetching USI: {usi}. {e.reason}") from e
    except TimeoutError as e:
        # socket.timeout has been an alias of TimeoutError since 3.10, and a read
        # that overruns ``timeout`` raises it directly rather than via URLError.
        raise ValueError(f"Timeout fetching USI: {usi}. {e}") from e
    except http.client.HTTPException as e:
        # IncompleteRead and friends come out of response.read() and are not
        # OSErrors, so without this they would escape the documented ValueError.
        raise ValueError(f"Malformed HTTP response for USI: {usi}. {e}") from e
    except OSError as e:
        # A connection reset or a DNS failure is not a timeout; say what it was.
        raise ValueError(f"Network error fetching USI: {usi}. {e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response for USI: {usi}. {e}") from e

    return spectrum_from_proxi_response(data, usi)
