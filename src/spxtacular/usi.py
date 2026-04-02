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
from typing import Any

import numpy as np

_PROXI_BACKENDS: dict[str, str] = {
    "aggregator": "https://proteomecentral.proteomexchange.org/api/proxi/v0.1/spectra",
    "pride": "https://www.ebi.ac.uk/pride/proxi/archive/v0.1/spectra",
    "massive": "https://massive.ucsd.edu/proxi/v0.1/spectra",
    "peptideatlas": "https://peptideatlas.org/api/proxi/v0.1/spectra",
    "jpost": "https://repository.jpostdb.org/proxi/spectra",
}


def _parse_proxi_response(
    data: list[dict[str, Any]],
    usi: str,
) -> dict[str, Any]:
    """Extract m/z, intensity, and precursor info from a PROXI response."""
    if not data:
        raise ValueError(f"Empty PROXI response for USI: {usi}")

    spectrum = data[0]

    mzs = spectrum.get("mzs") or spectrum.get("m/z array")
    intensities = spectrum.get("intensities") or spectrum.get("intensity array")

    if mzs is None or intensities is None:
        raise ValueError(
            f"PROXI response missing m/z or intensity data for USI: {usi}"
        )

    result: dict[str, Any] = {
        "mz": np.array(mzs, dtype=np.float64),
        "intensity": np.array(intensities, dtype=np.float64),
    }

    # Extract precursor attributes
    attributes = spectrum.get("attributes", [])
    for attr in attributes:
        accession = attr.get("accession", "")
        value = attr.get("value")
        if value is None:
            continue

        if accession in ("MS:1000827", "MS:1000744", "MS:1002234"):
            result.setdefault("precursor_mz", float(value))
        elif accession == "MS:1000041":
            result.setdefault("precursor_charge", int(value))

    return result


def fetch_usi(
    usi: str,
    backend: str = "aggregator",
    timeout: float = 30,
) -> Any:
    """Fetch a spectrum from a public repository via USI.

    Uses the PROXI protocol to retrieve spectra from aggregated proteomics
    repositories.

    Parameters
    ----------
    usi:
        Universal Spectrum Identifier, e.g.
        ``"mzspec:PXD000561:Adult_Frontalcortex_bRP_Elite_85_f09:scan:17555"``.
    backend:
        PROXI backend to query. Options: ``"aggregator"`` (default),
        ``"pride"``, ``"massive"``, ``"peptideatlas"``, ``"jpost"``.
        A full URL can also be provided directly.
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

    # Resolve backend URL
    if backend in _PROXI_BACKENDS:
        base_url = _PROXI_BACKENDS[backend]
    elif backend.startswith("http"):
        base_url = backend
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            f"Available: {', '.join(_PROXI_BACKENDS)}"
        )

    encoded_usi = urllib.parse.quote_plus(usi)
    url = f"{base_url}?resultType=full&usi={encoded_usi}"

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            data = json.loads(raw)
    except urllib.error.HTTPError as e:
        raise ValueError(
            f"HTTP {e.code} error fetching USI: {usi}. {e.reason}"
        ) from e
    except urllib.error.URLError as e:
        raise ValueError(
            f"Network error fetching USI: {usi}. {e.reason}"
        ) from e
    except (TimeoutError, OSError) as e:
        raise ValueError(
            f"Timeout fetching USI: {usi}. {e}"
        ) from e
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON response for USI: {usi}. {e}"
        ) from e

    parsed = _parse_proxi_response(data, usi)
    mz = parsed["mz"]
    intensity = parsed["intensity"]
    prec_mz = parsed.get("precursor_mz")
    prec_charge = parsed.get("precursor_charge")

    if prec_mz is not None:
        precursor = Precursor(
            mz=prec_mz,
            intensity=0.0,
            charge=prec_charge,
            is_monoisotopic=None,
        )
        return MsnSpectrum(
            mz=mz,
            intensity=intensity,
            precursors=[precursor],
            ms_level=2,
        )

    return Spectrum(mz=mz, intensity=intensity)
