"""
Tests for spxtacular.usi -- USI parsing, PROXI response handling and fetching.

Every request is mocked: these tests must never touch the network. The mocking
follows the pattern already used for ``Spectrum.from_usi`` -- patch
``spxtacular.usi.urllib.request.urlopen`` and hand back a context-manager-shaped
response object.
"""

from __future__ import annotations

import http.client
import json
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from spxtacular.core import MsnSpectrum, Spectrum, SpectrumType
from spxtacular.enums import Polarity
from spxtacular.usi import _parse_proxi_response, fetch_usi, spectrum_from_proxi_response

USI = "mzspec:TEST:test:scan:1"

SPECTRUM_ENTRY: dict[str, Any] = {
    "mzs": [100.0, 200.0, 300.0],
    "intensities": [10.0, 20.0, 30.0],
    "attributes": [{"accession": "MS:1000744", "value": "500.25"}],
}


def _response(payload: Any) -> MagicMock:
    """A urlopen return value shaped like an HTTPResponse context manager."""
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode()
    response.__enter__ = lambda s: s
    response.__exit__ = MagicMock(return_value=False)
    return response


# ---------------------------------------------------------------------------
# Picking the spectrum out of a PROXI response
# ---------------------------------------------------------------------------


class TestParseProxiResponse:
    def test_takes_the_first_entry_carrying_peaks(self) -> None:
        """An aggregator can lead with a stub for the repositories that missed."""
        stub = {"status": "ERROR", "usi": USI, "error": "not found in this repository"}
        parsed = _parse_proxi_response([stub, SPECTRUM_ENTRY], USI)
        np.testing.assert_allclose(parsed["mz"], [100.0, 200.0, 300.0])
        assert parsed["precursor_mz"] == pytest.approx(500.25)

    def test_skips_a_leading_non_dict_entry(self) -> None:
        parsed = _parse_proxi_response(["nonsense", SPECTRUM_ENTRY], USI)
        np.testing.assert_allclose(parsed["intensity"], [10.0, 20.0, 30.0])

    def test_first_entry_still_wins_when_several_carry_peaks(self) -> None:
        other = {"mzs": [1.0], "intensities": [2.0]}
        parsed = _parse_proxi_response([other, SPECTRUM_ENTRY], USI)
        np.testing.assert_allclose(parsed["mz"], [1.0])

    def test_an_empty_but_valid_spectrum_is_not_skipped(self) -> None:
        """``"mzs": []`` is falsy but is a real answer, not missing data."""
        parsed = _parse_proxi_response([{"mzs": [], "intensities": []}, SPECTRUM_ENTRY], USI)
        assert len(parsed["mz"]) == 0

    def test_error_when_no_entry_carries_peaks(self) -> None:
        with pytest.raises(ValueError, match="missing m/z or intensity data"):
            _parse_proxi_response([{"status": "ERROR"}, {"usi": USI}], USI)

    def test_error_when_entries_are_not_objects(self) -> None:
        with pytest.raises(ValueError, match="Expected a spectrum object"):
            _parse_proxi_response(["nonsense"], USI)

    def test_empty_list_still_reports_an_empty_response(self) -> None:
        with pytest.raises(ValueError, match="Empty PROXI response"):
            _parse_proxi_response([], USI)

    @pytest.mark.parametrize(
        ("attribute", "expected"),
        [
            ({"accession": "MS:1000127"}, "centroid"),
            ({"accession": "MS:1000128"}, "profile"),
            ({"accession": "MS:1000525", "value": "profile spectrum"}, "profile"),
        ],
    )
    def test_spectrum_representation_is_parsed(self, attribute: dict[str, str], expected: str) -> None:
        entry = {"mzs": [100.0], "intensities": [10.0], "attributes": [attribute]}
        assert _parse_proxi_response([entry], USI)["spectrum_type"] == expected

    @pytest.mark.parametrize(
        ("attribute", "expected"),
        [
            ({"accession": "MS:1000130"}, "positive"),
            ({"accession": "MS:1000129"}, "negative"),
            ({"accession": "MS:1000465", "value": "negative scan"}, "negative"),
        ],
    )
    def test_scan_polarity_is_parsed(self, attribute: dict[str, str], expected: str) -> None:
        entry = {"mzs": [100.0], "intensities": [10.0], "attributes": [attribute]}
        assert _parse_proxi_response([entry], USI)["polarity"] == expected

    def test_dense_data_without_representation_is_inferred_as_profile(self) -> None:
        mz = np.linspace(100.0, 101.0, 201)
        entry = {"mzs": mz.tolist(), "intensities": np.ones_like(mz).tolist()}
        assert _parse_proxi_response([entry], USI)["spectrum_type"] == "profile"

    def test_conflicting_polarities_are_rejected(self) -> None:
        entry = {
            "mzs": [100.0],
            "intensities": [10.0],
            "attributes": [{"accession": "MS:1000130"}, {"accession": "MS:1000129"}],
        }
        with pytest.raises(ValueError, match="conflicting scan polarities"):
            _parse_proxi_response([entry], USI)


class TestSpectrumFromProxiResponse:
    def test_public_parser_preserves_representation_and_polarity(self) -> None:
        entry = {
            "mzs": [100.0],
            "intensities": [10.0],
            "attributes": [
                {"accession": "MS:1000128"},
                {"accession": "MS:1000129"},
                {"accession": "MS:1000511", "value": "1"},
            ],
        }
        result = spectrum_from_proxi_response([entry], USI)
        assert isinstance(result, MsnSpectrum)
        assert result.spectrum_type == SpectrumType.PROFILE
        assert result.polarity == Polarity.NEGATIVE
        assert result.ms_level == 1
        assert result.scan_number == 1

    def test_public_parser_returns_plain_spectrum_without_msn_metadata(self) -> None:
        entry = {"mzs": [100.0], "intensities": [10.0], "attributes": [{"accession": "MS:1000127"}]}
        result = spectrum_from_proxi_response([entry], USI)
        assert type(result) is Spectrum
        assert result.spectrum_type == SpectrumType.CENTROID


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------


class TestRequestUrl:
    def _capture(self, backend: str) -> str:
        seen: list[str] = []

        def _urlopen(req: Any, timeout: float | None = None) -> MagicMock:
            seen.append(req.full_url)
            return _response([SPECTRUM_ENTRY])

        with patch("spxtacular.usi.urllib.request.urlopen", side_effect=_urlopen):
            fetch_usi(USI, backend=backend)
        return seen[0]

    def test_plain_backend_url_starts_the_query_string(self) -> None:
        url = self._capture("https://example.org/proxi/v0.1/spectra")
        assert url.startswith("https://example.org/proxi/v0.1/spectra?resultType=full&usi=")

    def test_backend_url_with_a_query_string_appends_to_it(self) -> None:
        """A double '?' makes the whole query unparseable server-side."""
        url = self._capture("https://example.org/proxi/v0.1/spectra?apikey=abc")
        assert url.count("?") == 1
        assert url == "https://example.org/proxi/v0.1/spectra?apikey=abc&resultType=full&usi=" + (
            "mzspec%3ATEST%3Atest%3Ascan%3A1"
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestFetchErrors:
    def test_http_error_response_body_is_closed(self) -> None:
        from email.message import Message
        from io import BytesIO
        from urllib.error import HTTPError

        body = BytesIO(b"Not found")
        error = HTTPError("https://example.org/spectrum", 404, "Not Found", Message(), body)
        with patch("spxtacular.usi.urllib.request.urlopen", side_effect=error):
            with pytest.raises(ValueError, match="HTTP 404"):
                fetch_usi(USI)
        assert body.closed

    def test_incomplete_read_is_wrapped_in_valueerror(self) -> None:
        """http.client exceptions are not OSErrors, so they escaped unwrapped."""
        response = _response([SPECTRUM_ENTRY])
        response.read.side_effect = http.client.IncompleteRead(b"partial")

        with patch("spxtacular.usi.urllib.request.urlopen", return_value=response):
            with pytest.raises(ValueError, match="Malformed HTTP response"):
                fetch_usi(USI)

    def test_timeout_is_reported_as_a_timeout(self) -> None:
        with patch("spxtacular.usi.urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            with pytest.raises(ValueError, match="Timeout fetching USI"):
                fetch_usi(USI)

    def test_other_oserrors_are_not_called_timeouts(self) -> None:
        with patch(
            "spxtacular.usi.urllib.request.urlopen",
            side_effect=ConnectionResetError("connection reset by peer"),
        ):
            with pytest.raises(ValueError, match="Network error fetching USI") as excinfo:
                fetch_usi(USI)
        assert "Timeout" not in str(excinfo.value)

    def test_invalid_json_is_wrapped(self) -> None:
        response = MagicMock()
        response.read.return_value = b"<html>not json</html>"
        response.__enter__ = lambda s: s
        response.__exit__ = MagicMock(return_value=False)

        with patch("spxtacular.usi.urllib.request.urlopen", return_value=response):
            with pytest.raises(ValueError, match="Invalid JSON response"):
                fetch_usi(USI)


# ---------------------------------------------------------------------------
# Successful fetch
# ---------------------------------------------------------------------------


class TestFetchUsi:
    def test_precursor_attributes_give_an_msn_spectrum(self) -> None:
        with patch("spxtacular.usi.urllib.request.urlopen", return_value=_response([SPECTRUM_ENTRY])):
            result = fetch_usi(USI)
        assert isinstance(result, MsnSpectrum)
        assert result.precursors is not None
        assert result.precursors[0].mz == pytest.approx(500.25)
        assert result.scan_number == 1

    def test_without_precursor_information_a_plain_spectrum_comes_back(self) -> None:
        entry = {"mzs": [100.0], "intensities": [10.0]}
        with patch("spxtacular.usi.urllib.request.urlopen", return_value=_response([entry])):
            result = fetch_usi(USI)
        assert type(result) is Spectrum

    def test_a_stub_entry_ahead_of_the_spectrum_is_survived(self) -> None:
        payload = [{"status": "ERROR", "usi": USI}, SPECTRUM_ENTRY]
        with patch("spxtacular.usi.urllib.request.urlopen", return_value=_response(payload)):
            result = fetch_usi(USI)
        assert len(result.mz) == 3
