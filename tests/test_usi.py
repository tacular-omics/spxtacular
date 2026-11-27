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
from spxtacular.errors import SpxtacularError
from spxtacular.usi import _parse_proxi_response, fetch_usi, parse_usi, spectrum_from_proxi_response

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
        assert result.polarity == "negative"
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
        assert result.precursors[0].precursor_mz == pytest.approx(500.25)
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


# ---------------------------------------------------------------------------
# parse_usi: run names that contain colons (USI 1.0 section 3.3.4)
# ---------------------------------------------------------------------------

# Real USIs quoted in the USI 1.0 specification (HUPO-PSI/usi,
# Specification/USI_FullSpecification_1.0.0_final.pdf), with the components it gives for each.
SPEC_USIS = [
    (
        "mzspec:PXD000561:Adult_Frontalcortex_bRP_Elite_85_f09:scan:17555:VLHPLEGAVVIIFK/2",
        ("PXD000561", "Adult_Frontalcortex_bRP_Elite_85_f09", "scan", "17555", "VLHPLEGAVVIIFK/2"),
    ),
    (
        "mzspec:PXD005175:CRC_iTRAQ_06:scan:11803:VEYTLGEESEAPGQR/3",
        ("PXD005175", "CRC_iTRAQ_06", "scan", "11803", "VEYTLGEESEAPGQR/3"),
    ),
    (
        "mzspec:PXD007592:good_responder_1_2.mgf:index:22627:TLM+15.994915TQIDGVNLAANSLVESGHPR/3",
        ("PXD007592", "good_responder_1_2.mgf", "index", "22627", "TLM+15.994915TQIDGVNLAANSLVESGHPR/3"),
    ),
    (
        "mzspec:PXD001464:CL_1hRP_rep3:nativeId:1,1,2740,10",
        ("PXD001464", "CL_1hRP_rep3", "nativeId", "1,1,2740,10", None),
    ),
    (
        "mzspec:PXD001587:18300_REP2_500ng_HumanLysate_SWATH_1:scan:4974:M[+15.994915]SAEDIEK",
        ("PXD001587", "18300_REP2_500ng_HumanLysate_SWATH_1", "scan", "4974", "M[+15.994915]SAEDIEK"),
    ),
    (
        # Provenance suffix (section 4): everything after the index stays in the interpretation.
        "mzspec:PXD000561:Adult_Frontalcortex_bRP_Elite_85_f09:scan:17555:VLHPLEGAVVIIFK/2:PR-G47",
        ("PXD000561", "Adult_Frontalcortex_bRP_Elite_85_f09", "scan", "17555", "VLHPLEGAVVIIFK/2:PR-G47"),
    ),
]


@pytest.mark.parametrize(("usi", "expected"), SPEC_USIS)
def test_parse_usi_specification_examples(usi: str, expected: tuple[str, ...]) -> None:
    assert tuple(parse_usi(usi)) == expected


def test_parse_usi_run_name_with_colons() -> None:
    parsed = parse_usi("mzspec:PXD000561:run:with:colon:scan:17555:VLHPLEGAVVIIFK/2")
    assert parsed.collection == "PXD000561"
    assert parsed.run == "run:with:colon"
    assert parsed.index_flag == "scan"
    assert parsed.index == "17555"
    assert parsed.interpretation == "VLHPLEGAVVIIFK/2"
    assert parsed.scan_number == 17555


@pytest.mark.parametrize(
    ("usi", "run", "flag", "index", "interpretation"),
    [
        ("mzspec:PXD1:a:b:index:7", "a:b", "index", "7", None),
        ("mzspec:PXD1:a:b:nativeId:1,1,2740,10", "a:b", "nativeId", "1,1,2740,10", None),
        ("mzspec:PXD1:a:b:trace:3", "a:b", "trace", "3", None),
        ("mzspec:PXD1:a:b:NativeID:1,2", "a:b", "nativeId", "1,2", None),  # flags stay case-insensitive
        # A run segment that looks like a flag but is not followed by a valid index stays in the run.
        ("mzspec:PXD1:x:scan:y:scan:12", "x:scan:y", "scan", "12", None),
        # Colons in the interpretation are kept there.
        ("mzspec:PXD1:a:b:scan:5:PEP[UNIMOD:35]TIDE/2", "a:b", "scan", "5", "PEP[UNIMOD:35]TIDE/2"),
    ],
)
def test_parse_usi_colon_run_variants(usi: str, run: str, flag: str, index: str, interpretation: str | None) -> None:
    parsed = parse_usi(usi)
    assert (parsed.run, parsed.index_flag, parsed.index, parsed.interpretation) == (run, flag, index, interpretation)


@pytest.mark.parametrize(
    ("usi", "run"),
    [
        # Section 3.6.1: a [subFolder] prefix may contain colons, even ones that look like an index.
        ("mzspec:PXD123456:[CaCo01]A01_100ng:scan:5", "[CaCo01]A01_100ng"),
        ("mzspec:PXD123456:[Ca:Co01]A01_100ng:scan:5", "[Ca:Co01]A01_100ng"),
        ("mzspec:PXD123456:[a:scan:1:x]A01_100ng:scan:5", "[a:scan:1:x]A01_100ng"),
        ("mzspec:PXD123456:[a:b]run:with:colon:scan:5", "[a:b]run:with:colon"),
    ],
)
def test_parse_usi_subfolder_with_colons(usi: str, run: str) -> None:
    parsed = parse_usi(usi)
    assert parsed.run == run
    assert (parsed.index_flag, parsed.index) == ("scan", "5")


@pytest.mark.parametrize(
    ("usi", "message"),
    [
        ("mzspec:PXD1:run:x:bogus:5", "Unknown index flag 'x'"),
        ("mzspec:PXD1:run:x:scan:abc", "Unknown index flag 'x'"),
        ("mzspec:PXD1:run:scan:abc", "scan index must be an integer"),
        ("mzspec:PXD1:run:scan:-1", "scan index must be non-negative"),
        ("mzspec:PXD1:run:scan:", "Index value is empty"),
        ("mzspec:PXD1::scan:5", "Run identifier is empty"),
        ("mzspec::run:scan:5", "Collection identifier is empty"),
        ("mzspc:PXD1:run:scan:5", "'mzspec' prefix"),
        ("mzspec:PXD1:run", "Expected"),
    ],
)
def test_parse_usi_errors_unchanged(usi: str, message: str) -> None:
    with pytest.raises(SpxtacularError, match=message):
        parse_usi(usi)


def test_parse_usi_matches_the_plain_split_whenever_the_run_has_no_colon() -> None:
    """Every USI that parsed before (run without a colon) gives the same components as before."""
    for flag in ("scan", "index", "nativeId", "trace", "SCAN"):
        for run in ("r", "scan", "index", "run.mzML", "[sub]run", "a[b"):
            for index in ("0", "17555", "1,1,2740,10"):
                if flag.lower() in ("scan", "index") and not index.isdecimal():
                    continue
                for tail in ("", ":PEPTIDE/2", ":PEP[UNIMOD:35]TIDE/2:PR-G47", ":scan:5"):
                    usi = f"mzspec:PXD1:{run}:{flag}:{index}{tail}"
                    _, collection, plain_run, _, plain_index, *rest = usi.split(":", 5)
                    parsed = parse_usi(usi)
                    assert (parsed.collection, parsed.run, parsed.index) == (collection, plain_run, plain_index)
                    assert parsed.interpretation == (rest[0] if rest and rest[0] else None)
