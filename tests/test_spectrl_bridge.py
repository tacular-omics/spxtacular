"""Tests for the spxtacular ↔ spectrl bridge."""

import numpy as np
import pytest

spectrl = pytest.importorskip("spectrl")

from spectrl.model import (  # noqa: E402
    SpectrlCvParam,
    SpectrlPrecursor,
    SpectrlScan,
    SpectrlScanWindow,
    SpectrlSelectedIon,
    SpectrlUserParam,
)

from spxtacular.core import MsnSpectrum, Precursor, Spectrum, SpectrumType  # noqa: E402
from spxtacular.spectrl_bridge import (  # noqa: E402
    from_decoded_spectrum,
    from_spectrl_token,
    from_spectrl_url,
    to_inline_spectrum,
    to_spectrl_token,
    to_spectrl_url,
)

# Accessions / user-param names the foreign-token tests below build by hand,
# spelled out rather than imported so the tests pin the wire format itself.
_SCAN_START_TIME = "MS:1000016"
_UNIT_SECOND = "UO:0000010"
_UNIT_MINUTE = "UO:0000031"
# Per the PSI-MS ontology MS:1000501 is the scan window *lower* limit and
# MS:1000500 the *upper* limit (the term names read counter-intuitively).
_SCAN_WINDOW_LOWER = "MS:1000501"
_SCAN_WINDOW_UPPER = "MS:1000500"
_SELECTED_ION_MZ = "MS:1000744"
_UP_PREC_MONOISOTOPIC = "spxtacular:precursor_is_monoisotopic"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _basic_spectrum() -> Spectrum:
    return Spectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype=np.float64),
        intensity=np.array([1000.0, 5000.0, 2500.0], dtype=np.float64),
        charge=np.array([1, 2, -1], dtype=np.int32),
        spectrum_type=SpectrumType.DECONVOLUTED,
    )


def _basic_msn() -> MsnSpectrum:
    return MsnSpectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype=np.float64),
        intensity=np.array([1000.0, 5000.0, 2500.0], dtype=np.float64),
        spectrum_type=SpectrumType.CENTROID,
        scan_number=42,
        ms_level=2,
        native_id="scan=42",
        rt=125.5,
        polarity="positive",
        mz_range=(50.0, 1500.0),
        collision_energy=28.0,
        activation_type="HCD",
        total_ion_current=1.234e6,
        precursors=[Precursor(mz=500.25, intensity=8000.0, charge=2, is_monoisotopic=True)],
        isolation_mz_range=(498.75, 501.75),
    )


# ---------------------------------------------------------------------------
# to_inline_spectrum
# ---------------------------------------------------------------------------


def test_to_inline_carries_peak_arrays() -> None:
    inline = to_inline_spectrum(_basic_spectrum())
    np.testing.assert_array_equal(inline.mz, [100.0, 200.0, 300.0])
    np.testing.assert_array_equal(inline.intensity, [1000.0, 5000.0, 2500.0])
    # charge cast to float64 for spectrl's array model
    np.testing.assert_array_equal(inline.charge, [1.0, 2.0, -1.0])
    assert inline.default_array_length == 3


def test_to_inline_msn_metadata() -> None:
    inline = to_inline_spectrum(_basic_msn())
    assert inline.id == "scan=42"
    accessions = {p.accession for p in inline.params}
    assert "MS:1000511" in accessions  # ms level
    assert "MS:1000130" in accessions  # positive polarity
    assert "MS:1000127" in accessions  # centroid


def test_to_inline_precursor_isolation_window() -> None:
    inline = to_inline_spectrum(_basic_msn())
    assert len(inline.precursors) == 1
    prec = inline.precursors[0]
    iw = prec.isolation_window
    assert iw is not None
    iw_accessions = {p.accession: p.value for p in iw.params}
    assert iw_accessions["MS:1000827"] == pytest.approx(500.25)
    assert iw_accessions["MS:1000828"] == pytest.approx(1.5)
    assert iw_accessions["MS:1000829"] == pytest.approx(1.5)


def test_to_inline_activation() -> None:
    inline = to_inline_spectrum(_basic_msn())
    activation = inline.precursors[0].activation
    assert activation is not None
    accessions = {p.accession for p in activation.params}
    assert "MS:1000045" in accessions  # collision energy
    assert "MS:1000422" in accessions  # HCD


def test_to_inline_activation_raw_accession_emitted_as_cv() -> None:
    # DReader/MzmlReader populate activation_type with a raw PSI-MS accession, not an
    # acronym. It must still be emitted as a standard dissociation-method CV param on
    # the precursor activation (not merely stashed in the spxtacular:activation_type
    # user_param), otherwise external mzML tooling loses the dissociation method.
    msn = MsnSpectrum(
        mz=np.array([100.0, 200.0], dtype=np.float64),
        intensity=np.array([10.0, 20.0], dtype=np.float64),
        ms_level=2,
        activation_type="MS:1002481",  # beam-type CID (Bruker PASEF), as DReader writes
        precursors=[Precursor(mz=500.0, intensity=8000.0, charge=2, is_monoisotopic=True)],
    )
    inline = to_inline_spectrum(msn)
    activation = inline.precursors[0].activation
    assert activation is not None
    assert "MS:1002481" in {p.accession for p in activation.params}


def test_to_inline_freetext_activation_not_emitted_as_cv() -> None:
    # A non-accession, unknown vendor string must NOT be passed to spectrl as a CV
    # accession (it would fail accession_tail()); it round-trips via the user_param only.
    msn = MsnSpectrum(
        mz=np.array([100.0, 200.0], dtype=np.float64),
        intensity=np.array([10.0, 20.0], dtype=np.float64),
        ms_level=2,
        collision_energy=25.0,
        activation_type="MyCustomActivation",
        precursors=[Precursor(mz=500.0, intensity=8000.0, charge=2, is_monoisotopic=True)],
    )
    inline = to_inline_spectrum(msn)
    activation = inline.precursors[0].activation
    # only collision energy survives as a CV param; the free-text activation is not one
    assert activation is not None
    assert {p.accession for p in activation.params} == {"MS:1000045"}
    # but it still round-trips losslessly via the user_param
    restored = from_spectrl_token(to_spectrl_token(msn))
    assert isinstance(restored, MsnSpectrum)
    assert restored.activation_type == "MyCustomActivation"
    inline = to_inline_spectrum(_basic_spectrum())
    assert inline.id is None
    assert inline.scans == []
    assert inline.precursors == []


# ---------------------------------------------------------------------------
# Token round-trip
# ---------------------------------------------------------------------------


def test_token_roundtrip_plain_spectrum() -> None:
    spec = _basic_spectrum()
    token = to_spectrl_token(spec, lossless=True)
    assert token.startswith("spectrl1.")

    restored = from_spectrl_token(token)
    np.testing.assert_allclose(restored.mz, spec.mz)
    np.testing.assert_allclose(restored.intensity, spec.intensity)
    np.testing.assert_array_equal(restored.charge, spec.charge)


def test_token_roundtrip_msn_spectrum() -> None:
    spec = _basic_msn()
    token = to_spectrl_token(spec, lossless=True)
    restored = from_spectrl_token(token)

    assert isinstance(restored, MsnSpectrum)
    assert restored.native_id == "scan=42"
    assert restored.ms_level == 2
    assert restored.polarity == "positive"
    assert restored.rt == pytest.approx(125.5)
    assert restored.mz_range == pytest.approx((50.0, 1500.0))
    assert restored.total_ion_current == pytest.approx(1.234e6)
    assert restored.collision_energy == pytest.approx(28.0)
    assert restored.activation_type == "HCD"
    assert restored.isolation_mz_range == pytest.approx((498.75, 501.75))
    assert restored.precursors is not None and len(restored.precursors) == 1
    prec = restored.precursors[0]
    assert prec.mz == pytest.approx(500.25)
    assert prec.charge == 2
    assert prec.intensity == pytest.approx(8000.0)


def test_token_roundtrip_lossy_within_tolerance() -> None:
    spec = _basic_msn()
    token = to_spectrl_token(spec)  # default: lossy MS-Numpress
    restored = from_spectrl_token(token)
    np.testing.assert_allclose(restored.mz, spec.mz, rtol=1e-5)
    np.testing.assert_allclose(restored.intensity, spec.intensity, rtol=1e-2)


def test_token_roundtrip_lossy_deconvoluted_with_singletons() -> None:
    """A deconvoluted spectrum with singleton charge sentinels (-1) must encode
    losslessly via the default (lossy) path without aborting; charge round-trips
    exactly. Regression for spectrl's PIC-codec abort on negative charge."""
    spec = _basic_spectrum()  # charge=[1, 2, -1], DECONVOLUTED
    restored = from_spectrl_token(to_spectrl_token(spec))  # default: lossy
    np.testing.assert_array_equal(restored.charge, spec.charge)


def test_token_roundtrip_with_ion_mobility() -> None:
    spec = MsnSpectrum(
        mz=np.array([100.0, 200.0], dtype=np.float64),
        intensity=np.array([1000.0, 2000.0], dtype=np.float64),
        im=np.array([0.85, 1.15], dtype=np.float64),
        im_type="ook0",
        ms_level=1,
    )
    token = to_spectrl_token(spec, lossless=True)
    restored = from_spectrl_token(token)
    assert isinstance(restored, MsnSpectrum)
    r_im, s_im = restored.im, spec.im
    assert r_im is not None and s_im is not None
    np.testing.assert_allclose(r_im, s_im)
    assert restored.im_type == "ook0"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_iso_score_roundtrips_via_extra_arrays() -> None:
    """iso_score is carried via spectrl's extra_arrays slot (MS:1000786 non-standard array)."""
    spec = Spectrum(
        mz=np.array([100.0, 200.0], dtype=np.float64),
        intensity=np.array([1.0, 2.0], dtype=np.float64),
        charge=np.array([1, 2], dtype=np.int32),
        iso_score=np.array([0.95, 0.42], dtype=np.float64),
        spectrum_type=SpectrumType.DECONVOLUTED,
    )
    restored = from_spectrl_token(to_spectrl_token(spec, lossless=True))
    r_score, s_score = restored.iso_score, spec.iso_score
    assert r_score is not None and s_score is not None
    np.testing.assert_allclose(r_score, s_score)


def test_iso_score_absent_when_not_set() -> None:
    """A spectrum without iso_score round-trips with iso_score still None."""
    spec = Spectrum(
        mz=np.array([100.0, 200.0], dtype=np.float64),
        intensity=np.array([1.0, 2.0], dtype=np.float64),
    )
    restored = from_spectrl_token(to_spectrl_token(spec, lossless=True))
    assert restored.iso_score is None


def test_decoded_to_spectrum_via_from_decoded() -> None:
    spec = _basic_msn()
    decoded = spectrl.decode_token(to_spectrl_token(spec, lossless=True))
    restored = from_decoded_spectrum(decoded)
    assert isinstance(restored, MsnSpectrum)
    assert restored.ms_level == 2


def test_negative_polarity_roundtrip() -> None:
    spec = MsnSpectrum(
        mz=np.array([100.0], dtype=np.float64),
        intensity=np.array([1.0], dtype=np.float64),
        ms_level=1,
        polarity="negative",
    )
    restored = from_spectrl_token(to_spectrl_token(spec, lossless=True))
    assert isinstance(restored, MsnSpectrum)
    assert restored.polarity == "negative"


def test_plain_spectrum_decodes_without_msn_fields() -> None:
    spec = Spectrum(
        mz=np.array([100.0, 200.0], dtype=np.float64),
        intensity=np.array([1.0, 2.0], dtype=np.float64),
    )
    restored = from_spectrl_token(to_spectrl_token(spec, lossless=True))
    # Plain Spectrum (no centroid flag, no MSn metadata) decodes back to plain Spectrum
    assert type(restored) is Spectrum


# ---------------------------------------------------------------------------
# user_params metadata round-trip (spxtacular scalars without a CV term)
# ---------------------------------------------------------------------------


def test_denoised_normalized_roundtrip_on_plain_spectrum() -> None:
    """denoised/normalized provenance strings survive on a plain Spectrum."""
    spec = Spectrum(
        mz=np.array([100.0, 200.0], dtype=np.float64),
        intensity=np.array([1.0, 2.0], dtype=np.float64),
        denoised="mad",
        normalized="tic",
    )
    restored = from_spectrl_token(to_spectrl_token(spec, lossless=True))
    assert type(restored) is Spectrum
    assert restored.denoised == "mad"
    assert restored.normalized == "tic"


def test_msn_scalar_fields_roundtrip_via_user_params() -> None:
    spec = MsnSpectrum(
        mz=np.array([100.0, 200.0], dtype=np.float64),
        intensity=np.array([1.0, 2.0], dtype=np.float64),
        ms_level=2,
        scan_number=4242,
        resolution=60000.0,
        analyzer="FTMS",
        ramp_time=100.0,
        im_range=(0.6, 1.4),
        isolation_im_range=(0.9, 1.1),
        denoised="mad",
        normalized="tic",
    )
    restored = from_spectrl_token(to_spectrl_token(spec, lossless=True))
    assert isinstance(restored, MsnSpectrum)
    assert restored.scan_number == 4242
    assert restored.resolution == pytest.approx(60000.0)
    assert restored.analyzer == "FTMS"
    assert restored.ramp_time == pytest.approx(100.0)
    assert restored.im_range == pytest.approx((0.6, 1.4))
    assert restored.isolation_im_range == pytest.approx((0.9, 1.1))
    assert restored.denoised == "mad"
    assert restored.normalized == "tic"


def test_scan_number_alone_forces_msn_spectrum() -> None:
    """An MSn-only user_param promotes the decode to MsnSpectrum even with no
    other MSn metadata present."""
    spec = MsnSpectrum(
        mz=np.array([100.0], dtype=np.float64),
        intensity=np.array([1.0], dtype=np.float64),
        scan_number=7,
    )
    restored = from_spectrl_token(to_spectrl_token(spec, lossless=True))
    assert isinstance(restored, MsnSpectrum)
    assert restored.scan_number == 7


# ---------------------------------------------------------------------------
# Foreign tokens — shapes spxtacular's own encoder never produces
# ---------------------------------------------------------------------------


def _foreign_decoded(**kwargs) -> spectrl.DecodedSpectrum:
    """A minimal DecodedSpectrum, as if handed to us by a non-spxtacular producer."""
    return spectrl.DecodedSpectrum(
        default_array_length=2,
        mz=np.array([100.0, 200.0], dtype=np.float64),
        intensity=np.array([1.0, 2.0], dtype=np.float64),
        **kwargs,
    )


def _selected_ion(mz: float | None) -> SpectrlSelectedIon:
    params = [] if mz is None else [SpectrlCvParam(accession=_SELECTED_ION_MZ, value=mz)]
    return SpectrlSelectedIon(params=params)


def test_monoisotopic_flag_keyed_by_precursor_not_selected_ion_count() -> None:
    """A precursor carrying several selected ions must not shift the
    is_monoisotopic flag onto the following precursor."""
    decoded = _foreign_decoded(
        precursors=[
            SpectrlPrecursor(selected_ions=[_selected_ion(500.0), _selected_ion(501.0)]),
            SpectrlPrecursor(selected_ions=[_selected_ion(700.0)]),
        ],
        user_params=[
            SpectrlUserParam(name=f"{_UP_PREC_MONOISOTOPIC}.0", value=0, type="xsd:boolean"),
            SpectrlUserParam(name=f"{_UP_PREC_MONOISOTOPIC}.1", value=1, type="xsd:boolean"),
        ],
    )
    restored = from_decoded_spectrum(decoded)
    assert isinstance(restored, MsnSpectrum)
    assert restored.precursors is not None
    # both ions of precursor 0 take precursor 0's flag; precursor 1 keeps its own
    assert [p.mz for p in restored.precursors] == [500.0, 501.0, 700.0]
    assert [p.is_monoisotopic for p in restored.precursors] == [False, False, True]


def test_monoisotopic_flag_survives_skipped_selected_ion() -> None:
    """A selected ion dropped for a missing m/z must not shift the
    is_monoisotopic flag of the precursors that follow it."""
    decoded = _foreign_decoded(
        precursors=[
            SpectrlPrecursor(selected_ions=[_selected_ion(None)]),  # no m/z — skipped
            SpectrlPrecursor(selected_ions=[_selected_ion(700.0)]),
        ],
        user_params=[
            SpectrlUserParam(name=f"{_UP_PREC_MONOISOTOPIC}.0", value=0, type="xsd:boolean"),
            SpectrlUserParam(name=f"{_UP_PREC_MONOISOTOPIC}.1", value=1, type="xsd:boolean"),
        ],
    )
    restored = from_decoded_spectrum(decoded)
    assert isinstance(restored, MsnSpectrum)
    assert restored.precursors is not None and len(restored.precursors) == 1
    assert restored.precursors[0].mz == pytest.approx(700.0)
    assert restored.precursors[0].is_monoisotopic is True


def test_monoisotopic_flag_roundtrips_for_own_tokens() -> None:
    """The precursor-index keying must leave spxtacular's own round-trip intact."""
    spec = _basic_msn()
    spec.precursors = [
        Precursor(mz=500.25, intensity=8000.0, charge=2, is_monoisotopic=True),
        Precursor(mz=700.5, intensity=4000.0, charge=3, is_monoisotopic=False),
    ]
    restored = from_spectrl_token(to_spectrl_token(spec, lossless=True))
    assert isinstance(restored, MsnSpectrum)
    assert restored.precursors is not None
    assert [p.is_monoisotopic for p in restored.precursors] == [True, False]


def test_scan_start_time_in_minutes_converted_to_seconds() -> None:
    """MS:1000016 with a minutes unit (the common mzML convention) must be
    converted — MsnSpectrum.rt is always seconds."""
    decoded = _foreign_decoded(
        scans=[SpectrlScan(params=[SpectrlCvParam(accession=_SCAN_START_TIME, value=2.5, unit_accession=_UNIT_MINUTE)])]
    )
    restored = from_decoded_spectrum(decoded)
    assert isinstance(restored, MsnSpectrum)
    assert restored.rt == pytest.approx(150.0)


@pytest.mark.parametrize("unit", [_UNIT_SECOND, None])
def test_scan_start_time_in_seconds_untouched(unit: str | None) -> None:
    """An explicit seconds unit — and an absent one — are both taken as seconds."""
    decoded = _foreign_decoded(
        scans=[SpectrlScan(params=[SpectrlCvParam(accession=_SCAN_START_TIME, value=2.5, unit_accession=unit)])]
    )
    restored = from_decoded_spectrum(decoded)
    assert isinstance(restored, MsnSpectrum)
    assert restored.rt == pytest.approx(2.5)


def test_scan_window_without_values_decodes_without_mz_range() -> None:
    """A degenerate scan window (terms present, values absent) must not raise."""
    decoded = _foreign_decoded(
        scans=[
            SpectrlScan(
                windows=[
                    SpectrlScanWindow(
                        params=[
                            SpectrlCvParam(accession=_SCAN_WINDOW_LOWER),
                            SpectrlCvParam(accession=_SCAN_WINDOW_UPPER),
                        ]
                    )
                ]
            )
        ]
    )
    restored = from_decoded_spectrum(decoded)
    assert isinstance(restored, MsnSpectrum)
    assert restored.mz_range is None


# ---------------------------------------------------------------------------
# URL sharing helpers
# ---------------------------------------------------------------------------


def test_url_fragment_roundtrip() -> None:
    spec = _basic_msn()
    url = to_spectrl_url(spec, "https://example.com/view", lossless=True)
    assert url.startswith("https://example.com/view#spectrl1.")
    restored = from_spectrl_url(url)
    assert isinstance(restored, MsnSpectrum)
    assert restored.scan_number == 42


def test_url_query_roundtrip() -> None:
    spec = _basic_spectrum()
    url = to_spectrl_url(spec, "https://example.com/view", mode="query", param="s", lossless=True)
    assert "s=spectrl1." in url
    restored = from_spectrl_url(url)
    np.testing.assert_allclose(restored.mz, spec.mz)


def test_url_data_uri_roundtrip_needs_no_base() -> None:
    spec = _basic_spectrum()
    uri = to_spectrl_url(spec, mode="data", lossless=True)
    assert uri.startswith("data:application/vnd.spectrl")
    restored = from_spectrl_url(uri)
    np.testing.assert_allclose(restored.mz, spec.mz)


def test_url_requires_base_for_fragment_and_query() -> None:
    spec = _basic_spectrum()
    with pytest.raises(ValueError):
        to_spectrl_url(spec, mode="fragment")
    with pytest.raises(ValueError):
        to_spectrl_url(spec, mode="query")


def test_url_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        to_spectrl_url(_basic_spectrum(), "https://example.com", mode="bogus")
