import base64
import pathlib

import numpy as np
import pytest

mzp = pytest.importorskip("mzmlpy")

from spxtacular.core import MsnSpectrum, SpectrumType  # noqa: E402
from spxtacular.reader import MzmlReader, Reader  # noqa: E402

DATA_DIR = pathlib.Path(__file__).parent / "data"
EXAMPLE_MZML = DATA_DIR / "example.mzML"
EXAMPLE_MZML_GZ = DATA_DIR / "example.mzML.gz"


@pytest.fixture(scope="module")
def ms1_spectrum():
    with MzmlReader(str(EXAMPLE_MZML)) as r:
        return next(iter(r.ms1))


@pytest.fixture(scope="module")
def ms2_spectrum():
    with MzmlReader(str(EXAMPLE_MZML)) as r:
        return next(iter(r.ms2))


# --- MS1 ---


def test_ms1_is_msn_spectrum(ms1_spectrum):
    assert isinstance(ms1_spectrum, MsnSpectrum)


def test_ms1_level(ms1_spectrum):
    assert ms1_spectrum.ms_level == 1


def test_ms1_spectrum_type_is_centroid(ms1_spectrum):
    assert ms1_spectrum.spectrum_type == SpectrumType.CENTROID


def test_ms1_arrays_nonempty(ms1_spectrum):
    assert len(ms1_spectrum.mz) > 0


def test_ms1_mz_intensity_same_length(ms1_spectrum):
    assert len(ms1_spectrum.mz) == len(ms1_spectrum.intensity)


def test_ms1_mz_sorted(ms1_spectrum):
    assert np.all(ms1_spectrum.mz[:-1] <= ms1_spectrum.mz[1:])


def test_ms1_intensity_positive(ms1_spectrum):
    assert np.all(ms1_spectrum.intensity > 0)


def test_ms1_rt_is_float(ms1_spectrum):
    assert isinstance(ms1_spectrum.rt, float)
    assert ms1_spectrum.rt > 0


def test_ms1_mz_range_set(ms1_spectrum):
    assert ms1_spectrum.mz_range is not None
    lo, hi = ms1_spectrum.mz_range
    assert hi > lo > 0


def test_ms1_native_id_set(ms1_spectrum):
    assert ms1_spectrum.native_id is not None
    assert len(ms1_spectrum.native_id) > 0


def test_ms1_no_precursors(ms1_spectrum):
    assert ms1_spectrum.precursors is None


def test_ms1_no_charge_array(ms1_spectrum):
    assert ms1_spectrum.charge is None


def test_ms1_no_ion_mobility(ms1_spectrum):
    assert ms1_spectrum.im is None


# --- MS2 ---


def test_ms2_is_msn_spectrum(ms2_spectrum):
    assert isinstance(ms2_spectrum, MsnSpectrum)


def test_ms2_level(ms2_spectrum):
    assert ms2_spectrum.ms_level == 2


def test_ms2_spectrum_type(ms2_spectrum):
    assert ms2_spectrum.spectrum_type in (SpectrumType.CENTROID, SpectrumType.PROFILE)


def test_ms2_arrays_nonempty(ms2_spectrum):
    assert len(ms2_spectrum.mz) > 0


def test_ms2_mz_intensity_same_length(ms2_spectrum):
    assert len(ms2_spectrum.mz) == len(ms2_spectrum.intensity)


def test_ms2_rt_is_float(ms2_spectrum):
    assert isinstance(ms2_spectrum.rt, float)
    assert ms2_spectrum.rt > 0


def test_ms2_has_precursor(ms2_spectrum):
    assert ms2_spectrum.precursors is not None
    assert len(ms2_spectrum.precursors) > 0


def test_ms2_precursor_mz_positive(ms2_spectrum):
    assert ms2_spectrum.precursors[0].mz > 0


def test_ms2_precursor_charge_set(ms2_spectrum):
    assert ms2_spectrum.precursors[0].charge is not None
    assert ms2_spectrum.precursors[0].charge > 0


def test_ms2_collision_energy_set(ms2_spectrum):
    assert ms2_spectrum.collision_energy is not None
    assert ms2_spectrum.collision_energy > 0


def test_ms2_activation_type_set(ms2_spectrum):
    assert ms2_spectrum.activation_type is not None


# --- __getitem__ ---


def test_getitem_by_index_returns_msn_spectrum():
    r = MzmlReader(str(EXAMPLE_MZML))
    spec = r[0]
    assert isinstance(spec, MsnSpectrum)


def test_getitem_by_index_matches_iteration():
    r = MzmlReader(str(EXAMPLE_MZML))
    spec_iter = next(iter(r.ms1))
    spec_item = r[0]
    assert spec_item.native_id == spec_iter.native_id
    np.testing.assert_array_equal(spec_item.mz, spec_iter.mz)


def test_getitem_by_native_id():
    r = MzmlReader(str(EXAMPLE_MZML))
    spec = r["scan=19"]
    assert isinstance(spec, MsnSpectrum)
    assert spec.native_id == "scan=19"


def test_getitem_by_native_id_matches_index():
    r = MzmlReader(str(EXAMPLE_MZML))
    by_index = r[0]
    assert by_index.native_id is not None
    by_id = r[by_index.native_id]
    np.testing.assert_array_equal(by_index.mz, by_id.mz)


def test_getitem_invalid_id_raises():
    r = MzmlReader(str(EXAMPLE_MZML))
    with pytest.raises(KeyError):
        r["scan=999999"]


# --- persistent handle (open/close) ---


def test_open_close_iter_matches_context_manager():
    r = MzmlReader(str(EXAMPLE_MZML))
    r.open()
    try:
        specs_open = list(r.ms1)
    finally:
        r.close()

    with MzmlReader(str(EXAMPLE_MZML)) as r2:
        specs_cm = list(r2.ms1)

    assert len(specs_open) == len(specs_cm)
    for a, b in zip(specs_open, specs_cm, strict=True):
        assert a.native_id == b.native_id
        np.testing.assert_array_equal(a.mz, b.mz)


def test_open_close_getitem_matches_no_open():
    r = MzmlReader(str(EXAMPLE_MZML))
    r.open()
    try:
        spec_open = r[0]
    finally:
        r.close()

    spec_noop = MzmlReader(str(EXAMPLE_MZML))[0]
    assert spec_open.native_id == spec_noop.native_id
    np.testing.assert_array_equal(spec_open.mz, spec_noop.mz)


# ---------------------------------------------------------------------------
# gzip mzML
# ---------------------------------------------------------------------------


def test_gz_ms1_readable():
    with MzmlReader(str(EXAMPLE_MZML_GZ)) as r:
        spec = next(iter(r.ms1))
    assert isinstance(spec, MsnSpectrum)
    assert spec.ms_level == 1


def test_gz_ms2_readable():
    with MzmlReader(str(EXAMPLE_MZML_GZ)) as r:
        spec = next(iter(r.ms2))
    assert isinstance(spec, MsnSpectrum)
    assert spec.ms_level == 2


def test_gz_matches_uncompressed():
    with MzmlReader(str(EXAMPLE_MZML)) as r:
        plain_ms1 = list(r.ms1)

    with MzmlReader(str(EXAMPLE_MZML_GZ)) as r:
        gz_ms1 = list(r.ms1)

    assert len(gz_ms1) == len(plain_ms1)
    for plain, gz in zip(plain_ms1, gz_ms1, strict=True):
        assert plain.native_id == gz.native_id
        np.testing.assert_array_equal(plain.mz, gz.mz)
        np.testing.assert_array_equal(plain.intensity, gz.intensity)


def test_gz_getitem_by_index():
    r = MzmlReader(str(EXAMPLE_MZML_GZ))
    spec = r[0]
    assert isinstance(spec, MsnSpectrum)


def test_gz_stream_mode_reads_sequentially_without_extraction():
    with MzmlReader(EXAMPLE_MZML_GZ, gzip_mode="stream", in_memory=False) as r:
        spec = next(iter(r.ms1))

    assert isinstance(spec, MsnSpectrum)
    assert spec.ms_level == 1


def test_unified_reader_forwards_mzml_options():
    with Reader(EXAMPLE_MZML_GZ, mzml_gzip_mode="stream", mzml_in_memory=False) as r:
        spec = next(iter(r.ms1))

    assert isinstance(spec, MsnSpectrum)
    assert spec.ms_level == 1


def test_mzml_options_apply_without_persistent_handle(monkeypatch):
    calls = []
    real_mzml = mzp.Mzml

    def recording_mzml(path, **kwargs):
        calls.append((path, kwargs))
        return real_mzml(path, **kwargs)

    monkeypatch.setattr(mzp, "Mzml", recording_mzml)
    reader = MzmlReader(EXAMPLE_MZML_GZ, gzip_mode="stream", in_memory=False)
    next(iter(reader.ms1))

    assert calls == [
        (
            EXAMPLE_MZML_GZ,
            {"gzip_mode": "stream", "in_memory": False, "extract_dir": None},
        )
    ]


# ---------------------------------------------------------------------------
# Deconvoluted spectra
#
# "charge deconvolution" (MS:1000034) is written on a <processingMethod>, not on
# the spectrum, so a spectrum only reveals it through its dataProcessingRef.
# example.mzML already declares such a processing entry, so these fixtures only
# have to add the charge array and the reference.
# ---------------------------------------------------------------------------

DECONVOLUTION_DP = "CompassXtract_x0020_processing"  # declares MS:1000034
CONVERSION_DP = "pwiz_processing"  # declares only "Conversion to mzML"


def _charge_binary_array(n_values: int) -> str:
    charges = np.full(n_values, 2.0, dtype=np.float64)
    encoded = base64.b64encode(charges.tobytes()).decode()
    return (
        f'<binaryDataArray encodedLength="{len(encoded)}">\n'
        '<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>\n'
        '<cvParam cvRef="MS" accession="MS:1000576" name="no compression" value=""/>\n'
        '<cvParam cvRef="MS" accession="MS:1000516" name="charge array" value=""/>\n'
        f"<binary>{encoded}</binary>\n"
        "</binaryDataArray>\n"
    )


def _mzml_with_charge_array(tmp_path, processing_ref=None, sole_processing=None):
    """A copy of example.mzML whose first spectrum carries a charge array.

    ``processing_ref`` is written onto the spectrum element; ``sole_processing``
    reduces the file to that one dataProcessing entry, which is the case where an
    unreferenced spectrum can still be resolved.
    """
    source = EXAMPLE_MZML.read_text(encoding="ISO-8859-1")
    # Drop the indexedmzML wrapper: editing the body invalidates its offsets, and
    # mzmlpy rebuilds the index for a plain mzML.
    body = source[source.index("<mzML ") : source.index("</mzML>") + len("</mzML>")]

    spectrum_tag = '<spectrum index="0" id="scan=19" defaultArrayLength="15">'
    if processing_ref is not None:
        body = body.replace(spectrum_tag, spectrum_tag[:-1] + f' dataProcessingRef="{processing_ref}">', 1)

    start = body.index('<binaryDataArrayList count="2">')
    end = body.index("</binaryDataArrayList>", start)
    body = body[:start] + body[start:end].replace('count="2"', 'count="3"') + _charge_binary_array(15) + body[end:]

    if sole_processing is not None:
        dropped = CONVERSION_DP if sole_processing == DECONVOLUTION_DP else DECONVOLUTION_DP
        block_start = body.index(f'<dataProcessing id="{dropped}">')
        block_end = body.index("</dataProcessing>", block_start) + len("</dataProcessing>")
        body = body[:block_start] + body[block_end:]
        body = body.replace('<dataProcessingList count="2">', '<dataProcessingList count="1">', 1)
        body = body.replace(f'defaultDataProcessingRef="{dropped}"', f'defaultDataProcessingRef="{sole_processing}"')

    path = tmp_path / "deconvoluted.mzML"
    path.write_text('<?xml version="1.0" encoding="ISO-8859-1"?>\n' + body, encoding="ISO-8859-1")
    return path


def test_charge_array_referencing_deconvolution_processing_is_deconvoluted(tmp_path):
    path = _mzml_with_charge_array(tmp_path, processing_ref=DECONVOLUTION_DP)
    with MzmlReader(str(path)) as r:
        spec = r[0]
    assert spec.charge is not None
    assert spec.spectrum_type == SpectrumType.DECONVOLUTED


def test_deconvolution_is_detected_when_iterating_too(tmp_path):
    path = _mzml_with_charge_array(tmp_path, processing_ref=DECONVOLUTION_DP)
    with MzmlReader(str(path)) as r:
        spec = next(iter(r.ms1))
    assert spec.spectrum_type == SpectrumType.DECONVOLUTED


def test_charge_array_referencing_plain_conversion_stays_centroid(tmp_path):
    """A charge array is usually just a per-peak annotation."""
    path = _mzml_with_charge_array(tmp_path, processing_ref=CONVERSION_DP)
    with MzmlReader(str(path)) as r:
        spec = r[0]
    assert spec.charge is not None
    assert spec.spectrum_type == SpectrumType.CENTROID


def test_unreferenced_spectrum_is_ambiguous_when_several_processings_exist(tmp_path):
    """mzmlpy does not expose defaultDataProcessingRef, so this stays centroid."""
    path = _mzml_with_charge_array(tmp_path)
    with MzmlReader(str(path)) as r:
        spec = r[0]
    assert spec.spectrum_type == SpectrumType.CENTROID


def test_unreferenced_spectrum_resolves_against_a_sole_processing_entry(tmp_path):
    path = _mzml_with_charge_array(tmp_path, sole_processing=DECONVOLUTION_DP)
    with MzmlReader(str(path)) as r:
        spec = r[0]
    assert spec.spectrum_type == SpectrumType.DECONVOLUTED


def test_a_file_without_deconvolution_processing_is_unaffected():
    """The stock fixture has no charge arrays; nothing may be promoted."""
    with MzmlReader(str(EXAMPLE_MZML)) as r:
        assert all(s.spectrum_type != SpectrumType.DECONVOLUTED for s in r.ms1)
