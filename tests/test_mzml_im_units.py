"""mzML ion-mobility arrays and precursor drift times are typed by their declared unit.

The fixture is the first spectrum of mzmlpy's ``bruker_ms2_im_combined_im.mzML``
(a mean 1/K0 array, MS:1003006 in Vs/cm²). Variants edit the array accession and
unit, as other converters write them.
"""

import pathlib
import warnings

import numpy as np
import pytest

pytest.importorskip("mzmlpy")

from spxtacular import IMType  # noqa: E402
from spxtacular.errors import SpxtacularError  # noqa: E402
from spxtacular.reader import MzmlReader  # noqa: E402

FIXTURE = pathlib.Path(__file__).parent / "data" / "bruker_im_one_spectrum.mzML"
ARRAY_TERM = 'accession="MS:1003006" name="mean inverse reduced ion mobility array"'
OOK0_UNIT = 'unitCvRef="MS" unitAccession="MS:1002814" unitName="volt-second per square centimeter"'
PEAK_INTENSITY = (
    'name="peak intensity" value="2614.0" unitCvRef="MS" unitAccession="MS:1000131" '
    'unitName="number of detector counts"/>'
)


def _read(tmp_path: pathlib.Path, text: str):
    path = tmp_path / "variant.mzML"
    path.write_text(text)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with MzmlReader(str(path)) as reader:
            return next(iter(reader.ms2))


def _variant(accession: str, name: str, unit: str | None) -> str:
    text = FIXTURE.read_text()
    assert ARRAY_TERM in text and OOK0_UNIT in text
    text = text.replace(ARRAY_TERM, f'accession="{accession}" name="{name}"')
    return text.replace(" " + OOK0_UNIT, "" if unit is None else " " + unit)


@pytest.fixture(scope="module")
def original_im():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with MzmlReader(str(FIXTURE)) as reader:
            spec = next(iter(reader.ms2))
    assert spec.im is not None
    return spec


MS_UNIT = 'unitCvRef="UO" unitAccession="UO:0000028" unitName="millisecond"'
S_UNIT = 'unitCvRef="UO" unitAccession="UO:0000010" unitName="second"'


def test_ook0_array_is_ook0(original_im):
    assert original_im.im_type == IMType.OOK0
    assert 0.5 < float(np.nanmedian(original_im.im)) < 2.0


@pytest.mark.parametrize(
    ("accession", "name", "unit", "expected", "scale"),
    [
        ("MS:1002816", "mean ion mobility array", OOK0_UNIT, IMType.OOK0, 1.0),
        ("MS:1002816", "mean ion mobility array", MS_UNIT, IMType.DRIFT_TIME_MS, 1.0),
        ("MS:1002816", "mean ion mobility array", None, IMType.IM, 1.0),
        ("MS:1002477", "mean ion mobility drift time array", MS_UNIT, IMType.DRIFT_TIME_MS, 1.0),
        ("MS:1002477", "mean ion mobility drift time array", S_UNIT, IMType.DRIFT_TIME_MS, 1000.0),
        ("MS:1003007", "raw ion mobility array", MS_UNIT, IMType.DRIFT_TIME_MS, 1.0),
        ("MS:1003153", "raw ion mobility drift time array", None, IMType.IM, 1.0),
    ],
)
def test_array_unit_decides_im_type(tmp_path, original_im, accession, name, unit, expected, scale):
    spec = _read(tmp_path, _variant(accession, name, unit))
    assert spec.im_type == expected
    np.testing.assert_allclose(spec.im, original_im.im * scale)


def _im_block_bounds(text: str) -> tuple[int, int]:
    """Where the ion-mobility ``binaryDataArray`` starts and ends in the fixture."""
    start = text.rindex("<binaryDataArray arrayLength=", 0, text.index(ARRAY_TERM))
    return start, text.index("</binaryDataArray>", start) + len("</binaryDataArray>")


def test_the_array_is_the_one_mzmlpy_pairs_with_the_peaks(original_im):
    """The combined-IM fixture: the values are mzmlpy's own choice, not a second selection.

    The reader asks :attr:`mzmlpy.Spectrum.ion_mobility_array` which array goes with the peaks
    and only reads the unit itself, so the array it reports must be that one element for
    element. Scaling would show up here too: this fixture's unit is Vs/cm², which is ook0, so
    the factor is 1.
    """
    import mzmlpy as mzp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with mzp.Mzml(str(FIXTURE)) as handle:
            theirs = handle.spectra[0].ion_mobility_array

    assert theirs is not None
    assert len(theirs) == len(original_im.mz)
    np.testing.assert_array_equal(original_im.im, theirs.astype(np.float64))


def test_multiple_im_arrays_take_the_first_in_document_order_and_warn(tmp_path, original_im):
    """With two per-point arrays, mzmlpy returns the first in document order, and says so.

    It was the first in *accession* order until the choice moved to mzmlpy, so this fixture,
    whose second array is appended after the first, now reports the original 1/K0 array rather
    than the drift-time one inserted below it.
    """
    text = FIXTURE.read_text()
    start, end = _im_block_bounds(text)
    block = text[start:end]
    second = block.replace(ARRAY_TERM, 'accession="MS:1002477" name="mean ion mobility drift time array"').replace(
        OOK0_UNIT, S_UNIT
    )
    text = text[:end] + "\n" + second + text[end:]
    text = text.replace('<binaryDataArrayList count="3">', '<binaryDataArrayList count="4">', 1)
    path = tmp_path / "two.mzML"
    path.write_text(text)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with MzmlReader(str(path)) as reader:
            spec = next(iter(reader.ms2))
    assert any("ion mobility arrays" in str(w.message) for w in caught), [str(w.message) for w in caught]
    assert spec.im_type == IMType.OOK0
    assert spec.im is not None
    np.testing.assert_allclose(spec.im, original_im.im)


def test_an_array_that_is_not_per_point_warns_and_leaves_im_unset(tmp_path):
    """A length that cannot be paired with the peaks gives a warning and no ``im``.

    It used to raise ``SpxtacularError``. mzmlpy makes the call now, and it declines to pair
    the two rather than refusing the spectrum, so the peaks still read. The variant moves the
    spectrum's ``defaultArrayLength`` instead of touching any array, because an array's own
    ``arrayLength`` attribute does not change what mzmlpy decodes (measured: setting it to 7
    still yields 113 values).
    """
    text = FIXTURE.read_text().replace('defaultArrayLength="113"', 'defaultArrayLength="112"', 1)
    path = tmp_path / "mismatched.mzML"
    path.write_text(text)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with MzmlReader(str(path)) as reader:
            spec = next(iter(reader.ms2))
    assert any("none of them is per-point" in str(w.message) for w in caught), [str(w.message) for w in caught]
    assert spec.im is None
    assert spec.im_type is None
    assert len(spec.mz) == 113, "the peaks still read; only the pairing was declined"


def test_the_identifier_declines_a_length_no_array_has():
    """``_mzml_im_binary_array`` answers None rather than guessing.

    Its rule is mzmlpy's: among the ion-mobility arrays, in document order, the one holding as
    many values as the property returned. Handed a length none of them has, there is no such
    array, and the reader turns that answer into the refusal below.
    """
    import mzmlpy as mzp

    from spxtacular import reader as reader_module

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with mzp.Mzml(str(FIXTURE)) as handle:
            spec = handle.spectra[0]
            assert spec.im_types, "the fixture must have an ion mobility array for this to mean anything"
            assert reader_module._mzml_im_binary_array(spec, np.zeros(3)) is None
            values = spec.ion_mobility_array
            assert values is not None, "the fixture's array must decode for the next line to mean anything"
            assert reader_module._mzml_im_binary_array(spec, values) is not None, "and it finds the real one"


def test_an_unidentifiable_array_is_refused(monkeypatch):
    """If mzmlpy's array cannot be matched to one of ``im_types``, the reader says so.

    The guard exists because the two facts it joins are mzmlpy's, not ours: the property
    returns values without naming their array, and the rule for finding it again (document
    order, one value per point) is repeated here. Should mzmlpy ever return an array this side
    cannot identify, the unit would be unknown and the ``IMType`` a guess, so it refuses rather
    than reporting an untyped array. Nothing in a real file reaches it, which is why it is
    reached here by hand.
    """
    from spxtacular import reader as reader_module

    monkeypatch.setattr(reader_module, "_mzml_im_binary_array", lambda _spec, _values: None)
    with pytest.raises(SpxtacularError, match=r"none of .* has that length"):
        with MzmlReader(str(FIXTURE)) as reader:
            next(iter(reader.ms2))


def test_scan_level_ion_mobility_only_has_no_array(tmp_path):
    """A spectrum whose mobility is a scan-level value has no ``im`` array and no warning.

    This is the timsTOF PASEF MS2 shape: the frame's 1/K0 sits on the scan, not on a
    per-point array. Nothing to pair means nothing to warn about.
    """
    text = FIXTURE.read_text()
    start, end = _im_block_bounds(text)
    text = text[:start] + text[end:]
    text = text.replace('<binaryDataArrayList count="3">', '<binaryDataArrayList count="2">', 1)
    ook0 = (
        '<cvParam cvRef="MS" accession="MS:1002815" name="inverse reduced ion mobility" value="0.88" '
        + OOK0_UNIT
        + "/>"
    )
    scan_start = '<cvParam cvRef="MS" accession="MS:1000016" name="scan start time"'
    assert scan_start in text
    text = text.replace(scan_start, ook0 + "\n" + scan_start, 1)
    path = tmp_path / "scan_level.mzML"
    path.write_text(text)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with MzmlReader(str(path)) as reader:
            spec = next(iter(reader.ms2))
    assert spec.im is None
    assert spec.im_type is None
    assert not [str(w.message) for w in caught if "ion mobility" in str(w.message)]
    assert len(spec.mz) == 113


@pytest.mark.parametrize(
    ("unit", "expected", "value"),
    [
        (MS_UNIT, IMType.DRIFT_TIME_MS, 12.5),
        (S_UNIT, IMType.DRIFT_TIME_MS, 12500.0),
        (None, IMType.IM, 12.5),
    ],
)
def test_precursor_drift_time_unit(tmp_path, unit, expected, value):
    text = FIXTURE.read_text()
    assert PEAK_INTENSITY in text
    unit_attrs = "" if unit is None else " " + unit
    drift = f'<cvParam cvRef="MS" accession="MS:1002476" name="ion mobility drift time" value="12.5"{unit_attrs}/>'
    spec = _read(tmp_path, text.replace(PEAK_INTENSITY, PEAK_INTENSITY + "\n" + drift))
    (precursor,) = spec.precursors
    assert precursor.im_type == expected
    assert precursor.im == pytest.approx(value)


def test_precursor_ook0(tmp_path):
    text = FIXTURE.read_text()
    ook0 = (
        '<cvParam cvRef="MS" accession="MS:1002815" name="inverse reduced ion mobility" value="0.88" '
        + OOK0_UNIT
        + "/>"
    )
    spec = _read(tmp_path, text.replace(PEAK_INTENSITY, PEAK_INTENSITY + "\n" + ook0))
    (precursor,) = spec.precursors
    assert precursor.im_type == IMType.OOK0
    assert precursor.im == pytest.approx(0.88)
