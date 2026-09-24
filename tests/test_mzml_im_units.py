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


def test_multiple_im_arrays_use_first_matching_and_warn(tmp_path, original_im):
    text = FIXTURE.read_text()
    start = text.index("<binaryDataArray arrayLength=")
    end = text.index("</binaryDataArray>", start) + len("</binaryDataArray>")
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
    assert any("multiple ion mobility arrays" in str(w.message) for w in caught)
    # accessions are sorted, so MS:1002477 (drift time, seconds) comes first
    assert spec.im_type == IMType.DRIFT_TIME_MS
    np.testing.assert_allclose(spec.im, original_im.im * 1000.0)


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
