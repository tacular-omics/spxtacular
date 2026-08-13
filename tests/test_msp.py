"""Tests for MSP spectral-library reading (MspReader) and writing (write_msp)."""

from __future__ import annotations

import gzip
import textwrap

import numpy as np
import pytest

from spxtacular import MsnSpectrum, MspReader, Precursor, Reader, Spectrum, SpectrumType, write_msp
from spxtacular.enums import Polarity

# A NIST/SpectraST-style peptide library record: charge rides on the Name,
# everything else hides in Comment key=value pairs, "Num peaks" is lowercase,
# and peaks carry quoted annotations.
NIST_RECORD = textwrap.dedent("""\
    Name: AAAAK/2
    MW: 430.253
    Comment: Spec=Consensus Parent=216.1343 CE=35 RT=1823.4 Mods=0 Charge=2
    Num peaks: 3
    101.0715 1200.5 "b1/0.002"
    172.1086 8000.0 "b2/0.001"
    303.1776 950.25 "y3/0.004 2/2"
""")

# A MoNA/MS-DIAL-style metabolomics record: SHOUTING keys, compound metadata
# spxtacular has no fields for, and a unit-suffixed collision energy.
MONA_RECORD = textwrap.dedent("""\
    NAME: Aspirin
    PRECURSORMZ: 181.0495
    PRECURSORTYPE: [M+H]+
    IONMODE: Positive
    RETENTIONTIME: 5.43
    COLLISIONENERGY: 35 eV
    FORMULA: C9H8O4
    SMILES: CC(=O)OC1=CC=CC=C1C(=O)O
    INCHIKEY: BSYNRYMUTXBXSQ-UHFFFAOYSA-N
    Num Peaks: 2
    92.0257 4500.0
    163.0390 12000.0
""")


@pytest.fixture
def two_dialects(tmp_path):
    path = tmp_path / "library.msp"
    path.write_text(NIST_RECORD + "\n" + MONA_RECORD)
    return path


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def test_iterates_all_records(two_dialects):
    with MspReader(two_dialects) as reader:
        spectra = list(reader)
    assert len(spectra) == 2
    for spec in spectra:
        assert isinstance(spec, MsnSpectrum)
        assert spec.ms_level == 2
        assert spec.spectrum_type == SpectrumType.CENTROID


def test_nist_peptide_dialect(two_dialects):
    spec = MspReader(two_dialects)[0]
    assert spec.native_id == "AAAAK/2"
    np.testing.assert_array_equal(spec.mz, [101.0715, 172.1086, 303.1776])
    np.testing.assert_array_equal(spec.intensity, [1200.5, 8000.0, 950.25])
    assert spec.precursors is not None
    prec = spec.precursors[0]
    assert prec.mz == 216.1343  # Comment Parent=
    assert prec.charge == 2  # Comment Charge= (agrees with the Name suffix)
    assert spec.collision_energy == 35.0  # Comment CE=
    assert spec.rt == 1823.4  # Comment RT=, verbatim
    assert spec.polarity == Polarity.POSITIVE  # implied by charge sign


def test_name_charge_suffix_is_fallback(tmp_path):
    path = tmp_path / "x.msp"
    path.write_text("Name: PEPTIDEK/3\nNum Peaks: 1\n100.0 1.0\n")
    spec = MspReader(path)[0]
    # No Charge header and no Comment — the /3 on the Name is the charge, but
    # with no precursor m/z anywhere there is no Precursor to attach it to.
    assert spec.precursors is None
    assert spec.polarity == Polarity.POSITIVE


def test_mona_metabolomics_dialect(two_dialects):
    spec = MspReader(two_dialects)[1]
    assert spec.native_id == "Aspirin"
    assert spec.precursors is not None
    prec = spec.precursors[0]
    assert prec.mz == 181.0495
    assert prec.charge is None
    assert spec.polarity == Polarity.POSITIVE  # IONMODE
    assert spec.rt == 5.43  # verbatim — no unit guessing
    assert spec.collision_energy == 35.0  # "35 eV"
    assert len(spec) == 2


def test_negative_ion_mode(tmp_path):
    path = tmp_path / "neg.msp"
    path.write_text("Name: X\nIon_mode: N\nPrecursorMZ: 179.03\nNum Peaks: 1\n89.02 100.0\n")
    spec = MspReader(path)[0]
    assert spec.polarity == Polarity.NEGATIVE


def test_key_normalisation_across_spellings(tmp_path):
    # Same fields, three spellings — all must land in the same places.
    path = tmp_path / "spellings.msp"
    path.write_text("Name: X\nPrecursor_mz: 100.5\nRetention_Time: 12.0\nnum peaks: 1\n50.0 1.0\n")
    spec = MspReader(path)[0]
    assert spec.precursors is not None
    assert spec.precursors[0].mz == 100.5
    assert spec.rt == 12.0


def test_semicolon_separated_peaks_on_one_line(tmp_path):
    path = tmp_path / "semi.msp"
    path.write_text("Name: X\nNum Peaks: 3\n100.0 10.0; 200.0 20.0;\n300.0 30.0\n")
    spec = MspReader(path)[0]
    np.testing.assert_array_equal(spec.mz, [100.0, 200.0, 300.0])
    np.testing.assert_array_equal(spec.intensity, [10.0, 20.0, 30.0])


def test_zero_peak_record(tmp_path):
    path = tmp_path / "empty.msp"
    path.write_text("Name: nothing\nNum Peaks: 0\n\nName: Y\nNum Peaks: 1\n100.0 1.0\n")
    with MspReader(path) as reader:
        spectra = list(reader)
    assert len(spectra) == 2
    assert len(spectra[0]) == 0
    assert spectra[0].native_id == "nothing"


def test_len_counts_records(two_dialects):
    reader = MspReader(two_dialects)
    assert len(reader) == 2


def test_lookup_by_index_and_name(two_dialects):
    reader = MspReader(two_dialects)
    assert reader[1].native_id == "Aspirin"
    assert reader["Aspirin"].native_id == "Aspirin"
    with pytest.raises(IndexError):
        reader[2]
    with pytest.raises(KeyError):
        reader["Caffeine"]


def test_duplicate_names_first_wins(tmp_path):
    path = tmp_path / "dup.msp"
    path.write_text(
        "Name: PEP/2\nCollision_energy: 25\nNum Peaks: 1\n100.0 1.0\n\n"
        "Name: PEP/2\nCollision_energy: 35\nNum Peaks: 1\n100.0 2.0\n"
    )
    assert MspReader(path)["PEP/2"].collision_energy == 25.0


def test_ms1_empty_ms2_full(two_dialects):
    with MspReader(two_dialects) as reader:
        assert list(reader.ms1) == []
        assert len(list(reader.ms2)) == 2


def test_gzip_by_magic_bytes(tmp_path, two_dialects):
    gz_path = tmp_path / "renamed.msp"  # wrong name on purpose — magic bytes decide
    gz_path.write_bytes(gzip.compress(two_dialects.read_bytes()))
    with MspReader(gz_path) as reader:
        assert len(list(reader)) == 2


def test_reader_autodetects_msp(two_dialects):
    with Reader(two_dialects) as reader:
        assert isinstance(reader._reader, MspReader)
        assert len(list(reader.ms2)) == 2


def test_comment_lines_skipped(tmp_path):
    path = tmp_path / "c.msp"
    path.write_text("# exported by test\nName: X\nNum Peaks: 1\n100.0 1.0\n")
    assert len(MspReader(path)[0]) == 1


# ---------------------------------------------------------------------------
# Structural errors
# ---------------------------------------------------------------------------


def test_missing_num_peaks_before_blank_line(tmp_path):
    path = tmp_path / "bad.msp"
    path.write_text("Name: X\nPrecursorMZ: 100.0\n\nName: Y\nNum Peaks: 1\n100.0 1.0\n")
    with pytest.raises(ValueError, match="no 'Num Peaks'"):
        list(MspReader(path))


def test_missing_num_peaks_at_eof(tmp_path):
    path = tmp_path / "bad.msp"
    path.write_text("Name: X\nPrecursorMZ: 100.0\n")
    with pytest.raises(ValueError, match="no 'Num Peaks'"):
        list(MspReader(path))


def test_truncated_peak_list(tmp_path):
    path = tmp_path / "bad.msp"
    path.write_text("Name: X\nNum Peaks: 3\n100.0 1.0\n200.0 2.0\n")
    with pytest.raises(ValueError, match="declares 3 peaks"):
        list(MspReader(path))


def test_blank_line_inside_peak_list(tmp_path):
    path = tmp_path / "bad.msp"
    path.write_text("Name: X\nNum Peaks: 3\n100.0 1.0\n\n200.0 2.0\n300.0 3.0\n")
    with pytest.raises(ValueError, match="ends after 1"):
        list(MspReader(path))


def test_excess_peaks_error(tmp_path):
    path = tmp_path / "bad.msp"
    path.write_text("Name: X\nNum Peaks: 1\n100.0 1.0\n200.0 2.0\n")
    with pytest.raises(ValueError, match="expected a 'Key: value' header"):
        list(MspReader(path))


def test_unparsable_peak_number(tmp_path):
    path = tmp_path / "bad.msp"
    path.write_text("Name: X\nNum Peaks: 1\n100.0 abc\n")
    with pytest.raises(ValueError, match=r"bad\.msp:3"):
        list(MspReader(path))


def test_peak_line_with_one_token(tmp_path):
    path = tmp_path / "bad.msp"
    path.write_text("Name: X\nNum Peaks: 1\n100.0\n")
    with pytest.raises(ValueError, match="expected 'mz intensity'"):
        list(MspReader(path))


# ---------------------------------------------------------------------------
# Writing and round trips
# ---------------------------------------------------------------------------


def _library_spectrum() -> MsnSpectrum:
    return MsnSpectrum(
        mz=np.array([101.07154321, 172.10864321, 303.17761234]),
        intensity=np.array([1200.512345, 8000.987654, 950.25]),
        spectrum_type=SpectrumType.CENTROID,
        ms_level=2,
        native_id="AAAAK/2",
        rt=1823.4,
        polarity=Polarity.POSITIVE,
        collision_energy=35.0,
        precursors=[Precursor(mz=216.13435678, intensity=0.0, charge=2, is_monoisotopic=None)],
    )


def test_round_trip_is_bit_exact(tmp_path):
    original = _library_spectrum()
    path = write_msp(original, tmp_path / "out.msp")
    restored = MspReader(path)[0]
    np.testing.assert_array_equal(restored.mz, original.mz)
    np.testing.assert_array_equal(restored.intensity, original.intensity)
    assert restored.native_id == original.native_id
    assert restored.precursors is not None and original.precursors is not None
    assert restored.precursors[0].mz == original.precursors[0].mz
    assert restored.precursors[0].charge == 2
    assert restored.polarity == Polarity.POSITIVE
    assert restored.rt == original.rt
    assert restored.collision_energy == original.collision_energy


def test_round_trip_negative_polarity(tmp_path):
    spec = _library_spectrum()
    spec.polarity = Polarity.NEGATIVE
    restored = MspReader(write_msp(spec, tmp_path / "neg.msp"))[0]
    # Ion_mode carries polarity explicitly — the charge stays positive.
    assert restored.polarity == Polarity.NEGATIVE
    assert restored.precursors is not None
    assert restored.precursors[0].charge == 2


def test_write_plain_spectrum(tmp_path):
    spec = Spectrum(mz=np.array([100.0, 200.0]), intensity=np.array([1.0, 2.0]))
    restored = MspReader(write_msp(spec, tmp_path / "plain.msp"))[0]
    np.testing.assert_array_equal(restored.mz, spec.mz)
    assert restored.native_id is None
    assert restored.precursors is None


def test_write_gzip_suffix(tmp_path):
    path = write_msp(_library_spectrum(), tmp_path / "out.msp.gz")
    assert path.read_bytes()[:2] == b"\x1f\x8b"
    assert MspReader(path)[0].native_id == "AAAAK/2"


def test_write_profile_refused(tmp_path):
    spec = Spectrum(mz=np.array([100.0, 100.01]), intensity=np.array([1.0, 2.0]), spectrum_type=SpectrumType.PROFILE)
    with pytest.raises(ValueError, match="centroid"):
        write_msp(spec, tmp_path / "nope.msp")


def test_write_many_then_len(tmp_path):
    specs = [_library_spectrum() for _ in range(5)]
    path = write_msp(specs, tmp_path / "many.msp")
    assert len(MspReader(path)) == 5
