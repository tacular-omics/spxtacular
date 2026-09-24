"""Isobaric reporter-ion extraction and isotope impurity correction.

Every synthetic spectrum is built from tacular's reporter m/z, so these tests also pin
that nothing in spxtacular types in a reporter mass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tacular import ISOBARIC_TAG_LOOKUP

from spxtacular import (
    MsnSpectrum,
    ReporterIons,
    Spectrum,
    SpxtacularError,
    correct_isotope_impurities,
    extract_reporter_ions,
    isotope_correction_matrix,
    reporter_ion_table,
)

TMT10 = ISOBARIC_TAG_LOOKUP["TMT10"]
TMT18 = ISOBARIC_TAG_LOOKUP["TMT18"]


def _spectrum(mzs: list[float], intensities: list[float], *, shuffle: bool = False) -> Spectrum:
    mz = np.asarray(mzs, dtype=float)
    inten = np.asarray(intensities, dtype=float)
    if shuffle:
        order = np.random.default_rng(1).permutation(len(mz))
        mz, inten = mz[order], inten[order]
    return Spectrum(mz, inten)


def _reporter_spectrum(info=TMT10, *, ppm_shift: float = 0.0, skip: tuple[str, ...] = ()) -> Spectrum:
    mzs, ints = [], []
    for k, ion in enumerate(info.reporter_ions):
        if ion.channel in skip:
            continue
        mzs.append(ion.mz * (1 + ppm_shift * 1e-6))
        ints.append(1000.0 * (k + 1))
    mzs += [110.07, 175.119, 500.0]  # unrelated peaks
    ints += [5e4, 7e4, 1e5]
    return _spectrum(mzs, ints, shuffle=True)


def test_exact_reporter_mz() -> None:
    ions = extract_reporter_ions(_reporter_spectrum(), "TMT10")
    assert isinstance(ions, ReporterIons)
    assert ions.plex == "TMT10"
    assert ions.channels == TMT10.channels
    np.testing.assert_allclose(ions.intensity, 1000.0 * np.arange(1, 11))
    np.testing.assert_allclose(ions.ppm_error, 0.0, atol=1e-6)
    assert ions.found.all()
    assert ions["127N"] == 2000.0
    assert ions["127n"] == 2000.0


def test_shift_within_and_outside_tolerance() -> None:
    inside = extract_reporter_ions(_reporter_spectrum(ppm_shift=15.0), "TMT10")
    np.testing.assert_allclose(inside.ppm_error, 15.0, atol=1e-6)
    np.testing.assert_allclose(inside.mz_error, inside.reporter_mz * 15e-6, rtol=1e-6)
    assert inside.found.all()

    outside = extract_reporter_ions(_reporter_spectrum(ppm_shift=22.0), "TMT10")
    assert not outside.found.any()
    np.testing.assert_array_equal(outside.intensity, 0.0)
    assert np.isnan(outside.ppm_error).all()

    wide = extract_reporter_ions(_reporter_spectrum(ppm_shift=22.0), "TMT10", tolerance=0.003, tolerance_unit="da")
    assert wide.found.all()


def test_most_intense_peak_in_window_wins() -> None:
    ref = TMT10.reporter_mzs[0]
    spectrum = _spectrum([ref - ref * 5e-6, ref + ref * 8e-6, ref * (1 + 30e-6)], [100.0, 900.0, 5000.0])
    ions = extract_reporter_ions(spectrum, "TMT10")
    assert ions["126"] == 900.0
    assert ions.ppm_error[0] == pytest.approx(8.0, abs=1e-6)
    assert ions.raw_intensity[0] == 900.0


def test_missing_channels_are_zero_and_nan() -> None:
    ions = extract_reporter_ions(_reporter_spectrum(skip=("127C", "131N")), "TMT10")
    for channel in ("127C", "131N"):
        i = ions.channels.index(channel)
        assert ions.intensity[i] == 0.0
        assert np.isnan(ions.observed_mz[i])
    assert ions.found.sum() == 8
    assert ions.to_dict()["observed_mz"][2] is None


def test_arrays_are_read_only() -> None:
    ions = extract_reporter_ions(_reporter_spectrum(), "TMT10")
    with pytest.raises(ValueError):
        ions.intensity[0] = 1.0


def test_tmt10_vs_tmtpro18_channel_names() -> None:
    assert extract_reporter_ions(_reporter_spectrum(), "TMT10plex").channels[-1] == "131N"
    pro = extract_reporter_ions(_reporter_spectrum(TMT18), "TMTpro18")
    assert pro.plex == "TMT18"
    assert len(pro) == 18
    assert pro.channels[-3:] == ("134N", "134C", "135N")
    table = reporter_ion_table([_reporter_spectrum(TMT18)], "tmtpro18")
    assert list(table.columns[5:]) == list(TMT18.channels)


def test_bad_inputs_raise() -> None:
    spectrum = _reporter_spectrum()
    with pytest.raises(SpxtacularError, match="Unknown isobaric plex"):
        extract_reporter_ions(spectrum, "TMT7")
    with pytest.raises(SpxtacularError, match="tolerance_unit"):
        extract_reporter_ions(spectrum, "TMT10", tolerance_unit="Da")  # ty: ignore[invalid-argument-type]
    with pytest.raises(SpxtacularError, match="overlap"):
        extract_reporter_ions(spectrum, "TMT10", tolerance=30.0)
    with pytest.raises(SpxtacularError, match="normalize"):
        extract_reporter_ions(spectrum, "TMT10", normalize="tic")  # ty: ignore[invalid-argument-type]
    decharged = Spectrum(np.array([126.0]), np.array([1.0]), charge=np.array([0]))
    with pytest.raises(SpxtacularError, match="decharged"):
        extract_reporter_ions(decharged, "TMT10")


def test_spectrum_method_and_normalize() -> None:
    ions = _reporter_spectrum().reporter_ions("TMT10", normalize="sum")
    assert ions.intensity.sum() == pytest.approx(1.0)
    assert ions.normalize == "sum"
    top = _reporter_spectrum().reporter_ions("TMT10", normalize="max")
    assert top.intensity.max() == pytest.approx(1.0)
    empty = _spectrum([500.0], [1.0]).reporter_ions("TMT10", normalize="sum")
    np.testing.assert_array_equal(empty.intensity, 0.0)


# ---------------------------------------------------------------------------
# Correction
# ---------------------------------------------------------------------------

LOT_SHEET = {
    "126": {"-2": 0.0, "-1": 0.0, "+1": 7.0, "+2": 0.2},
    "127N": {"-2": 0.0, "-1": 0.4, "+1": 6.5, "+2": 0.0},
    "127C": {"-2": 0.0, "-1": 0.8, "+1": 6.0, "+2": 0.1},
    "128N": {"-2": 0.0, "-1": 1.0, "+1": 5.5, "+2": 0.0},
    "128C": {"-2": 0.1, "-1": 1.5, "+1": 5.0, "+2": 0.0},
}


def test_correction_matrix_targets_by_isotope() -> None:
    m = isotope_correction_matrix("TMT10", LOT_SHEET)
    ch = TMT10.channels.index
    # +1 (13C) of 126 lands on 127C, not 127N; +2 on 128C.
    assert m[ch("127C"), ch("126")] == pytest.approx(0.07)
    assert m[ch("127N"), ch("126")] == 0.0
    assert m[ch("128C"), ch("126")] == pytest.approx(0.002)
    assert m[ch("126"), ch("126")] == pytest.approx(1 - 0.072)
    # -1 of 128C -> 127C, of 128N -> 127N; -1 of 127N has only 126 nominally below it.
    assert m[ch("127C"), ch("128C")] == pytest.approx(0.015)
    assert m[ch("127N"), ch("128N")] == pytest.approx(0.01)
    assert m[ch("126"), ch("127N")] == pytest.approx(0.004)
    # Channels absent from the sheet are pure.
    assert m[ch("131N"), ch("131N")] == 1.0
    # Explicit labels and a DataFrame give the same matrix.
    explicit = {"126": {"+13C": 7.0, "+2x13C": 0.2}}
    assert isotope_correction_matrix("TMT10", explicit)[ch("127C"), ch("126")] == pytest.approx(0.07)
    frame = pd.DataFrame(LOT_SHEET).T
    np.testing.assert_allclose(isotope_correction_matrix("TMT10", frame), m)


def test_correction_recovers_true_intensities() -> None:
    m = isotope_correction_matrix("TMT10", LOT_SHEET)
    true = np.array([1000.0, 0.0, 500.0, 2000.0, 50.0, 800.0, 0.0, 300.0, 1200.0, 700.0])
    observed = m @ true
    np.testing.assert_allclose(correct_isotope_impurities(observed, m), true, atol=1e-9)
    np.testing.assert_allclose(correct_isotope_impurities(observed, LOT_SHEET, "TMT10"), true, atol=1e-9)

    spectrum = _spectrum(list(TMT10.reporter_mzs), list(observed))
    ions = extract_reporter_ions(spectrum, "TMT10", impurities=LOT_SHEET)
    assert ions.corrected
    np.testing.assert_allclose(ions.intensity, true, atol=1e-6)
    np.testing.assert_allclose(ions.raw_intensity, observed)


def test_correction_is_non_negative() -> None:
    m = isotope_correction_matrix("TMT10", LOT_SHEET)
    # 127C read low relative to the 7 % that 126 spills into it: exact solve is negative.
    observed = m @ np.array([10000.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    observed[2] = 100.0
    assert np.linalg.solve(m, observed).min() < 0
    corrected = correct_isotope_impurities(observed, m)
    assert corrected.min() >= 0
    assert corrected[2] == 0.0
    batch = correct_isotope_impurities(np.vstack([observed, m @ np.ones(10)]), m)
    assert batch.shape == (2, 10)
    np.testing.assert_allclose(batch[1], 1.0, atol=1e-9)


def test_bad_impurities_raise() -> None:
    with pytest.raises(SpxtacularError, match="not in TMT10"):
        isotope_correction_matrix("TMT10", {"134N": {"+1": 1.0}})
    with pytest.raises(SpxtacularError, match="Unrecognised impurity shift"):
        isotope_correction_matrix("TMT10", {"126": {"+1C": 1.0}})
    with pytest.raises(SpxtacularError, match="more than 100"):
        isotope_correction_matrix("TMT10", {"126": {"+1": 80.0, "+2": 30.0}})
    with pytest.raises(SpxtacularError, match=">= 0"):
        isotope_correction_matrix("TMT10", {"126": {"+1": -1.0}})
    with pytest.raises(SpxtacularError, match="shape"):
        correct_isotope_impurities(np.ones(10), np.eye(6))
    with pytest.raises(SpxtacularError, match="needs plex"):
        correct_isotope_impurities(np.ones(10), LOT_SHEET)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def test_reporter_ion_table() -> None:
    spectra = [
        MsnSpectrum(_reporter_spectrum().mz, _reporter_spectrum().intensity, scan_number=7, ms_level=2, rt=12.5),
        MsnSpectrum(np.array([100.0]), np.array([1.0]), scan_number=8, ms_level=1),
        MsnSpectrum(
            _reporter_spectrum(skip=("126",)).mz, _reporter_spectrum(skip=("126",)).intensity, scan_number=9, ms_level=2
        ),
    ]
    table = reporter_ion_table(iter(spectra), "TMT10", ms_level=2, include_errors=True)
    assert list(table["scan_number"]) == [7, 9]
    assert list(table["spectrum_index"]) == [0, 2]
    assert table.loc[0, "127N"] == 2000.0
    assert table.loc[1, "126"] == 0.0
    assert np.isnan(table.loc[1, "126_ppm_error"])
    assert table.loc[0, "126_ppm_error"] == pytest.approx(0.0, abs=1e-6)

    normalized = reporter_ion_table(spectra, "TMT10", normalize="sum", impurities=LOT_SHEET)
    assert len(normalized) == 3
    sums = normalized[list(TMT10.channels)].sum(axis=1)
    np.testing.assert_allclose(sums, [1.0, 0.0, 1.0])

    empty = reporter_ion_table([], "iTRAQ4")
    assert list(empty.columns) == [
        "spectrum_index",
        "scan_number",
        "native_id",
        "ms_level",
        "rt",
        "114",
        "115",
        "116",
        "117",
    ]
    assert empty.empty


def test_reader_like_object_reads_ms2() -> None:
    class FakeReader:
        def __init__(self) -> None:
            self.ms2 = [MsnSpectrum(_reporter_spectrum().mz, _reporter_spectrum().intensity, ms_level=2)]

    table = reporter_ion_table(FakeReader(), "TMT10")
    assert len(table) == 1
