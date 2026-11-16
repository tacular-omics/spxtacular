"""Tests for :func:`spxtacular.recalibrate` and :class:`spxtacular.CalibrationProvenance`.

The sections follow the six validation items of the approved design: exact synthetic
recovery, no-op on clean data, real-PSM improvement, one test per refusal, robustness of
the median estimator, and the provenance round trip.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import peptacular as pt
import pytest
from jsonschema import validate

from spxtacular import (
    CalibrationProvenance,
    MatchedFragment,
    MsnSpectrum,
    Spectrum,
    get_json_schema,
    match_fragments,
    recalibrate,
    score,
)
from spxtacular.errors import SpxtacularError

PEPTIDE = "VLHPLEGAVVIIFK"
_FIXTURE = Path(__file__).resolve().parents[1] / "paper" / "abstract" / "data" / "stages.json"
_needs_fixture = pytest.mark.skipif(not _FIXTURE.exists(), reason="the paper's PSM fixture is not in this tree")


def _fragments(peptide: str = PEPTIDE):
    return pt.fragment(peptide, ion_types=("b", "y"), charges=(1,))


def _exact_spectrum(drift_ppm: float = 0.0, peptide: str = PEPTIDE):
    """A spectrum whose peaks are the theoretical fragment m/z times ``1 + drift``."""
    fragments = _fragments(peptide)
    true_mz = np.array(sorted(float(f.mz) for f in fragments))
    spectrum = Spectrum(
        mz=true_mz * (1.0 + drift_ppm * 1e-6),
        intensity=np.linspace(100.0, 1000.0, len(true_mz)),
    )
    matches = match_fragments(spectrum, fragments, tolerance=0.1)
    return fragments, true_mz, spectrum, matches


def _match(fragment, peak_index: int, peak_mz: float, ppm_error: float) -> MatchedFragment:
    """A hand-built match with an exactly consistent ``da_error``."""
    theoretical = peak_mz / (1.0 + ppm_error * 1e-6)
    return MatchedFragment(
        fragment=fragment,
        peak_index=peak_index,
        peak_mz=peak_mz,
        peak_intensity=100.0,
        intensity_pct=1.0,
        ppm_error=ppm_error,
        da_error=peak_mz - theoretical,
    )


def _real_stage(stage: str = "denoised"):
    """The real PSM the paper figure uses, at one pipeline stage, with its matches."""
    data = json.loads(_FIXTURE.read_text())
    spectrum = Spectrum(
        mz=np.asarray(data[stage]["mz"], dtype=np.float64),
        intensity=np.asarray(data[stage]["intensity"], dtype=np.float64),
    )
    fragments = _fragments(data["peptide"])
    tolerance = float(data["tolerance_ppm"])
    matches = match_fragments(spectrum, fragments, tolerance=tolerance, tolerance_unit="ppm")
    return spectrum, fragments, matches, tolerance


# ---------------------------------------------------------------------------
# Validation item 1: exact synthetic recovery (the sign-error test)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("drift", [-20.0, -5.0, 0.0, 5.0, 20.0])
def test_injected_offset_is_recovered_exactly(drift: float) -> None:
    """A known drift comes back with the right sign and the original m/z is restored."""
    _, true_mz, spectrum, matches = _exact_spectrum(drift)
    assert len(matches) == 28

    fixed = recalibrate(spectrum, matches)

    assert fixed.calibration is not None
    assert fixed.calibration.offset_ppm == pytest.approx(drift, abs=0.05)
    assert np.allclose(fixed.mz, true_mz, rtol=1e-6)
    # Nothing is left to correct, so the residual spread collapses.
    assert fixed.calibration.residual_ppm_sd == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("drift", [-20.0, -5.0, 5.0, 20.0])
@_needs_fixture
def test_injected_offset_is_recovered_on_a_real_spectrum(drift: float) -> None:
    """On real data the fit returns the spectrum's own median error plus the injection."""
    spectrum, fragments, matches, tolerance = _real_stage()
    base = recalibrate(spectrum, matches, tolerance=tolerance, tolerance_unit="ppm").calibration
    assert base is not None

    drifted = spectrum.update(mz=spectrum.mz * (1.0 + drift * 1e-6))
    drifted_matches = match_fragments(drifted, fragments, tolerance=50.0, tolerance_unit="ppm")
    fixed = recalibrate(drifted, drifted_matches, tolerance=50.0, tolerance_unit="ppm")

    assert fixed.calibration is not None
    assert fixed.calibration.offset_ppm == pytest.approx(base.offset_ppm + drift, abs=0.05)
    assert np.allclose(fixed.mz, spectrum.mz / (1.0 + base.offset_ppm * 1e-6), rtol=1e-6)


def test_unmatched_peaks_are_corrected_too() -> None:
    """The fit uses the matched ions; the correction applies to every peak."""
    fragments, true_mz, _, _ = _exact_spectrum()
    noise = np.array([123.456, 999.999, 1400.5])
    mz = np.sort(np.concatenate([true_mz, noise])) * (1.0 + 7.0e-6)
    spectrum = Spectrum(mz=mz, intensity=np.full(len(mz), 50.0))
    matches = match_fragments(spectrum, fragments, tolerance=0.1)

    fixed = recalibrate(spectrum, matches)

    assert fixed.calibration is not None
    assert fixed.calibration.n_matches == 28
    assert np.allclose(fixed.mz, np.sort(np.concatenate([true_mz, noise])), rtol=1e-6)


# ---------------------------------------------------------------------------
# Validation item 2: no-op on clean data
# ---------------------------------------------------------------------------


@_needs_fixture
def test_fit_on_clean_data_is_the_median_and_does_not_hurt() -> None:
    spectrum, fragments, matches, tolerance = _real_stage()
    errors = np.array([m.ppm_error for m in matches])
    before = score(spectrum, fragments, tolerance=tolerance, tolerance_unit="ppm")

    fixed = recalibrate(spectrum, matches, tolerance=tolerance, tolerance_unit="ppm")
    after = score(fixed, fragments, tolerance=tolerance, tolerance_unit="ppm")

    assert fixed.calibration is not None
    assert fixed.calibration.offset_ppm == pytest.approx(float(np.median(errors)))
    assert after["mean_ppm_error"] <= before["mean_ppm_error"]


def test_zero_error_data_leaves_the_axis_alone() -> None:
    _, true_mz, spectrum, matches = _exact_spectrum(0.0)

    fixed = recalibrate(spectrum, matches)

    assert fixed.calibration is not None
    assert fixed.calibration.offset_ppm == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(fixed.mz, true_mz, rtol=1e-12)


# ---------------------------------------------------------------------------
# Validation item 3: the real PSM improves
# ---------------------------------------------------------------------------


@_needs_fixture
def test_real_psm_mean_ppm_error_falls() -> None:
    """Measured on the denoised stage of the paper's PSM: 1.1925 -> 0.8436 ppm, 22 matches.

    The design quoted 1.083 -> 0.655 from the paper figure's own match list, which is
    taken on the *decharged* spectrum (``compute.py:16-18``, ``charges=0``). That is the
    one input :func:`recalibrate` refuses, since a neutral mass is not an m/z, so the
    equivalent numbers here are the charged stage the function accepts.
    """
    spectrum, fragments, matches, tolerance = _real_stage()
    before = score(spectrum, fragments, tolerance=tolerance, tolerance_unit="ppm")
    assert len(matches) == 22
    assert before["mean_ppm_error"] == pytest.approx(1.1925, abs=5e-4)

    fixed = recalibrate(spectrum, matches, tolerance=tolerance, tolerance_unit="ppm")
    after = score(fixed, fragments, tolerance=tolerance, tolerance_unit="ppm")

    assert after["mean_ppm_error"] == pytest.approx(0.8436, abs=5e-4)
    assert after["matched_fraction"] >= before["matched_fraction"]
    assert fixed.calibration is not None
    assert fixed.calibration.offset_ppm == pytest.approx(0.7898, abs=5e-4)
    assert fixed.calibration.mz_span[0] < fixed.calibration.mz_span[1]


@_needs_fixture
def test_linear_model_runs_on_the_real_psm() -> None:
    """The linear model is opt-in: it tightens the residual spread, not the mean error."""
    spectrum, fragments, matches, tolerance = _real_stage()
    offset = recalibrate(spectrum, matches, tolerance=tolerance, tolerance_unit="ppm")
    linear = recalibrate(
        spectrum,
        matches,
        model="linear",
        estimator="least_squares",
        tolerance=tolerance,
        tolerance_unit="ppm",
    )

    assert offset.calibration is not None
    assert linear.calibration is not None
    assert linear.calibration.slope_ppm_per_mz is not None
    assert linear.calibration.residual_ppm_sd < offset.calibration.residual_ppm_sd


# ---------------------------------------------------------------------------
# Validation item 4: the refusals, one test each
# ---------------------------------------------------------------------------


def test_refuses_too_few_matches() -> None:
    _, _, spectrum, matches = _exact_spectrum(5.0)

    with pytest.raises(SpxtacularError, match=r"min_matches=8 matched fragments; got 5"):
        recalibrate(spectrum, matches[:5])


def test_refuses_a_narrow_span_for_the_linear_model() -> None:
    fragments = _fragments()
    mz = np.linspace(500.0, 560.0, 8)
    spectrum = Spectrum(mz=mz, intensity=np.full(8, 100.0))
    matches = [_match(fragments[i], i, float(value), 1.0) for i, value in enumerate(mz)]

    with pytest.raises(SpxtacularError, match=r"min_mz_span=200.0; got 60"):
        recalibrate(spectrum, matches, model="linear", estimator="least_squares")

    # The offset model has no span requirement, and still fits the same points.
    assert recalibrate(spectrum, matches).calibration is not None


def test_refuses_a_correction_larger_than_the_bound() -> None:
    _, _, spectrum, matches = _exact_spectrum(5.0)

    with pytest.raises(SpxtacularError, match=r"correction of 5[\d.]* ppm, above max_abs_ppm=0.1"):
        recalibrate(spectrum, matches, max_abs_ppm=0.1)


def test_refuses_too_few_distinct_peaks() -> None:
    """Ten fragments on one peak are one observation, not ten."""
    fragments = _fragments()
    spectrum = Spectrum(mz=np.array([500.0]), intensity=np.array([100.0]))
    matches = [_match(fragments[i], 0, 500.0, 1.0) for i in range(10)]

    with pytest.raises(SpxtacularError, match=r"min_matches=8 distinct matched peaks; got 1 from 10"):
        recalibrate(spectrum, matches)


def test_refuses_a_decharged_spectrum() -> None:
    _, _, spectrum, matches = _exact_spectrum(5.0)
    neutral = spectrum.update(charge=np.zeros(len(spectrum.mz), dtype=np.int32))

    with pytest.raises(SpxtacularError, match=r"neutral masses \(28 of 28 peaks have charge 0\)"):
        recalibrate(neutral, matches)


def test_refuses_a_non_finite_fit() -> None:
    fragments = _fragments()
    mz = np.linspace(400.0, 1400.0, 10)
    spectrum = Spectrum(mz=mz, intensity=np.full(10, 100.0))
    matches = [_match(fragments[i], i, float(value), float("nan")) for i, value in enumerate(mz)]

    with pytest.raises(SpxtacularError, match=r"non-finite offset_ppm"):
        recalibrate(spectrum, matches)


def test_refuses_a_correction_that_inverts_the_axis() -> None:
    """The guard behind the ppm bound: a factor of zero or less is not a drift."""
    fragments = _fragments()
    mz = np.linspace(400.0, 1400.0, 10)
    spectrum = Spectrum(mz=mz, intensity=np.full(10, 100.0))
    matches = [_match(fragments[i], i, float(value), -1.5e6) for i, value in enumerate(mz)]

    with pytest.raises(SpxtacularError, match=r"inverts the m/z axis"):
        recalibrate(spectrum, matches, max_abs_ppm=2.0e6)


def test_refuses_an_unknown_model_or_estimator() -> None:
    _, _, spectrum, matches = _exact_spectrum(5.0)

    with pytest.raises(SpxtacularError, match=r"unknown calibration model 'quadratic'"):
        recalibrate(spectrum, matches, model="quadratic")  # ty: ignore[invalid-argument-type]
    with pytest.raises(SpxtacularError, match=r"unknown calibration estimator 'theil_sen'"):
        recalibrate(spectrum, matches, estimator="theil_sen")  # ty: ignore[invalid-argument-type]


def test_refuses_the_median_estimator_for_the_linear_model() -> None:
    """The linear model is a least-squares fit; the pairing is not silently changed."""
    _, _, spectrum, matches = _exact_spectrum(5.0)

    with pytest.raises(SpxtacularError, match=r"model='linear' is fitted by least squares"):
        recalibrate(spectrum, matches, model="linear")


# ---------------------------------------------------------------------------
# Validation item 5: the median estimator survives wrong matches
# ---------------------------------------------------------------------------


def test_median_resists_wrong_matches_and_least_squares_does_not() -> None:
    """Twenty percent of the matches are mis-assigned by +60 ppm; the truth is +5."""
    fragments = _fragments()
    mz = np.linspace(300.0, 1300.0, 25)
    spectrum = Spectrum(mz=mz, intensity=np.full(25, 100.0))
    matches = [
        _match(fragments[i % len(fragments)], i, float(value), 60.0 if i % 5 == 0 else 5.0)
        for i, value in enumerate(mz)
    ]
    assert sum(1 for m in matches if m.ppm_error > 50.0) == 5

    median = recalibrate(spectrum, matches)
    least_squares = recalibrate(spectrum, matches, estimator="least_squares")

    assert median.calibration is not None
    assert least_squares.calibration is not None
    assert median.calibration.offset_ppm == pytest.approx(5.0, abs=0.5)
    assert abs(least_squares.calibration.offset_ppm - 5.0) > 0.5
    # The spread it leaves behind is the honest signal that something is wrong.
    assert median.calibration.residual_ppm_sd > 10.0


# ---------------------------------------------------------------------------
# Validation item 6: the provenance round trip
# ---------------------------------------------------------------------------


def test_provenance_round_trips_through_dict_and_json() -> None:
    _, _, spectrum, matches = _exact_spectrum(5.0)
    fixed = recalibrate(spectrum, matches)
    assert fixed.calibration is not None

    assert CalibrationProvenance.from_dict(fixed.calibration.to_dict()) == fixed.calibration

    payload = fixed.to_dict()
    assert payload["schema_version"] == 3
    assert payload["metadata"]["calibration"]["schema_version"] == 1
    validate(payload, get_json_schema("spectrum"))
    restored = Spectrum.from_json(fixed.to_json())
    assert restored == fixed
    assert restored.calibration == fixed.calibration


def test_provenance_round_trips_through_npz(tmp_path: Path) -> None:
    _, _, spectrum, matches = _exact_spectrum(5.0)
    fixed = recalibrate(spectrum, matches)
    path = tmp_path / "calibrated.npz"

    fixed.save(path)

    assert Spectrum.load(path) == fixed


def test_msn_spectrum_carries_the_record() -> None:
    _, _, spectrum, matches = _exact_spectrum(5.0)
    msn = MsnSpectrum(mz=spectrum.mz, intensity=spectrum.intensity, scan_number=17555, ms_level=2)

    fixed = recalibrate(msn, matches)

    assert isinstance(fixed, MsnSpectrum)
    assert fixed.scan_number == 17555
    assert MsnSpectrum.from_json(fixed.to_json()) == fixed
    validate(fixed.to_dict(), get_json_schema("spectrum"))


def test_a_second_recalibration_records_both() -> None:
    _, true_mz, spectrum, matches = _exact_spectrum(5.0)
    once = recalibrate(spectrum, matches)
    second_matches = match_fragments(once, _fragments(), tolerance=0.1)

    twice = recalibrate(once, second_matches)

    assert twice.calibration is not None
    assert twice.calibration.previous is not None
    assert twice.calibration.previous.offset_ppm == pytest.approx(5.0, abs=0.05)
    assert twice.calibration.offset_ppm == pytest.approx(0.0, abs=0.05)
    assert np.allclose(twice.mz, true_mz, rtol=1e-6)
    assert Spectrum.from_json(twice.to_json()) == twice
    validate(twice.to_dict(), get_json_schema("spectrum"))


def test_rejects_an_unsupported_provenance_schema() -> None:
    _, _, spectrum, matches = _exact_spectrum(5.0)
    payload = recalibrate(spectrum, matches).to_dict()
    payload["metadata"]["calibration"]["schema_version"] = 2

    with pytest.raises(SpxtacularError, match=r"calibration.schema_version must be 1"):
        Spectrum.from_dict(payload)


def test_schema_rejects_an_incomplete_calibration_record() -> None:
    from jsonschema import ValidationError

    _, _, spectrum, matches = _exact_spectrum(5.0)
    payload = recalibrate(spectrum, matches).to_dict()
    del payload["metadata"]["calibration"]["offset_ppm"]

    with pytest.raises(ValidationError, match="offset_ppm"):
        validate(payload, get_json_schema("spectrum"))


# ---------------------------------------------------------------------------
# The transform contract
# ---------------------------------------------------------------------------


def test_returns_a_new_spectrum_sharing_no_arrays() -> None:
    _, _, spectrum, matches = _exact_spectrum(5.0)
    original_mz = spectrum.mz.copy()

    fixed = recalibrate(spectrum, matches)

    assert fixed is not spectrum
    assert np.array_equal(spectrum.mz, original_mz)
    assert not np.shares_memory(fixed.mz, spectrum.mz)
    assert not np.shares_memory(fixed.intensity, spectrum.intensity)
    assert spectrum.calibration is None


def test_inplace_updates_the_spectrum() -> None:
    _, true_mz, spectrum, matches = _exact_spectrum(5.0)

    returned = recalibrate(spectrum, matches, inplace=True)

    assert returned is spectrum
    assert spectrum.calibration is not None
    assert np.allclose(spectrum.mz, true_mz, rtol=1e-6)


def test_optional_arrays_are_carried_over() -> None:
    fragments, true_mz, _, _ = _exact_spectrum()
    n = len(true_mz)
    spectrum = Spectrum(
        mz=true_mz * (1.0 + 5.0e-6),
        intensity=np.full(n, 100.0),
        charge=np.ones(n, dtype=np.int32),
        im=np.linspace(0.9, 1.1, n),
        iso_score=np.full(n, 0.5),
    )
    matches = match_fragments(spectrum, fragments, tolerance=0.1)

    fixed = recalibrate(spectrum, matches)

    for before, after in (
        (spectrum.charge, fixed.charge),
        (spectrum.im, fixed.im),
        (spectrum.iso_score, fixed.iso_score),
    ):
        assert before is not None and after is not None
        assert np.array_equal(before, after)
        assert not np.shares_memory(before, after)
