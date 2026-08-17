"""
Tests for spxtacular.scoring.score.

Fragments are mocked with MagicMock — score() only accesses .mz, .ion_type,
and .position through match_fragments and the internal helpers.
"""

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from spxtacular.core import Spectrum
from spxtacular.enums import PeakSelectionLike, ToleranceLike
from spxtacular.scoring import _binom_log10_survival, _count_unique_ions, score

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXPECTED_KEYS = {
    "hyperscore",
    "probability_score",
    "total_matched_intensity",
    "matched_fraction",
    "intensity_fraction",
    "mean_ppm_error",
    "spectral_angle",
    "longest_run",
}


def _make_frag(mz: float, ion_type: str = "b", position: int = 1) -> MagicMock:
    f = MagicMock()
    f.mz = mz
    f.ion_type = ion_type
    f.position = position
    return f


def _spectrum() -> Spectrum:
    mz = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float64)
    intensity = np.array([10.0, 50.0, 20.0, 15.0], dtype=np.float64)
    return Spectrum(mz=mz, intensity=intensity)


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------


def test_score_returns_all_expected_keys() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert set(result.keys()) == _EXPECTED_KEYS


def test_score_all_values_are_floats() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    for key, val in result.items():
        assert isinstance(val, float), f"{key} is not float"


# ---------------------------------------------------------------------------
# No-match baseline
# ---------------------------------------------------------------------------


def test_score_no_matches_hyperscore_is_zero() -> None:
    spec = _spectrum()
    frag = _make_frag(999.0)  # far outside spectrum
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert result["hyperscore"] == pytest.approx(0.0)


def test_score_no_matches_total_matched_intensity_is_zero() -> None:
    spec = _spectrum()
    frag = _make_frag(999.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert result["total_matched_intensity"] == pytest.approx(0.0)


def test_score_no_matches_matched_fraction_is_zero() -> None:
    spec = _spectrum()
    frag = _make_frag(999.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert result["matched_fraction"] == pytest.approx(0.0)


def test_score_no_matches_intensity_fraction_is_zero() -> None:
    spec = _spectrum()
    frag = _make_frag(999.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert result["intensity_fraction"] == pytest.approx(0.0)


def test_score_no_matches_spectral_angle_is_zero() -> None:
    spec = _spectrum()
    frag = _make_frag(999.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert result["spectral_angle"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Matched fragment — positive scores
# ---------------------------------------------------------------------------


def test_hyperscore_positive_when_fragment_matches() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert result["hyperscore"] > 0.0


def test_total_matched_intensity_equals_matched_peak_intensity() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0)  # matches peak at index 1 with intensity 50.0
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert result["total_matched_intensity"] == pytest.approx(50.0)


def test_mean_ppm_error_zero_for_perfect_match() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0)  # exact m/z
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert result["mean_ppm_error"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Bounds checks
# ---------------------------------------------------------------------------


def test_matched_fraction_in_zero_to_one() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert 0.0 <= result["matched_fraction"] <= 1.0


def test_intensity_fraction_in_zero_to_one() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert 0.0 <= result["intensity_fraction"] <= 1.0


def test_spectral_angle_in_minus_one_to_one() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert -1.0 <= result["spectral_angle"] <= 1.0


# ---------------------------------------------------------------------------
# longest_run — consecutive ion series
# ---------------------------------------------------------------------------


def test_longest_run_three_consecutive_b_ions() -> None:
    spec = _spectrum()
    # b1=100, b2=200, b3=300 — all match spectrum peaks
    frags = [_make_frag(float(pos * 100), ion_type="b", position=pos) for pos in [1, 2, 3]]
    result = score(spec, frags, tolerance=0.02, tolerance_type="da")
    assert result["longest_run"] >= 3.0


def test_longest_run_zero_when_no_match() -> None:
    spec = _spectrum()
    frag = _make_frag(999.0, ion_type="b", position=1)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert result["longest_run"] == pytest.approx(0.0)


def test_longest_run_one_for_single_match() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0, ion_type="b", position=5)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert result["longest_run"] >= 1.0


# ---------------------------------------------------------------------------
# Empty fragments list
# ---------------------------------------------------------------------------


def test_score_empty_fragments_returns_all_zeros() -> None:
    spec = _spectrum()
    result = score(spec, [], tolerance=0.02, tolerance_type="da")
    for key in (
        "hyperscore",
        "total_matched_intensity",
        "matched_fraction",
        "intensity_fraction",
        "mean_ppm_error",
        "spectral_angle",
        "longest_run",
    ):
        assert result[key] == pytest.approx(0.0), f"{key} should be 0 with no fragments"


# ---------------------------------------------------------------------------
# _binom_log10_survival edge cases
# ---------------------------------------------------------------------------


def test_binom_log10_survival_k_zero_returns_zero() -> None:
    assert _binom_log10_survival(0, 10, 0.5) == pytest.approx(0.0)


def test_binom_log10_survival_k_negative_returns_zero() -> None:
    assert _binom_log10_survival(-1, 10, 0.5) == pytest.approx(0.0)


def test_binom_log10_survival_k_greater_than_n_returns_neg_inf() -> None:
    import math

    assert _binom_log10_survival(11, 10, 0.5) == -math.inf


def test_binom_log10_survival_p_zero_returns_neg_inf() -> None:
    import math

    assert _binom_log10_survival(1, 10, 0.0) == -math.inf


def test_binom_log10_survival_p_one_returns_zero() -> None:
    assert _binom_log10_survival(1, 10, 1.0) == pytest.approx(0.0)


def test_binom_log10_survival_p_greater_than_one_returns_zero() -> None:
    assert _binom_log10_survival(1, 10, 1.5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _count_unique_ions with dict input
# ---------------------------------------------------------------------------


def test_count_unique_ions_dict_sums_lengths() -> None:
    from peptacular import IonType

    frag_dict: dict = {
        (IonType.B, 1): [100.0, 200.0, 300.0],
        (IonType.Y, 1): [400.0, 500.0],
    }
    result = _count_unique_ions(frag_dict)
    assert result == 5


def test_count_unique_ions_dict_empty_returns_zero() -> None:
    result = _count_unique_ions({})
    assert result == 0


# ---------------------------------------------------------------------------
# _probability_score edge cases via public score()
# ---------------------------------------------------------------------------


def test_probability_score_zero_for_empty_spectrum() -> None:
    spec = Spectrum(
        mz=np.array([], dtype=np.float64),
        intensity=np.array([], dtype=np.float64),
    )
    frag = _make_frag(200.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert result["probability_score"] == pytest.approx(0.0)


def test_probability_score_zero_for_single_peak_zero_range() -> None:
    spec = Spectrum(
        mz=np.array([200.0], dtype=np.float64),
        intensity=np.array([100.0], dtype=np.float64),
    )
    frag = _make_frag(200.0)
    result = score(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert result["probability_score"] == pytest.approx(0.0)


def test_probability_score_ppm_tolerance_path() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0)
    result = score(spec, [frag], tolerance=10.0, tolerance_type="ppm")
    assert result["probability_score"] >= 0.0


# ---------------------------------------------------------------------------
# _probability_score must not assume sorted m/z
# ---------------------------------------------------------------------------


class TestProbabilityScoreIgnoresPeakOrder:
    """The m/z range is a span, not ``mz[-1] - mz[0]``.

    Unsorted input is ordinary: a timsTOF frame is ordered by ion-mobility scan,
    which is why ``match_fragments`` sorts a working copy. Taking the endpoints
    made the same peaks and matches score differently for a reordering.
    """

    def _frags(self) -> list[MagicMock]:
        return [_make_frag(float(pos * 100), ion_type="b", position=pos) for pos in (1, 2, 3)]

    @pytest.mark.parametrize("tolerance_type", ["da", "ppm"])
    def test_shuffled_spectrum_scores_the_same(self, tolerance_type: ToleranceLike) -> None:
        mz = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float64)
        intensity = np.array([10.0, 50.0, 20.0, 15.0], dtype=np.float64)
        order = np.array([2, 0, 3, 1])  # ion-mobility-style ordering

        tolerance = 0.02 if tolerance_type == "da" else 50.0
        sorted_result = score(
            Spectrum(mz=mz.copy(), intensity=intensity.copy()),
            self._frags(),
            tolerance=tolerance,
            tolerance_type=tolerance_type,
        )
        shuffled_result = score(
            Spectrum(mz=mz[order], intensity=intensity[order]),
            self._frags(),
            tolerance=tolerance,
            tolerance_type=tolerance_type,
        )
        assert sorted_result["probability_score"] > 0.0
        assert shuffled_result["probability_score"] == pytest.approx(sorted_result["probability_score"])

    def test_descending_spectrum_still_scores(self) -> None:
        """Ending below the start gave a negative range, silently scored as 0.0."""
        descending = Spectrum(
            mz=np.array([400.0, 300.0, 200.0, 100.0], dtype=np.float64),
            intensity=np.array([15.0, 20.0, 50.0, 10.0], dtype=np.float64),
        )
        result = score(descending, self._frags(), tolerance=0.02, tolerance_type="da")
        assert result["probability_score"] > 0.0


# ---------------------------------------------------------------------------
# matched_fraction stays a fraction under peak_selection="all"
# ---------------------------------------------------------------------------


class TestMatchedFractionCountsIons:
    """Numerator and denominator both count ions.

    With ``peak_selection="all"`` one theoretical ion claims every peak within
    tolerance, so counting matched *peaks* pushed the "fraction" above 1.0 —
    the same failure mode ``_spectral_angle`` guards against by trimming its
    observed vector back to ``n_unique``.
    """

    def _crowded(self) -> Spectrum:
        """Three peaks inside one tolerance window, plus an unrelated peak."""
        return Spectrum(
            mz=np.array([199.99, 200.0, 200.01, 400.0], dtype=np.float64),
            intensity=np.array([10.0, 50.0, 20.0, 15.0], dtype=np.float64),
        )

    def test_one_ion_on_three_peaks_is_a_full_match_not_three(self) -> None:
        frag = _make_frag(200.0, ion_type="b", position=1)
        result = score(self._crowded(), [frag], tolerance=0.05, tolerance_type="da", peak_selection="all")
        assert result["matched_fraction"] == pytest.approx(1.0)

    def test_half_the_ions_matched_is_a_half(self) -> None:
        frags = [
            _make_frag(200.0, ion_type="b", position=1),  # claims all three peaks
            _make_frag(999.0, ion_type="b", position=2),  # matches nothing
        ]
        result = score(self._crowded(), frags, tolerance=0.05, tolerance_type="da", peak_selection="all")
        assert result["matched_fraction"] == pytest.approx(0.5)

    @pytest.mark.parametrize("peak_selection", ["closest", "largest", "all"])
    def test_never_exceeds_one_for_any_peak_selection(self, peak_selection: PeakSelectionLike) -> None:
        frags = [_make_frag(200.0, ion_type="b", position=p) for p in (1, 2)]
        result = score(self._crowded(), frags, tolerance=0.05, tolerance_type="da", peak_selection=peak_selection)
        assert 0.0 < result["matched_fraction"] <= 1.0


# ---------------------------------------------------------------------------
# score() with dict fragments input
# ---------------------------------------------------------------------------


def test_score_dict_fragments_returns_expected_keys() -> None:
    from peptacular import IonType

    spec = _spectrum()
    frag_dict: dict = {
        (IonType.B, 1): [100.0, 200.0],
    }
    result = score(spec, frag_dict, tolerance=0.02, tolerance_type="da")
    assert set(result.keys()) == _EXPECTED_KEYS


# ---------------------------------------------------------------------------
# Hyperscore: the formula itself, not just "> 0"
# ---------------------------------------------------------------------------


class TestHyperscoreFormula:
    """Pin the hyperscore to X!Tandem.

    Previously the only assertions were ``> 0.0`` and ``== 0`` with no matches,
    so the formula could have been anything at all. These pin it to the published
    definition and to the property that makes it discriminating.
    """

    def _by_spectrum(self):
        """A spectrum with two b ions and two y ions of known intensity."""
        import peptacular as pt

        frags = list(pt.fragment("PEPTIDEK", ion_types=("b", "y"), charges=(1,)))
        b = [f for f in frags if str(f.ion_type) == "b"][:2]
        y = [f for f in frags if str(f.ion_type) == "y"][:2]
        chosen = b + y
        mz = np.array([f.mz for f in chosen], dtype=np.float64)
        # sum I_b = 30, sum I_y = 70
        inten = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
        order = np.argsort(mz)
        return Spectrum(mz=mz[order], intensity=inten[order]), chosen, b, y

    def test_matches_the_xtandem_formula_exactly(self) -> None:
        spec, frags, b, y = self._by_spectrum()
        result = score(spec, frags, tolerance=0.01, tolerance_type="da")
        # X!Tandem: log10(sum(I_b) * sum(I_y) * n_b! * n_y!) with sums 30 and 70, n=2 each
        expected = math.log10(30.0 * 70.0 * math.factorial(2) * math.factorial(2))
        assert result["hyperscore"] == pytest.approx(expected, rel=1e-12)

    def test_a_searched_series_with_no_signal_collapses_the_score(self) -> None:
        """The product is what discriminates: y-only evidence must not look good.

        A sum over all matched peaks would happily score this; the product form
        rejects it, which is what X!Tandem does and why it separates decoys better.
        """
        spec, frags, b, y = self._by_spectrum()
        # Keep only the y peaks in the spectrum; b was still searched for.
        y_mz = np.array(sorted(f.mz for f in y), dtype=np.float64)
        y_only = Spectrum(mz=y_mz, intensity=np.array([30.0, 40.0]))
        result = score(y_only, frags, tolerance=0.01, tolerance_type="da")
        assert result["matched_fraction"] > 0.0, "the y ions really did match"
        assert result["hyperscore"] == 0.0

    def test_generalises_beyond_b_and_y(self) -> None:
        """An ETD c/z search scores; X!Tandem's hardcoded b/y would give zero."""
        import peptacular as pt

        frags = pt.fragment("PEPTIDEK", ion_types=("c", "z"), charges=(1,))
        mz = np.array(sorted(f.mz for f in frags), dtype=np.float64)
        spec = Spectrum(mz=mz, intensity=np.linspace(10.0, 100.0, len(mz)))
        assert score(spec, frags, tolerance=0.01, tolerance_type="da")["hyperscore"] > 0.0

    def test_is_intensity_scale_dependent(self) -> None:
        """Documented caveat — pin it so it cannot change silently."""
        spec, frags, _, _ = self._by_spectrum()
        base = score(spec, frags, tolerance=0.01, tolerance_type="da")["hyperscore"]
        scaled = Spectrum(mz=spec.mz, intensity=spec.intensity * 100.0)
        got = score(scaled, frags, tolerance=0.01, tolerance_type="da")["hyperscore"]
        # Two series, so scaling every intensity by s shifts the score by 2*log10(s).
        assert got == pytest.approx(base + 2 * math.log10(100.0), rel=1e-12)


# ---------------------------------------------------------------------------
# Spectral angle against predicted intensities
# ---------------------------------------------------------------------------


class TestSpectralAngleWithPrediction:
    def _setup(self):
        import peptacular as pt

        frags = list(pt.fragment("PEPTIDEK", ion_types=("b", "y"), charges=(1,)))
        pred = np.linspace(1.0, 0.1, len(frags))
        mz = np.array([f.mz for f in frags], dtype=np.float64)
        order = np.argsort(mz)
        return frags, pred, mz, order

    def test_identical_to_prediction_scores_one(self) -> None:
        frags, pred, mz, order = self._setup()
        spec = Spectrum(mz=mz[order], intensity=pred[order])
        got = score(spec, frags, tolerance=0.01, tolerance_type="da", predicted_intensities=pred)
        # arccos is ill-conditioned near cos = 1, so ~1e-16 of float error in the
        # cosine surfaces as ~1e-8 here. That is inherent to the metric, not slack.
        assert got["spectral_angle"] == pytest.approx(1.0, abs=1e-7)

    def test_is_scale_invariant(self) -> None:
        """Unlike hyperscore, the spectral angle is a cosine — scaling must not move it."""
        frags, pred, mz, order = self._setup()
        spec = Spectrum(mz=mz[order], intensity=(pred * 1000.0)[order])
        got = score(spec, frags, tolerance=0.01, tolerance_type="da", predicted_intensities=pred)
        assert got["spectral_angle"] == pytest.approx(1.0, abs=1e-7)

    def test_charge_states_remain_distinct(self) -> None:
        import peptacular as pt

        frags = [frag for frag in pt.fragment("PEPTIDE", ion_types=("b",), charges=(1, 2)) if frag.position == 2]
        assert [frag.charge_state for frag in frags] == [1, 2]
        spec = Spectrum(mz=np.array([frags[1].mz]), intensity=np.array([100.0]))

        matched_prediction = score(
            spec,
            frags,
            tolerance=0.001,
            tolerance_type="da",
            predicted_intensities=[0.0, 1.0],
        )
        mismatched_prediction = score(
            spec,
            frags,
            tolerance=0.001,
            tolerance_type="da",
            predicted_intensities=[1.0, 0.0],
        )

        assert matched_prediction["spectral_angle"] == pytest.approx(1.0)
        assert mismatched_prediction["spectral_angle"] == pytest.approx(0.0)

    def test_spectrum_method_forwards_predicted_intensities(self) -> None:
        frags, pred, mz, order = self._setup()
        spec = Spectrum(mz=mz[order], intensity=pred[order])

        direct = score(spec, frags, tolerance=0.01, tolerance_type="da", predicted_intensities=pred)
        via_method = spec.score(frags, tolerance=0.01, tolerance_type="da", predicted_intensities=pred)

        assert via_method == direct

    def test_a_mismatched_pattern_scores_lower(self) -> None:
        frags, pred, mz, order = self._setup()
        spec = Spectrum(mz=mz[order], intensity=pred[::-1][order])
        got = score(spec, frags, tolerance=0.01, tolerance_type="da", predicted_intensities=pred)
        assert 0.0 <= got["spectral_angle"] < 0.9

    def test_length_mismatch_raises(self) -> None:
        frags, pred, mz, order = self._setup()
        spec = Spectrum(mz=mz[order], intensity=pred[order])
        with pytest.raises(ValueError, match="predicted_intensities"):
            score(spec, frags, tolerance=0.01, tolerance_type="da", predicted_intensities=pred[:-1])

    def test_dict_fragments_are_rejected(self) -> None:
        """Predictions must be pairable with ions, which the dict form cannot do."""
        import peptacular as pt

        frags, pred, mz, order = self._setup()
        spec = Spectrum(mz=mz[order], intensity=pred[order])
        as_dict = pt.ProFormaAnnotation.parse("PEPTIDEK").fast_fragment(ion_types="by", charges=[1])
        with pytest.raises(TypeError, match="Sequence"):
            score(spec, as_dict, tolerance=0.01, tolerance_type="da", predicted_intensities=pred)

    def test_without_prediction_the_coverage_measure_is_still_returned(self) -> None:
        frags, pred, mz, order = self._setup()
        spec = Spectrum(mz=mz[order], intensity=pred[order])
        got = score(spec, frags, tolerance=0.01, tolerance_type="da")
        assert 0.0 <= got["spectral_angle"] <= 1.0
