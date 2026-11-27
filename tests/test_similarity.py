"""
Tests for spectrum-to-spectrum similarity.

The properties that matter for a similarity metric are mathematical: identical
inputs score 1, disjoint inputs score 0, an overall rescale changes nothing, and
the result never leaves [0, 1] however the peaks are arranged. Those are what
these pin, rather than particular numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from spxtacular.core import MsnSpectrum, Precursor, Spectrum
from spxtacular.errors import SpxtacularError
from spxtacular.similarity import cosine, entropy_similarity, modified_cosine

MZ = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64)
INTENSITY = np.array([100.0, 80.0, 60.0, 40.0, 20.0], dtype=np.float64)

METRICS = [cosine, entropy_similarity]


def _spec(mz=MZ, intensity=INTENSITY) -> Spectrum:
    return Spectrum(mz=np.asarray(mz, dtype=np.float64), intensity=np.asarray(intensity, dtype=np.float64))


class TestSharedProperties:
    """Every metric must satisfy these; anything else is a bug in that metric."""

    @pytest.mark.parametrize("metric", METRICS)
    def test_identical_spectra_score_one(self, metric) -> None:
        assert metric(_spec(), _spec()) == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("metric", METRICS)
    def test_scale_invariant(self, metric) -> None:
        """A spectrum acquired at twice the signal is the same spectrum."""
        assert metric(_spec(), _spec(intensity=INTENSITY * 1000.0)) == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("metric", METRICS)
    def test_disjoint_spectra_score_zero(self, metric) -> None:
        assert metric(_spec(), _spec(mz=MZ + 50.0)) == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("metric", METRICS)
    def test_symmetric(self, metric) -> None:
        a, b = _spec(), _spec(intensity=INTENSITY[::-1])
        assert metric(a, b) == pytest.approx(metric(b, a), abs=1e-9)

    @pytest.mark.parametrize("metric", METRICS)
    def test_bounded(self, metric) -> None:
        rng = np.random.default_rng(0)
        for _ in range(25):
            a = _spec(mz=np.sort(rng.uniform(100, 600, 8)), intensity=rng.exponential(1e4, 8))
            b = _spec(mz=np.sort(rng.uniform(100, 600, 8)), intensity=rng.exponential(1e4, 8))
            assert 0.0 <= metric(a, b, tolerance=0.5) <= 1.0

    @pytest.mark.parametrize("metric", METRICS)
    def test_empty_spectra_score_zero(self, metric) -> None:
        empty = _spec(mz=np.zeros(0), intensity=np.zeros(0))
        assert metric(empty, _spec()) == 0.0
        assert metric(_spec(), empty) == 0.0
        assert metric(empty, empty) == 0.0

    @pytest.mark.parametrize("metric", METRICS)
    def test_all_zero_intensity_scores_zero(self, metric) -> None:
        assert metric(_spec(intensity=np.zeros(5)), _spec()) == 0.0

    @pytest.mark.parametrize("metric", METRICS)
    def test_unsorted_input_gives_the_same_answer(self, metric) -> None:
        """timsTOF frames are mobility-ordered, so unsorted m/z is normal."""
        perm = np.random.default_rng(1).permutation(MZ.size)
        shuffled = _spec(mz=MZ[perm], intensity=INTENSITY[perm])
        assert metric(_spec(), shuffled) == pytest.approx(metric(_spec(), _spec()), abs=1e-9)


class TestCosine:
    def test_partial_overlap_scores_between(self) -> None:
        half = _spec(mz=np.array([100.0, 200.0, 900.0, 950.0, 990.0]))
        score = cosine(_spec(), half)
        assert 0.0 < score < 1.0

    def test_a_peak_matches_at_most_once(self) -> None:
        """Without one-to-one alignment one intense peak matches several and the
        score runs past 1."""
        cluster = _spec(mz=np.array([300.0, 300.001, 300.002, 300.003, 300.004]))
        assert cosine(_spec(), cluster, tolerance=1.0) <= 1.0

    def test_tolerance_controls_matching(self) -> None:
        near = _spec(mz=MZ + 0.05)
        assert cosine(_spec(), near, tolerance=0.001) == pytest.approx(0.0, abs=1e-9)
        assert cosine(_spec(), near, tolerance=0.1) > 0.9

    def test_ppm_tolerance_scales_with_mz(self) -> None:
        shifted = _spec(mz=MZ * (1 + 10 / 1e6))  # 10 ppm everywhere
        assert cosine(_spec(), shifted, tolerance=20, tolerance_unit="ppm") > 0.99
        assert cosine(_spec(), shifted, tolerance=1, tolerance_unit="ppm") == pytest.approx(0.0, abs=1e-9)

    def test_sqrt_transform_reduces_the_pull_of_one_dominant_peak(self) -> None:
        spiky = _spec(intensity=np.array([1e6, 1.0, 1.0, 1.0, 1.0]))
        other = _spec(intensity=np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
        assert cosine(spiky, other, transform="sqrt") > cosine(spiky, other, transform="linear")

    def test_unknown_transform_raises(self) -> None:
        with pytest.raises(ValueError, match="transform must be"):
            cosine(_spec(), _spec(), transform="cbrt")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


class TestModifiedCosine:
    def test_recovers_a_precursor_mass_shift(self) -> None:
        """The whole point: a modification shifts every fragment containing it."""
        mod = 79.96633
        mz = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
        inten = np.array([100.0, 80.0, 60.0, 90.0, 70.0, 50.0])
        a = _spec(mz=mz, intensity=inten)
        shifted = mz.copy()
        shifted[3:] += mod  # only the fragments spanning the modified site move
        order = np.argsort(shifted)
        b = _spec(mz=shifted[order], intensity=inten[order])

        plain = cosine(a, b, tolerance=0.02)
        aware = modified_cosine(a, b, 500.0, 500.0 + mod, tolerance=0.02)
        assert plain < 0.7, "a plain cosine should miss the shifted half"
        assert aware > 0.99, "the modified cosine should recover it"

    def test_equals_cosine_when_precursors_match(self) -> None:
        a, b = _spec(), _spec(intensity=INTENSITY[::-1])
        assert modified_cosine(a, b, 500.0, 500.0) == pytest.approx(cosine(a, b), abs=1e-12)

    def test_stays_bounded_with_a_shift(self) -> None:
        rng = np.random.default_rng(2)
        for _ in range(20):
            a = _spec(mz=np.sort(rng.uniform(100, 600, 10)), intensity=rng.exponential(1e4, 10))
            b = _spec(mz=np.sort(rng.uniform(100, 600, 10)), intensity=rng.exponential(1e4, 10))
            assert 0.0 <= modified_cosine(a, b, 400.0, 480.0, tolerance=0.5) <= 1.0

    def test_a_shift_never_lowers_the_score(self) -> None:
        """The shifted offset only adds candidate pairs, so it cannot hurt."""
        mz = np.array([100.0, 200.0, 300.0, 400.0])
        a = _spec(mz=mz, intensity=np.array([100.0, 50.0, 25.0, 10.0]))
        b = _spec(mz=np.array([100.0, 250.0, 350.0, 450.0]), intensity=np.array([100.0, 50.0, 25.0, 10.0]))
        assert modified_cosine(a, b, 300.0, 350.0, tolerance=0.02) >= cosine(a, b, tolerance=0.02) - 1e-12


MOD = 79.96633
SHIFT_MZ = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
SHIFT_INTENSITY = np.array([100.0, 80.0, 60.0, 90.0, 70.0, 50.0])


def _msn(mz, intensity, precursor_mz: float | None) -> MsnSpectrum:
    """An MsnSpectrum with one precursor at ``precursor_mz``, or none when it is ``None``."""
    precursors = [] if precursor_mz is None else [Precursor(precursor_mz=precursor_mz, charge=2)]
    return MsnSpectrum(
        mz=np.asarray(mz, dtype=np.float64), intensity=np.asarray(intensity, dtype=np.float64), precursors=precursors
    )


def _shifted_pair(query_mz: float | None = 500.0, reference_mz: float | None = 500.0 + MOD):
    """A query and a reference whose upper half is displaced by ``MOD`` (#108)."""
    shifted = SHIFT_MZ.copy()
    shifted[3:] += MOD
    order = np.argsort(shifted)
    return (
        _msn(SHIFT_MZ, SHIFT_INTENSITY, query_mz),
        _msn(shifted[order], SHIFT_INTENSITY[order], reference_mz),
    )


class TestModifiedCosinePrecursorDefaults:
    """#108: the precursor m/z default to each spectrum's first precursor."""

    def test_two_argument_call_equals_the_explicit_one(self) -> None:
        a, b = _shifted_pair()
        explicit = modified_cosine(a, b, 500.0, 500.0 + MOD, tolerance=0.02)
        assert modified_cosine(a, b, tolerance=0.02) == explicit
        assert explicit > 0.99 > cosine(a, b, tolerance=0.02)

    def test_the_first_precursor_is_used(self) -> None:
        a, b = _shifted_pair()
        b.precursors.append(Precursor(precursor_mz=900.0, charge=3))
        assert modified_cosine(a, b, tolerance=0.02) == modified_cosine(a, b, 500.0, 500.0 + MOD, tolerance=0.02)

    def test_an_explicit_precursor_mz_overrides_the_spectrums_own(self) -> None:
        a, b = _shifted_pair()
        # Equal explicit precursors leave no shift to recover, whatever the spectra carry.
        assert modified_cosine(a, b, 500.0, 500.0, tolerance=0.02) == pytest.approx(cosine(a, b, tolerance=0.02))
        # Each side can be overridden on its own: here the reference keeps its own m/z.
        assert modified_cosine(a, b, query_precursor_mz=500.0 + MOD, tolerance=0.02) == pytest.approx(
            cosine(a, b, tolerance=0.02)
        )
        # ... and here the query does.
        assert modified_cosine(a, b, reference_precursor_mz=500.0, tolerance=0.02) == pytest.approx(
            cosine(a, b, tolerance=0.02)
        )

    def test_plain_spectra_still_take_explicit_values(self) -> None:
        a, b = _shifted_pair()
        plain_a, plain_b = _spec(mz=a.mz, intensity=a.intensity), _spec(mz=b.mz, intensity=b.intensity)
        assert modified_cosine(plain_a, plain_b, 500.0, 500.0 + MOD, tolerance=0.02) == modified_cosine(
            a, b, tolerance=0.02
        )

    @pytest.mark.parametrize(
        ("query_mz", "reference_mz", "side"),
        [(None, 500.0, "query"), (500.0, None, "reference"), (None, None, "query")],
    )
    def test_a_spectrum_without_a_precursor_raises_naming_its_side(self, query_mz, reference_mz, side) -> None:
        a, b = _shifted_pair(query_mz, reference_mz)
        with pytest.raises(SpxtacularError, match=f"for the {side} spectrum") as info:
            modified_cosine(a, b)
        assert f"{side}_precursor_mz" in str(info.value)

    def test_a_plain_spectrum_without_explicit_values_raises(self) -> None:
        with pytest.raises(SpxtacularError, match="for the query spectrum"):
            modified_cosine(_spec(), _msn(MZ, INTENSITY, 500.0))

    def test_an_explicit_value_covers_a_missing_precursor(self) -> None:
        a, b = _shifted_pair(None, None)
        assert modified_cosine(a, b, 500.0, 500.0 + MOD, tolerance=0.02) > 0.99

    def test_mirror_plot_similarity_uses_the_same_code_path(self, monkeypatch) -> None:
        from spxtacular import similarity, visualization

        a, b = _shifted_pair()
        calls = []
        real = similarity.modified_cosine

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return real(*args, **kwargs)

        monkeypatch.setattr(similarity, "modified_cosine", spy)
        result = visualization._similarity_value(a, b, "modified_cosine", 0.02, "da")
        assert result == ("modified cosine", real(a, b, tolerance=0.02))
        # No precursor m/z of its own: modified_cosine reads them from the spectra.
        assert calls == [((a, b), {"tolerance": 0.02, "tolerance_unit": "da"})]

        monkeypatch.undo()
        with pytest.raises(SpxtacularError, match="for the reference spectrum"):
            visualization._similarity_value(a, _msn(b.mz, b.intensity, None), "modified_cosine", 0.02, "da")


class TestEntropySimilarity:
    def test_entropy_and_cosine_differ_for_intensity_mismatch(self) -> None:
        """Different metrics need not assign the same score to an intensity mismatch."""
        a = _spec(intensity=np.array([100.0, 80.0, 60.0, 40.0, 20.0]))
        # Shares peak positions but a very different intensity pattern.
        b = _spec(intensity=np.array([20.0, 40.0, 60.0, 80.0, 100.0]))
        assert entropy_similarity(a, b) < cosine(a, b)

    def test_partial_overlap_scores_between(self) -> None:
        half = _spec(mz=np.array([100.0, 200.0, 900.0, 950.0, 990.0]))
        assert 0.0 < entropy_similarity(_spec(), half) < 1.0
