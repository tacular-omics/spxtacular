"""
Tests for spxtacular.noise.estimate_noise_level and private helpers.
"""

from typing import Any, cast

import numpy as np
import pytest

from spxtacular.noise import estimate_noise_level

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng_intensities(seed: int = 0, n: int = 500) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.exponential(scale=100.0, size=n).astype(np.float64)


# ---------------------------------------------------------------------------
# Numeric method passthrough
# ---------------------------------------------------------------------------


def test_numeric_int_returns_exact_value() -> None:
    arr = _rng_intensities()
    assert estimate_noise_level(arr, method=42) == 42.0


def test_numeric_float_returns_exact_value() -> None:
    arr = _rng_intensities()
    assert estimate_noise_level(arr, method=3.14) == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# MAD (default)
# ---------------------------------------------------------------------------


def test_mad_returns_positive_float() -> None:
    arr = _rng_intensities()
    result = estimate_noise_level(arr, method="mad")
    assert isinstance(result, float)
    assert result > 0.0


def test_mad_is_default_method() -> None:
    arr = _rng_intensities()
    assert estimate_noise_level(arr) == estimate_noise_level(arr, method="mad")


# ---------------------------------------------------------------------------
# percentile
# ---------------------------------------------------------------------------


def test_percentile_returns_positive_float() -> None:
    arr = _rng_intensities()
    result = estimate_noise_level(arr, method="percentile")
    assert isinstance(result, float)
    assert result > 0.0


def test_percentile_equals_fifth_percentile() -> None:
    arr = _rng_intensities()
    expected = float(np.percentile(arr, 5))
    assert estimate_noise_level(arr, method="percentile") == pytest.approx(expected)


# ---------------------------------------------------------------------------
# histogram
# ---------------------------------------------------------------------------


def test_histogram_returns_float() -> None:
    arr = _rng_intensities()
    result = estimate_noise_level(arr, method="histogram")
    assert isinstance(result, float)


def test_histogram_result_positive_for_positive_input() -> None:
    arr = np.abs(_rng_intensities()) + 1.0
    result = estimate_noise_level(arr, method="histogram")
    assert result > 0.0


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------


def test_baseline_returns_float() -> None:
    arr = _rng_intensities()
    result = estimate_noise_level(arr, method="baseline")
    assert isinstance(result, float)


def test_baseline_result_non_negative() -> None:
    arr = np.abs(_rng_intensities()) + 1.0
    result = estimate_noise_level(arr, method="baseline")
    assert result >= 0.0


# ---------------------------------------------------------------------------
# iterative_median
# ---------------------------------------------------------------------------


def test_iterative_median_returns_float() -> None:
    arr = _rng_intensities()
    result = estimate_noise_level(arr, method="iterative_median")
    assert isinstance(result, float)


def test_iterative_median_positive_for_positive_input() -> None:
    arr = np.abs(_rng_intensities()) + 1.0
    result = estimate_noise_level(arr, method="iterative_median")
    assert result > 0.0


def test_iterative_median_stops_early_on_small_array() -> None:
    arr = np.ones(50, dtype=np.float64) * 5.0
    result = estimate_noise_level(arr, method="iterative_median")
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Unknown method raises ValueError
# ---------------------------------------------------------------------------


def test_unknown_method_raises_value_error() -> None:
    arr = _rng_intensities()
    bad: Any = cast(Any, "bogus")
    with pytest.raises(ValueError, match="Unknown method"):
        estimate_noise_level(arr, method=bad)
