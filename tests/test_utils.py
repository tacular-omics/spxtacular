"""
Tests for spxtacular.utils -- the Da <-> ppm conversions.

The broad round-trip behaviour is covered in test_new_features.py; what is
pinned here is the degenerate reference m/z, where a relative error has no
meaning and the function has to say so rather than divide by zero.
"""

from __future__ import annotations

import pytest

from spxtacular.utils import (
    da_to_ppm,
    format_precursor_charge,
    ppm_to_da,
    precursor_charge_magnitude,
    signed_precursor_charge,
)


def test_da_to_ppm_zero_mz_raises_value_error() -> None:
    with pytest.raises(ValueError, match="mz must be non-zero"):
        da_to_ppm(0.001, 0.0)


def test_da_to_ppm_zero_delta_and_zero_mz_still_raises() -> None:
    """0/0 is no more meaningful than x/0."""
    with pytest.raises(ValueError, match="mz must be non-zero"):
        da_to_ppm(0.0, 0.0)


def test_ppm_to_da_accepts_zero_mz() -> None:
    """The inverse only multiplies, so a zero reference is well defined."""
    assert ppm_to_da(10.0, 0.0) == 0.0


@pytest.mark.parametrize(("charge", "expected"), [(2, 2), (-2, 2), (0, None), (None, None)])
def test_precursor_charge_magnitude(charge: int | None, expected: int | None) -> None:
    assert precursor_charge_magnitude(charge) == expected


def test_precursor_charge_sign_uses_explicit_sign_then_polarity() -> None:
    assert signed_precursor_charge(-2, "positive") == -2
    assert signed_precursor_charge(2, "negative") == -2
    assert signed_precursor_charge(2, "positive") == 2


def test_format_precursor_charge() -> None:
    assert format_precursor_charge(2, "positive") == "2+"
    assert format_precursor_charge(2, "negative") == "2-"
    assert format_precursor_charge(-2) == "2-"
    assert format_precursor_charge(None) is None
