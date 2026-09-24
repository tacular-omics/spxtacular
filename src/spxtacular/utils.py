"""
Utility functions for mass spectrometry calculations.
"""

from __future__ import annotations

from typing import Any

from tacular.tolerance import da_to_ppm, ppm_to_da

__all__ = [
    "da_to_ppm",
    "format_precursor_charge",
    "ppm_to_da",
    "precursor_charge_magnitude",
    "signed_precursor_charge",
]


def precursor_charge_magnitude(charge: int | None) -> int | None:
    """Positive magnitude of a known precursor charge, regardless of polarity."""
    if charge is None or charge == 0:
        return None
    return abs(int(charge))


def signed_precursor_charge(charge: int | None, polarity: Any = None) -> int | None:
    """Signed precursor charge, using explicit sign first and scan polarity second."""
    magnitude = precursor_charge_magnitude(charge)
    if magnitude is None:
        return None
    assert charge is not None
    negative = int(charge) < 0 or str(polarity).lower() == "negative"
    return -magnitude if negative else magnitude


def format_precursor_charge(charge: int | None, polarity: Any = None) -> str | None:
    """Display a precursor charge as ``2+`` or ``2-``; unknown charges return ``None``."""
    signed = signed_precursor_charge(charge, polarity)
    if signed is None:
        return None
    return f"{abs(signed)}{'-' if signed < 0 else '+'}"
