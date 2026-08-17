"""
Utility functions for mass spectrometry calculations.
"""

from __future__ import annotations

from typing import Any


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


def da_to_ppm(delta_mz: float, mz: float) -> float:
    """Convert a mass difference from Dalton to ppm.

    Parameters
    ----------
    delta_mz:
        Mass difference in Dalton.
    mz:
        Reference m/z value.

    Returns
    -------
    float
        Mass difference in ppm.

    Raises
    ------
    ValueError
        If ``mz`` is zero -- a relative error has no meaning without a reference.
    """
    if mz == 0.0:
        raise ValueError("mz must be non-zero to express a mass difference in ppm")
    return delta_mz / mz * 1e6


def ppm_to_da(delta_ppm: float, mz: float) -> float:
    """Convert a mass difference from ppm to Dalton.

    Parameters
    ----------
    delta_ppm:
        Mass difference in ppm.
    mz:
        Reference m/z value.

    Returns
    -------
    float
        Mass difference in Dalton.
    """
    return delta_ppm * mz / 1e6
