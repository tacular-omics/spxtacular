"""
Utility functions for mass spectrometry calculations.
"""

from __future__ import annotations


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
    """
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
