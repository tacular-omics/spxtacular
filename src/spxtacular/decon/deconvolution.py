"""
Main deconvolution function that uses the graph operations and navigation
methods to identify isotopic envelopes among the provided peaks.
"""

from typing import Literal

import numpy as np
import peptacular as pt
from numpy.typing import NDArray

from .dclass import DeconvolutedPeak, SpectrumPeak
from .graph_ops import construct_graph
from .navigation import navigate_left, navigate_right

# Global isotope lookup instance for performance
_GLOBAL_ISOTOPE_LOOKUP: pt.IsotopeLookup | None = None


def get_global_isotope_lookup() -> pt.IsotopeLookup:
    """Get or create the global isotope lookup instance."""
    global _GLOBAL_ISOTOPE_LOOKUP
    if _GLOBAL_ISOTOPE_LOOKUP is None:
        _GLOBAL_ISOTOPE_LOOKUP = pt.IsotopeLookup(
            mass_step=50,
            max_isotopes=25,
            min_abundance_threshold=0.005,
            use_neutron_count=True,
            is_abundance_sum=True,
        )
    return _GLOBAL_ISOTOPE_LOOKUP


def set_global_isotope_lookup(lookup: pt.IsotopeLookup | None) -> None:
    """
    Set a custom global isotope lookup instance.
    Pass None to reset to default (will be recreated on next use).
    """
    global _GLOBAL_ISOTOPE_LOOKUP
    _GLOBAL_ISOTOPE_LOOKUP = lookup


def deconvolute(
    mz: NDArray[np.float64],
    intensity: NDArray[np.float64],
    tolerance: float,
    tolerance_type: Literal["ppm", "da"],
    charge_range: tuple[int, int],
    max_left_decrease: float,
    max_right_decrease: float,
    isotope_mass: float,
    isotope_lookup: pt.IsotopeLookup | None,
) -> list[DeconvolutedPeak[SpectrumPeak]]:
    """
    Performs the main deconvolution procedure:
      1. Builds a graph of peaks connected by isotope spacing.
      2. Separates the graph by charge.
      3. Iterates over peaks in descending intensity,
         navigating left and right to form isotopic envelopes.

    Args:
        mz: Array of m/z values.
        intensity: Array of intensity values.
        tolerance: Numeric tolerance (in ppm or da).
        tolerance_type: Either 'ppm' or 'da'.
        charge_range: (min_charge, max_charge) to consider.
        max_left_decrease: Max fraction drop allowed to go left.
        max_right_decrease: Max fraction drop allowed to go right.
        isotope_mass: Mass difference between isotopes.
        isotope_lookup: Optional IsotopeLookup instance. If None, uses global instance.
    Returns:
        A list of DeconvolutedPeak objects containing identified isotopic envelopes.
    """
    # Use provided lookup or fall back to global
    if isotope_lookup is None:
        isotope_lookup = get_global_isotope_lookup()

    if len(mz) != len(intensity):
        raise ValueError("mz and intensity arrays must have same length")

    # Efficient convert to list of SpectrumPeak
    spectrum_peaks = [SpectrumPeak(mz=m, intensity=i) for m, i in zip(mz, intensity, strict=True)]

    graph = construct_graph(spectrum_peaks, tolerance, tolerance_type, charge_range, isotope_mass)

    # For quick peak lookup - we can use indices directly since peaks are stable
    dpeaks: list[DeconvolutedPeak[SpectrumPeak]] = []

    # Sort peaks by intensity descending
    sorted_peaks = np.argsort(intensity)[::-1]

    for peak_idx in sorted_peaks:
        peak_idx = int(peak_idx)  # numpy int to python int for graph indexing
        # Skip if peak is already visited
        if graph[peak_idx].seen:
            continue

        # Attempt each charge from highest to lowest,
        # build a set of candidate peaks using navigate_left/right
        results: dict[int, tuple[DeconvolutedPeak[SpectrumPeak], list[int]]] = {}
        for charge in range(charge_range[1], charge_range[0] - 1, -1):
            left_peaks = navigate_left(graph, peak_idx, charge, max_left_decrease)
            right_peaks = navigate_right(graph, peak_idx, charge, max_right_decrease)
            peaks_in_profile = sorted(set(left_peaks + right_peaks))

            decon_peak = DeconvolutedPeak(
                peaks=[graph[p].value for p in peaks_in_profile],
                charge=charge,
            )
            decon_peak.calculate_score(isotope_lookup)
            # Store both the decon_peak and the node indices
            results[charge] = (decon_peak, peaks_in_profile)

        # Pick the best (highest combined score) result among tested charges
        best_charge = max(
            results,
            key=lambda c: results[c][0].combined_score if results[c][0].combined_score is not None else -1.0,
        )

        best_result, best_node_indices = results[best_charge]

        # If only a single peak is found, we treat charge as unknown (None).
        if best_result.num_peaks == 1:
            best_result.charge = None

        # Mark all peaks in the best_result as seen using the node indices
        for node_idx in best_node_indices:
            graph[node_idx].seen = True

        dpeaks.append(best_result)

    # Set charge to None for single-peak results
    for dp in dpeaks:
        if dp.num_peaks <= 1:
            dp.charge = None

    return dpeaks
