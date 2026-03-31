"""
Tests for spxtacular.matching.match_fragments.

Fragments are mocked with MagicMock — match_fragments only reads .mz, .ion_type,
and .position, so the mock is sufficient for all matching tests.
"""

from unittest.mock import MagicMock

import numpy as np

from spxtacular.core import Spectrum
from spxtacular.matching import MatchedFragment, match_fragments

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frag(mz: float, ion_type: str = "b", position: int = 1, charge_state: int = 1) -> MagicMock:
    f = MagicMock()
    f.mz = mz
    f.ion_type = ion_type
    f.position = position
    f.charge_state = charge_state
    return f


def _spectrum() -> Spectrum:
    mz = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float64)
    intensity = np.array([10.0, 50.0, 20.0, 5.0], dtype=np.float64)
    return Spectrum(mz=mz, intensity=intensity)


# ---------------------------------------------------------------------------
# Basic Da tolerance matching
# ---------------------------------------------------------------------------


def test_closest_match_da_within_tolerance() -> None:
    spec = _spectrum()
    frag = _make_frag(100.005)
    result = match_fragments(spec, [frag], tolerance=0.02, tolerance_type="da", peak_selection="closest")
    assert len(result) == 1
    assert result[0].peak_index == 0  # peak at index 0 (100.0)


def test_no_match_outside_tolerance_da() -> None:
    spec = _spectrum()
    frag = _make_frag(100.5)
    result = match_fragments(spec, [frag], tolerance=0.02, tolerance_type="da", peak_selection="closest")
    assert result == []


def test_exact_mz_match_da() -> None:
    spec = _spectrum()
    frag = _make_frag(300.0)
    result = match_fragments(spec, [frag], tolerance=0.02, tolerance_type="da", peak_selection="closest")
    assert len(result) == 1
    assert result[0].peak_index == 2  # index of 300.0


# ---------------------------------------------------------------------------
# PPM tolerance matching
# ---------------------------------------------------------------------------


def test_ppm_tolerance_match() -> None:
    # 0.002 Da error at 200 Da = 10 ppm
    spec = _spectrum()
    frag = _make_frag(200.002)
    result = match_fragments(spec, [frag], tolerance=10, tolerance_type="ppm", peak_selection="closest")
    assert len(result) == 1
    assert result[0].peak_index == 1  # peak at 200.0


def test_ppm_no_match_when_error_exceeds_tolerance() -> None:
    # 0.003 Da at 200 Da = 15 ppm, exceeds 10 ppm tolerance
    spec = _spectrum()
    frag = _make_frag(200.003)
    result = match_fragments(spec, [frag], tolerance=10, tolerance_type="ppm", peak_selection="closest")
    assert result == []


# ---------------------------------------------------------------------------
# peak_selection modes
# ---------------------------------------------------------------------------


def test_closest_picks_nearest_peak() -> None:
    # Two peaks at 200.0 and 200.015; fragment at 200.01 — closest is 200.015 (delta=0.005)
    spec = Spectrum(
        mz=np.array([200.0, 200.015], dtype=np.float64),
        intensity=np.array([50.0, 5.0], dtype=np.float64),
    )
    frag = _make_frag(200.01)
    result = match_fragments(spec, [frag], tolerance=0.02, tolerance_type="da", peak_selection="closest")
    assert len(result) == 1
    assert result[0].peak_index == 1  # 200.015 is closer to 200.01


def test_largest_picks_highest_intensity_peak() -> None:
    # Two peaks at 200.0 (intensity=50) and 200.015 (intensity=5)
    spec = Spectrum(
        mz=np.array([200.0, 200.015], dtype=np.float64),
        intensity=np.array([50.0, 5.0], dtype=np.float64),
    )
    frag = _make_frag(200.01)
    result = match_fragments(spec, [frag], tolerance=0.02, tolerance_type="da", peak_selection="largest")
    assert len(result) == 1
    assert result[0].peak_index == 0  # 200.0 has higher intensity


def test_all_returns_both_peaks_in_tolerance() -> None:
    spec = Spectrum(
        mz=np.array([200.0, 200.015], dtype=np.float64),
        intensity=np.array([50.0, 5.0], dtype=np.float64),
    )
    frag = _make_frag(200.01)
    result = match_fragments(spec, [frag], tolerance=0.02, tolerance_type="da", peak_selection="all")
    assert len(result) == 2
    assert {r.peak_index for r in result} == {0, 1}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_no_fragments_returns_empty() -> None:
    spec = _spectrum()
    result = match_fragments(spec, [], tolerance=0.02, tolerance_type="da")
    assert result == []


def test_results_sorted_by_peak_index_ascending() -> None:
    spec = _spectrum()
    # Match fragments at 400.0 (index 3) and 100.0 (index 0) — submit in reverse order
    frags = [_make_frag(400.0), _make_frag(100.0)]
    result = match_fragments(spec, frags, tolerance=0.02, tolerance_type="da", peak_selection="closest")
    indices = [r.peak_index for r in result]
    assert indices == sorted(indices)


def test_multiple_fragments_match_same_peak_all_appear() -> None:
    spec = _spectrum()
    f1 = _make_frag(100.005, ion_type="b", position=1)
    f2 = _make_frag(100.008, ion_type="y", position=2)
    result = match_fragments(spec, [f1, f2], tolerance=0.02, tolerance_type="da", peak_selection="closest")
    assert len(result) == 2
    assert all(r.peak_index == 0 for r in result)


def test_matched_fragment_structure() -> None:
    spec = _spectrum()
    frag = _make_frag(200.0)
    result = match_fragments(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert len(result) == 1
    m = result[0]
    assert isinstance(m, MatchedFragment)
    assert m.peak_index == 1
    assert m.fragment is frag
    assert isinstance(m.peak_mz, float)
    assert isinstance(m.peak_intensity, float)
    assert isinstance(m.ppm_error, float)
    assert isinstance(m.da_error, float)
    assert isinstance(m.intensity_pct, float)


# ---------------------------------------------------------------------------
# Charge-state filtering (deconvoluted spectra)
# ---------------------------------------------------------------------------


def _decon_spectrum() -> Spectrum:
    """Two peaks: index 0 is z=1, index 1 is z=2."""
    return Spectrum(
        mz=np.array([200.0, 400.0], dtype=np.float64),
        intensity=np.array([100.0, 50.0], dtype=np.float64),
        charge=np.array([1, 2], dtype=np.int32),
    )


def test_charge_match_passes_when_charge_matches() -> None:
    spec = _decon_spectrum()
    frag = _make_frag(200.005, charge_state=1)
    result = match_fragments(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert len(result) == 1
    assert result[0].peak_index == 0


def test_charge_mismatch_excluded() -> None:
    spec = _decon_spectrum()
    frag = _make_frag(200.005, charge_state=2)  # peak at 200.0 is z=1 — mismatch
    result = match_fragments(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert result == []


def test_charge_filter_all_mode() -> None:
    """In 'all' mode, only peaks whose charge matches are returned."""
    spec = Spectrum(
        mz=np.array([200.0, 200.01], dtype=np.float64),
        intensity=np.array([100.0, 50.0], dtype=np.float64),
        charge=np.array([1, 2], dtype=np.int32),
    )
    frag = _make_frag(200.005, charge_state=1)
    result = match_fragments(spec, [frag], tolerance=0.02, tolerance_type="da", peak_selection="all")
    assert len(result) == 1
    assert result[0].peak_index == 0  # only the z=1 peak


def test_singleton_peaks_not_matched() -> None:
    """Peaks with charge=-1 (singletons) are never matched even if m/z is within tolerance."""
    spec = Spectrum(
        mz=np.array([200.0], dtype=np.float64),
        intensity=np.array([100.0], dtype=np.float64),
        charge=np.array([-1], dtype=np.int32),
    )
    frag = _make_frag(200.005, charge_state=1)
    result = match_fragments(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert result == []


def test_no_charge_filter_when_spectrum_has_no_charge_array() -> None:
    """Raw spectra (no charge array) match by m/z only — charge_state is ignored."""
    spec = _spectrum()  # no charge array
    frag = _make_frag(200.005, charge_state=99)  # any charge_state — should still match
    result = match_fragments(spec, [frag], tolerance=0.02, tolerance_type="da")
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Spectrum.match_fragments method
# ---------------------------------------------------------------------------


def test_method_matches_function_output() -> None:
    """spec.match_fragments(...) must return the same result as match_fragments(spec, ...)."""
    spec = _spectrum()
    frags = [_make_frag(100.005), _make_frag(200.005), _make_frag(999.0)]
    via_function = match_fragments(spec, frags, tolerance=0.02, tolerance_type="da")
    via_method = spec.match_fragments(frags, tolerance=0.02, tolerance_type="da")
    assert len(via_method) == len(via_function)
    for vm, vf in zip(via_method, via_function, strict=True):
        assert vm.peak_index == vf.peak_index
        assert vm.fragment is vf.fragment


# ---------------------------------------------------------------------------
# Dict fragments input (lines 150-163)
# ---------------------------------------------------------------------------


def test_dict_fragments_matches_correct_peak() -> None:
    from peptacular import IonType

    spec = _spectrum()
    frag_dict: dict = {(IonType.B, 1): [100.0, 200.0]}
    result = match_fragments(spec, frag_dict, tolerance=0.02, tolerance_type="da")
    peak_indices = {m.peak_index for m in result}
    assert 0 in peak_indices  # 100.0
    assert 1 in peak_indices  # 200.0


def test_dict_fragments_no_match_returns_empty() -> None:
    from peptacular import IonType

    spec = _spectrum()
    frag_dict: dict = {(IonType.B, 1): [999.0]}
    result = match_fragments(spec, frag_dict, tolerance=0.02, tolerance_type="da")
    assert result == []


def test_dict_fragments_largest_mode_picks_highest_intensity() -> None:
    from peptacular import IonType

    spec = Spectrum(
        mz=np.array([200.0, 200.015], dtype=np.float64),
        intensity=np.array([50.0, 5.0], dtype=np.float64),
    )
    frag_dict: dict = {(IonType.B, 1): [200.01]}
    result = match_fragments(spec, frag_dict, tolerance=0.02, tolerance_type="da", peak_selection="largest")
    assert len(result) == 1
    assert result[0].peak_index == 0


def test_dict_fragments_all_mode_returns_multiple_peaks() -> None:
    from peptacular import IonType

    spec = Spectrum(
        mz=np.array([200.0, 200.015], dtype=np.float64),
        intensity=np.array([50.0, 5.0], dtype=np.float64),
    )
    frag_dict: dict = {(IonType.B, 1): [200.01]}
    result = match_fragments(spec, frag_dict, tolerance=0.02, tolerance_type="da", peak_selection="all")
    assert len(result) == 2


# ---------------------------------------------------------------------------
# "largest" and "all" modes with ppm tolerance (lines 119-133)
# ---------------------------------------------------------------------------


def test_largest_mode_ppm_tolerance_picks_highest_intensity() -> None:
    spec = Spectrum(
        mz=np.array([200.0, 200.001], dtype=np.float64),
        intensity=np.array([80.0, 10.0], dtype=np.float64),
    )
    frag = _make_frag(200.0005)
    result = match_fragments(spec, [frag], tolerance=10.0, tolerance_type="ppm", peak_selection="largest")
    assert len(result) == 1
    assert result[0].peak_index == 0


def test_all_mode_ppm_tolerance_returns_all_within_tolerance() -> None:
    spec = Spectrum(
        mz=np.array([200.0, 200.001], dtype=np.float64),
        intensity=np.array([80.0, 10.0], dtype=np.float64),
    )
    frag = _make_frag(200.0005)
    result = match_fragments(spec, [frag], tolerance=10.0, tolerance_type="ppm", peak_selection="all")
    assert len(result) == 2
