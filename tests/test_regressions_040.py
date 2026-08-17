"""Regression guards for bugs fixed in the 0.4.0 cycle.

Each test pins behavior that previously regressed silently (a green suite did not
catch it). Grouped by the HISTORY.md fix they defend.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from spxtacular.core import Spectrum, SpectrumType
from spxtacular.matching import match_fragments
from spxtacular.scoring import score

_PROTON = 1.00727646688


def _frag(mz: float, ion_type: str = "b", position: int = 1, charge_state: int = 1) -> MagicMock:
    f = MagicMock()
    f.mz = mz
    f.ion_type = ion_type
    f.position = position
    f.charge_state = charge_state
    f.neutral_mass = mz * charge_state - charge_state * _PROTON
    return f


def _spec() -> Spectrum:
    return Spectrum(
        mz=np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64),
        intensity=np.array([10.0, 50.0, 30.0, 20.0, 5.0], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# tolerance_type default is DA (0.02 Da), uniformly. A peak 0.005 Da off matches
# under the DA default but not under PPM (0.02 ppm ≈ 2e-6 Da) — so this discriminates.
# ---------------------------------------------------------------------------


def test_match_fragments_default_tolerance_type_is_da() -> None:
    frag = _frag(100.005)  # 0.005 Da from the 100.0 peak
    assert len(match_fragments(_spec(), [frag])) == 1  # DA default → match
    assert match_fragments(_spec(), [frag], tolerance_type="ppm") == []  # PPM → no match


def test_score_default_tolerance_type_is_da() -> None:
    frags = [_frag(100.005)]
    default = score(_spec(), frags)
    da = score(_spec(), frags, tolerance_type="da")
    ppm = score(_spec(), frags, tolerance_type="ppm")
    assert default["total_matched_intensity"] == da["total_matched_intensity"]
    assert default["total_matched_intensity"] > 0.0
    assert ppm["total_matched_intensity"] == 0.0


# ---------------------------------------------------------------------------
# top_peaks(0) / filter(top_n=0) return ZERO peaks (the arr[-0:] negative-zero bug).
# ---------------------------------------------------------------------------


def test_top_peaks_zero_returns_empty() -> None:
    assert len(_spec().top_peaks(0)) == 0  # top_peaks returns a list[Peak]


def test_filter_top_n_zero_returns_empty() -> None:
    assert len(_spec().filter(top_n=0).mz) == 0


# ---------------------------------------------------------------------------
# inplace=False on an already-in-target-state spectrum returns a DISTINCT object;
# mutating it must not write through to the original.
# ---------------------------------------------------------------------------


def test_normalize_already_normalized_returns_distinct_copy() -> None:
    a = _spec().normalize()  # now normalized="max"
    with pytest.warns(UserWarning):
        b = a.normalize()  # early-return path
    assert b is not a
    b.intensity[0] = -999.0
    assert a.intensity[0] != -999.0


def test_deconvolute_already_deconvoluted_returns_distinct_copy() -> None:
    a = _spec().deconvolute(charge_range=(1, 2))
    with pytest.warns(UserWarning):
        b = a.deconvolute()  # early-return path
    assert b is not a
    b.mz[0] = -1.0
    assert a.mz[0] != -1.0


# ---------------------------------------------------------------------------
# match_fragments must not raise ZeroDivisionError on a 0.0 target mass under ppm.
# ---------------------------------------------------------------------------


def test_match_fragments_zero_target_mass_ppm_no_crash() -> None:
    # Degenerate 0.0 target mass under ppm previously raised ZeroDivisionError.
    # The contract of the fix is crash-safety; just assert the call completes.
    match_fragments(_spec(), [_frag(0.0)], tolerance_type="ppm")


# ---------------------------------------------------------------------------
# decharge() on a non-deconvoluted spectrum raises (documented contract).
# ---------------------------------------------------------------------------


def test_decharge_on_non_deconvoluted_raises() -> None:
    with pytest.raises(ValueError, match="deconvolute"):
        _spec().decharge()


def test_decharge_after_deconvolute_ok() -> None:
    decon = _spec().deconvolute(charge_range=(1, 2))
    assert decon.spectrum_type == SpectrumType.DECONVOLUTED
    # This synthetic spectrum has no isotope cluster, so every charge is unknown.
    with pytest.warns(UserWarning, match="unchanged"):
        result = decon.decharge()
    assert result == decon


def test_decharge_without_any_positive_charge_warns_and_preserves_spectrum() -> None:
    decon = Spectrum(
        mz=np.array([100.0, 200.0]),
        intensity=np.array([10.0, 20.0]),
        charge=np.array([-1, -1]),
        spectrum_type=SpectrumType.DECONVOLUTED,
    )
    with pytest.warns(UserWarning, match="unchanged"):
        result = decon.decharge()
    assert result == decon
    assert result is not decon


# ---------------------------------------------------------------------------
# Scored deconvolution conserves total intensity even when clusters are rejected
# (rejected multi-peak clusters no longer double-count the seed's intensity).
# ---------------------------------------------------------------------------


def test_deconvolute_rejected_cluster_conserves_total_intensity() -> None:
    # Three peaks ~1.003 Da apart form a charge-1 cluster candidate, plus a lone peak.
    mz = np.array([500.0, 501.003, 502.006, 400.0], dtype=np.float64)
    intensity = np.array([100.0, 60.0, 30.0, 50.0], dtype=np.float64)
    spec = Spectrum(mz=mz.copy(), intensity=intensity.copy())
    # min_score above any achievable Bhattacharyya score → every cluster rejected.
    decon = spec.deconvolute(charge_range=(1, 3), tolerance=50, tolerance_type="ppm", min_score=0.9999)
    # All peaks emitted as singletons; total intensity must equal the input (240),
    # not an inflated value from double-counting the rejected seed's cluster sum.
    assert decon.charge is not None
    assert np.all(decon.charge == -1)
    assert decon.intensity.sum() == pytest.approx(intensity.sum())


# ---------------------------------------------------------------------------
# Robustness: degenerate inputs don't crash or silently produce NaN.
# ---------------------------------------------------------------------------


def _empty() -> Spectrum:
    return Spectrum(mz=np.empty(0, dtype=np.float64), intensity=np.empty(0, dtype=np.float64))


def test_deconvolute_empty_returns_empty_deconvoluted() -> None:
    decon = _empty().deconvolute()
    assert decon.spectrum_type == SpectrumType.DECONVOLUTED
    assert len(decon.mz) == 0


@pytest.mark.parametrize("bad", [(0, 3), (3, 1), (-1, 2)])
def test_deconvolute_invalid_charge_range_raises(bad: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="charge_range"):
        _spec().deconvolute(charge_range=bad)


def test_normalize_all_zero_warns_and_returns_unchanged() -> None:
    spec = Spectrum(mz=np.array([100.0, 200.0]), intensity=np.array([0.0, 0.0]))
    with pytest.warns(UserWarning, match="all-zero"):
        out = spec.normalize()
    assert not np.isnan(out.intensity).any()
    assert np.all(out.intensity == 0.0)


def test_normalize_empty_is_noop() -> None:
    out = _empty().normalize()  # must not raise
    assert len(out.mz) == 0
