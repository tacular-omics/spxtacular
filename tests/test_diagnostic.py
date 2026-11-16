"""Diagnostic-ion search: named mzPAF reference and immonium ions in a centroided spectrum."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from tacular import REFMOL_LOOKUP

from spxtacular import DIAGNOSTIC_IONS, DiagnosticIons, Spectrum, SpxtacularError, find_diagnostic_ions

NAMES = ["r[TMT126]", "IY", "r[HexNAc]", "r[HexNAc(2)]", "IH"]


def _spectrum() -> Spectrum:
    mz = np.array([126.1277, 136.0757, 204.0866, 500.0])
    inten = np.array([5000.0, 800.0, 12000.0, 100.0])
    return Spectrum(mz, inten)


def test_named_ions_found_and_absent() -> None:
    """Acceptance 1: three of five found; the absent two are 0.0 / NaN / False."""
    result = find_diagnostic_ions(_spectrum(), NAMES)
    assert result.present() == ("r[TMT126]", "IY", "r[HexNAc]")
    assert result.ions == tuple(NAMES)
    np.testing.assert_array_equal(result.found, [True, True, True, False, False])
    np.testing.assert_array_equal(result.intensity, [5000.0, 800.0, 12000.0, 0.0, 0.0])
    assert np.isnan(result.observed_mz[3:]).all()
    np.testing.assert_allclose(result.observed_mz[:3], [126.1277, 136.0757, 204.0866])
    assert result.ion_mz[0] == pytest.approx(126.1277, abs=1e-4)
    assert result.ion_mz[3] == pytest.approx(407.1660, abs=1e-4)


def test_defaults_find_only_reference_and_immonium() -> None:
    """Acceptance 2: no glycan oxonium name is in REFMOL_LOOKUP, so the defaults miss HexNAc."""
    assert "HexNAc" not in REFMOL_LOOKUP
    assert find_diagnostic_ions(_spectrum()).present() == ("r[TMT126]", "IY")


def test_default_list_and_read_only_arrays() -> None:
    """Acceptance 3: 71 reference + 23 immonium names; every array is read-only."""
    assert len(DIAGNOSTIC_IONS) == 94
    assert DIAGNOSTIC_IONS[:71] == tuple(f"r[{n}]" for n in REFMOL_LOOKUP.keys())  # noqa: SIM118
    assert DIAGNOSTIC_IONS[71:] == tuple(f"I{aa}" for aa in "ACDEFGHIJKLMNOPQRSTUVWY")
    result = find_diagnostic_ions(_spectrum())
    assert len(result) == 94
    for name in ("ion_mz", "intensity", "observed_mz", "found"):
        array = getattr(result, name)
        assert len(array) == 94
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[0] = array[0]
    assert dataclasses.is_dataclass(DiagnosticIons)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.ions = ()  # ty: ignore[invalid-assignment]


def test_errors() -> None:
    """Acceptance 4: bad input raises SpxtacularError."""
    s = _spectrum()
    with pytest.raises(SpxtacularError, match="empty"):
        find_diagnostic_ions(s, [])
    with pytest.raises(SpxtacularError, match="NotAMolecule"):
        find_diagnostic_ions(s, ["r[NotAMolecule]"])
    with pytest.raises(SpxtacularError, match="tolerance_unit"):
        find_diagnostic_ions(s, ["IY"], tolerance_unit="PPM")  # ty: ignore[invalid-argument-type]
    with pytest.raises(SpxtacularError, match="tolerance"):
        find_diagnostic_ions(s, ["IY"], tolerance=float("nan"))
    decharged = Spectrum(np.array([136.0757]), np.array([1.0]), charge=np.array([0]))
    with pytest.raises(SpxtacularError, match=r"^Diagnostic ions.*decharged"):
        find_diagnostic_ions(decharged, ["IY"])
    bad = Spectrum(np.array([136.0757, 200.0]), np.array([np.nan, 1.0]))
    with pytest.raises(SpxtacularError, match=r"^Diagnostic ions.*finite"):
        find_diagnostic_ions(bad, ["IY"])


def test_shared_mz_all_report_and_overlap_does_not_raise() -> None:
    """Acceptance 5: names sharing an m/z all report the one peak."""
    s = Spectrum(np.array([114.1277]), np.array([300.0]))
    result = find_diagnostic_ions(s)
    assert result.present() == ("r[TMT126-ETD]", "r[TMT127C-ETD]")
    # Windows that overlap, even a very wide one, do not raise.
    wide = find_diagnostic_ions(_spectrum(), ["IY", "IH"], tolerance=100.0, tolerance_unit="da")
    assert wide.present() == ("IY", "IH")


def test_tolerance_and_most_intense_peak() -> None:
    """A tighter tolerance drops an off-mass peak; the most intense peak in a window wins."""
    mz = float(find_diagnostic_ions(_spectrum(), ["IY"]).ion_mz[0])
    s = Spectrum(np.array([mz - 0.001, mz + 0.0005]), np.array([10.0, 90.0]))
    assert find_diagnostic_ions(s, ["IY"], tolerance=1.0, tolerance_unit="da").intensity[0] == 90.0
    assert find_diagnostic_ions(s, ["IY"], tolerance=1.0, tolerance_unit="ppm").present() == ()
