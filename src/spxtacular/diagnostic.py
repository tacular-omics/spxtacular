"""Diagnostic ions: which named mzPAF reference and immonium ions a spectrum contains.

- :func:`find_diagnostic_ions` reads one spectrum into a :class:`DiagnosticIons`.
- :data:`DIAGNOSTIC_IONS` is the default list: the mzPAF reference molecules, then the
  immonium ions.

Ion names are mzPAF annotations, and their m/z comes from paftacular
(``paftacular.parse(name).mz()``); nothing here types in a mass:

- ``r[<name>]`` is a reference molecule (mzPAF 1.0.1 section 4.4.7, protonated and singly
  charged), taken from tacular's :data:`~tacular.REFMOL_LOOKUP`; ``r[TMT126]`` is the
  TMT 126 reporter ion.
- ``I<aa>`` is the immonium ion of a residue (section 4.4.5: the residue less CO),
  e.g. ``IY`` for tyrosine.

Choices (all documented on the function):

- Default tolerance 20 ppm, as for :func:`~spxtacular.extract_reporter_ions`.
- Several peaks in a window: the most intense one is taken (ties: the lowest m/z).
- An ion with no peak has intensity ``0.0`` and observed m/z ``NaN``, so "not found"
  stays visible.
- Windows may overlap, and names that share an m/z all report. Unlike a reporter plex, a
  list of diagnostic ions is not a set of channels that must be told apart.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import paftacular as paf
from numpy.typing import NDArray
from tacular import AMINO_ACID_INFOS, ORDERED_AMINO_ACIDS, REFMOL_LOOKUP, AminoAcid, tolerance_window
from tacular.types import ToleranceUnit

from .enums import check_tolerance_unit
from .errors import SpxtacularError
from .reporter import _pick

if TYPE_CHECKING:
    from .core import Spectrum

__all__ = [
    "DEFAULT_DIAGNOSTIC_TOLERANCE",
    "DEFAULT_DIAGNOSTIC_TOLERANCE_UNIT",
    "DIAGNOSTIC_IONS",
    "DiagnosticIons",
    "find_diagnostic_ions",
]

DEFAULT_DIAGNOSTIC_TOLERANCE = 20.0
"""Default diagnostic-ion m/z tolerance (with :data:`DEFAULT_DIAGNOSTIC_TOLERANCE_UNIT`, 20 ppm)."""
DEFAULT_DIAGNOSTIC_TOLERANCE_UNIT: ToleranceUnit = "ppm"

DIAGNOSTIC_IONS: tuple[str, ...] = tuple(f"r[{name}]" for name in REFMOL_LOOKUP.keys()) + tuple(  # noqa: SIM118
    f"I{aa}" for aa in ORDERED_AMINO_ACIDS if (AMINO_ACID_INFOS[AminoAcid(aa)].monoisotopic_mass or 0.0) > 0.0
)
"""The default ions of :func:`find_diagnostic_ions`: ``r[<name>]`` for every reference
molecule in tacular's ``REFMOL_LOOKUP`` (in its order), then ``I<aa>`` for every residue
with a positive monoisotopic mass (so not ``B``, ``X`` or ``Z``). Glycan oxonium ions are
not in ``REFMOL_LOOKUP``; pass their Unimod names, e.g. ``"r[HexNAc]"``, yourself."""


@cache
def _ion_mz(name: str) -> float:
    """m/z of one mzPAF ion name; raises :class:`SpxtacularError` naming it if it does not resolve."""
    try:
        return float(paf.parse(name).mz())
    except Exception as exc:  # paftacular raises its own error types; report the name
        raise SpxtacularError(f"Diagnostic ion {name!r} could not be resolved to an m/z: {exc}") from exc


@dataclass(frozen=True, slots=True)
class DiagnosticIons:
    """Which named diagnostic ions one spectrum contains.

    Arrays are read-only and in :attr:`ions` order. An ion with no peak in its window has
    ``intensity`` 0.0, ``observed_mz`` NaN and ``found`` False.
    """

    ions: tuple[str, ...]
    """The mzPAF ion names, in the order they were asked for."""
    ion_mz: NDArray[np.float64]
    """Theoretical m/z of each ion."""
    intensity: NDArray[np.float64]
    """Intensity of the picked peak (0.0 when none)."""
    observed_mz: NDArray[np.float64]
    """m/z of the picked peak, NaN when none."""
    found: NDArray[np.bool_]
    """Whether a peak was found in each ion's window."""

    def __post_init__(self) -> None:
        """Freeze the arrays."""
        for name, dtype in (
            ("ion_mz", np.float64),
            ("intensity", np.float64),
            ("observed_mz", np.float64),
            ("found", np.bool_),
        ):
            array = np.array(getattr(self, name), dtype=dtype)
            array.flags.writeable = False
            object.__setattr__(self, name, array)

    def present(self) -> tuple[str, ...]:
        """Names of the ions that were found, in :attr:`ions` order."""
        return tuple(name for name, hit in zip(self.ions, self.found, strict=True) if hit)

    def __len__(self) -> int:
        """Number of ions asked for."""
        return len(self.ions)


def find_diagnostic_ions(
    spectrum: "Spectrum",
    ions: Sequence[str] | None = None,
    *,
    tolerance: float = DEFAULT_DIAGNOSTIC_TOLERANCE,
    tolerance_unit: ToleranceUnit = DEFAULT_DIAGNOSTIC_TOLERANCE_UNIT,
) -> DiagnosticIons:
    """Report which named reference and immonium ions a (centroided) spectrum contains.

    Parameters
    ----------
    spectrum:
        A centroided :class:`~spxtacular.Spectrum` in m/z (not decharged).
    ions:
        mzPAF ion names, e.g. ``"r[TMT126]"`` (reference molecule) or ``"IY"`` (tyrosine
        immonium ion). ``None`` uses :data:`DIAGNOSTIC_IONS`. Glycan oxonium ions are not
        in the default list; pass their Unimod names, e.g. ``"r[HexNAc]"``.
    tolerance, tolerance_unit:
        Half-width of each ion's window around its m/z, in ``"ppm"`` (default 20) or
        ``"da"``.

    Returns
    -------
    DiagnosticIons
        In each window the most intense peak is taken (ties: lowest m/z). Ions with no
        peak have intensity 0.0, observed m/z NaN and ``found`` False. Windows may
        overlap, and names that share an m/z (``r[TMT126-ETD]`` and ``r[TMT127C-ETD]``)
        all report the same peak; nothing raises for that.

    Raises
    ------
    SpxtacularError
        An empty ``ions``, a name that does not resolve to an m/z, a bad tolerance or
        unit, a decharged spectrum, or NaN/inf intensities.
    """
    unit = check_tolerance_unit(tolerance_unit)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise SpxtacularError(f"tolerance must be a finite number >= 0, got {tolerance!r}.")
    names = DIAGNOSTIC_IONS if ions is None else tuple(ions)
    if not names:
        raise SpxtacularError("Diagnostic ions: ions is empty; pass at least one mzPAF ion name.")
    if any(not isinstance(name, str) for name in names):
        raise SpxtacularError("Diagnostic ions: every ion must be an mzPAF name string such as 'IY' or 'r[TMT126]'.")
    ion_mz = np.array([_ion_mz(name) for name in names], dtype=np.float64)
    if spectrum.is_decharged:
        raise SpxtacularError("Diagnostic ions need m/z values; this spectrum is decharged to neutral masses.")
    if not np.isfinite(spectrum.intensity).all():
        raise SpxtacularError("Diagnostic ions need finite intensities; this spectrum has NaN or inf intensities.")
    bounds = np.array([tolerance_window(float(mz), float(tolerance), tolerance_unit=unit) for mz in ion_mz])
    intensity, observed = _pick(spectrum, bounds[:, 0], bounds[:, 1])
    return DiagnosticIons(
        ions=names, ion_mz=ion_mz, intensity=intensity, observed_mz=observed, found=~np.isnan(observed)
    )
