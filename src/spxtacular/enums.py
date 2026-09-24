"""Enum types shared across spxtacular modules."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, cast, get_args

from tacular.types import Polarity, ToleranceUnit

from .errors import SpxtacularError

_TOLERANCE_UNITS: tuple[str, ...] = get_args(ToleranceUnit)
_POLARITIES: tuple[str, ...] = get_args(Polarity)


def check_tolerance_unit(value: object, name: str = "tolerance_unit") -> ToleranceUnit:
    """Return ``value`` if it is a :data:`tacular.types.ToleranceUnit` (``"da"`` or ``"ppm"``).

    Only the lowercase strings are accepted, as in tacular; anything else raises
    :class:`SpxtacularError` naming the parameter ``name``.
    """
    if isinstance(value, str) and value in _TOLERANCE_UNITS:
        return cast("ToleranceUnit", value)
    raise SpxtacularError(f"{name} must be 'da' or 'ppm', got {value!r}")


IMToleranceUnit = Literal["relative", "absolute"]
"""How an ion-mobility tolerance is read: a fraction of the peak's mobility or mobility units."""


def check_im_tolerance_unit(value: object, name: str = "im_tolerance_unit") -> IMToleranceUnit:
    """Return ``value`` lowercased if it is ``"relative"`` or ``"absolute"``, else raise."""
    folded = value.lower() if isinstance(value, str) else value
    if folded in ("relative", "absolute"):
        return cast("IMToleranceUnit", folded)
    raise SpxtacularError(f"{name} must be 'relative' or 'absolute', got {value!r}")


def check_polarity(value: object, name: str = "polarity") -> Polarity | None:
    """Return ``value`` if it is ``None`` or a :data:`tacular.types.Polarity`.

    Only the lowercase strings ``"positive"`` and ``"negative"`` are accepted;
    anything else raises :class:`SpxtacularError` naming the field ``name``.
    """
    if value is None:
        return None
    if isinstance(value, str) and value in _POLARITIES:
        return cast("Polarity", value)
    raise SpxtacularError(f"{name} must be 'positive' or 'negative', got {value!r}")


class _SpxEnum(StrEnum):
    """String enum that matches values case-insensitively and raises :class:`SpxtacularError`.

    ``PeakSelection("LARGEST")`` is ``PeakSelection.LARGEST``; an unknown value raises
    ``SpxtacularError`` (a ``ValueError``) listing the accepted values.
    """

    @classmethod
    def _aliases(cls) -> dict[str, str]:
        return {}

    @classmethod
    def _missing_(cls, value: object) -> Any:
        if isinstance(value, str):
            folded = value.strip().casefold()
            folded = cls._aliases().get(folded, folded)
            for member in cls:
                if member.value.casefold() == folded:
                    return member
        accepted = ", ".join(repr(member.value) for member in cls)
        raise SpxtacularError(f"{value!r} is not a valid {cls.__name__}; expected one of {accepted}")


class PeakSelection(_SpxEnum):
    """Rule for resolving multiple observed peaks near one target."""

    CLOSEST = "closest"
    LARGEST = "largest"
    ALL = "all"


class ActivationType(_SpxEnum):
    """Ion activation / dissociation method.

    Members mirror the acronyms used as keys in
    :data:`spxtacular.spectrl_bridge._ACTIVATION_ACCESSIONS`, which maps each to
    its PSI-MS "dissociation method" (MS:1000044) accession. Open vocabulary:
    :class:`~spxtacular.core.MsnSpectrum.activation_type` is typed
    ``ActivationType | str``. On construction, a member name in any case or a known
    PSI-MS accession (e.g. ``"MS:1002481"`` from ``DReader``) becomes the member;
    other non-blank vendor strings are kept as they are.
    """

    CID = "CID"  # collision-induced dissociation
    HCD = "HCD"  # beam-type collision-induced dissociation
    ETD = "ETD"  # electron transfer dissociation
    ECD = "ECD"  # electron capture dissociation
    ETHCD = "EThcD"  # electron-transfer/higher-energy collision dissociation
    ETCID = "ETciD"  # electron-transfer/collision-induced dissociation
    NETD = "NETD"  # negative electron transfer dissociation
    UVPD = "UVPD"  # ultraviolet photodissociation
    PD = "PD"  # photodissociation
    PQD = "PQD"  # pulsed q dissociation
    SID = "SID"  # surface-induced dissociation
    IRMPD = "IRMPD"  # infrared multiphoton dissociation
    BIRD = "BIRD"  # blackbody infrared radiative dissociation
    SORI = "SORI"  # sustained off-resonance irradiation
    PASEF = "PASEF"  # Bruker PASEF (fragmented via beam-type CID; see spectrl_bridge)

    @classmethod
    def from_accession(cls, accession: str) -> ActivationType | str:
        """Map a PSI-MS dissociation-method accession to its member.

        Readers see raw ``"MS:NNNNNNN"`` accessions (or vendor enums whose value
        is one). Normalising them here keeps ``spec.activation_type ==
        ActivationType.CID`` true for reader-produced spectra. Unrecognised
        accessions are returned unchanged — the field is an open vocabulary.
        """
        from .spectrl_bridge import _ACTIVATION_NAMES

        return _ACTIVATION_NAMES.get(str(accession), str(accession))


class IMType(_SpxEnum):
    """Ion-mobility measurement type.

    Canonical, lowercase members. Closed vocabulary: ``Precursor.im_type`` and
    ``MsnSpectrum.im_type`` coerce strings to a member (case-insensitive, with the
    aliases ``"1/k0"`` and ``"drift_time"``) and raise ``SpxtacularError`` for
    anything else.
    """

    OOK0 = "ook0"  # inverse reduced ion mobility (1/K0)
    IM = "im"  # generic ion mobility
    DRIFT_TIME_MS = "drift_time_ms"  # drift time (ms)
    CCS = "ccs"  # collision cross section

    @classmethod
    def _aliases(cls) -> dict[str, str]:
        return {"1/k0": "ook0", "drift_time": "drift_time_ms"}


class Analyzer(_SpxEnum):
    """Mass analyzer type.

    Members map to the PSI-MS "mass analyzer type" (MS:1000443) branch via
    :data:`spxtacular.spectrl_bridge._ANALYZER_ACCESSIONS`. Open vocabulary:
    :class:`~spxtacular.core.MsnSpectrum.analyzer` is typed ``Analyzer | str``.
    A member name in any case (``"TOF"``) or a PSI-MS accession (``"MS:1000484"``)
    becomes the member; other vendor shorthands (``"FTMS"``) are kept as they are.
    """

    ORBITRAP = "orbitrap"
    FT_ICR = "ft_icr"  # fourier transform ion cyclotron resonance
    TOF = "tof"  # time-of-flight
    QUADRUPOLE = "quadrupole"
    ION_TRAP = "ion_trap"
    LINEAR_ION_TRAP = "linear_ion_trap"
    QUADRUPOLE_ION_TRAP = "quadrupole_ion_trap"
    MAGNETIC_SECTOR = "magnetic_sector"
    ELECTROSTATIC_ENERGY_ANALYZER = "electrostatic_energy_analyzer"

    @classmethod
    def from_accession(cls, accession: str) -> Analyzer | str:
        """Map a PSI-MS mass-analyzer accession (``"MS:1000484"``) to its member.

        Unrecognised accessions are returned unchanged — the field is an open vocabulary.
        """
        from .spectrl_bridge import _ANALYZER_NAMES

        return _ANALYZER_NAMES.get(str(accession), str(accession))


PeakSelectionLike = PeakSelection | Literal["closest", "largest", "all"]
# Open vocabularies: known names and accessions become members, unknown vendor
# strings are kept as plain ``str``.
ActivationTypeLike = ActivationType | str
AnalyzerLike = Analyzer | str
# Closed: a string must name a member (case-insensitive, or an alias).
IMTypeLike = IMType | str

# Shared default (tolerance, tolerance_unit) for fragment-matching entry points
# (Spectrum.match_fragments/score/annotate/annot_plot_table/remove_precursor_peak/
# mass_error_plot/facet_plot and their matching.py/scoring.py/visualization.py/
# plot_table.py counterparts). Single source of truth so the defaults can't drift
# out of sync across entry points again.
DEFAULT_FRAGMENT_TOLERANCE = 0.02
DEFAULT_FRAGMENT_TOLERANCE_UNIT: ToleranceUnit = "da"
