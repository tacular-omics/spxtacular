"""Enum types shared across spxtacular modules."""

from enum import StrEnum
from typing import Literal


class ToleranceType(StrEnum):
    DA = "da"
    PPM = "ppm"


class PeakSelection(StrEnum):
    CLOSEST = "closest"
    LARGEST = "largest"
    ALL = "all"


class Polarity(StrEnum):
    """Scan polarity. Closed vocabulary (PSI-MS MS:1000130 / MS:1000129)."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


class ActivationType(StrEnum):
    """Ion activation / dissociation method.

    Members mirror the acronyms used as keys in
    :data:`spxtacular.spectrl_bridge._ACTIVATION_ACCESSIONS`, which maps each to
    its PSI-MS "dissociation method" (MS:1000044) accession. Open vocabulary:
    :class:`~spxtacular.core.MsnSpectrum.activation_type` is typed
    ``ActivationType | str``, so raw accessions (e.g. ``"MS:1002481"`` from
    ``DReader``) and unknown vendor strings still flow through untouched.
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


class IMType(StrEnum):
    """Ion-mobility measurement type.

    Canonical, lowercase members. Extra aliases (``"1/k0"``, ``"drift_time"``)
    remain accepted as raw strings by the accession lookup in
    :data:`spxtacular.spectrl_bridge._IM_TYPE_ACCESSIONS`. Open vocabulary:
    :class:`~spxtacular.core.MsnSpectrum.im_type` is typed ``IMType | str``.
    """

    OOK0 = "ook0"  # inverse reduced ion mobility (1/K0)
    IM = "im"  # generic ion mobility
    DRIFT_TIME_MS = "drift_time_ms"  # drift time (ms)
    CCS = "ccs"  # collision cross section


class Analyzer(StrEnum):
    """Mass analyzer type.

    Members map to the PSI-MS "mass analyzer type" (MS:1000443) branch via
    :data:`spxtacular.spectrl_bridge._ANALYZER_ACCESSIONS`. Open vocabulary:
    :class:`~spxtacular.core.MsnSpectrum.analyzer` is typed ``Analyzer | str``,
    so vendor shorthands (e.g. ``"TOF"``, ``"FTMS"``) still flow through.
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


ToleranceLike = ToleranceType | Literal["da", "ppm"]
PeakSelectionLike = PeakSelection | Literal["closest", "largest", "all"]
PolarityLike = Polarity | Literal["positive", "negative"]
# Open vocabularies — enum members give autocomplete/typo-safety, while raw
# accessions and unknown vendor strings keep flowing through as plain ``str``.
ActivationTypeLike = ActivationType | str
IMTypeLike = IMType | str
AnalyzerLike = Analyzer | str

# Shared default (tolerance, tolerance_type) for fragment-matching entry points
# (Spectrum.match_fragments/score/annotate/annot_plot_table/remove_precursor_peak/
# mass_error_plot/facet_plot and their matching.py/scoring.py/visualization.py/
# plot_table.py counterparts). Single source of truth so the defaults can't drift
# out of sync across entry points again.
DEFAULT_FRAGMENT_TOLERANCE = 0.02
DEFAULT_FRAGMENT_TOLERANCE_TYPE = ToleranceType.DA
