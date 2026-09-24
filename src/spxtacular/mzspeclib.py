"""HUPO-PSI mzSpecLib 1.0 spectral libraries, text and JSON.

mzSpecLib (https://psidev.info/mzSpecLib) stores library spectra as controlled
vocabulary (CV) attributes plus a peak list whose peaks carry mzPAF annotations.
This module reads and writes both serialisations and maps each entry onto the
spxtacular data model:

* the spectrum becomes an :class:`~spxtacular.core.MsnSpectrum` (precursor m/z and
  charge, retention time, ion mobility, polarity, collision energy, dissociation
  method, MS level, scan number, TIC, injection time);
* each analyte becomes an :class:`Analyte` whose peptidoform is a peptacular
  ``ProFormaAnnotation`` carrying the charge;
* peak annotations are parsed with paftacular into ``PafAnnotation`` objects;
* every attribute without a field keeps its CV form in an ``attributes`` tuple.

Reading resolves attribute sets (``<AttributeSet ...>`` / ``*_attribute_sets``)
into each element. Writing never emits attribute sets. :func:`read_mzspeclib`
returns the whole library; :class:`MzSpecLibReader` streams it one entry at a time.
"""

from __future__ import annotations

import bisect
import gzip
import json
import re
import weakref
from collections.abc import Generator, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Literal, Self

import numpy as np
import paftacular as paf
import peptacular as pt
from paftacular import PafAnnotation
from tacular.types import Polarity

from ._peak_annotations import align_peak_annotations, item_text
from .core import MsnSpectrum, Precursor, SpectrumType
from .enums import ActivationType, IMType
from .errors import SpxtacularError
from .peaklist import _check_writable, _fmt, _open_text_write
from .spectrl_bridge import _ACTIVATION_ACCESSIONS_LOWER, _ACTIVATION_NAMES

__all__ = [
    "Analyte",
    "CvParam",
    "Interpretation",
    "LibraryEntry",
    "MzSpecLibReader",
    "SpectralLibrary",
    "read_mzspeclib",
    "write_mzspeclib",
]

type CvValue = str | int | float | tuple[str | int | float, ...] | None
type PeakAttributeValue = str | int | float | None

# ---------------------------------------------------------------------------
# CV terms
# ---------------------------------------------------------------------------

_FORMAT_VERSION = ("MS:1003186", "library format version")
_SPECTRUM_KEY = ("MS:1003237", "library spectrum key")
_CLUSTER_KEY = ("MS:1003267", "spectrum cluster key")
_SPECTRUM_NAME = ("MS:1003061", "library spectrum name")
_SET_NAME = "MS:1003212"
_MONO_MZ = ("MS:1003208", "experimental precursor monoisotopic m/z")
_SELECTED_MZ = ("MS:1000744", "selected ion m/z")
_CHARGE = ("MS:1000041", "charge state")
_PRECURSOR_INTENSITY = ("MS:1003085", "previous MSn-1 scan precursor intensity")
_RT = ("MS:1000894", "retention time")
_MS_LEVEL = ("MS:1000511", "ms level")
_POLARITY = ("MS:1000465", "scan polarity")
_COLLISION_ENERGY = ("MS:1000045", "collision energy")
_DISSOCIATION = ("MS:1000044", "dissociation method")
_SCAN_NUMBER = ("MS:1003057", "scan number")
_NATIVE_ID = ("MS:1000767", "native spectrum identifier")
_TIC = ("MS:1000285", "total ion current")
_INJECTION_TIME = ("MS:1000927", "ion injection time")
_NUM_PEAKS = ("MS:1003059", "number of peaks")
_ANNOTATION_FORMAT = ("MS:1003103", "ion annotation format")
_MZPAF = ("MS:1003104", "mzPAF peptide ion annotation format")
_PROFORMA_ION = ("MS:1003270", "proforma peptidoform ion notation")
_PROFORMA_SEQUENCE = ("MS:1003169", "proforma peptidoform sequence")
_SCORE = ("MS:1002357", "PSM-level probability")
_MIXTURE_MEMBERS = ("MS:1003163", "analyte mixture members")
_UNIT = ("UO:0000000", "unit")

_POLARITY_TERMS: dict[Polarity, tuple[str, str]] = {
    "positive": ("MS:1000130", "positive scan"),
    "negative": ("MS:1000129", "negative scan"),
}
_POLARITY_FROM_ACCESSION = {acc: polarity for polarity, (acc, _) in _POLARITY_TERMS.items()}

_IM_TERMS: dict[IMType, tuple[str, str]] = {
    IMType.OOK0: ("MS:1002815", "inverse reduced ion mobility"),
    IMType.DRIFT_TIME_MS: ("MS:1002476", "ion mobility drift time"),
    IMType.CCS: ("MS:1002954", "collisional cross sectional area"),
}
_IM_FROM_ACCESSION = {acc: im_type for im_type, (acc, _) in _IM_TERMS.items()}

# Units a mapped term may carry, as {unit accession: factor to spxtacular's unit}.
_SECOND = ("UO:0000010", "second")
_ELECTRONVOLT = ("UO:0000266", "electronvolt")
_UNITS: dict[str, dict[str, float]] = {
    _RT[0]: {_SECOND[0]: 1.0, "UO:0000031": 60.0},
    _COLLISION_ENERGY[0]: {_ELECTRONVOLT[0]: 1.0},
    _INJECTION_TIME[0]: {"UO:0000028": 1.0},
    "MS:1002815": {"MS:1002814": 1.0},
    "MS:1002476": {"UO:0000028": 1.0},
    "MS:1002954": {"UO:0000324": 1.0},
}

_DISSOCIATION_NAMES = {
    "MS:1000133": "collision-induced dissociation",
    "MS:1000422": "beam-type collision-induced dissociation",
    "MS:1000598": "electron transfer dissociation",
    "MS:1000250": "electron capture dissociation",
    "MS:1002631": "electron-transfer/higher-energy collision dissociation",
    "MS:1003182": "electron-transfer/collision-induced dissociation",
    "MS:1003247": "negative electron transfer dissociation",
    "MS:1003246": "ultraviolet photodissociation",
    "MS:1000435": "photodissociation",
    "MS:1000599": "pulsed q dissociation",
    "MS:1000136": "surface-induced dissociation",
    "MS:1000262": "infrared multiphoton dissociation",
    "MS:1000242": "blackbody infrared radiative dissociation",
    "MS:1000282": "sustained off-resonance irradiation",
    "MS:1002481": "higher energy beam-type collision-induced dissociation",
}
# ETD plus a supplemental activation, the mzML way of writing EThcD / ETciD.
_SUPPLEMENTAL_ACTIVATION = {
    frozenset({"MS:1000598", "MS:1002678"}): ActivationType.ETHCD,
    frozenset({"MS:1000598", "MS:1002679"}): ActivationType.ETCID,
}

# Terms whose text values stay strings even when they look like numbers.
_STRING_TERMS = frozenset(
    {
        _FORMAT_VERSION[0],
        "MS:1003187",  # library identifier
        "MS:1003188",  # library name
        "MS:1003190",  # library version
        "MS:1003191",  # library URI
        _SPECTRUM_NAME[0],
        _NATIVE_ID[0],
        "MS:1003063",  # universal spectrum identifier
        "MS:1003203",  # constituent spectrum file
        "MS:1000512",  # filter string
        "MS:1000002",  # sample name
        "MS:1000888",  # stripped peptide sequence
        _PROFORMA_ION[0],
        _PROFORMA_SEQUENCE[0],
        "MS:1000885",  # protein accession
        "MS:1000886",  # protein name
        "MS:1000866",  # molecular formula
        "MS:1003275",  # other attribute name
        "MS:1003276",  # other attribute value
        "MS:1000529",  # instrument serial number
        "MS:1001088",  # protein description
        "MS:1003031",  # CPTAC accession number
        "MS:1003102",  # NIST msp comment
        "MS:1003151",  # SHA-256
        "MS:1003168",  # library spectrum comment
        "MS:1003189",  # library description
        "MS:1003197",  # license URI
        "MS:1003198",  # copyright notice
        "MS:1003199",  # change log
        "MS:1003200",  # software version
        "MS:1003204",  # constituent identification file
        "MS:1003205",  # constituent library file
        "MS:1003206",  # library creation log
        "MS:1003244",  # peptide accession number
        "MS:1003260",  # related spectrum USI
        "MS:1003261",  # related spectrum description
        "MS:1003264",  # similar spectrum USI
        "MS:1003269",  # spectrum cluster member USI
        "MS:1003272",  # peak annotation string
        "MS:1003299",  # contributing replicate spectrum USI
        "MS:1003403",  # InChI
        _SET_NAME,
    }
)
# Terms whose text values are comma-separated lists.
_LIST_TERMS = frozenset({_MIXTURE_MEMBERS[0], "MS:1003268"})

_CV_VALUE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*:[A-Za-z0-9_.]+)\|(.*)$")
_INT_RE = re.compile(r"^[-+]?\d+$")
_FLOAT_RE = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$|^[-+]?(?:inf|nan)$", re.IGNORECASE)
_ATTRIBUTE_RE = re.compile(r"^(?:\[([^\]]+)\])?([^|\s]+)\|(.*)$")
_SECTION_RE = re.compile(r"^<(\w+)(?:\s+(\w+))?(?:=(.*))?>$")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CvParam:
    """One mzSpecLib attribute: a CV term with an optional value.

    Attributes
    ----------
    accession
        Term accession, e.g. ``"MS:1003061"``.
    name
        Term name, e.g. ``"library spectrum name"``.
    value
        A string, number, tuple (for list-valued terms) or ``None``. For a
        CV-valued attribute this is the value term's name.
    value_accession
        Accession of the value term, for CV-valued attributes.
    group
        Attribute group number. Terms in one group qualify each other, such as a
        value and its unit. Groups are renumbered 1, 2, ... in order of first
        appearance within each attribute list.
    """

    accession: str
    name: str
    value: CvValue = None
    value_accession: str | None = None
    group: int | None = None

    def __post_init__(self) -> None:
        if not self.accession or "|" in self.accession or any(c.isspace() for c in self.accession):
            raise SpxtacularError(f"invalid CV accession {self.accession!r}")
        if isinstance(self.value, list):
            object.__setattr__(self, "value", tuple(self.value))
        if isinstance(self.value, bool):
            raise SpxtacularError(f"{self.accession}: boolean values are not allowed, use a string")


def _canonical_attributes(attributes: Iterable[CvParam], where: str) -> tuple[CvParam, ...]:
    """Renumber groups 1..n by order of first appearance."""
    renumber: dict[int, int] = {}
    out: list[CvParam] = []
    for attr in attributes:
        if not isinstance(attr, CvParam):
            raise SpxtacularError(f"{where}: attributes must be CvParam, got {type(attr).__name__}")
        if attr.group is not None:
            group = renumber.setdefault(attr.group, len(renumber) + 1)
            if group != attr.group:
                attr = CvParam(attr.accession, attr.name, attr.value, attr.value_accession, group)
        out.append(attr)
    return tuple(out)


def _as_peptidoform(value: pt.ProFormaAnnotation | str | None, where: str) -> pt.ProFormaAnnotation | None:
    if value is None or isinstance(value, pt.ProFormaAnnotation):
        return value
    if isinstance(value, str):
        try:
            parsed = pt.parse(value)
        except ValueError as exc:
            raise SpxtacularError(f"{where}: invalid ProForma {value!r}: {exc}") from exc
        if not isinstance(parsed, pt.ProFormaAnnotation):
            raise SpxtacularError(f"{where}: {value!r} is not a single peptidoform")
        return parsed
    raise SpxtacularError(f"{where}: peptidoform must be a ProFormaAnnotation or string, got {type(value).__name__}")


@dataclass(slots=True, kw_only=True)
class Analyte:
    """The molecule a library spectrum is assigned to.

    Attributes
    ----------
    id
        Analyte number within the entry (``<Analyte=N>``).
    peptidoform
        A peptacular ``ProFormaAnnotation`` (a ProForma string is parsed), or
        ``None`` for a non-peptide analyte.
    charge
        Analyte charge. When a peptidoform is given, the charge lives on it: a
        missing one is filled from here (on a copy), and a conflicting one raises.
    attributes
        Every other analyte attribute (protein, mass, formula, ...).
    """

    id: int = 1
    peptidoform: pt.ProFormaAnnotation | None = None
    charge: int | None = None
    attributes: tuple[CvParam, ...] = ()

    def __post_init__(self) -> None:
        where = f"Analyte {self.id}"
        self.id = _as_id(self.id, where)
        peptidoform = _as_peptidoform(self.peptidoform, where)
        if self.charge is not None:
            self.charge = _as_int(self.charge, f"{where} charge")
        if peptidoform is not None:
            own = peptidoform.charge
            if own is None and self.charge is not None:
                peptidoform = peptidoform.copy()
                peptidoform.charge = self.charge
            elif isinstance(own, int) and not isinstance(own, bool):
                if self.charge is not None and self.charge != own:
                    raise SpxtacularError(f"{where}: charge {self.charge} conflicts with peptidoform charge {own}")
                self.charge = own
        self.peptidoform = peptidoform
        self.attributes = _canonical_attributes(self.attributes, where)


@dataclass(slots=True, kw_only=True)
class Interpretation:
    """One explanation of a spectrum by one or more analytes.

    Attributes
    ----------
    id
        Interpretation number within the entry.
    members
        Analyte ids making up a chimeric interpretation (``MS:1003163``); empty
        when the entry has one analyte.
    score
        ``MS:1002357`` PSM-level probability, if given.
    attributes
        Every other interpretation attribute.
    member_attributes
        ``<InterpretationMember=N>`` attributes, keyed by analyte id.
    """

    id: int = 1
    members: tuple[int, ...] = ()
    score: float | None = None
    attributes: tuple[CvParam, ...] = ()
    member_attributes: dict[int, tuple[CvParam, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        where = f"Interpretation {self.id}"
        self.id = _as_id(self.id, where)
        self.members = tuple(_as_id(member, f"{where} member") for member in self.members)
        if self.score is not None:
            self.score = _as_float(self.score, f"{where} score")
        self.attributes = _canonical_attributes(self.attributes, where)
        self.member_attributes = {
            _as_id(member, f"{where} member"): _canonical_attributes(attrs, f"{where} member {member}")
            for member, attrs in self.member_attributes.items()
        }


@dataclass(slots=True, eq=False)
class LibraryEntry:
    """One library spectrum with its analytes, interpretations and peak annotations.

    Attributes
    ----------
    spectrum
        The spectrum and its acquisition metadata.
    key
        Library spectrum key (``<Spectrum=N>``). ``None`` is numbered by position
        when written.
    name
        Library spectrum name (``MS:1003061``).
    analytes
        Analytes the spectrum is assigned to.
    interpretations
        Interpretations of the spectrum.
    peak_annotations
        One tuple of ``PafAnnotation`` per peak, or ``None`` when no peak is
        annotated. Accepts the same forms as ``write_msp(annotations=...)``,
        including a list of :class:`~spxtacular.matching.MatchedFragment`.
    peak_attributes
        Values of the extra peak columns (defined by ``MS:1003254`` attributes),
        one tuple per peak, or ``None``.
    attributes
        Every other spectrum attribute.
    """

    spectrum: MsnSpectrum
    _: KW_ONLY
    key: int | None = None
    name: str | None = None
    analytes: tuple[Analyte, ...] = ()
    interpretations: tuple[Interpretation, ...] = ()
    peak_annotations: Any = None
    peak_attributes: Any = None
    attributes: tuple[CvParam, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.spectrum, MsnSpectrum):
            raise SpxtacularError(f"LibraryEntry.spectrum must be an MsnSpectrum, got {type(self.spectrum).__name__}")
        where = f"library entry {self.key}" if self.key is not None else "library entry"
        if self.key is not None:
            self.key = _as_id(self.key, where)
        self.analytes = tuple(self.analytes)
        self.interpretations = tuple(self.interpretations)
        for kind, items in (("analyte", self.analytes), ("interpretation", self.interpretations)):
            ids = [item.id for item in items]
            if len(set(ids)) != len(ids):
                raise SpxtacularError(f"{where}: duplicate {kind} ids {ids}")
        n_peaks = len(self.spectrum.mz)
        self.peak_annotations = _as_peak_annotations(self.peak_annotations, n_peaks, where)
        self.peak_attributes = _as_peak_attributes(self.peak_attributes, n_peaks, where)
        self.attributes = _canonical_attributes(self.attributes, where)

    @classmethod
    def from_spectrum(
        cls,
        spectrum: MsnSpectrum,
        peptidoform: pt.ProFormaAnnotation | str | None = None,
        *,
        charge: int | None = None,
        score: float | None = None,
        key: int | None = None,
        name: str | None = None,
        peak_annotations: Any = None,
        attributes: Iterable[CvParam] = (),
    ) -> LibraryEntry:
        """Build a single-analyte entry.

        ``charge`` defaults to the peptidoform's charge, then to the first
        precursor's charge. ``score`` becomes the ``MS:1002357`` PSM-level
        probability of interpretation 1.
        """
        analytes: tuple[Analyte, ...] = ()
        if peptidoform is not None or charge is not None:
            pf = _as_peptidoform(peptidoform, "LibraryEntry.from_spectrum")
            if charge is None and (pf is None or pf.charge is None) and spectrum.precursors:
                charge = spectrum.precursors[0].charge
            analytes = (Analyte(id=1, peptidoform=pf, charge=charge),)
        interpretations = (Interpretation(id=1, score=score),) if score is not None else ()
        return cls(
            spectrum,
            key=key,
            name=name,
            analytes=analytes,
            interpretations=interpretations,
            peak_annotations=peak_annotations,
            attributes=tuple(attributes),
        )

    @property
    def peptidoform(self) -> pt.ProFormaAnnotation | None:
        """Peptidoform of the first analyte, with its charge."""
        return self.analytes[0].peptidoform if self.analytes else None

    @property
    def charge(self) -> int | None:
        """Precursor charge, else the first analyte's charge."""
        if self.spectrum.precursors and self.spectrum.precursors[0].charge is not None:
            return self.spectrum.precursors[0].charge
        return self.analytes[0].charge if self.analytes else None

    @property
    def score(self) -> float | None:
        """Score of the first interpretation."""
        return self.interpretations[0].score if self.interpretations else None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LibraryEntry):
            return NotImplemented
        return (
            self.key == other.key
            and self.name == other.name
            and self.analytes == other.analytes
            and self.interpretations == other.interpretations
            and self.peak_annotations == other.peak_annotations
            and self.peak_attributes == other.peak_attributes
            and self.attributes == other.attributes
            and self.spectrum.to_dict() == other.spectrum.to_dict()
        )

    __hash__ = None  # type: ignore[assignment]


@dataclass(slots=True, eq=True)
class SpectralLibrary:
    """A whole mzSpecLib file.

    Attributes
    ----------
    entries
        Library spectra in file order.
    attributes
        Library-level attributes, without the format version.
    clusters
        ``<Cluster=N>`` attributes, keyed by cluster key, kept as read.
    """

    entries: list[LibraryEntry] = field(default_factory=list)
    attributes: tuple[CvParam, ...] = ()
    clusters: dict[int, tuple[CvParam, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.entries = list(self.entries)
        self.attributes = _canonical_attributes(self.attributes, "library")
        self.clusters = {
            _as_id(key, "cluster"): _canonical_attributes(attrs, f"cluster {key}")
            for key, attrs in self.clusters.items()
        }

    def __iter__(self) -> Iterator[LibraryEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> LibraryEntry:
        return self.entries[index]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _as_int(value: object, where: str) -> int:
    if isinstance(value, bool):
        raise SpxtacularError(f"{where}: expected an integer, got {value!r}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and _INT_RE.match(value.strip()):
        return int(value)
    raise SpxtacularError(f"{where}: expected an integer, got {value!r}")


def _as_id(value: object, where: str) -> int:
    number = _as_int(value, where)
    if number < 0:
        raise SpxtacularError(f"{where}: id must not be negative, got {number}")
    return number


def _as_float(value: object, where: str) -> float:
    if isinstance(value, bool):
        raise SpxtacularError(f"{where}: expected a number, got {value!r}")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    raise SpxtacularError(f"{where}: expected a number, got {value!r}")


def _parse_mzpaf(text: str, where: str) -> tuple[PafAnnotation, ...]:
    # A bare "?" is how mzSpecLib (and the JSON writer here) marks an unannotated
    # peak. "?" with more (``?^2``, ``?/0.5ppm``) still carries information and is kept.
    if text.strip() == "?":
        return ()
    try:
        return tuple(paf.parse_multi(text))
    except ValueError as exc:
        raise SpxtacularError(f"{where}: invalid mzPAF annotation {text!r}: {exc}") from exc


def _as_peak_annotations(value: Any, n_peaks: int, where: str) -> tuple[tuple[PafAnnotation, ...], ...] | None:
    if value is None:
        return None
    per_peak = align_peak_annotations(value, n_peaks, where=where)
    out: list[tuple[PafAnnotation, ...]] = []
    for index, items in enumerate(per_peak):
        peak: list[PafAnnotation] = []
        for item in items:
            if isinstance(item, PafAnnotation):
                peak.append(item)
            elif isinstance(item, str):
                peak.extend(_parse_mzpaf(item, f"{where} peak {index}"))
            else:
                peak.extend(_parse_mzpaf(item_text(item), f"{where} peak {index}"))
        out.append(tuple(peak))
    if not any(out):
        return None
    return tuple(out)


def _as_peak_attributes(value: Any, n_peaks: int, where: str) -> tuple[tuple[PeakAttributeValue, ...], ...] | None:
    """One tuple per peak, all padded with ``None`` to the widest; ``None`` when empty."""
    if value is None:
        return None
    rows = [tuple(row) for row in value]
    if len(rows) != n_peaks:
        raise SpxtacularError(f"{where}: {len(rows)} peak attribute rows for {n_peaks} peaks")
    for row in rows:
        for item in row:
            if item is not None and (isinstance(item, bool) or not isinstance(item, (str, int, float))):
                raise SpxtacularError(f"{where}: peak attribute values must be str, int, float or None, got {item!r}")
            if isinstance(item, str) and any(c in item for c in "\t\r\n"):
                raise SpxtacularError(f"{where}: peak attribute value {item!r} contains a tab or newline")
    width = max((len(row) for row in rows), default=0)
    # Trailing all-empty columns carry nothing.
    while width and all(len(row) < width or row[width - 1] is None for row in rows):
        width -= 1
    if width == 0:
        return None
    return tuple(row[:width] + (None,) * (width - len(row)) for row in rows)


# ---------------------------------------------------------------------------
# Raw (format-neutral) records
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _RawAttr:
    accession: str
    name: str
    value: CvValue
    value_accession: str | None
    group: str | None


@dataclass(slots=True)
class _RawElement:
    id: int
    attrs: list[_RawAttr] = field(default_factory=list)
    members: list[_RawElement] = field(default_factory=list)


@dataclass(slots=True)
class _RawSpectrum:
    key: int | None
    where: str
    attrs: list[_RawAttr] = field(default_factory=list)
    analytes: list[_RawElement] = field(default_factory=list)
    interpretations: list[_RawElement] = field(default_factory=list)
    mz: list[float] = field(default_factory=list)
    intensity: list[float] = field(default_factory=list)
    annotations: list[tuple[PafAnnotation, ...]] = field(default_factory=list)
    peak_attributes: list[tuple[PeakAttributeValue, ...]] = field(default_factory=list)


_SET_KINDS = ("spectrum", "analyte", "interpretation", "cluster")


def _resolve(attrs: list[_RawAttr], sets: Mapping[str, list[_RawAttr]], where: str) -> list[CvParam]:
    """Apply attribute sets: ``all`` first, then each claimed set, then local terms.

    A later layer replaces every earlier instance of a term it defines. Terms of
    a set claimed inside a group join that group when they are ungrouped.
    """
    layers: list[list[tuple[_RawAttr, tuple[int, str] | None]]] = []

    def layer_of(raw: list[_RawAttr], index: int, claim_group: tuple[int, str] | None) -> list:
        out = []
        for attr in raw:
            if attr.accession == _SET_NAME:
                continue
            group = (index, attr.group) if attr.group is not None else claim_group
            out.append((attr, group))
        return out

    local_index = 10**9
    if "all" in sets:
        layers.append(layer_of(sets["all"], 0, None))
    for attr in attrs:
        if attr.accession != _SET_NAME:
            continue
        name = str(attr.value)
        if name not in sets:
            raise SpxtacularError(f"{where}: unknown attribute set {name!r}")
        claim_group = (local_index, attr.group) if attr.group is not None else None
        layers.append(layer_of(sets[name], len(layers) + 1, claim_group))
    layers.append(layer_of(attrs, local_index, None))

    merged: list[tuple[_RawAttr, tuple[int, str] | None]] = []
    for layer in layers:
        defined = {attr.accession for attr, _ in layer}
        merged = [item for item in merged if item[0].accession not in defined] + layer

    numbers: dict[tuple[int, str], int] = {}
    out: list[CvParam] = []
    for attr, group in merged:
        number = None if group is None else numbers.setdefault(group, len(numbers) + 1)
        out.append(CvParam(attr.accession, attr.name, attr.value, attr.value_accession, number))
    return out


class _Attrs:
    """Resolved attributes of one element, from which mapped terms are consumed."""

    def __init__(self, attrs: list[CvParam], where: str) -> None:
        self.items: list[CvParam | None] = list(attrs)
        self.where = where

    def _indices(self, accession: str) -> list[int]:
        return [i for i, a in enumerate(self.items) if a is not None and a.accession == accession]

    def _group_members(self, group: int) -> list[int]:
        return [i for i, a in enumerate(self.items) if a is not None and a.group == group]

    def take_all(self, accession: str) -> list[CvParam]:
        taken = []
        for i in self._indices(accession):
            attr = self.items[i]
            if attr is not None and attr.group is None:
                taken.append(attr)
                self.items[i] = None
        return taken

    def peek(self, accession: str) -> CvParam | None:
        """The single ungrouped-or-unit-grouped instance, else ``None``."""
        found = self._indices(accession)
        if len(found) != 1:
            return None
        return self.items[found[0]]

    def take_value(self, accession: str) -> CvParam | None:
        """Consume the one instance of ``accession``, with its unit when it has one.

        Several instances, a group holding anything but a unit, or a unit not
        listed in ``_UNITS`` leave the term in place and return ``None``.
        """
        found = self._indices(accession)
        if len(found) != 1:
            return None
        index = found[0]
        attr = self.items[index]
        assert attr is not None
        if attr.group is None:
            self.items[index] = None
            return attr
        members = [i for i in self._group_members(attr.group) if i != index]
        if len(members) != 1:
            return None
        unit = self.items[members[0]]
        assert unit is not None
        factors = _UNITS.get(accession, {})
        if unit.accession != _UNIT[0] or unit.value_accession not in factors:
            return None
        factor = factors[unit.value_accession]
        self.items[index] = None
        self.items[members[0]] = None
        value = attr.value
        if factor != 1.0:
            value = _as_float(value, f"{self.where} {attr.name}") * factor
        return CvParam(attr.accession, attr.name, value, attr.value_accession)

    def remaining(self) -> tuple[CvParam, ...]:
        return _canonical_attributes((a for a in self.items if a is not None), self.where)


def _float_term(attrs: _Attrs, term: tuple[str, str]) -> float | None:
    attr = attrs.take_value(term[0])
    return None if attr is None else _as_float(attr.value, f"{attrs.where} {term[1]}")


def _int_term(attrs: _Attrs, term: tuple[str, str]) -> int | None:
    attr = attrs.take_value(term[0])
    return None if attr is None else _as_int(attr.value, f"{attrs.where} {term[1]}")


def _read_activation(attrs: _Attrs) -> ActivationType | str | None:
    attr = attrs.peek(_DISSOCIATION[0])
    if attr is not None:
        if attr.group is None and attr.value_accession in _ACTIVATION_NAMES:
            attrs.take_all(_DISSOCIATION[0])
            return _ACTIVATION_NAMES[attr.value_accession]
        return None
    ungrouped = [
        a
        for a in attrs.items
        if a is not None and a.accession == _DISSOCIATION[0] and a.group is None and a.value_accession
    ]
    combo = _SUPPLEMENTAL_ACTIVATION.get(frozenset(a.value_accession for a in ungrouped if a.value_accession))
    if combo is not None and len(ungrouped) == 2:
        attrs.take_all(_DISSOCIATION[0])
        return combo
    return None


def _build_entry(raw: _RawSpectrum, sets: Mapping[str, Mapping[str, list[_RawAttr]]]) -> LibraryEntry:
    where = raw.where
    attrs = _Attrs(_resolve(raw.attrs, sets["spectrum"], where), where)

    key_attr = attrs.take_value(_SPECTRUM_KEY[0])
    key = raw.key
    if key_attr is not None:
        attr_key = _as_id(key_attr.value, f"{where} {_SPECTRUM_KEY[1]}")
        if key is not None and attr_key != key:
            raise SpxtacularError(f"{where}: {_SPECTRUM_KEY[1]} {attr_key} does not match the section key {key}")
        key = attr_key
    if key is None:
        raise SpxtacularError(f"{where}: spectrum has no {_SPECTRUM_KEY[1]} ({_SPECTRUM_KEY[0]})")

    name_attr = attrs.take_value(_SPECTRUM_NAME[0])
    name = None if name_attr is None or name_attr.value is None else str(name_attr.value)
    native_attr = attrs.take_value(_NATIVE_ID[0])
    native_id = None if native_attr is None or native_attr.value is None else str(native_attr.value)

    n_declared = _int_term(attrs, _NUM_PEAKS)
    if n_declared is not None and n_declared != len(raw.mz):
        raise SpxtacularError(f"{where}: declares {n_declared} peaks but holds {len(raw.mz)}")

    # Annotation format: mzPAF is the default and the only one parsed.
    for attr in list(attrs.items):
        if attr is not None and attr.accession == _ANNOTATION_FORMAT[0] and attr.value_accession == _MZPAF[0]:
            attrs.items[attrs.items.index(attr)] = None
    if any(raw.annotations):
        other = [a for a in attrs.items if a is not None and a.accession == _ANNOTATION_FORMAT[0]]
        if other:
            raise SpxtacularError(f"{where}: unsupported peak annotation format {other[0].value!r}; only mzPAF is read")

    precursors = None
    mono = attrs.take_value(_MONO_MZ[0])
    selected = None if mono is not None else attrs.take_value(_SELECTED_MZ[0])
    if mono is not None or selected is not None:
        chosen = mono if mono is not None else selected
        assert chosen is not None
        im = im_type = None
        for accession, candidate_type in _IM_FROM_ACCESSION.items():
            if attrs.peek(accession) is None:
                continue
            value = _float_term(attrs, (accession, candidate_type.value))
            if value is not None:
                im, im_type = value, candidate_type
                break
        intensity = _float_term(attrs, _PRECURSOR_INTENSITY)
        precursors = [
            Precursor(
                precursor_mz=_as_float(chosen.value, f"{where} {chosen.name}"),
                intensity=0.0 if intensity is None else intensity,
                charge=_int_term(attrs, _CHARGE),
                im=im,
                im_type=im_type,
                is_monoisotopic=True if mono is not None else None,
            )
        ]

    polarity = None
    polarity_attr = attrs.peek(_POLARITY[0])
    if polarity_attr is not None and polarity_attr.group is None:
        polarity = _POLARITY_FROM_ACCESSION.get(polarity_attr.value_accession or "")
        if polarity is not None:
            attrs.take_all(_POLARITY[0])

    ms_level = _int_term(attrs, _MS_LEVEL)
    spectrum = MsnSpectrum(
        mz=np.asarray(raw.mz, dtype=np.float64),
        intensity=np.asarray(raw.intensity, dtype=np.float64),
        spectrum_type=SpectrumType.CENTROID,
        ms_level=2 if ms_level is None else ms_level,
        scan_number=_int_term(attrs, _SCAN_NUMBER),
        native_id=native_id,
        rt=_float_term(attrs, _RT),
        injection_time=_float_term(attrs, _INJECTION_TIME),
        total_ion_current=_float_term(attrs, _TIC),
        polarity=polarity,
        collision_energy=_float_term(attrs, _COLLISION_ENERGY),
        activation_type=_read_activation(attrs),
        precursors=precursors,
    )

    analytes = tuple(_build_analyte(element, sets["analyte"], where) for element in raw.analytes)
    interpretations = tuple(
        _build_interpretation(element, sets["interpretation"], where) for element in raw.interpretations
    )
    return LibraryEntry(
        spectrum,
        key=key,
        name=name,
        analytes=analytes,
        interpretations=interpretations,
        peak_annotations=raw.annotations if any(raw.annotations) else None,
        peak_attributes=raw.peak_attributes if any(raw.peak_attributes) else None,
        attributes=attrs.remaining(),
    )


def _build_analyte(element: _RawElement, sets: Mapping[str, list[_RawAttr]], where: str) -> Analyte:
    where = f"{where} analyte {element.id}"
    attrs = _Attrs(_resolve(element.attrs, sets, where), where)
    peptidoform = None
    ion = attrs.take_value(_PROFORMA_ION[0])
    if ion is not None:
        peptidoform = _as_peptidoform(str(ion.value), where)
    else:
        sequence = attrs.take_value(_PROFORMA_SEQUENCE[0])
        if sequence is not None:
            peptidoform = _as_peptidoform(str(sequence.value), where)
    charge = None
    if peptidoform is None or peptidoform.charge is None:
        charge = _int_term(attrs, _CHARGE)
    elif isinstance(peptidoform.charge, int):
        # A redundant analyte charge that agrees with the peptidoform is absorbed.
        charge_attr = attrs.peek(_CHARGE[0])
        if charge_attr is not None and charge_attr.group is None and charge_attr.value == peptidoform.charge:
            attrs.take_all(_CHARGE[0])
    return Analyte(id=element.id, peptidoform=peptidoform, charge=charge, attributes=attrs.remaining())


def _build_interpretation(element: _RawElement, sets: Mapping[str, list[_RawAttr]], where: str) -> Interpretation:
    where = f"{where} interpretation {element.id}"
    attrs = _Attrs(_resolve(element.attrs, sets, where), where)
    members: tuple[int, ...] = ()
    members_attr = attrs.take_value(_MIXTURE_MEMBERS[0])
    if members_attr is not None:
        value = members_attr.value
        values = value if isinstance(value, tuple) else (value,)
        members = tuple(_as_id(v, f"{where} {_MIXTURE_MEMBERS[1]}") for v in values)
    score = _float_term(attrs, _SCORE)
    member_attributes = {
        member.id: _canonical_attributes(
            (CvParam(a.accession, a.name, a.value, a.value_accession, _group_number(a.group)) for a in member.attrs),
            f"{where} member {member.id}",
        )
        for member in element.members
    }
    return Interpretation(
        id=element.id,
        members=members,
        score=score,
        attributes=attrs.remaining(),
        member_attributes=member_attributes,
    )


def _group_number(group: str | None) -> int | None:
    """Raw group ids are only compared; any text is mapped to a stable integer."""
    if group is None:
        return None
    return int(group) if _INT_RE.match(group) else hash(group) & 0x7FFFFFFF


def _plain_attributes(raw: list[_RawAttr], where: str) -> tuple[CvParam, ...]:
    return _canonical_attributes(
        (CvParam(a.accession, a.name, a.value, a.value_accession, _group_number(a.group)) for a in raw), where
    )


# ---------------------------------------------------------------------------
# Text format: reading
# ---------------------------------------------------------------------------


def _text_value(accession: str, text: str) -> tuple[CvValue, str | None]:
    """Parse the value side of an attribute line."""
    if text == "":
        return None, None
    match = _CV_VALUE_RE.match(text)
    if match is not None and accession not in _STRING_TERMS:
        return match.group(2), match.group(1)
    if accession in _STRING_TERMS:
        return text, None
    if accession in _LIST_TERMS:
        return tuple(_scalar(part.strip()) for part in text.split(",")), None
    return _scalar(text), None


def _scalar(text: str) -> str | int | float:
    if _INT_RE.match(text):
        return int(text)
    if _FLOAT_RE.match(text):
        return float(text)
    return text


def _parse_attribute_line(line: str, where: str) -> _RawAttr:
    match = _ATTRIBUTE_RE.match(line)
    if match is None:
        raise SpxtacularError(f"{where}: expected 'ACCESSION|name=value', got {line!r}")
    group, accession, rest = match.groups()
    if rest.startswith('"'):
        end = rest.find('"', 1)
        if end < 0 or rest[end + 1 : end + 2] != "=":
            raise SpxtacularError(f"{where}: unterminated quoted attribute name in {line!r}")
        name, value_text = rest[1:end], rest[end + 2 :]
    else:
        name, sep, value_text = rest.partition("=")
        if not sep:
            raise SpxtacularError(f"{where}: attribute line has no '=': {line!r}")
    value, value_accession = _text_value(accession, value_text)
    return _RawAttr(accession, name, value, value_accession, group)


def _parse_peak_line(line: str, raw: _RawSpectrum, where: str) -> None:
    parts = line.split("\t") if "\t" in line else line.split(None, 2)
    if len(parts) < 2:
        raise SpxtacularError(f"{where}: expected 'mz<TAB>intensity[<TAB>annotation]', got {line!r}")
    try:
        mz, intensity = float(parts[0]), float(parts[1])
    except ValueError:
        raise SpxtacularError(f"{where}: peak m/z and intensity must be numbers, got {line!r}") from None
    raw.mz.append(mz)
    raw.intensity.append(intensity)
    annotation = parts[2].strip() if len(parts) > 2 else ""
    raw.annotations.append(_parse_mzpaf(annotation, where) if annotation else ())
    raw.peak_attributes.append(tuple(None if p.strip() == "" else _scalar(p.strip()) for p in parts[3:]))


@dataclass(slots=True)
class _RawHeader:
    """Everything outside spectra: library attributes, attribute sets, clusters."""

    library_attrs: list[_RawAttr] = field(default_factory=list)
    sets: dict[str, dict[str, list[_RawAttr]]] = field(default_factory=lambda: {kind: {} for kind in _SET_KINDS})
    clusters: dict[int, list[_RawAttr]] = field(default_factory=dict)


def _iter_text(lines: Iterable[str], path: Path, header: _RawHeader) -> Iterator[_RawSpectrum | None]:
    """Parse text lines, filling ``header`` as it goes.

    Yields ``None`` once when the header ends (at the first ``<Spectrum>``, or at
    the end of a file without spectra), then each spectrum as soon as the next
    section starts, so only one spectrum is held at a time.
    """
    sets = header.sets
    target: list[_RawAttr] | None = header.library_attrs
    spectrum: _RawSpectrum | None = None
    interpretation: _RawElement | None = None
    in_peaks = False
    seen_header = False
    header_done = False
    seen_cluster = False

    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\r\n")
        if line_no == 1:
            line = line.lstrip("﻿")
        where = f"{path}:{line_no}"
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not seen_header:
            if not stripped.startswith("<mzSpecLib"):
                raise SpxtacularError(f"{where}: not an mzSpecLib text file (expected '<mzSpecLib>')")
            seen_header = True
            continue

        section = _SECTION_RE.match(stripped) if stripped.startswith("<") else None
        if section is not None:
            tag, kind, value = section.groups()
            in_peaks = False
            if tag in ("AttributeSet", "Spectrum", "Cluster") and spectrum is not None:
                yield spectrum
                spectrum = None
            if tag == "AttributeSet":
                if header_done or seen_cluster:
                    raise SpxtacularError(f"{where}: attribute sets must come before the first spectrum")
                if kind is None or kind.lower() not in sets or not value:
                    raise SpxtacularError(f"{where}: malformed attribute set header {stripped!r}")
                target = sets[kind.lower()].setdefault(value, [])
            elif tag == "Spectrum":
                key = _as_id(value, f"{where} spectrum key")
                if not header_done:
                    header_done = True
                    yield None
                spectrum = _RawSpectrum(key=key, where=where)
                target = spectrum.attrs
                interpretation = None
            elif tag == "Cluster":
                key = _as_id(value, f"{where} cluster key")
                seen_cluster = True
                target = header.clusters.setdefault(key, [])
            elif tag in ("Analyte", "Interpretation", "InterpretationMember", "Peaks"):
                if spectrum is None:
                    raise SpxtacularError(f"{where}: <{tag}> outside a <Spectrum> section")
                if tag == "Peaks":
                    if value is not None:
                        raise SpxtacularError(f"{where}: malformed section header {stripped!r}")
                    in_peaks = True
                    target = None
                elif tag == "Analyte":
                    element = _RawElement(id=_as_id(value, f"{where} analyte id"))
                    spectrum.analytes.append(element)
                    target = element.attrs
                elif tag == "Interpretation":
                    interpretation = _RawElement(id=_as_id(value, f"{where} interpretation id"))
                    spectrum.interpretations.append(interpretation)
                    target = interpretation.attrs
                else:
                    if interpretation is None:
                        raise SpxtacularError(f"{where}: <InterpretationMember> outside an <Interpretation>")
                    member = _RawElement(id=_as_id(value, f"{where} interpretation member id"))
                    interpretation.members.append(member)
                    target = member.attrs
            else:
                raise SpxtacularError(f"{where}: unknown section <{tag}>")
            continue

        if in_peaks:
            assert spectrum is not None
            _parse_peak_line(line, spectrum, where)
            continue
        if target is None:
            raise SpxtacularError(f"{where}: unexpected line {stripped!r}")
        target.append(_parse_attribute_line(stripped, where))

    if not seen_header:
        raise SpxtacularError(f"{path}: empty file, not an mzSpecLib library")
    if not header_done:
        yield None
    if spectrum is not None:
        yield spectrum


def _library_attributes(library_attrs: list[_RawAttr], path: Path) -> tuple[CvParam, ...]:
    """Check the format version and return the other library attributes."""
    versions = [a for a in library_attrs if a.accession == _FORMAT_VERSION[0]]
    if len(versions) != 1:
        raise SpxtacularError(f"{path}: expected one {_FORMAT_VERSION[1]} ({_FORMAT_VERSION[0]}), got {len(versions)}")
    version = str(versions[0].value)
    if version.split(".")[0] != "1":
        raise SpxtacularError(f"{path}: unsupported mzSpecLib format version {version!r} (1.x is supported)")
    return _plain_attributes([a for a in library_attrs if a.accession != _FORMAT_VERSION[0]], f"{path} library")


def _build_clusters(
    clusters: dict[int, list[_RawAttr]], sets: Mapping[str, Mapping[str, list[_RawAttr]]], path: Path
) -> dict[int, tuple[CvParam, ...]]:
    for key, attrs in clusters.items():
        # The text form may repeat the key as an attribute inside <Cluster=N>; the
        # JSON form must. Either way it is the section key, not an attribute.
        for attr in attrs:
            if attr.accession == _CLUSTER_KEY[0] and _as_id(attr.value, f"{path} cluster {key}") != key:
                raise SpxtacularError(f"{path}: {_CLUSTER_KEY[1]} {attr.value} does not match the section key {key}")
        attrs[:] = [a for a in attrs if a.accession != _CLUSTER_KEY[0]]
    return {
        key: tuple(
            _resolve(attrs, sets["cluster"], f"{path} cluster {key}")
            if any(a.accession == _SET_NAME for a in attrs) or "all" in sets["cluster"]
            else _plain_attributes(attrs, f"{path} cluster {key}")
        )
        for key, attrs in clusters.items()
    }


# ---------------------------------------------------------------------------
# JSON format: reading
# ---------------------------------------------------------------------------


def _json_attrs(items: object, where: str) -> list[_RawAttr]:
    if isinstance(items, Mapping) and "attributes" in items:
        items = items["attributes"]
    if not isinstance(items, list):
        raise SpxtacularError(f"{where}: attributes must be a list")
    out = []
    for item in items:
        if not isinstance(item, Mapping) or "accession" not in item or "name" not in item:
            raise SpxtacularError(f"{where}: each attribute needs 'accession' and 'name', got {item!r}")
        value = item.get("value")
        if isinstance(value, list):
            value = tuple(value)
        if isinstance(value, bool) or not (value is None or isinstance(value, (str, int, float, tuple))):
            raise SpxtacularError(f"{where}: unsupported value {value!r} for {item['accession']}")
        group = item.get("cv_param_group")
        accession = str(item["accession"])
        value_accession = None if item.get("value_accession") is None else str(item["value_accession"])
        if accession in _STRING_TERMS and isinstance(value, (int, float)):
            # Text-valued terms read back from the text form as strings; match that.
            value = repr(value) if isinstance(value, float) else str(value)
        if value_accession is not None and accession in _STRING_TERMS:
            # Some writers split a string such as "sp|P12345|NAME" at its first "|"
            # as if it were a CV value; string terms never take CV values.
            value = f"{value_accession}|{'' if value is None else value}"
            value_accession = None
        group_text = None if group is None else str(group)
        out.append(_RawAttr(accession, str(item["name"]), value, value_accession, group_text))
    return out


def _json_elements(items: object, where: str, kind: str) -> list[_RawElement]:
    if items is None:
        return []
    if not isinstance(items, Mapping):
        raise SpxtacularError(f"{where}: {kind}s must be an object keyed by id")
    out = []
    for key, payload in items.items():
        if not isinstance(payload, Mapping):
            raise SpxtacularError(f"{where}: {kind} {key!r} must be an object")
        element = _RawElement(id=_as_id(payload.get("id", key), f"{where} {kind} id"))
        element.attrs = _json_attrs(payload.get("attributes", []), f"{where} {kind} {key}")
        members = payload.get("members", payload.get("member_interpretations"))
        element.members = _json_elements(members, where, f"{kind} member")
        out.append(element)
    return out


def _json_spectrum(payload: object, index: int, path: Path) -> _RawSpectrum:
    where = f"{path} spectrum {index}"
    if not isinstance(payload, Mapping):
        raise SpxtacularError(f"{where}: must be an object")
    raw = _RawSpectrum(key=None, where=where)
    raw.attrs = _json_attrs(payload.get("attributes", []), where)
    raw.analytes = _json_elements(payload.get("analytes"), where, "analyte")
    raw.interpretations = _json_elements(payload.get("interpretations"), where, "interpretation")
    mzs, intensities = payload.get("mzs"), payload.get("intensities")
    if not isinstance(mzs, list) or not isinstance(intensities, list) or len(mzs) != len(intensities):
        raise SpxtacularError(f"{where}: 'mzs' and 'intensities' must be lists of equal length")
    raw.mz = [_as_float(v, f"{where} m/z") for v in mzs]
    raw.intensity = [_as_float(v, f"{where} intensity") for v in intensities]

    annotations = payload.get("peak_annotations")
    if annotations is None:
        raw.annotations = [() for _ in mzs]
    else:
        if not isinstance(annotations, list) or len(annotations) != len(mzs):
            raise SpxtacularError(f"{where}: 'peak_annotations' must have one entry per peak")
        for i, entry in enumerate(annotations):
            texts = [entry] if isinstance(entry, str) else entry
            if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                raise SpxtacularError(
                    f"{where}: peak {i} annotation must be an mzPAF string or a list of them "
                    "(JSON annotation objects are not supported)"
                )
            peak: list[PafAnnotation] = []
            for text in texts:
                if text.strip():
                    peak.extend(_parse_mzpaf(text.strip(), f"{where} peak {i}"))
            raw.annotations.append(tuple(peak))

    aggregations = payload.get("aggregations", payload.get("aggregation_metadata"))
    if aggregations is None:
        raw.peak_attributes = [() for _ in mzs]
    else:
        if not isinstance(aggregations, list) or len(aggregations) != len(mzs):
            raise SpxtacularError(f"{where}: peak aggregations must have one entry per peak")
        for i, row in enumerate(aggregations):
            if row in ("", None):
                row = []
            if not isinstance(row, list):
                raise SpxtacularError(f"{where}: peak {i} aggregations must be a list")
            raw.peak_attributes.append(tuple(None if v == "" else v for v in row))
    return raw


def _json_header(members: Mapping[str, Any], path: Path) -> _RawHeader:
    """Library attributes and attribute sets from the top-level members (no spectra)."""
    if "format_version" not in members:
        raise SpxtacularError(f"{path}: missing 'format_version'")
    library_attrs = _json_attrs(members.get("attributes", []), f"{path} library")
    if not any(a.accession == _FORMAT_VERSION[0] for a in library_attrs):
        library_attrs.insert(0, _RawAttr(*_FORMAT_VERSION, str(members["format_version"]), None, None))

    sets: dict[str, dict[str, list[_RawAttr]]] = {}
    for kind in _SET_KINDS:
        found = members.get(f"{kind}_attribute_sets", members.get(f"library_{kind}_attribute_sets", {})) or {}
        if not isinstance(found, Mapping):
            raise SpxtacularError(f"{path}: {kind}_attribute_sets must be an object")
        sets[kind] = {name: _json_attrs(attrs, f"{path} {kind} set {name}") for name, attrs in found.items()}
    return _RawHeader(library_attrs=library_attrs, sets=sets)


def _json_clusters(items: Any, path: Path) -> dict[int, list[_RawAttr]]:
    clusters: dict[int, list[_RawAttr]] = {}
    for i, item in enumerate(items or []):
        where = f"{path} cluster {i}"
        attrs = _json_attrs(item.get("attributes", []) if isinstance(item, Mapping) else item, where)
        keys = [a for a in attrs if a.accession == _CLUSTER_KEY[0]]
        if len(keys) != 1:
            raise SpxtacularError(f"{where}: needs one {_CLUSTER_KEY[1]} ({_CLUSTER_KEY[0]})")
        clusters[_as_id(keys[0].value, where)] = [a for a in attrs if a.accession != _CLUSTER_KEY[0]]
    return clusters


# Members that, once all seen, let the header be read without scanning past "spectra". Cluster
# sets are not needed by entries; they are collected by the iteration pass, after the spectra.
_JSON_HEADER_KEYS = frozenset(
    {"format_version", "attributes"} | {f"{kind}_attribute_sets" for kind in _SET_KINDS if kind != "cluster"}
)


def _json_header_key(key: str) -> str:
    """``library_<kind>_attribute_sets`` counts as ``<kind>_attribute_sets`` for the header check."""
    return key.removeprefix("library_") if key.endswith("_attribute_sets") else key


# Characters a JSON number can continue with, when a chunk boundary cuts it.
_JSON_NUMBER_TAIL = re.compile(r"[0-9.eE+-]*")
_JSON_WS = re.compile(r"[ \t\n\r]*")
_JSON_CHUNK = 1 << 16


class _BadJson(Exception):
    """The document is not valid JSON; the caller re-parses it whole for the exact error."""


class _JsonStream:
    """Decode one JSON value at a time from a text handle, with the standard-library decoder.

    Only the value being decoded is held in memory. A value cut by a chunk
    boundary fails to decode and is retried with more text; the buffer grows by
    doubling, so a large value still costs linear time.
    """

    def __init__(self, handle: IO[str]) -> None:
        self.handle = handle
        self.buffer = ""
        self.pos = 0
        self.eof = False
        self.decoder = json.JSONDecoder()

    def _more(self) -> bool:
        if self.eof:
            return False
        if self.pos:
            self.buffer = self.buffer[self.pos :]
            self.pos = 0
        chunk = self.handle.read(max(_JSON_CHUNK, len(self.buffer)))
        if not chunk:
            self.eof = True
            return False
        self.buffer += chunk
        return True

    def peek(self) -> str:
        """Next non-whitespace character, or ``""`` at the end of the file."""
        while True:
            match = _JSON_WS.match(self.buffer, self.pos)
            assert match is not None
            self.pos = match.end()
            if self.pos < len(self.buffer):
                return self.buffer[self.pos]
            if not self._more():
                return ""

    def expect(self, chars: str) -> str:
        char = self.peek()
        if not char or char not in chars:
            raise _BadJson
        self.pos += 1
        return char

    def value(self) -> Any:
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.pos)
            except json.JSONDecodeError:
                if self._more():
                    continue
                raise _BadJson from None
            # A value that ends at the buffer's end, or a number followed only by what could
            # continue it ("1." cut from "1.5"), may continue in the next chunk.
            if (
                end >= len(self.buffer)
                or (
                    isinstance(value, int | float)
                    and not isinstance(value, bool)
                    and _JSON_NUMBER_TAIL.fullmatch(self.buffer, end) is not None
                )
            ) and self._more():
                continue
            self.pos = end
            return value

    def array_items(self) -> Iterator[Any]:
        self.expect("[")
        if self.peek() == "]":
            self.pos += 1
            return
        while True:
            yield self.value()
            if self.expect(",]") == "]":
                return

    def members(self) -> Iterator[tuple[str, Any]]:
        """Top-level ``(key, value)`` pairs. A ``"spectra"`` array comes as an iterator of its items;
        items the consumer does not take are decoded and dropped."""
        self.expect("{")
        if self.peek() == "}":
            self.pos += 1
        else:
            while True:
                key = self.value()
                if not isinstance(key, str):
                    raise _BadJson
                self.expect(":")
                if key == "spectra" and self.peek() == "[":
                    items = self.array_items()
                    yield key, items
                    for _ in items:
                        pass
                else:
                    yield key, self.value()
                if self.expect(",}") == "}":
                    break
        if self.peek():
            raise _BadJson


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _open_library_text(path: Path) -> IO[str]:
    """Open a library as UTF-8 text (a BOM is dropped), decompressing gzip found by its magic bytes."""
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8-sig")
    return open(path, encoding="utf-8-sig")


def _invalid_json(path: Path) -> SpxtacularError:
    """The error ``json.loads`` gives for the whole file, so streamed and whole reads fail alike.

    This reads the whole file into memory; it runs only when the stream finds invalid JSON.
    """
    with _open_library_text(path) as fh:
        text = fh.read()
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        return SpxtacularError(f"{path}: invalid JSON: {exc}")
    return SpxtacularError(
        f"{path}: internal error: the streaming JSON reader rejected a file that json.loads accepts; please report this"
    )


class _SeenKeys:
    """Spectrum keys seen so far.

    Keys above every key seen so far extend or start a run (``O(1)``), so a library
    with sequential keys costs almost nothing. Any other key goes in a ``set``,
    about 70 bytes each.
    """

    __slots__ = ("ends", "others", "starts")

    def __init__(self) -> None:
        self.starts: list[int] = []
        self.ends: list[int] = []
        self.others: set[int] = set()

    def add(self, key: int) -> bool:
        """Record ``key``; ``False`` if it was already seen."""
        if not self.ends or key > self.ends[-1] + 1:
            self.starts.append(key)
            self.ends.append(key)
            return True
        if key == self.ends[-1] + 1:
            self.ends[-1] = key
            return True
        i = bisect.bisect_right(self.starts, key) - 1
        if (i >= 0 and key <= self.ends[i]) or key in self.others:
            return False
        self.others.add(key)
        return True


@dataclass(slots=True)
class _Header:
    format: Literal["text", "json"]
    attributes: tuple[CvParam, ...]
    raw: _RawHeader


class MzSpecLibReader:
    """Stream an mzSpecLib library (text or JSON, optionally gzipped) one entry at a time.

    The library attributes are read on :meth:`open` (or first use) without
    reading any spectrum. Iterating yields :class:`LibraryEntry` objects in file
    order while holding only one spectrum in memory, with the same results and
    errors as :func:`read_mzspeclib`.

    Parameters
    ----------
    path
        Library file (``.mzspeclib.txt``, ``.mzspeclib.json``, optionally ``.gz``).
        The format is detected from the content and gzip from its magic bytes.

    Examples
    --------
    >>> with MzSpecLibReader("library.mzspeclib.txt") as reader:  # doctest: +SKIP
    ...     print(reader.attributes)
    ...     for entry in reader:
    ...         print(entry.key, entry.peptidoform)

    Notes
    -----
    Every iteration opens its own handle, so iterations are independent and the
    reader can be iterated again. An iteration closes its handle when it ends,
    when the iterator is dropped (``break`` out of a ``for`` loop), or on
    :meth:`close`. A JSON file whose attribute sets follow the ``"spectra"``
    array (as in files written with alphabetically sorted keys) is scanned once
    first to read them. Memory does not grow with the number of spectra, apart
    from about 70 bytes per spectrum key that is out of order (kept to detect
    duplicates). A JSON file with more than one ``"spectra"`` member is rejected.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._header: _Header | None = None
        self._clusters: dict[int, tuple[CvParam, ...]] | None = None
        self._iterations: weakref.WeakSet[Generator[LibraryEntry]] = weakref.WeakSet()

    # -- lifecycle -------------------------------------------------------------

    def open(self) -> None:
        """Check the file exists and read its header.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        SpxtacularError
            If the header is not a valid mzSpecLib 1.x header.
        """
        self._load_header()

    def close(self) -> None:
        """Close the handle of every unfinished iteration."""
        for iteration in list(self._iterations):
            iteration.close()

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- header ----------------------------------------------------------------

    @property
    def format(self) -> Literal["text", "json"]:
        """``"text"`` or ``"json"``, detected from the content."""
        return self._load_header().format

    @property
    def attributes(self) -> tuple[CvParam, ...]:
        """Library-level attributes, without the format version."""
        return self._load_header().attributes

    @property
    def clusters(self) -> dict[int, tuple[CvParam, ...]]:
        """``<Cluster=N>`` attributes, keyed by cluster key.

        Clusters may follow the spectra, so they are known after a complete
        iteration; before that, reading this property runs one.
        """
        if self._clusters is None:
            for _ in self:
                pass
        assert self._clusters is not None
        return self._clusters

    def _load_header(self) -> _Header:
        if self._header is None:
            with self._decoding():
                self._header = self._read_header()
        return self._header

    @contextmanager
    def _decoding(self) -> Iterator[None]:
        try:
            yield
        except UnicodeDecodeError as exc:
            raise SpxtacularError(f"{self.path}: not UTF-8 text: {exc}") from exc

    def _read_header(self) -> _Header:
        with _open_library_text(self.path) as fh:
            chunk = fh.read(_JSON_CHUNK)
            while chunk and not chunk.lstrip():
                chunk = fh.read(_JSON_CHUNK)
            fh.seek(0)
            if chunk.lstrip().startswith("{"):
                members: dict[str, Any] = {}
                try:
                    for key, value in _JsonStream(fh).members():
                        if key != "spectra":
                            members[key] = value
                        elif {_json_header_key(k) for k in members} >= _JSON_HEADER_KEYS:
                            break
                except _BadJson:
                    raise _invalid_json(self.path) from None
                raw = _json_header(members, self.path)
                return _Header("json", _library_attributes(raw.library_attrs, self.path), raw)
            raw = _RawHeader()
            next(_iter_text(fh, self.path, raw))
            return _Header("text", _library_attributes(raw.library_attrs, self.path), raw)

    # -- entries ---------------------------------------------------------------

    def __iter__(self) -> Iterator[LibraryEntry]:
        """Library entries in file order, one at a time."""
        iteration = self._iter_entries()
        self._iterations.add(iteration)
        return iteration

    def _iter_entries(self) -> Generator[LibraryEntry]:
        header = self._load_header()
        keys = _SeenKeys()
        with self._decoding(), _open_library_text(self.path) as fh:
            if header.format == "json":
                raws = self._iter_json(fh, header.raw)
            else:
                raws = self._iter_text(fh)
            for raw, sets in raws:
                entry = _build_entry(raw, sets)
                assert entry.key is not None
                if not keys.add(entry.key):
                    raise SpxtacularError(f"{self.path}: duplicate library spectrum keys")
                yield entry

    def _iter_text(self, fh: IO[str]) -> Iterator[tuple[_RawSpectrum, Mapping[str, Mapping[str, list[_RawAttr]]]]]:
        raw_header = _RawHeader()
        for raw in _iter_text(fh, self.path, raw_header):
            if raw is not None:
                yield raw, raw_header.sets
        self._clusters = _build_clusters(raw_header.clusters, raw_header.sets, self.path)

    def _iter_json(
        self, fh: IO[str], raw_header: _RawHeader
    ) -> Iterator[tuple[_RawSpectrum, Mapping[str, Mapping[str, list[_RawAttr]]]]]:
        members: dict[str, Any] = {}
        seen_spectra = False
        try:
            for key, value in _JsonStream(fh).members():
                if key != "spectra":
                    members[key] = value
                    continue
                if seen_spectra:
                    raise SpxtacularError(f"{self.path}: more than one 'spectra' member")
                seen_spectra = True
                if not isinstance(value, Iterator):
                    raise SpxtacularError(f"{self.path}: 'spectra' must be a list")
                for index, item in enumerate(value):
                    yield _json_spectrum(item, index, self.path), raw_header.sets
        except _BadJson:
            raise _invalid_json(self.path) from None
        # Cluster sets may follow the spectra, so take them from this pass, not the header.
        sets = {**raw_header.sets, "cluster": _json_header(members, self.path).sets["cluster"]}
        clusters = _json_clusters(members.get("clusters", []), self.path)
        self._clusters = _build_clusters(clusters, sets, self.path)


def read_mzspeclib(path: str | Path) -> SpectralLibrary:
    """Read a whole mzSpecLib library in text or JSON form.

    The format is detected from the content (a JSON document starts with ``{``),
    and gzip by its magic bytes. To read one entry at a time without holding
    the library in memory, use :class:`MzSpecLibReader`.

    Parameters
    ----------
    path
        Library file (``.mzspeclib.txt``, ``.mzspeclib.json``, optionally ``.gz``).

    Returns
    -------
    SpectralLibrary
        Library attributes, entries in file order, and clusters.

    Raises
    ------
    SpxtacularError
        If the file is not a valid mzSpecLib 1.x library, a peak annotation is
        not valid mzPAF, or a ProForma string does not parse.
    """
    reader = MzSpecLibReader(path)
    entries = list(reader)
    return SpectralLibrary(entries=entries, attributes=reader.attributes, clusters=reader.clusters)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _emit_spectrum_attributes(entry: LibraryEntry, key: int) -> list[CvParam]:
    """Mapped fields as CV params, then the entry's own attributes."""
    spec = entry.spectrum
    where = f"library entry {key}"
    out: list[CvParam] = []
    group = 0

    def add(term: tuple[str, str], value: CvValue, value_accession: str | None = None) -> None:
        out.append(CvParam(term[0], term[1], value, value_accession))

    def add_with_unit(term: tuple[str, str], value: float, unit: tuple[str, str]) -> None:
        nonlocal group
        group += 1
        out.append(CvParam(term[0], term[1], value, None, group))
        out.append(CvParam(_UNIT[0], _UNIT[1], unit[1], unit[0], group))

    if entry.name is not None:
        add(_SPECTRUM_NAME, entry.name)
    if spec.ms_level is not None and spec.ms_level != 2:
        add(_MS_LEVEL, int(spec.ms_level))
    if spec.scan_number is not None:
        add(_SCAN_NUMBER, int(spec.scan_number))
    if spec.native_id is not None:
        add(_NATIVE_ID, str(spec.native_id))
    if spec.polarity is not None:
        accession, name = _POLARITY_TERMS[spec.polarity]
        add(_POLARITY, name, accession)
    if spec.precursors:
        if len(spec.precursors) > 1:
            raise SpxtacularError(f"{where}: mzSpecLib stores one precursor per spectrum, got {len(spec.precursors)}")
        prec = spec.precursors[0]
        add(_MONO_MZ if prec.is_monoisotopic else _SELECTED_MZ, float(prec.precursor_mz))
        if prec.charge is not None:
            add(_CHARGE, int(prec.charge))
        if prec.intensity != 0.0:
            add(_PRECURSOR_INTENSITY, float(prec.intensity))
        if prec.im is not None:
            im_type = prec.im_type if prec.im_type is not None else spec.im_type
            if im_type not in _IM_TERMS:
                raise SpxtacularError(
                    f"{where}: precursor ion mobility needs im_type ook0, drift_time_ms or ccs to be written, "
                    f"got {im_type!r}"
                )
            add(_IM_TERMS[IMType(im_type)], float(prec.im))
    if spec.rt is not None:
        add_with_unit(_RT, float(spec.rt), _SECOND)
    if spec.collision_energy is not None:
        add_with_unit(_COLLISION_ENERGY, float(spec.collision_energy), _ELECTRONVOLT)
    if spec.activation_type is not None:
        accession = _ACTIVATION_ACCESSIONS_LOWER.get(str(spec.activation_type).lower())
        if accession is None:
            raise SpxtacularError(
                f"{where}: activation type {spec.activation_type!r} has no PSI-MS dissociation method; "
                "put it in attributes instead"
            )
        add(_DISSOCIATION, _DISSOCIATION_NAMES[accession], accession)
    if spec.injection_time is not None:
        add(_INJECTION_TIME, float(spec.injection_time))
    if spec.total_ion_current is not None:
        add(_TIC, float(spec.total_ion_current))
    if entry.peak_annotations is not None:
        add(_ANNOTATION_FORMAT, _MZPAF[1], _MZPAF[0])
    add(_NUM_PEAKS, len(spec.mz))

    written = {attr.accession for attr in out if attr.accession != _UNIT[0]}
    reserved = written | {_SPECTRUM_KEY[0], _SET_NAME}
    for attr in entry.attributes:
        if attr.accession in reserved:
            raise SpxtacularError(
                f"{where}: attribute {attr.accession} ({attr.name}) duplicates a field that is already written"
            )
        if attr.accession == _ANNOTATION_FORMAT[0] and entry.peak_annotations is not None:
            raise SpxtacularError(f"{where}: {attr.accession} ({attr.name}) conflicts with the mzPAF peak annotations")
    return out + _offset_groups(entry.attributes, group)


def _offset_groups(attrs: Iterable[CvParam], offset: int) -> list[CvParam]:
    return [
        attr
        if attr.group is None or offset == 0
        else CvParam(attr.accession, attr.name, attr.value, attr.value_accession, attr.group + offset)
        for attr in attrs
    ]


def _analyte_attributes(analyte: Analyte, where: str) -> list[CvParam]:
    out: list[CvParam] = []
    pf = analyte.peptidoform
    if pf is not None:
        if pf.charge is not None:
            out.append(CvParam(*_PROFORMA_ION, pf.serialize()))
        else:
            out.append(CvParam(*_PROFORMA_SEQUENCE, pf.serialize()))
    if analyte.charge is not None and (pf is None or pf.charge is None):
        out.append(CvParam(*_CHARGE, int(analyte.charge)))
    written = {attr.accession for attr in out} | {_SET_NAME}
    if pf is not None:
        written |= {_PROFORMA_ION[0], _PROFORMA_SEQUENCE[0]}
    for attr in analyte.attributes:
        if attr.accession in written:
            raise SpxtacularError(f"{where} analyte {analyte.id}: attribute {attr.accession} duplicates a field")
    return out + list(analyte.attributes)


def _interpretation_attributes(interpretation: Interpretation, where: str) -> list[CvParam]:
    out: list[CvParam] = []
    if interpretation.members:
        out.append(CvParam(*_MIXTURE_MEMBERS, interpretation.members))
    if interpretation.score is not None:
        out.append(CvParam(*_SCORE, float(interpretation.score)))
    written = {attr.accession for attr in out} | {_SET_NAME}
    for attr in interpretation.attributes:
        if attr.accession in written:
            raise SpxtacularError(
                f"{where} interpretation {interpretation.id}: attribute {attr.accession} duplicates a field"
            )
    return out + list(interpretation.attributes)


def _check_entry(entry: LibraryEntry, key: int) -> None:
    if not isinstance(entry, LibraryEntry):
        raise SpxtacularError(f"expected LibraryEntry objects, got {type(entry).__name__}")
    _check_writable(entry.spectrum, key, "mzSpecLib")


def _keyed_entries(entries: Iterable[LibraryEntry]) -> list[tuple[int, LibraryEntry]]:
    keyed: list[tuple[int, LibraryEntry]] = []
    seen: set[int] = set()
    for position, entry in enumerate(entries, start=1):
        key = entry.key if isinstance(entry, LibraryEntry) and entry.key is not None else position
        _check_entry(entry, key)
        if key in seen:
            raise SpxtacularError(f"duplicate library spectrum key {key}")
        seen.add(key)
        keyed.append((key, entry))
    return keyed


# -- text ------------------------------------------------------------------


def _text_scalar(value: str | int | float, where: str) -> str:
    if isinstance(value, float):
        return repr(value)
    text = str(value)
    if any(c in text for c in "\r\n"):
        raise SpxtacularError(f"{where}: attribute value {text!r} contains a newline")
    return text


def _text_attribute(attr: CvParam, where: str) -> str:
    name = f'"{attr.name}"' if "=" in attr.name else attr.name
    if any(c in attr.name for c in '"\r\n'):
        raise SpxtacularError(f"{where}: attribute name {attr.name!r} contains a quote or newline")
    if attr.value_accession is not None and attr.accession in _STRING_TERMS:
        raise SpxtacularError(f"{where}: {attr.accession} ({attr.name}) holds text and cannot take a CV value")
    if attr.value is None:
        value = ""
    elif isinstance(attr.value, tuple):
        value = ",".join(_text_scalar(v, where) for v in attr.value)
    else:
        value = _text_scalar(attr.value, where)
    if attr.value_accession is not None:
        value = f"{attr.value_accession}|{value}"
    prefix = f"[{attr.group}]" if attr.group is not None else ""
    return f"{prefix}{attr.accession}|{name}={value}\n"


def _text_peak_value(value: PeakAttributeValue) -> str:
    if value is None:
        return ""
    return repr(value) if isinstance(value, float) else str(value)


def _write_text(fh: Any, library: SpectralLibrary, keyed: list[tuple[int, LibraryEntry]]) -> None:
    fh.write("<mzSpecLib>\n")
    fh.write(_text_attribute(CvParam(*_FORMAT_VERSION, "1.0"), "library"))
    for attr in library.attributes:
        if attr.accession == _FORMAT_VERSION[0]:
            raise SpxtacularError("library attributes must not include the format version; it is written for you")
        fh.write(_text_attribute(attr, "library"))
    for cluster_key, attrs in library.clusters.items():
        fh.write(f"<Cluster={cluster_key}>\n")
        for attr in attrs:
            fh.write(_text_attribute(attr, f"cluster {cluster_key}"))
    for key, entry in keyed:
        where = f"library entry {key}"
        fh.write(f"<Spectrum={key}>\n")
        for attr in _emit_spectrum_attributes(entry, key):
            fh.write(_text_attribute(attr, where))
        for analyte in entry.analytes:
            fh.write(f"<Analyte={analyte.id}>\n")
            for attr in _analyte_attributes(analyte, where):
                fh.write(_text_attribute(attr, where))
        for interpretation in entry.interpretations:
            fh.write(f"<Interpretation={interpretation.id}>\n")
            for attr in _interpretation_attributes(interpretation, where):
                fh.write(_text_attribute(attr, where))
            for member, attrs in interpretation.member_attributes.items():
                fh.write(f"<InterpretationMember={member}>\n")
                for attr in attrs:
                    fh.write(_text_attribute(attr, where))
        fh.write("<Peaks>\n")
        spec = entry.spectrum
        for i in range(len(spec.mz)):
            columns = [_fmt(spec.mz[i]), _fmt(spec.intensity[i])]
            annotations = entry.peak_annotations[i] if entry.peak_annotations is not None else ()
            extras = entry.peak_attributes[i] if entry.peak_attributes is not None else ()
            if annotations or extras:
                columns.append(",".join(a.serialize() for a in annotations))
            columns.extend(_text_peak_value(v) for v in extras)
            fh.write("\t".join(columns).rstrip("\t") + "\n")
        fh.write("\n")


# -- JSON ------------------------------------------------------------------


def _json_attribute(attr: CvParam) -> dict[str, Any]:
    out: dict[str, Any] = {"accession": attr.accession, "name": attr.name}
    out["value"] = list(attr.value) if isinstance(attr.value, tuple) else attr.value
    if attr.value_accession is not None:
        out["value_accession"] = attr.value_accession
    if attr.group is not None:
        out["cv_param_group"] = attr.group
    return out


def _json_document(library: SpectralLibrary, keyed: list[tuple[int, LibraryEntry]]) -> dict[str, Any]:
    for attr in library.attributes:
        if attr.accession == _FORMAT_VERSION[0]:
            raise SpxtacularError("library attributes must not include the format version; it is written for you")
    spectra = []
    for key, entry in keyed:
        where = f"library entry {key}"
        attributes = [_json_attribute(CvParam(*_SPECTRUM_KEY, key))]
        attributes += [_json_attribute(a) for a in _emit_spectrum_attributes(entry, key)]
        spec = entry.spectrum
        payload: dict[str, Any] = {
            "attributes": attributes,
            "analytes": {
                str(a.id): {"id": str(a.id), "attributes": [_json_attribute(x) for x in _analyte_attributes(a, where)]}
                for a in entry.analytes
            },
            "interpretations": {},
            "mzs": [float(v) for v in spec.mz],
            "intensities": [float(v) for v in spec.intensity],
            # One comma-joined mzPAF string per peak, "?" when unannotated: the form
            # the upstream examples and mzspeclib-py use.
            "peak_annotations": [
                ",".join(a.serialize() for a in peak) or "?" for peak in (entry.peak_annotations or [()] * len(spec.mz))
            ],
        }
        for interpretation in entry.interpretations:
            item: dict[str, Any] = {
                "id": str(interpretation.id),
                "attributes": [_json_attribute(x) for x in _interpretation_attributes(interpretation, where)],
            }
            if interpretation.member_attributes:
                item["members"] = {
                    str(member): {"id": str(member), "attributes": [_json_attribute(x) for x in attrs]}
                    for member, attrs in interpretation.member_attributes.items()
                }
            payload["interpretations"][str(interpretation.id)] = item
        if entry.peak_attributes is not None:
            payload["aggregations"] = [list(row) for row in entry.peak_attributes]
        spectra.append(payload)
    clusters = [
        {"attributes": [_json_attribute(CvParam(*_CLUSTER_KEY, key))] + [_json_attribute(a) for a in attrs]}
        for key, attrs in library.clusters.items()
    ]
    return {
        "format_version": "1.0",
        "attributes": [_json_attribute(CvParam(*_FORMAT_VERSION, "1.0"))]
        + [_json_attribute(a) for a in library.attributes],
        "spectrum_attribute_sets": {},
        "analyte_attribute_sets": {},
        "interpretation_attribute_sets": {},
        "cluster_attribute_sets": {},
        "spectra": spectra,
        "clusters": clusters,
    }


def _format_of(path: Path, format: str | None) -> Literal["text", "json"]:
    if format is None:
        suffixes = [s.lower() for s in path.suffixes]
        if suffixes and suffixes[-1] == ".gz":
            suffixes = suffixes[:-1]
        return "json" if suffixes and suffixes[-1] == ".json" else "text"
    if format not in ("text", "json"):
        raise SpxtacularError(f"format must be 'text' or 'json', got {format!r}")
    return format  # type: ignore[return-value]


def write_mzspeclib(
    entries: SpectralLibrary | Iterable[LibraryEntry] | LibraryEntry,
    path: str | Path,
    *,
    format: Literal["text", "json"] | None = None,
) -> Path:
    """Write library entries as an mzSpecLib 1.0 text or JSON file.

    Parameters
    ----------
    entries
        A :class:`SpectralLibrary` (its attributes and clusters are written
        too), an iterable of :class:`LibraryEntry`, or one entry.
    path
        Output path. A ``.gz`` suffix gzips the output.
    format
        ``"text"`` or ``"json"``. By default JSON when the name ends in
        ``.json`` (before any ``.gz``), text otherwise.

    Returns
    -------
    Path
        The path written.

    Raises
    ------
    SpxtacularError
        If a spectrum is profile data or holds neutral masses, has more than one
        precursor, an ion mobility of type ``im``, an activation type without a
        PSI-MS term, a duplicate key, or an attribute that repeats a written field.

    Notes
    -----
    Spectrum fields without a mzSpecLib term (native id, resolution, analyzer,
    ranges, per-peak charge / ion mobility arrays, ...) are not written. Entries
    without a key are numbered by position.
    """
    out = Path(path)
    library = entries if isinstance(entries, SpectralLibrary) else None
    if library is None:
        items: Iterable[LibraryEntry] = [entries] if isinstance(entries, LibraryEntry) else entries  # type: ignore[list-item]
        library = SpectralLibrary(entries=list(items))
    keyed = _keyed_entries(library.entries)
    chosen = _format_of(out, format)
    if chosen == "json":
        document = _json_document(library, keyed)
        with _open_text_write(out) as fh:
            json.dump(document, fh, indent=1, allow_nan=False)
            fh.write("\n")
    else:
        with _open_text_write(out) as fh:
            _write_text(fh, library, keyed)
    return out
