"""MGF, MS2, and MSP peak-list reading and writing — pure standard library.

Unlike the Bruker, mzML, and Thermo readers, nothing here is behind an optional
extra. MGF, MS2, and MSP (the NIST spectral-library format) are plain text and
are parsed and written with nothing but the standard library plus numpy. All
three formats are always available.

All three hold fragmentation spectra only, so every spectrum read back is an
:class:`~spxtacular.core.MsnSpectrum` with ``ms_level=2`` and
``spectrum_type=SpectrumType.CENTROID``.

Reading::

    from spxtacular import MgfReader, Ms2Reader, MspReader

    with MgfReader("run.mgf") as reader:      # .mgf.gz works too
        for spec in reader:
            print(spec.precursors[0].mz, len(spec))

Writing::

    from spxtacular import write_mgf, write_ms2, write_msp

    write_mgf(spectra, "out.mgf")
    write_ms2(spectra, "out.ms2.gz")          # gzipped by suffix
    write_msp(spectra, "library.msp")
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import IO, Self

import numpy as np
import peptacular as pt

from .core import MsnSpectrum, Precursor, Spectrum, SpectrumType
from .enums import Polarity
from .utils import format_precursor_charge, signed_precursor_charge

__all__ = ["MgfReader", "Ms2Reader", "MspReader", "PeakListLookup", "write_mgf", "write_ms2", "write_msp"]


# ---------------------------------------------------------------------------
# Low-level text helpers
# ---------------------------------------------------------------------------

_GZIP_MAGIC = b"\x1f\x8b"

# Lines opening with one of these are comments in the wild (Mascot uses ``#``,
# MS2 writers have used all four). They are skipped wherever they appear.
_COMMENT_PREFIXES = "#;!/"

_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_INT_RE = re.compile(r"[-+]?\d+")
_CHARGE_RE = re.compile(r"^([-+]?)(\d+)([-+]?)$")
# Peak lines are whitespace separated; some writers use commas.
_PEAK_SPLIT_RE = re.compile(r"[\s,]+")


def _is_gzip(path: Path) -> bool:
    """True when the file starts with the gzip magic bytes."""
    with open(path, "rb") as fh:
        return fh.read(2) == _GZIP_MAGIC


def _open_text(path: Path) -> IO[str]:
    """Open a peak-list file for reading, transparently decompressing gzip.

    Detection is by magic bytes, so a gzipped file keeps working under any name.
    Undecodable bytes are replaced rather than raising — vendor TITLE lines carry
    all sorts of encodings and a spectrum file should not be unreadable over one.
    """
    if _is_gzip(path):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def _open_text_write(path: Path) -> IO[str]:
    """Open a peak-list file for writing, gzipping when the path ends in ``.gz``."""
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "wt", encoding="utf-8", newline="\n")
    return open(path, "w", encoding="utf-8", newline="\n")


def _fmt(value: float) -> str:
    """Format a float at repr precision, so a write→read round trip is exact."""
    return repr(float(value))


def _parse_float(text: str, *, field_name: str, path: Path, line_no: int) -> float:
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"{path}:{line_no}: could not parse {field_name} from {text!r}") from None


def _first_float(text: str, *, field_name: str, path: Path, line_no: int) -> float:
    """Leading float of a value, tolerating ranges such as ``"120.5-130.5"``."""
    match = _FLOAT_RE.match(text.strip())
    if match is None:
        raise ValueError(f"{path}:{line_no}: could not parse {field_name} from {text!r}")
    return float(match.group())


def _first_int(text: str, *, field_name: str, path: Path, line_no: int) -> int:
    """Leading integer of a value, tolerating ranges such as ``"1024-1030"``."""
    match = _INT_RE.match(text.strip())
    if match is None:
        raise ValueError(f"{path}:{line_no}: could not parse {field_name} from {text!r}")
    return int(match.group())


def _parse_charge(text: str, *, path: Path, line_no: int) -> int:
    """Parse one charge token: ``2``, ``2+``, ``+2``, ``3-`` all work.

    A trailing (or leading) ``-`` yields a negative charge, which is how both
    formats encode negative-mode data. Multi-charge values (``"2+ and 3+"``,
    ``"2+,3+"``) are reduced to their first entry by :func:`_first_charge`.
    """
    token = text.strip()
    match = _CHARGE_RE.match(token)
    if match is None:
        raise ValueError(f"{path}:{line_no}: could not parse charge from {text!r}")
    lead, digits, trail = match.groups()
    sign = trail or lead or "+"
    return -int(digits) if sign == "-" else int(digits)


def _first_charge(text: str, *, path: Path, line_no: int) -> int | None:
    """First charge of a possibly multi-charge value; ``None`` if there is none.

    MGF permits ``CHARGE=2+ and 3+`` (and converters emit ``CHARGE=2+,3+``).
    Only the first state is kept — ``Precursor.charge`` is a single value.
    """
    tokens = [t for t in re.split(r"[\s,;]+", text.strip()) if t and t.lower() != "and"]
    if not tokens:
        return None
    return _parse_charge(tokens[0], path=path, line_no=line_no)


def _polarity_of(charge: int | None) -> Polarity | None:
    """Polarity implied by a signed charge — neither format states it outright."""
    if charge is None or charge == 0:
        return None
    return Polarity.NEGATIVE if charge < 0 else Polarity.POSITIVE


# ---------------------------------------------------------------------------
# MGF parsing
# ---------------------------------------------------------------------------


@dataclass
class _MgfBlock:
    """One ``BEGIN IONS`` … ``END IONS`` block, mid-parse."""

    begin_line: int
    # header key (upper-cased) -> (raw value, line number it was seen on)
    headers: dict[str, tuple[str, int]] = field(default_factory=dict)
    mz: list[float] = field(default_factory=list)
    intensity: list[float] = field(default_factory=list)
    charge: list[int | None] = field(default_factory=list)


def _iter_mgf(handle: IO[str], path: Path) -> Iterator[MsnSpectrum]:
    """Yield one :class:`MsnSpectrum` per ``BEGIN IONS`` block."""
    block: _MgfBlock | None = None

    for line_no, raw in enumerate(handle, start=1):
        line = raw.strip()
        if not line or line[0] in _COMMENT_PREFIXES:
            continue

        upper = line.upper()
        if upper == "BEGIN IONS":
            if block is not None:
                raise ValueError(
                    f"{path}:{line_no}: 'BEGIN IONS' inside the spectrum block opened at "
                    f"line {block.begin_line} (missing 'END IONS')"
                )
            block = _MgfBlock(begin_line=line_no)
            continue

        if upper == "END IONS":
            if block is None:
                raise ValueError(f"{path}:{line_no}: 'END IONS' without a matching 'BEGIN IONS'")
            yield _mgf_spectrum(block, path=path)
            block = None
            continue

        if block is None:
            # Global headers (``SEARCH=MIS``, ``CHARGE=2+,3+``, …) and any stray
            # text between blocks. Deliberately ignored rather than rejected.
            continue

        if "=" in line:
            key, _, value = line.partition("=")
            block.headers[key.strip().upper()] = (value.strip(), line_no)
            continue

        parts = _PEAK_SPLIT_RE.split(line)
        if len(parts) < 2:
            raise ValueError(f"{path}:{line_no}: expected 'mz intensity' on an ion line, got {line!r}")
        block.mz.append(_parse_float(parts[0], field_name="peak m/z", path=path, line_no=line_no))
        block.intensity.append(_parse_float(parts[1], field_name="peak intensity", path=path, line_no=line_no))
        # A third column is an optional per-peak charge (``100.0 25.0 1+``).
        block.charge.append(_parse_charge(parts[2], path=path, line_no=line_no) if len(parts) > 2 else None)

    if block is not None:
        raise ValueError(f"{path}:{block.begin_line}: unterminated spectrum block (missing 'END IONS')")


def _mgf_spectrum(block: _MgfBlock, *, path: Path) -> MsnSpectrum:
    """Build an :class:`MsnSpectrum` from a finished MGF block."""
    headers = block.headers

    precursor_mz: float | None = None
    precursor_intensity = 0.0
    if "PEPMASS" in headers:
        value, line_no = headers["PEPMASS"]
        parts = _PEAK_SPLIT_RE.split(value.strip())
        if not parts or not parts[0]:
            raise ValueError(f"{path}:{line_no}: PEPMASS has no m/z value")
        precursor_mz = _parse_float(parts[0], field_name="PEPMASS m/z", path=path, line_no=line_no)
        if len(parts) > 1 and parts[1]:
            precursor_intensity = _parse_float(parts[1], field_name="PEPMASS intensity", path=path, line_no=line_no)

    charge: int | None = None
    if "CHARGE" in headers:
        value, line_no = headers["CHARGE"]
        charge = _first_charge(value, path=path, line_no=line_no)

    scan_number: int | None = None
    if "SCANS" in headers:
        value, line_no = headers["SCANS"]
        scan_number = _first_int(value, field_name="SCANS", path=path, line_no=line_no)

    rt: float | None = None
    if "RTINSECONDS" in headers:
        value, line_no = headers["RTINSECONDS"]
        rt = _first_float(value, field_name="RTINSECONDS", path=path, line_no=line_no)
    elif "RTINMINUTES" in headers:
        # Not in the Matrix Science description of MGF, but written by enough
        # converters to be worth honouring. MsnSpectrum.rt is always seconds.
        value, line_no = headers["RTINMINUTES"]
        rt = _first_float(value, field_name="RTINMINUTES", path=path, line_no=line_no) * 60.0

    title = headers["TITLE"][0] if "TITLE" in headers else None

    precursors = None
    if precursor_mz is not None:
        precursors = [
            Precursor(
                mz=precursor_mz,
                intensity=precursor_intensity,
                charge=charge,
                im=None,
                is_monoisotopic=None,
            )
        ]

    # Per-peak charges are only kept when every peak carries one — a partly
    # charged array cannot be represented, and the parallel-array contract
    # forbids a shorter one.
    charge_array: np.ndarray | None = None
    if block.mz and all(z is not None for z in block.charge):
        charge_array = np.asarray(block.charge, dtype=np.int32)

    return MsnSpectrum(
        mz=np.asarray(block.mz, dtype=np.float64),
        intensity=np.asarray(block.intensity, dtype=np.float64),
        charge=charge_array,
        im=None,
        spectrum_type=SpectrumType.CENTROID,
        scan_number=scan_number,
        ms_level=2,
        native_id=title,
        rt=rt,
        polarity=_polarity_of(charge),
        precursors=precursors,
    )


# ---------------------------------------------------------------------------
# MS2 parsing
# ---------------------------------------------------------------------------

# ``I`` info keys this reader understands, upper-cased. Everything else is kept
# out of the spectrum rather than guessed at.
_MS2_RTIME_KEYS = frozenset({"RTIME", "RETTIME"})


@dataclass
class _Ms2Block:
    """One ``S`` scan record, mid-parse."""

    scan_line: int
    scan_number: int | None = None
    precursor_mz: float | None = None
    precursor_intensity: float = 0.0
    charges: list[int] = field(default_factory=list)
    rt: float | None = None
    injection_time: float | None = None
    total_ion_current: float | None = None
    activation_type: str | None = None
    mz: list[float] = field(default_factory=list)
    intensity: list[float] = field(default_factory=list)


def _iter_ms2(handle: IO[str], path: Path) -> Iterator[MsnSpectrum]:
    """Yield one :class:`MsnSpectrum` per ``S`` record."""
    block: _Ms2Block | None = None

    for line_no, raw in enumerate(handle, start=1):
        line = raw.strip()
        if not line or line[0] in _COMMENT_PREFIXES:
            continue

        fields = line.split()
        tag = fields[0].upper()

        if tag == "H":  # file-level header
            continue
        if tag == "D":  # per-scan analysis data (charge-state predictions, …)
            continue

        if tag == "S":
            if block is not None:
                yield _ms2_spectrum(block)
            if len(fields) < 4:
                raise ValueError(
                    f"{path}:{line_no}: expected 'S <first_scan> <last_scan> <precursor_mz>', got {line!r}"
                )
            block = _Ms2Block(
                scan_line=line_no,
                scan_number=_first_int(fields[1], field_name="scan number", path=path, line_no=line_no),
                precursor_mz=_parse_float(fields[3], field_name="precursor m/z", path=path, line_no=line_no),
            )
            continue

        if block is None:
            raise ValueError(f"{path}:{line_no}: {line!r} appears before any 'S' scan line")

        if tag == "Z":
            if len(fields) < 2:
                raise ValueError(f"{path}:{line_no}: expected 'Z <charge> <mass>', got {line!r}")
            # Repeated Z lines mean several candidate charge states; all are
            # parsed (so a malformed one still errors) and the first is used.
            block.charges.append(_parse_charge(fields[1], path=path, line_no=line_no))
            continue

        if tag == "I":
            if len(fields) < 3:
                continue  # a key with no value — nothing to record
            key = fields[1].upper()
            value = " ".join(fields[2:])
            if key in _MS2_RTIME_KEYS:
                # RTime is minutes in every writer that emits it; rt is seconds.
                block.rt = _first_float(value, field_name=fields[1], path=path, line_no=line_no) * 60.0
            elif key == "IONINJECTIONTIME":
                block.injection_time = _first_float(value, field_name=fields[1], path=path, line_no=line_no)
            elif key == "PRECURSORINT":
                block.precursor_intensity = _first_float(value, field_name=fields[1], path=path, line_no=line_no)
            elif key == "TIC":
                block.total_ion_current = _first_float(value, field_name=fields[1], path=path, line_no=line_no)
            elif key == "ACTIVATIONTYPE":
                block.activation_type = value
            continue

        # Anything else must be an ion line.
        if len(fields) < 2:
            raise ValueError(f"{path}:{line_no}: expected 'mz intensity' on an ion line, got {line!r}")
        block.mz.append(_parse_float(fields[0], field_name="peak m/z", path=path, line_no=line_no))
        block.intensity.append(_parse_float(fields[1], field_name="peak intensity", path=path, line_no=line_no))

    if block is not None:
        yield _ms2_spectrum(block)


def _ms2_spectrum(block: _Ms2Block) -> MsnSpectrum:
    """Build an :class:`MsnSpectrum` from a finished MS2 ``S`` record."""
    charge = block.charges[0] if block.charges else None
    precursors = None
    if block.precursor_mz is not None:
        precursors = [
            Precursor(
                mz=block.precursor_mz,
                intensity=block.precursor_intensity,
                charge=charge,
                im=None,
                is_monoisotopic=None,
            )
        ]
    return MsnSpectrum(
        mz=np.asarray(block.mz, dtype=np.float64),
        intensity=np.asarray(block.intensity, dtype=np.float64),
        charge=None,
        im=None,
        spectrum_type=SpectrumType.CENTROID,
        scan_number=block.scan_number,
        ms_level=2,
        native_id=f"scan={block.scan_number}" if block.scan_number is not None else None,
        rt=block.rt,
        injection_time=block.injection_time,
        total_ion_current=block.total_ion_current,
        polarity=_polarity_of(charge),
        activation_type=block.activation_type,
        precursors=precursors,
    )


# ---------------------------------------------------------------------------
# MSP parsing
# ---------------------------------------------------------------------------

# MSP has no formal spec, and the proteomics (NIST peptide libraries) and
# metabolomics (MoNA, GNPS, MS-DIAL) dialects spell their headers differently:
# ``Num Peaks`` vs ``Num peaks``, ``PrecursorMZ`` vs ``PRECURSORMZ`` vs
# ``Precursor_mz``, ``Ion_mode`` vs ``IONMODE``. Keys are therefore normalised
# by upper-casing and dropping spaces/underscores/hyphens before lookup.
_MSP_KEY_STRIP_RE = re.compile(r"[\s_\-]+")


def _msp_key(key: str) -> str:
    return _MSP_KEY_STRIP_RE.sub("", key).upper()


_MSP_NUM_PEAKS_KEY = "NUMPEAKS"

# ``Comment:`` in NIST peptide libraries is a run of space-separated
# ``Key=value`` pairs, values optionally double-quoted.
_MSP_COMMENT_PAIR_RE = re.compile(r'([\w./-]+)=("[^"]*"|\S+)')

# A trailing ``/2`` on a peptide ``Name`` is the precursor charge.
_MSP_NAME_CHARGE_RE = re.compile(r"/(\d+)\s*$")


@dataclass
class _MspBlock:
    """One MSP record, mid-parse."""

    start_line: int
    # normalised header key -> (raw value, line number it was seen on)
    headers: dict[str, tuple[str, int]] = field(default_factory=dict)
    num_peaks: int | None = None
    num_peaks_line: int = 0
    mz: list[float] = field(default_factory=list)
    intensity: list[float] = field(default_factory=list)


def _iter_msp(handle: IO[str], path: Path) -> Iterator[MsnSpectrum]:
    """Yield one :class:`MsnSpectrum` per MSP record.

    MSP records have no BEGIN/END markers: a record is header lines up to
    ``Num Peaks: N``, then exactly N peaks. Parsing is count-driven — the
    declared count is both metadata and the record terminator, so a mismatch in
    either direction is a structural error, not a guess.
    """
    block: _MspBlock | None = None

    for line_no, raw in enumerate(handle, start=1):
        line = raw.strip()
        if line and line[0] in _COMMENT_PREFIXES:
            continue

        if block is not None and block.num_peaks is not None:
            # Inside the peak list. A blank line here means the record declared
            # more peaks than it holds.
            if not line:
                raise ValueError(
                    f"{path}:{line_no}: record starting at line {block.start_line} declares "
                    f"{block.num_peaks} peaks but ends after {len(block.mz)}"
                )
            # Several peaks may share a line, semicolon-separated (NIST allows
            # it). Within one chunk the first two tokens are m/z and intensity;
            # anything after — usually a quoted annotation — is ignored.
            for chunk in line.split(";"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                if len(block.mz) >= block.num_peaks:
                    raise ValueError(
                        f"{path}:{line_no}: record starting at line {block.start_line} declares "
                        f"{block.num_peaks} peaks but holds more"
                    )
                parts = _PEAK_SPLIT_RE.split(chunk)
                if len(parts) < 2:
                    raise ValueError(f"{path}:{line_no}: expected 'mz intensity' on a peak line, got {chunk!r}")
                block.mz.append(_parse_float(parts[0], field_name="peak m/z", path=path, line_no=line_no))
                block.intensity.append(_parse_float(parts[1], field_name="peak intensity", path=path, line_no=line_no))
            if len(block.mz) == block.num_peaks:
                yield _msp_spectrum(block, path=path)
                block = None
            continue

        if not line:
            if block is not None:
                raise ValueError(
                    f"{path}:{block.start_line}: record has no 'Num Peaks' line before the blank line at line {line_no}"
                )
            continue

        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"{path}:{line_no}: expected a 'Key: value' header line, got {line!r}")
        if block is None:
            block = _MspBlock(start_line=line_no)
        normalised = _msp_key(key)
        if normalised == _MSP_NUM_PEAKS_KEY:
            block.num_peaks = _first_int(value, field_name="Num Peaks", path=path, line_no=line_no)
            block.num_peaks_line = line_no
            if block.num_peaks < 0:
                raise ValueError(f"{path}:{line_no}: negative 'Num Peaks' count {block.num_peaks}")
            if block.num_peaks == 0:
                yield _msp_spectrum(block, path=path)
                block = None
        else:
            block.headers[normalised] = (value.strip(), line_no)

    if block is not None:
        if block.num_peaks is None:
            raise ValueError(f"{path}:{block.start_line}: record has no 'Num Peaks' line before end of file")
        raise ValueError(
            f"{path}:{block.start_line}: record declares {block.num_peaks} peaks "
            f"but the file ends after {len(block.mz)}"
        )


def _msp_comment_pairs(block: _MspBlock) -> dict[str, str]:
    """``Key=value`` pairs of the ``Comment:``/``Comments:`` header, keys upper-cased."""
    for key in ("COMMENT", "COMMENTS"):
        if key in block.headers:
            value, _ = block.headers[key]
            return {k.upper(): v.strip('"') for k, v in _MSP_COMMENT_PAIR_RE.findall(value)}
    return {}


def _msp_spectrum(block: _MspBlock, *, path: Path) -> MsnSpectrum:
    """Build an :class:`MsnSpectrum` from a finished MSP record."""
    headers = block.headers
    comment = _msp_comment_pairs(block)

    def header(*keys: str) -> tuple[str, int] | None:
        for key in keys:
            if key in headers:
                return headers[key]
        return None

    name = header("NAME")
    native_id = name[0] if name is not None else None

    precursor_mz: float | None = None
    found = header("PRECURSORMZ", "PRECURSORM/Z")
    if found is not None:
        precursor_mz = _first_float(found[0], field_name="PrecursorMZ", path=path, line_no=found[1])
    elif "PARENT" in comment:
        precursor_mz = _first_float(comment["PARENT"], field_name="Comment Parent", path=path, line_no=block.start_line)

    charge: int | None = None
    found = header("CHARGE")
    if found is not None:
        charge = _first_charge(found[0], path=path, line_no=found[1])
    elif "CHARGE" in comment:
        charge = _first_charge(comment["CHARGE"], path=path, line_no=block.start_line)
    elif native_id is not None:
        match = _MSP_NAME_CHARGE_RE.search(native_id)
        if match is not None:
            charge = int(match.group(1))

    polarity: Polarity | None = None
    found = header("IONMODE", "POLARITY")
    if found is not None:
        mode = found[0].strip().upper()
        if mode.startswith("P"):
            polarity = Polarity.POSITIVE
        elif mode.startswith("N"):
            polarity = Polarity.NEGATIVE
    if polarity is None:
        polarity = _polarity_of(charge)

    # MSP has no retention-time unit convention — NIST peptide libraries write
    # seconds, most metabolomics exporters write minutes. The value is kept
    # exactly as given rather than silently guessed at; know your library.
    rt: float | None = None
    found = header("RETENTIONTIME", "RT")
    if found is not None:
        rt = _first_float(found[0], field_name="RetentionTime", path=path, line_no=found[1])
    elif "RT" in comment:
        rt = _first_float(comment["RT"], field_name="Comment RT", path=path, line_no=block.start_line)

    collision_energy: float | None = None
    found = header("COLLISIONENERGY", "CE")
    if found is not None:
        # Real files hold "35", "35 eV", "HCD 35%": the first number is the energy.
        collision_energy = _first_float(found[0], field_name="Collision_energy", path=path, line_no=found[1])
    elif "CE" in comment:
        collision_energy = _first_float(comment["CE"], field_name="Comment CE", path=path, line_no=block.start_line)

    precursors = None
    if precursor_mz is not None:
        precursors = [
            Precursor(
                mz=precursor_mz,
                intensity=0.0,
                charge=charge,
                im=None,
                is_monoisotopic=None,
            )
        ]

    return MsnSpectrum(
        mz=np.asarray(block.mz, dtype=np.float64),
        intensity=np.asarray(block.intensity, dtype=np.float64),
        charge=None,
        im=None,
        spectrum_type=SpectrumType.CENTROID,
        scan_number=None,
        ms_level=2,
        native_id=native_id,
        rt=rt,
        polarity=polarity,
        collision_energy=collision_energy,
        precursors=precursors,
    )


# ---------------------------------------------------------------------------
# Lookup object
# ---------------------------------------------------------------------------


class PeakListLookup:
    """Iterable + index-accessible spectra from a peak-list file.

    Iteration yields spectra filtered to ``ms_level`` (if given). Since MGF and
    MS2 hold fragmentation spectra only, ``reader.ms1`` is a valid but always
    empty walk, and ``reader.ms2`` yields the whole file.

    Index access (``lookup[int]`` or ``lookup[str]``) fetches by 0-based position
    in the file or by ``native_id``, and is *not* level-filtered (matching
    ``MzmlSpectraLookup``). Both forms stream from the start of the file, so
    random access is O(n) — iterate when you want every spectrum.
    """

    def __init__(self, reader: _PeakListReader, ms_level: int | None = None) -> None:
        self._reader = reader
        self._ms_level = ms_level

    def __iter__(self) -> Iterator[MsnSpectrum]:
        for spec in self._reader._iter_spectra():
            if self._ms_level is not None and spec.ms_level != self._ms_level:
                continue
            yield spec

    def __getitem__(self, key: int | str) -> MsnSpectrum:
        """Fetch a single spectrum by 0-based index or native ID string."""
        if isinstance(key, int):
            if key < 0:
                raise IndexError(f"negative indices are not supported by {type(self._reader).__name__}: {key}")
            for i, spec in enumerate(self._reader._iter_spectra()):
                if i == key:
                    return spec
            raise IndexError(f"spectrum index {key} out of range for {self._reader.path}")
        for spec in self._reader._iter_spectra():
            if spec.native_id == key:
                return spec
        raise KeyError(f"no spectrum with native_id {key!r} in {self._reader.path}")


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


class _PeakListReader:
    """Shared plumbing for the text peak-list readers.

    Every walk streams its own file handle, so iterations are independent and may
    be nested; :meth:`open` and :meth:`close` exist for interface symmetry with
    the other readers and hold nothing open.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._n_spectra: int | None = None

    # -- interface symmetry with DReader / MzmlReader -------------------------

    def open(self) -> None:
        """Check the file is present. No handle is held — each walk opens its own."""
        if not self.path.exists():
            raise FileNotFoundError(f"{type(self).__name__}: no such file: {self.path}")

    def close(self) -> None:
        """No-op — peak-list readers never hold a file handle between walks."""

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

    # -- format hooks --------------------------------------------------------

    def _parse(self, handle: IO[str]) -> Iterator[MsnSpectrum]:
        raise NotImplementedError

    def _is_record_start(self, line: str) -> bool:
        raise NotImplementedError

    # -- public API ----------------------------------------------------------

    def _iter_spectra(self) -> Iterator[MsnSpectrum]:
        with _open_text(self.path) as handle:
            yield from self._parse(handle)

    def __iter__(self) -> Iterator[MsnSpectrum]:
        """Every spectrum in the file, in file order."""
        return self._iter_spectra()

    def __len__(self) -> int:
        """Number of spectra in the file.

        Counted with one pass that only looks at record-start lines (no peaks are
        parsed), then cached — so it is cheap, but not free, on first call.
        """
        if self._n_spectra is None:
            with _open_text(self.path) as handle:
                self._n_spectra = sum(1 for line in handle if self._is_record_start(line.strip()))
        return self._n_spectra

    def __getitem__(self, key: int | str) -> MsnSpectrum:
        """Fetch a single spectrum by 0-based index or native ID string."""
        return PeakListLookup(self)[key]

    @property
    def ms1(self) -> PeakListLookup:
        """Always empty — peak lists carry no survey scans. Present so generic code works."""
        return PeakListLookup(self, ms_level=1)

    @property
    def ms2(self) -> PeakListLookup:
        """Every spectrum in the file — supports iteration and index/native-ID access."""
        return PeakListLookup(self, ms_level=2)


class MgfReader(_PeakListReader):
    """Read Mascot Generic Format (``.mgf``, optionally gzipped).

    Parsing is deliberately lenient: unknown ``KEY=VALUE`` headers, comment lines
    (``#;!/``), blank lines, and text outside ``BEGIN IONS``/``END IONS`` are all
    skipped. Structural damage — a stray ``END IONS``, a nested ``BEGIN IONS``, an
    unterminated block, an unparsable number — raises ``ValueError`` naming the
    file and line number.

    ``PEPMASS``, ``CHARGE``, ``TITLE``, ``SCANS``, ``RTINSECONDS`` and the
    non-standard ``RTINMINUTES`` are mapped onto ``MsnSpectrum`` fields; see the
    Readers documentation for the table.
    """

    def _parse(self, handle: IO[str]) -> Iterator[MsnSpectrum]:
        return _iter_mgf(handle, self.path)

    def _is_record_start(self, line: str) -> bool:
        return line.upper() == "BEGIN IONS"


class Ms2Reader(_PeakListReader):
    """Read the MS2 peak-list format (``.ms2``, optionally gzipped).

    ``H`` header lines and ``D`` analysis lines are skipped; ``S`` opens a scan,
    ``Z`` gives a candidate precursor charge (repeatable — the first is used),
    ``I`` carries info values, and everything else is an ion line. Structural
    damage raises ``ValueError`` naming the file and line number.
    """

    def _parse(self, handle: IO[str]) -> Iterator[MsnSpectrum]:
        return _iter_ms2(handle, self.path)

    def _is_record_start(self, line: str) -> bool:
        return line[:1].upper() == "S" and (len(line) == 1 or line[1].isspace())


class MspReader(_PeakListReader):
    """Read the MSP spectral-library format (``.msp``, optionally gzipped).

    Handles both dialects found in the wild: NIST/SpectraST peptide libraries
    (``Name: PEPTIDE/2``, metadata in ``Comment:`` ``Key=value`` pairs) and
    metabolomics exports from MoNA / GNPS / MS-DIAL (``PRECURSORMZ:``,
    ``IONMODE:``, …) — header keys are matched case-insensitively, ignoring
    spaces, underscores, and hyphens. Unknown headers (``Formula:``,
    ``SMILES:``, ``InChIKey:``, ``Synon:``…) and per-peak annotation columns
    are skipped: ``MsnSpectrum`` has no fields for them.

    Records are count-driven — ``Num Peaks: N`` ends the header and exactly
    ``N`` peaks must follow, so a count mismatch, a record with no ``Num
    Peaks`` line, or an unparsable number raises ``ValueError`` naming the
    file and line number.
    """

    def _parse(self, handle: IO[str]) -> Iterator[MsnSpectrum]:
        return _iter_msp(handle, self.path)

    def _is_record_start(self, line: str) -> bool:
        key, sep, _ = line.partition(":")
        return bool(sep) and _msp_key(key) == _MSP_NUM_PEAKS_KEY


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _as_spectra(spectra: Iterable[Spectrum] | Spectrum) -> Iterable[Spectrum]:
    """Allow a lone spectrum where an iterable is expected."""
    if isinstance(spectra, Spectrum):
        return [spectra]
    return spectra


def _check_writable(spec: Spectrum, index: int, fmt: str) -> None:
    if spec.spectrum_type == SpectrumType.PROFILE:
        raise ValueError(
            f"cannot write spectrum {index} to {fmt}: peak lists hold centroid data, not profile data. "
            "Call .centroid() first."
        )


def _meta(spec: Spectrum) -> MsnSpectrum | None:
    """The spectrum as an MsnSpectrum when it carries MS metadata, else ``None``."""
    return spec if isinstance(spec, MsnSpectrum) else None


def _first_precursor(spec: Spectrum) -> Precursor | None:
    msn = _meta(spec)
    if msn is None or not msn.precursors:
        return None
    return msn.precursors[0]


def _written_charge(spec: Spectrum) -> int | None:
    """Signed precursor charge to write.

    Neither format records polarity directly — the sign of the charge carries it.
    A negative-polarity spectrum whose precursor charge was recorded as positive
    is therefore written negative (and reads back negative).
    """
    prec = _first_precursor(spec)
    msn = _meta(spec)
    return signed_precursor_charge(
        prec.charge if prec is not None else None,
        msn.polarity if msn is not None else None,
    )


def write_mgf(spectra: Iterable[Spectrum] | Spectrum, path: str | Path) -> Path:
    """Write spectra to a Mascot Generic Format file.

    Parameters
    ----------
    spectra:
        Spectra to write. A lone :class:`~spxtacular.core.Spectrum` is accepted.
        Metadata is taken from :class:`~spxtacular.core.MsnSpectrum` fields where
        present; a plain ``Spectrum`` writes a block of peaks and nothing else.
    path:
        Output path. A ``.gz`` suffix gzips the output.

    Returns
    -------
    Path
        The path written.

    Raises
    ------
    ValueError
        If any spectrum is ``SpectrumType.PROFILE`` — peak lists are centroid data.

    Notes
    -----
    ``mz`` and ``intensity`` are written at repr precision, so reading the file
    back reproduces them exactly.
    """
    out = Path(path)
    with _open_text_write(out) as fh:
        for index, spec in enumerate(_as_spectra(spectra)):
            _check_writable(spec, index, "MGF")
            msn = _meta(spec)
            fh.write("BEGIN IONS\n")

            title = msn.native_id if msn is not None else None
            if title is None and msn is not None and msn.scan_number is not None:
                title = f"scan={msn.scan_number}"
            if title is not None:
                fh.write(f"TITLE={title}\n")

            if msn is not None and msn.scan_number is not None:
                fh.write(f"SCANS={msn.scan_number}\n")
            if msn is not None and msn.rt is not None:
                fh.write(f"RTINSECONDS={_fmt(msn.rt)}\n")

            prec = _first_precursor(spec)
            if prec is not None:
                # An intensity of 0.0 is what an absent one reads back as, so it
                # is left off rather than written out.
                if prec.intensity != 0.0:
                    fh.write(f"PEPMASS={_fmt(prec.mz)} {_fmt(prec.intensity)}\n")
                else:
                    fh.write(f"PEPMASS={_fmt(prec.mz)}\n")
            charge = _written_charge(spec)
            if charge is not None:
                fh.write(f"CHARGE={format_precursor_charge(charge)}\n")

            peak_charges = spec.charge
            for i in range(len(spec.mz)):
                line = f"{_fmt(spec.mz[i])} {_fmt(spec.intensity[i])}"
                if peak_charges is not None:
                    z = int(peak_charges[i])
                    line += f" {abs(z)}{'-' if z < 0 else '+'}"
                fh.write(line + "\n")

            fh.write("END IONS\n\n")
    return out


def write_ms2(spectra: Iterable[Spectrum] | Spectrum, path: str | Path) -> Path:
    """Write spectra to an MS2 peak-list file.

    Parameters
    ----------
    spectra:
        Spectra to write. A lone :class:`~spxtacular.core.Spectrum` is accepted.
    path:
        Output path. A ``.gz`` suffix gzips the output.

    Returns
    -------
    Path
        The path written.

    Raises
    ------
    ValueError
        If any spectrum is ``SpectrumType.PROFILE`` — peak lists are centroid data.

    Notes
    -----
    ``S`` needs a scan number and a precursor m/z, which a plain ``Spectrum`` does
    not have: the 1-based position in the input stands in for the scan number and
    the precursor m/z is written as ``0.0``. The ``Z`` line's mass is the singly
    protonated mass derived from the precursor m/z and charge; it is regenerated
    on write and ignored on read, so nothing depends on its precision.
    """
    out = Path(path)
    with _open_text_write(out) as fh:
        fh.write(f"H\tCreationDate\t{datetime.now().isoformat(timespec='seconds')}\n")
        fh.write("H\tExtractor\tspxtacular\n")
        fh.write(f"H\tExtractorVersion\t{_spxtacular_version()}\n")

        for index, spec in enumerate(_as_spectra(spectra)):
            _check_writable(spec, index, "MS2")
            msn = _meta(spec)
            prec = _first_precursor(spec)

            scan = msn.scan_number if msn is not None and msn.scan_number is not None else index + 1
            precursor_mz = prec.mz if prec is not None else 0.0
            fh.write(f"S\t{scan}\t{scan}\t{_fmt(precursor_mz)}\n")

            if msn is not None and msn.rt is not None:
                fh.write(f"I\tRTime\t{_fmt(msn.rt / 60.0)}\n")
            if msn is not None and msn.injection_time is not None:
                fh.write(f"I\tIonInjectionTime\t{_fmt(msn.injection_time)}\n")
            if msn is not None and msn.total_ion_current is not None:
                fh.write(f"I\tTIC\t{_fmt(msn.total_ion_current)}\n")
            if msn is not None and msn.activation_type is not None:
                fh.write(f"I\tActivationType\t{msn.activation_type}\n")
            if prec is not None and prec.intensity != 0.0:
                fh.write(f"I\tPrecursorInt\t{_fmt(prec.intensity)}\n")

            charge = _written_charge(spec)
            if charge is not None and prec is not None:
                fh.write(f"Z\t{charge}\t{_fmt(_mh_mass(prec.mz, charge))}\n")

            for i in range(len(spec.mz)):
                fh.write(f"{_fmt(spec.mz[i])} {_fmt(spec.intensity[i])}\n")
    return out


def write_msp(spectra: Iterable[Spectrum] | Spectrum, path: str | Path) -> Path:
    """Write spectra to an MSP spectral-library file.

    Parameters
    ----------
    spectra:
        Spectra to write. A lone :class:`~spxtacular.core.Spectrum` is accepted.
        Metadata is taken from :class:`~spxtacular.core.MsnSpectrum` fields where
        present; a plain ``Spectrum`` writes ``Num Peaks`` and the peaks alone.
    path:
        Output path. A ``.gz`` suffix gzips the output.

    Returns
    -------
    Path
        The path written.

    Raises
    ------
    ValueError
        If any spectrum is ``SpectrumType.PROFILE`` — peak lists are centroid data.

    Notes
    -----
    ``mz`` and ``intensity`` are written at repr precision, so reading the file
    back reproduces them exactly. Polarity goes out as an explicit
    ``Ion_mode: P``/``N`` line (MSP's own field — unlike MGF/MS2, no charge-sign
    encoding is needed), and ``rt`` is written verbatim under
    ``RetentionTime:``; spxtacular's ``rt`` is seconds, but MSP has no unit
    convention, so no conversion is applied in either direction.
    """
    out = Path(path)
    with _open_text_write(out) as fh:
        for index, spec in enumerate(_as_spectra(spectra)):
            _check_writable(spec, index, "MSP")
            msn = _meta(spec)

            name = msn.native_id if msn is not None else None
            if name is None and msn is not None and msn.scan_number is not None:
                name = f"scan={msn.scan_number}"
            if name is not None:
                fh.write(f"Name: {name}\n")

            prec = _first_precursor(spec)
            if prec is not None:
                fh.write(f"PrecursorMZ: {_fmt(prec.mz)}\n")
                if prec.charge is not None:
                    fh.write(f"Charge: {int(prec.charge)}\n")
            if msn is not None and msn.polarity is not None:
                fh.write(f"Ion_mode: {'N' if msn.polarity == Polarity.NEGATIVE else 'P'}\n")
            if msn is not None and msn.rt is not None:
                fh.write(f"RetentionTime: {_fmt(msn.rt)}\n")
            if msn is not None and msn.collision_energy is not None:
                fh.write(f"Collision_energy: {_fmt(msn.collision_energy)}\n")

            fh.write(f"Num Peaks: {len(spec.mz)}\n")
            for i in range(len(spec.mz)):
                fh.write(f"{_fmt(spec.mz[i])} {_fmt(spec.intensity[i])}\n")
            fh.write("\n")
    return out


def _mh_mass(mz: float, charge: int) -> float:
    """Singly protonated (M+H) mass implied by an m/z and charge — the ``Z`` line mass."""
    z = abs(charge)
    if z == 0:
        return float(mz)
    return (float(mz) - pt.PROTON_MASS) * z + pt.PROTON_MASS


def _spxtacular_version() -> str:
    # Imported lazily: spxtacular/__init__.py imports this module.
    from . import __version__

    return __version__
