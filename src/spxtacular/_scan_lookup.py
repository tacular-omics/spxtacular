"""Helpers shared by every reader's ``get_by_scan`` / ``get_by_native_id`` / ``get_by_sage_scannr``.

Kept in its own module because ``reader.py`` imports the peak-list and Thermo
readers, so the shared pieces cannot live there. Standard library only.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, NamedTuple, Protocol

from .errors import SpxtacularError

if TYPE_CHECKING:
    from .core import MsnSpectrum


#: Native-id keys that may sit next to ``scan=`` without making it ambiguous
#: (Thermo: ``controllerType=0 controllerNumber=1 scan=19``).
_THERMO_ID_KEYS = frozenset({"controllertype", "controllernumber"})


def native_id_scan_number(native_id: str) -> int | None:
    """The scan number a native id carries on its own, or ``None``.

    Only ids whose number identifies the spectrum by itself count:

    - ``scan=19`` and Thermo ``controllerType=0 controllerNumber=1 scan=19`` -> 19
    - ``index=5`` (multiple peak list) and ``spectrum=5`` -> 5, verbatim

    Every other id gives ``None``: in Bruker ``frame=… scan=…`` and Waters
    ``function=… process=… scan=…`` the ``scan`` value repeats across frames or
    functions, and SCIEX ``sample=… period=… cycle=… experiment=…`` has no
    single number.
    """
    keys: dict[str, str] = {}
    for token in native_id.split():
        key, sep, value = token.partition("=")
        if sep:
            keys[key.lower()] = value
    if "scan" in keys and set(keys) - {"scan"} <= _THERMO_ID_KEYS:
        value = keys["scan"]
    elif set(keys) == {"index"}:
        value = keys["index"]
    elif set(keys) == {"spectrum"}:
        value = keys["spectrum"]
    else:
        return None
    try:
        return int(value)
    except ValueError:
        return None


class IdIndex[K](NamedTuple):
    """Scan-number and native-id index of a file, built once per reader.

    ``K`` is whatever the reader fetches a record by (an mzML native id, a byte
    offset into a peak list).

    Attributes
    ----------
    scans:
        Scan number -> record, for scan numbers held by exactly one spectrum.
    ids:
        Native id -> record, for native ids held by exactly one spectrum.
    duplicate_scans:
        Scan numbers held by more than one spectrum.
    duplicate_ids:
        Native ids held by more than one spectrum.
    """

    scans: dict[int, K]
    ids: dict[str, K]
    duplicate_scans: frozenset[int]
    duplicate_ids: frozenset[str]


def build_id_index[K](records: Iterable[tuple[int | None, str | None, K]]) -> IdIndex[K]:
    """Index ``(scan_number, native_id, record)`` triples, setting every duplicate aside."""
    scans: dict[int, K] = {}
    ids: dict[str, K] = {}
    duplicate_scans: set[int] = set()
    duplicate_ids: set[str] = set()
    for scan_number, native_id, record in records:
        if native_id is not None:
            if native_id in ids:
                duplicate_ids.add(native_id)
            ids[native_id] = record
        if scan_number is not None:
            if scan_number in scans:
                duplicate_scans.add(scan_number)
            scans[scan_number] = record
    for scan_number in duplicate_scans:
        del scans[scan_number]
    for native_id in duplicate_ids:
        del ids[native_id]
    return IdIndex(scans, ids, frozenset(duplicate_scans), frozenset(duplicate_ids))


def check_ms_level(ms_level: int | None) -> None:
    """Reject an ``ms_level`` that is not ``None`` or a positive integer."""
    if ms_level is None:
        return
    if isinstance(ms_level, bool) or not isinstance(ms_level, int) or ms_level < 1:
        raise SpxtacularError(f"ms_level must be a positive integer or None, got {ms_level!r}")


def check_scan_number(scan_number: object) -> int:
    """Return ``scan_number`` as an ``int``, rejecting anything that is not one."""
    if isinstance(scan_number, bool) or not isinstance(scan_number, int):
        raise SpxtacularError(f"scan_number must be an int, got {type(scan_number).__name__} {scan_number!r}")
    return scan_number


class _NativeIdAndScanReader(Protocol):
    def get_by_native_id(self, native_id: str) -> MsnSpectrum: ...

    def get_by_scan(self, scan_number: int, *, ms_level: int | None = None) -> MsnSpectrum: ...


def by_sage_scannr(reader: _NativeIdAndScanReader, scannr: str | int) -> MsnSpectrum:
    """Resolve a Sage ``scannr`` against any non-Bruker reader.

    Sage writes the spectrum's id verbatim into ``results.sage.tsv``: the mzML
    native id, or the MGF ``TITLE``. Its ``.pin`` output instead keeps only the
    number from ``scan=(\\d+)``. So the value is tried as a native id and, when it
    is a bare integer, as a scan number too.

    Raises
    ------
    SpxtacularError
        If a bare integer names one spectrum as a native id and a different one
        as a scan number (an MGF with ``TITLE=12`` on one spectrum and
        ``SCANS=12`` on another).
    KeyError
        If neither lookup finds a spectrum.
    """
    if isinstance(scannr, bool) or not isinstance(scannr, (str, int)):
        raise SpxtacularError(f"scannr must be a str or int, got {type(scannr).__name__} {scannr!r}")
    text = str(scannr).strip()
    if not text:
        raise SpxtacularError("scannr is empty")
    try:
        by_id: MsnSpectrum | None = reader.get_by_native_id(text)
    except KeyError:
        if not text.isdigit():
            raise
        by_id = None
    if not text.isdigit():
        assert by_id is not None
        return by_id
    if by_id is None:
        return reader.get_by_scan(int(text))
    try:
        by_scan = reader.get_by_scan(int(text))
    except (KeyError, SpxtacularError):
        return by_id
    if not _same_spectrum(by_id, by_scan):
        raise SpxtacularError(
            f"scannr {text!r} is ambiguous: it is the native id {by_id.native_id!r} "
            f"and the scan number of spectrum {by_scan.native_id!r}"
        )
    return by_id


def _same_spectrum(a: MsnSpectrum, b: MsnSpectrum) -> bool:
    if a.native_id is not None or b.native_id is not None:
        return a.native_id == b.native_id and a.scan_number == b.scan_number
    return a.to_dict() == b.to_dict()
