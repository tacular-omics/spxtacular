"""The scan-number index must not answer for an mzML offset index that has since been rebuilt.

mzmlpy 0.10.3 made the document the authority and the offset index a cache of it: reading a
file whose index does not list every spectrum lengthens that index mid-iteration.
:class:`~spxtacular.reader.MzmlReader` caches a scan-number index built from those ids, so the
cache has to notice.

The document builder below is a copy of mzmlpy's ``tests/test_index_document_agreement.py``,
reduced to spectra, rather than an import: that module is another package's test code and
nothing installs it.
"""

import tempfile
import warnings
from pathlib import Path

import pytest

pytest.importorskip("mzmlpy")

from spxtacular.reader import MzmlReader  # noqa: E402

HEADER = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<indexedmzML xmlns="http://psi.hupo.org/ms/mzml">\n'
    '<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0" id="agreement">\n'
    '<cvList count="1"><cv id="MS" fullName="PSI-MS" URI="https://purl.obolibrary.org/obo/ms.obo"/></cvList>\n'
    "<fileDescription><fileContent/></fileDescription>\n"
    '<softwareList count="1"><software id="sw" version="1"/></softwareList>\n'
    '<instrumentConfigurationList count="1"><instrumentConfiguration id="IC"/></instrumentConfigurationList>\n'
    '<dataProcessingList count="1"><dataProcessing id="DP"/></dataProcessingList>\n'
    '<run id="r" defaultInstrumentConfigurationRef="IC">\n'
)


def _array(accession: str, name: str) -> str:
    """An empty 64-bit binary data array. ``MzmlReader`` needs both m/z and intensity."""
    return (
        '<binaryDataArray encodedLength="0">'
        '<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>'
        '<cvParam cvRef="MS" accession="MS:1000576" name="no compression" value=""/>'
        f'<cvParam cvRef="MS" accession="{accession}" name="{name}" value=""/>'
        "<binary></binary></binaryDataArray>"
    )


def _record(index: int, identifier: str) -> str:
    return (
        f'<spectrum index="{index}" id="{identifier}" defaultArrayLength="0">'
        '<cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="1"/>'
        '<cvParam cvRef="MS" accession="MS:1000127" name="centroid spectrum" value=""/>'
        '<binaryDataArrayList count="2">'
        f"{_array('MS:1000514', 'm/z array')}{_array('MS:1000515', 'intensity array')}"
        "</binaryDataArrayList></spectrum>\n"
    )


def build(*, spectra: int, declared_spectra: int | None = None, indexed_spectra: list[int] | None = None) -> bytes:
    """An indexedmzML document whose index may name fewer spectra than the document holds.

    ``declared_spectra`` sets the ``spectrumList count`` attribute (by default the honest
    count) and ``indexed_spectra`` chooses which record positions get an ``<offset>`` entry
    (by default all of them).
    """
    body = bytearray(HEADER.encode())
    offsets: list[tuple[str, int]] = []
    count = spectra if declared_spectra is None else declared_spectra
    body += f'<spectrumList count="{count}" defaultDataProcessingRef="DP">\n'.encode()
    for position in range(spectra):
        identifier = f"scan={position}"
        offsets.append((identifier, len(body)))
        body += _record(position, identifier).encode()
    body += b"</spectrumList>\n</run>\n</mzML>\n"

    chosen = offsets if indexed_spectra is None else [offsets[position] for position in indexed_spectra]
    index_offset = len(body)
    body += b'<indexList count="1"><index name="spectrum">'
    for identifier, offset in chosen:
        body += f'<offset idRef="{identifier}">{offset}</offset>'.encode()
    body += b"</index></indexList>\n"
    body += f"<indexListOffset>{index_offset}</indexListOffset>\n</indexedmzML>\n".encode()
    return bytes(body)


def _written(tmp_path: Path, name: str, data: bytes) -> str:
    path = tmp_path / f"{name}.mzML"
    path.write_bytes(data)
    return str(path)


def _mzmlpy_lengthens_a_short_index() -> bool:
    """Whether reading a file whose index is short lengthens that index (mzmlpy 0.10.3).

    mzmlpy 0.10.2 already yields all three records, from the document and without warning, but
    leaves its offset index, count and ids short: the record stays unreachable by id or scan
    number whatever this reader caches, so there is nothing here to fix on that version. The
    count is probed rather than read off ``__version__``, which cannot tell a released 0.10.2
    from a checkout carrying the fix before its bump.
    """
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = _written(Path(directory), "probe", build(spectra=3, declared_spectra=2, indexed_spectra=[0, 2]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with MzmlReader(path) as reader:
                    for _ in reader.ms1:
                        pass
                    handle = reader._mzml_handle  # noqa: SLF001 - the probe is about mzmlpy's own index
                    return handle is not None and handle.spectra.count == 3
    except Exception:  # noqa: BLE001 - any failure to read the probe means no rebuild happens
        return False


@pytest.mark.skipif(
    not _mzmlpy_lengthens_a_short_index(),
    reason="this mzmlpy does not rebuild a short offset index while reading (needs mzmlpy 0.10.3)",
)
def test_scan_lookup_recovers_the_spectrum_the_mzml_index_omitted(tmp_path: Path) -> None:
    """A spectrum missing from the middle of the index is reachable once reading has proved it.

    The declared count matches the two entries the index does name and the last of them ends
    the list, so nothing bounded can tell at open that a third spectrum sits between them.
    Reading proves it, and mzmlpy rebuilds its offset index then; a scan-number index built
    from the short one would keep raising for a spectrum the walk has already yielded.
    """
    path = _written(tmp_path, "middle", build(spectra=3, declared_spectra=2, indexed_spectra=[0, 2]))
    expected = ["scan=0", "scan=1", "scan=2"]
    with MzmlReader(path) as reader:
        with pytest.raises(KeyError):
            reader.get_by_scan(1)  # the file's own index does not name it, so far
        # Every warning of the walk is captured together: rebuilding the index warns that it
        # was short, and then warns again that the declared count disagrees. Under
        # ``filterwarnings = ["error"]`` a warning left outside ``pytest.warns`` fails the test.
        with pytest.warns(UserWarning) as records:
            assert [spectrum.native_id for spectrum in reader.ms1] == expected
        messages = [str(record.message) for record in records]
        assert any("does not list every spectrum" in message for message in messages), messages

        assert reader.get_by_scan(1).native_id == "scan=1"
        assert reader["scan=1"].native_id == "scan=1"
        assert reader.get_by_native_id("scan=1").native_id == "scan=1"
        assert [spectrum.native_id for spectrum in reader.ms1] == expected
        for position, identifier in enumerate(expected):
            assert reader.get_by_scan(position).native_id == identifier
            assert reader[position].native_id == identifier


def test_complete_index_is_read_once_and_not_rebuilt(tmp_path: Path) -> None:
    """The invalidation must not fire on a sound file: no warning, and every accessor agrees."""
    path = _written(tmp_path, "complete", build(spectra=3))
    expected = ["scan=0", "scan=1", "scan=2"]
    with MzmlReader(path) as reader:
        assert [spectrum.native_id for spectrum in reader.ms1] == expected
        for position, identifier in enumerate(expected):
            assert reader.get_by_scan(position).native_id == identifier
            assert reader[identifier].native_id == identifier
