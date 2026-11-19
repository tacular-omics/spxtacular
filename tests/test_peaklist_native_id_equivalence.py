"""``reader[native_id]`` must return what it returned before the native-id index existed.

The index (PR #128) made the string form of ``reader[key]`` seek instead of stream. That is a
performance change, so no lookup may give a different answer. This module checks that against
**recorded behaviour, not against a belief about it**:

``tests/data/peaklist_native_id_behaviour.json`` was produced by
``scripts/record_peaklist_native_id_behaviour.py`` run on commit ``ffeee28`` — main before the
index landed, whose ``peaklist.py`` differs from the 0.10.0 release only in docstrings. The
file records, for every corpus file and every probe key, either the spectrum that came back
(native id, scan number, peaks, and a digest of the full ``to_dict()``) or the exception type
and message. Its ``recorded_from_commit`` field is asserted below, so the expectations cannot
quietly become "whatever this tree does".

The corpus is written by ``tests/peaklist_native_id_corpus.py``: MGF, MS2 and MSP, each as
plain, CRLF, bare-CR, mixed-ending, gzip, truncated-gzip, no-final-newline, damaged-later-
record, id-less-record and duplicate-id variants, plus an MGF that ends inside a block.

Measured when this test was written: against the recording, the merge of #128 without the
fallback was identical in 301 of 370 lookups and returned a spectrum in 36 of the 69
differences; with the fallback it is 370 of 370.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from spxtacular import MgfReader, Ms2Reader, MsnSpectrum, MspReader
from tests.peaklist_native_id_corpus import PROBE_KEYS, build_corpus

BEHAVIOUR = pathlib.Path(__file__).parent / "data" / "peaklist_native_id_behaviour.json"
RECORDED_FROM = "ffeee283eeaa29ac1eeb0df746628b44357c58af"
READERS = {"mgf": MgfReader, "ms2": Ms2Reader, "msp": MspReader}


def _reader(case: str, path: pathlib.Path):
    """The reader for a case name, which always ends in the format it holds."""
    fmt = case.rsplit(".", 1)[-1]
    if fmt == "gz":
        fmt = case.rsplit(".", 2)[-2]
    return READERS[fmt](path)


def _digest(spec: MsnSpectrum) -> str:
    """The same digest the recorder wrote: every field of the serialised spectrum."""
    return hashlib.sha256(json.dumps(spec.to_dict(), sort_keys=True, default=repr).encode()).hexdigest()[:16]


def _outcome(case: str, path: pathlib.Path, key: object, root: pathlib.Path) -> dict:
    """What this tree does, in the recorder's shape so the two can be compared directly."""
    try:
        spec = _reader(case, path)[key]  # type: ignore[index]
    except Exception as exc:  # noqa: BLE001 - the recorded expectation may be an exception
        return {"error": type(exc).__name__, "message": str(exc).replace(str(root), "<corpus>")}
    return {
        "spectrum": {
            "native_id": spec.native_id,
            "scan_number": spec.scan_number,
            "mz": [float(v) for v in spec.mz],
            "intensity": [float(v) for v in spec.intensity],
            "digest": _digest(spec),
        }
    }


@pytest.fixture(scope="module")
def recorded() -> dict:
    """The behaviour recorded on main, with the commit it came from asserted."""
    payload = json.loads(BEHAVIOUR.read_text())
    assert payload["recorded_from_commit"] == RECORDED_FROM, (
        "the expectations were re-recorded on another commit; re-read the diff before trusting them"
    )
    assert payload["lookups"] == payload["cases"] * len(PROBE_KEYS)
    return payload


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> tuple[pathlib.Path, dict[str, pathlib.Path]]:
    root = tmp_path_factory.mktemp("native-id-corpus")
    return root, build_corpus(root)


def test_the_recording_covers_what_it_claims(recorded, corpus):
    """Every corpus file is in the recording and vice versa, and it is not mostly errors."""
    _, files = corpus
    assert set(recorded["results"]) == set(files), "the corpus and the recording have drifted apart"
    returned = sum(1 for keys in recorded["results"].values() for o in keys.values() if "spectrum" in o)
    assert returned == 171, f"the recording holds {returned} successful lookups, expected 171"
    assert recorded["lookups"] == 370


def test_every_lookup_matches_what_main_returned(recorded, corpus):
    """The whole point: 370 of 370, with no lookup that used to work now failing."""
    root, files = corpus
    identical = different = 0
    lost: list[str] = []
    changed: list[str] = []
    for case, path in sorted(files.items()):
        for key in PROBE_KEYS:
            want = recorded["results"][case][repr(key)]
            got = _outcome(case, path, key, root)
            if got == want:
                identical += 1
                continue
            different += 1
            where = f"{case}[{key!r}]: main {want} -> now {got}"
            changed.append(where)
            if "spectrum" in want:
                lost.append(where)
    assert not lost, "lookups that used to return a spectrum and no longer do:\n" + "\n".join(lost[:10])
    assert not changed, "lookups whose outcome changed:\n" + "\n".join(changed[:10])
    assert identical == recorded["lookups"], f"compared {identical + different}, identical {identical}"


@pytest.mark.parametrize(
    "case",
    [
        # One per class the review measured as changed by the index, all of which main
        # answered with a spectrum. Named so a regression says which class broke.
        "barecr.mgf",
        "barecr.ms2",
        "mixed_eol.mgf",
        "damaged_tail.mgf",
        "damaged_tail.ms2",
        "unterminated.mgf",
        "truncated_gzip.mgf",
    ],
)
def test_the_classes_the_index_cannot_answer(recorded, corpus, case):
    """Each of these files has at least one id the index misses, and it must still resolve."""
    root, files = corpus
    recoverable = [k for k in PROBE_KEYS if "spectrum" in recorded["results"][case][repr(k)]]
    assert recoverable, f"{case} records no successful lookup, so this case proves nothing"
    for key in recoverable:
        assert _outcome(case, files[case], key, root) == recorded["results"][case][repr(key)]


def test_reader_none_still_finds_an_id_less_record(recorded, corpus):
    """``reader[None]`` is outside the ``int | str`` hint, and it is not changed here.

    Main's string path compared ``spec.native_id == key`` for any non-int key, so ``None``
    matched the first record without an id. The recorded expectation is main's.
    """
    root, files = corpus
    for case in ("no_id_block_first.mgf", "no_id_block_first.msp"):
        want = recorded["results"][case]["None"]
        assert "spectrum" in want, f"{case} should have an id-less record for None to find"
        assert _outcome(case, files[case], None, root) == want
