"""Deterministic peak-list corpus for the ``reader[native_id]`` equivalence test.

This module is imported by :mod:`tests.test_peaklist_native_id_equivalence` **and** by
``scripts/record_peaklist_native_id_behaviour.py``, which is run against an older tree to
record what that tree returned. It therefore uses nothing but the public writers and plain
file writes, so the same corpus is produced on either tree.

Every file is small and written byte-exactly (``newline=""``), because the line endings are
the point of several cases.
"""

from __future__ import annotations

import gzip
import pathlib
import re

# Three MGF records with distinct peaks, so a wrong record is obvious in the comparison.
MGF = (
    "BEGIN IONS\nTITLE=first\nPEPMASS=400.5 1000.0\nCHARGE=2+\nSCANS=11\nRTINSECONDS=60.0\n"
    "100.0 10.0 1+\n110.0 11.0\nEND IONS\n"
    "BEGIN IONS\nTITLE=second\nPEPMASS=500.25 2000.0\nCHARGE=3+\nSCANS=12\nRTINSECONDS=70.5\n"
    "200.0 20.0 2+\n210.0 21.0\nEND IONS\n"
    "BEGIN IONS\nTITLE=third\nPEPMASS=600.125\nCHARGE=2-\nSCANS=13\n"
    "300.0 30.0\n310.0 31.0\nEND IONS\n"
)
MS2 = (
    "H\tCreationDate\t2026-11-18\n"
    "S\t11\t11\t400.5\nI\tRTime\t1.0\nI\tNativeID\tfirst\nZ\t2\t800.0\n100.0 10.0\n110.0 11.0\n"
    "S\t12\t12\t500.25\nI\tRTime\t1.2\nI\tNativeID\tsecond\nZ\t3\t1499.0\n200.0 20.0\n210.0 21.0\n"
    "S\t13\t13\t600.125\nI\tRTime\t1.4\nI\tNativeID\tthird\nZ\t2\t1199.0\n300.0 30.0\n310.0 31.0\n"
)
MSP = (
    "Name: first\nMW: 800.0\nComment: Parent=400.5\nNum Peaks: 2\n100.0 10.0\n110.0 11.0\n\n"
    "Name: second\nMW: 1499.0\nComment: Parent=500.25\nNum Peaks: 2\n200.0 20.0\n210.0 21.0\n\n"
    "Name: third\nMW: 1199.0\nComment: Parent=600.125\nNum Peaks: 2\n300.0 30.0\n310.0 31.0\n\n"
)

#: Ids present in the three-record bodies, plus keys that are deliberately absent or odd.
PRESENT_IDS = ("first", "second", "third")
PROBE_KEYS: tuple[object, ...] = (*PRESENT_IDS, "missing", "None", "", None, 0, 1, [1])

# A record with no id: no TITLE in MGF, no Name in MSP, no `I NativeID` in MS2.
MGF_NO_TITLE = "BEGIN IONS\nPEPMASS=350.0\nSCANS=9\n50.0 5.0\nEND IONS\n"
MS2_NO_ID = "S\t9\t9\t350.0\nZ\t2\t699.0\n50.0 5.0\n"
MSP_NO_NAME = "MW: 699.0\nComment: Parent=350.0\nNum Peaks: 1\n50.0 5.0\n\n"

BODIES = {"mgf": MGF, "ms2": MS2, "msp": MSP}
NO_ID_BLOCKS = {"mgf": MGF_NO_TITLE, "ms2": MS2_NO_ID, "msp": MSP_NO_NAME}
# What damage looks like in each format: an unparsable number in a later record.
DAMAGED_TAIL = {
    "mgf": "BEGIN IONS\nTITLE=fourth\nPEPMASS=abc\n400.0 40.0\nEND IONS\n",
    "ms2": "S\t14\t14\tabc\nZ\t2\t1399.0\n400.0 40.0\n",
    "msp": "Name: fourth\nNum Peaks: 2\n400.0 40.0\n\n",  # declares 2 peaks, has 1
}


def mask_path(message: str, path: pathlib.Path) -> str:
    """``message`` with any spelling of ``path`` replaced by ``<file>``.

    Error messages name the file, and the corpus lives in a temporary directory, so the
    message cannot be compared as it stands. Masking only ``str(path)`` is not enough: the
    separator survives it (``<corpus>/x.mgf`` on POSIX, ``<corpus>\\x.mgf`` on Windows) and a
    path can reach the message in another spelling than the one used to build it. So every
    run of non-space characters ending in the file's name goes, separator and all. Measured:
    without this, the equivalence test failed on windows-latest with the full
    ``C:\\Users\\runneradmin\\...`` path in the message while the recording held
    ``<corpus>/...``.

    Both the recorder and the test import this, so the two cannot drift apart.
    """
    return re.sub(r"[^\s'\"]*" + re.escape(path.name), "<file>", message)


def _write(path: pathlib.Path, text: str) -> pathlib.Path:
    """Write ``text`` with no newline translation, so the endings in it survive."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return path


def build_corpus(directory: pathlib.Path) -> dict[str, pathlib.Path]:
    """Write the corpus into ``directory`` and return ``{case name: path}``.

    Case names are stable: they are the keys of the recorded expectations.
    """
    directory.mkdir(parents=True, exist_ok=True)
    files: dict[str, pathlib.Path] = {}
    for fmt, body in BODIES.items():
        files[f"plain.{fmt}"] = _write(directory / f"plain.{fmt}", body)
        files[f"crlf.{fmt}"] = _write(directory / f"crlf.{fmt}", body.replace("\n", "\r\n"))
        files[f"barecr.{fmt}"] = _write(directory / f"barecr.{fmt}", body.replace("\n", "\r"))
        # LF for the first record, bare CR for the rest: the id asked for is reachable by
        # streaming but the file as a whole is not what the index assumes.
        head, _, tail = body.partition("\n" + body.split("\n")[0])
        files[f"mixed_eol.{fmt}"] = _write(
            directory / f"mixed_eol.{fmt}", head + "\n" + body.split("\n")[0] + tail.replace("\n", "\r")
        )
        files[f"no_final_newline.{fmt}"] = _write(directory / f"no_final_newline.{fmt}", body.rstrip("\n"))
        files[f"damaged_tail.{fmt}"] = _write(directory / f"damaged_tail.{fmt}", body + DAMAGED_TAIL[fmt])
        files[f"no_id_block_first.{fmt}"] = _write(directory / f"no_id_block_first.{fmt}", NO_ID_BLOCKS[fmt] + body)
        files[f"no_id_block_last.{fmt}"] = _write(directory / f"no_id_block_last.{fmt}", body + NO_ID_BLOCKS[fmt])
        files[f"duplicate_ids.{fmt}"] = _write(directory / f"duplicate_ids.{fmt}", body + body)
        files[f"empty.{fmt}"] = _write(directory / f"empty.{fmt}", "")
        gz = directory / f"plain.{fmt}.gz"
        gz.write_bytes(gzip.compress(body.encode("utf-8")))
        files[f"gzip.{fmt}"] = gz
        # Truncated inside the compressed stream, after the first record's bytes.
        trunc = directory / f"truncated.{fmt}.gz"
        whole = gzip.compress(body.encode("utf-8"))
        trunc.write_bytes(whole[: max(len(whole) - 12, 20)])
        files[f"truncated_gzip.{fmt}"] = trunc
    # MGF only: a file that ends in the middle of a block, after a complete record.
    files["unterminated.mgf"] = _write(
        directory / "unterminated.mgf", MGF + "BEGIN IONS\nTITLE=fourth\nPEPMASS=700.0\n"
    )
    return files
