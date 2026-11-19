"""Record what ``reader[key]`` does on every corpus file, for the equivalence test.

Run it on a tree whose behaviour you want to pin — the point is to run it on the tree
*before* a change, so the expectations in
``tests/data/peaklist_native_id_behaviour.json`` are that tree's results and not the new
tree's. It prints the commit it ran on, which goes into the file.

    python scripts/record_peaklist_native_id_behaviour.py <out.json>

The generator lives in ``tests/peaklist_native_id_corpus.py`` so both sides build the same
bytes; copy that file alongside when recording on an older checkout that lacks it.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tests"))

from peaklist_native_id_corpus import PROBE_KEYS, build_corpus

from spxtacular import MgfReader, Ms2Reader, MsnSpectrum, MspReader

READERS = {"mgf": MgfReader, "ms2": Ms2Reader, "msp": MspReader}


def reader_for(case: str, path: pathlib.Path):
    """The reader class for a case name, which always ends in the format it holds."""
    fmt = case.rsplit(".", 1)[-1]
    if fmt == "gz":  # "truncated_gzip.mgf.gz" style names
        fmt = case.rsplit(".", 2)[-2]
    return READERS[fmt](path)


def digest(spec: MsnSpectrum) -> str:
    """A hash of the whole serialised spectrum, so a change in any other field still shows up."""
    return hashlib.sha256(json.dumps(spec.to_dict(), sort_keys=True, default=repr).encode()).hexdigest()[:16]


def outcome(case: str, path: pathlib.Path, key: object, root: pathlib.Path) -> dict:
    """What ``reader[key]`` does: which spectrum came back, or the exception it raised.

    A returned spectrum is recorded as the few fields a reader of the fixture can check by
    eye, plus a digest of the complete ``to_dict()``, so the file stays legible and a change
    in any field it does not spell out still fails the comparison.
    """
    try:
        spec = reader_for(case, path)[key]  # type: ignore[index]
    except Exception as exc:
        # Paths differ per run, so the message is recorded with the corpus dir masked.
        return {"error": type(exc).__name__, "message": str(exc).replace(str(root), "<corpus>")}
    return {
        "spectrum": {
            "native_id": spec.native_id,
            "scan_number": spec.scan_number,
            "mz": [float(v) for v in spec.mz],
            "intensity": [float(v) for v in spec.intensity],
            "digest": digest(spec),
        }
    }


def main() -> int:
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "peaklist_native_id_behaviour.json")
    root = pathlib.Path(tempfile.mkdtemp(prefix="nativeid-corpus-"))
    files = build_corpus(root)
    rev = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False, cwd=pathlib.Path(__file__).parent
    ).stdout.strip()
    records = {
        case: {repr(key): outcome(case, path, key, root) for key in PROBE_KEYS} for case, path in sorted(files.items())
    }
    payload = {
        "recorded_from_commit": rev,
        "cases": len(records),
        "lookups": sum(len(v) for v in records.values()),
        "results": records,
    }
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    spectra = sum(1 for v in records.values() for o in v.values() if "spectrum" in o)
    print(
        f"recorded {payload['lookups']} lookups over {payload['cases']} files from {rev}: {spectra} returned a spectrum"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
