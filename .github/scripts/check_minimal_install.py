"""Assert spxtacular works when installed with no optional extras.

CLAUDE.md promises that ``DReader`` and ``MzmlReader`` stay importable from
``spxtacular`` whether or not their backends are installed, and that only
*instantiation* raises ``ImportError``. Downstream libraries (e.g. pydiode)
rely on this to depend on spxtacular without pulling in the raw-file readers.

Run against a venv containing *only* the base install: ``uv pip install .``
"""

from __future__ import annotations

import sys

OPTIONAL_BACKENDS = ("tdfpy", "mzmlpy", "numba", "spectrl")


def main() -> int:
    import spxtacular
    from spxtacular import DReader, MzmlReader

    failures: list[str] = []

    # The job is only meaningful if the optional backends really are absent.
    for mod in OPTIONAL_BACKENDS:
        try:
            __import__(mod)
        except ImportError:
            print(f"ok: {mod} is absent, as expected for a minimal install")
        else:
            failures.append(f"{mod} must NOT be installed in the minimal job")

    for cls in (DReader, MzmlReader):
        name = cls.__name__
        try:
            cls("nonexistent.path")
        except ImportError as exc:
            print(f"ok: {name}(...) raised ImportError: {exc}")
        except Exception as exc:
            failures.append(f"{name} raised {type(exc).__name__} instead of ImportError: {exc}")
        else:
            failures.append(f"{name} should raise ImportError without its backend")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    print(f"minimal install OK: spxtacular {spxtacular.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
