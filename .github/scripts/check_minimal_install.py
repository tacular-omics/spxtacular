"""Assert spxtacular works when installed with no optional extras.

CLAUDE.md promises that ``DReader`` and ``MzmlReader`` stay importable from
``spxtacular`` whether or not their backends are installed, and that only
*instantiation* raises ``ImportError``. Downstream libraries (e.g. pydiode)
rely on this to depend on spxtacular without pulling in the raw-file readers.

Run against a venv containing *only* the base install: ``uv pip install .``
"""

from __future__ import annotations

import sys

OPTIONAL_BACKENDS = ("tdfpy", "mzmlpy", "numba", "spectrl", "matchms", "spectrum_utils", "fisher_py")


def main() -> int:
    from importlib.resources import files

    import numpy as np

    import spxtacular
    from spxtacular import (
        DReader,
        MsnSpectrum,
        MzmlReader,
        Precursor,
        Spectrum,
        ThermoReader,
        to_matchms,
        to_spectrum_utils,
    )
    from spxtacular.serialization import get_json_schema

    failures: list[str] = []

    # The job is only meaningful if the optional backends really are absent.
    for mod in OPTIONAL_BACKENDS:
        try:
            __import__(mod)
        except ImportError:
            print(f"ok: {mod} is absent, as expected for a minimal install")
        else:
            failures.append(f"{mod} must NOT be installed in the minimal job")

    for cls in (DReader, MzmlReader, ThermoReader):
        name = cls.__name__
        try:
            cls("nonexistent.path")
        except ImportError as exc:
            print(f"ok: {name}(...) raised ImportError: {exc}")
        except Exception as exc:
            failures.append(f"{name} raised {type(exc).__name__} instead of ImportError: {exc}")
        else:
            failures.append(f"{name} should raise ImportError without its backend")

    optional_calls = (
        (
            "to_matchms",
            lambda: to_matchms(Spectrum(mz=np.array([100.0]), intensity=np.array([1.0]))),
        ),
        (
            "to_spectrum_utils",
            lambda: to_spectrum_utils(
                MsnSpectrum(
                    mz=np.array([100.0]),
                    intensity=np.array([1.0]),
                    native_id="scan=1",
                    precursors=[Precursor(mz=500.0, intensity=0.0, charge=2, is_monoisotopic=None)],
                )
            ),
        ),
    )
    for name, call in optional_calls:
        try:
            call()
        except ImportError as exc:
            print(f"ok: {name}(...) raised ImportError: {exc}")
        except Exception as exc:
            failures.append(f"{name} raised {type(exc).__name__} instead of ImportError: {exc}")
        else:
            failures.append(f"{name} should raise ImportError without its backend")

    for kind in ("spectrum", "chromatogram"):
        assert get_json_schema(kind)["$schema"].endswith("2020-12/schema")
    assert files("spxtacular").joinpath("py.typed").is_file()
    spectrum = Spectrum(mz=np.array([100.0]), intensity=np.array([1.0]))
    assert Spectrum.from_json(spectrum.to_json()) == spectrum

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    print(f"minimal install OK: spxtacular {spxtacular.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
