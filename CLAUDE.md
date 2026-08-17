# spxtacular coding guide

Python 3.12+ mass-spectrometry spectrum processing library; `peptacular` supplies peptide and fragment math.

## Commands

Use `just` recipes when available and `uv run` otherwise:

```bash
just --list
just test
just lint
just format
just check
```

Equivalent direct commands are `uv run pytest tests/ -v`, `uv run ruff check src/ tests/`, `uv run ruff format src/ tests/`, and `uv run ty check src tests`.

## Source map

- `core.py`: `Spectrum`, `MsnSpectrum`, `Peak`, `SpectrumType`, and spectrum transforms.
- `reader.py`, `thermo.py`, `peaklist.py`: Bruker/mzML, Thermo RAW, and MGF/MS2/MSP I/O.
- `decon/`: greedy isotope-cluster finding and separate scored selection.
- `matching.py`, `scoring.py`, `similarity.py`: fragment matching, PSM scoring, and spectrum similarity.
- `interop.py`, `spectrl_bridge.py`, `usi.py`: ecosystem conversion, serialization, and remote identifiers.
- `chromatogram.py`, `noise.py`, `isotopes.py`, `ionization.py`: domain algorithms.
- `theme.py`, `plot_table.py`, `visualization.py`: plot styling, intermediate tables, and figures.

## Load-bearing rules

- Preserve all parallel peak arrays (`mz`, `intensity`, `charge`, `im`, `iso_score`) with the same length and permutation.
- Do not assume m/z is sorted; timsTOF frames are ordered by ion-mobility scan.
- `charge > 0` means assigned, `-1` means singleton/unassigned, and `0` means neutralized after `decharge()`.
- Call `deconvolute()` before `decharge()`; `SpectrumType` guards this state transition.
- Keep cluster finding in `decon/greedy.py` and isotope scoring in `decon/scored.py`.
- Keep optional backends lazy: reader classes must remain importable without their extras, and `fisher_py` must not load at package import time.
- Keep `peaklist.py` standard-library-only apart from numpy.
- Use `theme.py` as the only source of plot colors; ion type is categorical, charge is ordinal, and isotope score/mobility are sequential.
- Render profile spectra as traces and decimate with min/max buckets, never stride sampling.
- Keep labels capped and collision-avoided; full data remains available in hover/table output.
- Prefer immutable transformations; non-inplace methods must not share mutable arrays with their input.
- Coerce and validate enum-like inputs rather than silently selecting a fallback.

## Style

- Add precise type hints to production code and reasonable hints to tests.
- Prefer Python 3.12 syntax and built-in generic types.
- Choose readable helpers over clever expressions.
- Keep docstrings brief, document non-obvious reasons, and include `Raises` when useful.
- Add a concise `HISTORY.md` bullet only for user-visible changes.

Consult the relevant source, tests, and `docs/` page for API details instead of expanding this file.
