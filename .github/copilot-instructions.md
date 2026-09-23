# Copilot instructions

The canonical guide is [`CLAUDE.md`](../CLAUDE.md) at the repo root: commands, module map,
public API, conventions and gotchas. Read it before non-trivial changes.

Hardest rules:

1. Use the `just` recipes (`just test`, `just lint`, `just fmt-check`, `just check` = ty only);
   otherwise `uv run <tool>`. Never `pip install` or call bare `python`/`pytest`.
2. Keep `mz`, `intensity`, `charge`, `im`, `iso_score` equal-length and co-permuted; never assume
   m/z is sorted; `charge` > 0 assigned, -1 singleton, 0 neutral (never test by truthiness).
3. `deconvolute()` before `decharge()`; cluster finding stays in `decon/greedy.py`, scoring in
   `decon/scored.py`; non-inplace methods must not share mutable arrays with their input.
4. Optional backends stay optional: readers import without their extras, `fisher_py`/`matchms`/
   `spectrum_utils` load lazily, `peaklist.py` uses only the standard library and numpy.
5. All plot colours come from `theme.py`; validate enum-like inputs instead of silently falling
   back; pytest treats warnings as errors.
