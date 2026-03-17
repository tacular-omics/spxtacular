# spxtacular

Mass spectrometry spectrum processing library. Companion to [peptacular](../peptacular).

## Commands

```bash
uv run pytest tests/ -v          # run all tests
uv run ruff check src/ tests/    # lint
uv run ruff format src/ tests/   # format
uv run ty check src/             # type check
```

## Architecture

```
src/spxtacular/
├── core.py          # Spectrum, MsnSpectrum, Peak — all processing lives here
├── reader.py        # DReader (Bruker timsTOF via tdfpy), MzmlReader
├── decon/
│   └── greedy.py    # greedy numpy deconvolution (no numba, no graph)
├── compress.py      # spectrum matrix compression/decompression
├── noise.py         # noise estimation (MAD, fixed threshold)
└── visualization.py # plotly-based plotting (stub)
```

## Core concepts

**`Spectrum`** — central class. Holds `mz`, `intensity`, and optionally `charge` (int32 array) and `im` (ion mobility). Methods return a new `Spectrum` (or mutate inplace) and are chainable:

```python
spec.filter(min_mz=100).normalize().deconvolute(charge_range=(1, 5)).decharge()
```

**`SpectrumType`** — `CENTROID | PROFILE | DECONVOLUTED`. Guards prevent calling `.decharge()` before `.deconvolute()`.

**`MsnSpectrum`** — extends `Spectrum` with MS metadata: `scan_number`, `ms_level`, `rt`, `precursors`, `collision_energy`, etc.

**`Peak`** — frozen dataclass for a single peak `(mz, intensity, charge, im)`.

## Deconvolution pipeline

```python
# 1. identify isotope clusters → monoisotopic m/z + charge state
decon = spec.deconvolute(charge_range=(1, 5), tolerance=10, tolerance_type="ppm")
# decon.charge: -1 = singleton/unassigned, >0 = assigned charge state

# 2. convert charged peaks to neutral masses (drops singletons)
neutral = decon.decharge()
```

**How the greedy algorithm works** (`decon/greedy.py`):
- Seeds on the most-intense unused peak, tries every charge in `charge_range`
- For each charge, extends forward in m/z space by `NEUTRON_MASS / charge` steps (10 peaks max, no skips)
- Picks the charge that yields the longest chain; ties broken by total intensity
- Singletons (no neighbours found at any charge) get `charge = -1`
- Every input peak is consumed exactly once

## Charge conventions

| `charge` value | meaning |
|---|---|
| `> 0` | assigned isotope cluster |
| `-1` | singleton / unassigned |
| `0` | after `decharge()` (neutral mass, charge unknown) |

## Key dependencies

- **peptacular** — `pt.C13_NEUTRON_MASS`, `pt.PROTON_MASS` (editable local install)
- **tdfpy** — Bruker `.d` file reading (editable local install)
- **mzmlpy** — mzML reading (editable local install)
- **numpy** — all numeric operations; no numba, no scipy

## What NOT to do

- Do not add numba to the deconvolution — the greedy implementation is intentionally pure numpy.
- Do not introduce isotopic pattern scoring into `greedy.py` — scoring belongs in a separate layer if added.
- Do not call `decharge()` on a non-deconvoluted spectrum — it will raise `ValueError`.
