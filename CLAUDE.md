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
│   ├── greedy.py    # isotope cluster finder (optionally JIT'd with numba)
│   └── scored.py    # scored deconvolution entry point (Bhattacharyya scoring)
├── matching.py      # fragment peak matching (match_fragments)
├── scoring.py       # peptide-spectrum match scoring (hyperscore, spectral_angle, …)
├── compress.py      # spectrum matrix compression/decompression
├── noise.py         # noise estimation (MAD, fixed threshold)
├── plot_table.py    # intermediate DataFrame API (build_plot_table, plot_from_table)
└── visualization.py # plotly-based plotting (plot_spectrum, mirror_plot, annotate_spectrum)
```

## Core concepts

**`Spectrum`** — central class. Holds `mz`, `intensity`, and optionally `charge` (int32 array),
`im` (ion mobility), and `score` (per-peak isotopic profile score from deconvolution). Methods
return a new `Spectrum` (or mutate inplace) and are chainable:

```python
spec.filter(min_mz=100).normalize().deconvolute(charge_range=(1, 5)).decharge()
```

**`SpectrumType`** — `CENTROID | PROFILE | DECONVOLUTED`. Guards prevent calling `.decharge()` before `.deconvolute()`.

**`MsnSpectrum`** — extends `Spectrum` with MS metadata: `scan_number`, `ms_level`, `rt`, `precursors`, `collision_energy`, etc.

**`Peak`** — frozen dataclass for a single peak `(mz, intensity, charge, im, score)`.

## Deconvolution pipeline

```python
# 1. identify isotope clusters → monoisotopic m/z + charge state + isotopic profile score
decon = spec.deconvolute(charge_range=(1, 5), tolerance=10, tolerance_type="ppm")
# decon.charge: -1 = singleton/unassigned, >0 = assigned charge state
# decon.score:  0.0 for singletons, Bhattacharyya score (0–1) for clusters

# 2. filter by score quality
filtered = decon.filter(min_score=0.5)

# 3. convert charged peaks to neutral masses (drops singletons)
neutral = filtered.decharge()
```

**How the scored algorithm works** (`decon/scored.py` + `decon/greedy.py`):
- Seeds on the most-intense unused peak, tries every charge in `charge_range`
- For each charge, `_find_isotope_cluster` extends forward by `NEUTRON_MASS / charge` steps (10 peaks max)
- Each candidate cluster is scored by Bhattacharyya coefficient against a theoretical isotope template, penalised for missed detectable peaks (score range 0–1)
- Picks the charge with the highest score; ties broken by cluster size
- Clusters below `min_score` are recorded as singletons (`charge = -1`, `score = 0.0`); their peaks remain available for future seeds
- Singletons (no neighbours found at any charge) also get `charge = -1`

## Charge conventions

| `charge` value | meaning |
|---|---|
| `> 0` | assigned isotope cluster |
| `-1` | singleton / unassigned |
| `0` | after `decharge()` (neutral mass, charge unknown) |

## Key dependencies

- **peptacular** — isotope distribution estimation, `pt.PROTON_MASS`
- **paftacular** — fragment label serialisation (mzPAF format)
- **numpy** — all numeric operations
- **pandas** — plot table DataFrames (`plot_table.py`)
- **plotly** — interactive visualisation
- **tdfpy** *(optional)* — Bruker `.d` file reading; required only for `DReader`. Install with `pip install spxtacular[bruker]`
- **mzmlpy** *(optional)* — mzML reading; required only for `MzmlReader`. Install with `pip install spxtacular[mzml]`
- **numba** *(optional)* — JIT-compiles `_find_isotope_cluster` and `_score_cluster` for ~3–4× speedup; install with `pip install spxtacular[numba]`

`DReader` and `MzmlReader` remain importable from `spxtacular` regardless of
whether their backends are installed; only instantiation raises `ImportError`
when the corresponding optional dep is missing. This lets downstream libraries
(e.g. `pydiode`) depend on `spxtacular` without pulling in the raw-file readers.

## What NOT to do

- Do not call `decharge()` on a non-deconvoluted spectrum — it will raise `ValueError`.
- Do not move isotope scoring logic into `greedy.py` — keep cluster finding and scoring separate.
