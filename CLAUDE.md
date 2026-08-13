# spxtacular

Mass spectrometry spectrum processing library. Companion to [peptacular](../peptacular).

## Commands

```bash
uv run pytest tests/ -v          # run all tests
uv run ruff check src/ tests/    # lint
uv run ruff format src/ tests/   # format
uv run ty check src tests        # type check (CI checks tests too — must stay clean)
```

## Architecture

```
src/spxtacular/
├── core.py          # Spectrum, MsnSpectrum, Peak, SpectrumType — all processing lives here
├── enums.py         # StrEnums: ToleranceType, PeakSelection, Polarity, ActivationType, IMType, Analyzer
├── reader.py        # Reader (auto-detect), DReader (Bruker timsTOF via tdfpy), MzmlReader, CentroidConfig
├── thermo.py        # ThermoReader (Thermo .raw via fisher-py; lazy import — fisher_py boots .NET at import time)
├── peaklist.py      # MGF / MS2 read + write (MgfReader, Ms2Reader, write_mgf, write_ms2) — pure stdlib
├── usi.py           # fetch_usi — USI / PROXI spectrum fetching
├── utils.py         # da_to_ppm / ppm_to_da
├── decon/
│   ├── greedy.py    # isotope cluster finder (optionally JIT'd with numba)
│   └── scored.py    # scored deconvolution entry point (Bhattacharyya scoring)
├── matching.py      # fragment peak matching (match_fragments)
├── scoring.py       # peptide-spectrum match scoring (hyperscore, spectral_angle, …)
├── similarity.py    # spectrum-to-spectrum similarity (cosine, modified_cosine, entropy)
├── spectrl_bridge.py # spectrl token / URL serialisation bridge (optional [spectrl] extra)
├── chromatogram.py  # run-level extraction (extract_chromatogram, extract_xic)
├── noise.py         # noise estimation (MAD, fixed threshold)
├── theme.py         # plot palettes + plotly template (single source of truth for colour)
├── plot_table.py    # intermediate DataFrame API (build_plot_table, plot_from_table, table_view)
└── visualization.py # plotly plotting (plot_spectrum, mirror_plot, annotate_spectrum,
                     #   sequence_coverage_plot, mass_error_plot, facet_plot, save_figure)
```

## Core concepts

**`Spectrum`** — central class. Holds `mz`, `intensity`, and optionally `charge` (int32 array),
`im` (ion mobility), and `iso_score` (per-peak isotopic profile score from deconvolution). Methods
return a new `Spectrum` (or mutate inplace) and are chainable:

```python
spec.filter(min_mz=100).normalize().deconvolute(charge_range=(1, 5)).decharge()
```

**`SpectrumType`** — `CENTROID | PROFILE | DECONVOLUTED`. Guards prevent calling `.decharge()` before `.deconvolute()`.

**`MsnSpectrum`** — extends `Spectrum` with MS metadata: `scan_number`, `ms_level`, `rt`, `precursors`, `collision_energy`, etc.

**`Peak`** — frozen dataclass for a single peak `(mz, intensity, charge, im, iso_score)`.

**Metadata enums** (`enums.py`) — `Polarity`, `ActivationType`, `IMType`, `Analyzer` are `StrEnum`s
typing the `MsnSpectrum.polarity`/`activation_type`/`im_type`/`analyzer` fields as `Enum | str`
(open vocabulary — raw PSI-MS accessions and vendor strings still pass through). `ActivationType`
and `Analyzer` are the single source of truth for the PSI-MS accession maps in `spectrl_bridge.py`.

## Deconvolution pipeline

```python
# 1. identify isotope clusters → monoisotopic m/z + charge state + isotopic profile score
decon = spec.deconvolute(charge_range=(1, 5), tolerance=10, tolerance_type="ppm")
# decon.charge: -1 = singleton/unassigned, >0 = assigned charge state
# decon.iso_score:  0.0 for singletons, Bhattacharyya score (0–1) for clusters

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
- Clusters below `min_score` are recorded as singletons (`charge = -1`, `iso_score = 0.0`); their peaks remain available for future seeds
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
- **fisher-py** *(optional)* — Thermo `.raw` reading; required only for `ThermoReader`. Install with `pip install spxtacular[thermo]`. Needs a .NET runtime on the machine; `import fisher_py` boots it and raises `RuntimeError` without one, so `thermo.py` imports it lazily (never at `import spxtacular` time) — keep it that way
- **numba** *(optional)* — JIT-compiles `_find_isotope_cluster` and `_score_cluster` for ~3–4× speedup; install with `pip install spxtacular[numba]`
- **spectrl** *(optional)* — token / URL spectrum serialisation used by `spectrl_bridge.py`; required only for `to_spectrl_token`/`to_spectrl_url` and their inverses. Install with `pip install spxtacular[spectrl]`

`DReader`, `MzmlReader`, and `ThermoReader` remain importable from `spxtacular`
regardless of whether their backends are installed; only instantiation raises
`ImportError` when the corresponding optional dep is missing. This lets
downstream libraries (e.g. `pydiode`) depend on `spxtacular` without pulling in
the raw-file readers.

`peaklist.py` (MGF / MS2, read and write) has **no** optional dependency — it is
pure standard library plus numpy, so those formats are always available. Keep it
that way: do not reach for a parsing library there.

## Plot colour

`theme.py` is the single source of truth — never hardcode a hex value in `plot_table.py` or
`visualization.py`. Colour is assigned by the **job** it does:

| Job | Encoding |
|---|---|
| ion type | nominal categorical following the proteomics convention: b blue, y red, a green, c teal, x purple, z orange; anything else folds to neutral |
| charge state | **ordinal** — one hue, light→dark; clamps past the ramp; `charge <= 0` is neutral |
| `iso_score`, ion mobility | sequential — one hue, light→dark |
| unmatched peaks | recessive grey, thinner and dimmer |

The palettes were validated for colour-vision deficiency (protanopia/deuteranopia) against both the
light and dark surfaces. Both modes are explicit sets of steps; dark is not an inversion of light.

## What NOT to do

- Do not call `decharge()` on a non-deconvoluted spectrum — it will raise `ValueError`.
- Do not move isotope scoring logic into `greedy.py` — keep cluster finding and scoring separate.
- Do not colour charge states with a categorical cycle — charge is ordinal, and a cycle repeats
  (the old 10-colour version made `z=1` and `z=11` identical).
- Do not label every annotated peak — labels are capped and collision-avoided on purpose; a
  deconvoluted spectrum with a label per peak is an unreadable smear.
- Do not add a hex value straight into a plot function; add it to `theme.py` and validate it.
- Do not assume `spectrum.mz` is sorted — a timsTOF frame is ordered by ion-mobility scan,
  so roughly half its m/z steps descend. Sort a working copy and map indices back.
- Do not draw profile spectra as sticks — `plot_spectrum` renders `SpectrumType.PROFILE` as a
  continuous trace. And never thin a profile by taking every Nth sample; that deletes peaks.
  Use `_decimate_profile` (min/max per bucket).
