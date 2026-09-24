# spxtacular — Claude Code Guide

Canonical guide for AI coding agents working in this repo. `AGENTS.md` and
`.github/copilot-instructions.md` point here. Users of the package (not developers) want
`llms.txt` / `llms-full.txt` instead.

## Project overview

spxtacular is a Python 3.12+ library for mass-spectrum processing behind one chainable
`Spectrum` / `MsnSpectrum` object: filtering, denoising, centroiding, isotope deconvolution,
charge assignment and decharging, fragment matching, PSM scoring, spectrum similarity,
chromatograms, readers/writers for Bruker `.d`, mzML, Thermo `.raw`, MGF/MS2/MSP, and
accessible plotly figures. Users are people writing proteomics, metabolomics, lipidomics,
glycomics or oligonucleotide analysis code.

Place in the tacular-omics graph (tier 2, the top of the core workspace):

- **Upstream (runtime):** `peptacular` (peptide/fragment math, `Fragment` objects) and
  `paftacular`; `tacular` arrives transitively (its `IonType` appears in `FragmentInput`).
- **Upstream (extras):** `tdfpy` (`[bruker]`, `DReader`), `mzmlpy` (`[mzml]`, `MzmlReader`),
  plus external `fisher-py` (`[thermo]`), `spectrl` (`[spectrl]`), `matchms` /
  `spectrum-utils` (`[interop]`), `numba` (`[numba]`).
- **Downstream:** `msbit`, `pepbit` (not in the workspace). Note breaking changes for them.

Other runtime deps: numpy, pandas, plotly. Version sources and pins are in `pyproject.toml`.

## Commands

Every recipe runs `uv run --locked ...`. All verified on 2026-09-23 (1128 passed, 13 skipped;
the skips are Thermo tests that need a .NET runtime).

```bash
just --list          # all recipes
just test            # pytest tests  (--timeout=15, filterwarnings=error)
just test-cov        # pytest with coverage.xml + junit.xml (CI)
just lint            # ruff check src tests .github/scripts benchmarks
just fmt-check       # ruff format --check (what CI runs)
just format          # ruff isort fix + ruff format  (REWRITES files)
just check           # ty check src tests  (type check ONLY, not a full check)
just docs            # mkdocs serve on localhost:8003
just docs-build      # mkdocs build --strict
just benchmark       # benchmarks/deconvolution.py accuracy + timing
just check-version   # __init__.py / CITATION.cff / CHANGELOG version agreement
just                 # default: lint, format, check, test  (format rewrites files)
```

Pre-PR set (from CONTRIBUTING.md): `just lint && just fmt-check && just check && just test-cov && just docs-build`.
`just paper*` recipes build the JOSS draft and need Docker, LibreOffice and poppler.

## Architecture

```
src/spxtacular/
  __init__.py        # public re-exports + __version__ (version source for hatch)
  core.py            # Peak, Precursor, Spectrum, MsnSpectrum, SpectrumType; every transform
                     # (filter, normalize, denoise, centroid, merge, deconvolute, decharge, ...)
  enums.py           # PeakSelection, ActivationType, IMType, Analyzer + *Like aliases; unit/polarity checkers
  decon/greedy.py    # isotope-cluster finding helpers
  decon/scored.py    # greedy deconvolution with isotope-profile scoring (numba JIT if installed)
  isotopes.py        # IsotopeModel, BRAIN isotope distributions, built-in average-composition models
  ionization.py      # IonizationModel (H+, H-, Na+, NH4+, custom), DeconvolutionProvenance
  noise.py           # noise-floor estimators behind denoise()/centroid(min_intensity="noise")
  matching.py        # match_fragments, MatchedFragment, FragmentInput
  scoring.py         # score(): hyperscore, probability, spectral angle, coverage metrics
  similarity.py      # cosine, modified_cosine, entropy_similarity
  chromatogram.py    # Chromatogram, extract_chromatogram (TIC/BPC), extract_xic
  reporter.py        # TMT/TMTpro/iTRAQ reporter ions (tacular ISOBARIC_TAG_LOOKUP), impurity correction (NNLS)
  reader.py          # Reader (format auto-detect), DReader (tdfpy), MzmlReader (mzmlpy), CentroidConfig, AcquisitionType
  thermo.py          # ThermoReader (fisher-py, lazy)
  peaklist.py        # MGF / MS2 / MSP readers and writers, standard library + numpy only
  mzml.py            # write_indexed_mzml_gzip (thin mzmlpy wrapper)
  interop.py         # matchms / spectrum_utils adapters (lazy imports)
  spectrl_bridge.py  # spectrl tokens / URLs (optional)
  usi.py             # fetch_usi / Spectrum.from_usi via PROXI
  serialization.py   # versioned to_dict/to_json helpers; schemas/*.schema.json + get_json_schema
  utils.py           # da_to_ppm, ppm_to_da
  theme.py           # the ONLY source of plot colours, templates, light/dark mode
  plot_table.py      # plot tables (DataFrame between data and figure), plot_from_table, table_view
  visualization.py   # plot_spectrum, annotate_spectrum, mirror_plot, facet_plot, ... save_figure
```

Data flow: a reader yields `MsnSpectrum` objects -> chained transforms return new spectra
(`filter` -> `centroid`/`denoise` -> `deconvolute` -> `decharge`) -> `match_fragments`/`score`
against peptacular fragments, or `similarity` against another spectrum -> `plot_table` ->
figure. Other dirs: `tests/` (+ `tests/data` Bruker `.d`, mzML, `.raw` fixtures,
`tests/reference` analytical fixtures), `benchmarks/`, `docs/` (mkdocs), `paper/` (JOSS),
`.github/scripts/` (wheel, minimal-install and no-skip checks used by CI).

## Public API

All names below are in `spxtacular.__all__` and import from the top level (checked by
importing each one).

- **Data model:** `Spectrum`, `MsnSpectrum`, `Peak`, `Precursor`, `SpectrumType`.
- **Enums / aliases:** `PeakSelection`, `PeakSelectionLike`, `ActivationType`,
  `ActivationTypeLike`, `IMType`, `IMTypeLike`, `Analyzer`, `AnalyzerLike`. Tolerance units and
  polarity are plain lowercase strings typed by `tacular.types.ToleranceUnit` (`"da"`/`"ppm"`)
  and `tacular.types.Polarity` (`"positive"`/`"negative"`); spxtacular does not re-export them.
- **Isotope models:** `IsotopeModel`, `IsotopeModelLike`, `IsotopeModelType`, `ISOTOPE_MODELS`,
  `PEPTIDE_/GLYCAN_/LIPID_/DNA_/RNA_ISOTOPE_MODEL`, `brain_isotopic_distribution`,
  `resolve_isotope_model`.
- **Ionization:** `IonizationModel`, `IonizationModelLike`, `DeconvolutionProvenance`,
  `PROTONATED`, `DEPROTONATED`, `SODIATED`, `AMMONIATED`, `IONIZATION_MODELS`,
  `resolve_ionization_model`.
- **Matching / scoring / similarity:** `match_fragments`, `score`, `cosine`, `modified_cosine`,
  `entropy_similarity`.
- **Chromatograms:** `Chromatogram`, `extract_chromatogram`, `extract_xic`.
- **Reporter ions:** `ReporterIons`, `extract_reporter_ions`, `reporter_ion_table`,
  `isotope_correction_matrix`, `correct_isotope_impurities`.
- **I/O:** `Reader`, `DReader`, `MzmlReader`, `ThermoReader`, `CentroidConfig`,
  `AcquisitionType`, `MgfReader`, `Ms2Reader`, `MspReader`, `write_mgf`, `write_ms2`,
  `write_msp`, `write_indexed_mzml_gzip`, `fetch_usi`, `spectrum_from_proxi_response`.
- **Interop / sharing:** `to_matchms`, `from_matchms`, `to_spectrum_utils`,
  `from_spectrum_utils`, `to_spectrl_token`, `from_spectrl_token`, `to_spectrl_url`,
  `from_spectrl_url`, `to_inline_spectrum`, `get_json_schema`.
- **Plotting:** `plot_spectrum`, `annotate_spectrum`, `mirror_plot`, `facet_plot`,
  `mass_error_plot`, `sequence_coverage_plot`, `profile_centroid_plot`, `plot_chromatogram`,
  `plot_xic`, `save_figure`, `build_plot_table`, `build_annot_plot_table`, `plot_from_table`,
  `table_view`, `theme` (module).
- **Utils:** `da_to_ppm`, `ppm_to_da`.

`__all__` is deliberately unsorted (ruff `RUF022` suppressed); add new names anywhere, but
add them to both the import block and `__all__`.

## Conventions

- **Typing:** precise hints on production code, reasonable hints in tests; Python 3.12 syntax
  and built-in generics. `ty check src tests` must pass.
- **Style:** ruff, line length 120, double quotes. Prefer frozen slotted dataclasses, small
  readable helpers over clever one-liners, immutable transformations.
- **Docstrings:** NumPy style (`Parameters` / `Returns` / `Raises` with `----------`
  underlines; there are no Google `Args:` blocks in `src/`), Sphinx-style cross references
  (`:class:`, `:meth:`). Keep them brief; explain non-obvious reasons; include `Raises` when useful.
  Do not restate what type hints say. `docstring-code-format` is off on purpose (it would
  reformat the `>>>` examples in `core.py` / `reader.py`).
- **Errors:** coerce and validate enum-like inputs and raise `SpxtacularError` (a `ValueError`
  subclass, from `spxtacular.errors`) rather than silently picking a fallback. No placeholder TODOs; raise `NotImplementedError` with a reason.
- **Warnings:** pytest runs with `filterwarnings = ["error"]`, so any `UserWarning` the code
  emits on a test path fails the test unless the test expects it (`pytest.warns`).
- **Tests:** `tests/test_<area>.py`; `--timeout=15` per test. Keep tests focused.
- **Changelog:** a concise `CHANGELOG.md` bullet under `[Unreleased]` only for user-visible
  changes (the file's heading is "History", but the file is `CHANGELOG.md`).
- **ruff per-file-ignores** in `pyproject.toml` are suppressions of real findings: fix the code
  and delete the entry rather than adding more.

## Gotchas

Load-bearing rules (kept from the previous guide, all still true of the code):

- Keep all parallel peak arrays (`mz`, `intensity`, `charge`, `im`, `iso_score`) the same
  length and in the same permutation.
- Do not assume m/z is sorted; timsTOF frames are ordered by ion-mobility scan.
- `charge > 0` is assigned, `-1` is singleton/unassigned, `0` is neutral mass after
  `decharge()`. Never test charge or `iso_score` by truthiness.
- Call `deconvolute()` before `decharge()`; `SpectrumType` guards the transition
  (`decharge()` on a spectrum with no charge array raises `ValueError`; on an already-neutral
  spectrum it returns it unchanged silently; on an all-singleton spectrum it warns and returns
  it unchanged).
- Keep cluster finding in `decon/greedy.py` and isotope scoring in `decon/scored.py`.
- Optional backends: reader classes must stay importable without their extras and fail only
  when instantiated. `tdfpy`, `mzmlpy`, `spectrl` and `numba` are imported at package import
  time when installed (guarded by `try/except`); `fisher_py`, `matchms` and `spectrum_utils`
  must stay lazy and must not load at `import spxtacular`.
- Keep `peaklist.py` standard-library-only apart from numpy.
- `theme.py` is the only source of plot colours: ion type is categorical, charge is ordinal,
  isotope score and mobility are sequential.
- Render profile spectra as traces and decimate with min/max buckets, never stride sampling.
- Keep labels capped and collision-avoided; full data stays in hover/table output.
- Non-inplace methods must not share mutable arrays (or precursor lists) with their input.

Other traps:

- `just check` is only the ty type check. Use the pre-PR set above for a full check.
- Reader `.ms1` / `.ms2` views are iterable and indexable but are **not** iterators and have
  no `len()`: use `next(iter(reader.ms2))`.
- `Spectrum.combine` raises on mixing neutral masses with m/z, or charged inputs with different
  deconvolution provenance; it always returns a base `Spectrum`.
- Hyperscore changed in 0.7.0 (dot product + factorial terms); scores from 0.6.0 are not
  comparable. Do not claim numerical equivalence with X!Tandem/OpenMS/Comet/MSFragger.
- `centroid()` with no `min_intensity` turns every local maximum into a peak; real data needs a
  floor (`"noise"` or a number).
- Thermo tests skip unless a .NET 8 runtime is installed; CI has a job where they are required.
- spectrl is pinned `>=3.0,<4` (token format `spectrl.v3`); each spectrl major changes the token
  format, so the major cap is deliberate. spectrl 3 rejects boolean parameter values, so the
  bridge writes flags as int 0/1.

## Releasing

Only the tacular-omics overseer bumps versions or publishes. See `RELEASING.md` and
`just --list` (`set-version`, `sync-version`, `check-version`). The version source is
`__version__` in `src/spxtacular/__init__.py` (`[tool.hatch.version]`), mirrored in
`CITATION.cff` and the `CHANGELOG.md` heading. Releases also go to Zenodo and feed the JOSS
paper in `paper/`.

## Workspace note

This repo is also developed inside the tacular-omics uv workspace
(`~/Repos/tacular-omics/packages/spxtacular`); there `uv run` and the `just` recipes use the
shared `.venv` and the root `uv.lock`, so siblings resolve to local checkouts. See the
workspace CLAUDE.md.
