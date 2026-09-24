# History

User-visible changes only; implementation details belong in commits and pull requests.

## [Unreleased]

## [0.9.0] (unreleased)

Breaking release. Every rename and removal, old -> new, is in the migration guide
(`docs/migration.md`).

### Changed

- Requires tacular 2, peptacular 5, paftacular 2, and (extras) tdfpy 5 and mzmlpy 0.10. Build fragments with `ion_types=("b", "y")`: peptacular 5 reads `"by"` as one ion type.
- `Precursor` is a standalone keyword-only dataclass (no longer a `Peak`): `mz` -> `precursor_mz`, plus a new `im_type` naming the kind of mobility in `im`. `intensity` defaults to 0.0 and `is_monoisotopic` to `None`.
- `MsnSpectrum.isolation_im_range` -> `isolation_ook0_range`, always `(low, high)`. 0.8 gave Bruker windows as `(high, low)`; 0.8 JSON, npz and spectrl tokens are read back sorted.
- Parameters with a default are keyword-only across the public API (`spec.normalize(method="tic")`, `match_fragments(spec, frags, tolerance=10, tolerance_type="ppm")`), and so is everything after `intensity` in `Spectrum`, `Peak`, `Chromatogram` and `decon.deconvolve_spectrum`, plus the optional fields of `IsotopeModel`, `IonizationModel` and `DeconvolutionProvenance`.
- `decon.deconvolve_spectrum(is_ppm=)` -> `tolerance_type=`; `has_peak`/`get_peak`/`get_peaks(im_tol=)` -> `im_tolerance=`; `get_peak(collision=)` -> `peak_selection=`.
- Removed the deprecated `show_charges=` of `Spectrum.plot`, `Spectrum.plot_table` and `plot_spectrum`; use `color=`.
- Invalid input raises `SpxtacularError`, a `ValueError` subclass, so existing `except ValueError` still works. So do enum coercion, bad JSON payloads (was `TypeError`), corrupt `.npz` files and malformed mzML (was mzmlpy's `MzmlParseError`). Reader lookups before `open()` raise it instead of `RuntimeError`.
- `im_type` and `polarity` are validated on construction; `activation_type` and `analyzer` turn member names (any case) and PSI-MS accessions (`"MS:1000484"` -> `Analyzer.ORBITRAP`) into enum members and keep other vendor strings. Enum coercion is case-insensitive.
- `normalize` on a normalized spectrum rescales it; `denoise`, `centroid`, `deconvolute` and `decharge` on a spectrum already in that state return it unchanged without a warning.
- Spectrum JSON is schema version 2 (`precursor_mz`, `im_type`, `isolation_ook0_range`; `schemas/spectrum-v2.schema.json`). Version-1 files still load.
- `MzmlReader(extract_dir=)`, `Reader(mzml_extract_dir=)` and `gzip_mode="extract"` are removed with mzmlpy 0.10's disk extraction.
- mzML `scan_number` comes from the native id when it identifies the spectrum on its own (`scan=`, Thermo, `index=`, `spectrum=`), else `None`, instead of the 0-based spectrum index. `write_ms2` numbers such spectra by position and keeps the native id as `I NativeID`; `write_mgf` still writes `SCANS` only for a set scan number, with the native id as `TITLE`.
- mzML ion mobility type comes from the declared unit: Vs/cm² is `ook0`, milliseconds `drift_time_ms`, seconds `drift_time_ms` scaled by 1000, no unit the generic `im`.
- tdfpy 5 changes some Bruker peaks: MS1 spectra can differ by one or two peaks (about 40 ppm of summed intensity) and some DDA MS2 peaks that 0.8 merged come out separately.
- `AcquisitionType` is tdfpy's enum. `da_to_ppm`/`ppm_to_da` are tacular's (relative to `abs(mz)`).
- `core.JSON_SCHEMA_VERSION` is gone; the spectrum schema version is `serialization.SPECTRUM_SCHEMA_VERSION` (2); `PeakListLookup` and `ThermoScanLookup` are imported from the package root, not `spxtacular.reader`.
- `MatchedFragment` is frozen, slotted and keyword-only, with an `annotation` property that returns the match as a paftacular `PafAnnotation`.
- Fragment labels come straight from paftacular 2's mzPAF writer, including negative charges.
- `DReader` MS1 spectra have `native_id` `"frame=F"` and DDA MS2 spectra `"precursor=P"` (was `None`), so `write_mgf` writes those as `TITLE`.

### Added

- `SpxtacularError`, the `SpectrumLookup` protocol and the reader lookup types (`DReaderMs1Lookup`, `DReaderMs2Lookup`, `MzmlSpectraLookup`, `ThermoScanLookup`, `PeakListLookup`) are exported from the package root.
- `DReader` MS1 spectra carry `total_ion_current`; precursors from `.d` and mzML files carry `im_type`.
- Every reader and `Reader` fetch one spectrum with `get_by_scan(n, ms_level=)`, `get_by_native_id(id)` and `get_by_sage_scannr(scannr)`. A missing key raises `KeyError`; a key that names no single spectrum (no scan numbers in the file, duplicates, an ambiguous Bruker number) raises `SpxtacularError`. Peak-list lookups index the file once, then seek.
- `read_mzspeclib` and `write_mzspeclib` read and write HUPO-PSI mzSpecLib 1.0 spectral libraries, text and JSON. Entries are `LibraryEntry` objects: an `MsnSpectrum`, analytes with peptacular ProForma peptidoforms, scores, mzPAF peak annotations as paftacular objects, and the remaining CV attributes. See `docs/mzspeclib.md`.
- `write_msp` and `write_mgf` take `annotations=` to write mzPAF peak annotations (strings, `PafAnnotation`s or `match_fragments` output) as a quoted peak column. Default output is unchanged, and `MgfReader` now skips a quoted annotation column.
- `get_by_sage_scannr` reads Sage's `scannr` for mzML, MGF, Thermo and Bruker DDA `.d` (precursor id = scannr + 1 for upstream Sage; `precursor_offset=0` for Sage on timsrust 0.6 or later).

### Performance

- `match_fragments` searches all fragments at once (about 2x faster for 3,000 fragments).
- `Spectrum.merge` runs as a single greedy kernel, numba-compiled when installed (about 50x faster without ion mobility, 2x with it, on 50,000 peaks).
- Bruker DDA MS2 reading uses tdfpy's batched per-precursor peaks (about 8x faster).
- Label collision checks in plot tables are O(n log n).

## [0.8.0] (2026-09-23)

### Fixed

- `Spectrum.load("x")` now finds the `x.npz` that `save("x")` writes, instead of raising `FileNotFoundError`.
- `mass_error_plot` and `facet_plot` treat `unit` case-insensitively (`"PPM"` plotted Da errors under a ppm label) and raise `ValueError` for an unknown unit.
- Centroiding keeps a profile peak sampled at only three points instead of dropping it.
- The missing-backend error from `write_indexed_mzml_gzip` names mzMLPy 0.9, the version actually required.
- `MzmlReader` keeps a precursor whose selected ion has no peak intensity (MS:1000042 is optional) and reads its intensity as 0.0, as the MGF reader does. It was dropped with a warning, losing its m/z and charge.
- Deconvolution and `remove_precursor_peak` space isotope peaks by the 13C-12C mass difference, 1.00335483507 Da. peptacular's rounded 1.003350 put neutral masses decharged from an A+n apex off by n * 4.8e-6 Da, so deconvolved masses of larger molecules shift by a few ppb.
- `Spectrum.sort` is stable in both directions, so tied peaks keep their input order and sorting a sorted spectrum returns it unchanged. `reverse=True` flipped tied peaks.
- `denoise` and `estimate_noise_level` ignore NaN and infinite intensities. One NaN peak made the `"histogram"` method raise and the other methods return a NaN threshold that removed every peak.
- `denoise("histogram")` no longer raises when the intensities span too narrow a range to cut into 100 bins.
- `filter(top_n=...)` raises `ValueError` for a negative count instead of returning an empty spectrum.
- `match_fragments`, `score`, `annotate` and the other fragment-matching entry points raise a clear `TypeError` when given a string such as `"PEPTIDE/2"` instead of fragments, instead of `AttributeError: 'str' object has no attribute 'mz'`.
- `Reader` raises `FileNotFoundError` at construction when the path does not exist, instead of failing later on first access.

### Changed

- Required spectrl 3 (`spectrl>=3.0,<4`). **Breaking for stored or shared tokens:** `to_spectrl_token`/`to_spectrl_url` now write `spectrl.v3.…` tokens, and `from_spectrl_token`/`from_spectrl_url` reject the `spectrl.v1` tokens earlier spxtacular releases wrote. Every spxtacular field still round-trips. The default lossy profile now picks the smallest bounded-error encoding per array (m/z within 0.1 ppm), so token bytes differ from before; `lossless=True` stays bit-exact. Decoding applies spectrl's default untrusted-input limits (1,000,000 peaks, 64 MiB of arrays). Downstream packages that read spxtacular tokens (`msbit`, `pepbit`) need spectrl 3 too.
- Capped the sibling requirements (peptacular, paftacular, tdfpy, mzmlpy) at their next breaking version, so installs no longer pick up a new major release before spxtacular is tested against it. Raised the minimums to peptacular 4.2, paftacular 1.4 and tdfpy 4.1 (`[bruker]`).
- Fragment labels in annotated plots and plot tables follow paftacular 1.4's mzPAF output: peptacular z fragments are labelled `z3-H` instead of `z3`, and d, v and w ions (including `da`/`db`/`wa`/`wb` and side-chain variants such as `d-valine`) get labels such as `d3^2` instead of raising `ValueError`.

### Deprecated

- `Spectrum.plot_table(show_charges=...)` warns with `DeprecationWarning`, as `plot(show_charges=...)` already did. Use the new keyword `color="charge"` (default) or `color=None`.

## [0.7.0] (2026-09-04)

- Required peptacular 3.3.0 and tdfpy 4.0.1. Verified position-free precursor fragment scoring and preservation of fractional-scan precursor mobility. Bruker mobility values can change because tdfpy now preserves the recorded fractional coordinate.
- Closed HTTP-error response bodies during USI fetch failures and removed the broad resource-warning suppression from tests.
- Updated the complete dependency lock to current compatible releases. Required mzMLPy 0.9.0, paftacular 1.2.0, and spectrl 1.1.0, and verified native mzML array conversion. Adjusted optional-backend annotations for the current type checker.
- Fixed hyperscore to use a matched-intensity dot product and factorial terms. Scores and thresholds from 0.6.0 must be recomputed. Removed unsupported cross-engine equivalence claims and clarified unweighted entropy similarity.
- Prevented combining neutral masses with m/z values or discarding conflicting charged-spectrum provenance. Decharge incompatible inputs separately before combining. Empty inputs no longer erase measurement metadata.
- Preserved wide flat-topped peaks during centroiding and removed symmetric Gaussian fit bias.
- Invalidated normalization after peak selection, intensity updates, concatenation, and rounding. Non-inplace transformations now own their precursor lists.
- Preserved the neutral-mass marker through rounding and rejected neutral masses in m/z-only peak-list exports. Peak-list writers now replace destinations atomically after successful completion.
- Distinguished scan-index chromatograms from retention times in seconds and rejected mixed or nonfinite time axes.
- Added independent analytical regression fixtures and a reproducible deconvolution benchmark. Expanded CI to isolated wheels on Linux, Windows, and macOS, Python 3.12 through 3.14, lowest installable direct dependencies, strict documentation builds, and required Thermo runtime tests. Locked development installs and pinned workflow actions to commits.

## [0.6.0] (2026-08-28)

### Added

- Added versioned `to_dict`/`from_dict` and `to_json`/`from_json` transport for `Spectrum`, `MsnSpectrum`, and `Chromatogram`, including packaged JSON Schema documents.
- Added configurable mzML gzip handling so large compressed runs can use low-latency sequential streaming.
- Added automatic mzML access selection, observable `access_strategy`, and a thin self-indexed gzip creation helper backed by mzMLPy 0.7.

### Changed

- Added strict validation for nested deconvolution provenance, scan identifiers, JSON object keys, and packaged Draft 2020-12 schemas.
- Changed mzML reading defaults to `gzip_mode="auto"` and `in_memory=False` for scalable random access.

### Fixed

- Accepted prefixed MGF scan identifiers such as `SCANS=F1:2478` while preserving numeric range behavior.

## [0.5.0] (2026-08-17)

### Added

- Added cached BRAIN-style isotope envelopes and configurable `IsotopeModel` presets for peptides, glycans, lipids, DNA, and RNA.
- Made isotope-envelope length adaptive and shared the selected model with automatic precursor removal.
- Added apex-aware, bidirectional deconvolution that can infer a missing monoisotopic peak and reject abundance or mobility mismatches.
- Added isotope-model and envelope parameters to deconvolution provenance while retaining schema-v1 compatibility.
- Added `to_matchms`/`from_matchms` with namespaced metadata preservation and filtered-peak array realignment.
- Added `to_spectrum_utils`/`from_spectrum_utils` for centroided MS/MS annotation and plotting workflows.
- Added `[matchms]`, `[spectrum-utils]`, and combined `[interop]` optional extras.
- Added MGF, MS2, and MSP readers and writers with gzip support and `Reader` auto-detection.
- Added `ThermoReader` through the optional `[thermo]` extra with lazy .NET initialization.
- Added `Spectrum.filter(top_n_per_window=(n, width))` for fixed-width window filtering.
- Added spectrum similarity metrics: `cosine`, `modified_cosine`, and `entropy_similarity`.
- Added run-level TIC/BPC/XIC extraction and plotting, including ion-mobility windows and one-pass multi-target extraction.
- Added profile-spectrum rendering, min/max profile decimation, and `profile_centroid_plot()`.
- Added `spectrum_from_proxi_response()` for clients that fetch PROXI JSON themselves.
- Added relative-intensity and sqrt/log plot scaling, precursor markers, hover hit areas, and responsive sizing.
- Added `sequence_coverage_plot()`, `table_view()`, texture encoding, `save_figure()`, and palette customization.
- Added a centralized light/dark plotting theme with conventional ion-series colors and ordinal charge ramps.
- Added vertical, collision-aware fragment labels with configurable `label_angle`.

### Changed

- The `[spectrl]` integration now requires Spectrl 1.0, emits the frozen `spectrl.v1` token format, and preserves ion mobility through accession-keyed auxiliary arrays.
- `hyperscore` implemented a product-of-series-sums heuristic, incorrectly identified as X!Tandem. This was corrected after 0.6.0. Stored thresholds must be retuned.
- `spectral_angle` now implements the literature metric when `predicted_intensities` are supplied and otherwise retains the documented flat-reference fallback.
- Deconvolution now uses complete-envelope scoring, float64 arithmetic, expected-position stepping, and isotope templates through 20,000 Da.
- Matching and scoring now accept unsorted m/z arrays by sorting a working copy and mapping indices back.
- `plot_spectrum()` now chooses sticks or a continuous trace from `spectrum_type`, with `render=` available as an override.
- Plot tables now distinguish plotted `intensity` from true `intensity_abs` and preserve rows with missing grouping values.
- Ion colors follow the proteomics convention: b blue, y red, a green, c teal, x purple, and z orange.
- `facet_plot()` groups peaks into traces instead of creating one trace per peak.
- Direct labels are capped and collision-avoided; complete values remain in hover text and table output.
- Generated plot HTML is no longer tracked and is rebuilt by the documentation hook.
- Deconvolution, merging, and plotting were optimized without changing results.

### Fixed

- Peak queries now reject unavailable charge/ion-mobility filters, validate `collision=`, and use stable top-N tie-breaking.
- Centroiding now accepts an intensity threshold and detects flat apexes.
- Deconvolution now recovers monoisotopic masses when the observed apex is A+1/A+2 and scores one-peak clusters as zero.
- Deconvolution now preserves ion mobility, validates `charge_range`, and warns when `max_dpeaks` truncates work.
- Rejected total-intensity clusters no longer double-count their remaining peaks.
- Spectrum equality now compares arrays element-wise.
- Non-inplace transformations no longer share numpy buffers or alias `self` on no-op paths.
- Explicit `spectrum_type` is no longer overwritten merely because a charge array exists.
- Tolerance and peak-selection enums are validated consistently instead of silently falling through.
- `filter()` now rejects criteria for dimensions absent from the spectrum.
- `merge()` preserves maximum `iso_score`, and `round_mz()` resets incompatible processing state.
- `centroid()` clears stale isotope scores, and inplace updates revalidate array shapes.
- `get_peak()`/`get_peaks()` preserve isotope scores and return Python scalars.
- `decharge()` uses `peptacular.PROTON_MASS`, treats nonpositive charges as unknown, rejects non-deconvoluted input, and no longer erases spectra when every charge is unknown.
- Normalization, scaling, denoising, serialization, and list-backed construction now handle invalid or degenerate inputs safely.
- Fragment matching now skips charge-incompatible neighbors correctly and validates fragment charges.
- Negative-mode fragments now match deconvoluted peaks by charge magnitude.
- Negative-mode fragment annotations now render without mzPAF charge errors.
- Scoring now handles NaNs, zero tolerances, dict fragments, and internal-ion runs correctly.
- Histogram noise estimation now focuses on the low-intensity bulk, and all estimators return zero for empty input.
- Corrected swapped mzML scan-window accessions and several activation/ion-mobility CV mappings.
- Spectrl round-trips now preserve precursor mobility, injection time, monoisotopic flags, enums, and unknown metadata strings.
- Readers now emit canonical enums and avoid incorrectly classifying centroid spectra as deconvoluted.
- `fetch_usi()` now validates before network access, uses deterministic precursor precedence, and preserves identifiers, spectrum representation, and scan polarity.
- Reader handles are closed reliably; uppercase `.D`, `.mzML.gz`, and broken optional native backends are handled correctly.
- Plotting now handles zero/NaN intensities, mzPAF labels, missing labels, mirror hover values, and required-column validation.
- Corrected documentation examples, signatures, parameter names, reader iteration examples, and dependency descriptions.

### Packaging and tests

- Reduced the source distribution from 91 MB to about 140 KB by excluding fixtures and generated plots.
- Switched publishing to PyPI trusted publishing and expanded CI to pull requests, Python 3.12/3.13, and minimal installs.
- Moved dev dependencies to PEP 735 groups and added typed-package, license, pytest, and coverage metadata.
- Removed stale generated files, scripts, manifests, and workflows.
- Strengthened deconvolution, filtering, sorting, plotting, query, and numba/Python parity tests.

## [0.4.0] (2026-07-09)

### Breaking changes

- Removed `spxtacular.compress`, `spxtacular.urlparams`, and their `Spectrum` methods in favor of spectrl tokens and URLs; `.npz` persistence is unchanged.
- Replace `spec.compress()`/`spec.to_url_params()` with `spec.to_spectrl_token()` and use `Spectrum.from_spectrl_token()` to decode.

### Added

- Added `spxtacular.spectrl_bridge`, `Spectrum.to_spectrl_token()`, `Spectrum.from_spectrl_token()`, and standalone conversion helpers.
- Added spectrl URL/data-URI helpers with fragment, query, and data modes.
- Preserved `iso_score` and spxtacular-only scalar metadata in spectrl round-trips.
- Added open-vocabulary `Polarity`, `ActivationType`, `IMType`, and `Analyzer` string enums.
- Added the optional `[spectrl]` extra.

### Fixed

- Restored a consistent default fragment tolerance of `0.02 Da` across matching, scoring, and visualization APIs.
- Made plot color options keyword-only to prevent positional argument misbinding.
- Added `Spectrum.is_decharged` and safe repeated-decharge behavior.
- Fixed zero-mass matching, `top_n=0`, case-insensitive merge mobility modes, and multiple mzML mobility arrays.
- Fixed DReader lifecycle handling and ion-mobility plotting for empty/NaN data.
- Preserved unknown spectrl metadata, precursor mobility, MSn classification, and standard PSI-MS activation accessions.
- Non-inplace processing methods now always return independent objects.
- Deconvolution now handles rejected clusters, invalid charge ranges, empty spectra, and all-zero normalization safely.
- Updated dependency floors to tested Python 3.12-compatible versions.

## [0.3.1] (2026-05-14)

### Added

- Moved `tdfpy` and `mzmlpy` to optional reader extras while keeping their reader classes importable.
- Added ion-mobility coloring to spectrum plots.
- Added URL query-parameter serialization and `iso_score` support to the legacy compressor.

### Changed

- Temporarily changed `Spectrum.match_fragments()` tolerance units to PPM; version 0.4.0 restored Da consistently.
- Updated reader and `paftacular` dependency packaging.

### Fixed

- Supported singleton charges in compressed spectra.
- Deprecated `show_charges` in favor of `color=` without breaking calls.
- Fixed missing retention times in `MsnSpectrum.__str__`.
- Made fragment matching adapt to centroided, deconvoluted, singleton, and decharged spectra.
- Unified persistence implementations and retained backward compatibility for the old `score` key.
- Supported both released and upcoming `tdfpy` centroid APIs.

## [0.3.0] (2026-04-07)

- Added Bruker PRM reading, USI loading, precursor-envelope removal, intensity scaling, and m/z rounding.
- Added mass-error and faceted plots plus `Spectrum.annotate()`.
- Added `da_to_ppm()` and `ppm_to_da()`.

## [0.2.0] (2026-03-18)

- Added the editable plot-table API and `Spectrum` plot-table convenience methods.
- Added scored deconvolution, isotope scores, and score-based filtering.
- Added charge-aware fragment matching and eight peptide-spectrum scoring metrics.
- Added mirror and annotated spectrum plots.
- Added `.npz` persistence for `Spectrum` and `MsnSpectrum`.
- Added spectrum combination, charge-aware peak merging, and unified reader auto-detection.
- Added pandas as a runtime dependency and completed package metadata/LICENSE cleanup.

## [0.1.0] (2026-01-16)

- First PyPI release.
