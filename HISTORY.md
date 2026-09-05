# History

User-visible changes only; implementation details belong in commits and pull requests.

## Unreleased

## 0.7.0 (2026-09-04)

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

## 0.6.0 (2026-08-28)

### Added

- Added versioned `to_dict`/`from_dict` and `to_json`/`from_json` transport for `Spectrum`, `MsnSpectrum`, and `Chromatogram`, including packaged JSON Schema documents.
- Added configurable mzML gzip handling so large compressed runs can use low-latency sequential streaming.
- Added automatic mzML access selection, observable `access_strategy`, and a thin self-indexed gzip creation helper backed by mzMLPy 0.7.

### Changed

- Added strict validation for nested deconvolution provenance, scan identifiers, JSON object keys, and packaged Draft 2020-12 schemas.
- Changed mzML reading defaults to `gzip_mode="auto"` and `in_memory=False` for scalable random access.

### Fixed

- Accepted prefixed MGF scan identifiers such as `SCANS=F1:2478` while preserving numeric range behavior.

## 0.5.0 (2026-08-17)

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

## 0.4.0 (2026-07-09)

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

## 0.3.1 (2026-05-14)

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

## 0.3.0 (2026-04-07)

- Added Bruker PRM reading, USI loading, precursor-envelope removal, intensity scaling, and m/z rounding.
- Added mass-error and faceted plots plus `Spectrum.annotate()`.
- Added `da_to_ppm()` and `ppm_to_da()`.

## 0.2.0 (2026-03-18)

- Added the editable plot-table API and `Spectrum` plot-table convenience methods.
- Added scored deconvolution, isotope scores, and score-based filtering.
- Added charge-aware fragment matching and eight peptide-spectrum scoring metrics.
- Added mirror and annotated spectrum plots.
- Added `.npz` persistence for `Spectrum` and `MsnSpectrum`.
- Added spectrum combination, charge-aware peak merging, and unified reader auto-detection.
- Added pandas as a runtime dependency and completed package metadata/LICENSE cleanup.

## 0.1.0 (2026-01-16)

- First PyPI release.
