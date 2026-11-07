# History

User-visible changes only; implementation details belong in commits and pull requests.

## [Unreleased]

## [0.9.4] (2026-11-07)

### Added

- Documentation: a [Development](https://tacular-omics.github.io/spxtacular/development/) page —
  clone, uv environment, every `just` recipe, the checks to run before a pull request, and what
  CI adds on top — linked from the README, Getting started and `llms.txt`.

### Changed

- Documentation: the README and `CONTRIBUTING.md` say which install is for using spxtacular (pip
  or `uv add`) and which is for working on it (a clone with uv and the `just` recipes).
- Documentation: every runnable snippet in the README and in the documentation carries the imports
  it needs. Thirty blocks across the API reference, Readers, Scoring, Spectrum and Visualization
  pages previously ran only because an earlier block on the page had imported the name. Each page
  already said that its blocks share the data its setup block defines; now only the data is
  shared, not the imports.
- Documentation: the API reference describes `IsotopeModel`'s methods (`distribution`,
  `adaptive_distribution`, `apex_index`, `estimate_composition`, `to_dict`/`from_dict`) and its
  fields, which only the migration guide referred to before.
- The `thermo`, `readers` and `all` extras now require `fisher-py>=2.0.2` instead of
  `>=2.0.0`: fisher-py 2.0.0 and 2.0.1 were never published to PyPI, where 2.0.2 is the only
  2.x release, so the old floor named a version no installer could supply.
- `pandas>=2.1.2` replaces `pandas>=2.0`. No pandas 2.0.x or 2.1.0 ships a wheel for Python
  3.12, the oldest Python spxtacular supports, but each of them does publish a source
  distribution that claims to support it, so an installer selects one and then fails to build
  it. 2.0.0 and 2.0.3 abort on a `pkg_resources` build dependency they do not declare, and
  2.1.0 aborts because numpy's Cython headers now require Cython 3. The one installable
  release this floor removes is 2.1.1, the first with a 3.12 wheel, and it is removed on
  purpose: 2.1.1 requires `numpy>=1.26.0` with no upper cap while shipping the numpy 1.x
  binary interface, so an installer pairs it with numpy 2.x and importing pandas raises
  `numpy.dtype size changed`. 2.1.2 is the first release that caps `numpy<2` and imports
  cleanly.
- The `numba` and `all` extras require `numba>=0.59` instead of `>=0.64.0`. 0.59.0 is the
  first numba with a Python 3.12 wheel, and it installs and imports at that floor. Only an
  install that asks for `numba` without the interop extras can reach it: numba 0.59 requires
  `numpy<1.27` while matchms requires `numpy>=2.0.0`, so `spxtacular[all]` with numba pinned
  to 0.59.0 has no solution at all, and the lowest numba a full install resolves to is
  0.61.0. That is the version the lowest-resolution CI job tests, so the suite is not
  exercised at 0.59 itself.
- `plotly>=5.15` replaces `plotly>=6.6.0` for the base install, and the `plotly-export` and
  `all` extras now require `plotly>=6.1.1` alongside `kaleido>=1.0`. 5.15.0 is the lowest
  release the test suite passes on. Static image export is the only feature that needs a
  newer plotly, so its requirement sits in the extra that asks for export rather than in the
  base install. plotly and kaleido are a pair there, because export goes through whichever
  kaleido interface a plotly release knows: plotly 6.0.1 and below raise an error with
  kaleido 1.x, plotly 6.1.0 writes no file and raises nothing, and 6.1.1 is the first release
  that exports correctly with kaleido 1.0.0 and later. 6.1.0 is why the extra names a floor
  at all, because it leaves a user with a missing figure and no traceback, and only a floor
  of 6.1.1 keeps a resolver away from it. The test suite now exports a plotly figure for
  real, so this pair is exercised at its floor and not only at current versions. That one
  test is marked `slow`, skips unless kaleido, a Chrome or Chromium binary and plotly 6.1.1
  or newer are all present, and carries a 60 second per-test timeout on purpose, because the
  suite's 15 second default does not fit a cold browser start.
- The `matchms`, `interop` and `all` extras require `matchms>=0.30` instead of `>=0.33.1`,
  and the `spectrum-utils`, `interop` and `all` extras require `spectrum-utils>=0.4.0`
  instead of `>=0.5.0`. The interop tests pass at both floors, which the lowest-resolution
  CI job installs. Lowering the matchms floor does not lower the numpy or pandas a user ends
  up with, because matchms 0.30.0 itself requires `numpy>=2.0.0` and `pandas>=2.2.3`: any
  install that includes matchms stays above this package's own numpy and pandas floors.
- The `bruker`, `readers`, `all` and dev requirements now require `tdfpy>=5.1,<6` instead of
  `>=5,<6`. tdfpy 5.1.0 is the first release whose `isolation_ook0_range` ends at the window's
  last scan rather than one past it, and `MsnSpectrum.isolation_ook0_range` is that window
  verbatim, so every range this package reported with tdfpy 5.0.x was one scan too wide.

- `save_figure()` says what to do when plotly static export finds no browser. kaleido drives a
  Chrome or Chromium browser and never downloads one, so a machine without one got plotly's
  bare `RuntimeError`. The error now also names `plotly_get_chrome`, the command that installs
  a browser, and keeps the original on `__cause__`; any other `RuntimeError` from the writer is
  re-raised untouched. The browser requirement is documented wherever the `plotly-export` extra
  is.

### Fixed

- `remove_precursor_peak(isotopes="auto")` now removes the whole isotope envelope the isotope
  model reports, including the monoisotopic peak of a heavy precursor. It re-filtered that
  envelope against its own apex, so from roughly 12 kDa upwards the monoisotopic peak and the
  isotopes next to it fall below the 1% threshold and were left in the spectrum: a 20 kDa
  precursor at m/z 2001.0073 kept three of its peaks. Peptide-sized precursors are unaffected,
  since their monoisotopic peak is at or near the apex.
- `MzmlReader`'s scan-number lookups no longer answer for an mzML index that has since been
  rebuilt. mzmlpy 0.10.3 treats the document as the authority and its offset index as a cache
  of it, so reading a file whose index does not list every spectrum lengthens that index
  mid-read; the index `get_by_scan()`, `get_by_native_id()` and `get_by_sage_scannr()` share
  was built from the old ids and kept raising `KeyError` for a spectrum the walk had just
  yielded. It is now rebuilt when an open file's spectrum count changes, and dropped when the
  reader is closed.

## [0.9.3] (2026-10-26)

### Changed

- Documentation: the Spectrum guide is one runnable walkthrough of a single spectrum, and
  every signature and parameter table now lives in the API reference, instead of the two
  being interleaved. Each step of the guide links to the reference entry for the method it
  uses.
- Documentation: the README points to the Spectra web application, which opens a spectrum
  from a `to_spectrl_url()` link in the browser.

### Fixed

- Documentation: every signature shown in the API reference and the guides matches the
  code (keyword-only markers, `backend`/`style`/`size` on the plotting functions,
  `da_to_ppm(delta, ...)`, `Precursor` as a standalone class), and the public names the
  reference omitted are listed.
- Documentation: every Python example in the guides and the README runs as written,
  against the small mzML, Bruker `.d` and mzSpecLib files the repository ships. A block
  that is not meant to be executed says so above it (a signature listing, or one that
  needs your own data file, network access or the Thermo .NET runtime), and
  `pytest tests/test_docs.py` runs the rest page by page so CI catches an example that
  stops working.

## [0.9.2] (2026-10-08)

### Fixed

- `resolve_ionization_model` (and every `ionization_model=` argument) accepts a
  ProForma 2.1 charge carrier, the notation peptacular parses: `"K:z+1"`,
  `"Li:z+1"`, `"Cl:z-1"`, `"H:z+1^-1"`. Mass and charge come from peptacular's
  parser, so any singly charged carrier it knows works; one matching a preset
  returns the preset. Mixed carriers (`"Na:z+1,H:z+1"`) and multiply charged
  carriers raise `SpxtacularError`. The sodium and ammonium cation masses are now
  computed from tacular's element masses instead of local literals (unchanged).
- `sequence_coverage_plot` places every fragment ion type tacular marks as forward
  or backward on the ladder (c-H, z., z+H, d, v, w, ...); it used to drop all but
  a/b/c/x/y/z. Ion colours and dash patterns give a variant its base series'
  colour (z. and z+H are z orange, c-H is c teal) instead of neutral grey.

- `write_ms2` writes the `Z` line mass as the singly protonated `[M+H]+` mass for
  negative-mode and adduct precursors too. It used to assume a protonated
  positive ion, so `M = 1000` at `z = -2` was written as 996.978 instead of
  1001.007. The carrier comes from the spectrum's deconvolution provenance, else
  the charge sign; the new `ionization_model=` overrides it (for example
  `"sodiated"`).
- `IsotopeModel.distribution`, `IsotopeModel.apex_index` and
  `brain_isotopic_distribution` no longer cut every envelope at 32 peaks by
  default, which put the apex of a 200 kDa protein at 31 instead of about 124.
  `max_isotopes` now defaults to `None`: 32 peaks (results below about 6 kDa are
  unchanged), extended when the mass needs more. An integer still truncates.
- `Ms2Reader` keeps `I` line values verbatim. It rebuilt them from
  whitespace-split fields, so a `NativeID` such as `run 1  "fraction=3"` read
  back with its double space collapsed and no longer matched what `write_ms2`
  wrote.
- `annotate_spectrum`, `build_annot_plot_table` and `mass_error_plot` no longer
  crash on fragments of ion type `n` (the intact neutral peptacular generates),
  which mzPAF cannot express. Such fragments are labelled with the ion type and
  charge (`n`, `n^2`) instead of raising `PaftacularError`. Every other fragment
  mzPAF cannot write (an uncharged one, for example) still raises.
- `write_mgf`, `write_ms2` and `write_msp` raise `SpxtacularError` when a
  `native_id` (MGF `TITLE`, MS2 `I NativeID`, MSP `Name`) or an MS2
  `activation_type` contains a line break. It used to be written raw, so the file
  read back with the value cut short and the rest parsed as extra peaks or
  fields, without an error.

## [0.9.1] (2026-10-06)

### Added

- `NATURAL_ISOTOPE_ABUNDANCES` and the monoisotopic masses used by `IsotopeModel`
  and `brain_isotopic_distribution` now cover every element with a natural
  abundance (84, including the halogens), built from Tacular's element data
  instead of a hard-coded C, H, N, O, P, S table (#43).

## [0.9.0] (2026-09-24)

Breaking release. Every rename and removal, old -> new, is in the migration guide
(`docs/migration.md`).

### Changed

- Requires tacular 2, peptacular 5, paftacular 2, and (extras) tdfpy 5 and mzmlpy 0.10. Build fragments with `ion_types=("b", "y")`: peptacular 5 reads `"by"` as one ion type.
- `Precursor` is a standalone keyword-only dataclass (no longer a `Peak`): `mz` -> `precursor_mz`, plus a new `im_type` naming the kind of mobility in `im`. `intensity` defaults to 0.0 and `is_monoisotopic` to `None`.
- `MsnSpectrum.isolation_im_range` -> `isolation_ook0_range`, always `(low, high)`. 0.8 gave Bruker windows as `(high, low)`; 0.8 JSON, npz and spectrl tokens are read back sorted.
- Parameters with a default are keyword-only across the public API (`spec.normalize(method="tic")`, `match_fragments(spec, frags, tolerance=10, tolerance_unit="ppm")`), and so is everything after `intensity` in `Spectrum`, `Peak`, `Chromatogram` and `decon.deconvolve_spectrum`, plus the optional fields of `IsotopeModel`, `IonizationModel` and `DeconvolutionProvenance`.
- `decon.deconvolve_spectrum(is_ppm=)` -> `tolerance_unit=`; `has_peak`/`get_peak`/`get_peaks(im_tol=)` -> `im_tolerance=`; `get_peak(collision=)` -> `peak_selection=`.
- Removed the deprecated `show_charges=` of `Spectrum.plot`, `Spectrum.plot_table` and `plot_spectrum`; use `color=`.
- Invalid input raises `SpxtacularError`, a `ValueError` subclass, so existing `except ValueError` still works. So do enum coercion, bad JSON payloads (was `TypeError`), corrupt `.npz` files and malformed mzML (was mzmlpy's `MzmlParseError`). Reader lookups before `open()` raise it instead of `RuntimeError`.
- `im_type` and `polarity` are validated on construction; `activation_type` and `analyzer` turn member names (any case) and PSI-MS accessions (`"MS:1000484"` -> `Analyzer.ORBITRAP`) into enum members and keep other vendor strings. `im_type` coercion is case-insensitive; tolerance units and polarity are lowercase only.
- Every `tolerance_type=` / `mz_tolerance_type=` / `im_tolerance_type=` parameter and field is now `tolerance_unit=` / `mz_tolerance_unit=` / `im_tolerance_unit=` (matching, scoring, similarity, `merge`, `deconvolute`, `extract_xic`, peak queries, plotting, `CentroidConfig`, `Chromatogram`, `DeconvolutionProvenance`). Values are the lowercase strings `"da"`/`"ppm"`; `"Da"` and `"PPM"` raise. `DEFAULT_FRAGMENT_TOLERANCE_TYPE` -> `DEFAULT_FRAGMENT_TOLERANCE_UNIT`.
- `ToleranceType`, `ToleranceLike`, `Polarity` and `PolarityLike` are removed. Units and polarity are plain strings typed by tacular's `tacular.types.ToleranceUnit` and `tacular.types.Polarity`; `Spectrum.polarity`, `IonizationModel.polarity` and the presets hold `"positive"`/`"negative"`.
- Deconvolution provenance is schema version 3 (`tolerance_unit`, `im_tolerance_unit` keys) and chromatogram JSON is schema version 2 (`tolerance_unit`; `schemas/chromatogram-v2.schema.json` replaces `chromatogram-v1.schema.json`). Older files with `tolerance_type` keys still load.
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
- Figures are drawn through a backend-neutral figure layer and restyled for print: horizontal typeset fragment labels (y₇²⁺ as subscript and superscript), italic *m/z*, "Mass error (ppm)" and "Intensity (×10ⁿ)" axis titles, and new fonts and sizes in plotly too. Plotting functions return `Any` (a plotly `Figure` by default).
- Plot tables drop the `label_font`, `label_yshift` and `label_xanchor` columns; `label_angle` defaults to 0 (was -90) and `label_size` is NaN for the style default. `plot_from_table` requires fewer columns.
- `save_figure(scale=)` defaults to `None` (the figure style's resolution, 600 dpi for `"paper"`) instead of 2.0, takes `dpi=`, writes matplotlib figures and figure specs as well as plotly, and raises `SpxtacularError` for an unsupported suffix. A missing kaleido (plotly to PNG, SVG or PDF) raises `ImportError` naming the `spxtacular[plotly-export]` extra.
- `style="paper"` draws no default title (pass `title=` to add one). `backend="matplotlib"` defaults to `style="paper"`; plotly keeps `"screen"`.
- Plot-table `linewidth` is relative to the default 1.6 and scaled to the style's stick width. `label_size`, `label_angle` and `label_color` are honoured per row. `label_color` stays as set; only rows that keep the series colour get the style's label colour. Unmatched peaks no longer block labels: a label may sit over a grey stick, on a background patch. A plot table without the `intensity_scale` attr is treated as relative when its y label starts with "Relative" and no intensity exceeds 100.
- Labelled plotly figures in the `"screen"` style keep their design width instead of autosizing, so placed labels do not collide at other widths. Unlabelled ones still autosize.
- `sequence_coverage_plot` sizes its height to the sequence rows, without the fixed blank band at the bottom. `reporter_ion_plot` labels both panels "Relative intensity (%)" with ticks at 0-100 when normalised. `facet_plot` fixes the mirror ticks at -100 and -50 only for relative intensities.

### Added

- `SpxtacularError`, the `SpectrumLookup` protocol and the reader lookup types (`DReaderMs1Lookup`, `DReaderMs2Lookup`, `MzmlSpectraLookup`, `ThermoScanLookup`, `PeakListLookup`) are exported from the package root.
- `DReader` MS1 spectra carry `total_ion_current`; precursors from `.d` and mzML files carry `im_type`.
- Every reader and `Reader` fetch one spectrum with `get_by_scan(n, ms_level=)`, `get_by_native_id(id)` and `get_by_sage_scannr(scannr)`. A missing key raises `KeyError`; a key that names no single spectrum (no scan numbers in the file, duplicates, an ambiguous Bruker number, a Sage `scannr` that is one spectrum's native id and another's scan number) raises `SpxtacularError`. Peak-list lookups index the file once, then seek.
- `read_mzspeclib` and `write_mzspeclib` read and write HUPO-PSI mzSpecLib 1.0 spectral libraries, text and JSON. Entries are `LibraryEntry` objects: an `MsnSpectrum`, analytes with peptacular ProForma peptidoforms, scores, mzPAF peak annotations as paftacular objects, and the remaining CV attributes. See `docs/mzspeclib.md`.
- `MzSpecLibReader` streams an mzSpecLib library one `LibraryEntry` at a time, in memory that does not grow with the number of spectra (text and JSON, gzipped or not), with the library attributes available before iterating. `read_mzspeclib` is built on it and returns the same results.
- `write_msp` and `write_mgf` take `annotations=` to write mzPAF peak annotations (strings, `PafAnnotation`s or `match_fragments` output) as a quoted peak column. Default output is unchanged, and `MgfReader` now skips a quoted annotation column.
- Isobaric reporter-ion quantification for TMT, TMTpro and iTRAQ: `extract_reporter_ions` / `Spectrum.reporter_ions` (one spectrum, `ReporterIons`) and `reporter_ion_table` (many spectra or a reader, one DataFrame row each). Channels and m/z come from tacular; the most intense peak within `tolerance` (default 20 ppm) is used. Optional isotope impurity correction from the reagent lot sheet or a matrix (`isotope_correction_matrix`, `correct_isotope_impurities`, non-negative least squares) and `normalize="sum"|"max"`. An impurity counts on a channel within min(0.02 Da, half the plex's smallest channel spacing), as in OpenMS; numeric lot-sheet shifts are 13C, 15N impurities need `"-15N"` labels.
- `get_by_sage_scannr` reads Sage's `scannr` for mzML, MGF, Thermo and Bruker DDA `.d` (precursor id = scannr + 1 for upstream Sage; `precursor_offset=0` for Sage on timsrust 0.6 or later).
- Publication figures. Every plot takes `backend="plotly"|"matplotlib"|"spec"`, `style="paper"|"screen"|"talk"` (or a `FigureStyle`) and `size="single"|"onehalf"|"double"`, a width in mm, or `(width, height)` mm. `"paper"` is 7 pt text at 85 mm and 600 dpi; matplotlib writes PDF/SVG/EPS with embedded TrueType fonts. New extras: `spxtacular[matplotlib]` and `spxtacular[plotly-export]` (kaleido), both in `all`.
- `compose_figure` lays out `backend="spec"` figures as one lettered multi-panel figure; `render`, `FigureSpec`, `FigureStyle` and `get_style` are exported. `size=None` (default) is the double-column width, or a 16:9 slide (254 × 143 mm) with `style="talk"`, and a panel too short for the style's text raises a `UserWarning`.
- Axes fit small panels in every figure: tick count follows axis length and font size, long axis titles are abbreviated ("Rel. int. (%)") or set smaller, the sequence header shrinks to its width, and crowded category labels are rotated or thinned with room reserved.
- `reporter_ion_plot(spectrum, plex="TMT10")`: channel bars from `extract_reporter_ions` with the reporter m/z region above, missing channels marked "n.d.". Takes any tacular plex name, an `IsobaricTagInfo` or a `ReporterIons`.
- `annotate_spectrum(peptide=)` draws the sequence with b/y cleavage marks above the spectrum, and `mass_error_panel=True` adds an error strip below. `mirror_plot` takes `fragments=` (annotate both halves), `names=` and `similarity=` (cosine, modified cosine, entropy or a number), plus `lower_fragments=` for a library half with another annotation. `mirror_labels="auto"` (default, also on `facet_plot`) labels an ion both halves share once, on the upper half; `"both"` and `"top"` label both halves or the upper only. `absolute_axis=True` shows absolute intensity with a ×10ⁿ exponent. `sequence_coverage_plot` accepts a ProForma annotation.

### Performance

- `match_fragments` searches all fragments at once (about 2x faster for 3,000 fragments).
- `Spectrum.merge` runs as a single greedy kernel, numba-compiled when installed (about 50x faster without ion mobility, 2x with it, on 50,000 peaks).
- Bruker DDA MS2 reading uses tdfpy's batched per-precursor peaks (about 8x faster).
- Label collision checks in plot tables are O(n log n).
- Label placement is vectorised: 500 labels on 20,000 peaks resolve in about 0.15 s.

### Fixed

- `Spectrum.centroid` no longer returns huge apex intensities (up to ~1e36x the data) for peaks whose neighbours are near zero, and keeps single-sample peaks next to zero or negative samples instead of dropping them. Such peaks fall back to a parabola on the intensities; well-sampled peaks are unchanged.

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
