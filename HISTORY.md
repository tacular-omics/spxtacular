# History

## Unreleased

### Documentation

* `docs/visualization.md` gained a "How these plots are built" preamble (relative intensity, label
  thinning, the hover layer, colour-by-job), sections for `sequence_coverage_plot`, `table_view`,
  `save_figure` and the theme (light/dark, brand palettes, texture), and updated parameter tables
  for every plotting function.
* `docs/api.md` gained a full `Theme` section, the three new functions, a rewritten plot-table
  schema (including `intensity_abs` vs `intensity`), and explicit notes that `mirror_plot`,
  `facet_plot` and `mass_error_plot` do *not* take `intensity_scale` / `intensity_transform` /
  `texture`.
* `llms.txt` documents the new surface plus the pitfalls an assistant would otherwise get wrong.
* `README.md` and `docs/index.md` cover the theme, the coverage ladder, and the accessible table.
* `CLAUDE.md` gained `theme.py` in the architecture tree, a "Plot colour" section recording the
  colour-by-job rules, and new entries under "What NOT to do".
* `plot_example.py` now also generates the sequence coverage ladder, a dark-mode spectrum, and a
  log-intensity spectrum, so the docs show them.
* Enabled the `admonition` and `pymdownx.details` mkdocs extensions — `!!! warning` blocks were
  previously rendering as literal text.

### Visualisation — new capabilities

* **Hover layer.** Sticks are ~1.6px wide, so the pointer previously had to land on
  a hairline to get a tooltip. Each figure now carries a transparent hit layer sized
  well beyond the mark, plus an m/z crosshair, so being *near* a peak is enough.
* **Relative intensity is the default axis** (`intensity_scale="relative"`), matching
  the convention every MS viewer uses: base peak = 100%. Raw counts remain available
  with `intensity_scale="absolute"`, and the true value is what every tooltip reports
  regardless — rescaling changes the axis, never the reported number. A new
  `intensity_transform` (`"sqrt"` / `"log"`) compresses dynamic range so
  low-abundance matched ions stay visible next to a dominant base peak.
* **`sequence_coverage_plot`** — the standard proteomics ladder: residues left to
  right, N-terminal (a/b/c) ticks above, C-terminal (x/y/z) below, with the fraction
  of backbone bonds covered in the title. It answers *where along the peptide* the
  evidence sits, which an annotated spectrum alone does not show.
* **Precursor marker.** `plot_spectrum` and `annotate_spectrum` now draw the
  precursor m/z and its isolation window as recessive reference chrome on any
  `MsnSpectrum` that carries them (`show_precursor=False` to suppress).
* **`table_view`** renders a plot table as an accessible HTML table. Labels are
  deliberately thinned from the figure, and a hover is unusable for keyboard and
  screen-reader users, so the values need a home that isn't the tooltip. Label text
  is HTML-escaped.
* **Texture channel** — `texture=True` gives each ion series its own dash pattern, so
  identity survives print, forced-colours, and readers who cannot separate two hues.
* **`theme.set_palette`** allows brand colours to replace any of the three palettes,
  with a docstring that is explicit that substituted hues are not validated for you.
* **`save_figure`** picks the writer from the file extension and reports a missing
  `kaleido` with the install command rather than an exception from inside plotly.
* Figures set `autosize`, so they fill their container in docs pages and notebooks
  instead of a fixed default box.

### Visualisation — new theme

* **New `spxtacular.theme` module** — the single source of truth for plot colour,
  replacing the palettes that were duplicated across `plot_table.py` and
  `visualization.py` and "kept in sync" by comment. Every palette is validated for
  colour-vision deficiency (protanopia/deuteranopia) in both light and dark modes.
* **Charge state is now an ordinal ramp, not a categorical cycle.** Charge has a
  natural order, so it takes one hue running light to dark and the reader sees
  1+ < 2+ < 3+ in the colour. This also fixes a real bug: the old ten-colour cycle
  made `z=1` and `z=11` identical, and colours depended on which charge states
  happened to be present rather than on the charge itself.
* **Dark mode** — `theme.set_plot_theme("dark")` globally, or `theme_mode="dark"`
  per call. The dark palette is its own validated set of steps for the dark
  surface, not an automatic inversion.
* Ion series take a fixed eight-slot categorical order (b and y first, the pair
  that co-occurs most often). Internal fragments and anything past the eighth slot
  fold to a neutral colour rather than inventing an indistinguishable ninth hue.
* Ion mobility uses a single-hue sequential ramp instead of Viridis; a multi-hue
  ramp bands a magnitude that has no bands.
* Chrome is recessive: no panel fill, solid hairline horizontal gridlines, no
  vertical grid, generous margins, consistent sans typography, and unmatched peaks
  drawn thinner and dimmer so the annotated peaks lead.

### Visualisation — fixes

* **`facet_plot` built one plotly trace per peak.** A 3000-peak spectrum produced
  3000 traces in 1.37s where the identical picture takes 1 trace and 0.05s; at
  5000 peaks the figure was effectively unusable in a browser. It now groups like
  `plot_from_table` does. It also silently dropped the ion labels that are the
  whole point of the annotated panel.
* **Direct labels are thinned instead of flooding the plot.** Every annotated peak
  used to get a layout annotation, so a deconvoluted spectrum rendered thousands of
  overlapping labels as an unreadable smear along the baseline. Labels are now
  capped (`max_labels`, default 25) *and* collision-avoided along the m/z axis,
  strongest peak wins. Dropped values remain in the hover text and the plot table.
* **`plot_from_table` silently deleted peaks** whose `series` or `color` was NA —
  `groupby` drops NA keys by default, and NA is exactly what a `merge`/`reindex`
  on a user-edited table produces. A 4-peak table rendered 3 peaks with no error.
* **`mirror_plot` hover reported normalised values** under an "intensity" label,
  so a peak of 50 000 showed as `5.00e-01`. It now reports the true intensity.
* `mirror_plot` produced an all-NaN, silently blank panel when either half had zero
  maximum intensity, and its charge colours did not match `plot_spectrum`'s, so the
  same spectrum changed colour between the two figures.
* `mass_error_plot` and `facet_plot` raised `ZeroDivisionError` on an all-zero
  intensity match set (real for thresholded or background-subtracted data) — the
  existing guard only caught an empty list. `mass_error_plot` also labelled `b3^1`
  and `b3^2` identically as `b3`; it now uses mzPAF labels.
* An NA label rendered the literal text `"nan"` onto the figure.
* `plot_from_table` now validates its required columns up front, instead of failing
  part-way through rendering or only on data that happens to carry labels.
* The zero-error reference line in `mass_error_plot` is a solid hairline rather than
  dashed — dashing reads as a threshold when it is just a reference.

### Tests

* New `tests/test_theme_and_viz.py` (24 tests) pins the properties that make these
  plots correct: trace counts, label thinning and collision separation, the sign of
  the mirrored half, hover values, colour assignment, and degenerate inputs. The
  existing plotting tests asserted only that a `Figure` came back, so none of the
  defects above would have been caught.

### Fixes — deconvolution (scientific correctness)
* **Monoisotopic peak recovery.** Cluster finding seeded on the most intense peak and extended
  *forward only*. Above ~1900 Da the A+1 peak is more intense than A (and above ~3500 Da it is
  A+2), so the reported monoisotopic mass was systematically one or two neutrons too high, and the
  charge state was sometimes wrong as well. The isotope score did not catch it — misaligned
  envelopes still scored 0.79–0.83. Deconvolution now searches backwards from the seed for
  candidate anchors and picks the alignment that best fits the isotope template. Verified exact
  recovery from 1000–4000 Da at charges 1–3.
* **Single-peak clusters no longer score 1.0.** A one-peak candidate scored a perfect 1.0 (a vector
  is trivially identical to itself after normalisation), beating and destroying genuine multi-peak
  clusters. This fired whenever `min_intensity` exceeded the seed intensity — i.e. exactly when
  feeding `estimate_noise_level()` in, as the docs recommend. Such clusters now score `0.0`.
* Cluster extension measures each step from the *expected* position rather than the previously
  matched peak, so a chain of peaks each just inside tolerance can no longer ratchet a cluster off
  target.
* Deconvolution now runs in float64 throughout; output m/z is no longer truncated through float32
  (~0.02 ppm). This also removes a float32-accumulator divergence between the numba and pure-Python
  backends, which now produce bit-identical results.
* `deconvolute()` preserves ion mobility instead of discarding it — previously the entire IM
  dimension was destroyed with no warning, defeating the main purpose of `DReader`.
* Isotope templates extend to 20000 Da (was 5000, silently clamped).
* `deconvolve_spectrum` validates `charge_range`; a reversed range silently returned every peak as
  a singleton. Hitting `max_dpeaks` now warns instead of truncating silently.

### Fixes — core Spectrum API
* `Spectrum == Spectrum` raised `ValueError` ("truth value of an array is ambiguous"), which also
  broke `in`, `list.remove`, and `assert spec == expected`. Equality is now element-wise.
* Methods documented as returning a new `Spectrum` returned objects **sharing numpy buffers** with
  the original, so writing into the result silently mutated the source. `update()` now copies any
  array field the caller did not replace.
* A `charge` array no longer forces `spectrum_type = DECONVOLUTED` over an explicit value. This had
  defeated the `SpectrumType` guard, allowing `decharge()` on never-deconvoluted centroid data.
* `tolerance_type` is coerced through `ToleranceType` and raises on unknown values. `"PPM"` used to
  fall through to Da — a window a million times too wide, silently. Applies across `core`,
  `matching` and `scoring`.
* `filter()` raises when given a criterion for a dimension the spectrum lacks, instead of silently
  ignoring it and returning every peak.
* `merge()` now carries `iso_score` through (max over the merged group) rather than dropping it.
* `round_mz()` resets `spectrum_type` when it drops the charge array, instead of leaving the
  spectrum wedged — claiming to be deconvoluted, refusing `decharge()`, and no-oping `deconvolute()`.
* `centroid()` resets `iso_score`; previously the inplace path left arrays at mismatched lengths and
  reported fabricated per-peak scores. `update(inplace=True)` now re-validates shapes.
* `get_peak()` / `get_peaks()` populate `iso_score` (previously dropped) and return Python scalars.
* `decharge()` uses `pt.PROTON_MASS` rather than a hardcoded constant, and treats `charge <= 0` as
  unknown instead of collapsing charge-0 peaks to a neutral mass of 0.0.
* `normalize()` guards non-finite normalisation factors; `scale_intensity()` clears the `normalized`
  flag so re-normalisation after a transform is not silently skipped, and rejects `degree=0`.
* `save()` no longer fails with `TypeError` on numpy scalars in reader-produced metadata.
* `denoise()` on an empty spectrum no longer emits numpy RuntimeWarnings.
* `__post_init__` coerces array dtypes, so Python lists no longer construct successfully and fail
  later inside `filter()`.

### Fixes — scoring, matching, noise
* `spectral_angle` no longer reports a **perfect 1.0 for NaN input**, and builds its observed vector
  at fixed length so the cosine cannot exceed 1. Its docstring now states plainly that it is not the
  literature spectral angle (there is no predicted-intensity vector).
* `probability_score` returns a finite value when `tolerance=0` (was `+inf`).
* `match_fragments` with `peak_selection="closest"` walks outward past charge-incompatible
  neighbours instead of missing charge-compatible peaks well inside tolerance.
* `peak_selection` and negative/zero fragment charges are validated rather than silently corrupting
  results; unsorted input m/z raises instead of silently returning no matches.
* `n_theoretical` agrees between dict and `Sequence[Fragment]` input; `longest_run` handles
  internal-ion (tuple) positions; `_binom_log10_survival` vectorised (~5x faster, bit-identical).
* `hyperscore` documented as X!Tandem-*style* and intensity-scale-dependent (math unchanged).
* Histogram noise estimation now bins the low-intensity bulk; binning the full dynamic range put
  every noise peak in bin 0 and overestimated the level by roughly two orders of magnitude. All
  estimators return `0.0` on an empty array instead of NaN or `IndexError`.

### Fixes — mzML / spectrl interoperability
* **Swapped scan-window accessions.** `MS:1000500` is "scan window *upper* limit" and `MS:1000501`
  is the lower limit; these were reversed. Round-trips were symmetric so tests passed, but emitted
  tokens told external mzML consumers the window was inverted.
* Precursor ion mobility uses the scalar ion-selection terms `MS:1002815` / `MS:1002476` rather than
  a binary-data-array accession (legacy accessions still decode).
* `MS:1002481` decodes to `HCD` rather than `PASEF`; `MS:1003007` no longer claims to mean CCS.
* `injection_time` and `Precursor.is_monoisotopic` now survive the round-trip; isolation window and
  activation attach only to the first precursor.
* Readers emit the canonical enums, so `spec.activation_type == ActivationType.CID` is now true for
  reader-produced spectra.
* `MzmlReader` no longer marks ordinary centroid data as `DECONVOLUTED` merely because it carries a
  charge array.
* `fetch_usi` validates the USI locally before any network call, resolves precursor m/z by fixed
  accession precedence rather than server ordering, and retains `scan_number` / `native_id`.
* `MzmlReader.open()` no longer orphans an existing handle; `DReader` no longer leaks a sqlite
  connection per instance; `Reader` accepts uppercase `.D` and `.mzML.gz`.
* Optional backends that fail with `OSError` (broken native library) no longer break
  `import spxtacular`.

### Packaging
* **The sdist is 74 KB, down from 91 MB.** It had swept in 74 MB of Bruker test fixtures and both
  generated plot directories, leaving releases one fixture away from PyPI's 100 MB limit. Tests are
  no longer shipped.
* Publishing uses PyPI trusted publishing (OIDC) instead of a long-lived API token.
* CI gained a `pull_request` trigger (packaging changes previously got no check at all), a 3.12/3.13
  matrix, and a job that installs **without** extras to test that the readers stay importable —
  the load-bearing promise in `CLAUDE.md`, previously untested.
* `[tool.uv] dev-dependencies` moved to PEP 735 `[dependency-groups]`; `Typing :: Typed` classifier
  added; PEP 639 license metadata; pytest/coverage config added with `filterwarnings = ["error"]`.
* Removed the committed `junit.xml`, the stray `try.py`, the redundant `plots/` directory, the
  vestigial `MANIFEST.in`, and the dead `draft-pdf.yml` workflow.

### Tests
* `test_numba_fallback.py` restored modules by rewriting `sys.modules`, which does nothing after
  `importlib.reload` mutates the module dict in place — so **every test collected after it ran the
  pure-Python path**, and the JIT path was silently uncovered. Fixed, with a new test asserting the
  two backends produce identical results and a guard against the leak recurring.

### Docs
* The README and docs landing-page quick starts produced an **empty spectrum** (the `.denoise()`
  threshold exceeded every peak in the sample data). Replaced with realistic data and real output.
* The deconvolution basic-usage and worked examples showed output that did not match what the code
  produces; both are now generated from actual runs, and the algorithm description covers the new
  anchor search.
* Fixed API references that raised on copy-paste: `reader.aquisition_type` (misspelled),
  `mirror_plot(raw, decon=...)` (the parameter is `deconvoluted`), `next(reader.ms2)` (the lookup
  objects are iterable but not iterators), `facet_plot(spectra)` in `llms.txt`, and a `decharge()`
  call on non-deconvoluted data that violated the docs' own guidance.
* `Spectrum` constructor signature now lists `iso_score`; plotly and pandas correctly documented as
  required rather than optional; `docs/scoring.md` added to the nav; dead anchors repointed.

## 0.4.0 (2026-07-09)

### Breaking changes
* **Removed `spxtacular.compress` and `spxtacular.urlparams`.** The in-house hex-delta + gzip wire format and the URL query-param encoder have been deleted in favour of the [spectrl](https://github.com/tacular-omics/spectrl) token format. Removed APIs:
  * `Spectrum.compress()` / `Spectrum.from_compressed()`
  * `Spectrum.to_url_params()` / `Spectrum.from_url_params()`
  * `spxtacular.spectrum_to_query_params` / `spectrum_to_query_string` / `spectrum_from_query_params`
  * The `compress_spectra` / `decompress_spectra` functions
* `.npz` persistence (`Spectrum.save` / `Spectrum.load`) is unchanged.

### Migration
* Replace `spec.compress()` and `spec.to_url_params()` calls with `spec.to_spectrl_token()`.
* Replace `Spectrum.from_compressed(s)` and `Spectrum.from_url_params(p)` with `Spectrum.from_spectrl_token(t)`.
* The spectrl token format is mzML-faithful (PSI-MS CV params, a single CBOR document, MS-Numpress compression, SHA-256 integrity hash) and well-suited for embedding in URLs, QR codes, notebooks, and papers.

### New features
* **`spxtacular.spectrl_bridge`** — encode/decode `Spectrum` / `MsnSpectrum` to/from spectrl tokens.
  * `Spectrum.to_spectrl_token(*, lossless=False, max_len=None)` — encode to a `spectrl1.…` token.
  * `Spectrum.from_spectrl_token(token)` — classmethod decode.
  * Standalone helpers: `to_inline_spectrum`, `to_spectrl_token`, `from_spectrl_token`, and `spxtacular.spectrl_bridge.from_decoded_spectrum`.
* **URL sharing helpers** — `Spectrum.to_spectrl_url(base, *, mode="fragment"|"query"|"data", param="d", …)` and `Spectrum.from_spectrl_url(url)` (plus standalone `to_spectrl_url` / `from_spectrl_url`) build and parse a shareable URL or `data:` URI in one call, replacing the removed `urlparams` convenience.
* **`iso_score` is preserved** through the round-trip via spectrl's `extra_arrays` slot (encoded as a non-standard mzML binary array, `MS:1000786`). Other tools that don't recognise the array name ignore it cleanly.
* **Lossless scalar-metadata round-trip** — spxtacular fields without an mzML CV counterpart (`denoised`, `normalized`, `scan_number`, `resolution`, `analyzer`, `ramp_time`, `im_range`, `isolation_im_range`) are carried as namespaced (`spxtacular:`) free-text `user_params`, so the round-trip is faithful.
* **Typed metadata enums** — new `StrEnum`s `Polarity`, `ActivationType`, `IMType`, and `Analyzer` (exported from `spxtacular`) give autocomplete and typo-safety when hand-authoring an `MsnSpectrum` (`activation_type=ActivationType.HCD`). The fields stay open vocabularies (`ActivationType | str`, etc.), so raw PSI-MS accessions from `DReader` (`"MS:1002481"`) and unknown vendor strings still flow through untouched. `ActivationType`/`Analyzer` are the single source of truth for `spectrl_bridge`'s `_ACTIVATION_ACCESSIONS` / new `_ANALYZER_ACCESSIONS` PSI-MS accession maps (keyed by enum member), so the acronym list and its accessions can no longer drift apart.
* spectrl is gated behind the `[spectrl]` optional extra, sourced from PyPI (`spectrl>=0.2.1`). The token is a single CBOR document; 0.2.1 fixes a native abort when lossy-encoding charge arrays that contain singleton sentinels (`charge=-1`).

### Fixes
* `Spectrum.match_fragments()` / `Spectrum.score()` default `tolerance_type` reverted to `DA` (it had drifted to `PPM` while keeping `tolerance=0.02`, making the default call match almost nothing).
* `plot_spectrum()` / `Spectrum.plot()`: `color`, `show_scores`, and `show_charges` are now keyword-only, closing off a silent-positional-argument hazard introduced when `color` was inserted ahead of the old `show_charges` slot.
* `match_fragments()` no longer raises `ZeroDivisionError` when a fragment's target mass is exactly `0.0` under `tolerance_type="ppm"`; the dict-fragment branch also builds `Fragment` objects lazily again (only for confirmed matches), restoring the pre-rewrite performance on this per-PSM hot path.
* New `Spectrum.is_decharged` property replaces three separate inline re-derivations of the same check (`core.py`, `matching.py`).
* `Spectrum.decharge()` now warns instead of silently zeroing every m/z value when called on an already-decharged spectrum (see the aliasing fix below for what it returns).
* `Spectrum.top_peaks(0)` / `Spectrum.filter(top_n=0)` now correctly return zero peaks instead of all of them (a `arr[-0:]` negative-zero slicing bug).
* `Spectrum.merge(im_tolerance_type=...)` validation was case-insensitive but the comparison wasn't, so e.g. `"RELATIVE"` silently used absolute-tolerance semantics.
* `MzmlReader` spectra with multiple ion-mobility arrays now use the first length-matching array (previously the loop kept overwriting its result and could end up using the last array, or none, contradicting its own warning).
* `DReader.close()` now clears its internal reader handle so the "must be opened" guard can't be bypassed by using a closed reader; `DReader.open()` now closes a previously-open reader instead of leaking its handle on re-open.
* `_plot_spectrum_im` (the `color="im"` plot path) no longer corrupts the whole color scale when a single peak's `im` is `NaN`, and no longer crashes on an empty spectrum.
* `spectrl_bridge`: unrecognised `im_type` strings now round-trip losslessly via a namespaced `spxtacular:im_type` user_param instead of being silently coerced to a default accession; unrecognised `activation_type` strings now round-trip losslessly via a new `spxtacular:activation_type` user_param instead of raising an `IndexError` (they were previously passed to spectrl as a raw, non-`MS:NNNNN` CV accession, which crashed encode); `Precursor.im` is now carried through encode/decode; an `MsnSpectrum` whose only MSn-specific data is `im`/`im_type`/`activation_type` no longer downgrades to a plain `Spectrum` on decode. `_ACTIVATION_ACCESSIONS` (`Spectrum.activation_type` → PSI-MS CV accession) expanded from 5 to 15 entries — added `EThcD`, `ETciD`, `NETD`, `UVPD`, `PD`, `PQD`, `SID`, `IRMPD`, `BIRD`, `SORI` — and its `"PASEF"` entry's comment now clarifies it maps to the "higher energy beam-type CID" accession (`MS:1002481`), since PASEF is a Bruker acquisition scheme with no PSI-MS term of its own, not a mislabelled CV term.
* `uv run ty check` no longer errors under `tdfpy>=2.0.0` (which ships a `py.typed` marker): the `MergePeaksCentroider` import-fallback assignment was missing a `ty: ignore[invalid-assignment]` alongside its existing `type: ignore`, so `ty` now flagged it as a real type error once it could resolve the import. Verified against the currently-released `tdfpy==2.0.0`/`mzmlpy==0.4.0` and, ahead of their releases, the local `tdfpy` (`release/v2.1.0`) and `mzmlpy` (`release/v0.5.0`) branches — full test suite passes against both.
* `Spectrum.decharge(inplace=False)`, `.normalize(inplace=False)`, `.denoise(inplace=False)`, `.centroid(inplace=False)`, and `.deconvolute(inplace=False)` called on a spectrum already in the target state now return a distinct object instead of aliasing `self`; previously the caller's original spectrum could be silently mutated through the "new" object.
* `spxtacular.score()`, `match_fragments()`, `annotate_spectrum()`, `Spectrum.mass_error_plot()`/`facet_plot()`, and their standalone `visualization` counterparts now default `tolerance_type` to `DA`, matching `Spectrum.score()`/`Spectrum.match_fragments()` — these had drifted to `PPM` in some entry points but not others, so calling different parts of the API with default arguments on the same inputs silently produced very different match counts.
* Removed a stale, now-unused `ty: ignore[unresolved-import]` comment in `reader.py` flagged by `ty check`.
* `Spectrum.decharge()` now raises `ValueError` when called on a non-deconvoluted spectrum, matching its documented contract, instead of silently calling `deconvolute()` with hidden default parameters. Call `deconvolute()` explicitly first.
* `spectrl_bridge`: an `activation_type` that is already a valid PSI-MS accession (e.g. `"MS:1002481"` / `"MS:1000133"`, as both `DReader` and `MzmlReader` produce) is once again emitted as a standard dissociation-method CV param on the encoded precursor, not just carried in the `spxtacular:activation_type` user_param; only genuinely free-text vendor strings fall through to the user_param. Restores mzML CV fidelity for reader-produced MS2 spectra while keeping the encode crash-safe.
* Scored deconvolution (`deconvolute(min_score=…, intensity="total")`): a rejected multi-peak cluster no longer records the *whole cluster's* summed intensity on its seed singleton — it records the seed's own intensity, so the remaining cluster peaks (which stay available and are re-seeded later) are no longer double-counted in the output. Only affected `min_score > 0` in the default `"total"` intensity mode; `"base"` mode and the default `min_score=0.0` were already correct.
* `Spectrum.deconvolute()` now validates `charge_range` (must be `(min, max)` with `1 <= min <= max`) with a clear `ValueError` instead of a downstream divide-by-zero, and returns an empty deconvoluted spectrum for an empty input instead of raising on the `min()` of an empty intensity array.
* `Spectrum.normalize()` on an all-zero-intensity spectrum now warns and returns the spectrum unchanged instead of dividing by zero and silently producing `NaN`s; on an empty spectrum it is a no-op instead of raising.

### Dependencies
* Added an explicit `numpy>=1.26` floor (the first numpy supporting Python 3.12, the project's minimum) — previously `numpy` was unbounded.
* Raised dependency floors to the versions 0.4.0 is tested against: `peptacular>=3.1.2`, `paftacular>=1.1.0`, and the optional reader extras `tdfpy>=2.0.0` (was `>=1.2.0` — now requires the tdfpy 2.x API) and `mzmlpy>=0.5.0` (was `>=0.4.0`). `spectrl>=0.2.1` unchanged. Verified: full suite passes against `peptacular==3.1.2`, `paftacular==1.1.0`, `tdfpy==2.2.0`, `mzmlpy==0.5.0`.

## 0.3.1 (2026-05-14)

### New features
* **Optional reader backends** — `tdfpy` and `mzmlpy` moved to optional extras (`bruker`, `mzml`, `readers`, `all`). `DReader` and `MzmlReader` remain importable from `spxtacular` regardless of whether their backends are installed; only instantiation raises `ImportError` pointing to the correct extra. Lets downstream consumers (e.g. `pydiode`) depend on `spxtacular` for the core processing API without pulling in native reader deps.
* **Ion-mobility plotting** — `plot_spectrum()` and `Spectrum.plot()` gained a `color` parameter (`"im" | "charge" | None`). New internal `_plot_spectrum_im` renders sticks with a quantized Viridis scale and IM colorbar; respects `Spectrum.im_type` (e.g. `ook0` → `1/K0`).
* **URL query-param encoding** — new `spxtacular.urlparams` module (`spectrum_to_query_params`, `spectrum_to_query_string`, `spectrum_from_query_params`, plus `Spectrum.to_url_params` / `Spectrum.from_url_params`) for round-tripping `Spectrum`/`MsnSpectrum` through URL query strings. Peak arrays are routed through the binary compressor with `url_safe=True`; MSn scalar metadata is emitted as plain, human-readable params. Wire format is versioned (`version=1`).
* **Compressor carries per-peak `iso_score`** — `Spectrum.compress()` / `compress_spectra()` gained an `iso_score_precision` kwarg and now encode `score` as an optional 5th length-prefixed chunk. Payloads without `iso_score` stay byte-identical to the previous format, so old payloads remain decodable and pre-0.3.1 decoders can still read new payloads (they stop after the 4th chunk).

### API changes
* `Spectrum.match_fragments()` default `tolerance_type` changed from `DA` to `PPM`. New `is_monoisotopic: bool = True` parameter forwarded to the underlying `match_fragments` function.

### Dependencies
* `tdfpy>=1.2.0` (was `>=1.1.0`) — now under the `bruker` extra.
* `paftacular` extra dropped: now `paftacular>=1.0.0` (was `paftacular[peptacular]>=1.0.0`) for micropip/Pyodide compatibility. Core `paftacular` is sufficient — only `pft.to_mzpaf` is used.

### Fixes
* `Spectrum.compress()` no longer crashes on deconvoluted spectra containing singletons (`charge=-1`). The wire format now reserves hex `'f'` for the singleton sentinel; charge state 15 is no longer supported (charge states above 14 are vanishingly rare).
* `plot_spectrum(show_charges=...)` now warns and forwards to the new `color=` argument instead of failing inside plotly. Tests and call-sites should migrate to `color="charge"` / `color=None`.
* `MsnSpectrum.__str__` no longer raises `TypeError` when `rt` is `None`.
* `match_fragments()` (and therefore `Spectrum.annotate()` / `annotate_spectrum()`) now adapts to the spectrum's processing state. Singletons (`charge == -1`, unknown charge) act as a wildcard and match by m/z. Decharged spectra (every peak's `charge == 0`, m/z values are neutral masses) match fragments against `Fragment.neutral_mass` instead of `Fragment.mz`, so any `charge_state` fragment can match the same neutral peak. Previously these states silently produced zero matches.

### Internal
* `Spectrum.save`/`load` and `MsnSpectrum.save`/`load` unified onto base-class implementations with `_meta_dict()` / `_meta_kwargs()` hooks. The persisted `iso_score` array is now stored under the `iso_score` key in `.npz` files (was `score` on the base `Spectrum`); the loader transparently falls back to the old `score` key for backward compatibility.
* `DReader` now works against both the current `tdfpy` 1.2.0 release and the post-1.2.0 smoothing branch, which reshaped `Frame.centroid()` / `DiaWindow.centroid()` / `PrmTransition.centroid()` to take `centroid=MergePeaksCentroider(…)` and `noise=…` instead of the older flat keyword args. The adapter is transparent: `CentroidConfig` is unchanged.

## 0.3.0 (2026-04-07)

### New features
* **PRM support** — `DReader` now opens PRM `.d` folders via the dedicated `tdfpy.PRM` reader. MS2 iteration yields one `MsnSpectrum` per `PrmTransition` (frame × target slice), with target metadata exposed via the `precursors` field and isolation window/collision energy populated from the transition. Native ID format is `"{frame_id}@t{target_id}"`. PRM MS2 lookup by integer ID raises `NotImplementedError` (transitions are keyed by `(frame_id, target_id)`).
* **USI loading** — `Spectrum.from_usi()` and the underlying `spxtacular.usi.fetch_usi` retrieve spectra from public repositories (PRIDE, MassIVE, PeptideAtlas, jPOST, or the PROXI aggregator) by Universal Spectrum Identifier. Returns an `MsnSpectrum` when precursor info is present, otherwise a plain `Spectrum`.
* **Precursor peak removal** — `Spectrum.remove_precursor_peak()` strips the precursor, its isotope envelope, and (optionally) all charge states from 1..z. Adapts to centroid / deconvoluted / decharged spectra; auto-detects precursors from `MsnSpectrum.precursors` when no explicit `precursor_mz` is given.
* **Intensity scaling** — `Spectrum.scale_intensity(method="root"|"log"|"rank")` for dynamic-range compression. Independent of `normalize()` (which divides by a reference value).
* **m/z rounding** — `Spectrum.round_mz(decimals, combine="sum"|"max")` rounds m/z values and merges duplicates with sum or max-intensity reduction.
* **New visualisations** — `mass_error_plot()` (bubble chart of fragment mass errors vs m/z, sized by intensity, coloured by ion series) and `facet_plot()` (multi-panel spectrum + mass-error + optional mirror). Convenience methods on `Spectrum` (`.mass_error_plot()`, `.facet_plot()`) included.
* **Unit conversion** — `spxtacular.da_to_ppm` and `spxtacular.ppm_to_da` helpers.
* **`Spectrum.annotate()`** — convenience method calling `annotate_spectrum` for fragment-labelled plots.

## 0.2.0 (2026-03-18)

### New features
* **Plot table API** — `build_plot_table()`, `build_annot_plot_table()`, `plot_from_table()` provide an intermediate `pandas.DataFrame` layer between data and plotting.  Users can freely modify colours, line widths, labels, and font settings before rendering.
* `Spectrum.plot_table()` and `Spectrum.annot_plot_table()` convenience methods.
* **Scored deconvolution** — `Spectrum.deconvolute()` now uses Bhattacharyya-coefficient isotope-profile scoring; peaks carry a `score` array (0–1).
* `Spectrum.filter(min_score=, max_score=)` for quality-based peak filtering.
* `min_intensity` and `min_score` parameters added to `Spectrum.deconvolute()`.
* **Fragment matching** — `match_fragments()` supports charge-state filtering when the spectrum has a `charge` array; singletons (`charge == -1`) are excluded.
* **PSM scoring** — `score()` function with eight metrics: `hyperscore`, `probability_score`, `total_matched_intensity`, `matched_fraction`, `intensity_fraction`, `mean_ppm_error`, `spectral_angle`, `longest_run`.
* `mirror_plot()` and `annotate_spectrum()` added to the visualization module.
* `show_scores` parameter added to `plot_spectrum()` and `Spectrum.plot()`.
* **Persistence** — `Spectrum.save()` / `Spectrum.load()` and `MsnSpectrum.save()` / `MsnSpectrum.load()` round-trip spectra to compact `.npz` files. Arrays are stored natively; scalar metadata (and `MsnSpectrum.precursors`) is serialised as JSON under a `meta` key.
* **`Spectrum.combine()`** — classmethod that concatenates peaks from multiple spectra into a single new `Spectrum`, m/z-sorted. Optional per-peak arrays (`charge`, `im`, `iso_score`) carry over only when **all** input spectra provide them.
* **`Spectrum.merge()`** — greedy intensity-ordered peak merging. Replaces the older single-tolerance signature with split `mz_tolerance` / `mz_tolerance_type` and `im_tolerance` / `im_tolerance_type` (relative or absolute) kwargs; charge-aware (only peaks of matching charge are merged).
* **Unified `Reader`** — `spxtacular.Reader(path)` auto-detects `.d` vs `.mzML` and dispatches to `DReader` or `MzmlReader`.

### Dependencies
* `pandas>=2.0` added as a runtime dependency.

### Fixes & polish
* Missing `stacklevel` added to all `warnings.warn()` calls in `reader.py`.
* Dead code (`_fragment_label` in `visualization.py`) removed.
* `pyproject.toml`: added `[project.urls]`, `license`, and `keywords` fields.
* `LICENSE` (MIT) file added to repository root.

## 0.1.0 (2026-01-16)

* First release on PyPI.
