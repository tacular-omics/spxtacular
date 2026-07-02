# History

## 0.4.0 (2026-06-28)

### Breaking changes
* **Removed `spxtacular.compress` and `spxtacular.urlparams`.** The in-house hex-delta + gzip wire format and the URL query-param encoder have been deleted in favour of the [spectrl](https://github.com/pgarrett-scripps/spectrl) token format. Removed APIs:
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
* spectrl is gated behind the `[spectrl]` optional extra, sourced from PyPI (`spectrl>=0.2.1`). The token is a single CBOR document; 0.2.1 fixes a native abort when lossy-encoding charge arrays that contain singleton sentinels (`charge=-1`).

### Fixes
* `Spectrum.match_fragments()` / `Spectrum.score()` default `tolerance_type` reverted to `DA` (it had drifted to `PPM` while keeping `tolerance=0.02`, making the default call match almost nothing).
* `plot_spectrum()` / `Spectrum.plot()`: `color`, `show_scores`, and `show_charges` are now keyword-only, closing off a silent-positional-argument hazard introduced when `color` was inserted ahead of the old `show_charges` slot.
* `match_fragments()` no longer raises `ZeroDivisionError` when a fragment's target mass is exactly `0.0` under `tolerance_type="ppm"`; the dict-fragment branch also builds `Fragment` objects lazily again (only for confirmed matches), restoring the pre-rewrite performance on this per-PSM hot path.
* New `Spectrum.is_decharged` property replaces three separate inline re-derivations of the same check (`core.py`, `matching.py`).
* `Spectrum.decharge()` now warns and returns the original spectrum instead of silently zeroing every m/z value when called on an already-decharged spectrum.
* `Spectrum.top_peaks(0)` / `Spectrum.filter(top_n=0)` now correctly return zero peaks instead of all of them (a `arr[-0:]` negative-zero slicing bug).
* `Spectrum.merge(im_tolerance_type=...)` validation was case-insensitive but the comparison wasn't, so e.g. `"RELATIVE"` silently used absolute-tolerance semantics.
* `MzmlReader` spectra with multiple ion-mobility arrays now use the first length-matching array (previously the loop kept overwriting its result and could end up using the last array, or none, contradicting its own warning).
* `DReader.close()` now clears its internal reader handle so the "must be opened" guard can't be bypassed by using a closed reader; `DReader.open()` now closes a previously-open reader instead of leaking its handle on re-open.
* `_plot_spectrum_im` (the `color="im"` plot path) no longer corrupts the whole color scale when a single peak's `im` is `NaN`, and no longer crashes on an empty spectrum.
* `spectrl_bridge`: unrecognised `activation_type` and `im_type` strings now round-trip losslessly instead of being silently coerced to a default accession; `Precursor.im` is now carried through encode/decode; an `MsnSpectrum` whose only MSn-specific data is `im`/`im_type` no longer downgrades to a plain `Spectrum` on decode.

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
