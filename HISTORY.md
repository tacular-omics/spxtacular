# History

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
