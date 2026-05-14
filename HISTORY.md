# History

## 0.3.1 (2026-05-14)

### New features
* **Optional reader backends** — `tdfpy` and `mzmlpy` moved to optional extras (`bruker`, `mzml`, `readers`, `all`). `DReader` and `MzmlReader` remain importable from `spxtacular` regardless of whether their backends are installed; only instantiation raises `ImportError` pointing to the correct extra. Lets downstream consumers (e.g. `pydiode`) depend on `spxtacular` for the core processing API without pulling in native reader deps.
* **Ion-mobility plotting** — `plot_spectrum()` and `Spectrum.plot()` gained a `color` parameter (`"im" | "charge" | None`). New internal `_plot_spectrum_im` renders sticks with a quantized Viridis scale and IM colorbar; respects `Spectrum.im_type` (e.g. `ook0` → `1/K0`).
* **URL query-param encoding** — new `spxtacular.urlparams` module (`spectrum_to_query_params`, `query_params_to_spectrum`) for round-tripping `Spectrum`/`MsnSpectrum` through URL query strings. Peak arrays are routed through the binary compressor with `url_safe=True`; MSn scalar metadata is emitted as plain, human-readable params. Wire format is versioned (`version=1`).
* **Compressor carries per-peak `iso_score`** — `Spectrum.compress()` / `compress_spectra()` gained an `iso_score_precision` kwarg and now encode `score` as an optional 5th length-prefixed chunk. Payloads without `iso_score` stay byte-identical to the previous format, so old payloads remain decodable and pre-0.3.1 decoders can still read new payloads (they stop after the 4th chunk).

### API changes
* `Spectrum.match()` default `tolerance_type` changed from `DA` to `PPM`. New `is_monoisotopic: bool = True` parameter forwarded to `match_fragments`.

### Dependencies
* `tdfpy>=1.2.0` (was `>=1.1.0`) — now under the `bruker` extra.
* `paftacular` extra dropped: now `paftacular>=1.0.0` (was `paftacular[peptacular]>=1.0.0`) for micropip/Pyodide compatibility. Core `paftacular` is sufficient — only `pft.to_mzpaf` is used.

## 0.3.0 (2026-04-07)

### New features
* **PRM support** — `DReader` now opens PRM `.d` folders via the dedicated `tdfpy.PRM` reader. MS2 iteration yields one `MsnSpectrum` per `PrmTransition` (frame × target slice), with target metadata exposed via the `precursors` field and isolation window/collision energy populated from the transition. Native ID format is `"{frame_id}@t{target_id}"`. PRM MS2 lookup by integer ID raises `NotImplementedError` (transitions are keyed by `(frame_id, target_id)`).

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

### Dependencies
* `pandas>=2.0` added as a runtime dependency.

### Fixes & polish
* Missing `stacklevel` added to all `warnings.warn()` calls in `reader.py`.
* Dead code (`_fragment_label` in `visualization.py`) removed.
* `pyproject.toml`: added `[project.urls]`, `license`, and `keywords` fields.
* `LICENSE` (MIT) file added to repository root.

## 0.1.0 (2026-01-16)

* First release on PyPI.
