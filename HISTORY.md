# History

## Unreleased

### New features
* **PRM support** — `DReader` now opens PRM `.d` folders via the dedicated `tdfpy.PRM` reader. MS2 iteration yields one `MsnSpectrum` per `PrmTransition` (frame × target slice), with target metadata exposed via the `precursors` field and isolation window/collision energy populated from the transition. Native ID format is `"{frame_id}@t{target_id}"`. PRM MS2 lookup by integer ID raises `NotImplementedError` (transitions are keyed by `(frame_id, target_id)`).

### Dependencies
* `tdfpy` is temporarily pinned to a git ref on `main` until a release containing the `PRM` class is cut. `tool.hatch.metadata.allow-direct-references` is enabled to permit this.

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
