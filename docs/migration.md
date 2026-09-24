# Migrating from 0.8 to 0.9

spxtacular 0.9 is a breaking release. It moves to the new majors of its sibling packages
(tacular 2, peptacular 5, paftacular 2, tdfpy 5, mzmlpy 0.10), renames a few fields so each
name says exactly what it holds, and makes optional parameters keyword-only. This page lists
every rename and removal as **old -> new**.

## Dependencies

| 0.8 | 0.9 |
|---|---|
| `peptacular>=4.2,<5` | `peptacular>=5,<6` |
| `paftacular>=1.4,<2` | `paftacular>=2,<3` |
| (transitive) `tacular` | `tacular>=2,<3` (direct) |
| `tdfpy>=4.1,<5` (`[bruker]`) | `tdfpy>=5,<6` |
| `mzmlpy>=0.9,<0.10` (`[mzml]`) | `mzmlpy>=0.10,<0.11` |

peptacular 5 changes affect code that builds fragments to pass in: `ion_types="by"` is now
**one** ion type named `"by"` (an internal fragment), so write `ion_types=("b", "y")`; and
`Fragment.losses` is now `Fragment.deltas`.

## Precursor

`Precursor` is now its own frozen, slotted, keyword-only dataclass. It is no longer a `Peak`
subclass.

| 0.8 | 0.9 |
|---|---|
| `Precursor(mz=500.2, ...)` | `Precursor(precursor_mz=500.2, ...)` |
| `Precursor(500.2, 1e4, 2, ...)` (positional) | keyword-only: `Precursor(precursor_mz=500.2, intensity=1e4, charge=2)` |
| `precursor.mz` | `precursor.precursor_mz` |
| `is_monoisotopic` required | `is_monoisotopic=None` by default; `intensity=0.0` by default |
| `isinstance(precursor, Peak)` is `True` | `False` |
| (none) | `precursor.im_type`: the kind of mobility stored in `precursor.im` |

`Precursor.im` keeps its name because the value is whatever the source recorded: 1/K0 for
Bruker and most mzML files, drift time for some mzML files. `im_type` (an `IMType`, or
`None` when unknown) says which. Readers fill it in: Bruker DDA and PRM set `"ook0"`; mzML
sets `"ook0"` for inverse reduced mobility and `"drift_time_ms"` for drift time.

## MsnSpectrum

| 0.8 | 0.9 |
|---|---|
| `MsnSpectrum(isolation_im_range=...)` | `MsnSpectrum(isolation_ook0_range=...)` |
| `spec.isolation_im_range` | `spec.isolation_ook0_range` |

The isolation window is always 1/K0 (it comes from Bruker's quadrupole/TIMS isolation), so
the name now says so.

## Spectrum and Peak constructors

Everything after `intensity` is keyword-only:

| 0.8 | 0.9 |
|---|---|
| `Spectrum(mz, intensity, charge)` | `Spectrum(mz, intensity, charge=charge)` |
| `Peak(500.0, 1e4, 2)` | `Peak(500.0, 1e4, charge=2)` |

## Keyword-only parameters

Required parameters stay positional. Every parameter with a default is now keyword-only in the
public functions and methods below. `spec.normalize("tic")` becomes
`spec.normalize(method="tic")`; `match_fragments(spec, frags, 10, "ppm")` becomes
`match_fragments(spec, frags, tolerance=10, tolerance_type="ppm")`.

| Function or method | Now keyword-only |
|---|---|
| `Spectrum.filter` | all parameters |
| `Spectrum.normalize`, `Spectrum.denoise` | `method`, `inplace` |
| `Spectrum.centroid` | `min_intensity`, `inplace` |
| `Spectrum.merge` | `mz_tolerance`, `mz_tolerance_type`, `im_tolerance`, `im_tolerance_type`, `inplace` |
| `Spectrum.sort`, `Spectrum.top_peaks` | `by`, `reverse` (and `inplace` for `sort`) |
| `Spectrum.update`, `Spectrum.decharge` | `inplace` |
| `Spectrum.deconvolute` | all parameters |
| `Spectrum.remove_precursor_peak` | all parameters, including `precursor_mz` and `precursor_charge` |
| `Spectrum.scale_intensity` | `method`, `degree`, `base`, `inplace` |
| `Spectrum.round_mz` | `decimals`, `combine`, `inplace` |
| `Spectrum.has_peak`, `get_peak`, `get_peaks` | `tolerance`, `tolerance_type`, `target_charge`, `target_im`, `im_tol` (and `collision` for `get_peak`) |
| `Spectrum.match_fragments`, `match_fragments` | `tolerance`, `tolerance_type`, `peak_selection`, `is_monoisotopic` |
| `Spectrum.score`, `score` | `tolerance`, `tolerance_type`, `peak_selection`, `predicted_intensities` |
| `Spectrum.annotate`, `Spectrum.annot_plot_table` | `tolerance`, `tolerance_type`, `title`, `peak_selection`, `include_sequence` |
| `Spectrum.plot`, `Spectrum.plot_table` | `title`; `show_charges`, `show_scores` |
| `Spectrum.mass_error_plot`, `Spectrum.facet_plot` | all optional parameters, including `fragments` and `mirror_spectrum` for `facet_plot` |
| `Spectrum.to_spectrl_url`, `to_spectrl_url` | `base` |
| `Spectrum.from_usi`, `fetch_usi` | `backend`, `timeout` |
| `cosine`, `modified_cosine`, `entropy_similarity` | `tolerance`, `tolerance_type` (and `transform`); the two precursor m/z of `modified_cosine` stay positional |
| `extract_chromatogram` | `mode`, `mz_range` |
| `extract_xic`, `plot_xic` | `tolerance`, `tolerance_type`, `im_window`, `aggregate` (and plot options) |
| `brain_isotopic_distribution` | `max_isotopes`, `isotope_abundances` |
| `IsotopeModel.adaptive_distribution` | `min_relative_abundance`, `max_isotopes` |
| `estimate_noise_level` | `method` |
| `build_plot_table`, `build_annot_plot_table`, `plot_from_table`, `table_view` | all optional parameters |
| `plot_spectrum`, `annotate_spectrum`, `mirror_plot`, `plot_chromatogram` | all optional parameters |
| `profile_centroid_plot` | `centroids`, `title`, `theme_mode`, `max_points` |
| `sequence_coverage_plot`, `mass_error_plot`, `facet_plot` | all optional parameters, including `fragments` for `facet_plot` |
| `save_figure` | `scale` |
| `Reader`, `DReader` | `centroid_config` |
| `ThermoReader` | `prefer_vendor_centroid` |
| `CentroidConfig` | all fields |
| `MatchedFragment` | all fields (it is built by `match_fragments`, not by hand) |

`IsotopeModel.distribution(mass, max_isotopes)` and `apex_index` are unchanged.

## Readers

| 0.8 | 0.9 |
|---|---|
| `MzmlReader(..., extract_dir=...)` | removed (mzmlpy 0.10 no longer extracts gzip files to disk) |
| `Reader(..., mzml_extract_dir=...)` | removed |
| `gzip_mode="extract"` | removed; use `"auto"` (default), `"indexed"` or `"stream"` |
| `access_strategy == "extracted"` | no longer returned; `"memory"` or `"stream"` are new values |
| mzML `scan_number` = 0-based spectrum index | the `scan=` value of the native id (`"scan=19"` -> `19`), or `None` when the id has no `scan` key |
| `AcquisitionType.UNKNOWN == "UNKNOWN"` (values `"DDA"`, ...) | `AcquisitionType` is tdfpy's enum; compare members, not strings |
| lookup before `open()` raises `RuntimeError` | raises `SpxtacularError` |
| `CentroidConfig` applied to DDA MS2 | DDA MS2 spectra use tdfpy's per-precursor merged peaks; `CentroidConfig` affects MS1, DIA and PRM |

New in 0.9: `DReader` MS1 spectra carry `total_ion_current`, and the reader lookup types
(`DReaderMs1Lookup`, `DReaderMs2Lookup`, `MzmlSpectraLookup`, `ThermoScanLookup`,
`PeakListLookup`) and a `SpectrumLookup` protocol are exported from the package root, so
`Reader.ms1` / `.ms2` can be type-annotated.

## Transforms that are already done

In 0.8, calling `normalize`, `denoise`, `centroid`, `deconvolute` or `decharge` on a
spectrum that had already been through that step warned and returned it unchanged. In 0.9:

| Method | 0.9 behaviour |
|---|---|
| `normalize` | always rescales with the requested method (no warning) |
| `denoise`, `centroid`, `deconvolute`, `decharge` | silent no-op: returns a copy, or `self` when `inplace=True` |

Code that wrapped these calls in `pytest.warns` or `warnings.catch_warnings` can drop that.

## Errors

| 0.8 | 0.9 |
|---|---|
| `ValueError` for invalid input | `SpxtacularError`, a `ValueError` subclass: `except ValueError` still works |
| `da_to_ppm(delta, 0)` raises `ValueError` | raises tacular's `TacularError` (also a `ValueError`) |
| `da_to_ppm(delta, mz)` divides by `mz` | divides by `abs(mz)`, so a negative reference keeps the error's sign |

Enum coercion errors (for example an unknown `tolerance_type`) are still plain `ValueError`.

## MatchedFragment

`MatchedFragment` is frozen, slotted and keyword-only. Its fields are unchanged. The new
`annotation` property returns the match as a paftacular `PafAnnotation` (with the mass error
in ppm), built on first access and cached.

## JSON

| 0.8 | 0.9 |
|---|---|
| spectrum `schema_version` 1 | 2 (0.9 still reads version 1) |
| precursor key `mz` | `precursor_mz`; new `im_type` key |
| metadata key `isolation_im_range` | `isolation_ook0_range` |
| `spxtacular/schemas/spectrum-v1.schema.json` | `spxtacular/schemas/spectrum-v2.schema.json` |

The chromatogram schema stays at version 1. spectrl tokens and URLs keep their wire keys, so
tokens written by 0.8 decode in 0.9.

## Fragment labels

Plot labels come straight from paftacular 2's mzPAF writer, including negative charges
(`y3^-2`). The `include_sequence` option controls whether the peptide sequence is part of the
label.
