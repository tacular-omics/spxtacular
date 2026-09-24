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

tdfpy 5 changes some Bruker peak values. MS1 spectra from `DReader` can differ by one or two
peaks, with summed intensity about 40 ppm different. Some DDA MS2 peaks that 0.8 merged into one
now come out as separate peaks.

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
`None` when unknown) says which. Readers fill it in: Bruker DDA and PRM set `"ook0"`. mzML
takes it from the declared unit, for precursor values and for ion mobility arrays alike:

| mzML unit | `im_type` | values |
|---|---|---|
| MS:1002814 (volt-second per square centimeter) | `"ook0"` | as stored |
| UO:0000028 (millisecond) | `"drift_time_ms"` | as stored |
| UO:0000010 (second) | `"drift_time_ms"` | multiplied by 1000 |
| no unit | `"im"` (generic) | as stored |

When a spectrum has several ion mobility arrays, the first whose length matches the peaks is
used, with a warning.

## MsnSpectrum

| 0.8 | 0.9 |
|---|---|
| `MsnSpectrum(isolation_im_range=...)` | `MsnSpectrum(isolation_ook0_range=...)` |
| `spec.isolation_im_range` | `spec.isolation_ook0_range` |

The isolation window is always 1/K0 (it comes from Bruker's quadrupole/TIMS isolation), so
the name now says so.

| 0.8 | 0.9 |
|---|---|
| Bruker `isolation_im_range` was `(high, low)`, e.g. `(1.3062, 1.29)` | `isolation_ook0_range` is `(low, high)`: `(1.29, 1.3062)` |

0.8 JSON, npz files and spectrl tokens are read back as `(low, high)`.

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
| `Spectrum.has_peak`, `get_peak`, `get_peaks` | `tolerance`, `tolerance_type`, `target_charge`, `target_im`, `im_tolerance` (and `peak_selection` for `get_peak`) |
| `Spectrum.match_fragments`, `match_fragments` | `tolerance`, `tolerance_type`, `peak_selection`, `is_monoisotopic` |
| `Spectrum.score`, `score` | `tolerance`, `tolerance_type`, `peak_selection`, `predicted_intensities` |
| `Spectrum.annotate`, `Spectrum.annot_plot_table` | `tolerance`, `tolerance_type`, `title`, `peak_selection`, `include_sequence` |
| `Spectrum.plot`, `Spectrum.plot_table` | all parameters |
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
| `spxtacular.decon.deconvolve_spectrum` | everything after `intensity` |
| `Chromatogram` | everything after `intensity` |
| `IsotopeModel` | everything after `atoms_per_da` |
| `IonizationModel` | `carrier` |
| `DeconvolutionProvenance` | everything from `isotope_model_definition` on |

`ParsedUsi` is a `NamedTuple`, so its fields stay positional: tuple unpacking depends on it.

`IsotopeModel.distribution(mass, max_isotopes)` and `apex_index` are unchanged.

## Renamed and removed parameters

| 0.8 | 0.9 |
|---|---|
| `has_peak`/`get_peak`/`get_peaks(im_tol=...)` | `im_tolerance=...`, the name used everywhere else |
| `get_peak(collision="largest"\|"closest")` | `get_peak(peak_selection="largest"\|"closest")`, the name `match_fragments` uses. Default is still `"largest"`; `"all"` raises (use `get_peaks`) |
| `Spectrum.plot(show_charges=...)`, `plot_spectrum(show_charges=...)` | removed (deprecated in 0.8); use `color="charge"` or `color=None` |
| `Spectrum.plot_table(show_charges=...)` | removed; use `color="charge"` or `color=None` |
| `deconvolve_spectrum(..., is_ppm=True)` | `tolerance_type="ppm"` (or `"da"`), as in `Spectrum.deconvolute` |

`build_plot_table(show_charges=...)` and `mirror_plot(show_charges=...)` keep their option: there
it is not a deprecated alias.

## Removed names

| 0.8 | 0.9 |
|---|---|
| `spxtacular.core.JSON_SCHEMA_VERSION`, `spxtacular.serialization.JSON_SCHEMA_VERSION` (`1`) | `spxtacular.serialization.SPECTRUM_SCHEMA_VERSION` (now `2`) |
| `spxtacular.reader.PeakListLookup`, `spxtacular.reader.ThermoScanLookup` | import from the package root: `from spxtacular import PeakListLookup, ThermoScanLookup` |

## Readers

| 0.8 | 0.9 |
|---|---|
| `MzmlReader(..., extract_dir=...)` | removed (mzmlpy 0.10 no longer extracts gzip files to disk) |
| `Reader(..., mzml_extract_dir=...)` | removed |
| `gzip_mode="extract"` | removed; use `"auto"` (default), `"indexed"` or `"stream"` |
| `access_strategy == "extracted"` | no longer returned; `"memory"` or `"stream"` are new values |
| mzML `scan_number` = 0-based spectrum index | the number in the native id when it identifies the spectrum on its own: `scan=19` or Thermo `controllerType=0 controllerNumber=1 scan=19` -> `19`, `index=5` / `spectrum=5` -> `5`. Otherwise `None` (Bruker `frame=… scan=…`, Waters `function=… scan=…`, SCIEX `cycle=…`), because the `scan` value repeats or is missing |
| `write_mgf` wrote no `SCANS` without a scan number | `SCANS` is the 1-based position in the input, and `TITLE` keeps the native id. `write_ms2` also writes the native id as `I NativeID` when it is not `scan=<n>`, and `Ms2Reader` reads it back |
| malformed mzML raised mzmlpy's `MzmlParseError` | raises `SpxtacularError`, with the mzmlpy error as `__cause__` (a missing spectrum is still `KeyError`) |
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
| enum coercion (`ToleranceType("foo")`) raises plain `ValueError` | raises `SpxtacularError`, listing the accepted values. Coercion is case-insensitive (`"PPM"`, `"Positive"`) |
| `im_type`, `polarity` accepted any value | coerced to `IMType` / `Polarity` on construction (`Precursor`, `MsnSpectrum`, JSON); anything else raises `SpxtacularError` |
| `activation_type`, `analyzer` kept any value as given | a member name in any case or a PSI-MS accession becomes the member (`"MS:1002481"` -> `ActivationType.HCD`, `"TOF"` -> `Analyzer.TOF`); other non-blank strings are kept; non-strings and blanks raise |
| a bad JSON payload (wrong types) raised `TypeError` | raises `SpxtacularError` |
| a corrupt or non-spectrum `.npz` raised numpy/zipfile/JSON errors | raises `SpxtacularError`, chained to the original |

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
