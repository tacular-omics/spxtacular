# API Reference

All public names importable from `spxtacular`:

```python
from spxtacular import (
    Spectrum, MsnSpectrum, Peak,
    DReader, MzmlReader,
    match_fragments, score,
    plot_spectrum, mirror_plot, annotate_spectrum,
    build_plot_table, build_annot_plot_table, plot_from_table,
)
```

---

## Core classes

### `Spectrum`

Central data structure for a mass spectrum. Holds parallel numpy arrays for `mz`, `intensity`, and optionally `charge` and `im` (ion mobility). All processing methods return a new `Spectrum` and are chainable.

**Constructor:**

```python
Spectrum(
    mz: NDArray[np.float64],
    intensity: NDArray[np.float64],
    charge: NDArray[np.int32] | None = None,
    im: NDArray[np.float64] | None = None,
    spectrum_type: SpectrumType | str | None = None,
    denoised: str | None = None,
    normalized: str | None = None,
)
```

**Methods:**

| Method | Returns | Summary |
|---|---|---|
| `.peaks` | `list[Peak]` | All peaks as `Peak` objects |
| `.top_peaks(n, by, reverse)` | `list[Peak]` | Top N peaks sorted by attribute |
| `.has_peak(target_mz, ...)` | `bool` | Check for a peak near target m/z |
| `.get_peak(target_mz, ...)` | `Peak \| None` | Single best-matching peak |
| `.get_peaks(target_mz, ...)` | `list[Peak]` | All peaks matching criteria |
| `.filter(...)` | `Spectrum` | Remove peaks outside bounds |
| `.normalize(method)` | `Spectrum` | Scale intensities (max / tic / median) |
| `.denoise(method)` | `Spectrum` | Remove peaks below noise threshold |
| `.centroid()` | `Spectrum` | Convert profile to centroid via Gaussian fit |
| `.merge(...)` | `Spectrum` | Merge nearby peaks by weighted average |
| `.deconvolute(...)` | `Spectrum` | Assign isotope clusters and charge states |
| `.decharge()` | `Spectrum` | Convert charged m/z to neutral masses |
| `.compress(...)` | `str` | Serialise to compact ASCII string |
| `.from_compressed(s)` | `Spectrum` | Deserialise from compressed string (classmethod) |
| `.update(**kwargs)` | `Spectrum` | Return copy with specified fields replaced |
| `.plot_table(show_charges, show_scores)` | `pd.DataFrame` | Build an editable plot table (one row per peak) |
| `.annot_plot_table(fragments, ...)` | `pd.DataFrame` | Build an editable annotated plot table with fragment labels |

Full documentation: [Spectrum reference](spectrum.md)

---

### `MsnSpectrum`

Extends `Spectrum` with instrument metadata fields. Returned by both readers.

**Additional fields (all optional):**

| Field | Type | Description |
|---|---|---|
| `scan_number` | `int \| None` | Native scan or frame number |
| `ms_level` | `int \| None` | MS level (1, 2, …) |
| `native_id` | `str \| None` | Instrument-specific scan identifier |
| `rt` | `float \| None` | Retention time in seconds |
| `injection_time` | `float \| None` | Ion accumulation time in ms |
| `mz_range` | `tuple[float, float] \| None` | Acquisition m/z window |
| `im_range` | `tuple[float, float] \| None` | Ion mobility window |
| `im_type` | `str \| None` | Ion mobility unit string |
| `polarity` | `"positive" \| "negative" \| None` | Scan polarity |
| `resolution` | `float \| None` | Instrument resolution |
| `analyzer` | `str \| None` | Mass analyser type |
| `ramp_time` | `float \| None` | timsTOF ramp time in ms |
| `collision_energy` | `float \| None` | Fragmentation collision energy |
| `activation_type` | `str \| None` | Fragmentation type (HCD, CID, PASEF, …) |
| `precursors` | `list[TargetIon] \| None` | Precursor ions (MS2 only) |

Full documentation: [Spectrum reference — MsnSpectrum](spectrum.md#msnspectrum)

---

### `Peak`

Frozen dataclass for a single spectral peak. Returned by peak access methods.

```python
Peak(mz: float, intensity: float, charge: int | None = None, im: float | None = None)
```

---

### `TargetIon`

Frozen dataclass, subclass of `Peak`, representing a selected precursor ion.

```python
# All fields are keyword-only (kw_only=True)
TargetIon(mz=..., intensity=..., charge=..., im=..., is_monoisotopic=...)
```

---

### `SpectrumType`

Not exported from the package root. Import via:

```python
from spxtacular.core import SpectrumType
```

`StrEnum` with three members:

| Member | Value | Meaning |
|---|---|---|
| `CENTROID` | `"centroid"` | Peak-picked data |
| `PROFILE` | `"profile"` | Raw continuous data |
| `DECONVOLUTED` | `"deconvoluted"` | Isotope clusters assigned |

---

## Readers

### `MzmlReader`

Reads `.mzML` files. No context manager required.

```python
MzmlReader(mzml_path: str)
```

| Property | Yields |
|---|---|
| `.ms1` | `Generator[MsnSpectrum]` — all MS1 spectra |
| `.ms2` | `Generator[MsnSpectrum]` — all MS2 spectra |

Full documentation: [Readers — MzmlReader](readers.md#mzmlreader)

---

### `DReader`

Reads Bruker timsTOF `.d` directories. **Must be used as a context manager.**

```python
DReader(analysis_dir: str)
```

| Property / Attribute | Type | Description |
|---|---|---|
| `.ms1` | `Generator[MsnSpectrum]` | All MS1 frames |
| `.ms2` | `Generator[MsnSpectrum]` | All MS2 spectra |
| `.acquisition_type` | `AcquisitionType` | DDA / DIA / PRM / UNKNOWN |

Full documentation: [Readers — DReader](readers.md#dreader)

---

### `AcquisitionType`

Not exported from the package root. Import via:

```python
from spxtacular.reader import AcquisitionType
```

`StrEnum` with four members: `DDA`, `DIA`, `PRM`, `UNKNOWN`.

---

## Noise estimation

`estimate_noise_level` is not exported from the package root but is the function backing `Spectrum.denoise()`.

```python
from spxtacular.noise import estimate_noise_level

threshold = estimate_noise_level(intensity_array, method="mad")
```

| `method` | Strategy |
|---|---|
| `"mad"` | `median + 3 × 1.4826 × MAD` |
| `"percentile"` | 5th percentile |
| `"histogram"` | Histogram mode + 3 σ |
| `"baseline"` | Bottom-quartile mean + 3 σ |
| `"iterative_median"` | Three-pass iterative median refinement |
| `float` or `int` | Used directly as the absolute threshold |

---

## Compression utilities

Not exported from the package root. Available via:

```python
from spxtacular.compress import compress_spectra, decompress_spectra
```

Prefer the `Spectrum.compress()` / `Spectrum.from_compressed()` API instead.

| Function | Summary |
|---|---|
| `compress_spectra(spectrum, ...)` | Serialise a `Spectrum` to a string |
| `decompress_spectra(compressed_str)` | Deserialise back to a `Spectrum` |

---

## Visualization

Requires `plotly` (`pip install plotly`). All three functions return a `plotly.graph_objects.Figure`.

Full documentation: [Visualization](visualization.md)

### `plot_spectrum`

```python
from spxtacular import plot_spectrum
```

```python
plot_spectrum(
    spectrum: Spectrum,
    title: str | None = None,
    show_charges: bool = True,
    show_scores: bool = True,
    **layout_kwargs,
)
```

### `mirror_plot`

```python
from spxtacular import mirror_plot
```

```python
mirror_plot(
    raw: Spectrum,
    decon: Spectrum,
    title: str | None = None,
    normalize: bool = True,
    show_scores: bool = True,
    **layout_kwargs,
)
```

### `annotate_spectrum`

```python
from spxtacular import annotate_spectrum
```

```python
annotate_spectrum(
    spectrum: Spectrum,
    fragments,
    mz_tol: float = 0.02,
    mz_tol_type: str = "da",
    title: str | None = None,
    **layout_kwargs,
)
```

---

## Matching and scoring

Full documentation: [Fragment matching and scoring](scoring.md)

### `match_fragments`

```python
from spxtacular import match_fragments
```

```python
match_fragments(
    spectrum: Spectrum,
    fragments,
    mz_tol: float = 0.02,
    mz_tol_type: str = "Da",
    peak_selection: Literal["closest", "largest", "all"] = "closest",
)
```

### `score`

```python
from spxtacular import score
```

```python
score(
    spectrum: Spectrum,
    fragments,
    mz_tol: float = 10,
    mz_tol_type: str = "ppm",
) -> dict
```

Returns a dict of PSM metrics: `hyperscore`, `probability_score`, `total_matched_intensity`, `matched_fraction`, `intensity_fraction`, `mean_ppm_error`, `spectral_angle`, `longest_run`.

---

## Plot table API

Provides an intermediate `pandas.DataFrame` that holds all data and visual properties for a spectrum plot. Users can freely modify the DataFrame before passing it to `plot_from_table`.

Full documentation: [Visualization — Plot table API](visualization.md#plot-table-api)

### `build_plot_table`

```python
from spxtacular import build_plot_table
```

```python
build_plot_table(
    spectrum: Spectrum,
    show_charges: bool = True,
    show_scores: bool = True,
) -> pd.DataFrame
```

### `build_annot_plot_table`

```python
from spxtacular import build_annot_plot_table
```

```python
build_annot_plot_table(
    spectrum: Spectrum,
    fragments,
    mz_tol: float = 0.02,
    mz_tol_type: Literal["Da", "ppm"] = "Da",
    peak_selection: Literal["closest", "largest", "all"] = "closest",
    include_sequence: bool = False,
) -> pd.DataFrame
```

### `plot_from_table`

```python
from spxtacular import plot_from_table
```

```python
plot_from_table(
    table: pd.DataFrame,
    title: str | None = None,
    **layout_kwargs,
) -> go.Figure
```
