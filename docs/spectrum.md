# Spectrum and MsnSpectrum

## SpectrumType

```python
class SpectrumType(StrEnum):
    CENTROID = "centroid"
    PROFILE  = "profile"
    DECONVOLUTED = "deconvoluted"
```

`SpectrumType` tags what stage the data is in. Several methods check or set this flag to prevent out-of-order operations (e.g., calling `.decharge()` on a non-deconvoluted spectrum raises `ValueError`).

---

## Peak

```python
@dataclass(frozen=True, slots=True)
class Peak:
    mz: float
    intensity: float
    charge: int | None = None
    im: float | None = None
    score: float | None = None
```

A frozen dataclass representing a single detected peak. `charge`, `im`, and `score` are optional. `Peak` objects are returned by `.peaks`, `.top_peaks()`, `.get_peak()`, and `.get_peaks()` — they are read-only views, not references into the underlying arrays.

`score` holds the isotopic profile score (0–1) assigned during deconvolution, or `None` for peaks that have not been through deconvolution.

```python
>>> peak = Peak(mz=500.1, intensity=1e5, charge=2)
>>> repr(peak)
'Peak(mz=500.1000, int=1.00e+05, z=2)'
```

### Charge conventions

| `charge` value | Meaning |
|---|---|
| `> 0` | Peak belongs to an assigned isotope cluster with that charge state |
| `-1` | Singleton — no isotope neighbours found at any tested charge |
| `0` | After `.decharge()` — neutral mass, charge state no longer tracked |

---

## Spectrum

```python
@dataclass(slots=True)
class Spectrum:
    mz: NDArray[np.float64]
    intensity: NDArray[np.float64]
    charge: NDArray[np.int32] | None = None
    im: NDArray[np.float64] | None = None
    score: NDArray[np.float64] | None = None
    spectrum_type: SpectrumType | str | None = None
    denoised: str | None = None
    normalized: str | None = None
```

The central data structure. `mz` and `intensity` must have the same length. `charge`, `im`, and `score` must also match that length when provided.

**Fields:**

| Field | Type | Description |
|---|---|---|
| `mz` | `NDArray[np.float64]` | Peak m/z values, sorted ascending |
| `intensity` | `NDArray[np.float64]` | Parallel peak intensities |
| `charge` | `NDArray[np.int32] \| None` | Charge state per peak. `None` before deconvolution |
| `im` | `NDArray[np.float64] \| None` | Ion mobility per peak. `None` if not acquired |
| `score` | `NDArray[np.float64] \| None` | Per-peak isotopic profile score (0–1). Populated after `deconvolute()`; `None` otherwise. Singletons have `score=0.0`. |
| `spectrum_type` | `SpectrumType \| str \| None` | Stage tag: `CENTROID`, `PROFILE`, or `DECONVOLUTED` |
| `denoised` | `str \| None` | Name of the denoising method applied, or `None` |
| `normalized` | `str \| None` | Name of the normalization method applied, or `None` |

**Validation rules enforced in `__post_init__`:**

- `len(charge) == len(mz)` when `charge` is not `None`
- `len(im) == len(mz)` when `im` is not `None`
- `len(score) == len(mz)` when `score` is not `None`
- A `charge` array may only be present when `spectrum_type == DECONVOLUTED`

```python
import numpy as np
from spxtacular import Spectrum

spec = Spectrum(
    mz=np.array([500.1, 800.2, 1200.5], dtype=np.float64),
    intensity=np.array([1e5, 2e5, 9e4], dtype=np.float64),
)
print(spec)
# Spectrum(n_peaks=3, type=None, denoised=None, normalized=None)
```

---

### Peak access

#### `peaks` property

```python
@property
def peaks(self) -> list[Peak]
```

Returns all peaks as a list of `Peak` objects. Iterates the full spectrum; prefer numpy operations on `.mz` / `.intensity` for performance on large spectra.

```python
for peak in spec.peaks:
    print(peak.mz, peak.intensity)
```

#### `top_peaks`

```python
def top_peaks(
    self,
    n: int,
    by: Literal["intensity", "mz", "charge", "im"] = "intensity",
    reverse: bool = True,
) -> list[Peak]
```

Returns the top `n` peaks sorted by the chosen attribute.

| Parameter | Description |
|---|---|
| `n` | Number of peaks to return |
| `by` | Sort key: `"intensity"` (default), `"mz"`, `"charge"`, `"im"` |
| `reverse` | `True` (default) returns highest values first |

`"charge"` requires a charge array to be present; `"im"` requires an ion mobility array. Both raise `ValueError` otherwise.

```python
# Five most intense peaks
top5 = spec.top_peaks(5)

# Lowest-mz three peaks
low_mz = spec.top_peaks(3, by="mz", reverse=False)
```

---

### Peak finding

#### `has_peak`

```python
def has_peak(
    self,
    target_mz: float,
    mz_tol: float = 0.01,
    mz_tol_type: Literal["Da", "ppm"] = "Da",
    target_charge: int | None = None,
    target_im: float | None = None,
    im_tol: float = 0.01,
) -> bool
```

Returns `True` if at least one peak matches all supplied criteria.

```python
spec.has_peak(500.1, mz_tol=0.02)
spec.has_peak(500.1, mz_tol=10, mz_tol_type="ppm", target_charge=2)
```

#### `get_peak`

```python
def get_peak(
    self,
    target_mz: float,
    mz_tol: float = 0.01,
    mz_tol_type: Literal["Da", "ppm"] = "Da",
    target_charge: int | None = None,
    target_im: float | None = None,
    im_tol: float = 0.01,
    collision: Literal["largest", "closest"] = "largest",
) -> Peak | None
```

Returns a single matching peak, or `None` if no match is found. When multiple peaks fall within tolerance, `collision="largest"` picks the most intense; `collision="closest"` picks the nearest in m/z.

```python
peak = spec.get_peak(800.2, mz_tol=5, mz_tol_type="ppm")
if peak:
    print(f"Found: {peak}")
```

#### `get_peaks`

```python
def get_peaks(
    self,
    target_mz: float,
    mz_tol: float = 0.01,
    mz_tol_type: Literal["Da", "ppm"] = "Da",
    target_charge: int | None = None,
    target_im: float | None = None,
    im_tol: float = 0.01,
) -> list[Peak]
```

Returns all peaks matching the criteria (may be empty).

---

### Filtering and processing

All processing methods accept `inplace: bool = False`. When `inplace=False` (the default) a new `Spectrum` is returned, leaving the original unchanged and allowing method chaining.

#### `filter`

```python
def filter(
    self,
    min_mz: float | None = None,
    max_mz: float | None = None,
    min_intensity: float | None = None,
    max_intensity: float | None = None,
    min_charge: int | None = None,
    max_charge: int | None = None,
    min_im: float | None = None,
    max_im: float | None = None,
    top_n: int | None = None,
    inplace: bool = False,
) -> Self
```

Removes peaks outside the given bounds. All parameters are optional and combinable. `top_n` is applied last — after all range filters — keeping the `top_n` most intense survivors.

Charge, ion mobility, and score filters are silently ignored if the spectrum lacks those arrays.

**Score filter parameters:**

| Parameter | Type | Description |
|---|---|---|
| `min_score` | `float \| None` | Keep peaks with score >= this value. Only effective when `score` array is present. |
| `max_score` | `float \| None` | Keep peaks with score <= this value. Only effective when `score` array is present. |

```python
# Keep peaks between 200 and 1500 Da with intensity >= 1000
filtered = spec.filter(min_mz=200, max_mz=1500, min_intensity=1000)

# Keep only the 50 most intense peaks after m/z filtering
filtered = spec.filter(min_mz=200, top_n=50)
```

#### `normalize`

```python
def normalize(
    self,
    method: Literal["max", "tic", "median"] = "max",
    inplace: bool = False,
) -> Self
```

Scales all intensities so that the chosen reference equals 1.0.

| `method` | Normalization factor |
|---|---|
| `"max"` (default) | Most intense peak |
| `"tic"` | Total ion current (sum of all intensities) |
| `"median"` | Median intensity |

Calling `normalize` on an already-normalized spectrum emits a `UserWarning` and returns `self` unchanged.

```python
norm = spec.normalize()            # max normalization
norm = spec.normalize("tic")       # TIC normalization
```

#### `denoise`

```python
def denoise(
    self,
    method: Literal["mad", "percentile", "histogram", "baseline", "iterative_median"]
            | float | int = "mad",
    inplace: bool = False,
) -> Self
```

Removes peaks below an estimated noise threshold. Peaks at or above the threshold are kept.

| `method` | Threshold strategy |
|---|---|
| `"mad"` (default) | `median + 3 × 1.4826 × MAD` |
| `"percentile"` | 5th percentile of intensities |
| `"histogram"` | Mode of 100-bin histogram + 3 σ (FWHM-derived) |
| `"baseline"` | Mean + 3 σ of the bottom 25th percentile |
| `"iterative_median"` | Iteratively refines median/MAD estimate over 3 passes |
| `float` or `int` | Used directly as the absolute threshold |

Calling `denoise` on an already-denoised spectrum emits a `UserWarning` and returns `self` unchanged.

```python
spec.denoise()                       # MAD (robust, recommended for most spectra)
spec.denoise("histogram")            # histogram mode estimate
spec.denoise(5000.0)                 # fixed absolute threshold
```

#### `centroid`

```python
def centroid(self, inplace: bool = False) -> Self
```

Converts a profile-mode spectrum to centroid mode using vectorized Gaussian fitting. Detects local maxima, fits a Gaussian to each triplet of points, and returns sub-bin peak positions. Ion mobility data is preserved at the apex value.

Calling this on an already-centroided spectrum emits a `UserWarning` and returns `self` unchanged.

```python
centroided = profile_spec.centroid()
```

#### `merge`

```python
def merge(
    self,
    mz_tolerance: float = 0.01,
    mz_tolerance_type: Literal["ppm", "da"] = "da",
    im_tolerance: float = 0.05,
    im_tolerance_type: Literal["relative", "absolute"] = "relative",
    inplace: bool = False,
) -> Self
```

Merges nearby peaks using a greedy intensity-ordered strategy. Peaks are processed from most to least intense; each unused neighbour within the tolerance window is merged into the current peak. The merged peak carries the intensity-weighted average m/z (and ion mobility if present) and the summed intensity. Charge arrays are preserved — only peaks with matching charge are merged together.

```python
merged = spec.merge(mz_tolerance=0.02, mz_tolerance_type="da")
merged = spec.merge(mz_tolerance=5, mz_tolerance_type="ppm")
```

#### `deconvolute`

```python
def deconvolute(
    self,
    tolerance: float = 50,
    tolerance_type: Literal["ppm", "da"] = "ppm",
    charge_range: tuple[int, int] = (1, 3),
    intensity: Literal["base", "total"] = "total",
    max_dpeaks: int = 2000,
    inplace: bool = False,
) -> Self
```

Assigns each peak to an isotope cluster and records the charge state. Returns a spectrum with `spectrum_type=DECONVOLUTED` and a populated `charge` array.

| Parameter | Description |
|---|---|
| `tolerance` | Peak matching tolerance (default 50 ppm) |
| `tolerance_type` | `"ppm"` (default) or `"da"` |
| `charge_range` | `(min_charge, max_charge)` inclusive; default `(1, 3)` |
| `intensity` | `"total"` (default) sums the whole cluster; `"base"` uses only the monoisotopic peak |
| `max_dpeaks` | Maximum output peaks (default 2000) |
| `min_intensity` | `float \| "min"` — Absolute intensity floor for isotope detectability. The sentinel `"min"` (default) uses the spectrum's own minimum intensity as the S/N floor. |
| `min_score` | `float` — Clusters whose best isotopic profile score falls below this threshold are recorded as singletons. Default `0.0` accepts all clusters. |

After deconvolution the `charge` array follows the [charge conventions](#charge-conventions) table: `> 0` for assigned clusters, `-1` for singletons.

See [Deconvolution](deconvolution.md) for a detailed walkthrough.

```python
decon = spec.deconvolute(charge_range=(1, 5), tolerance=10, tolerance_type="ppm")
```

#### `decharge`

```python
def decharge(self, inplace: bool = False) -> Self
```

Converts deconvoluted m/z values to neutral monoisotopic masses using `neutral_mass = (mz × charge) - (charge × proton_mass)`. Singletons (`charge == -1`) are dropped. The resulting `charge` array is set to all zeros (meaning "charge unknown / neutral mass").

Raises `ValueError` if the spectrum is not in `DECONVOLUTED` state.

> The `score` array is propagated through `decharge()` — each surviving neutral-mass peak retains the score of its charged precursor.

```python
neutral = decon.decharge()
# neutral.mz now contains neutral masses sorted ascending
# neutral.charge is all zeros
# neutral.score carries through from the deconvoluted spectrum
```

#### `update`

```python
def update(self, inplace: bool = False, **kwargs) -> Self
```

Low-level helper to create a new `Spectrum` with arbitrary fields replaced. Prefer the named methods above for normal use.

```python
renamed = spec.update(spectrum_type="centroid")
```

---

### Compression

#### `compress`

```python
def compress(
    self,
    url_safe: bool = False,
    mz_precision: int | None = None,
    intensity_precision: int | None = None,
    im_precision: int | None = None,
    compression: str = "gzip",
) -> str
```

Serialises the spectrum to a compact ASCII string. m/z values are delta-encoded; intensities and ion mobility use raw float32 hex encoding. The result is compressed with gzip, zlib, or brotli (requires `pip install brotli`) and then base85-encoded (default) or base64 URL-safe encoded when `url_safe=True`.

Optional `*_precision` parameters round the corresponding arrays before encoding, reducing compressed size at the cost of numeric precision.

```python
blob = spec.compress()
blob_url = spec.compress(url_safe=True, mz_precision=4, compression="zlib")
```

#### `Spectrum.from_compressed`

```python
@classmethod
def from_compressed(cls, compressed_str: str) -> Spectrum
```

Round-trips a string produced by `.compress()` back to a `Spectrum`.

```python
recovered = Spectrum.from_compressed(blob)
```

---

### Visualization

#### `plot`

```python
def plot(
    self,
    title: str | None = None,
    show_charges: bool = True,
    show_scores: bool = True,
    **layout_kwargs,
) -> Figure
```

Returns a Plotly `Figure` (stick plot). Requires `plotly` (`pip install plotly`).

| Parameter | Description |
|---|---|
| `title` | Plot title |
| `show_charges` | Colour sticks by charge state when a `charge` array is present |
| `show_scores` | Annotate scored peaks with their score value when a `score` array is present |

```python
spec.plot(title="My spectrum").show()
decon.plot(show_charges=True, show_scores=True).show()
```

See [Visualization](visualization.md) for `mirror_plot()` and `annotate_spectrum()`.

#### `plot_table`

```python
def plot_table(
    self,
    show_charges: bool = True,
    show_scores: bool = True,
) -> pd.DataFrame
```

Returns a `pandas.DataFrame` with one row per peak. Each row contains both the raw peak data (`mz`, `intensity`, `charge`, `score`, `im`) and all visual properties (`color`, `linewidth`, `opacity`, `series`, `label`, `label_size`, `label_font`, `label_color`, `label_yshift`, `label_xanchor`, `hover`). Modify the table freely, then render it with `plot_from_table()`.

```python
tbl = decon.plot_table()
tbl.loc[tbl["charge"] == 2, "color"] = "red"
tbl.loc[tbl["intensity"] > 1e5, "linewidth"] = 2.0
fig = plot_from_table(tbl, title="Custom plot")
fig.show()
```

#### `annot_plot_table`

```python
def annot_plot_table(
    self,
    fragments,
    mz_tol: float = 0.02,
    mz_tol_type: Literal["Da", "ppm"] = "Da",
    peak_selection: Literal["closest", "largest", "all"] = "closest",
    include_sequence: bool = False,
) -> pd.DataFrame
```

Like `plot_table()` but matched peaks are coloured by ion series and labelled with their fragment identifier. Unmatched peaks are grey. Modify the returned table and call `plot_from_table()` to render.

```python
tbl = spec.annot_plot_table(fragments, mz_tol=10, mz_tol_type="ppm")
tbl.loc[tbl["label"] != "", "label_size"] = 14
fig = plot_from_table(tbl, title="Annotated")
fig.show()
```

See [Visualization — Plot table API](visualization.md#plot-table-api) for full column reference.

---

## MsnSpectrum

`MsnSpectrum` extends `Spectrum` with instrument-level metadata. It is what the readers (`DReader`, `MzmlReader`) yield. All `Spectrum` methods are available unchanged.

```python
@dataclass(slots=True, kw_only=True)
class MsnSpectrum(Spectrum):
    # Scan identification
    scan_number: int | None = None
    ms_level: int | None = None
    native_id: str | None = None

    # Timing
    rt: float | None = None          # retention time, seconds
    injection_time: float | None = None  # ion accumulation time, ms

    # Acquisition windows
    mz_range: tuple[float, float] | None = None
    im_range: tuple[float, float] | None = None
    im_type: str | None = None       # e.g. "1/K0", "drift_time_ms"

    # Instrument settings
    polarity: Literal["positive", "negative"] | None = None

    # Optional metadata
    resolution: float | None = None
    analyzer: str | None = None      # e.g. "TOF", "FTMS", "ITMS"
    ramp_time: float | None = None
    collision_energy: float | None = None
    activation_type: str | None = None
    precursors: list[TargetIon] | None = None
```

### TargetIon

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class TargetIon(Peak):
    is_monoisotopic: bool | None
```

Represents a precursor ion selected for fragmentation. Stored in `MsnSpectrum.precursors`.

### Example: inspecting an MS2 spectrum

```python
from spxtacular import MzmlReader

reader = MzmlReader("run.mzML")
for spec in reader.ms2:
    print(f"Scan {spec.scan_number}, RT={spec.rt:.1f}s, CE={spec.collision_energy}")
    if spec.precursors:
        prec = spec.precursors[0]
        print(f"  Precursor: {prec.mz:.4f} m/z, z={prec.charge}")
    break
```
