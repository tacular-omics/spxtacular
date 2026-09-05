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
    iso_score: float | None = None
```

A frozen dataclass representing a single detected peak. `charge`, `im`, and `iso_score` are optional. `Peak` objects are returned by `.peaks`, `.top_peaks()`, `.get_peak()`, and `.get_peaks()`. They are read-only snapshots, not references into the underlying arrays.

`iso_score` holds the isotopic profile score (0–1) assigned during deconvolution, or `None` for peaks that have not been through deconvolution.

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
    iso_score: NDArray[np.float64] | None = None
    spectrum_type: SpectrumType | str | None = None
    denoised: str | None = None
    normalized: str | None = None
```

The central data structure. `mz` and `intensity` must have the same length. `charge`, `im`, and `iso_score` must also match that length when provided.

**Fields:**

| Field | Type | Description |
|---|---|---|
| `mz` | `NDArray[np.float64]` | Peak m/z values, sorted ascending |
| `intensity` | `NDArray[np.float64]` | Parallel peak intensities |
| `charge` | `NDArray[np.int32] \| None` | Charge state per peak. `None` before deconvolution |
| `im` | `NDArray[np.float64] \| None` | Ion mobility per peak. `None` if not acquired |
| `iso_score` | `NDArray[np.float64] \| None` | Per-peak isotopic profile score (0–1). Populated after `deconvolute()`; `None` otherwise. Singletons have `iso_score=0.0`. |
| `spectrum_type` | `SpectrumType \| str \| None` | Stage tag: `CENTROID`, `PROFILE`, or `DECONVOLUTED` |
| `denoised` | `str \| None` | Name of the denoising method applied, or `None` |
| `normalized` | `str \| None` | Name of the normalization method applied, or `None` |

**Validation rules enforced in `__post_init__`:**

- `len(charge) == len(mz)` when `charge` is not `None`
- `len(im) == len(mz)` when `im` is not `None`
- `len(iso_score) == len(mz)` when `iso_score` is not `None`
- Supplying a `charge` array forces `spectrum_type` to `DECONVOLUTED` (it is rewritten, not rejected)

**`is_decharged` property** — `True` when every (non-dropped) peak's `charge == 0`, i.e. the spectrum has already been through `decharge()`. Used internally by `decharge()`, `remove_precursor_peak()`, and `match_fragments()` to detect neutral-mass spectra.

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
    by: Literal["intensity", "mz", "charge", "im", "score"] = "intensity",
    reverse: bool = True,
) -> list[Peak]
```

Returns the top `n` peaks sorted by the chosen attribute.

| Parameter | Description |
|---|---|
| `n` | Number of peaks to return |
| `by` | Sort key: `"intensity"` (default), `"mz"`, `"charge"`, `"im"`, `"score"` |
| `reverse` | `True` (default) returns highest values first |

`"charge"` requires a charge array, `"im"` an ion mobility array, and `"score"` an `iso_score` array. Each raises `ValueError` when the corresponding array is absent.

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
    tolerance: float = 0.01,
    tolerance_type: Literal["da", "ppm"] = "da",
    target_charge: int | None = None,
    target_im: float | None = None,
    im_tol: float = 0.01,
) -> bool
```

Returns `True` if at least one peak matches all supplied criteria.

```python
spec.has_peak(500.1, tolerance=0.02)
spec.has_peak(500.1, tolerance=10, tolerance_type="ppm", target_charge=2)
```

#### `get_peak`

```python
def get_peak(
    self,
    target_mz: float,
    tolerance: float = 0.01,
    tolerance_type: Literal["da", "ppm"] = "da",
    target_charge: int | None = None,
    target_im: float | None = None,
    im_tol: float = 0.01,
    collision: Literal["largest", "closest"] = "largest",
) -> Peak | None
```

Returns a single matching peak, or `None` if no match is found. When multiple peaks fall within tolerance, `collision="largest"` picks the most intense; `collision="closest"` picks the nearest in m/z.

```python
peak = spec.get_peak(800.2, tolerance=5, tolerance_type="ppm")
if peak:
    print(f"Found: {peak}")
```

#### `get_peaks`

```python
def get_peaks(
    self,
    target_mz: float,
    tolerance: float = 0.01,
    tolerance_type: Literal["da", "ppm"] = "da",
    target_charge: int | None = None,
    target_im: float | None = None,
    im_tol: float = 0.01,
) -> list[Peak]
```

Returns all peaks matching the criteria (may be empty).

---

### Filtering and processing

Transformation methods that expose `inplace` default to `False`. In that mode they return a new
`Spectrum`, leave the input unchanged, and support method chaining.

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
    min_score: float | None = None,
    max_score: float | None = None,
    top_n: int | None = None,
    top_n_per_window: tuple[int, float] | None = None,
    inplace: bool = False,
) -> Self
```

Removes peaks outside the given bounds. All parameters are optional and combinable. `top_n` is applied last — after all range filters — keeping the `top_n` most intense survivors. `top_n_per_window=(n, width)` is its windowed counterpart: keep the `n` most intense peaks per fixed-width m/z window (bins anchored at 0, `[k*width, (k+1)*width)`), preserving quiet regions that a global `top_n` erases. The two are mutually exclusive.

Charge, ion mobility, and score filters are silently ignored if the spectrum lacks those arrays.

**Score filter parameters:**

| Parameter | Type | Description |
|---|---|---|
| `min_score` | `float \| None` | Keep peaks with score >= this value. Only effective when the `iso_score` array is present. |
| `max_score` | `float \| None` | Keep peaks with score <= this value. Only effective when the `iso_score` array is present. |

```python
# Keep peaks between 200 and 1500 Da with intensity >= 1000
filtered = spec.filter(min_mz=200, max_mz=1500, min_intensity=1000)

# Keep only the 50 most intense peaks after m/z filtering
filtered = spec.filter(min_mz=200, top_n=50)

# Keep the 10 most intense peaks per 100 Th window (search-engine-style preprocessing)
filtered = spec.filter(top_n_per_window=(10, 100.0))
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

Calling `normalize` on an already-normalized spectrum emits a `UserWarning` and leaves its data
unchanged. The default non-inplace path returns an independent copy.

Removing peaks, replacing intensities, rounding peaks, or combining multiple nonempty spectra
clears the normalization marker. Normalize again after those operations when the resulting
spectrum needs a unit reference.

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

Calling `denoise` on an already-denoised spectrum emits a `UserWarning` and leaves its data
unchanged. The default non-inplace path returns an independent copy.

```python
spec.denoise()                       # MAD (robust, recommended for most spectra)
spec.denoise("histogram")            # histogram mode estimate
spec.denoise(5000.0)                 # fixed absolute threshold
```

#### `centroid`

```python
def centroid(
    self,
    min_intensity: float | Literal["noise"] | None = None,
    inplace: bool = False,
) -> Self
```

Converts a profile-mode spectrum to centroid mode using vectorized Gaussian fitting. Sharp local
maxima use a three-point log-space fit for sub-bin positions. Flat maxima use the midpoint of the
plateau's m/z bounds and its observed height, since a Gaussian fit is not identifiable there.
Ion mobility is taken from the apex sample, or the lower middle sample of a plateau.
`min_intensity="noise"` uses the MAD noise estimate. A number applies an absolute floor,
and `None` applies no intensity floor. Boundary peaks without both flanks are excluded.

Calling this on an already-centroided spectrum emits a `UserWarning` and leaves its data unchanged.
The default non-inplace path returns an independent copy.

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
    min_intensity: float | Literal["min"] = "min",
    min_score: float = 0.0,
    isotope_model: IsotopeModel | IsotopeModelType | str = "peptide",
    min_isotope_abundance: float = 0.01,
    max_isotope_fold_error: float = 2.0,
    max_isotope_gaps: int = 0,
    max_isotopes: int | None = None,
    im_tolerance: float = 0.05,
    im_tolerance_type: Literal["relative", "absolute"] = "relative",
    ionization_model: IonizationModel | str | float | None = None,
) -> Self
```

Assigns each peak to an isotope cluster and records the charge state. Returns a spectrum with `spectrum_type=DECONVOLUTED` and a populated `charge` array.

| Parameter | Description |
|---|---|
| `tolerance` | Peak matching tolerance (default 50 ppm) |
| `tolerance_type` | `"ppm"` (default) or `"da"` |
| `charge_range` | `(min_charge, max_charge)` inclusive; default `(1, 3)` |
| `intensity` | `"total"` sums matched peaks; `"base"` uses observed A+0 or zero when it is absent |
| `max_dpeaks` | Maximum output peaks (default 2000) |
| `min_intensity` | `float \| "min"` — Absolute intensity floor for isotope detectability. The sentinel `"min"` (default) uses the spectrum's own minimum intensity as the S/N floor. |
| `min_score` | `float` — Clusters whose best isotopic profile score falls below this threshold are recorded as singletons. Default `0.0` accepts all clusters. |
| `isotope_model` | Built-in name or custom `IsotopeModel`. Available presets are `"peptide"`, `"glycan"`, `"lipid"`, `"dna"`, and `"rna"`. |
| `min_isotope_abundance` | Relative theoretical abundance at which directional expansion stops. Default `0.01`. |
| `max_isotope_fold_error` | Hard observed-to-expected intensity gate. Default `2.0` accepts 0.5x to 2x expected. |
| `max_isotope_gaps` | Missing positions allowed before stopping one direction. Default `0`. |
| `max_isotopes` | Optional hard envelope-length limit. Default `None` is adaptive. |
| `im_tolerance` | Candidate-to-seed mobility tolerance when ion mobility is available. Default `0.05`. |
| `im_tolerance_type` | `"relative"` (default) or `"absolute"`. |
| `ionization_model` | Adduct preset, signed carrier mass, or custom model. Defaults from scan polarity |

After deconvolution the `charge` array follows the [charge conventions](#charge-conventions) table: `> 0` for assigned clusters, `-1` for singletons.

See [Deconvolution](deconvolution.md) for a detailed walkthrough.

```python
decon = spec.deconvolute(charge_range=(1, 5), tolerance=10, tolerance_type="ppm")
```

#### `decharge`

```python
def decharge(
    self,
    inplace: bool = False,
    *,
    ionization_model: IonizationModel | str | float | None = None,
) -> Self
```

Converts deconvoluted m/z values to neutral monoisotopic masses using the ionization model recorded
by `deconvolute()`. Singletons (`charge == -1`) are dropped. The resulting `charge` array is set to
all zeros, which marks neutral masses.

Pass `ionization_model` only to override recorded provenance. Without provenance, positive scans
use `[M+H]+` and negative scans use `[M-H]-`.

Raises `ValueError` if the spectrum is not in `DECONVOLUTED` state.

Calling `decharge()` again on an already-decharged spectrum (`spec.is_decharged`) warns and leaves
the neutral m/z values unchanged. The default non-inplace path returns an independent copy.

> The `iso_score` array is propagated through `decharge()` — each surviving neutral-mass peak retains the score of its charged precursor.

```python
neutral = decon.decharge()
# neutral.mz now contains neutral masses sorted ascending
# neutral.charge is all zeros
# neutral.iso_score carries through from the deconvoluted spectrum
```

#### `sort`

```python
def sort(
    self,
    by: Literal["mz", "intensity", "charge", "im", "score"] = "mz",
    reverse: bool = False,
    inplace: bool = False,
) -> Self
```

Reorders every parallel array by the chosen attribute. `by="charge"`, `"im"`, and `"score"` raise
`ValueError` when the corresponding array is absent. Any other value of `by` raises `ValueError`.

```python
by_intensity = spec.sort(by="intensity", reverse=True)   # most intense first
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

### JSON transport

#### `to_dict` / `Spectrum.from_dict`

```python
def to_dict(self) -> dict[str, Any]

@classmethod
def from_dict(cls, payload: Mapping[str, Any]) -> Spectrum | MsnSpectrum
```

Convert a spectrum to or from the versioned `spxtacular.spectrum` transport
format. The payload contains only JSON-native values. Peak columns stay in
parallel arrays, and `kind` records whether the original object was a
`Spectrum` or `MsnSpectrum`.

```python
payload = spec.to_dict()
restored = Spectrum.from_dict(payload)

assert type(restored) is type(spec)
assert restored == spec
```

Calling `Spectrum.from_dict()` dispatches to `MsnSpectrum` when appropriate.
Calling `MsnSpectrum.from_dict()` requires an MSn payload. Version 1 rejects
missing fields, unknown fields, malformed arrays, and non-finite numeric values
instead of silently discarding data.

#### `to_json` / `Spectrum.from_json`

```python
def to_json(self, *, indent: int | None = None) -> str

@classmethod
def from_json(cls, value: str | bytes | bytearray) -> Spectrum | MsnSpectrum
```

Encode or decode the same transport as strict JSON. Compact JSON is the
default. Pass `indent=2` for readable output.

```python
message = spec.to_json()
restored = Spectrum.from_json(message)
```

The format is designed for APIs and browser visualization. It is not a storage
replacement for `.npz`, which is smaller and retains NumPy arrays natively.

The packaged JSON Schema is available without an extra dependency:

```python
from spxtacular import get_json_schema

schema = get_json_schema("spectrum")
```

---

### Serialisation (spectrl token)

#### `to_spectrl_token` / `Spectrum.from_spectrl_token`

```python
def to_spectrl_token(self, *, lossless: bool = False, max_len: int | None = None) -> str

@classmethod
def from_spectrl_token(cls, token: str) -> Spectrum | MsnSpectrum
```

Encode the spectrum as a [spectrl](https://github.com/pgarrett-scripps/spectrl) `spectrl.v1.…` URL-safe token, or decode one back to a `Spectrum`/`MsnSpectrum`. The token mirrors mzML semantics (PSI-MS CV params, a single CBOR document, MS-Numpress compression, CRC-32 integrity checksum) and is suitable for sharing in URLs, QR codes, notebooks, and papers.

Requires the optional `[spectrl]` extra.

```python
token = spec.to_spectrl_token()                      # lossy MS-Numpress
token_exact = spec.to_spectrl_token(lossless=True)   # bit-exact float64 + zlib
restored = Spectrum.from_spectrl_token(token)
```

The round-trip is faithful — every spxtacular field is carried. Ion mobility rides in spectrl's `extra_arrays` slot under its exact PSI-MS array accession; `iso_score` uses the same slot as a non-standard mzML binary array (`MS:1000786`) under the descriptor name `"iso_score"`. Scalar fields without an mzML CV counterpart — `denoised`/`normalized` provenance, `scan_number`, `resolution`, `analyzer`, `ramp_time`, `im_range`, `isolation_im_range` — are carried losslessly as namespaced free-text `user_params` (`spxtacular:` prefix).

#### `to_spectrl_url` / `Spectrum.from_spectrl_url`

```python
def to_spectrl_url(
    self,
    base: str | None = None,
    *,
    mode: str = "fragment",   # "fragment" | "query" | "data"
    param: str = "d",
    lossless: bool = False,
    max_len: int | None = None,
) -> str

@classmethod
def from_spectrl_url(cls, url: str) -> Spectrum | MsnSpectrum
```

Bind a token into a shareable URL (or `data:` URI), or extract and decode one. `mode="fragment"` (default) puts the token after `#` so it never reaches the server; `mode="query"` uses `base?<param>=…`; `mode="data"` emits a `data:application/vnd.spectrl;v=1,…` URI (`base` ignored). `base` is required for `"fragment"` and `"query"`.

```python
url = spec.to_spectrl_url("https://example.com/view")             # …#spectrl.v1.…
uri = spec.to_spectrl_url(mode="data")                            # data: URI
restored = Spectrum.from_spectrl_url(url)
```

---

### Persistence (.npz)

#### `save` / `Spectrum.load`

```python
def save(self, path: str | Path) -> None

@classmethod
def load(cls, path: str | Path) -> Spectrum
```

Serialise to / from a numpy `.npz` archive. Peak arrays are stored natively; scalar metadata is JSON-encoded under a `meta` key. `.npz` extension is appended automatically when missing.

```python
spec.save("scan_001.npz")
restored = Spectrum.load("scan_001.npz")
```

For `MsnSpectrum`, all MSn metadata (scan number, RT, precursors, isolation window, …) is preserved.

#### `Spectrum.from_usi`

```python
@classmethod
def from_usi(
    cls,
    usi: str,
    backend: str = "aggregator",
    timeout: float = 30,
) -> Spectrum | MsnSpectrum
```

Fetch a spectrum from a public proteomics repository via Universal Spectrum Identifier. It uses the PROXI REST API. Backends are `"aggregator"` (default), `"pride"`, `"massive"`, `"peptideatlas"`, `"jpost"`, or a full URL. It returns `MsnSpectrum` when scan-level metadata or precursor information is present.

```python
spec = Spectrum.from_usi(
    "mzspec:PXD000561:Adult_Frontalcortex_bRP_Elite_85_f09:scan:17555"
)
```

---

### Visualization

#### `plot`

```python
def plot(
    self,
    title: str | None = None,
    *,
    color: Literal["charge", "im"] | None = "charge",
    show_scores: bool = True,
    show_charges: bool | None = None,  # deprecated alias
    **layout_kwargs,
) -> Figure
```

`color`, `show_scores`, and `show_charges` are keyword-only.

Returns a Plotly `Figure` (stick plot). `plotly` is a required dependency, so no extra install is needed.

| Parameter | Description |
|---|---|
| `title` | Plot title |
| `color` | `"charge"` colours sticks by charge state, `"im"` by ion mobility on the theme's single-hue sequential scale, `None` for uniform colour |
| `show_scores` | Annotate scored peaks with their score value when an `iso_score` array is present |
| `show_charges` | Deprecated. Use `color="charge"` or `color=None` instead |

```python
spec.plot(title="My spectrum").show()
decon.plot(color="charge", show_scores=True).show()
```

See [Visualization](visualization.md) for `mirror_plot()` and `annotate_spectrum()`.

#### `annotate`

```python
def annotate(
    self,
    fragments,
    tolerance: float = 0.02,
    tolerance_type: Literal["da", "ppm"] = "da",
    title: str | None = None,
    peak_selection: Literal["closest", "largest", "all"] = "closest",
    include_sequence: bool = False,
    **layout_kwargs,
) -> Figure
```

Convenience wrapper around `annotate_spectrum()`. Plots the spectrum with matched fragment ion labels — matched peaks are coloured by ion series, unmatched peaks rendered in grey.

```python
fig = ms2.annotate(fragments, tolerance=10, tolerance_type="ppm")
fig.show()
```

#### `mass_error_plot`

```python
def mass_error_plot(
    self,
    fragments,
    tolerance: float = 0.02,
    tolerance_type: Literal["da", "ppm"] = "da",
    peak_selection: Literal["closest", "largest", "all"] = "closest",
    unit: Literal["ppm", "da"] = "ppm",
    title: str | None = None,
    **layout_kwargs,
) -> Figure
```

Bubble chart of fragment mass errors vs m/z. Each matched fragment is a bubble whose x-position is the observed m/z, y-position is the mass error (ppm or Da), and size is proportional to the peak intensity. Bubbles are coloured by ion series. Useful for spotting calibration drifts or systematic mass errors.

```python
ms2.mass_error_plot(fragments, tolerance=20, tolerance_type="ppm", unit="ppm").show()
```

#### `facet_plot`

```python
def facet_plot(
    self,
    fragments=None,
    mirror_spectrum: Spectrum | None = None,
    title: str | None = None,
    tolerance: float = 0.02,
    tolerance_type: Literal["da", "ppm"] = "da",
    peak_selection: Literal["closest", "largest", "all"] = "closest",
    include_sequence: bool = False,
    **layout_kwargs,
) -> Figure
```

Multi-panel plot on a shared m/z axis. Panel 1 is always the (optionally annotated) spectrum. Passing `fragments` adds a mass-error panel; passing `mirror_spectrum` adds an inverted mirror panel below.

```python
ms2.facet_plot(fragments, mirror_spectrum=ms2.deconvolute().decharge()).show()
```

#### `plot_table`

```python
def plot_table(
    self,
    show_charges: bool = True,
    show_scores: bool = True,
) -> pd.DataFrame
```

Returns a `pandas.DataFrame` with one row per peak. Each row contains both the raw peak data (`mz`, `intensity`, `charge`, `score`, `im`) and all visual properties (`color`, `linewidth`, `opacity`, `series`, `label`, `label_size`, `label_font`, `label_color`, `label_yshift`, `label_xanchor`, `label_angle`, `hover`). Modify the table freely, then render it with `plot_from_table()`.

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
    tolerance: float = 0.02,
    tolerance_type: Literal["da", "ppm"] = "da",
    peak_selection: Literal["closest", "largest", "all"] = "closest",
    include_sequence: bool = False,
) -> pd.DataFrame
```

Like `plot_table()` but matched peaks are coloured by ion series and labelled with their fragment identifier. Unmatched peaks are grey. Modify the returned table and call `plot_from_table()` to render.

```python
tbl = spec.annot_plot_table(fragments, tolerance=10, tolerance_type="ppm")
tbl.loc[tbl["label"] != "", "label_size"] = 14
fig = plot_from_table(tbl, title="Annotated")
fig.show()
```

See [`plot_table`](#plot_table) above for the full column list, and
[API reference — Plot table API](api.md#plot-table-api) for the module-level
`build_plot_table` / `build_annot_plot_table` / `plot_from_table` signatures.

---

### Fragment matching & PSM scoring

#### `match_fragments`

```python
def match_fragments(
    self,
    fragments,
    tolerance: float = 0.02,
    tolerance_type: Literal["da", "ppm"] = "da",
    peak_selection: Literal["closest", "largest", "all"] = "closest",
    is_monoisotopic: bool = True,
) -> list[MatchedFragment]
```

Match fragment ions against this spectrum's peaks. Thin wrapper around `spxtacular.match_fragments`. When `fragments` is a `dict[(IonType, charge_state), list[float]]` (peptacular's `fast_fragment` output), `is_monoisotopic` is forwarded to the constructed `Fragment` objects; otherwise it has no effect.

#### `score`

```python
def score(
    self,
    fragments,
    tolerance: float = 0.02,
    tolerance_type: Literal["da", "ppm"] = "da",
    peak_selection: Literal["closest", "largest", "all"] = "closest",
    predicted_intensities: Sequence[float] | None = None,
) -> dict[str, float]
```

Returns a dict with eight PSM metrics: `hyperscore`, `probability_score`, `total_matched_intensity`, `matched_fraction`, `intensity_fraction`, `mean_ppm_error`, `spectral_angle`, `longest_run`. Pass one predicted intensity per fragment to compute the literature spectral-angle metric.

---

### Precursor removal & scaling

#### `remove_precursor_peak`

```python
def remove_precursor_peak(
    self,
    precursor_mz: float | None = None,
    precursor_charge: int | None = None,
    tolerance: float = 0.02,
    tolerance_type: Literal["da", "ppm"] = "da",
    isotopes: int | Literal["auto"] = "auto",
    isotope_threshold: float = 0.01,
    remove_charge_states: bool = True,
    inplace: bool = False,
    isotope_model: IsotopeModel | IsotopeModelType | str = "peptide",
    ionization_model: IonizationModel | str | float | None = None,
) -> Self
```

Remove the precursor peak, its isotope envelope, and (optionally) all lower charge states. Adapts to spectrum state:

| State | Behaviour |
|---|---|
| **Centroid** | Removes all charge states 1..z and their isotope envelopes |
| **Deconvoluted** | Targets only the monoisotopic peak at the precursor charge (charge-aware) |
| **Decharged** | Targets the precursor neutral mass directly |
| **Profile** | Raises `ValueError` (call `.centroid()` first) |

When called on an `MsnSpectrum` without an explicit `precursor_mz`, the method auto-detects from `MsnSpectrum.precursors` and removes peaks for **all** precursors. With `isotopes="auto"`, it uses the selected isotope model to determine the significant isotopes. The ionization model defaults to recorded deconvolution provenance, then scan polarity.

```python
# Auto from MsnSpectrum.precursors
cleaned = msn.remove_precursor_peak(tolerance=10, tolerance_type="ppm")

# Explicit
cleaned = spec.remove_precursor_peak(precursor_mz=450.25, precursor_charge=2)
```

#### `scale_intensity`

```python
def scale_intensity(
    self,
    method: Literal["root", "log", "rank"] = "root",
    degree: int = 2,
    base: float = 2.0,
    inplace: bool = False,
) -> Self
```

Non-linear intensity scaling (independent of `normalize()`, which divides by a reference). `"root"` applies an nth-root transform (default sqrt), `"log"` applies `log_base(intensity + 1)`, `"rank"` replaces intensities with their rank.

#### `round_mz`

```python
def round_mz(
    self,
    decimals: int = 0,
    combine: Literal["sum", "max"] = "sum",
    inplace: bool = False,
) -> Self
```

Round m/z values to `decimals` decimals and merge duplicate peaks via `sum` or `max` of their intensities.
Clears normalization, mobility, and isotope scores. Charged spectra also lose their charge assignments
and deconvolution record. Already decharged spectra retain their zero-charge marker and provenance
so the rounded values remain identifiable as neutral masses.

---

### Combining spectra

#### `Spectrum.combine`

```python
@classmethod
def combine(cls, spectra: list[Spectrum]) -> Spectrum
```

Concatenate peaks from multiple spectra into a single new `Spectrum`, sorted by m/z ascending.
Empty inputs do not erase metadata from nonempty inputs. Optional per-peak arrays (`charge`, `im`,
`iso_score`) are carried over when every contributing input provides them. Combining multiple
nonempty inputs clears normalization. Other scalar metadata is preserved when all inputs agree.

Combining neutral masses with m/z values raises `ValueError`. Charged inputs with differing
deconvolution records also raise, including records that differ only in processing parameters.
Decharge those inputs separately before combining them, so each conversion uses its own carrier.
A charged scan whose negative polarity would be lost in the base `Spectrum` also requires this
explicit conversion.

```python
combined = Spectrum.combine([spec1, spec2, spec3])
```

The return type is always the base `Spectrum` — MSn metadata cannot be sensibly combined across scans.

#### Recipe: consensus spectrum from replicates

`combine` and `merge` together build a consensus spectrum from repeat measurements of the same
analyte: normalize each replicate so no single acquisition dominates, concatenate, then collapse
peaks that agree within tolerance into intensity-weighted centroids.

```python
replicates = [spec1, spec2, spec3]          # e.g. the same precursor across runs

consensus = Spectrum.combine(
    [s.normalize(method="tic") for s in replicates]
).merge(mz_tolerance=10, mz_tolerance_type="ppm")
```

Peaks seen in every replicate accumulate intensity across the merge window while one-off noise
peaks stay small, so a follow-up `filter(min_intensity=...)` — thresholded at, say, a fraction of
`1 / len(replicates)` of the summed TIC — keeps only reproducible signal.

---

## MsnSpectrum

`MsnSpectrum` extends `Spectrum` with instrument-level metadata. Every file reader yields it. All
`Spectrum` methods remain available.

```python
@dataclass(slots=True, kw_only=True)
class MsnSpectrum(Spectrum):
    # Scan identification
    scan_number: int | None = None
    ms_level: int | None = None
    native_id: str | None = None

    # Timing
    rt: float | None = None              # retention time, seconds
    injection_time: float | None = None  # ion accumulation time, ms
    total_ion_current: float | None = None

    # Acquisition windows (the full scan range, NOT isolation windows)
    mz_range: tuple[float, float] | None = None
    im_range: tuple[float, float] | None = None
    im_type: IMType | str | None = None       # e.g. IMType.OOK0, "drift_time_ms"

    # Instrument settings
    polarity: Polarity | Literal["positive", "negative"] | None = None

    # Optional metadata
    resolution: float | None = None
    analyzer: Analyzer | str | None = None      # e.g. Analyzer.TOF, "FTMS"
    ramp_time: float | None = None
    collision_energy: float | None = None
    activation_type: ActivationType | str | None = None
    precursors: list[Precursor] | None = None

    # MS2 precursor isolation windows
    isolation_mz_range: tuple[float, float] | None = None
    isolation_im_range: tuple[float, float] | None = None
```

All fields are keyword-only (`kw_only=True`), including the inherited `Spectrum` fields.
`mz_range` / `im_range` describe the **acquisition** window of the scan; `isolation_mz_range` /
`isolation_im_range` describe the **precursor isolation** window used to select ions for MS2.

`im_type`, `analyzer`, and `activation_type` are **open vocabulary**: an enum member gives autocomplete and typo-safety, but raw PSI-MS accessions (e.g. `"MS:1002481"` from `DReader`) and unknown vendor strings still pass through untouched. `polarity` is **closed vocabulary** — only `Polarity.POSITIVE`/`Polarity.NEGATIVE` or the literal strings `"positive"`/`"negative"` are valid. See [API reference — Metadata enums](api.md#metadata-enums) for the full member list of `Polarity`, `ActivationType`, `IMType`, and `Analyzer`.

### Precursor

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class Precursor(Peak):
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


---

## Centroiding

```python
spec.centroid()                          # every local maximum becomes a peak
spec.centroid(min_intensity="noise")     # MAD-estimated floor
spec.centroid(min_intensity=2000)        # absolute floor
```

`min_intensity` is the floor an apex must clear to count as a peak.

!!! warning
    Without one, **every** local maximum is a peak. On data with any noise floor that returns far
    more centroids than there are real peaks — a test spectrum with 6 real peaks and a modest noise
    floor gives 769 centroids at the default, 6 with `min_intensity=2000`. The default is `None`
    for backwards compatibility, but on real data you almost always want a floor.

Use [`profile_centroid_plot()`](visualization.md#checking-centroiding) to see what the centroider
actually did.

A flat apex counts as one peak: requiring a strict `prev < curr > next` would discard any peak
whose maximum spans two or more equal samples, which is routine in quantised or saturated data.
