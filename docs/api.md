# API Reference

Every name in `spxtacular.__all__`, grouped by area:

```python
from spxtacular import (
    # Core data structures
    Spectrum, MsnSpectrum, Peak, Precursor,
    # Enums and their permissive type aliases
    ToleranceType, ToleranceLike,
    PeakSelection, PeakSelectionLike,
    Polarity, PolarityLike,
    ActivationType, ActivationTypeLike,
    IMType, IMTypeLike,
    Analyzer, AnalyzerLike,
    # Readers
    Reader, DReader, MzmlReader, CentroidConfig,
    # Matching and scoring
    match_fragments, score,
    # Visualization
    plot_spectrum, mirror_plot, annotate_spectrum, mass_error_plot, facet_plot,
    # Plot tables
    build_plot_table, build_annot_plot_table, plot_from_table,
    # Utilities
    da_to_ppm, ppm_to_da,
    # Remote / serialised spectra
    fetch_usi,
    to_inline_spectrum,
    to_spectrl_token, from_spectrl_token,
    to_spectrl_url, from_spectrl_url,
)
```

`SpectrumType`, `AcquisitionType`, `MatchedFragment`, `estimate_noise_level`, and the reader lookup
classes are **not** exported from the package root; import them from their defining modules as shown
in the relevant sections below.

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
    iso_score: NDArray[np.float64] | None = None,
    spectrum_type: SpectrumType | str | None = None,
    denoised: str | None = None,
    normalized: str | None = None,
)
```

`iso_score` sits between `im` and `spectrum_type` — pass the trailing fields by keyword to avoid
positional mix-ups.

**Methods:**

| Method | Returns | Summary |
|---|---|---|
| `.peaks` | `list[Peak]` | All peaks as `Peak` objects |
| `.is_decharged` | `bool` | Property: `True` once every peak's `charge == 0` |
| `len(spec)` | `int` | Number of peaks (`__len__`) |
| `.top_peaks(n, by, reverse)` | `list[Peak]` | Top N peaks sorted by attribute |
| `.has_peak(target_mz, ...)` | `bool` | Check for a peak near target m/z |
| `.get_peak(target_mz, ...)` | `Peak \| None` | Single best-matching peak |
| `.get_peaks(target_mz, ...)` | `list[Peak]` | All peaks matching criteria |
| `.filter(...)` | `Spectrum` | Remove peaks outside bounds |
| `.normalize(method)` | `Spectrum` | Scale intensities (max / tic / median) |
| `.scale_intensity(method, ...)` | `Spectrum` | Non-linear scaling: `"root"`, `"log"`, `"rank"` |
| `.denoise(method)` | `Spectrum` | Remove peaks below noise threshold |
| `.centroid()` | `Spectrum` | Convert profile to centroid via Gaussian fit |
| `.merge(mz_tolerance, mz_tolerance_type, im_tolerance, im_tolerance_type)` | `Spectrum` | Merge nearby peaks by weighted average |
| `.round_mz(decimals, combine)` | `Spectrum` | Round m/z, then sum / max-reduce duplicates |
| `.deconvolute(...)` | `Spectrum` | Assign isotope clusters and charge states |
| `.decharge()` | `Spectrum` | Convert charged m/z to neutral masses |
| `.remove_precursor_peak(...)` | `Spectrum` | Strip precursor + isotopes + charge states |
| `.sort(by, reverse)` | `Spectrum` | Reorder peaks by attribute |
| `.copy()` | `Spectrum` | Deep copy with all arrays duplicated |
| `Spectrum.combine(spectra)` | `Spectrum` | Classmethod: concatenate multiple spectra |
| `.match_fragments(fragments, ...)` | `list[MatchedFragment]` | Fragment-to-peak matching |
| `.score(fragments, ...)` | `dict[str, float]` | All PSM scores |
| `.to_spectrl_token(...)` | `str` | Encode as a `spectrl1.…` URL-safe token (requires `[spectrl]` extra) |
| `Spectrum.from_spectrl_token(t)` | `Spectrum` | Decode a `spectrl1.…` token (classmethod) |
| `.to_spectrl_url(base, mode, ...)` | `str` | Encode as a shareable URL or `data:` URI (requires `[spectrl]` extra) |
| `Spectrum.from_spectrl_url(url)` | `Spectrum` | Decode a token from a URL fragment, query, or `data:` URI (classmethod) |
| `Spectrum.from_usi(usi, ...)` | `Spectrum` | Fetch via PROXI from USI (classmethod) |
| `.save(path)` | `None` | Serialise to `.npz` |
| `Spectrum.load(path)` | `Spectrum` | Load from `.npz` (classmethod) |
| `.update(**kwargs)` | `Spectrum` | Return copy with specified fields replaced |
| `.plot(title, color, show_scores, ...)` | `go.Figure` | Stick plot (requires plotly) |
| `.annotate(fragments, ...)` | `go.Figure` | Plot with fragment annotations |
| `.mass_error_plot(fragments, ...)` | `go.Figure` | Bubble chart of fragment mass errors |
| `.facet_plot(fragments, mirror_spectrum, ...)` | `go.Figure` | Multi-panel facet plot |
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
| `total_ion_current` | `float \| None` | Total ion current for the scan |
| `mz_range` | `tuple[float, float] \| None` | Acquisition m/z window |
| `im_range` | `tuple[float, float] \| None` | Ion mobility acquisition window |
| `isolation_mz_range` | `tuple[float, float] \| None` | MS2 precursor isolation window (m/z) |
| `isolation_im_range` | `tuple[float, float] \| None` | MS2 precursor isolation window (ion mobility) |
| `im_type` | `IMType \| str \| None` | Ion mobility unit (open vocabulary) |
| `polarity` | `Polarity \| "positive" \| "negative" \| None` | Scan polarity (closed vocabulary) |
| `resolution` | `float \| None` | Instrument resolution |
| `analyzer` | `Analyzer \| str \| None` | Mass analyser type (open vocabulary) |
| `ramp_time` | `float \| None` | timsTOF ramp time in ms |
| `collision_energy` | `float \| None` | Fragmentation collision energy |
| `activation_type` | `ActivationType \| str \| None` | Fragmentation type (open vocabulary) |
| `precursors` | `list[Precursor] \| None` | Precursor ions (MS2 only) |

See [Metadata enums](#metadata-enums) below for the `Polarity`, `ActivationType`, `IMType`, and `Analyzer` member lists.

Full documentation: [Spectrum reference — MsnSpectrum](spectrum.md#msnspectrum)

---

### Metadata enums

Four `StrEnum`s are exported from `spxtacular` root and back the `MsnSpectrum` fields above:

```python
from spxtacular import Polarity, ActivationType, IMType, Analyzer
```

| Enum | Vocabulary | Members |
|---|---|---|
| `Polarity` | Closed | `POSITIVE` (`"positive"`), `NEGATIVE` (`"negative"`) |
| `ActivationType` | Open | `CID`, `HCD`, `ETD`, `ECD`, `ETHCD` (`"EThcD"`), `ETCID` (`"ETciD"`), `NETD`, `UVPD`, `PD`, `PQD`, `SID`, `IRMPD`, `BIRD`, `SORI`, `PASEF` |
| `IMType` | Open | `OOK0` (`"ook0"`), `IM` (`"im"`), `DRIFT_TIME_MS` (`"drift_time_ms"`), `CCS` (`"ccs"`) |
| `Analyzer` | Open | `ORBITRAP`, `FT_ICR`, `TOF`, `QUADRUPOLE`, `ION_TRAP`, `LINEAR_ION_TRAP`, `QUADRUPOLE_ION_TRAP`, `MAGNETIC_SECTOR`, `ELECTROSTATIC_ENERGY_ANALYZER` |

`Polarity` is closed vocabulary: `MsnSpectrum.polarity` only accepts a `Polarity` member or the literal strings `"positive"`/`"negative"`. The other three are open vocabulary — `MsnSpectrum.im_type`, `.analyzer`, and `.activation_type` are typed `Enum | str`, so an enum member gives autocomplete/typo-safety while raw PSI-MS accessions (e.g. `"MS:1002481"` from `DReader`) and unknown vendor strings still pass through unchanged.

```python
from spxtacular import MsnSpectrum, ActivationType

spec = MsnSpectrum(mz=mz, intensity=intensity, activation_type=ActivationType.HCD)
```

#### Type aliases

Every enum ships a `…Like` alias — the permissive union that the library's own parameters and
fields are annotated with. Use them when you type your own wrappers so callers can pass either an
enum member or a plain string.

```python
from spxtacular import ToleranceLike, PeakSelectionLike, PolarityLike
from spxtacular import ActivationTypeLike, IMTypeLike, AnalyzerLike
```

| Alias | Definition |
|---|---|
| `ToleranceLike` | `ToleranceType \| Literal["da", "ppm"]` |
| `PeakSelectionLike` | `PeakSelection \| Literal["closest", "largest", "all"]` |
| `PolarityLike` | `Polarity \| Literal["positive", "negative"]` |
| `ActivationTypeLike` | `ActivationType \| str` |
| `IMTypeLike` | `IMType \| str` |
| `AnalyzerLike` | `Analyzer \| str` |

The first three are closed unions (only the listed literals type-check); the last three are open —
any `str` is accepted so raw PSI-MS accessions and vendor shorthands pass through.

`ToleranceType` (`DA` = `"da"`, `PPM` = `"ppm"`) and `PeakSelection` (`CLOSEST`, `LARGEST`, `ALL`)
are the processing-side enums behind the `tolerance_type` and `peak_selection` parameters used
throughout matching, scoring, and plotting.

---

### `Peak`

Frozen dataclass for a single spectral peak. Returned by peak access methods.

```python
Peak(mz: float, intensity: float, charge: int | None = None, im: float | None = None, iso_score: float | None = None)
```

---

### `Precursor`

Frozen dataclass, subclass of `Peak`, representing a selected precursor ion. Exported from the package root: `from spxtacular import Precursor`.

```python
# mz, intensity, charge, im, iso_score are inherited from Peak (positional-or-keyword);
# is_monoisotopic is keyword-only (kw_only=True) and has no default
Precursor(mz=..., intensity=..., charge=..., im=..., iso_score=..., is_monoisotopic=...)
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

`.ms1` and `.ms2` are **lookup objects**, not generators: iterable *and* indexable, but not
iterators. `next(reader.ms2)` raises `TypeError` — use `next(iter(reader.ms2))`.

### `Reader`

Format-agnostic entry point. Detects `.d` (Bruker timsTOF) or `.mzML` from the path suffix and
delegates to `DReader` / `MzmlReader`; any other suffix raises `ValueError`.

```python
Reader(path: str | Path, centroid_config: CentroidConfig | None = None)
```

| Property / Method | Type | Description |
|---|---|---|
| `.ms1` | `DReaderMs1Lookup \| MzmlSpectraLookup` | MS1 spectra — iterate or index |
| `.ms2` | `DReaderMs2Lookup \| MzmlSpectraLookup` | MS2 spectra — iterate or index |
| `.open()` / `.close()` | `None` | Open / close the delegate; also driven by `with` |

```python
from spxtacular import Reader

with Reader("run.mzML") as r:   # or Reader("/data/sample.d")
    for spec in r.ms1:
        ...
```

Full documentation: [Readers — Reader](readers.md#reader)

---

### `MzmlReader`

Reads `.mzML` files. A context manager is optional but recommended — it keeps one file handle open
instead of reopening the file per operation.

```python
MzmlReader(mzml_path: str | Path)
```

| Property / Method | Type | Description |
|---|---|---|
| `.ms1` | `MzmlSpectraLookup` | MS1 spectra — iterate, or index by overall index / native ID |
| `.ms2` | `MzmlSpectraLookup` | MS2 spectra — iterate, or index by overall index / native ID |
| `reader[key]` | `MsnSpectrum` | Spectrum by 0-based index (`reader[0]`) or native ID (`reader["scan=19"]`) |
| `.open()` / `.close()` | `None` | Open / close the persistent `mzmlpy` handle |

Index access is not MS-level filtered — `reader.ms2[0]` is the first spectrum in the file, not the
first MS2 spectrum.

Full documentation: [Readers — MzmlReader](readers.md#mzmlreader)

---

### `DReader`

Reads Bruker timsTOF `.d` directories. **Must be opened before use** — via `open()`/`close()` or, preferably, as a context manager.

```python
DReader(analysis_dir: str | Path, centroid_config: CentroidConfig | None = None)
```

| Property / Attribute | Type | Description |
|---|---|---|
| `.ms1` | `DReaderMs1Lookup` | All MS1 frames — iterate, or index by tdfpy `frame_id` |
| `.ms2` | `DReaderMs2Lookup` | All MS2 spectra — iterate, or index by tdfpy `precursor_id` (DDA only; DIA/PRM raise `NotImplementedError`) |
| `.acquisition_type` | `AcquisitionType` | DDA / DIA / PRM / UNKNOWN |
| `.open()` / `.close()` | `None` | Open / close the underlying `tdfpy` reader |

Full documentation: [Readers — DReader](readers.md#dreader)

---

### `CentroidConfig`

Dataclass of parameters forwarded to `tdfpy`'s `frame.centroid()`. Only used by `DReader` (and by
`Reader` for `.d` inputs); ignored by `MzmlReader`.

```python
from spxtacular import CentroidConfig

CentroidConfig(
    mz_tolerance: float = 8.0,
    mz_tolerance_type: Literal["ppm", "da"] = "ppm",
    im_tolerance: float = 0.1,
    im_tolerance_type: Literal["relative", "absolute"] = "relative",
    min_peaks: int = 3,
    noise_filter: Literal["mad", "percentile", "histogram", "baseline", "iterative_median"]
                  | float | None = None,
)
```

Full documentation: [Readers — CentroidConfig](readers.md#centroidconfig)

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

## Token serialisation (spectrl)

The single supported wire format for sharing a spectrum as a string is the
[spectrl](https://github.com/tacular-omics/spectrl) token. Encodes a full
spectrum (peaks, metadata, precursors) into a compact URL-safe token that
mirrors mzML semantics, with PSI-MS CV params, a single CBOR document,
MS-Numpress compression, and a SHA-256 integrity hash.

Requires the optional ``[spectrl]`` extra.

```python
from spxtacular import to_spectrl_token, from_spectrl_token, to_inline_spectrum

token = spec.to_spectrl_token()                      # lossy MS-Numpress, default
token_exact = spec.to_spectrl_token(lossless=True)   # bit-exact float64 + zlib
restored = Spectrum.from_spectrl_token(token)
inline = to_inline_spectrum(spec)                    # → spectrl.InlineSpectrum
```

The round-trip is faithful — every spxtacular field is carried.

Via mzML-native CV params: `mz`, `intensity`, `charge` (including singletons),
`im` + `im_type`, spectrum type, and — for `MsnSpectrum` — `native_id`,
`ms_level`, `polarity`, `rt`, `mz_range`, `total_ion_current`, `precursors`,
`isolation_mz_range`, `collision_energy`, `activation_type`.

`iso_score` rides in spectrl's `extra_arrays` slot under key `"iso_score"`
(encoded as a non-standard mzML binary array `MS:1000786`).

spxtacular scalar fields without an mzML CV counterpart —
`denoised`/`normalized` provenance strings, `scan_number`, `resolution`,
`analyzer`, `ramp_time`, `im_range`, `isolation_im_range` — are carried
losslessly as namespaced free-text `user_params` (`spxtacular:` prefix).

### URL sharing

`to_spectrl_url` / `from_spectrl_url` bind a token into a shareable link (or
decode one back). Also available as `Spectrum.to_spectrl_url` /
`Spectrum.from_spectrl_url`.

```python
from spxtacular import to_spectrl_url, from_spectrl_url

url  = to_spectrl_url(spec, "https://example.com/view")            # fragment (default)
url  = to_spectrl_url(spec, "https://example.com/view", mode="query", param="d")
uri  = to_spectrl_url(spec, mode="data")                          # data: URI, no base
spec = from_spectrl_url(url)                                       # extract + decode
```

`mode` selects the binding:

| `mode` | Result | `base` |
|---|---|---|
| `"fragment"` (default) | `base#spectrl1.…` — token in the URL fragment (never sent to the server) | required |
| `"query"` | `base?<param>=spectrl1.…` — token as a query parameter | required |
| `"data"` | `data:application/vnd.spectrl;v=1,…` URI | ignored |

`lossless` and `max_len` are forwarded to the token encoder.

---

## USI loading

Fetch spectra from public proteomics repositories by Universal Spectrum
Identifier via the PROXI protocol.

```python
from spxtacular import fetch_usi
# or via Spectrum.from_usi(...) for the same result

spec = fetch_usi(
    "mzspec:PXD000561:Adult_Frontalcortex_bRP_Elite_85_f09:scan:17555",
    backend="aggregator",  # or "pride", "massive", "peptideatlas", "jpost", or a full URL
    timeout=30,
)
```

Returns an `MsnSpectrum` when the response includes precursor info, otherwise a
plain `Spectrum`.

---

## Persistence (.npz)

Serialise spectra to / from numpy `.npz` archives. Arrays are stored natively;
scalar metadata is JSON-encoded under the `meta` key. The `.npz` extension is
appended automatically when missing.

```python
spec.save("scan_001.npz")
restored = Spectrum.load("scan_001.npz")

msn.save("scan_001.npz")
restored_msn = MsnSpectrum.load("scan_001.npz")
```

`MsnSpectrum.save` / `MsnSpectrum.load` preserve all MSn metadata (scan number,
RT, precursors, isolation window, …) in addition to the peak arrays.

---

## Visualization

`plotly` is a required dependency of `spxtacular`, so these are available out of the box — nothing
extra to install. All of them return a `plotly.graph_objects.Figure`.

Full documentation: [Visualization](visualization.md)

### `plot_spectrum`

```python
from spxtacular import plot_spectrum
```

```python
plot_spectrum(
    spectrum: Spectrum,
    title: str | None = None,
    *,
    color: Literal["charge", "im"] | None = "charge",
    show_scores: bool = True,
    show_charges: bool | None = None,  # deprecated alias of color
    **layout_kwargs,
)
```

`color`, `show_scores`, and `show_charges` are keyword-only.

``color="charge"`` (default) colours sticks by charge state. ``color="im"``
colours by ion mobility on a Viridis scale (falls back to ``"charge"`` when no
IM array is present). ``color=None`` renders all sticks in a uniform colour.
``show_charges`` is kept as a deprecated alias mapping to ``color="charge"`` /
``color=None``.

### `mirror_plot`

```python
from spxtacular import mirror_plot
```

```python
mirror_plot(
    raw: Spectrum,
    deconvoluted: Spectrum,
    title: str | None = None,
    normalize: bool = True,
    show_charges: bool = True,
    show_scores: bool = True,
    **layout_kwargs,
)
```

The second parameter is named `deconvoluted`. `show_charges` colours the deconvoluted (upper) half
by charge state; `show_scores` annotates its peaks with their isotope profile score.

### `annotate_spectrum`

```python
from spxtacular import annotate_spectrum
```

```python
annotate_spectrum(
    spectrum: Spectrum,
    fragments,
    tolerance: float = 0.02,
    tolerance_type: Literal["da", "ppm"] = "da",
    title: str | None = None,
    peak_selection: Literal["closest", "largest", "all"] = "closest",
    include_sequence: bool = False,
    **layout_kwargs,
)
```

### `mass_error_plot`

```python
from spxtacular import mass_error_plot
```

```python
mass_error_plot(
    spectrum: Spectrum,
    fragments,
    tolerance: float = 0.02,
    tolerance_type: Literal["da", "ppm"] = "da",
    peak_selection: Literal["closest", "largest", "all"] = "closest",
    unit: Literal["ppm", "da"] = "ppm",
    title: str | None = None,
    **layout_kwargs,
)
```

### `facet_plot`

```python
from spxtacular import facet_plot
```

```python
facet_plot(
    spectrum: Spectrum,
    fragments=None,
    mirror_spectrum: Spectrum | None = None,
    title: str | None = None,
    tolerance: float = 0.02,
    tolerance_type: Literal["da", "ppm"] = "da",
    peak_selection: Literal["closest", "largest", "all"] = "closest",
    include_sequence: bool = False,
    unit: str = "ppm",
    **layout_kwargs,
)
```

Takes a **single** spectrum, not a list. The optional second spectrum is `mirror_spectrum`.

---

## Utilities

```python
from spxtacular import da_to_ppm, ppm_to_da
```

```python
da_to_ppm(delta_mz: float, mz: float) -> float   # delta_mz / mz * 1e6
ppm_to_da(delta_ppm: float, mz: float) -> float  # delta_ppm * mz / 1e6
```

Convert a mass difference between Dalton and ppm at a given reference `mz`. Both take the
*difference* first and the reference m/z second.

```python
da_to_ppm(0.01, 500.0)   # 20.0 ppm
ppm_to_da(20.0, 500.0)   # 0.01 Da
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
    tolerance: float = 0.02,
    tolerance_type: Literal["da", "ppm"] = "da",
    peak_selection: Literal["closest", "largest", "all"] = "closest",
    is_monoisotopic: bool = True,
) -> list[MatchedFragment]
```

When `fragments` is a `dict[tuple[IonType, int], list[float]]` (the output of
`peptacular.ProFormaAnnotation.fast_fragment`), `is_monoisotopic` is forwarded
to the `Fragment` constructor; otherwise it has no effect. Returns a list sorted
by ascending `peak_index`.

### `score`

```python
from spxtacular import score
```

```python
score(
    spectrum: Spectrum,
    fragments,
    tolerance: float = 0.02,
    tolerance_type: Literal["da", "ppm"] = "da",
    peak_selection: Literal["closest", "largest", "all"] = "closest",
) -> dict[str, float]
```

Returns a dict of PSM metrics: `hyperscore`, `probability_score`, `total_matched_intensity`, `matched_fraction`, `intensity_fraction`, `mean_ppm_error`, `spectral_angle`, `longest_run`.

---

## Plot table API

Provides an intermediate `pandas.DataFrame` that holds all data and visual properties for a spectrum
plot. Users can freely modify the DataFrame before passing it to `plot_from_table`. `pandas` is a
required dependency, so this API is always available.

Full documentation: [Spectrum reference — `plot_table`](spectrum.md#plot_table)

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
    tolerance: float = 0.02,
    tolerance_type: Literal["da", "ppm"] = "da",
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
