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
    # Readers and peak-list writers
    Reader, DReader, MzmlReader, ThermoReader, CentroidConfig,
    MgfReader, Ms2Reader, MspReader, write_mgf, write_ms2, write_msp,
    # Matching and scoring
    match_fragments, score,
    # Isotope envelopes and average-composition models
    IsotopeModel, IsotopeModelType, IsotopeModelLike,
    ISOTOPE_MODELS,
    PEPTIDE_ISOTOPE_MODEL, GLYCAN_ISOTOPE_MODEL, LIPID_ISOTOPE_MODEL,
    DNA_ISOTOPE_MODEL, RNA_ISOTOPE_MODEL,
    brain_isotopic_distribution, resolve_isotope_model,
    # Visualization
    plot_spectrum, mirror_plot, annotate_spectrum, mass_error_plot, facet_plot,
    sequence_coverage_plot, save_figure,
    # Plot tables
    build_plot_table, build_annot_plot_table, plot_from_table, table_view,
    # Theme (a submodule, not a function)
    theme,
    # Utilities
    da_to_ppm, ppm_to_da,
    # Remote / serialised spectra
    fetch_usi, spectrum_from_proxi_response,
    to_inline_spectrum,
    to_spectrl_token, from_spectrl_token,
    to_spectrl_url, from_spectrl_url,
    # Ecosystem interoperability
    to_matchms, from_matchms,
    to_spectrum_utils, from_spectrum_utils,
)
```

`SpectrumType`, `AcquisitionType`, `MatchedFragment`, `estimate_noise_level`, and the reader lookup
classes (including `PeakListLookup`) are **not** exported from the package root; import them from
their defining modules as shown in the relevant sections below.

---

## Ecosystem interoperability

The adapters import their optional dependency only when called. Install `spxtacular[matchms]`,
`spxtacular[spectrum-utils]`, or `spxtacular[interop]` for both.

### `to_matchms` / `from_matchms`

```python
to_matchms(
    spectrum: Spectrum,
    *,
    extra_metadata: Mapping[str, object] | None = None,
    include_spxtacular_metadata: bool = True,
) -> matchms.Spectrum

from_matchms(
    spectrum: matchms.Spectrum,
    *,
    prefer_spxtacular_metadata: bool = True,
) -> Spectrum
```

`to_matchms` stable-sorts peaks by m/z and populates matchms fields including `id`,
`precursor_mz`, `charge`, `retention_time`, `ionmode`, `scan_number`, and `collision_energy`.
`extra_metadata` adds values outside spxtacular's model, such as `smiles` or `inchikey`;
spxtacular-derived values win on a key collision.

By default, `spxtacular_metadata` contains a namespaced JSON payload with all spxtacular metadata
and the per-peak `charge`, `im`, and `iso_score` arrays. A matchms operation that removes peaks is
supported: return conversion aligns surviving m/z values with those arrays. If an operation changes
m/z values and alignment is no longer possible, the extension arrays are dropped with a warning
rather than attached to the wrong peaks. Set `include_spxtacular_metadata=False` for a conventional,
intentionally lossy matchms object; set `prefer_spxtacular_metadata=False` to ignore an existing
payload while importing.

### `to_spectrum_utils` / `from_spectrum_utils`

```python
to_spectrum_utils(
    spectrum: MsnSpectrum,
    *,
    precursor_index: int = 0,
    identifier: str | None = None,
    warn_on_loss: bool = True,
) -> spectrum_utils.spectrum.MsmsSpectrum

from_spectrum_utils(
    spectrum: spectrum_utils.spectrum.MsmsSpectrum,
    *,
    warn_on_loss: bool = True,
) -> MsnSpectrum
```

The target model requires centroided peaks, one precursor m/z and charge, and an identifier.
`identifier` falls back to `native_id`, then `scan=<scan_number>`; absent required information
raises instead of being invented. Use `precursor_index` when an `MsnSpectrum` has multiple
precursors.

This conversion is explicitly lossy. spectrum_utils has no fields for per-peak charge, ion
mobility, isotope score, multiple precursors, or most acquisition metadata, and internally stores
intensities as `float32`. Populated unsupported fields produce a `UserWarning`. ProForma
annotations applied by spectrum_utils are likewise warned about and dropped by
`from_spectrum_utils`, because `MsnSpectrum` has no persistent annotation field.

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
    deconvolution: DeconvolutionProvenance | None = None,
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
| `.filter(...)` | `Spectrum` | Remove peaks outside bounds; `top_n` / `top_n_per_window=(n, width)` keep the most intense peaks globally or per fixed-width m/z window |
| `.normalize(method)` | `Spectrum` | Scale intensities (max / tic / median) |
| `.scale_intensity(method, ...)` | `Spectrum` | Non-linear scaling: `"root"`, `"log"`, `"rank"` |
| `.denoise(method)` | `Spectrum` | Remove peaks below noise threshold |
| `.centroid()` | `Spectrum` | Convert profile to centroid via Gaussian fit |
| `.merge(mz_tolerance, mz_tolerance_type, im_tolerance, im_tolerance_type)` | `Spectrum` | Merge nearby peaks by weighted average |
| `.round_mz(decimals, combine)` | `Spectrum` | Round m/z, then sum / max-reduce duplicates |
| `.deconvolute(..., ionization_model=...)` | `Spectrum` | Assign isotope clusters and charge magnitudes using a selected adduct/carrier model |
| `.decharge(..., ionization_model=...)` | `Spectrum` | Convert charged m/z to neutral masses, reusing recorded ionization provenance by default |
| `.remove_precursor_peak(...)` | `Spectrum` | Strip precursor + isotopes + charge states |
| `.sort(by, reverse)` | `Spectrum` | Reorder peaks by attribute |
| `.copy()` | `Spectrum` | Deep copy with all arrays duplicated |
| `Spectrum.combine(spectra)` | `Spectrum` | Classmethod: concatenate multiple spectra |
| `.match_fragments(fragments, ...)` | `list[MatchedFragment]` | Fragment-to-peak matching |
| `.score(fragments, ...)` | `dict[str, float]` | All PSM scores |
| `.to_spectrl_token(...)` | `str` | Encode as a `spectrl2.…` URL-safe token (requires `[spectrl]` extra) |
| `Spectrum.from_spectrl_token(t)` | `Spectrum` | Decode a `spectrl2.…` token (classmethod) |
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

Format-agnostic entry point. Detects `.d` (Bruker timsTOF), `.mzML`, `.raw` (Thermo), `.mgf`,
`.ms2`, or `.msp` from the path suffix — a trailing `.gz` is stripped first — and delegates to `DReader` /
`MzmlReader` / `ThermoReader` / `MgfReader` / `Ms2Reader` / `MspReader`; any other suffix raises
`ValueError`.

```python
Reader(path: str | Path, centroid_config: CentroidConfig | None = None)
```

| Property / Method | Type | Description |
|---|---|---|
| `.ms1` | `DReaderMs1Lookup \| MzmlSpectraLookup \| ThermoScanLookup \| PeakListLookup` | MS1 spectra — iterate or index (empty for `.mgf` / `.ms2`) |
| `.ms2` | `DReaderMs2Lookup \| MzmlSpectraLookup \| ThermoScanLookup \| PeakListLookup` | MS2 spectra — iterate or index |
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

### `ThermoReader`

Reads Thermo `.raw` files via `fisher-py` (Thermo's RawFileReader .NET assemblies — a .NET runtime
must be installed on the machine). **Must be opened before use** — via `open()`/`close()` or,
preferably, as a context manager.

```python
ThermoReader(raw_path: str | Path, prefer_vendor_centroid: bool = True)
```

| Property / Method | Type | Description |
|---|---|---|
| `.ms1` | `ThermoScanLookup` | MS1 spectra — iterate, or index by native 1-based scan number |
| `.ms2` | `ThermoScanLookup` | MS2 spectra — iterate, or index by native 1-based scan number |
| `reader[scan]` | `MsnSpectrum` | Spectrum of any MS level by native scan number |
| `.open()` / `.close()` | `None` | Open / release the RawFileReader handle |

With `prefer_vendor_centroid=True` (default), profile-mode FTMS scans yield Thermo's own centroid
stream (`CENTROID`, with per-peak charge annotations — unknown charge arrives as `-1`); with
`False`, they yield the full `PROFILE` trace.

Full documentation: [Readers — ThermoReader](readers.md#thermoreader)

---

### `MgfReader` / `Ms2Reader` / `MspReader`

Read the MGF, MS2, and MSP (NIST spectral-library) peak-list formats. Pure standard library — no
optional extra, always available. All three formats hold fragmentation spectra only: every
spectrum comes back with `ms_level=2` and `spectrum_type=SpectrumType.CENTROID`. Gzip is detected
by magic bytes, so `.mgf.gz` / `.ms2.gz` / `.msp.gz` just work. `MspReader` handles both the
NIST/SpectraST peptide-library dialect and metabolomics exports (MoNA, GNPS, MS-DIAL) with
case-insensitive header matching.

```python
MgfReader(path: str | Path)
Ms2Reader(path: str | Path)
MspReader(path: str | Path)
```

| Property / Method | Type | Description |
|---|---|---|
| `iter(reader)` | `Iterator[MsnSpectrum]` | Every spectrum, in file order |
| `len(reader)` | `int` | Spectra in the file — one counting pass, then cached |
| `reader[key]` | `MsnSpectrum` | Spectrum by 0-based position (`reader[0]`) or `native_id` (`reader["scan=19"]`); O(n) |
| `.ms1` | `PeakListLookup` | Always empty — peak lists carry no survey scans |
| `.ms2` | `PeakListLookup` | Every spectrum — iterate or index |
| `.open()` / `.close()` | `None` | `open()` checks the file exists; `close()` is a no-op (each walk streams its own handle) |

Malformed input raises `ValueError` naming the file and line number. Unknown headers, multi-charge
values, comment lines, and peak-less blocks are tolerated.

Full documentation: [Readers — MGF / MS2 / MSP](readers.md#mgf-ms2-msp)

---

### `write_mgf` / `write_ms2` / `write_msp`

Write spectra to a peak-list file, returning the path. A `.gz` suffix gzips the output.

```python
write_mgf(spectra: Iterable[Spectrum] | Spectrum, path: str | Path) -> Path
write_ms2(spectra: Iterable[Spectrum] | Spectrum, path: str | Path) -> Path
write_msp(spectra: Iterable[Spectrum] | Spectrum, path: str | Path) -> Path
```

| Behaviour | Detail |
|---|---|
| Peak values | `mz` / `intensity` at repr precision — a write → read round trip is exact |
| `SpectrumType.PROFILE` | Raises `ValueError`; peak lists are centroid data |
| Polarity | MGF/MS2: carried by the sign of the written charge (`CHARGE=2-`, `Z -2`). MSP: explicit `Ion_mode: P`/`N` line |
| Missing metadata | Omitted, except MS2's mandatory `S` fields (scan number → 1-based position, precursor m/z → `0.0`) |

Full documentation: [Readers — Writing](readers.md#writing)

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
| `"fragment"` (default) | `base#spectrl2.…` — token in the URL fragment (never sent to the server) | required |
| `"query"` | `base?<param>=spectrl2.…` — token as a query parameter | required |
| `"data"` | `data:application/vnd.spectrl;v=1,…` URI | ignored |

`lossless` and `max_len` are forwarded to the token encoder.

---

## USI loading

Fetch spectra from public proteomics repositories by Universal Spectrum
Identifier via the PROXI protocol.

```python
from spxtacular import fetch_usi, spectrum_from_proxi_response
# or via Spectrum.from_usi(...) for the same result

spec = fetch_usi(
    "mzspec:PXD000561:Adult_Frontalcortex_bRP_Elite_85_f09:scan:17555",
    backend="aggregator",  # or "pride", "massive", "peptideatlas", "jpost", or a full URL
    timeout=30,
)

# For clients that perform the HTTP request themselves:
spec = spectrum_from_proxi_response(decoded_proxi_json, usi)
```

The parser preserves PROXI centroid/profile representation and scan polarity
metadata. It returns an `MsnSpectrum` when scan-level metadata or precursor
information is available, otherwise a plain `Spectrum`.

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

### Conventions shared by every figure

These defaults apply to all the plotting functions below; they are described once here rather than
repeated in each parameter table.

| Behaviour | Detail |
|---|---|
| **Relative intensity by default** | Table-driven figures (`plot_spectrum`, `annotate_spectrum`, `plot_from_table`) scale the y-axis so the base peak is 100% and title the axis `Relative intensity (%)`. Pass `intensity_scale="absolute"` for raw counts. Tooltips always report the **true** intensity, whatever the scaling. |
| **Optional intensity transform** | `intensity_transform="sqrt"` or `"log"` compresses a range spanning orders of magnitude; the axis title is prefixed accordingly (`√ relative intensity (%)`, `log₁₀ …`). |
| **Labels are vertical, capped and collision-avoided** | Labels are rotated to read bottom-to-top (`label_angle`, default `-90`), the spectrum-viewer convention — a rotated label occupies about one line-height rather than its full text width, so several times as many peaks can be labelled. `max_labels` (default `60`) keeps only the strongest, and any label falling within 0.9% of the m/z span of a stronger one is dropped. `max_labels=None` removes the count cap but *not* the collision pass. Dropped values stay in the hover text, in the plot table, and in `table_view()`. |
| **Hovering does not require precision** | Table-driven figures carry a transparent hit layer of 22px markers on the peak tips — the sticks themselves are `hoverinfo="skip"` — so being *near* a peak is enough rather than landing on a 1.6px hairline. Every figure additionally gets the m/z crosshair from the theme template (`showspikes`, `spikemode="across"`, snapped to the cursor), with `hoverdistance=24`. |
| **Autosize** | Figures fill their container (`autosize=True`, from the template) rather than a fixed pixel box, so they lay out correctly in notebooks and docs pages. |
| **Theme** | Every function takes `theme_mode="light" \| "dark"`; `None` (default) uses the module default from `theme.set_plot_theme()`. See [Theme](#theme). |

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
    max_labels: int | None = 60,
    theme_mode: Literal["light", "dark"] | None = None,
    intensity_scale: Literal["absolute", "relative"] = "relative",
    intensity_transform: Literal["sqrt", "log"] | None = None,
    show_precursor: bool = True,
    **layout_kwargs,
)
```

Everything after `title` is keyword-only.

| Parameter | Default | Description |
|---|---|---|
| `color` | `"charge"` | `"charge"` colours sticks by charge state on the ordinal ramp; `"im"` colours by ion mobility on the single-hue sequential scale with a colourbar (falls back to `"charge"` when no IM array is present); `None` renders every stick in one colour |
| `show_scores` | `True` | Label peaks whose `iso_score > 0` with their score |
| `show_charges` | `None` | Deprecated alias — `True` → `color="charge"`, `False` → `color=None`; emits `DeprecationWarning` |
| `max_labels` | `60` | Cap on directly drawn labels, strongest first; `None` for no count cap |
| `theme_mode` | `None` | `"light"` / `"dark"`; `None` uses the global default |
| `intensity_scale` | `"relative"` | `"relative"` (base peak = 100%) or `"absolute"` |
| `intensity_transform` | `None` | `None`, `"sqrt"`, or `"log"` |
| `show_precursor` | `True` | On an `MsnSpectrum` carrying precursors, draw the precursor m/z hairline and the isolation window as recessive chrome behind the peaks |

`color="im"` takes a separate rendering path that bins ion mobility into 20 steps of the sequential
scale; `intensity_scale`, `intensity_transform`, and `show_precursor` apply there as on the other
colour modes. Profile spectra cannot take the IM path — centroid first, or pass `render="sticks"`
explicitly.

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
    max_labels: int | None = 60,
    theme_mode: Literal["light", "dark"] | None = None,
    intensity_scale: Literal["absolute", "relative"] = "relative",
    intensity_transform: Literal["sqrt", "log"] | None = None,
    texture: bool = False,
    show_precursor: bool = True,
    **layout_kwargs,
)
```

| Parameter | Default | Description |
|---|---|---|
| `tolerance` / `tolerance_type` | `0.02` / `"da"` | Matching tolerance |
| `peak_selection` | `"closest"` | `"closest"`, `"largest"`, or `"all"` |
| `include_sequence` | `False` | Embed the residue sequence in each label (`b3{PEP}` instead of `b3`) |
| `max_labels` | `60` | Cap on directly drawn ion labels |
| `theme_mode` | `None` | `"light"` / `"dark"` |
| `intensity_scale` / `intensity_transform` | `"relative"` / `None` | y-axis scaling, as above |
| `texture` | `False` | Also encode ion series as a dash pattern (the non-colour channel) — for print, forced-colours modes, and readers who cannot separate two hues. Off by default because at stick density dashes add noise |
| `show_precursor` | `True` | Draw precursor m/z + isolation window when present |

Matched peaks are coloured by ion series and labelled with their mzPAF identifier; unmatched peaks
are drawn in recessive grey, thinner (`1.0` vs `1.6`) and dimmer (opacity `0.55`).

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
    max_labels: int | None = 60,
    theme_mode: Literal["light", "dark"] | None = None,
    **layout_kwargs,
)
```

The second parameter is named `deconvoluted`. `show_charges` colours the deconvoluted (upper) half
by charge state; `show_scores` annotates its peaks with their isotope profile score, capped by
`max_labels`. The `raw` half is drawn in the unmatched grey.

`mirror_plot` does **not** take `intensity_scale`, `intensity_transform`, or `texture` — its y-axis
scaling is controlled by `normalize`, which scales each half independently to its own maximum so the
two fill their halves symmetrically. Either way the hover reports the pre-normalisation intensity.

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
    unit: str = "ppm",              # "ppm" or "da"
    title: str | None = None,
    theme_mode: Literal["light", "dark"] | None = None,
    **layout_kwargs,
)
```

Bubbles are coloured by ion series from the categorical palette and labelled with their mzPAF
identifier, so a 2+ and a 1+ of the same ion do not both render as `b3`. There is no `max_labels`
here — every match is labelled — and no `intensity_scale`, `intensity_transform`, or `texture`:
the y-axis is mass error, not intensity.

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
    max_labels: int | None = 60,
    theme_mode: Literal["light", "dark"] | None = None,
    **layout_kwargs,
)
```

Takes a **single** spectrum, not a list. The optional second spectrum is `mirror_spectrum`.
`max_labels` caps the ion labels in the annotated panel. Like `mirror_plot` and `mass_error_plot`,
`facet_plot` accepts no `intensity_scale`, `intensity_transform`, or `texture`; its panels are built
from plot tables at their defaults (so relative intensity), and each panel axis is titled
`Intensity`.

### `sequence_coverage_plot`

```python
from spxtacular import sequence_coverage_plot
```

```python
sequence_coverage_plot(
    spectrum: Spectrum,
    peptide: str,
    fragments,
    tolerance: float = 0.02,
    tolerance_type: Literal["da", "ppm"] = "da",
    peak_selection: Literal["closest", "largest", "all"] = "closest",
    title: str | None = None,
    theme_mode: Literal["light", "dark"] | None = None,
    **layout_kwargs,
) -> go.Figure
```

The coverage ladder: an annotated spectrum shows *that* peaks matched, this shows **where along the
peptide** they matched — which is what tells you whether an identification is localised or leaning
on one end of the molecule.

| Parameter | Default | Description |
|---|---|---|
| `spectrum` | | The spectrum the fragments are matched against |
| `peptide` | | Residue sequence, **one character per residue**. Pass the *stripped* sequence — ProForma modification brackets are not rendered. Raises `ValueError` when empty |
| `fragments` | | Fragment objects, as for `match_fragments` |
| `tolerance` / `tolerance_type` / `peak_selection` | `0.02` / `"da"` / `"closest"` | Matching parameters |
| `title` | `None` | Overrides the generated title |
| `theme_mode` | `None` | `"light"` / `"dark"` |

**Tick convention.** Residues run left to right. A tick drawn **above and to the left** of a residue
marks an N-terminal (a/b/c) fragment that *ended* at that bond; a tick **below and to the right**
marks a C-terminal (x/y/z) fragment that *started* there. A bond carrying ticks on both sides is
confirmed from both directions. N-terminal ticks take the `b` colour, C-terminal ticks the `y`
colour, and both are named in the legend.

The default title reports the count of distinct bonds covered — e.g.
`Sequence coverage — 17/17 backbone bonds covered (100%)`.

```python
import numpy as np
import peptacular as pt
import spxtacular as spx

peptide = "FDSFGDLSSASAIMGNPK"
fragments = pt.fragment(peptide, ion_types=("b", "y"), charges=(1, 2))

# A toy spectrum: one peak per theoretical fragment (m/z must be sorted).
mz = np.sort(np.array([f.mz for f in fragments]))
spectrum = spx.Spectrum(mz=mz, intensity=np.linspace(1e4, 1e5, len(mz)))

fig = spx.sequence_coverage_plot(spectrum, peptide, fragments)   # note: spectrum first
print(fig.layout.title.text)
# Sequence coverage — 17/17 backbone bonds covered (100%)
```

### `save_figure`

```python
from spxtacular import save_figure
```

```python
save_figure(fig: go.Figure, path: str | Path, scale: float = 2.0, **kwargs) -> Path
```

| Parameter | Default | Description |
|---|---|---|
| `fig` | | Figure to write |
| `path` | | Destination; the **suffix picks the writer** |
| `scale` | `2.0` | Device pixel ratio for raster formats — `2.0` stays sharp on a high-density display or in print |
| `**kwargs` | | Forwarded to `fig.write_html` / `fig.write_image` |

| Suffix | Backend | Extra install |
|---|---|---|
| `.html`, or no suffix (`.html` is appended) | `fig.write_html` | none — always works |
| `.png`, `.svg`, `.pdf`, `.jpg`, `.jpeg`, `.webp`, `.eps` | `fig.write_image` | `pip install kaleido` |
| anything else | — | raises `ValueError` |

A missing static-export backend is reported as an `ImportError` naming `kaleido`, rather than as a
bare exception from inside plotly. Returns the path actually written (useful when the suffix was
appended).

```python
import numpy as np
import spxtacular as spx

spectrum = spx.Spectrum(mz=np.array([100.0, 200.0]), intensity=np.array([10.0, 40.0]))
fig = spx.plot_spectrum(spectrum)

path = spx.save_figure(fig, "spectrum")     # -> Path('spectrum.html')
print(path)
```

---

## Theme

```python
from spxtacular import theme
```

`spxtacular.theme` is the single source of truth for plot colour — both `plot_table.py` and
`visualization.py` read from it, so a palette change lands on every figure at once rather than being
kept in sync by comment. Every plotting function's `theme_mode` argument selects the mode for one
figure; `theme.set_plot_theme()` sets the default for all of them.

### Colour is assigned by job, not by taste

| Job | Encoding | Lookup |
|---|---|---|
| Fragment ion series | **Nominal categorical** — eight fixed hues in the fixed slot order `b, y, a, c, x, z, p, i` | `ion_color` |
| Charge state | **Ordinal** — one hue, running light → dark as charge rises, so the reader sees 1+ < 2+ < 3+ in the colour | `charge_color` |
| Continuous magnitude — ion mobility (`plot_spectrum(color="im")`), and any per-peak score you colour yourself | **Sequential** — one hue, light → dark, with a colourbar. Not Viridis: a multi-hue ramp invents banding that is not in the data | `sequential_scale` |
| Unmatched peaks | **Recessive grey**, also thinner and dimmer — unmatched peaks are context, not subject | `unmatched_color` |

Three consequences worth knowing:

* **Ion hues never cycle.** Slots are assigned in order and stop at eight. Anything not in the eight
  slots — including internal fragments, whose ion types are two letters like `"by"` — folds to
  `neutral_color()` rather than being handed a ninth hue that would collide with an existing series.
  `b` and `y` take the first two slots because they are by far the most common pair, so the pair
  that co-occurs most often is the most separable.
* **The charge ramp clamps, it does not wrap.** The shipped ramp has five steps, and charges past
  its end all take the far end: `charge_color(11) == charge_color(5)`. The previous 10-colour cycle
  rendered `z=1` and `z=11`
  in *identical* colours, which is the one failure an ordinal encoding must not have.
  `charge <= 0` — singletons (`-1`) and decharged peaks (`0`) — is neutral grey: absence of
  identity, not another category.
* **Dark mode is not an inversion.** The dark charge ramp runs dark → light so it stays legible
  against the dark surface, and the sequential scale is reversed to match.

### Functions

| Function | Signature | Returns |
|---|---|---|
| `set_plot_theme` | `(mode: ThemeMode) -> None` | Sets the default mode for every subsequent plot. `ValueError` on anything but `"light"` / `"dark"` |
| `resolve_mode` | `(theme: ThemeMode \| None = None) -> ThemeMode` | The effective mode — the argument if given, otherwise the global default |
| `set_palette` | `(*, categorical=None, charge_ramp=None, sequential=None) -> None` | Replaces a palette wholesale (see below) |
| `ion_color` | `(ion_type: str, theme=None) -> str` | Hex colour for a fragment series; neutral for anything outside the eight slots |
| `charge_color` | `(charge: int, theme=None) -> str` | Hex colour from the ordinal ramp; clamped at the end, neutral for `charge <= 0` |
| `ion_dash` | `(ion_type: str) -> str` | Plotly dash pattern for a series — the texture channel used by `texture=True`. Mode-independent, so it takes no `theme` |
| `sequential_scale` | `(theme=None) -> list[list]` | Plotly colourscale for continuous magnitude — `[[stop, hex], …]`, five stops |
| `surface` | `(theme=None) -> str` | Chart surface (paper and plot background) |
| `text_color` | `(level: Literal["primary","secondary","muted"] = "secondary", theme=None) -> str` | Ink. Labels never wear the series colour — identity comes from the mark |
| `unmatched_color` | `(theme=None) -> str` | Colour for peaks carrying no annotation |
| `neutral_color` | `(theme=None) -> str` | Colour for singletons and any category past the eighth slot |
| `template` | `(theme=None) -> go.layout.Template` | The plotly template: recessive chrome, horizontal gridlines only, m/z crosshair, autosize |
| `apply` | `(fig: go.Figure, theme=None) -> go.Figure` | Applies that template to an existing figure in place and returns it |

`ThemeMode` is the type alias for the mode: `Literal["light", "dark"]`.

Note the naming: inside `theme` the mode argument is called `theme` (`ion_color("b", theme="dark")`),
while the plotting functions call the same thing `theme_mode` — there, `theme` would shadow the
module.

```python
import plotly.graph_objects as go
from spxtacular import theme

theme.set_plot_theme("dark")            # global default for every subsequent figure

theme.ion_color("b")                    # '#3987e5'  (dark-mode slot 0)
theme.ion_color("by")                   # neutral — internal fragments get no hue of their own
theme.ion_color("y", theme="light")     # '#eb6834' — one-off override, global default untouched
theme.charge_color(11) == theme.charge_color(5)   # True — the ramp clamps
theme.charge_color(-1) == theme.neutral_color()   # True — singletons are not a category

fig = theme.apply(go.Figure())          # borrow the template for your own figure
theme.set_plot_theme("light")
```

### `set_palette`

```python
theme.set_palette(
    *,
    categorical: dict[ThemeMode, list[str]] | None = None,   # >= 8 hues per mode
    charge_ramp: dict[ThemeMode, list[str]] | None = None,
    sequential: dict[ThemeMode, list[list]] | None = None,
) -> None
```

Each argument takes a `{"light": [...], "dark": [...]}` mapping and replaces that palette in both
modes. `categorical` and `charge_ramp` take lists of hex strings; `sequential` takes a plotly
colourscale, `[[0.0, "#…"], …, [1.0, "#…"]]`. `ValueError` is raised when a mapping is missing a
mode, or when a categorical palette has fewer entries than there are ion slots (8).

> **Substituted palettes are not validated for colour-vision deficiency.**
> The shipped palettes were checked with a CVD validator (protanopia and deuteranopia,
> Machado-Oliveira-Fernandes at severity 1.0) against both surfaces. A palette you pass to
> `set_palette` is **not** checked — the only validation is the structural one above (both modes
> present, at least 8 categorical hues). Validate your own hues before relying on them, or you lose
> the property the defaults were chosen for. Categorical hues want a fixed order with adjacent pairs
> kept far apart; a charge ramp wants a single hue with monotone lightness.

```python
from spxtacular import theme

theme.set_palette(
    categorical={
        "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
        "dark": ["#3987e5", "#d95926", "#199e70", "#c98500",
                 "#d55181", "#008300", "#9085e9", "#e66767"],
    },
    charge_ramp={
        "light": ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"],
        "dark": ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"],
    },
)
```

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

### Table schema

Both builders return the same columns, in this order:

```text
mz, intensity, intensity_abs, charge, score, im,
color, linewidth, opacity, dash, series,
label, label_size, label_font, label_color, label_yshift, label_xanchor, label_angle,
hover
```

| Column | dtype | Read by `plot_from_table`? | Meaning |
|---|---|---|---|
| `mz` | `float64` | yes | Peak m/z |
| `intensity` | `float64` | yes | **The plotted value** — relative-scaled (base peak = 100) unless `intensity_scale="absolute"`, then transformed if `intensity_transform` was given |
| `intensity_abs` | `float64` | no | **The true intensity**, always unscaled. This is what the tooltips report and what `table_view(max_rows=…)` ranks by |
| `charge` | `Int64` (nullable; `pd.NA` when the spectrum has no charge array) | no | Charge state |
| `score` | `float64` (`NaN` when absent) | no | `iso_score` |
| `im` | `float64` (`NaN` when absent) | no | Ion mobility |
| `color` | `str` | yes | Hex colour, from `theme` |
| `linewidth` | `float` | yes, from the first row of each group | `1.6` matched / `1.0` unmatched |
| `opacity` | `float` | yes, from the first row of each group | `1.0` matched / `0.55` unmatched |
| `dash` | `str` | yes, from the first row of each group (only when not `"solid"`) | Texture channel — set per ion series when `texture=True` |
| `series` | `str` | yes | Trace name and grouping key |
| `label` | `str` | yes | Direct label; `""` for peaks whose label was capped or collided away |
| `label_size`, `label_font`, `label_color`, `label_yshift`, `label_xanchor` | | yes | Label styling |
| `label_angle` | `float64` | yes | Label rotation in degrees. Defaults to `-90` (vertical, reading bottom-to-top); set `0` for horizontal |
| `hover` | `str` | yes | Tooltip text, baked in by the builder — to change a tooltip edit `hover` itself, not the value behind it |

`table.attrs["intensity_label"]` carries the y-axis title that matches the scaling applied
(`"Intensity"`, `"Relative intensity (%)"`, `"√ relative intensity (%)"`, `"log₁₀ …"`).
`plot_from_table` reads it, so a rescaled table titles its own axis.

Because `intensity` and `intensity_abs` are separate, tooltips are unaffected by rescaling: change
`intensity_scale` and the axis changes, never the number the reader is told.

`series` values:

| Table | `series` values |
|---|---|
| `build_plot_table` with charge data | `"z=1"`, `"z=2"`, … plus `"singleton"` (`charge == -1`) and `"decharged"` (`charge == 0`) |
| `build_plot_table` without charge data, or `show_charges=False` | `"peaks"` |
| `build_annot_plot_table` | the ion type of the matched fragment (`"b"`, `"y"`, …), or `"unmatched"` |

### `build_plot_table`

```python
from spxtacular import build_plot_table
```

```python
build_plot_table(
    spectrum: Spectrum,
    show_charges: bool = True,
    show_scores: bool = True,
    max_labels: int | None = 60,
    theme_mode: Literal["light", "dark"] | None = None,
    intensity_scale: Literal["absolute", "relative"] = "relative",
    intensity_transform: Literal["sqrt", "log"] | None = None,
    texture: bool = False,
) -> pd.DataFrame
```

| Parameter | Default | Description |
|---|---|---|
| `show_charges` | `True` | Colour peaks by charge state on the ordinal ramp and set `series` to `"z=N"` / `"singleton"` / `"decharged"` |
| `show_scores` | `True` | Label peaks whose `iso_score > 0` with their score |
| `max_labels` | `60` | Cap on labels kept non-empty, strongest first, after the collision pass |
| `theme_mode` | `None` | `"light"` / `"dark"` — decides the hex values written into `color` and `label_color` |
| `intensity_scale` | `"relative"` | Scaling written into `intensity` (`intensity_abs` is unaffected) |
| `intensity_transform` | `None` | `None`, `"sqrt"`, or `"log"` |
| `texture` | `False` | Accepted for signature parity with `build_annot_plot_table`; a plain spectrum has no ion series to texture, so `dash` stays `"solid"` either way |

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
    max_labels: int | None = 60,
    theme_mode: Literal["light", "dark"] | None = None,
    intensity_scale: Literal["absolute", "relative"] = "relative",
    intensity_transform: Literal["sqrt", "log"] | None = None,
    texture: bool = False,
) -> pd.DataFrame
```

Same trailing parameters as `build_plot_table`, and here `texture=True` does have an effect: each
matched peak's `dash` is set from `theme.ion_dash(ion_type)`. When one peak matches several ions,
the colour and `series` are chosen by the fixed ion slot order rather than by input order, so
reordering the fragment list never silently repaints the plot; the label lists every matching ion,
joined by `<br>`.

### `plot_from_table`

```python
from spxtacular import plot_from_table
```

```python
plot_from_table(
    table: pd.DataFrame,
    title: str | None = None,
    theme_mode: Literal["light", "dark"] | None = None,
    **layout_kwargs,
) -> go.Figure
```

Renders one `go.Scatter` trace per unique `(series, color)` group, plus the transparent hit-target
trace, plus one annotation per row with a non-empty `label`.

The required columns are validated up front — a missing one raises
`ValueError: plot table is missing required column(s): …` immediately, rather than part-way through
rendering or only on data that happens to carry labels:

```text
mz, intensity, series, color, linewidth, opacity, hover,
label, label_size, label_font, label_color, label_yshift, label_xanchor, label_angle
```

`intensity_abs` and `dash` are *not* required, so a table built by an older version still renders.
Grouping keeps NA keys (`dropna=False`): a row whose `series` or `color` came back NA — easy to
produce with `merge` / `reindex` / `concat` on a hand-edited table — is drawn in the unmatched
colour under the series name `"unlabelled"` instead of silently vanishing from the figure.

The legend is shown only when the table holds more than one `series`.

### `table_view`

```python
from spxtacular import table_view
```

```python
table_view(
    table: pd.DataFrame,
    max_rows: int | None = None,
    annotated_only: bool = False,
) -> str
```

Renders a plot table as an accessible HTML `<table>` string — the companion to the figure, not a
replacement for it. It exists because a tooltip should enhance, never gate: label capping
deliberately drops labels off the figure, and hovering is unusable for keyboard and screen-reader
users, so every value needs a non-hover route.

| Parameter | Default | Description |
|---|---|---|
| `table` | | A table from `build_plot_table` or `build_annot_plot_table` |
| `max_rows` | `None` | Keep only this many most-intense peaks (ranked on `intensity_abs`); `None` keeps all |
| `annotated_only` | `False` | Keep only peaks carrying a label — useful beside an annotated spectrum, where unmatched peaks are context rather than results |

Rows are emitted in m/z order. The columns are `m/z` and `Intensity` (the **true** intensity from
`intensity_abs`), plus `z`, `Score`, and `Ion mobility` when those columns hold any non-NA value,
plus `Annotation` when any label is present. Label text is HTML-escaped, and the `<br>` separators
the plot uses between co-matching ions become commas.

```python
import numpy as np
import peptacular as pt
import spxtacular as spx

peptide = "FDSFGDLSSASAIMGNPK"
fragments = pt.fragment(peptide, ion_types=("b", "y"), charges=(1, 2))

# A toy spectrum: one peak per theoretical fragment (m/z must be sorted).
mz = np.sort(np.array([f.mz for f in fragments]))
spectrum = spx.Spectrum(mz=mz, intensity=np.linspace(1e4, 1e5, len(mz)))

table = spx.build_annot_plot_table(spectrum, fragments)
print(table.attrs["intensity_label"])           # Relative intensity (%)

html = spx.table_view(table, max_rows=3, annotated_only=True)
print(html)
# <table><caption>Peak list</caption>…
```
