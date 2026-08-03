# Readers

spxtacular provides two format-specific reader classes — `MzmlReader` and `DReader` — plus a
format-agnostic `Reader` that picks the right one from the file extension. All three expose a
uniform interface for iterating over `MsnSpectrum` objects regardless of the underlying file format.

Both readers yield `MsnSpectrum` instances populated with as much instrument metadata as the format provides. All spectrum-processing methods (`.filter()`, `.denoise()`, `.deconvolute()`, etc.) are immediately available on each yielded object.

## Lookup objects

`.ms1` and `.ms2` are **not** generators. Each property returns a small lookup object that is both
**iterable** (`for spec in reader.ms1:`) and **indexable** (`reader.ms2[42]`):

| Reader | `.ms1` type | `.ms2` type |
|---|---|---|
| `MzmlReader` | `MzmlSpectraLookup` | `MzmlSpectraLookup` |
| `DReader` | `DReaderMs1Lookup` | `DReaderMs2Lookup` |
| `Reader` | whichever the detected backend provides | whichever the detected backend provides |

Because they are iterables and not iterators, `next(reader.ms2)` raises `TypeError` — use
`next(iter(reader.ms2))` to pull the first spectrum:

```python
first_ms2 = next(iter(reader.ms2))
```

Index semantics differ per backend:

| Expression | Meaning |
|---|---|
| `MzmlReader.ms1[key]` / `.ms2[key]` | Spectrum by **overall** 0-based index or native ID string. No MS-level filtering is applied on random access, so `reader.ms2[0]` is the first spectrum in the file, not the first MS2 spectrum |
| `MzmlReader[key]` | Same as above, straight off the reader (`reader[0]`, `reader["scan=19"]`) |
| `DReader.ms1[frame_id]` | MS1 spectrum by tdfpy `frame_id` |
| `DReader.ms2[precursor_id]` | MS2 spectrum by tdfpy `precursor_id` — **DDA only**. DIA and PRM raise `NotImplementedError` (their MS2 records are not keyed by a single id); iterate instead |

`DReader` lookups raise `RuntimeError` if the reader has not been opened.

`polarity`, `activation_type`, `im_type`, and `analyzer` are populated as plain strings straight from the underlying format (including raw PSI-MS accessions such as `"MS:1002481"`) — they also accept the `Polarity`, `ActivationType`, `IMType`, and `Analyzer` enums documented in [API reference — Metadata enums](api.md#metadata-enums) if you want to set or compare them with autocomplete/typo-safety.

---

## Reader

`Reader` is the format-agnostic entry point: it inspects the path suffix and delegates to `DReader`
(`.d`) or `MzmlReader` (`.mzml`, case-insensitive). Anything else raises `ValueError`. Usage is
identical regardless of the underlying format.

```python
class Reader:
    def __init__(
        self,
        path: str | Path,
        centroid_config: CentroidConfig | None = None,
    ): ...

    def open(self) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> Reader: ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...

    @property
    def ms1(self) -> DReaderMs1Lookup | MzmlSpectraLookup: ...

    @property
    def ms2(self) -> DReaderMs2Lookup | MzmlSpectraLookup: ...
```

```python
from spxtacular import Reader

with Reader("run.mzML") as r:      # or Reader("/data/sample.d")
    for spec in r.ms1:
        print(spec)

with Reader("/data/sample.d") as r:
    ms2 = r.ms2[42]                # DDA precursor_id
```

`centroid_config` is only meaningful for `.d` inputs; it is ignored for mzML. `Reader` exposes
`ms1`, `ms2`, `open`, and `close` only — backend-specific members (`MzmlReader.__getitem__`,
`DReader.acquisition_type`) are not proxied.

---

## MzmlReader

Reads standard `.mzML` files using `mzmlpy`. No context manager is required, but one is strongly
recommended — see [File handles](#file-handles) below.

```python
class MzmlReader:
    def __init__(self, mzml_path: str | Path): ...

    def open(self) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> MzmlReader: ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...

    @property
    def ms1(self) -> MzmlSpectraLookup: ...

    @property
    def ms2(self) -> MzmlSpectraLookup: ...

    def __getitem__(self, key: int | str) -> MsnSpectrum: ...
```

### Properties

| Property | Contents |
|---|---|
| `ms1` | All MS1 spectra in scan order (iteration); index/native-ID access is unfiltered |
| `ms2` | All MS2 spectra in scan order, including parsed precursor information (iteration); index/native-ID access is unfiltered |

### Metadata populated from mzML

| Field | Source |
|---|---|
| `scan_number` | Spectrum index |
| `ms_level` | `msLevel` CV param |
| `native_id` | Raw spectrum `id` attribute |
| `rt` | `scan start time` (converted to seconds) |
| `mz_range` | `scan window lower/upper limit` |
| `polarity` | `positive scan` / `negative scan` CV params |
| `spectrum_type` | `centroid spectrum` / `profile spectrum` CV params |
| `charge` (array) | Per-peak charge array when present in the binary data |
| `im` (array) | First ion mobility binary array when present |
| `precursors` | MS2 only: selected ion m/z, intensity, charge, and activation info |
| `collision_energy` | MS2 only: from activation element |
| `activation_type` | MS2 only: from activation element |

### Examples

**Iterate MS1 spectra:**

```python
from spxtacular import MzmlReader

reader = MzmlReader("run.mzML")
for spec in reader.ms1:
    print(spec)
    # MsnSpectrum(scan=0, ms_level=1, rt=1.23s, polarity=positive, n_peaks=4521)
```

**Filter and denoise each MS1 scan:**

```python
from spxtacular import MzmlReader

reader = MzmlReader("run.mzML")
for spec in reader.ms1:
    processed = spec.filter(min_mz=200, max_mz=1600).denoise("mad")
    print(f"Scan {spec.scan_number}: {len(processed)} peaks after denoise")
```

**Iterate MS2 spectra with precursor info:**

```python
from spxtacular import MzmlReader

reader = MzmlReader("run.mzML")
for spec in reader.ms2:
    if not spec.precursors:
        continue
    prec = spec.precursors[0]
    print(
        f"Scan {spec.scan_number} | "
        f"Precursor {prec.mz:.4f} m/z, z={prec.charge} | "
        f"CE={spec.collision_energy} eV"
    )
```

**Full deconvolution pipeline on MS1:**

```python
from spxtacular import MzmlReader

reader = MzmlReader("run.mzML")
for spec in reader.ms1:
    neutral = (
        spec
        .filter(min_mz=300, min_intensity=1000)
        .denoise("mad")
        .deconvolute(charge_range=(1, 5), tolerance=10, tolerance_type="ppm")
        .decharge()
    )
    for peak in neutral.top_peaks(10):
        print(f"  mass={peak.mz:.4f} Da  intensity={peak.intensity:.2e}")
    break  # first scan only
```

**Fetch a single spectrum by index or native ID:**

```python
from spxtacular import MzmlReader

with MzmlReader("run.mzML") as reader:
    first = reader[0]              # by overall 0-based index
    named = reader["scan=19"]      # by native ID
    first_ms2 = next(iter(reader.ms2))   # first MS2 spectrum
```

### File handles

`MzmlReader.open()` opens a persistent `mzmlpy` handle and `close()` releases it; `__enter__` /
`__exit__` call them, so `__exit__` is **not** a no-op. While a handle is open, iteration and index
access reuse it (the fast path). Without one, every `.ms1` / `.ms2` / `reader[...]` operation opens
and closes the file again — correct, but considerably slower. Prefer the context manager:

```python
with MzmlReader("run.mzML") as reader:
    for spec in reader.ms1:
        ...
```

---

## DReader

Reads Bruker timsTOF `.d` directories using `tdfpy`. **Must be opened before use** — either with
`open()` / `close()` or (preferred) as a context manager. The underlying `tdfpy` handle is opened on
`__enter__` and closed on `__exit__`; touching `.ms1` / `.ms2` before `open()` raises `RuntimeError`.

```python
class DReader:
    def __init__(
        self,
        analysis_dir: str | Path,
        centroid_config: CentroidConfig | None = None,
    ): ...

    def open(self) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> DReader: ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...

    @property
    def ms1(self) -> DReaderMs1Lookup: ...

    @property
    def ms2(self) -> DReaderMs2Lookup: ...
```

The acquisition type (DDA, DIA, PRM) is detected automatically from the `.d` directory at construction time and stored as `reader.acquisition_type` (`AcquisitionType` enum).

### `CentroidConfig`

timsTOF frames arrive as raw (frame, scan) points, so `DReader` centroids them via `tdfpy`.
`CentroidConfig` holds the parameters forwarded to that step. It is exported from the package root
and ignored by `MzmlReader`.

```python
from spxtacular import CentroidConfig, DReader

cfg = CentroidConfig(mz_tolerance=8.0, min_peaks=3, noise_filter="mad")
with DReader("/data/sample.d", centroid_config=cfg) as reader:
    ...
```

| Field | Default | Description |
|---|---|---|
| `mz_tolerance` | `8.0` | m/z merge window |
| `mz_tolerance_type` | `"ppm"` | `"ppm"` or `"da"` |
| `im_tolerance` | `0.1` | Ion mobility merge window |
| `im_tolerance_type` | `"relative"` | `"relative"` or `"absolute"` |
| `min_peaks` | `3` | Minimum raw points required to emit a centroid |
| `noise_filter` | `None` | `"mad"`, `"percentile"`, `"histogram"`, `"baseline"`, `"iterative_median"`, a `float` threshold, or `None` |

Passing `centroid_config=None` (the default) uses `CentroidConfig()` with the values above.

### Properties

| Property | Contents |
|---|---|
| `ms1` | All MS1 frames, centroided and merged by `tdfpy`; `ms1[frame_id]` fetches one frame |
| `ms2` | All MS2 spectra (DDA: per-precursor; DIA: per isolation window; PRM: per transition); `ms2[precursor_id]` fetches one DDA spectrum |

### Metadata populated from timsTOF

**MS1:**

| Field | Source |
|---|---|
| `scan_number` | `frame_id` |
| `ms_level` | Always `1` |
| `rt` | Frame acquisition time (seconds) |
| `injection_time` | Frame accumulation time (ms) |
| `mz_range` | Instrument acquisition range from metadata |
| `im_range` | 1/K0 acquisition range from metadata |
| `im` (array) | Per-peak 1/K0 values |
| `analyzer` | Always `"TOF"` |
| `ramp_time` | Frame ramp time (ms) |
| `polarity` | From frame polarity field |
| `spectrum_type` | Always `CENTROID` (timsTOF data arrives centroided) |

**MS2 (DDA):**

| Field | Source |
|---|---|
| `scan_number` | `precursor_id` |
| `ms_level` | Always `2` |
| `rt` | Retention time (seconds) |
| `isolation_mz_range` | Precursor isolation window |
| `isolation_im_range` | 1/K0 range of precursor |
| `precursors` | Single `Precursor` with monoisotopic m/z (or largest peak m/z if unavailable), intensity, charge, and 1/K0 |
| `collision_energy` | From precursor record |
| `activation_type` | `"MS:1002481"` (PASEF) |

**MS2 (DIA):**

| Field | Source |
|---|---|
| `scan_number` | `frame_id` |
| `native_id` | `"{frame_id}@w{window_index}"` |
| `ms_level` | Always `2` |
| `rt` | Retention time (seconds) |
| `isolation_mz_range` | Isolation window m/z range |
| `isolation_im_range` | Isolation window 1/K0 range |
| `im` (array) | Per-peak 1/K0 values |
| `collision_energy` | From window record |
| `precursors` | `None` — DIA windows have no defined precursor |

**MS2 (PRM):**

| Field | Source |
|---|---|
| `scan_number` | `frame_id` |
| `native_id` | `"{frame_id}@t{target_id}"` |
| `ms_level` | Always `2` |
| `rt` | Retention time (seconds) |
| `isolation_mz_range` / `isolation_im_range` | Transition isolation windows |
| `precursors` | Single `Precursor` built from the PRM target (monoisotopic m/z, charge, 1/K0); intensity is the summed MS2 peak intensity, since PRM targets carry no measured precursor intensity |
| `collision_energy` | From transition record |

### Examples

**Iterate MS1 frames (DDA or DIA):**

```python
from spxtacular import DReader

with DReader("/data/sample.d") as reader:
    print(f"Acquisition type: {reader.acquisition_type}")
    for spec in reader.ms1:
        print(spec)
        # MsnSpectrum(scan=1, ms_level=1, rt=0.42s, polarity=positive, n_peaks=8234)
        break
```

**MS1 with ion mobility filtering:**

```python
from spxtacular import DReader

with DReader("/data/sample.d") as reader:
    for spec in reader.ms1:
        # Keep only peaks in a specific 1/K0 window
        filtered = spec.filter(min_im=0.8, max_im=1.2, min_intensity=500)
        if len(filtered) == 0:
            continue
        neutral = (
            filtered
            .deconvolute(charge_range=(1, 5), tolerance=15, tolerance_type="ppm")
            .decharge()
        )
        print(f"Frame {spec.scan_number}: {len(neutral)} neutral masses")
```

**MS2 DDA — inspect precursors:**

```python
from spxtacular import DReader

with DReader("/data/sample_dda.d") as reader:
    for spec in reader.ms2:
        if not spec.precursors:
            continue
        prec = spec.precursors[0]
        print(
            f"Precursor {spec.scan_number}: "
            f"m/z={prec.mz:.4f}, z={prec.charge}, "
            f"1/K0={prec.im:.3f}, "
            f"monoisotopic={prec.is_monoisotopic}"
        )
        break
```

**MS2 DIA — iterate isolation windows:**

```python
from spxtacular import DReader

with DReader("/data/sample_dia.d") as reader:
    for spec in reader.ms2:
        print(
            f"{spec.native_id}: "
            f"mz_range={spec.mz_range}, "
            f"CE={spec.collision_energy}"
        )
        break
```

---

## AcquisitionType

```python
class AcquisitionType(StrEnum):
    DDA = "DDA"
    DIA = "DIA"
    PRM = "PRM"
    UNKNOWN = "UNKNOWN"
```

Detected automatically by `DReader` from the `.d` directory. Accessible as `reader.acquisition_type`.
`UNKNOWN` falls back to the DDA reader; `PRM` has its own MS2 iteration path (per transition).
