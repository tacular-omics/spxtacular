# Readers

spxtacular provides two reader classes that expose a uniform interface for iterating over `MsnSpectrum` objects regardless of the underlying file format.

Both readers yield `MsnSpectrum` instances populated with as much instrument metadata as the format provides. All spectrum-processing methods (`.filter()`, `.denoise()`, `.deconvolute()`, etc.) are immediately available on each yielded object.

---

## MzmlReader

Reads standard `.mzML` files using `mzmlpy`. No context manager is required, but one is supported.

```python
class MzmlReader:
    def __init__(self, mzml_path: str): ...

    @property
    def ms1(self) -> Generator[MsnSpectrum, None, None]: ...

    @property
    def ms2(self) -> Generator[MsnSpectrum, None, None]: ...
```

### Properties

| Property | Yields |
|---|---|
| `ms1` | All MS1 spectra in scan order |
| `ms2` | All MS2 spectra in scan order, including parsed precursor information |

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

**Use as a context manager (optional):**

`MzmlReader` supports the context manager protocol but the `__exit__` method is a no-op — the underlying file handle is managed internally per-property call.

```python
with MzmlReader("run.mzML") as reader:
    for spec in reader.ms1:
        ...
```

---

## DReader

Reads Bruker timsTOF `.d` directories using `tdfpy`. **Must be used as a context manager** — the underlying `tdfpy` handle is opened on `__enter__` and closed on `__exit__`.

```python
class DReader:
    def __init__(self, analysis_dir: str): ...

    def __enter__(self) -> DReader: ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...

    @property
    def ms1(self) -> Generator[MsnSpectrum, None, None]: ...

    @property
    def ms2(self) -> Generator[MsnSpectrum, None, None]: ...
```

The acquisition type (DDA, DIA, PRM) is detected automatically from the `.d` directory at construction time and stored as `reader.aquisition_type` (`AcquisitionType` enum).

### Properties

| Property | Yields |
|---|---|
| `ms1` | All MS1 frames, centroided and merged by `tdfpy` |
| `ms2` | All MS2 spectra (DDA: per-precursor; DIA: per isolation window) |

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
| `mz_range` | Precursor isolation window |
| `im_range` | 1/K0 range of precursor |
| `precursors` | Single `TargetIon` with monoisotopic m/z (or largest peak m/z if unavailable), intensity, charge, and 1/K0 |
| `collision_energy` | From precursor record |
| `activation_type` | `"MS:1002481"` (PASEF) |

**MS2 (DIA):**

| Field | Source |
|---|---|
| `scan_number` | `frame_id` |
| `native_id` | `"{frame_id}@w{window_index}"` |
| `ms_level` | Always `2` |
| `rt` | Retention time (seconds) |
| `mz_range` | Isolation window m/z range |
| `im` (array) | Per-peak 1/K0 values |
| `collision_energy` | From window record |
| `precursors` | `None` — DIA windows have no defined precursor |

### Examples

**Iterate MS1 frames (DDA or DIA):**

```python
from spxtacular import DReader

with DReader("/data/sample.d") as reader:
    print(f"Acquisition type: {reader.aquisition_type}")
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

Detected automatically by `DReader` from the `.d` directory. Accessible as `reader.aquisition_type`. PRM and UNKNOWN acquisition types are handled the same way as DDA for MS2 iteration.
