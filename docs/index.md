# spxtacular

Mass spectrometry spectrum processing library. Companion to [peptacular](https://github.com/pgarrett-scripps/peptacular).

## Install

```bash
pip install spxtacular
```

For the Bruker timsTOF reader (`DReader`), `tdfpy` must be available. For mzML files, `mzmlpy` is required. Both are installed automatically if you pull from the editable source tree.

## Quick start

### Build a spectrum and run the full processing pipeline

```python
import numpy as np
from spxtacular import Spectrum

mz = np.array([500.1, 500.6, 501.1, 800.2, 800.7, 1200.5], dtype=np.float64)
intensity = np.array([1e5, 8e4, 3e4, 2e5, 1.5e5, 9e4], dtype=np.float64)

spec = Spectrum(mz=mz, intensity=intensity)

# Denoise, normalize, deconvolute, then convert to neutral masses — all chainable
neutral = (
    spec
    .denoise(method="mad")
    .normalize(method="max")
    .deconvolute(charge_range=(1, 5), tolerance=10, tolerance_type="ppm")
    .decharge()
)

for peak in neutral.peaks:
    print(peak)
```

### Read from an mzML file

```python
from spxtacular import MzmlReader

reader = MzmlReader("run.mzML")
for spec in reader.ms1:
    filtered = spec.filter(min_mz=200, min_intensity=1e3)
    print(filtered)
```

### Read from a Bruker timsTOF `.d` directory

```python
from spxtacular import DReader

with DReader("/data/sample.d") as reader:
    for spec in reader.ms1:
        print(spec)
```

## Key concepts

| Concept | Summary |
|---|---|
| `Spectrum` | Central class. Holds `mz`, `intensity`, and optionally `charge` and `im` arrays. All processing methods return a new `Spectrum` and are chainable. |
| `MsnSpectrum` | Extends `Spectrum` with instrument metadata: scan number, MS level, retention time, precursors, etc. |
| `Peak` | Frozen dataclass for a single `(mz, intensity, charge, im)` observation. |
| `SpectrumType` | Enum: `CENTROID`, `PROFILE`, or `DECONVOLUTED`. Guards prevent calling `.decharge()` before `.deconvolute()`. |

## Documentation

- [Spectrum reference](spectrum.md) — all `Spectrum` and `MsnSpectrum` methods
- [Deconvolution](deconvolution.md) — how the greedy algorithm works and how to use it
- [Readers](readers.md) — loading data from mzML and Bruker `.d` files
- [API reference](api.md) — concise listing of all public names
