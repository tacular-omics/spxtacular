<p align="center">
  <img src="spxtacular_logo.svg" alt="spxtacular logo" width="400"/>
</p>

# spxtacular

Mass spectrometry spectrum processing library. Companion to [peptacular](https://github.com/tacular-omics/peptacular).

## Install

```bash
pip install spxtacular
```

Reader backends are optional extras:

```bash
pip install spxtacular[bruker]   # Bruker timsTOF (.d) via tdfpy
pip install spxtacular[mzml]     # mzML via mzmlpy
pip install spxtacular[readers]  # both
pip install spxtacular[spectrl]  # share spectra as URL-safe tokens (spectrl)
pip install spxtacular[all]      # readers + numba JIT + spectrl
```

`DReader` and `MzmlReader` remain importable from `spxtacular` regardless of which backends are
installed; only instantiation raises a clear `ImportError` pointing to the right extra when the
corresponding dependency is missing.

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

### Share a spectrum as a URL-safe token

Requires the `[spectrl]` extra. The whole spectrum lives in the token — no backend needed.

```python
token = spec.to_spectrl_token()                        # spectrl1.… token
restored = Spectrum.from_spectrl_token(token)

url = spec.to_spectrl_url("https://example.com/view")  # …#spectrl1.… (shareable link)
restored = Spectrum.from_spectrl_url(url)
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
