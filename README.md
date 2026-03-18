# spxtacular

Mass spectrometry spectrum processing library. Companion to [peptacular](https://github.com/pgarrett-scripps/peptacular).

## Install

```bash
pip install spxtacular

# Optional: Numba JIT acceleration for deconvolution (~3-4x faster)
pip install spxtacular[numba]
```

## Quick start

```python
import numpy as np
import spxtacular as spx

# Build a spectrum
spec = spx.Spectrum(mz=mz_array, intensity=intensity_array)

# Denoise -> deconvolute -> decharge
neutral = (
    spec
    .denoise(method="mad", snr=3.0)
    .deconvolute(charge_range=(1, 5), tolerance=10, tolerance_type="ppm", min_score=0.4)
    .decharge()
)

# Plot
neutral.plot(title="Neutral masses").show()
```

## Features

- **Isotope deconvolution** — Bhattacharyya-scored greedy algorithm; optional numba acceleration
- **Quality filtering** — reject low-confidence clusters with `min_score`, filter by m/z, intensity, charge, or ion mobility
- **Neutral mass conversion** — `decharge()` converts charged clusters to neutral masses
- **Fragment matching** — `match_fragments()` with ppm/Da tolerance, closest/largest/all peak selection
- **PSM scoring** — `score()` returns hyperscore, spectral angle, matched fraction, and more
- **Visualization** — stick plots, mirror plots (raw vs deconvoluted), annotated fragment spectra
- **File reading** — Bruker timsTOF `.d` files (`DReader`) and mzML (`MzmlReader`)

## Documentation

Full API reference and guides: [docs/](docs/)

- [Spectrum API](docs/spectrum.md)
- [Deconvolution](docs/deconvolution.md)
- [Readers](docs/readers.md)
- [Visualization](docs/visualization.md)
- [Scoring](docs/scoring.md)

## License

See [LICENSE](LICENSE).
