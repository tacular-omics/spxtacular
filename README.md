<p align="center">
  <img src="https://raw.githubusercontent.com/tacular-omics/spxtacular/main/spxtacular_logo.svg" alt="spxtacular logo" width="400"/>
</p>

<p align="center">
  <a href="https://github.com/tacular-omics/spxtacular/actions/workflows/python-package.yml">
    <img src="https://github.com/tacular-omics/spxtacular/actions/workflows/python-package.yml/badge.svg" alt="CI"/>
  </a>
  <a href="https://codecov.io/gh/tacular-omics/spxtacular" > 
 <img src="https://codecov.io/gh/tacular-omics/spxtacular/graph/badge.svg?token=QbHHfY504R"/> 
 </a>
  <a href="https://pypi.org/project/spxtacular/">
    <img src="https://img.shields.io/pypi/v/spxtacular.svg" alt="PyPI"/>
  </a>
  <a href="https://pypi.org/project/spxtacular/">
    <img src="https://img.shields.io/pypi/pyversions/spxtacular.svg" alt="Python versions"/>
  </a>
  <a href="https://tacular-omics.github.io/spxtacular/">
    <img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue" alt="Docs"/>
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License"/>
  </a>
</p>

# spxtacular

**spxtacular** is a Python library for mass spectrometry spectrum processing. It provides a chainable `Spectrum` API covering the full centroid-to-neutral-mass pipeline: denoising, isotope deconvolution, neutral mass conversion, fragment matching, and PSM scoring — with interactive Plotly visualizations throughout.

> Part of the [tacular-omics](https://github.com/tacular-omics) ecosystem alongside [peptacular](https://github.com/tacular-omics/peptacular), [paftacular](https://github.com/tacular-omics/paftacular), and [mzmlpy](https://github.com/tacular-omics/mzmlpy).

## Install

```bash
pip install spxtacular

# Optional: Numba JIT acceleration (~3–4× faster deconvolution)
pip install spxtacular[numba]

# Optional: share spectra as compact URL-safe tokens (spectrl)
pip install spxtacular[spectrl]

# Optional: raw-file readers — Bruker .d (bruker) and/or mzML (mzml)
pip install spxtacular[bruker]      # tdfpy — DReader
pip install spxtacular[mzml]        # mzmlpy — MzmlReader
pip install spxtacular[readers]     # both readers

# Everything (numba + both readers + spectrl)
pip install spxtacular[all]
```

## Quick start

```python
import numpy as np
import spxtacular as spx

# A 2+ envelope near m/z 500 and a 3+ envelope near m/z 801, over a noise floor.
mz = np.array([
    352.1100, 418.4400, 476.9200,
    500.2573, 500.7590, 501.2606,
    655.3100, 733.0800,
    801.3073, 801.6417, 801.9762, 802.3106,
    918.6500, 1102.4000,
])
intensity = np.array([
    820.0, 1350.0, 690.0,
    100000.0, 51973.0, 11066.0,
    1580.0, 1015.0,
    52335.0, 60000.0, 34070.0, 12544.0,
    745.0, 1240.0,
])

spec = spx.Spectrum(mz=mz, intensity=intensity)

# Full pipeline: denoise → deconvolute → neutral mass
neutral = (
    spec
    .denoise(method="mad")
    .deconvolute(charge_range=(1, 5), tolerance=15, tolerance_type="ppm", min_score=0.4)
    .decharge()
)

for peak in neutral.peaks:
    print(peak)
# Peak(mz=998.5000, int=1.52e+05, z=0, score=1.000)
# Peak(mz=2400.9001, int=1.46e+05, z=0, score=0.997)

neutral.plot(title="Neutral masses").show()
```

Reading raw files works the same either way — `Reader` picks `DReader` or `MzmlReader` from the
path suffix:

```python
with spx.Reader("run.mzML") as reader:   # or spx.Reader("/data/sample.d")
    for spec in reader.ms1:              # .ms1/.ms2 are iterable *and* indexable
        ...
```

## Features

| Feature | Description |
|---|---|
| **Isotope deconvolution** | Bhattacharyya-scored greedy algorithm; optional Numba JIT acceleration |
| **Quality filtering** | `min_score`, m/z, intensity, charge, and ion mobility filters |
| **Neutral mass conversion** | `decharge()` converts charged clusters to neutral masses |
| **Fragment matching** | `match_fragments()` with ppm/Da tolerance |
| **PSM scoring** | Hyperscore, spectral angle, matched fraction, and more |
| **Interactive visualization** | Stick plots, mirror plots, annotated fragment spectra (Plotly) |
| **File reading** | Bruker timsTOF `.d` files (`DReader`) and mzML (`MzmlReader`), or `Reader` to auto-detect the format from the path |
| **Spectrum sharing** | Encode a full spectrum to a compact, URL-safe [spectrl](https://github.com/tacular-omics/spectrl) token or link (`to_spectrl_token` / `to_spectrl_url`) |

## Deconvolution pipeline

```python
# 1. Find isotope clusters → assign monoisotopic m/z + charge + Bhattacharyya score
decon = spec.deconvolute(charge_range=(1, 5), tolerance=10, tolerance_type="ppm")

# charge > 0  → assigned cluster
# charge = -1 → singleton / unassigned
# score 0–1   → isotope profile quality (0.0 for singletons)

# 2. Keep only high-confidence clusters
filtered = decon.filter(min_score=0.5)

# 3. Convert to neutral masses (drops singletons)
neutral = filtered.decharge()
```

## Sharing spectra

With the optional `[spectrl]` extra, encode a complete spectrum (peaks, charges,
ion mobility, and MSn metadata) into a single compact, URL-safe token — or a
ready-to-share link — with no backend required.

```python
token = spec.to_spectrl_token()                       # spectrl1.… token
restored = spx.Spectrum.from_spectrl_token(token)

url = spec.to_spectrl_url("https://example.com/view")  # …#spectrl1.… (shareable)
restored = spx.Spectrum.from_spectrl_url(url)
```

## Documentation

Full documentation with API reference, guides, and interactive plots is available at
**[tacular-omics.github.io/spxtacular](https://tacular-omics.github.io/spxtacular/)**.

- [Spectrum API](https://tacular-omics.github.io/spxtacular/spectrum/)
- [Deconvolution](https://tacular-omics.github.io/spxtacular/deconvolution/)
- [Readers](https://tacular-omics.github.io/spxtacular/readers/)
- [Matching & Scoring](https://tacular-omics.github.io/spxtacular/scoring/)
- [Visualization](https://tacular-omics.github.io/spxtacular/visualization/)
- [API Reference](https://tacular-omics.github.io/spxtacular/api/)

## License

[MIT](LICENSE)
