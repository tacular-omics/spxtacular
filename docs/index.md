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
pip install spxtacular[thermo]   # Thermo .raw via fisher-py (also needs a .NET runtime)
pip install spxtacular[readers]  # all three
pip install spxtacular[spectrl]  # share spectra as URL-safe tokens (spectrl)
pip install spxtacular[all]      # readers + numba JIT + spectrl
```

`DReader`, `MzmlReader`, and `ThermoReader` remain importable from `spxtacular` regardless of which
backends are installed; only instantiation raises a clear `ImportError` pointing to the right extra
when the corresponding dependency is missing.

## Quick start

### Build a spectrum and run the full processing pipeline

```python
import numpy as np
from spxtacular import Spectrum

# A 2+ envelope near m/z 500 and a 3+ envelope near m/z 801, over a noise floor.
mz = np.array([
    352.1100, 418.4400, 476.9200,
    500.2573, 500.7590, 501.2606,                       # 2+ isotope envelope
    655.3100, 733.0800,
    801.3073, 801.6417, 801.9762, 802.3106,             # 3+ isotope envelope
    918.6500, 1102.4000,
], dtype=np.float64)
intensity = np.array([
    820.0, 1350.0, 690.0,
    100000.0, 51973.0, 11066.0,
    1580.0, 1015.0,
    52335.0, 60000.0, 34070.0, 12544.0,
    745.0, 1240.0,
], dtype=np.float64)

spec = Spectrum(mz=mz, intensity=intensity)

# Denoise, deconvolute, then convert to neutral masses — all chainable
neutral = (
    spec
    .denoise(method="mad")
    .deconvolute(charge_range=(1, 5), tolerance=15, tolerance_type="ppm")
    .decharge()
)

for peak in neutral.peaks:
    print(peak)
```

```text
Peak(mz=998.5000, int=1.52e+05, z=0, score=1.000)
Peak(mz=2400.9001, int=1.46e+05, z=0, score=0.997)
```

Fourteen input peaks collapse to two neutral masses. Note the 3+ envelope's most
intense peak is its *second* isotope, not the monoisotopic one — above roughly
1900 Da that is the norm, and deconvolution anchors the cluster accordingly.

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

### Or let the format be detected for you

```python
from spxtacular import Reader

with Reader("/data/sample.d") as reader:   # or Reader("run.mzML")
    for spec in reader.ms1:
        print(spec)
```

### Plot, annotate, and show sequence coverage

```python
import peptacular as pt
import spxtacular as spx

spx.theme.set_plot_theme("dark")   # global default: "light" (default) or "dark"

frags = pt.fragment("PEPTIDE", ion_types=("b", "y"), charges=(1, 2))

fig = spec.annotate(frags)                                   # annotated fragment spectrum
ladder = spx.sequence_coverage_plot(spec, "PEPTIDE", frags)  # backbone coverage ladder
html = spx.table_view(spx.build_annot_plot_table(spec, frags))  # accessible peak table

spx.save_figure(fig, "spectrum.html")   # .png/.svg/.pdf also work — those need kaleido
```

Peaks are plotted as a percentage of the base peak by default (`intensity_scale="absolute"`
restores raw counts), direct labels are capped at 25 and collision-avoided along m/z, and every
tooltip reports the true intensity regardless of scaling. See
[Visualization](visualization.md) for the full set of plots and options.

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
| `Spectrum` | Central class. Holds `mz`, `intensity`, and optionally `charge`, `im`, and `iso_score` arrays. All processing methods return a new `Spectrum` and are chainable. |
| `MsnSpectrum` | Extends `Spectrum` with instrument metadata: scan number, MS level, retention time, precursors, etc. |
| `Peak` | Frozen dataclass for a single `(mz, intensity, charge, im, iso_score)` observation. |
| `SpectrumType` | Enum: `CENTROID`, `PROFILE`, or `DECONVOLUTED`. Guards prevent calling `.decharge()` before `.deconvolute()`. |
| `Reader` | Format-agnostic file reader — detects `.d`, `.mzML`, `.raw`, `.mgf`, or `.ms2` from the path and delegates to `DReader` / `MzmlReader` / `ThermoReader` / the peak-list readers. |
| `spxtacular.theme` | Single source of truth for plot colour. `set_plot_theme("light"\|"dark")` sets the global mode; `set_palette()` swaps in your own hues. The shipped palettes were validated for colour-vision deficiency in both modes — substituted ones are not. |
| Plot table | `build_plot_table()` / `build_annot_plot_table()` return the `DataFrame` behind every figure; `plot_from_table()` draws it and `table_view()` renders it as an accessible HTML table. |

## Plotting

- **One theme, two modes.** Colour is assigned by job: ion type is a fixed 8-slot categorical
  palette, charge state is an *ordinal* single-hue ramp, and `iso_score` / ion mobility use a
  sequential ramp. Unmatched peaks stay recessive grey.
- **Relative intensity by default.** The y-axis is a percentage of the base peak; pass
  `intensity_scale="absolute"` for raw counts, or `intensity_transform="sqrt"` / `"log"` to
  compress a large dynamic range. Tooltips always report the true intensity.
- **Readable labels.** Direct labels are vertical (the spectrum-viewer convention, so far more
  fit before colliding), capped (`max_labels=60`) and collision-avoided; dropped
  labels remain in the hover and in the plot table.
- **Beyond colour.** On annotated plots, `texture=True` adds per-ion-series dash patterns for
  print and forced-colours displays, and `table_view()` gives a non-visual route to the same data.
- **Plots available.** `plot_spectrum`, `annotate_spectrum`, `mirror_plot`, `facet_plot`,
  `mass_error_plot`, and `sequence_coverage_plot`, plus `save_figure()` to write HTML or (with
  kaleido) static images.

## Documentation

- [Spectrum reference](spectrum.md) — all `Spectrum` and `MsnSpectrum` methods
- [Deconvolution](deconvolution.md) — how the greedy algorithm works and how to use it
- [Readers](readers.md) — loading data from mzML, Bruker `.d`, Thermo `.raw`, and MGF / MS2 files
- [Matching & scoring](scoring.md) — `match_fragments()` and PSM `score()` metrics
- [Visualization](visualization.md) — stick, mirror, annotated, facet, mass-error, and sequence coverage plots; theming, intensity scaling, and the accessible table view
- [API reference](api.md) — concise listing of all public names
