# Getting started

## Install

spxtacular needs Python 3.12 or newer.

```bash
pip install spxtacular
# or
uv add spxtacular
```

The core install processes spectra and draws plots (numpy, pandas, plotly and
[peptacular](https://github.com/tacular-omics/peptacular) are required dependencies). File readers
and integrations are optional extras:

```bash
pip install "spxtacular[bruker]"          # Bruker timsTOF (.d) via tdfpy
pip install "spxtacular[mzml]"            # mzML via mzmlpy
pip install "spxtacular[thermo]"          # Thermo .raw via fisher-py (also needs a .NET runtime)
pip install "spxtacular[readers]"         # all three readers
pip install "spxtacular[numba]"           # JIT-compiled deconvolution
pip install "spxtacular[spectrl]"         # share spectra as URL-safe tokens (spectrl)
pip install "spxtacular[matchms]"         # matchms adapter
pip install "spxtacular[spectrum-utils]"  # spectrum_utils adapter
pip install "spxtacular[interop]"         # both adapters
pip install "spxtacular[all]"             # everything above
```

With uv, use `uv add "spxtacular[mzml]"` and so on.

`DReader`, `MzmlReader`, and `ThermoReader` stay importable from `spxtacular` whichever extras are
installed. Only creating one without its backend raises an `ImportError` that names the extra to
install. The MGF, MS2, and MSP readers need no extra.

## Process a spectrum

Build a `Spectrum` from m/z and intensity arrays, then chain the processing steps. Every step
returns a new spectrum.

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

# Denoise, deconvolute, then convert to neutral masses.
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
Peak(mz=2400.8999, int=1.46e+05, z=0, score=0.998)
```

Fourteen input peaks collapse to two neutral masses. The most intense peak of the 3+ envelope is
its *second* isotope, not the monoisotopic one. Above roughly 1900 Da that is the norm, and
deconvolution anchors the cluster accordingly. [Deconvolution](deconvolution.md) explains how.

## Read a file

`Reader` detects the format from the path (`.d`, `.mzML`, `.raw`, `.mgf`, `.ms2`, `.msp`) and
hands off to the matching reader. This example uses `tests/data/example.mzML` from a clone of the
[repository](https://github.com/tacular-omics/spxtacular) and needs the `[mzml]` extra:

```python
from spxtacular import Reader

with Reader("tests/data/example.mzML") as reader:
    for spec in reader.ms1:
        print(spec)
    ms2 = next(iter(reader.ms2))
    print(ms2.precursors)
```

```text
MsnSpectrum(scan=0, ms_level=1, rt=353.43s, polarity=positive, n_peaks=15)
MsnSpectrum(scan=2, ms_level=1, rt=None, polarity=positive, n_peaks=0)
MsnSpectrum(scan=3, ms_level=1, rt=42.05s, polarity=positive, n_peaks=15)
[Precursor(mz=445.34, intensity=120053.0, charge=2, im=None, iso_score=None, is_monoisotopic=None)]
```

A Bruker timsTOF `.d` directory works the same way (`Reader("run.d")` or `DReader("run.d")`, with
the `[bruker]` extra). Readers return `MsnSpectrum` objects, which add scan metadata such as
`rt`, `ms_level`, and `precursors` to `Spectrum`. See [Readers](readers.md) for every format and
its options.

## Plot and annotate

Plots are interactive plotly figures. Pass fragments from peptacular to annotate a spectrum:

```python
import peptacular as pt
import spxtacular as spx

frags = pt.fragment("PEPTIDE", ion_types=("b", "y"), charges=(1, 2))

fig = spec.annotate(frags)                                   # annotated fragment spectrum
ladder = spx.sequence_coverage_plot(spec, "PEPTIDE", frags)  # backbone coverage ladder
html = spx.table_view(spx.build_annot_plot_table(spec, frags))  # accessible peak table

spx.save_figure(fig, "spectrum.html")   # .png/.svg/.pdf also work but need kaleido
```

Peaks are plotted as a percentage of the base peak by default, and every tooltip reports the true
intensity. `spx.theme.set_plot_theme("dark")` switches every later figure to the dark palette.
See [Visualization](visualization.md) for all plots and options.

## Share and convert spectra

With the `[spectrl]` extra, a whole spectrum fits in a URL-safe token, so no backend is needed to
share it:

```python
token = spec.to_spectrl_token()
restored = Spectrum.from_spectrl_token(token)

url = spec.to_spectrl_url("https://example.com/view")  # token in the URL fragment
restored = Spectrum.from_spectrl_url(url)
```

With the `[interop]` extra, `spx.to_matchms()` / `spx.from_matchms()` and
`spx.to_spectrum_utils()` / `spx.from_spectrum_utils()` convert to and from those libraries. See
[API reference: Ecosystem interoperability](api.md#ecosystem-interoperability) for what each
conversion keeps.

## Key concepts

| Concept | Summary |
|---|---|
| `Spectrum` | Central class. Holds `mz`, `intensity`, and optionally `charge`, `im`, and `iso_score` arrays. Transformations are chainable and return a new object unless `inplace=True` is requested. |
| `MsnSpectrum` | Extends `Spectrum` with instrument metadata: scan number, MS level, retention time, precursors, and more. |
| `Peak` | Frozen dataclass for a single `(mz, intensity, charge, im, iso_score)` observation. |
| `SpectrumType` | Enum: `CENTROID`, `PROFILE`, or `DECONVOLUTED`. Guards prevent calling `.decharge()` before `.deconvolute()`. |
| `Reader` | Format-agnostic file reader. Detects the format from the path and delegates to `DReader`, `MzmlReader`, `ThermoReader`, or the peak-list readers. |
| `spxtacular.theme` | Single source of plot colour. `set_plot_theme("light"\|"dark")` sets the global mode; `set_palette()` swaps in your own hues. The shipped palettes were checked for colour-vision deficiency in both modes; substituted ones are not. |
| Plot table | `build_plot_table()` / `build_annot_plot_table()` return the `DataFrame` behind every figure; `plot_from_table()` draws it and `table_view()` renders it as an accessible HTML table. |

## Where next

- [Spectrum](spectrum.md): every `Spectrum` and `MsnSpectrum` method
- [Deconvolution](deconvolution.md): how the greedy algorithm works and how to tune it
- [Readers](readers.md): mzML, Bruker `.d`, Thermo `.raw`, MGF, MS2, and MSP
- [Matching & scoring](scoring.md): `match_fragments()`, PSM `score()`, and spectrum similarity
- [Visualization](visualization.md): stick, mirror, annotated, facet, mass-error, and coverage plots
