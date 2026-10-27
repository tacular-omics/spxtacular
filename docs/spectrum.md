# Spectrum and MsnSpectrum

`Spectrum` holds the peak arrays of one mass spectrum: `mz` and `intensity`, and optionally
`charge`, `im` (ion mobility) and `iso_score`. `MsnSpectrum` adds the scan metadata a reader knows —
scan number, MS level, retention time, precursors. Processing methods return a new spectrum instead
of changing the one they were called on, so a piece of work reads as a single chain.

This page opens with the sequence most work follows, then works through the remaining methods
on a real spectrum. Signatures and parameter tables live in the
[API reference](api.md#spectrum-methods).

## The usual sequence

Load a spectrum, inspect it, clean it up, assign charge states, match fragments, plot. Every block
below runs as written from a clone of the
[repository](https://github.com/tacular-omics/spxtacular) and continues the one before it. The
input is `tests/data/mzspeclib/fetal_brain_tiny.mzSpecLib.txt.gz`, a four-spectrum excerpt of a
human fetal brain HCD spectral library that needs no optional extra. The instrument-file fixtures
are listed under [Readers — Example files](readers.md#example-files).

### 1. Load a spectrum

`read_mzspeclib` yields `LibraryEntry` objects, each carrying an `MsnSpectrum`. A reader
(`Reader`, `DReader`, `MzmlReader`, `ThermoReader`, `MgfReader`) yields the same `MsnSpectrum`
objects from an instrument file, so every later step is identical whatever the input was.

```python
from pathlib import Path

import peptacular as pt
import spxtacular as spx

library = Path("tests/data/mzspeclib/fetal_brain_tiny.mzSpecLib.txt.gz")
peptide = "LAQANGWGVM[Oxidation]VSHR/2"

entries = [e for e in spx.read_mzspeclib(library) if e.name and peptide in e.name]
spec = entries[0].spectrum  # one of four HCD acquisitions of this peptidoform
print(spec)
```

```text
MsnSpectrum(scan=None, ms_level=2, rt=1501.00s, polarity=positive, n_peaks=339)
```

### 2. Inspect it

`len(spec)` is the peak count. [`peaks`](api.md#peaks) and [`top_peaks`](api.md#top_peaks) return
read-only [`Peak`](api.md#peak) snapshots; [`has_peak`](api.md#has_peak),
[`get_peak`](api.md#get_peak) and [`get_peaks`](api.md#get_peaks) look for one m/z. The arrays
themselves (`spec.mz`, `spec.intensity`) stay available for numpy work.

```python
print(len(spec), "peaks,", "MS level", spec.ms_level, "at", spec.rt, "s")
print("precursor m/z", spec.precursors[0].precursor_mz)
for peak in spec.top_peaks(3):
    print(peak)
```

```text
339 peaks, MS level 2 at 1501.0 s
precursor m/z 771.3848
Peak(mz=737.4050, int=1.00e+04)
Peak(mz=101.0709, int=9.48e+03)
Peak(mz=980.5052, int=7.26e+03)
```

### 3. Clean it up

[`filter`](api.md#filter) removes peaks outside bounds (m/z, intensity, charge, ion mobility,
isotope score) or keeps only the most intense ones. This library spectrum is already centroided and
fairly clean, so a modest intensity floor only drops the low-intensity tail.

```python
clean = spec.filter(min_intensity=200.0)
print(len(spec), "->", len(clean), "peaks")
```

```text
339 -> 181 peaks
```

On raw data the two companion steps come first: [`centroid`](api.md#centroid) turns profile data
into peaks, and [`denoise`](api.md#denoise) estimates a noise floor from the spectrum itself rather
than taking a number from you. [`normalize`](api.md#normalize) and
[`scale_intensity`](api.md#scale_intensity) adjust the intensity scale, and
[`remove_precursor_peak`](api.md#remove_precursor_peak) strips the surviving precursor and its
isotopes before scoring.

### 4. Assign charge states and go to neutral masses

[`deconvolute`](api.md#deconvolute) groups peaks into isotope clusters and records a charge
magnitude and an isotopic profile score per peak. [`decharge`](api.md#decharge) then converts each
cluster's monoisotopic m/z to a neutral mass. The order matters: `decharge()` on a spectrum that has
not been deconvoluted raises `SpxtacularError`.

```python
deconv = clean.deconvolute(charge_range=(1, 2), tolerance=20, tolerance_unit="ppm")
assigned = int((deconv.charge > 0).sum())
print(deconv.spectrum_type, "-", assigned, "peaks in clusters,", len(deconv), "peaks total")

neutral = deconv.decharge()
print(neutral.top_peaks(1)[0])
```

```text
deconvoluted - 24 peaks in clusters, 157 peaks total
Peak(mz=736.3977, int=1.31e+04, z=0, score=0.903)
```

A `charge` of `-1` marks a singleton with no isotope neighbours, `0` a neutral mass after
`decharge()`; see [Charge conventions](api.md#charge-conventions). Fragment matching works on either
m/z or neutral masses, so step 5 can take `clean`, `deconv` or `neutral`.

### 5. Match fragments and score the match

[`match_fragments`](api.md#match_fragments) pairs peptacular fragments with peaks;
[`score`](api.md#score) runs the same matching and returns the PSM metrics.

```python
fragments = pt.parse(peptide).fragment(ion_types=("b", "y"), charges=[1, 2])
matches = clean.match_fragments(fragments, tolerance=20, tolerance_unit="ppm")
print(len(matches), "of", len(fragments), "fragments matched")
for m in matches[:3]:
    print(f"{m.fragment.to_mzpaf():>8s}  {m.peak_mz:9.4f}  {m.ppm_error:+5.1f} ppm  {m.intensity_pct:4.1f}%")

scores = clean.score(fragments, tolerance=20, tolerance_unit="ppm")
print(f"hyperscore {scores['hyperscore']:.2f}, spectral angle {scores['spectral_angle']:.3f}")
```

```text
22 of 56 fragments matched
   y1{R}   175.1187   -1.4 ppm   2.4%
  b2{LA}   185.1282   -1.4 ppm   1.8%
  y2{HR}   312.1775   -1.2 ppm   1.1%
hyperscore 16.19, spectral angle 0.459
```

Every metric is defined in [Matching and scoring](scoring.md).

### 6. Plot it

[`plot`](api.md#plot_spectrum) draws the spectrum, [`annotate`](api.md#annotate_spectrum) draws it
with the fragment labels, and [`mass_error_plot`](api.md#mass_error_plot) and
[`facet_plot`](api.md#facet_plot) add the diagnostic panels. `spx.save_figure(fig,
"annotated.html")` writes the figure; `.png`, `.svg` and `.pdf` work too.

```python
fig = spec.annotate(fragments, tolerance=20, tolerance_unit="ppm")
print(type(fig).__name__)
```

```text
Figure
```

[`plot_table`](api.md#build_plot_table) and [`annot_plot_table`](api.md#build_annot_plot_table)
return the `DataFrame` behind a figure, so the numbers a plot shows can be read or edited directly.
Every figure and option is in [Visualization](visualization.md).

---

## Working through a spectrum

The rest of this page walks the remaining methods in the order a script tends to need them. Every
block below runs as written after this one and continues the one before it, with one exception:
the [USI fetch](#moving-a-spectrum-somewhere-else), which needs network access and so is not run
here. `spec` is a three-peak spectrum built by hand; `ms2`, `deconvoluted`, `fragments`, the
replicates and `profile_spec` stand in for a real scan, its fragments, repeat acquisitions and
profile data.

```python
import numpy as np
import peptacular as pt
import spxtacular as spx
from spxtacular import Peak, Spectrum, plot_from_table

spec = Spectrum(
    mz=np.array([500.1, 800.2, 1200.5], dtype=np.float64),
    intensity=np.array([1e5, 2e5, 9e4], dtype=np.float64),
)
print(spec)
# Spectrum(n_peaks=3, type=None, denoised=None, normalized=None)

# A real HCD scan, its fragments and three replicates of it, from the example library
peptide = "LAQANGWGVM[Oxidation]VSHR/2"
entries = [
    e
    for e in spx.read_mzspeclib("tests/data/mzspeclib/fetal_brain_tiny.mzSpecLib.txt.gz")
    if e.name and peptide in e.name
]
ms2 = entries[0].spectrum
deconvoluted = ms2.deconvolute(charge_range=(1, 2), tolerance=20, tolerance_unit="ppm")
fragments = pt.parse(peptide).fragment(ion_types=("b", "y"), charges=[1, 2])
spec1, spec2, spec3 = (e.spectrum for e in entries[:3])

# A single Gaussian peak sampled in profile mode, for the centroiding examples
grid = np.arange(499.80, 500.40, 0.005)
profile_spec = Spectrum(
    mz=grid,
    intensity=1e5 * np.exp(-((grid - 500.1) ** 2) / (2 * 0.01**2)),
    spectrum_type="profile",
)
```

### Looking at peaks

[`peaks`](api.md#peaks) returns every peak as a read-only [`Peak`](api.md#peak) snapshot, so peak
fields read as attributes rather than parallel array indices. For anything numeric on a large
spectrum, work on `spec.mz` and `spec.intensity` instead.

```python
for peak in spec.peaks:
    print(peak.mz, peak.intensity)
```

[`top_peaks`](api.md#top_peaks) sorts by any of the peak attributes — intensity, m/z, charge, ion
mobility or isotope score — and returns the first `n`.

```python
# Five most intense peaks
top5 = spec.top_peaks(5)

# Lowest-mz three peaks
low_mz = spec.top_peaks(3, by="mz", reverse=False)
```

[`has_peak`](api.md#has_peak), [`get_peak`](api.md#get_peak) and
[`get_peaks`](api.md#get_peaks) look for one m/z within a tolerance. `has_peak` answers yes or no,
`get_peak` returns the best single match (most intense, or nearest in m/z with
`peak_selection="closest"`), and `get_peaks` returns all of them. A charge or ion mobility
criterion needs the matching array, so it needs a deconvoluted spectrum or one read with ion
mobility.

```python
spec.has_peak(500.1, tolerance=0.02)

# target_charge and target_im need the matching array, so they need a deconvoluted
# spectrum or one read with ion mobility: `deconvoluted` comes from the setup block.
deconvoluted.has_peak(736.4050, tolerance=10, tolerance_unit="ppm", target_charge=1)
```

```python
peak = spec.get_peak(800.2, tolerance=5, tolerance_unit="ppm")
if peak:
    print(f"Found: {peak}")
```

### Cleaning up

[`filter`](api.md#filter) removes peaks outside bounds on m/z, intensity, charge, ion mobility or
isotope score, and `top_n` / `top_n_per_window` keep only the most intense survivors — globally, or
per fixed-width m/z window, which is the usual search-engine preprocessing step.

```python
# Keep peaks between 200 and 1500 Da with intensity >= 1000
filtered = spec.filter(min_mz=200, max_mz=1500, min_intensity=1000)

# Keep only the 50 most intense peaks after m/z filtering
filtered = spec.filter(min_mz=200, top_n=50)

# Keep the 10 most intense peaks per 100 Th window (search-engine-style preprocessing)
filtered = spec.filter(top_n_per_window=(10, 100.0))
```

[`normalize`](api.md#normalize) divides by a reference (the base peak, the total ion current or the
median) so intensities from different acquisitions are comparable.

```python
norm = spec.normalize()            # max normalization
norm = spec.normalize(method="tic")       # TIC normalization
```

[`denoise`](api.md#denoise) estimates a noise floor from the spectrum itself and drops everything
below it, rather than taking a number from you. The `"mad"` default is robust on most spectra; a
plain number is used as an absolute threshold.

```python
spec.denoise()                       # MAD (robust, recommended for most spectra)
spec.denoise(method="histogram")            # histogram mode estimate
spec.denoise(method=5000.0)          # fixed absolute threshold
```

[`centroid`](api.md#centroid) turns profile data into peaks by fitting each local maximum. On real
data give it a floor — see [Centroiding](#centroiding) at the end of this page for why.

```python
centroided = profile_spec.centroid()
```

[`merge`](api.md#merge) collapses peaks that agree within a tolerance into one intensity-weighted
centroid, working from the most intense peak outwards. Only peaks of the same charge merge.

```python
merged = spec.merge(mz_tolerance=0.02, mz_tolerance_unit="da")
merged = spec.merge(mz_tolerance=5, mz_tolerance_unit="ppm")
```

### Charge states and neutral masses

[`deconvolute`](api.md#deconvolute) groups peaks into isotope clusters, recording a charge
magnitude and an isotopic profile score per peak, plus the isotope and ionization models it used.
[Deconvolution](deconvolution.md) is the full walkthrough, including the isotope models for
glycans, lipids and nucleic acids.

```python
decon = spec.deconvolute(charge_range=(1, 5), tolerance=10, tolerance_unit="ppm")
```

[`decharge`](api.md#decharge) then converts each cluster's monoisotopic m/z to a neutral mass,
dropping the singletons. It reuses the ionization model `deconvolute()` recorded, so the carrier
that was added is the carrier that is removed. Afterwards every `charge` is `0` — see
[charge conventions](api.md#charge-conventions) — and `iso_score` carries through.

```python
neutral = deconvoluted.decharge()
# neutral.mz now contains neutral masses sorted ascending
# neutral.charge is all zeros
# neutral.iso_score carries through from the deconvoluted spectrum
```

### Reordering and rebuilding

[`sort`](api.md#sort) reorders every parallel array together, by any peak attribute.

```python
by_intensity = spec.sort(by="intensity", reverse=True)   # most intense first
```

[`update`](api.md#update) is the low-level escape hatch: a copy with named fields replaced. Prefer
the named methods above for everything they cover.

```python
renamed = spec.update(spectrum_type="centroid")
```

### Moving a spectrum somewhere else

[`to_dict`](api.md#json-transport) and `from_dict` convert to and from the versioned
`spxtacular.spectrum` transport format, which holds only JSON-native values and records whether the
original was a `Spectrum` or an `MsnSpectrum`.

```python
payload = spec.to_dict()
restored = Spectrum.from_dict(payload)

assert type(restored) is type(spec)
assert restored == spec
```

[`to_json`](api.md#json-transport) and `from_json` are the same transport as strict JSON text, for
APIs and browser visualization. For local storage prefer `.npz`, which is smaller and keeps the
arrays as NumPy arrays.

```python
message = spec.to_json()
restored = Spectrum.from_json(message)
```

The packaged JSON Schema is available without an extra dependency:

```python
from spxtacular import get_json_schema

schema = get_json_schema("spectrum")
```

[`to_spectrl_token`](api.md#token-serialisation-spectrl) encodes the whole spectrum — peaks,
metadata, precursors — as one URL-safe [spectrl](https://github.com/tacular-omics/spectrl) token
that mirrors mzML semantics. The default is lossy within a bounded error; `lossless=True` keeps the
arrays bit-exact. Requires the optional `[spectrl]` extra.

```python
token = spec.to_spectrl_token()                      # lossy (bounded-error), default
token_exact = spec.to_spectrl_token(lossless=True)   # bit-exact arrays
restored = Spectrum.from_spectrl_token(token)
```

[`to_spectrl_url`](api.md#url-sharing) binds that token into a link, or a `data:` URI. With the
default fragment binding the token sits after `#`, so the spectrum never reaches the server hosting
the page that draws it: the [Spectra web application](https://spectra.tacular.dev/) decodes such a
link in the browser, and `from_spectrl_url` reads one back here.

```python
url = spec.to_spectrl_url(base="https://example.com/view")             # …#spectrl.…
uri = spec.to_spectrl_url(mode="data")                            # data: URI
restored = Spectrum.from_spectrl_url(url)
```

[`save`](api.md#persistence-npz) and `load` write a numpy `.npz` archive: peak arrays natively,
scalar metadata JSON-encoded beside them. All MSn metadata survives the round trip.

```python
spec.save("scan_001.npz")
restored = Spectrum.load("scan_001.npz")
```

[`Spectrum.from_usi`](api.md#usi-loading) fetches a spectrum from a public proteomics repository by
Universal Spectrum Identifier, over the PROXI REST API.

```python title="needs network access - not executed"
fetched = Spectrum.from_usi(
    "mzspec:PXD000561:Adult_Frontalcortex_bRP_Elite_85_f09:scan:17555"
)
```

### Figures

[`plot`](api.md#plot_spectrum) draws the spectrum — sticks or profile, chosen from the spectrum
type — colouring by charge state or ion mobility.

```python
spec.plot(title="My spectrum").show()
decon.plot(color="charge", show_scores=True).show()
```

[`annotate`](api.md#annotate_spectrum) draws it with fragment ion labels: matched peaks coloured by
ion series, unmatched peaks grey.

```python
fig = ms2.annotate(fragments, tolerance=10, tolerance_unit="ppm")
fig.show()
```

[`mass_error_plot`](api.md#mass_error_plot) plots the mass error of each matched fragment against
its m/z, which is where a calibration drift shows up.

```python
ms2.mass_error_plot(fragments, tolerance=20, tolerance_unit="ppm", unit="ppm").show()
```

[`facet_plot`](api.md#facet_plot) stacks those panels on a shared m/z axis, optionally with a
second spectrum mirrored below.

```python
ms2.facet_plot(
    fragments=fragments,
    mirror_spectrum=deconvoluted.decharge(),
).show()
```

[`plot_table`](api.md#build_plot_table) and
[`annot_plot_table`](api.md#build_annot_plot_table) return the `DataFrame` behind a figure: one row
per peak, carrying both the data and every visual property. Edit it and render it with
`plot_from_table()`, which is how a figure gets a detail the plotting functions do not expose.

```python
tbl = decon.plot_table()
tbl.loc[tbl["charge"] == 2, "color"] = "red"
tbl.loc[tbl["intensity"] > 1e5, "linewidth"] = 2.0
fig = plot_from_table(tbl, title="Custom plot")
fig.show()
```

```python
tbl = spec.annot_plot_table(fragments, tolerance=10, tolerance_unit="ppm")
tbl.loc[tbl["label"] != "", "label_size"] = 14
fig = plot_from_table(tbl, title="Annotated")
fig.show()
```

Every figure, style and export option is in [Visualization](visualization.md); the column list is
in [Plot table API](api.md#plot-table-api).

### Precursor removal, rescaling and rounding

[`remove_precursor_peak`](api.md#remove_precursor_peak) strips the surviving precursor, its isotope
envelope and its lower charge states, which is worth doing before scoring. On an `MsnSpectrum` it
reads the precursors from the scan; it adapts to whether the spectrum is centroided, deconvoluted
or decharged.

```python
# Auto from MsnSpectrum.precursors
cleaned = ms2.remove_precursor_peak(tolerance=10, tolerance_unit="ppm")

# Explicit
cleaned = spec.remove_precursor_peak(precursor_mz=450.25, precursor_charge=2)
```

[`scale_intensity`](api.md#scale_intensity) applies a non-linear transform — square root, log or
rank — which keeps a few tall peaks from dominating a similarity score.
[`round_mz`](api.md#round_mz) rounds m/z to a given number of decimals and sums or max-reduces the
duplicates, for coarse binning.

### Combining spectra

[`Spectrum.combine`](api.md#spectrumcombine) concatenates the peaks of several spectra into one new
`Spectrum`, sorted by m/z. Mixing neutral masses with m/z values, or charged inputs with different
deconvolution records, raises `SpxtacularError` rather than silently producing a spectrum whose
axis means two things.

```python
combined = Spectrum.combine([spec1, spec2, spec3])
```

`combine` and `merge` together build a consensus spectrum from repeat measurements of the same
analyte: normalize each replicate so no single acquisition dominates, concatenate, then collapse
peaks that agree within tolerance into intensity-weighted centroids.

```python
replicates = [spec1, spec2, spec3]          # e.g. the same precursor across runs

consensus = Spectrum.combine(
    [s.normalize(method="tic") for s in replicates]
).merge(mz_tolerance=10, mz_tolerance_unit="ppm")
```

Peaks seen in every replicate accumulate intensity across the merge window while one-off noise
peaks stay small, so a follow-up `filter(min_intensity=...)` — thresholded at, say, a fraction of
`1 / len(replicates)` of the summed TIC — keeps only reproducible signal.

---

## MsnSpectrum

`MsnSpectrum` extends `Spectrum` with the instrument metadata a reader knows: scan number, MS
level, retention time, acquisition and isolation windows, collision energy, activation type,
analyzer, and the [`Precursor`](api.md#precursor) objects selected for fragmentation. Every file
reader yields it, and every `Spectrum` method above works on it unchanged. The field list is in
[API reference — `MsnSpectrum`](api.md#msnspectrum), and the `ActivationType`, `IMType` and
`Analyzer` vocabularies are in [Metadata enums](api.md#metadata-enums).

```python
from spxtacular import MzmlReader

reader = MzmlReader("tests/data/example.mzML")   # or your own file; see [Example files](readers.md#example-files)
for scan in reader.ms2:
    print(f"Scan {scan.scan_number}, RT={scan.rt:.1f}s, CE={scan.collision_energy}")
    if scan.precursors:
        prec = scan.precursors[0]
        print(f"  Precursor: {prec.precursor_mz:.4f} m/z, z={prec.charge}")
    break
```

---

## Centroiding

```python
spec.centroid()                          # every local maximum becomes a peak
spec.centroid(min_intensity="noise")     # MAD-estimated floor
spec.centroid(min_intensity=2000)        # absolute floor
```

`min_intensity` is the floor an apex must clear to count as a peak.

!!! warning
    Without one, **every** local maximum is a peak. On data with any noise floor that returns far
    more centroids than there are real peaks — a test spectrum with 6 real peaks and a modest noise
    floor gives 769 centroids at the default, 6 with `min_intensity=2000`. The default is `None`
    for backwards compatibility, but on real data you almost always want a floor.

Use [`profile_centroid_plot()`](visualization.md#checking-centroiding) to see what the centroider
actually did.

A flat apex counts as one peak: requiring a strict `prev < curr > next` would discard any peak
whose maximum spans two or more equal samples, which is routine in quantised or saturated data.
