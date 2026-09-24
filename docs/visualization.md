# Visualization

Every plot function on this page draws the same figure with one of three backends:

| `backend=` | Returns | For | Needs |
|---|---|---|---|
| `"plotly"` (default) | `plotly.graph_objects.Figure` | notebooks, dashboards, HTML reports: hover, zoom | nothing (plotly is required) |
| `"matplotlib"` | `matplotlib.figure.Figure` | papers: vector PDF/SVG with embedded fonts | `pip install 'spxtacular[matplotlib]'` |
| `"spec"` | `spxtacular.FigureSpec` | composing multi-panel figures, testing, custom renderers | nothing |

A plot is first described as a backend-neutral `FigureSpec` (panels, marks, axes, rich-text
labels). One layout pass then fixes the tick positions, the label placement and the sizes in
points, and the backend only draws what it is given. So a figure looks the same in both backends,
and a label that clears its neighbours in plotly clears them in the PDF too.

```python
import spxtacular as spx

fig = spx.annotate_spectrum(spec, fragments, peptide="PEPTIDEK")                  # plotly, on screen
fig = spx.annotate_spectrum(spec, fragments, peptide="PEPTIDEK",
                            backend="matplotlib", style="paper", size="single")  # print
spx.save_figure(fig, "figure2.pdf")
```

## How these plots are built

A few conventions apply to every figure on this page. They are worth knowing up front, because
they explain defaults that would otherwise look surprising.

**Relative intensity is the default y-axis.** Peaks are scaled so the base peak is 100%, matching
the convention every MS viewer uses. Raw counts are still available with
`intensity_scale="absolute"`, and the tooltip always reports the *true* intensity either way —
rescaling changes the axis, never the number you are told. For data spanning orders of magnitude,
`intensity_transform="sqrt"` or `"log"` compresses the range so low-abundance matched ions stay
visible beside a dominant base peak.

**Labels are typeset and never overlap.** Ion labels are rich text: `b₅`, `y₇²⁺`, `y₄−H₂O`, with a
real minus sign and subscripted formulas, coloured by ion series. They read horizontally and are
placed in two dimensions: a label that would collide with a stronger one moves up, and a leader
line joins it back to its peak when it has moved far. A label with no free space is dropped rather
than drawn over another. Set `label_angle` to `-90` in the plot table for the old vertical labels.

Labels are also capped (`max_labels`, default 60), strongest peak first: labelling every annotated
peak turns a deconvoluted spectrum into an unreadable smear.
Nothing is lost — the dropped values remain in the hover text, in the plot table, and in
[`table_view()`](#table_view).

**Hovering does not require precision.** Every figure gets an m/z crosshair from the shared
template. Sticks are only about 1.5 pixels wide, so `plot_spectrum()` and `annotate_spectrum()`
additionally lay a transparent hit layer over the peak tips — being *near* a peak is enough.
(The other plots hover on their traces directly, which is fine for bubbles and subplot panels.)

**Profile spectra are drawn as a trace, not sticks.** `plot_spectrum()` reads `spectrum_type`:
profile data becomes a continuous line with a light fill so the peak *shape* — the only reason
profile data exists — survives. Everything else stays a stick plot. Override with `render=`.

**Colour is assigned by the job it does**, not by taste — see [Theme](#theme) below.

---

## Publication figures

### Styles

`style=` sets fonts, font sizes, line weights, tick density and colour for the medium:

| Style | Base font | Width | For |
|---|---|---|---|
| `"paper"` | 7 pt Arial/Helvetica | 85 mm | journal figures: thin lines, no title (the caption is the title), 600 dpi raster |
| `"screen"` | 11 px system sans | 900 px (238 mm) | notebooks and reports; the default for `backend="plotly"` |
| `"talk"` | 14 pt Arial/Helvetica | 254 mm (10 in, 16:9) | slides: heavy lines, few ticks |

`"paper"` is the default for `backend="matplotlib"` and `"spec"`. `spx.get_style("paper")` returns
the `FigureStyle` dataclass; change any field with `.with_(...)` and pass the result as `style=`:

```python
tight = spx.get_style("paper").with_(font_size=6.0, label_size=5.5)
fig = spx.annotate_spectrum(spec, fragments, backend="matplotlib", style=tight)
```

An unknown style name raises `SpxtacularError`.

### Sizes

`size=` takes a journal column width or millimetres. Heights follow the style's aspect ratio, with
fixed extra height for a sequence header or an error strip, so those do not squash the spectrum.

| `size=` | Width |
|---|---|
| `"single"` | 85 mm (one column) |
| `"onehalf"` | 114 mm |
| `"double"` | 175 mm (full page width) |
| `120` | 120 mm |
| `(120, 60)` | 120 mm by 60 mm |

Fonts are set in points at the final size, so a `"paper"` figure placed at 100% in the manuscript
has 7 pt text, which is what most journals ask for. Do not rescale it in the layout program.

### Output

`save_figure(fig, "fig.pdf")` from a matplotlib figure writes vector PDF with the fonts embedded as
TrueType (Type 42), not Type 3. Journals accept this, and the text stays editable in Illustrator or
Inkscape. SVG keeps text as `<text>` elements. PNG is rendered at the style's `dpi` (600 for
`"paper"`). The plotly backend writes the same formats through kaleido (`spxtacular[plotly-export]`).

### What changes in print

- The m/z axis title is an italic *m/z*, and absolute intensity axes carry a shared exponent
  (`Intensity (×10⁵)`) instead of `1.2e5` on every tick.
- Unmatched peaks are a darker grey than on screen, so they survive printing and photocopying.
- The precursor is labelled `[M+2H]²⁺` rather than `precursor 500.2500 (2+)`.
- Mass-error axes are symmetric around zero and span the matching tolerance when it is in the
  displayed unit, so a point near the edge is near the tolerance.

### Gallery

`docs/gallery/build.py` renders every figure in both backends and both styles, as PNG and SVG,
from the mzSpecLib fixture in the tests and a few synthetic spectra:

```bash
python docs/gallery/build.py                                # everything into docs/gallery/out
python docs/gallery/build.py --only annotated --backend matplotlib --style paper
```

---

## `plot_spectrum()`

```python
from spxtacular.visualization import plot_spectrum

fig = plot_spectrum(
    spectrum,
    title=None,                  # plot title
    color="charge",              # "charge" | "im" | None
    show_scores=True,            # annotate scored peaks with their score value
    max_labels=60,               # cap on direct labels; None for no cap
    theme_mode=None,             # "light" | "dark"; None uses the global default
    intensity_scale="relative",  # "relative" (base peak = 100%) | "absolute"
    intensity_transform=None,    # None | "sqrt" | "log"
    show_precursor=True,         # draw precursor m/z + isolation window on MSn
    render=None,                 # None | "sticks" | "profile"
    max_points=4000,             # profile-sample cap, or None for every sample
    absolute_axis=False,         # absolute intensities with a x10^n axis exponent
    backend="plotly",            # "plotly" | "matplotlib" | "spec"
    style=None,                  # "paper" | "screen" | "talk" | FigureStyle
    size=None,                   # "single" | "onehalf" | "double" | mm | (w, h) mm
    **layout_kwargs,             # plotly only: passed to fig.update_layout()
)
fig.show()
```

Draws a stick plot of any `Spectrum`. With `color="charge"` (default), sticks are coloured by
charge state when `charge` is present, on an **ordinal ramp** running light to dark — charge is
ordered, so you read 1+ < 2+ < 3+ out of the colour. Unassigned peaks (`charge = -1`) and decharged
ones (`charge = 0`) are neutral grey. With `color="im"`, sticks are coloured by ion mobility on a
single-hue sequential scale (falls back to `"charge"` when no IM array is present). With
`color=None`, every stick takes one colour. When `iso_score` is present and `show_scores=True`,
score values label the strongest scored peaks.

On an `MsnSpectrum` carrying precursor information, the precursor m/z and its isolation window are
drawn as recessive reference chrome behind the peaks. Pass `show_precursor=False` to suppress them.

Every function on this page takes `backend=`, `style=` and `size=` as above. `**layout_kwargs` is
plotly-only; passing it with `backend="matplotlib"` raises `SpxtacularError`.

`Spectrum.plot()` is a convenience wrapper around this function:

```python
spec.plot(title="My spectrum", color="charge").show()
```

**Raw spectrum:**

<iframe src="../plots/raw.html" width="100%" height="500" frameborder="0"></iframe>

**Deconvoluted spectrum (coloured by charge state):**

<iframe src="../plots/deconvoluted.html" width="100%" height="500" frameborder="0"></iframe>

**Deconvoluted + filtered (score ≥ 0.5):**

<iframe src="../plots/deconvoluted_filtered.html" width="100%" height="500" frameborder="0"></iframe>

---

## `mirror_plot()`

```python
from spxtacular.visualization import mirror_plot

fig = mirror_plot(
    raw,                 # Spectrum -- drawn inverted below the x-axis
    deconvoluted,        # Spectrum -- drawn upright above the x-axis
    fragments=None,      # annotate both halves with matched fragments
    names=None,          # ("query", "library"): labels for the two halves
    similarity=None,     # "cosine" | "modified_cosine" | "entropy" | a number to print
    title=None,
    normalize=True,      # scale each half to its own maximum independently
    show_charges=True,   # colour the deconvoluted half by charge state
    show_scores=True,    # annotate deconvoluted peaks with iso_score
    max_labels=60,       # cap on score labels, strongest first
    theme_mode=None,     # "light" | "dark"
    **layout_kwargs,
)
fig.show()
```

Mirror plot for comparing a raw spectrum (inverted, below) against its deconvoluted counterpart
(upright, above). Useful for visually confirming that isotope clusters have been correctly
identified and scored. With `show_charges=True` (default) deconvoluted peaks are coloured by charge
state, using the same ordinal ramp as `plot_spectrum()` so a spectrum keeps its colours when the
two figures sit side by side; with `show_scores=True` the `iso_score` annotations appear above each
cluster. Each half is normalised to its own maximum, but the hover reports the true intensity.

The second parameter is named `deconvoluted` — pass it by that name if you use keywords.

The same function compares a query against a library spectrum. With `fragments=`, both halves are
annotated and coloured by ion series, and `names=` labels each half. `similarity=` computes the
score and prints it in the corner (`"modified_cosine"` needs a precursor m/z on both spectra and
raises `SpxtacularError` otherwise):

```python
fig = mirror_plot(query, library, fragments=fragments, names=("query", "library"), similarity="cosine")
```

**Example:**

```python
from spxtacular import Spectrum
from spxtacular.visualization import mirror_plot

decon = spec.deconvolute(charge_range=(1, 5), tolerance=10, tolerance_unit="ppm")
fig = mirror_plot(spec, decon, title="Raw vs deconvoluted")
fig.show()
```

**Raw vs deconvoluted:**

<iframe src="../plots/mirror.html" width="100%" height="500" frameborder="0"></iframe>

**Raw vs deconvoluted + filtered:**

<iframe src="../plots/mirror_filtered.html" width="100%" height="500" frameborder="0"></iframe>

**Neutral mass spectrum:**

<iframe src="../plots/neutral_mass.html" width="100%" height="500" frameborder="0"></iframe>

**Neutral mass + filtered:**

<iframe src="../plots/neutral_mass_filtered.html" width="100%" height="500" frameborder="0"></iframe>

---

## `annotate_spectrum()`

```python
import peptacular as pt
from spxtacular.visualization import annotate_spectrum

fragments = pt.fragment("PEPTIDE", ion_types=("b", "y"), charges=(1, 2))
fig = annotate_spectrum(
    spectrum,
    fragments,
    tolerance=0.02,
    tolerance_unit="da",         # or "ppm"
    title=None,
    peak_selection="closest",    # "closest" | "largest" | "all"
    include_sequence=False,
    max_labels=60,
    theme_mode=None,
    intensity_scale="relative",
    intensity_transform=None,
    texture=False,
    show_precursor=True,
    peptide=None,                # draw the sequence with fragment ticks above the spectrum
    mass_error_panel=False,      # add a mass-error strip below
    absolute_axis=False,
    backend="plotly",
    style=None,
    size=None,
    **layout_kwargs,
)
fig.show()
```

Draws the spectrum as a stick plot and overlays matched fragment ion labels, coloured by ion
series. Unmatched peaks stay in a recessive grey, drawn thinner and dimmer, so the annotated peaks
lead rather than competing with the context behind them.

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `spectrum` | | `Spectrum` to plot |
| `fragments` | | Iterable of fragment objects from `peptacular` |
| `tolerance` | `0.02` | Matching tolerance |
| `tolerance_unit` | `"da"` | `"da"` or `"ppm"` |
| `title` | `None` | Plot title |
| `peak_selection` | `"closest"` | How to resolve multiple peaks within tolerance — `"closest"`, `"largest"`, or `"all"` |
| `include_sequence` | `False` | Embed the residue sequence in each label (e.g. `b3{PEP}`) |
| `max_labels` | `60` | Cap on direct labels, strongest first; `None` for no cap |
| `theme_mode` | `None` | `"light"` or `"dark"`; `None` uses the global default |
| `intensity_scale` | `"relative"` | `"relative"` (base peak = 100%) or `"absolute"` |
| `intensity_transform` | `None` | `None`, `"sqrt"` or `"log"` |
| `texture` | `False` | Give each ion series its own dash pattern |
| `show_precursor` | `True` | Draw the precursor m/z and isolation window on an `MsnSpectrum` |
| `peptide` | `None` | Sequence (string or peptacular annotation) drawn above the spectrum with a tick for each observed b/y cleavage |
| `mass_error_panel` | `False` | Add a mass-error strip below the spectrum, sharing its m/z axis |

When one peak matches several ions, its colour is chosen by the fixed ion-series order rather than
by whichever fragment you happened to list first, so reordering your fragment list never repaints
the plot.

**Annotated spectrum:**

!!! note
    This figure and the mass-error, coverage and facet figures below are drawn from a *simulated*
    MS2 — this peptide's own fragments displaced by a few ppm, over a noise floor — so the
    annotations and mass errors are genuine. The raw, deconvolution and mirror figures above use a
    real spectrum, where no peptide is involved.

<iframe src="../plots/annotated.html" width="100%" height="500" frameborder="0"></iframe>

---

## `mass_error_plot()`

```python
from spxtacular.visualization import mass_error_plot

fig = mass_error_plot(
    spectrum,
    fragments,
    tolerance=0.02,
    tolerance_unit="da",         # or "ppm"
    peak_selection="closest",    # "closest" | "largest" | "all"
    unit="ppm",                  # error units
    title=None,
    max_labels=60,
    theme_mode=None,
    **layout_kwargs,
)
fig.show()
```

Bubble chart of fragment mass errors vs m/z. Each matched fragment is a bubble whose x-position is
the observed m/z, y-position is the mass error (ppm or Da), and size is proportional to the peak
intensity. Bubbles are coloured by ion series. Useful for spotting calibration drifts or
systematic mass errors. Also available as `Spectrum.mass_error_plot()`.

<iframe src="../plots/mass_errors.html" width="100%" height="500" frameborder="0"></iframe>

A well-calibrated instrument gives a cloud centred on zero and comfortably inside the search
tolerance, as above. A cloud offset from zero means a systematic calibration error; one that fans
out with m/z means the calibration is drifting across the mass range.

---

## `facet_plot()`

```python
from spxtacular.visualization import facet_plot

fig = facet_plot(
    spectrum,
    fragments=None,           # adds annotated panel + mass-error panel when provided
    mirror_spectrum=None,     # adds a mirror panel below when provided
    title=None,
    tolerance=0.02,
    tolerance_unit="da",
    peak_selection="closest",
    include_sequence=False,
    unit="ppm",
    max_labels=60,
    theme_mode=None,
    **layout_kwargs,
)
fig.show()
```

Multi-panel plot combining (1) the annotated spectrum, (2) the mass-error bubble chart, and (3) a
mirror spectrum — all on a shared m/z axis. Panels 2 and 3 are opt-in; supplying `fragments`
enables the mass-error panel and the annotations, supplying `mirror_spectrum` enables the mirror.
Also available as `Spectrum.facet_plot()`.

<iframe src="../plots/facet.html" width="100%" height="920" frameborder="0"></iframe>

The shared m/z axis is the point: zooming one panel zooms all three, so you can follow a single
peak from its annotation, to its mass error, to its deconvoluted counterpart.

---

## `sequence_coverage_plot()`

```python
from spxtacular.visualization import sequence_coverage_plot

fig = sequence_coverage_plot(
    spectrum,
    "FDSFGDLSSASAIMGNPK",   # stripped residue sequence
    fragments,
    tolerance=5,
    tolerance_unit="da",
    theme_mode=None,
    **layout_kwargs,
)
fig.show()
```

The coverage ladder: **where along the peptide** the evidence sits. An annotated spectrum tells you
that peaks matched; this tells you which backbone bonds those matches actually confirm, which is
what distinguishes a localised identification from one leaning on a single end of the molecule.

Residues run left to right. A tick **above and to the left** of a residue marks an N-terminal
(a/b/c) fragment ending at that bond; a tick **below and to the right** marks a C-terminal (x/y/z)
fragment starting there. A bond with ticks on both sides is confirmed from both directions. The
title reports the fraction of backbone bonds covered.

<iframe src="../plots/sequence_coverage.html" width="100%" height="260" frameborder="0"></iframe>

Pass the **stripped** sequence — ProForma modification brackets are not rendered. An empty peptide
raises `ValueError`.

---

## Chromatograms and XICs

Everything above works on a single spectrum. These two work on a *run* — any iterable of spectra
carrying retention times, which is what `reader.ms1` yields.

```python
with spx.Reader("run.d") as reader:
    spx.plot_chromatogram(reader.ms1).show()          # TIC, extracted for you
```

```python
with spx.Reader("run.d") as reader:
    spx.plot_xic(reader.ms1, [599.3262, 599.8268, 600.3306], tolerance=20).show()
```

`extract_chromatogram()` and `extract_xic()` return `Chromatogram` objects if you want the numbers
rather than a figure — each carries `rt`, `intensity`, `apex_rt` and `total`.

Extractors expect supplied retention times in seconds. When every scan lacks a retention time,
they use scan indices and record `meta["rt_unit"] = "scan_index"`. Plots label that axis as
`Scan index`. Mixing measured times with fallback indices raises `ValueError`, as does overlaying
traces with different axis units. `total` is a sum of intensities, without time integration.

Two things shape the API:

**Everything is one pass.** A reader is expensive to walk (several seconds for a 65-frame timsTOF
run) and `reader.ms1` may be a generator that cannot be replayed. So `extract_xic()` takes a *list*
of targets and extracts them all together — twenty traces cost one walk, not twenty.

**Any m/z order is accepted.** A timsTOF frame is ordered by ion-mobility scan and only sorted by
m/z *within* each scan, so it is not globally sorted. Each frame is sorted once on arrival when
needed, after which every target is a binary search.

### Ion mobility

On mobility data an `im_window` is what makes a trace selective — two co-eluting species at the
same m/z usually separate in mobility:

```python
spx.plot_xic(reader.ms1, [599.3262], tolerance=20, im_window=(1.01, 1.11))
```

`aggregate="sum"` (default) totals the peaks in the window, the quantification convention;
`aggregate="max"` takes the largest.

---

## Profile spectra

A profile spectrum samples a continuous signal, so drawing each sample as its own stick from the
baseline throws away the peak shape and triples the coordinate count. `plot_spectrum()` detects
`SpectrumType.PROFILE` and draws a connected trace instead:

```python
spec = Spectrum(mz=mz, intensity=intensity, spectrum_type=SpectrumType.PROFILE)
spec.plot()                      # continuous trace with a light fill
spec.plot(render="sticks")       # force the stick rendering
```

### Thinning

Real profile scans run to hundreds of thousands of samples, well past what a screen can resolve or
a browser can draw. Above `max_points` (default 4000) the trace is thinned by keeping the
**minimum and maximum of each bucket**.

!!! warning
    This matters more than it sounds. The obvious alternative — keeping every *N*th sample — will
    step straight over the two or three samples that form a narrow peak, and that peak silently
    disappears from the plot. On a 200,000-sample test spectrum with 40 narrow peaks, every-*N*th
    sampling preserved **0 of 40** apexes; min/max preserved **39 of 40**, and the global maximum
    exactly. Pass `max_points=None` to draw every sample.

### Checking centroiding

`profile_centroid_plot()` puts the centroided peaks on top of the profile trace, which is how you
confirm centroiding did the right thing:

```python
from spxtacular import profile_centroid_plot

profile_centroid_plot(profile)                    # centroids computed for you
profile_centroid_plot(profile, centroids=my_centroids)      # or supply your own
```

A stick off the apex means a mis-assigned centre. An apex with no stick means a peak was removed by
thresholding or was not detected. `centroid()` uses no intensity floor by default, so every local
maximum becomes a peak. On noisy data, pass `min_intensity="noise"` for the MAD-estimated floor or
provide an absolute threshold. Flat-topped peaks are supported and produce one centroid at the
middle of the plateau.

---

## `reporter_ion_plot()`

```python
fig = spx.reporter_ion_plot(
    spectrum,
    "TMT10",                 # plex name, IsobaricTagInfo, or an extracted ReporterIons
    tolerance=20.0,
    tolerance_unit="ppm",    # or "da"
    impurities=None,         # lot-sheet impurity table, for corrected bars
    show_spectrum=True,      # the raw reporter region above the bars
    normalize=True,          # bars as % of the strongest channel
)
```

One bar per channel, in channel order, from
[`extract_reporter_ions`](scoring.md) (so the plot and the quantification read the same peaks).
Above it, the raw reporter m/z region, with the peak picked for each channel highlighted: an
interfering peak or a missing channel is visible, not hidden in a bar. A channel with no peak is
marked *n.d.*. Pass an already extracted `ReporterIons` to plot corrected or normalized values
exactly as you computed them.

---

## `compose_figure()`

```python
parts = [
    spx.annotate_spectrum(spec, fragments, peptide=peptide, backend="spec"),
    spx.mirror_plot(query, library, fragments=fragments, backend="spec"),
    spx.mass_error_plot(spec, fragments, backend="spec"),
    spx.reporter_ion_plot(spec, "TMT10", backend="spec"),
]
fig = spx.compose_figure(parts, ncols=2, size="double", backend="matplotlib")
spx.save_figure(fig, "figure3.pdf")
```

Lays out several `FigureSpec`s (from `backend="spec"`) as one multi-panel figure, with bold panel
letters (`labels="abc"`, `"ABC"`, a list of strings, or `None`). The panels share one style and one
width, so fonts and line weights match across the figure. Rendered figures (plotly or matplotlib) are rejected with
`SpxtacularError`: build the parts with `backend="spec"`.

`spx.render(spec, "plotly")` (or `spec.render("matplotlib")`) draws a single spec.

---

## Theme

Colour lives in `spxtacular.theme`, and is assigned by the *job* it does rather than by taste:

| Job | What it encodes | How it is coloured |
|---|---|---|
| Ion type | which fragment series | The proteomics convention: **b** blue · **y** red · **a** green · **c** teal · **x** purple · **z** orange |
| Charge state | 1+, 2+, 3+ … | **Ordinal** — one hue, light to dark |
| `iso_score`, ion mobility | magnitude | Sequential — one hue, light to dark |
| Unmatched peaks | context, not subject | Recessive grey, thinner and dimmer |

Ion colours follow the convention used by Skyline, MetaDraw, IPSA and `spectrum_utils`, so a
spectrum from spxtacular reads the way you expect. The hues are this palette's own validated steps
picked from each conventional family, rather than copied hex values, so the convention is honoured
without giving up the colour-vision checks — **b** and **y**, the pair present in nearly every
spectrum, separate at CVD ΔE 21.6 (light) and 19.2 (dark), well above the ≥8 target.

The one pair to know about is **a vs y** — green against red, the classic confusion pair — which
sits at ΔE 7.2 in light mode. That is inside the band that is only safe alongside a second channel;
annotated spectra always carry direct ion labels, which provides it, and `texture=True` adds dash
patterns if you want more. Dark mode clears the target outright at 8.6.

Two further consequences. Charge is *ordinal*, so it takes a ramp rather than a categorical
cycle — you see the ordering in the colour, and charges beyond the ramp clamp to its dark end
instead of wrapping around to an earlier colour. And ion types past the eighth slot — including
internal fragments, whose types are two letters like `by` — fold to a neutral colour rather than
inventing a ninth hue that nobody could distinguish.

Every palette was checked with a colour-vision-deficiency validator (protanopia and deuteranopia)
against both the light and dark surfaces.

### Light and dark

```python
from spxtacular import theme

theme.set_plot_theme("dark")          # global default for every later plot
spec.plot(theme_mode="dark")          # or per call
```

The dark palette is its own set of steps chosen for the dark surface, not an automatic inversion of
the light one.

<iframe src="../plots/annotated_dark.html" width="100%" height="500" frameborder="0"></iframe>

### Compressed dynamic range

`intensity_transform="log"` (or `"sqrt"`) keeps low-abundance matched ions readable when one base
peak would otherwise flatten everything else:

<iframe src="../plots/annotated_log.html" width="100%" height="500" frameborder="0"></iframe>

### Brand colours

```python
theme.set_palette(
    categorical={"light": [...8 hues...], "dark": [...8 hues...]},
)
```

`set_palette` also accepts `charge_ramp` and `sequential`. Each takes both modes and raises
`ValueError` if one is missing, or if a categorical palette has fewer than eight hues.

> Substituted palettes are **not** validated for you. The shipped hues were chosen to stay
> distinguishable under colour-vision deficiency; if you replace them, check your own.

### Texture

`texture=True` on `annotate_spectrum()` gives each ion series its own dash pattern, so identity
survives print, forced-colours, and readers who cannot separate two hues:

```python
spx.annotate_spectrum(spec, fragments, tolerance=5, tolerance_unit="da", texture=True)
```

---

## `table_view()`

```python
from spxtacular import build_annot_plot_table, table_view

table = build_annot_plot_table(spec, fragments, tolerance=5, tolerance_unit="da")
html = table_view(table, max_rows=50, annotated_only=True)
```

Renders a plot table as an HTML `<table>`. A tooltip enhances a figure, it should never gate it —
and label capping deliberately drops labels off the plot, while a hover is unusable for keyboard
and screen-reader users. This gives those values a home that is not the tooltip. Label text is
HTML-escaped.

`annotated_only=True` keeps just the peaks carrying a label; `max_rows` keeps the *n* most intense.

The result renders as an ordinary table — here the six most intense annotated peaks from the
spectrum above:

<table><caption>Peak list</caption><thead><tr><th scope="col">m/z</th><th scope="col">Intensity</th><th scope="col">Annotation</th></tr></thead><tbody><tr><td>122.5865</td><td>1.599e+05</td><td>y2^2</td></tr><tr><td>649.7990</td><td>1.862e+05</td><td>b13^2</td></tr><tr><td>956.3997</td><td>1.194e+05</td><td>b9</td></tr><tr><td>1290.6364</td><td>9.6e+04</td><td>y13</td></tr><tr><td>1486.6503</td><td>1.283e+05</td><td>b15</td></tr><tr><td>1600.6938</td><td>1.022e+05</td><td>b16</td></tr></tbody></table>

Note the `Intensity` column carries the **true** value, not the relative-scaled one the y-axis
shows — the table is the place values are reported exactly.

---

## `save_figure()`

```python
from spxtacular import save_figure

save_figure(fig, "spectrum.html")            # plotly: always works
save_figure(fig, "figure.pdf")               # matplotlib: vector, fonts embedded
save_figure(fig, "figure.png", dpi=300)      # raster at 300 dpi (default: the style's dpi)
save_figure(spec, "figure.svg")              # a FigureSpec: drawn with matplotlib if installed
```

The file extension picks the writer, and the figure type picks the backend:

- **matplotlib figure:** `.pdf`, `.svg`, `.eps`, `.png`, `.jpg`, `.tif`, `.webp`. PDF and SVG are
  vector with TrueType fonts; rasters use `dpi` (default: the style's `dpi`, 600 for `"paper"`).
- **plotly figure:** `.html` (or no suffix) needs nothing extra. `.png`, `.svg`, `.pdf`, `.jpg`,
  `.webp` go through kaleido and raise `ImportError` naming `spxtacular[plotly-export]` if it is
  missing. `scale=` sets the device pixel ratio and overrides `dpi`.
- **`FigureSpec`:** drawn with matplotlib when it is installed, else plotly; `.html` always uses
  plotly.

An unsupported suffix raises `SpxtacularError`. Returns the path written.
