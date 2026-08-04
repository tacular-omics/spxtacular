# Visualization

spxtacular uses [Plotly](https://plotly.com/python/) for interactive HTML visualizations.
Every function on this page returns a `plotly.graph_objects.Figure` object. Plotly is a **required**
dependency of spxtacular (as is `pandas`, which backs the plot-table API), so nothing extra needs
installing.

## How these plots are built

A few conventions apply to every figure on this page. They are worth knowing up front, because
they explain defaults that would otherwise look surprising.

**Relative intensity is the default y-axis.** Peaks are scaled so the base peak is 100%, matching
the convention every MS viewer uses. Raw counts are still available with
`intensity_scale="absolute"`, and the tooltip always reports the *true* intensity either way —
rescaling changes the axis, never the number you are told. For data spanning orders of magnitude,
`intensity_transform="sqrt"` or `"log"` compresses the range so low-abundance matched ions stay
visible beside a dominant base peak.

**Labels are deliberately sparse.** Direct labels are capped (`max_labels`, default 25) and
collision-avoided along the m/z axis, strongest peak first. Labelling every annotated peak turns a
deconvoluted spectrum into an unreadable smear along the baseline. Nothing is lost: the dropped
values remain in the hover text, in the plot table, and in [`table_view()`](#table_view).

**Hovering does not require precision.** Every figure gets an m/z crosshair from the shared
template. Sticks are only about 1.5 pixels wide, so `plot_spectrum()` and `annotate_spectrum()`
additionally lay a transparent hit layer over the peak tips — being *near* a peak is enough.
(The other plots hover on their traces directly, which is fine for bubbles and subplot panels.)

**Colour is assigned by the job it does**, not by taste — see [Theme](#theme) below.

---

## `plot_spectrum()`

```python
from spxtacular.visualization import plot_spectrum

fig = plot_spectrum(
    spectrum,
    title=None,                  # plot title
    color="charge",              # "charge" | "im" | None
    show_scores=True,            # annotate scored peaks with their score value
    max_labels=25,               # cap on direct labels; None for no cap
    theme_mode=None,             # "light" | "dark"; None uses the global default
    intensity_scale="relative",  # "relative" (base peak = 100%) | "absolute"
    intensity_transform=None,    # None | "sqrt" | "log"
    show_precursor=True,         # draw precursor m/z + isolation window on MSn
    **layout_kwargs,             # passed to fig.update_layout()
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

The legacy `show_charges=True/False` keyword is still accepted as a deprecated alias mapping to
`color="charge"` / `color=None`.

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
    title=None,
    normalize=True,      # scale each half to its own maximum independently
    show_charges=True,   # colour the deconvoluted half by charge state
    show_scores=True,    # annotate deconvoluted peaks with iso_score
    max_labels=25,       # cap on score labels, strongest first
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

**Example:**

```python
from spxtacular import Spectrum
from spxtacular.visualization import mirror_plot

decon = spec.deconvolute(charge_range=(1, 5), tolerance=10, tolerance_type="ppm")
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
    tolerance_type="da",         # or "ppm"
    title=None,
    peak_selection="closest",    # "closest" | "largest" | "all"
    include_sequence=False,
    max_labels=25,
    theme_mode=None,
    intensity_scale="relative",
    intensity_transform=None,
    texture=False,
    show_precursor=True,
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
| `tolerance_type` | `"da"` | `"da"` or `"ppm"` |
| `title` | `None` | Plot title |
| `peak_selection` | `"closest"` | How to resolve multiple peaks within tolerance — `"closest"`, `"largest"`, or `"all"` |
| `include_sequence` | `False` | Embed the residue sequence in each label (e.g. `b3{PEP}`) |
| `max_labels` | `25` | Cap on direct labels, strongest first; `None` for no cap |
| `theme_mode` | `None` | `"light"` or `"dark"`; `None` uses the global default |
| `intensity_scale` | `"relative"` | `"relative"` (base peak = 100%) or `"absolute"` |
| `intensity_transform` | `None` | `None`, `"sqrt"` or `"log"` |
| `texture` | `False` | Give each ion series its own dash pattern |
| `show_precursor` | `True` | Draw the precursor m/z and isolation window on an `MsnSpectrum` |

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
    tolerance_type="da",         # or "ppm"
    peak_selection="closest",    # "closest" | "largest" | "all"
    unit="ppm",                  # error units
    title=None,
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
    tolerance_type="da",
    peak_selection="closest",
    include_sequence=False,
    unit="ppm",
    max_labels=25,
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
    tolerance_type="da",
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

## Theme

Colour lives in `spxtacular.theme`, and is assigned by the *job* it does rather than by taste:

| Job | What it encodes | How it is coloured |
|---|---|---|
| Ion type | which fragment series | Eight fixed categorical slots, in order `b y a c x z p i` |
| Charge state | 1+, 2+, 3+ … | **Ordinal** — one hue, light to dark |
| `iso_score`, ion mobility | magnitude | Sequential — one hue, light to dark |
| Unmatched peaks | context, not subject | Recessive grey, thinner and dimmer |

Two consequences worth knowing. Charge is *ordinal*, so it takes a ramp rather than a categorical
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
spx.annotate_spectrum(spec, fragments, tolerance=5, tolerance_type="da", texture=True)
```

---

## `table_view()`

```python
from spxtacular import build_annot_plot_table, table_view

table = build_annot_plot_table(spec, fragments, tolerance=5, tolerance_type="da")
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

save_figure(fig, "spectrum.html")        # always works
save_figure(fig, "figure.png", scale=2)  # needs: pip install kaleido
```

The file extension picks the writer. `.html` (or no suffix) needs nothing extra. Static formats —
`.png`, `.svg`, `.pdf`, `.jpg`, `.webp`, `.eps` — go through Plotly's static export and raise
`ImportError` naming `kaleido` if it is not installed. An unrecognised suffix raises `ValueError`.
`scale=2` renders at twice the device resolution, which is what you want for a paper figure.
