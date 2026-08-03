# Visualization

spxtacular uses [Plotly](https://plotly.com/python/) for interactive HTML visualizations.
Every function on this page returns a `plotly.graph_objects.Figure` object. Plotly is a **required**
dependency of spxtacular (as is `pandas`, which backs the plot-table API), so nothing extra needs
installing.

---

## `plot_spectrum()`

```python
from spxtacular.visualization import plot_spectrum

fig = plot_spectrum(
    spectrum,
    title=None,          # plot title
    color="charge",      # "charge" | "im" | None
    show_scores=True,    # annotate scored peaks with their score value
    **layout_kwargs,     # passed to fig.update_layout()
)
fig.show()
```

Draws a stick plot of any `Spectrum`. With `color="charge"` (default), sticks are coloured by
charge state when `charge` is present (z=1, z=2, … in different colours; z=-1 singletons in grey).
With `color="im"`, sticks are coloured by ion mobility on a continuous Viridis scale (falls back
to `"charge"` when no IM array is present). With `color=None`, all sticks render in a uniform
steelblue. When `iso_score` is present and `show_scores=True`, score values are shown above each
scored peak.

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
    **layout_kwargs,
)
fig.show()
```

Mirror plot for comparing a raw spectrum (inverted, below) against its deconvoluted counterpart
(upright, above). Useful for visually confirming that isotope clusters have been correctly
identified and scored. With `show_charges=True` (default) deconvoluted peaks are coloured by charge
state; with `show_scores=True` the `iso_score` annotations appear above each cluster.

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
    **layout_kwargs,
)
fig.show()
```

Draws the spectrum as a stick plot and overlays matched fragment ion labels (b/y ions coloured
by ion type). Unmatched peaks are shown in grey.

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

**Annotated spectrum:**

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
    **layout_kwargs,
)
fig.show()
```

Bubble chart of fragment mass errors vs m/z. Each matched fragment is a bubble whose x-position is
the observed m/z, y-position is the mass error (ppm or Da), and size is proportional to the peak
intensity. Bubbles are coloured by ion series. Useful for spotting calibration drifts or
systematic mass errors. Also available as `Spectrum.mass_error_plot()`.

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
    **layout_kwargs,
)
fig.show()
```

Multi-panel plot combining (1) the annotated spectrum, (2) the mass-error bubble chart, and (3) a
mirror spectrum — all on a shared m/z axis. Panels 2 and 3 are opt-in; supplying `fragments`
enables the mass-error panel and the annotations, supplying `mirror_spectrum` enables the mirror.
Also available as `Spectrum.facet_plot()`.
