# Visualization

spxtacular uses [Plotly](https://plotly.com/python/) for interactive HTML visualizations.
All three functions return a `plotly.graph_objects.Figure` object. Plotly is an optional
dependency — install it with `pip install plotly`.

---

## `plot_spectrum()`

```python
from spxtacular.visualization import plot_spectrum

fig = plot_spectrum(
    spectrum,
    title=None,          # plot title
    show_charges=True,   # colour sticks by charge state
    show_scores=True,    # annotate scored peaks with their score value
    **layout_kwargs,     # passed to fig.update_layout()
)
fig.show()
```

Draws a stick plot of any `Spectrum`. When `charge` is present, sticks are coloured by charge
state (z=1, z=2, ... in different colours; z=-1 singletons in grey). When `score` is present and
`show_scores=True`, score values are shown above each scored peak.

`Spectrum.plot()` is a convenience wrapper around this function:

```python
spec.plot(title="My spectrum", show_charges=True).show()
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
    raw,             # Spectrum -- drawn inverted below the x-axis
    decon,           # Spectrum -- drawn upright above the x-axis
    title=None,
    normalize=True,  # scale each half to its own maximum independently
    show_scores=True,
    **layout_kwargs,
)
fig.show()
```

Mirror plot for comparing a raw spectrum (inverted, below) against its deconvoluted counterpart
(upright, above). Useful for visually confirming that isotope clusters have been correctly
identified and scored. Deconvoluted peaks are coloured by charge state; score annotations appear
above each cluster.

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
    mz_tol=0.02,
    mz_tol_type="da",   # or "ppm"
    title=None,
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
| `mz_tol` | `0.02` | Matching tolerance |
| `mz_tol_type` | `"da"` | `"da"` or `"ppm"` |
| `title` | `None` | Plot title |

**Annotated spectrum:**

<iframe src="../plots/annotated.html" width="100%" height="500" frameborder="0"></iframe>
