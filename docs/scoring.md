# Fragment matching and scoring

spxtacular provides two functions for peptide-spectrum match (PSM) scoring:
`match_fragments()` for matching and `score()` for computing all metrics at once.

---

## `match_fragments()`

```python
from spxtacular import match_fragments
import peptacular as pt

fragments = pt.fragment("PEPTIDE", ion_types=("b", "y"), charges=(1, 2))
matches = match_fragments(
    spectrum,
    fragments,
    tolerance=0.02,
    tolerance_type="da",         # "da" or "ppm"
    peak_selection="closest",    # "closest", "largest", or "all"
)
# matches: list[MatchedFragment], sorted by peak index
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `spectrum` | | `Spectrum` to search |
| `fragments` | | Iterable of fragment objects from `peptacular` |
| `tolerance` | `0.02` | Matching tolerance |
| `tolerance_type` | `"da"` | `"da"` or `"ppm"` |
| `peak_selection` | `"closest"` | How to resolve multiple peaks within tolerance (see below) |
| `is_monoisotopic` | `True` | Forwarded to the constructed `Fragment` objects when `fragments` is a `dict[(IonType, charge_state), list[float]]` (peptacular's `fast_fragment` output); otherwise has no effect |

**`peak_selection` modes:**

| Mode | Behaviour |
|---|---|
| `"closest"` | Keep the single peak with smallest m/z error (default) |
| `"largest"` | Keep the single peak with highest intensity |
| `"all"` | Keep every peak within tolerance |

**Return value:**

`match_fragments()` returns a `list[MatchedFragment]`, sorted by ascending `peak_index`. Each `MatchedFragment` carries both the fragment and the matched peak's metadata:

| Field | Description |
|---|---|
| `fragment` | The matched `Fragment` object |
| `peak_index` | Index of the matched peak in `spectrum.mz`/`spectrum.intensity` |
| `peak_mz` | m/z of the matched peak |
| `peak_intensity` | Intensity of the matched peak |
| `intensity_pct` | `peak_intensity / total_spectrum_intensity * 100` |
| `ppm_error` | Signed error: `(peak_mz - theoretical_mz) / theoretical_mz * 1e6` |
| `da_error` | Signed error: `peak_mz - theoretical_mz` |

**Example:**

```python
matches = match_fragments(spec, fragments, tolerance=10, tolerance_type="ppm")
for m in matches:
    print(f"  Peak {m.peak_index} ({m.peak_mz:.4f} m/z) matched {m.fragment} (Δ={m.ppm_error:+.1f} ppm)")
```

---

## `score()`

Runs `match_fragments()` internally and returns all scoring metrics as a dict.

```python
from spxtacular import score

result = score(spectrum, fragments, tolerance=10, tolerance_type="ppm")
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `spectrum` | | `Spectrum` to score against |
| `fragments` | | Iterable of fragment objects from `peptacular` |
| `tolerance` | `0.02` | Matching tolerance |
| `tolerance_type` | `"da"` | `"da"` or `"ppm"` |
| `peak_selection` | `"closest"` | How to resolve multiple peaks within tolerance — `"closest"`, `"largest"`, or `"all"` |
| `predicted_intensities` | `None` | Optional predicted intensity for every fragment, in the same order. Enables the literature spectral-angle metric |

**Returned metrics:**

| Key | Description |
|---|---|
| `hyperscore` | X!Tandem hyperscore: log10(∏ₛ ΣIₛ) + Σₛ log10(nₛ!) over the searched ion series |
| `probability_score` | -log10 P(>= k matches by chance) |
| `total_matched_intensity` | Sum of matched peak intensities |
| `matched_fraction` | Fraction of theoretical ions matched |
| `intensity_fraction` | Fraction of total spectrum intensity explained by matches |
| `mean_ppm_error` | Mean absolute ppm error of matches |
| `spectral_angle` | Spectral angle vs `predicted_intensities` (0–1). Without a prediction, a coverage/evenness fallback — see below |
| `longest_run` | Longest consecutive ion sequence matched |

Neutral-loss and isotope variants of the same fragment share `(ion_type, position)` and are
collapsed to avoid inflating factorial terms in the hyperscore.

### Hyperscore

For a b/y search this is numerically identical to the X!Tandem hyperscore, so values are
comparable with X!Tandem, Comet and MSFragger. The product runs over whichever series you
*searched* rather than a hardcoded b/y, so an ETD c/z search is scored the same way.

The product is what makes it discriminating: a searched series with no signal collapses the whole
score to zero, so a PSM supported only by b ions cannot look as convincing as one corroborated
from both directions.

!!! warning
    The intensity term uses **raw** intensities, so the score shifts by `log10(s)` per series if
    the spectrum is scaled by `s`, and can go negative on a normalised spectrum. Only compare
    hyperscores computed on identically scaled spectra.

### Spectral angle

Supply `predicted_intensities` — one value per fragment, in the same order — and you get the
spectral angle of the literature (Toprak et al.; the metric Prosit and Spectronaut report):

```python
result = spx.score(
    spec, fragments,
    tolerance=10, tolerance_type="ppm",
    predicted_intensities=predicted,   # aligned with `fragments`
)
```

It is a cosine, so it is scale-invariant: 1.0 means the observed pattern matches the prediction.

!!! warning
    Without `predicted_intensities` there is nothing to compare against, and the value falls back
    to a cosine against a *flat* reference — which measures intensity evenness × coverage, not
    similarity to a predicted spectrum. A perfect match with realistic intensities `[100, 50, 10, 1]`
    scores 0.509 that way. Do not compare the fallback to published spectral angles.

**Example:**

```python
import peptacular as pt
from spxtacular import MzmlReader, score

with MzmlReader("run.mzML") as reader:
    # reader.ms2 is an iterable lookup object, not an iterator — wrap it in iter()
    spec = next(iter(reader.ms2))

fragments = pt.fragment("ACDEFGHIK", ion_types=("b", "y"), charges=(1, 2))
result = score(spec, fragments, tolerance=10, tolerance_type="ppm")

print(f"Hyperscore:      {result['hyperscore']:.3f}")
print(f"Spectral angle:  {result['spectral_angle']:.3f}")
print(f"Matched ions:    {result['matched_fraction']:.1%}")
```


---

## Spectrum-to-spectrum similarity

`score()` answers *how well does this peptide explain this spectrum*. These answer *how alike are
these two spectra*, which is what spectral library search, replicate comparison and clustering are
built on.

```python
from spxtacular import cosine, modified_cosine, entropy_similarity

cosine(query, reference, tolerance=20, tolerance_type="ppm")   # 0-1
entropy_similarity(query, reference, tolerance=0.02)           # 0-1
```

| Function | What it is |
|---|---|
| `cosine` | The standard spectral dot product: sqrt-transformed intensities, unit-normalised, peaks matched one-to-one |
| `modified_cosine` | Cosine that also matches peaks displaced by the precursor mass difference — the GNPS molecular-networking metric |
| `entropy_similarity` | Entropy similarity (Li et al. 2021), which discriminates more sharply and has largely displaced cosine for library search |

All three are symmetric, scale-invariant, and bounded in `[0, 1]`: identical spectra score 1,
spectra with no shared peaks score 0.

Matching is **one-to-one** — a peak may back at most one match, resolved greedily by descending
contribution. Allowing every pair within tolerance instead would let one intense peak match several
neighbours and push the score past 1.

### Modified cosine

Two spectra of the same molecule differing by one modification share many fragments, but every
fragment containing the modified site is shifted by the modification's mass. A plain cosine reads
those as mismatches:

```python
# same peptide, one +79.966 phospho on the C-terminal half
cosine(a, b, tolerance=0.02)                              # 0.53 - looks unrelated
modified_cosine(a, b, 500.0, 579.966, tolerance=0.02)     # 1.00 - recovered
```

It reduces exactly to `cosine` when the two precursors are equal.
