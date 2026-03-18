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
    mz_tol=0.02,
    mz_tol_type="Da",            # "Da" or "ppm"
    peak_selection="closest",    # "closest", "largest", or "all"
)
# matches: list of (peak_index, Fragment) pairs, sorted by peak index
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `spectrum` | | `Spectrum` to search |
| `fragments` | | Iterable of fragment objects from `peptacular` |
| `mz_tol` | `0.02` | Matching tolerance |
| `mz_tol_type` | `"Da"` | `"Da"` or `"ppm"` |
| `peak_selection` | `"closest"` | How to resolve multiple peaks within tolerance (see below) |

**`peak_selection` modes:**

| Mode | Behaviour |
|---|---|
| `"closest"` | Keep the single peak with smallest m/z error (default) |
| `"largest"` | Keep the single peak with highest intensity |
| `"all"` | Keep every peak within tolerance |

**Example:**

```python
matches = match_fragments(spec, fragments, mz_tol=10, mz_tol_type="ppm")
for peak_idx, frag in matches:
    print(f"  Peak {peak_idx} ({spec.mz[peak_idx]:.4f} m/z) matched {frag}")
```

---

## `score()`

Runs `match_fragments()` internally and returns all scoring metrics as a dict.

```python
from spxtacular import score

result = score(spectrum, fragments, mz_tol=10, mz_tol_type="ppm")
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `spectrum` | | `Spectrum` to score against |
| `fragments` | | Iterable of fragment objects from `peptacular` |
| `mz_tol` | `10` | Matching tolerance |
| `mz_tol_type` | `"ppm"` | `"Da"` or `"ppm"` |

**Returned metrics:**

| Key | Description |
|---|---|
| `hyperscore` | log10(sum matched intensities) + sum log10(n!) per ion series |
| `probability_score` | -log10 P(>= k matches by chance) |
| `total_matched_intensity` | Sum of matched peak intensities |
| `matched_fraction` | Fraction of theoretical ions matched |
| `intensity_fraction` | Fraction of total spectrum intensity explained by matches |
| `mean_ppm_error` | Mean absolute ppm error of matches |
| `spectral_angle` | Normalised spectral angle (0-1, higher is better) |
| `longest_run` | Longest consecutive ion sequence matched |

Neutral-loss and isotope variants of the same fragment share `(ion_type, position)` and are
collapsed to avoid inflating factorial terms in the hyperscore.

**Example:**

```python
import peptacular as pt
from spxtacular import MzmlReader, score

reader = MzmlReader("run.mzML")
spec = next(reader.ms2)

fragments = pt.fragment("ACDEFGHIK", ion_types=("b", "y"), charges=(1, 2))
result = score(spec, fragments, mz_tol=10, mz_tol_type="ppm")

print(f"Hyperscore:      {result['hyperscore']:.3f}")
print(f"Spectral angle:  {result['spectral_angle']:.3f}")
print(f"Matched ions:    {result['matched_fraction']:.1%}")
```
