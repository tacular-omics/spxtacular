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
result = score(spec, fragments, tolerance=10, tolerance_type="ppm")

print(f"Hyperscore:      {result['hyperscore']:.3f}")
print(f"Spectral angle:  {result['spectral_angle']:.3f}")
print(f"Matched ions:    {result['matched_fraction']:.1%}")
```
