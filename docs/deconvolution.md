# Deconvolution

Deconvolution takes a raw centroid spectrum and groups peaks into **isotope clusters**, assigning each cluster a monoisotopic m/z and a charge state. The output is still a spectrum — same `mz` and `intensity` arrays — but now with a `charge` array and `spectrum_type = DECONVOLUTED`.

Converting those charged peaks to neutral masses is a separate step: see [`decharge`](spectrum.md#decharge).

---

## Basic usage

```python
from spxtacular import Spectrum
import numpy as np

# A 2+ cluster of three peaks, then a 1+ cluster of two.
mz = np.array([500.2573, 500.7590, 501.2606, 800.2000, 801.2033], dtype=np.float64)
intensity = np.array([1e5, 5.2e4, 1.1e4, 2e5, 6.5e4], dtype=np.float64)
spec = Spectrum(mz=mz, intensity=intensity)

decon = spec.deconvolute(
    charge_range=(1, 5),
    tolerance=15,
    tolerance_type="ppm",
)

print(decon.mz)         # monoisotopic m/z, one entry per cluster (or singleton)
print(decon.charge)     # charge state per peak; -1 for singletons
print(decon.intensity)  # summed over the cluster
print(decon.iso_score)  # isotopic profile fit, 0–1
```

```text
[500.2573 800.2   ]
[2 1]
[163000. 265000.]
[0.99834984 0.8899464 ]
```

The output has **one peak per identified cluster**, not one peak per input peak: five input peaks became two entries. A cluster of three isotope peaks at z=2 collapses to a single output entry at the monoisotopic m/z with `charge=2`, and its intensity is the sum over the cluster.

---

## How the algorithm works

The implementation lives in `src/spxtacular/decon/` — pure NumPy, no graph construction.

### 1. Seed selection

Pick the **most intense unused peak** as the seed for the next cluster. High-intensity clusters are therefore assigned first.

### 2. Anchor search

The seed is the most intense peak, but that is **not** the same as the monoisotopic peak. As mass
grows the isotope envelope's apex drifts: above roughly 1900 Da the A+1 peak is more intense than
A, and above ~3500 Da it is A+2. Anchoring blindly on the seed would report a monoisotopic mass one
or two neutrons too high.

So for each charge, the algorithm first walks *backwards* from the seed in steps of
`NEUTRON_MASS / z`, collecting up to four candidate anchor positions. Each candidate is then built
out and scored, and the alignment that best fits the isotope template wins.

### 3. Cluster building

From a given anchor, extend forward in steps of `NEUTRON_MASS / z` Da, where `NEUTRON_MASS` is the
C13 neutron mass (~1.00335 Da). Up to 9 additional peaks are added (maximum cluster size: 10).

At each step, the algorithm looks for an unused peak within the tolerance window of the *expected*
position `anchor + k * step` — measuring from the expected position rather than the previous match
prevents a chain of peaks each just inside tolerance from ratcheting the cluster off target. The
closest candidate is chosen. If none exists, the cluster terminates — **no skips are allowed**.

### 4. Scoring

Each candidate cluster is scored against a theoretical isotope distribution using the
**Bhattacharyya coefficient**, penalised for missed peaks that should have been detectable above
`min_intensity`. The score is in the range 0–1, where 1 is a perfect match to the theoretical
envelope. A cluster of a single peak scores `0.0` — one peak is no evidence of a charge state.

### 5. Charge and anchor assignment

The (charge, anchor) combination with the **highest score** wins. Ties are broken by cluster size.
A candidate whose cluster does not reach back to the seed is skipped, since it describes a
different feature.

### 6. Rejection

If the winning score is below `min_score`, the seed is marked as a **singleton** (`charge=-1`, `iso_score=0.0`). The other peaks that were tested as cluster members remain available as seeds for future iterations.

### 7. Repeat

All peaks in the winning cluster are marked as used. The cycle restarts from the next most-intense unused peak until every input peak has been consumed or `max_dpeaks` is reached.

---

## Parameters

```python
def deconvolute(
    self,
    tolerance: float = 50,
    tolerance_type: Literal["ppm", "da"] = "ppm",
    charge_range: tuple[int, int] = (1, 3),
    intensity: Literal["base", "total"] = "total",
    max_dpeaks: int = 2000,
    inplace: bool = False,
    min_intensity: float | Literal["min"] = "min",
    min_score: float = 0.0,
) -> Self
```

Note the positional order: `inplace` comes **before** `min_intensity` and `min_score`. Pass the
latter two by keyword.

| Parameter | Default | Description |
|---|---|---|
| `tolerance` | `50` | Peak matching tolerance |
| `tolerance_type` | `"ppm"` | `"ppm"` or `"da"` |
| `charge_range` | `(1, 3)` | Min and max charge to try, inclusive. Requires `1 <= min <= max`, else `ValueError` |
| `intensity` | `"total"` | `"total"` sums all cluster peaks; `"base"` uses only the seed (monoisotopic) peak |
| `max_dpeaks` | `2000` | Upper bound on output peaks |
| `inplace` | `False` | Mutate in place instead of returning a new `Spectrum` |
| `min_intensity` | `"min"` | Intensity floor for detectability scoring. `"min"` uses the spectrum minimum |
| `min_score` | `0.0` | Minimum profile score to accept a cluster; `0.0` accepts everything |

Calling `deconvolute()` on an already-`DECONVOLUTED` spectrum emits a `UserWarning` and returns it
unchanged.

---

## Score output

After deconvolution, `spectrum.iso_score` is a `float64` array parallel to `mz`/`intensity`. Each assigned cluster carries a score in 0–1 representing how well its observed intensity distribution matches the theoretical isotope envelope. Singletons always have `iso_score=0.0`.

```python
decon = spec.deconvolute(charge_range=(1, 5), tolerance=10, tolerance_type="ppm")
print(decon.iso_score)   # array of float64, same length as decon.mz

# Keep only well-matched clusters (score >= 0.5) and assigned peaks (charge > 0)
confident = decon.filter(min_score=0.5, min_charge=1)
```

The `iso_score` array is propagated through `.decharge()`, so neutral-mass peaks retain their cluster score.

---

**`charge_range`:** Cover the full range you expect. A wider range increases runtime linearly. For tryptic peptides `(1, 5)` is typical; for intact proteins `(5, 50)` or wider.

**`tolerance`:** The default 50 ppm is conservative. For high-resolution instruments (Orbitrap, timsTOF) use 5–15 ppm.

**`intensity` mode:** `"total"` is recommended for quantification — it captures the full isotope envelope signal. Use `"base"` if downstream tools expect monoisotopic intensity only.

---

## Charge conventions

| `charge` value | Meaning |
|---|---|
| `> 0` | Assigned isotope cluster with this charge state |
| `-1` | Singleton — no isotope neighbours found at any tested charge |
| `0` | After `.decharge()` — neutral mass, charge state no longer tracked |

---

## Worked example

```python
import numpy as np
from spxtacular import Spectrum

# Simulated z=2 peptide of neutral mass 1398.70, with its isotope envelope.
# Isotope spacing at z=2 is 1.00335 / 2 ≈ 0.5017 Da.
mz = np.array([
    450.2000,                                       # singleton
    700.3573, 700.8590, 701.3606, 701.8623,         # z=2 cluster
], dtype=np.float64)
intensity = np.array([1e3, 80000.0, 58614.0, 21276.0, 3828.0], dtype=np.float64)

spec = Spectrum(mz=mz, intensity=intensity)
decon = spec.deconvolute(charge_range=(1, 4), tolerance=10, tolerance_type="ppm")

for mz_val, z, inten, score in zip(decon.mz, decon.charge, decon.intensity, decon.iso_score):
    label = f"z={z}" if z != -1 else "singleton"
    print(f"  mz={mz_val:.4f}  {label:<9} intensity={inten:.2e}  score={score:.3f}")
```

Expected output:

```
  mz=450.2000  singleton intensity=1.00e+03  score=0.000
  mz=700.3573  z=2       intensity=1.64e+05  score=0.968
```

Four input peaks collapsed into one cluster entry at the monoisotopic m/z, with the intensities
summed. The singleton is preserved with `charge=-1` and `iso_score=0.0`.

Note that the envelope matters: supply only the first two or three isotope peaks of a large
cluster and the missing tail is penalised, which can leave a z=1 reading of the same peaks scoring
marginally higher. Deconvolution is most reliable on complete envelopes.

To convert the assigned peaks to neutral masses, chain `.decharge()`:

```python
neutral = decon.decharge()
# drops singletons, outputs neutral monoisotopic masses
```
