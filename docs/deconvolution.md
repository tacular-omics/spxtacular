# Deconvolution

Deconvolution takes a raw centroid spectrum and groups peaks into **isotope clusters**, assigning each cluster a monoisotopic m/z and a charge state. The output is still a spectrum — same `mz` and `intensity` arrays — but now with a `charge` array and `spectrum_type = DECONVOLUTED`.

Converting those charged peaks to neutral masses is a separate step: see [`decharge`](spectrum.md#decharge).

---

## Basic usage

```python
from spxtacular import Spectrum
import numpy as np

mz = np.array([500.10, 500.60, 501.11, 800.20, 800.70], dtype=np.float64)
intensity = np.array([1e5, 8e4, 3e4, 2e5, 1.5e5], dtype=np.float64)
spec = Spectrum(mz=mz, intensity=intensity)

decon = spec.deconvolute(
    charge_range=(1, 5),
    tolerance=10,
    tolerance_type="ppm",
)

print(decon.mz)      # monoisotopic m/z, one entry per cluster (or singleton)
print(decon.charge)  # charge state per peak; -1 for singletons
print(decon.intensity)
```

The output has **one peak per identified cluster**, not one peak per input peak. A cluster of four isotope peaks at z=2 collapses to a single output entry at the monoisotopic m/z with `charge=2`.

---

## How the algorithm works

The implementation lives in `src/spxtacular/decon/` — pure NumPy, no graph construction.

### 1. Seed selection

Pick the **most intense unused peak** as the seed for the next cluster. High-intensity clusters are therefore assigned first.

### 2. Cluster building

For each charge state in `charge_range`, extend forward from the seed in steps of `NEUTRON_MASS / z` Da, where `NEUTRON_MASS` is the C13 neutron mass (~1.00335 Da). Up to 9 additional peaks are added (maximum cluster size: 10).

At each step, the algorithm looks for an unused peak within the tolerance window. The closest candidate is chosen. If none exists, the cluster terminates — **no skips are allowed**.

### 3. Scoring

Each candidate cluster is scored against a theoretical isotope distribution using the **Bhattacharyya coefficient**, penalised for missed peaks that should have been detectable above `min_intensity`. The score is in the range 0–1, where 1 is a perfect match to the theoretical envelope.

### 4. Charge assignment

The charge with the **highest score** wins. Ties are broken by cluster size.

### 5. Rejection

If the winning score is below `min_score`, the seed is marked as a **singleton** (`charge=-1`, `score=0.0`). The other peaks that were tested as cluster members remain available as seeds for future iterations.

### 6. Repeat

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
    min_intensity: float | str = "min",
    min_score: float = 0.0,
    inplace: bool = False,
) -> Self
```

| Parameter | Default | Description |
|---|---|---|
| `tolerance` | `50` | Peak matching tolerance |
| `tolerance_type` | `"ppm"` | `"ppm"` or `"da"` |
| `charge_range` | `(1, 3)` | Min and max charge to try, inclusive |
| `intensity` | `"total"` | `"total"` sums all cluster peaks; `"base"` uses only the seed (monoisotopic) peak |
| `max_dpeaks` | `2000` | Upper bound on output peaks |
| `min_intensity` | `"min"` | Intensity floor for detectability scoring. `"min"` uses the spectrum minimum |
| `min_score` | `0.0` | Minimum profile score to accept a cluster; `0.0` accepts everything |
| `inplace` | `False` | Mutate in place instead of returning a new `Spectrum` |

---

## Score output

After deconvolution, `spectrum.score` is a `float64` array parallel to `mz`/`intensity`. Each assigned cluster carries a score in 0–1 representing how well its observed intensity distribution matches the theoretical isotope envelope. Singletons always have `score=0.0`.

```python
decon = spec.deconvolute(charge_range=(1, 5), tolerance=10, tolerance_type="ppm")
print(decon.score)   # array of float64, same length as decon.mz

# Keep only well-matched clusters (score >= 0.5) and assigned peaks (charge > 0)
confident = decon.filter(min_score=0.5, min_charge=1)
```

The `score` array is propagated through `.decharge()`, so neutral-mass peaks retain their cluster score.

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

---

## Worked example

```python
import numpy as np
from spxtacular import Spectrum

# Simulated z=2 peptide: monoisotopic at 700.35, two isotope peaks
# Expected isotope spacing: 1.00335 / 2 ≈ 0.5017 Da
mz = np.array([
    700.35, 700.85, 701.35,   # z=2 cluster
    450.20,                    # singleton
], dtype=np.float64)
intensity = np.array([8e4, 6e4, 2e4, 1e3], dtype=np.float64)

spec = Spectrum(mz=mz, intensity=intensity)
decon = spec.deconvolute(charge_range=(1, 4), tolerance=10, tolerance_type="ppm")

for mz_val, z, inten in zip(decon.mz, decon.charge, decon.intensity):
    label = f"z={z}" if z != -1 else "singleton"
    print(f"  mz={mz_val:.4f}  {label}  intensity={inten:.2e}")
```

Expected output:

```
  mz=450.2000  singleton  intensity=1.00e+03
  mz=700.3500  z=2        intensity=1.60e+05
```

Two input peaks collapsed into one cluster entry. The singleton is preserved with `charge=-1`.

To convert the assigned peaks to neutral masses, chain `.decharge()`:

```python
neutral = decon.decharge()
# drops singletons, outputs neutral monoisotopic masses
```
