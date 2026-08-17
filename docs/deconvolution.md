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

### 2. Apex alignment

The seed is treated as the observed envelope apex. For every charge in `charge_range`, the selected
isotope model predicts which isotope is most abundant. The algorithm tests the exact theoretical
apex and the contiguous near-apex positions whose predicted abundance is at least 90% of the
maximum. It matches and scores the complete envelope for each alignment. This prevents small
intensity fluctuations across a broad high-mass apex from causing a one-isotope error in the
inferred monoisotopic m/z. The mass and theoretical apex are recalculated until stable.

### 3. Cluster building

Starting at the apex, the algorithm walks left and right independently in steps of
`NEUTRON_MASS / z`, where `NEUTRON_MASS` is the C13 neutron mass (~1.00335 Da). At each position,
all unused peaks inside the m/z window are considered. Candidates outside the abundance fold gate
or ion-mobility window are rejected. The remaining candidates are ranked by the sum of squared,
normalised m/z error, log-abundance error, and ion-mobility error. Ion mobility participates only
when the spectrum contains it. Expansion in that direction stops when the theoretical abundance is
too low, no peak is found beyond the allowed gap count, or every candidate fails a hard gate. A
blocking peak is left unused for a later greedy pass.

Envelope length is adaptive by default. `max_isotopes` can impose a hard limit when runtime or a
known acquisition range warrants one.

### 4. Scoring

Each candidate cluster is scored against a theoretical isotope distribution using the
**Bhattacharyya coefficient**. This comparison is two-sided: both missing predicted intensity and
unexpected intensity among the aligned candidate peaks lower the coefficient. Aligned observed
entries participate even where the corresponding model entry falls below the detectability
cutoffs. Missing peaks that should have been detectable above `min_intensity` receive an additional
penalty; absent theoretical peaks below that floor do not. The score is in the range 0–1, where 1 is
a perfect match to the theoretical envelope. A cluster of a single peak scores `0.0` — one peak is
no evidence of a charge state.

The theoretical envelope is calculated with a BRAIN-style recurrence and cached at one-Dalton
resolution. Built-in average-composition models are available for peptides, glycans, lipids, DNA,
and RNA. The peptide model is the default.

### 5. Charge assignment

Every charge candidate is evaluated without modifying the input state. The candidate with the
**highest score** wins, with ties broken by matched peak count.

### 6. Rejection

If the winning score is below `min_score`, the seed is marked as a **singleton** (`charge=-1`, `iso_score=0.0`). The other peaks that were tested as cluster members remain available as seeds for future iterations.

### 7. Repeat

Only peaks accepted into the winning cluster are marked as used. Missing, abundance-rejected, and
fold-rejected peaks remain available. The cycle restarts from the next most-intense unused peak
until every input peak has been consumed or `max_dpeaks` is reached.

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
    isotope_model: IsotopeModel | IsotopeModelType | str = "peptide",
    min_isotope_abundance: float = 0.01,
    max_isotope_fold_error: float = 2.0,
    max_isotope_gaps: int = 0,
    max_isotopes: int | None = None,
    im_tolerance: float = 0.05,
    im_tolerance_type: Literal["relative", "absolute"] = "relative",
    ionization_model: IonizationModel | str | float | None = None,
) -> Self
```

Note the positional order: `inplace` comes **before** `min_intensity` and `min_score`. Pass the
latter two by keyword.

| Parameter | Default | Description |
|---|---|---|
| `tolerance` | `50` | Peak matching tolerance |
| `tolerance_type` | `"ppm"` | `"ppm"` or `"da"` |
| `charge_range` | `(1, 3)` | Min and max charge to try, inclusive. Requires `1 <= min <= max`, else `ValueError` |
| `intensity` | `"total"` | `"total"` sums matched peaks; `"base"` uses observed A+0, or zero when A+0 is absent |
| `max_dpeaks` | `2000` | Upper bound on output peaks |
| `inplace` | `False` | Mutate in place instead of returning a new `Spectrum` |
| `min_intensity` | `"min"` | Intensity floor for detectability scoring. `"min"` uses the spectrum minimum |
| `min_score` | `0.0` | Minimum profile score to accept a cluster; `0.0` accepts everything |
| `isotope_model` | `"peptide"` | `"peptide"`, `"glycan"`, `"lipid"`, `"dna"`, `"rna"`, or a custom `IsotopeModel` |
| `min_isotope_abundance` | `0.01` | Stop when theoretical abundance falls below 1% of the apex |
| `max_isotope_fold_error` | `2.0` | Stop when observed intensity is outside 0.5x to 2x expected |
| `max_isotope_gaps` | `0` | Missing isotope positions allowed before stopping one direction |
| `max_isotopes` | `None` | Adaptive by default; an integer imposes a hard envelope-length limit |
| `im_tolerance` | `0.05` | Candidate-to-seed mobility gate when the spectrum contains ion mobility |
| `im_tolerance_type` | `"relative"` | Scale the mobility tolerance by the seed value or use it as an absolute difference |
| `ionization_model` | `None` | Adduct preset/alias, custom model, or signed carrier mass. Defaults from scan polarity, with positive protonation as the fallback |

Calling `deconvolute()` on an already-`DECONVOLUTED` spectrum emits a `UserWarning` and returns it
unchanged.

### Isotope models

Select a built-in model by string or enum:

```python
from spxtacular import IsotopeModelType

glycan = spectrum.deconvolute(isotope_model="glycan")
rna = spectrum.deconvolute(isotope_model=IsotopeModelType.RNA)
```

Define a custom average composition as expected atoms per Dalton. A fixed composition represents
atoms that occur once rather than scaling with mass.

```python
from spxtacular import IsotopeModel

model = IsotopeModel(
    atoms_per_da={"C": 0.05, "H": 0.08, "N": 0.01, "O": 0.02},
    fixed_composition={"H": 2, "O": 1},
)
custom = spectrum.deconvolute(isotope_model=model)
```

The general lipid preset spans several lipid classes and is necessarily approximate. Prefer an
exact formula or a class-specific custom model when that information is available.

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

### Deconvolution provenance

The output spectrum records the resolved ionization and isotope models together with every
parameter that affects envelope construction and matching. This includes the abundance threshold,
fold-error gate, allowed gaps, envelope-length limit, and ion-mobility tolerance. Custom
`IsotopeModel` compositions and isotope abundances are stored by value, so native `.npz`, matchms,
and spectrl round-trips preserve the exact model. Schema-v1 provenance remains readable.

---

**`charge_range`:** Cover the full range you expect. A wider range increases runtime linearly. For tryptic peptides `(1, 5)` is typical; for intact proteins `(5, 50)` or wider.

**`tolerance`:** The default 50 ppm is conservative. For high-resolution instruments (Orbitrap, timsTOF) use 5–15 ppm.

**`intensity` mode:** `"total"` is recommended for quantification. `"base"` returns the observed
A+0 intensity and returns zero when the monoisotopic peak was inferred rather than observed.

---

## Charge conventions

| `charge` value | Meaning |
|---|---|
| `> 0` | Assigned isotope cluster with this charge state |
| `-1` | Singleton — no isotope neighbours found at any tested charge |
| `0` | After `.decharge()` — neutral mass, charge state no longer tracked |

Charge values are positive magnitudes. Polarity and carrier mass are recorded
separately in deconvolution provenance.

### Polarity and adducts

```python
negative = spectrum.deconvolute(isotope_model="rna", ionization_model="[M-H]-")
sodium = spectrum.deconvolute(isotope_model="lipid", ionization_model="[M+Na]+")
neutral = sodium.decharge()  # reuses the recorded sodium carrier
```

Built-ins cover `[M+H]+`, `[M-H]-`, `[M+Na]+`, and `[M+NH4]+`.
`IonizationModel` accepts custom signed carrier masses.

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
