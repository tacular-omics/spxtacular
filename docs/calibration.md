# m/z recalibration

An instrument drifts. When a spectrum is identified, its matched fragments are the only
peaks whose true m/z is known, so they are what the drift can be measured against.
`recalibrate` fits that measurement and applies it to **every** peak in the spectrum,
matched or not.

This is a per-spectrum correction from one PSM's matches. It is not a run-wide or
retention-time-dependent calibration model, and it does not touch the precursor.

## One call

```python
import numpy as np
import peptacular as pt
from spxtacular import Spectrum, match_fragments, recalibrate

# A spectrum of the b and y ions of PEPTIDESK, 4 ppm high across the whole range.
fragments = pt.fragment("PEPTIDESK", ion_types=("b", "y"), charges=(1,))
true_mz = np.array(sorted(f.mz for f in fragments))
spectrum = Spectrum(mz=true_mz * (1 + 4.0e-6), intensity=np.full(len(true_mz), 1000.0))

matches = match_fragments(spectrum, fragments, tolerance=0.02)
fixed = recalibrate(spectrum, matches)

print(round(fixed.calibration.offset_ppm, 3))      # 4.0
print(np.allclose(fixed.mz, true_mz, rtol=1e-9))   # True
```

A new spectrum comes back, as with every other transform; pass `inplace=True` to correct
the one you have. `intensity`, `charge`, `im` and `iso_score` are carried over unchanged
and share no array with the input.

## What is fitted, and what it is fitted on

Each `MatchedFragment` already carries the signed error of its peak against the
theoretical m/z, in ppm. Those errors are fitted against observed m/z:

| `model` | Fitted | Correction applied |
|---|---|---|
| `"offset"` (default) | one number, `ppm = a` | `mz / (1 + a * 1e-6)` |
| `"linear"` | `ppm = a + b * mz` | the same, with `a + b * mz` in place of `a` |

`estimator` chooses how. `"median"` takes the median signed error, which a minority of
mis-assigned matches cannot move; `"least_squares"` takes the mean. The default is the
median for a measured reason — on the real PSM behind the package's own figure
(`mzspec:PXD000561:Adult_Frontalcortex_bRP_Elite_85_f09:scan:17555`, VLHPLEGAVVIIFK, 22
matched b/y ions from 147 to 1242 m/z) the median signed error is +0.790 ppm and the mean
is +0.908, a 0.12 ppm difference created by a single +5.46 ppm outlier out of 22. The
median offset takes the mean absolute error from 1.19 ppm to 0.84 ppm and leaves
`matched_fraction` unchanged.

The linear model is opt-in, because the slope spends a degree of freedom that one outlier
can pull: on the same PSM it tightens the residual spread from 1.40 to 1.29 ppm while
leaving a larger mean absolute error. It is for an instrument whose error genuinely fans
out with m/z, which is the shape `mass_error_plot` is there to show you. It is fitted by
least squares, so it requires `estimator="least_squares"` rather than silently changing
the estimator you asked for.

One pass only. There is no refit-and-rematch loop, which is where a recalibration starts
agreeing with its own errors.

## When it refuses

`recalibrate` raises `SpxtacularError` rather than returning a spectrum it cannot stand
behind. Never a silent no-op, and never a half-fit:

1. Fewer than `min_matches` matched fragments (8). With fewer points the answer is
   whichever ions happened to match.
2. A matched m/z span below `min_mz_span` (200 Da), for `model="linear"` only. A slope
   fitted over a narrow span is extrapolation across the rest of the spectrum.
3. A correction above `max_abs_ppm` (50) anywhere in the spectrum's m/z range. That is a
   wrong peptide or the wrong tolerance unit, not a calibration drift.
4. Fewer than `min_matches` **distinct** peaks: two fragments matching one peak are one
   observation, not two.
5. A decharged spectrum. A neutral mass is not an m/z, so recalibrate before
   `decharge()`, not after.
6. A non-finite fitted coefficient.

```python
import numpy as np
import peptacular as pt
from spxtacular import Spectrum, match_fragments, recalibrate
from spxtacular.errors import SpxtacularError

fragments = pt.fragment("PEPTIDESK", ion_types=("b", "y"), charges=(1,))
true_mz = np.array(sorted(f.mz for f in fragments))
spectrum = Spectrum(mz=true_mz * (1 + 4.0e-6), intensity=np.full(len(true_mz), 1000.0))
matches = match_fragments(spectrum, fragments, tolerance=0.02)

try:
    recalibrate(spectrum, matches[:4])
except SpxtacularError as error:
    print(error)  # names min_matches and what it got
```

Each message names the quantity and the threshold, so a caller can decide whether to
lower the bar deliberately or leave the spectrum alone.

## What it records

The returned spectrum carries a `CalibrationProvenance` on `calibration`, beside the
`deconvolution` record:

```python
import numpy as np
import peptacular as pt
from spxtacular import Spectrum, match_fragments, recalibrate

fragments = pt.fragment("PEPTIDESK", ion_types=("b", "y"), charges=(1,))
true_mz = np.array(sorted(f.mz for f in fragments))
spectrum = Spectrum(mz=true_mz * (1 + 4.0e-6), intensity=np.full(len(true_mz), 1000.0))
matches = match_fragments(spectrum, fragments, tolerance=0.02)

record = recalibrate(spectrum, matches).calibration
print(record.model, record.estimator)                      # offset median
print(record.n_matches, record.n_unique_peaks)             # 18 18
print(round(record.residual_ppm_sd, 6))                    # 0.0
print(round(record.correction_ppm(1000.0), 3))             # 4.0
```

`residual_ppm_sd` and `residual_ppm_mad` are the spread the correction left behind, which
is how a caller sees a fit that explained nothing: a median offset that leaves a 20 ppm
residual spread has found a wrong peptide, not a drift.

The record survives `to_dict()`, `to_json()`, `save()` and `load()`, and is part of the
packaged JSON Schema (spectrum schema version 3; versions 1 and 2 predate it and read back
with no calibration). Peak-list writers drop it, as they drop the other provenance.

Recalibrating twice keeps both corrections:

```python
import numpy as np
import peptacular as pt
from spxtacular import Spectrum, match_fragments, recalibrate

fragments = pt.fragment("PEPTIDESK", ion_types=("b", "y"), charges=(1,))
true_mz = np.array(sorted(f.mz for f in fragments))
spectrum = Spectrum(mz=true_mz * (1 + 4.0e-6), intensity=np.full(len(true_mz), 1000.0))
matches = match_fragments(spectrum, fragments, tolerance=0.02)

once = recalibrate(spectrum, matches)
again = recalibrate(once, match_fragments(once, fragments, tolerance=0.02))

print(round(again.calibration.previous.offset_ppm, 3))  # 4.0
print(round(again.calibration.offset_ppm, 6))           # 0.0, nothing left to correct
```

## Where it fits in a pipeline

Match, recalibrate, then re-match or re-score on the corrected axis:

```python
import numpy as np
import peptacular as pt
from spxtacular import Spectrum, match_fragments, recalibrate, score

fragments = pt.fragment("PEPTIDESK", ion_types=("b", "y"), charges=(1,))
true_mz = np.array(sorted(f.mz for f in fragments))
spectrum = Spectrum(mz=true_mz * (1 + 4.0e-6), intensity=np.full(len(true_mz), 1000.0))

matches = match_fragments(spectrum, fragments, tolerance=0.02)
fixed = recalibrate(spectrum, matches)

before = score(spectrum, fragments, tolerance=0.02)
after = score(fixed, fragments, tolerance=0.02)
print(round(before["mean_ppm_error"], 3), "->", round(after["mean_ppm_error"], 3))
```

A tighter mean error on the same matches is the point; a `matched_fraction` that *falls*
after recalibration means the fit moved the axis the wrong way, and is worth looking at
rather than ignoring.
