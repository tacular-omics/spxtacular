# Fragment matching, scoring and reporter ions

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
    tolerance_unit="da",         # "da" or "ppm"
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
| `tolerance_unit` | `"da"` | `"da"` or `"ppm"` |
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
matches = match_fragments(spec, fragments, tolerance=10, tolerance_unit="ppm")
for m in matches:
    print(f"  Peak {m.peak_index} ({m.peak_mz:.4f} m/z) matched {m.fragment} (Δ={m.ppm_error:+.1f} ppm)")
```

---

## `score()`

Runs `match_fragments()` internally and returns all scoring metrics as a dict.

```python
from spxtacular import score

result = score(spectrum, fragments, tolerance=10, tolerance_unit="ppm")
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `spectrum` | | `Spectrum` to score against |
| `fragments` | | Iterable of fragment objects from `peptacular` |
| `tolerance` | `0.02` | Matching tolerance |
| `tolerance_unit` | `"da"` | `"da"` or `"ppm"` |
| `peak_selection` | `"closest"` | How to resolve multiple peaks within tolerance — `"closest"`, `"largest"`, or `"all"` |
| `predicted_intensities` | `None` | Optional predicted intensity for every fragment, in the same order. Enables the literature spectral-angle metric |

**Returned metrics:**

| Key | Description |
|---|---|
| `hyperscore` | Base-10 matched-intensity dot product plus log-factorial terms per ion series |
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

The score uses `log10(sum(I_matched)) + sum_s(log10(n_s!))`, with unit theoretical
intensities. Each observed peak contributes once to the intensity sum. Each distinct
`(ion_type, position)` contributes once to its series count, so charge, neutral-loss,
and isotope variants do not inflate the factorial terms. Missing series contribute
`0! = 1`, and one-sided evidence can have a positive score. No matches or a nonpositive
matched-intensity sum returns zero.

This follows the dot-product/factorial structure described in
[OpenMS's HyperScore documentation](https://openms.de/current_doxygen/html/HyperScore_8h_source.html).
The base-10 logarithm, unit theoretical intensities, preprocessing, and duplicate
handling are explicit spxtacular conventions. Scores are not guaranteed numerically
interchangeable with X!Tandem, OpenMS, Comet, or MSFragger outputs.

!!! warning
    Versions through 0.6.0 multiplied the intensity sums of different ion series.
    That was a different heuristic, incorrectly described as identical to X!Tandem.
    Recompute stored scores and retune thresholds when upgrading. Scaling all
    intensities by `s` now shifts a positive-signal score by `log10(s)`, once rather
    than once per ion series. Normalized inputs can produce negative scores.

### Spectral angle

Supply `predicted_intensities` — one value per fragment, in the same order — and you get the
spectral angle of the literature (Toprak et al.; the metric Prosit and Spectronaut report):

```python
result = spx.score(
    spec, fragments,
    tolerance=10, tolerance_unit="ppm",
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
result = score(spec, fragments, tolerance=10, tolerance_unit="ppm")

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

cosine(query, reference, tolerance=20, tolerance_unit="ppm")   # 0-1
entropy_similarity(query, reference, tolerance=0.02)           # 0-1
```

| Function | What it is |
|---|---|
| `cosine` | The standard spectral dot product: sqrt-transformed intensities, unit-normalised, peaks matched one-to-one |
| `modified_cosine` | Cosine that also matches peaks displaced by the precursor mass difference — the GNPS molecular-networking metric |
| `entropy_similarity` | Unweighted entropy similarity with one-to-one greedy peak alignment |

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

`entropy_similarity` uses the unweighted entropy expression. It does not apply
entropy-dependent intensity weights or automatically remove precursor and noise peaks.
Negative intensities are clipped to zero and each spectrum is normalized to a probability
distribution. Preprocess both inputs consistently. The
[MS Entropy reference](https://msentropy.readthedocs.io/en/latest/classical_entropy_similarity.html)
distinguishes weighted and unweighted variants and documents its own cleaning defaults.
The tests include independent values for unambiguous peak alignments. They do not establish
that one metric outperforms another for compound identification.

## Isobaric reporter ions (TMT, TMTpro, iTRAQ)

spxtacular reads the reporter-ion intensities of isobaric-labelled spectra: TMT 0/2/6/10/11,
TMTpro 0/16/18 and iTRAQ 4/8. Channel names and reporter m/z come from tacular's
`ISOBARIC_TAG_LOOKUP`, which computes each reporter m/z from its isotopic composition, so no
reporter mass is typed into spxtacular.

```python
import spxtacular as spx

spec = next(iter(spx.Reader("run.mzML").ms2))
ions = spec.reporter_ions("TMT10")          # or spx.extract_reporter_ions(spec, "TMT10")
ions.channels        # ('126', '127N', '127C', ..., '131N')
ions.intensity       # one value per channel, 0.0 where no peak was found
ions.ppm_error       # observed - theoretical m/z in ppm, NaN where no peak was found
ions["127N"]         # one channel's intensity

table = spx.reporter_ion_table(spx.Reader("run.mzML"), "TMTpro18", include_errors=True)
# spectrum_index  scan_number  native_id  ms_level  rt  126  127N  ...  135N  126_ppm_error ...
```

### Plex names

Pass a name or alias from tacular, case-insensitive: `TMT0`, `TMT2`, `TMT6`, `TMT10`
(`TMT10plex`), `TMT11`, `TMTpro0`, `TMT16` (`TMTpro16`), `TMT18` (`TMTpro18`), `iTRAQ4`,
`iTRAQ8`, or a `tacular.IsobaricTagInfo`. `ReporterIons.plex` is tacular's name for it
(`"TMT18"` for `"TMTpro18"`). An unknown name raises `SpxtacularError` listing the valid
ones.

### How peaks are picked

| Choice | Behaviour |
|---|---|
| Tolerance | `tolerance=20.0, tolerance_unit="ppm"` by default (`"da"` also accepted). The closest channels, the TMT/TMTpro N/C pairs, are 6.32 mDa (about 47-50 ppm) apart, so +/-20 ppm windows never overlap. A tolerance that makes two windows overlap raises `SpxtacularError`. |
| Several peaks in a window | The most intense peak is used (ties: lowest m/z). |
| No peak in a window | Intensity `0.0`; `observed_mz` and the error are `NaN`; `found` is `False`. Zero keeps sums, ratios and the impurity correction well defined, and NaN in the error columns keeps "not found" visible. |
| Tolerance limit | The overlap error names the largest tolerance that fits, e.g. "Use a tolerance below 24.28 ppm" for TMT10. |
| Input | Centroided spectra in m/z; the m/z array need not be sorted. A decharged spectrum (neutral masses) or NaN/inf intensities raise `SpxtacularError`. |

### Many spectra

`reporter_ion_table(spectra, plex, ...)` takes any iterable of spectra (a list, a generator,
`reader.ms2`). Any object with an `ms2` attribute (every spxtacular reader) is read through
`.ms2`, even if it is iterable itself, so a reader gives its MS2 spectra. It returns one row per spectrum:
`spectrum_index` (position in the input), `scan_number`, `native_id`, `ms_level`, `rt`
(`None` for a plain `Spectrum`), one float column per channel, and with
`include_errors=True` a `<channel>_ppm_error` column per channel. `ms_level=` keeps only
spectra of that level; for SPS-MS3 data pass an iterable of the MS3 spectra (e.g.
`iter(reader)`) with `ms_level=3`.

### Isotope impurity correction

Each TMT/iTRAQ reagent lot ships with a sheet of isotope impurities: for every channel, the
percentage of its reporter signal that appears 2 or 1 Da below and 1 or 2 Da above. Pass the
sheet as `impurities=` and the intensities are corrected:

```python
lot = {
    "126":  {"-2": 0.0, "-1": 0.0, "+1": 7.0, "+2": 0.2},
    "127N": {"-2": 0.0, "-1": 0.4, "+1": 6.5, "+2": 0.0},
    # ... one row per channel; channels left out are taken as pure
}
ions = spec.reporter_ions("TMT10", impurities=lot)
ions.raw_intensity   # what was measured
ions.intensity       # corrected

table = spx.reporter_ion_table(reader, "TMT10", impurities=lot, normalize="sum")
```

A `pandas.DataFrame` with channels as the index and shifts as columns works as well. Empty
cells (`None`, NaN) are 0. Column labels are:

- **Numbers** (`-2`, `-1`, `+1`, `+2`): always that many **13C**, never 15N.
- **Explicit substitutions**: `"-13C"`, `"+2x13C"`, `"-15N"`, `"+13C+15N"`, `"-18O"`.
- **A main-peak column** (`0`, `"0"`, `"main"`, `"reporter"`, `"monoisotopic"`, any case):
  ignored, because the main peak is always `100 - sum(impurities)`.

TMT10/11 and TMTpro N channels carry a 15N, and a reagent missing it loses 0.997 Da, not
the 1.003 Da of a 13C. The two land 6.3 mDa apart, on different channels. If your sheet lists
15N and 13C impurities of N channels separately (newer Thermo TMTpro sheets do), label the
15N columns `"-15N"`/`"+15N"`; a numeric `-1` there is read as 13C and placed on the wrong
m/z.

`isotope_correction_matrix(plex, lot)` returns the matrix `M` that is solved,
`observed = M @ true`. `M[i, j]` is the fraction of reagent `j`'s signal seen in channel `i`:

- A reagent keeps `100 - sum(its impurities)` percent in its own channel.
- Each impurity goes to the channel nearest the shifted reporter m/z, if it is within
  `min(0.02 Da, half the plex's smallest channel spacing)`, as in OpenMS. That is 0.02 Da
  for TMT6 and iTRAQ, whose channels are about 1 Da apart, and about 3.2 mDa for TMT10/11
  and TMTpro, whose N/C pairs are 6.32 mDa apart. So the -1 of TMT `128C` goes to `127C`
  and of `128N` to `127N`, since they differ by one 13C.
- An impurity that lands on no channel of the plex is lost signal and only lowers the
  diagonal. In TMT10 the -1 (13C) of `127N` lands 6.3 mDa below `126` and the +1 of `130C`
  lands on the `131C` position, 6.3 mDa above `131N`: neither is inside a picking window,
  so neither is counted. In TMT11 the +1 of `130C` goes to `131C`.

You can also pass a precomputed square matrix as `impurities=`, or call
`correct_isotope_impurities(intensities, matrix_or_lot, plex=...)` (`plex` is keyword-only) on your own arrays (one
spectrum or a 2-D array of many).

**Non-negativity.** Each spectrum is first solved exactly. With noise, the exact solution can
go slightly negative in a weak channel next to a strong one. Those spectra are re-solved by
non-negative least squares (a small Lawson-Hanson NNLS in numpy, since scipy is not a
dependency). That gives the exact answer whenever it is already non-negative. Otherwise it
gives the closest non-negative fit, which, unlike clipping negatives to zero, passes the
remainder on to the neighbouring channels.

### Normalization

`normalize="sum"` divides each spectrum's channels by their sum and `normalize="max"` by the
largest channel, after correction. A spectrum with no reporter signal stays all zero.
