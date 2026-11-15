# PTM site localization

A search engine often knows that a peptide carries a modification without knowing which
residue carries it. `localize` answers that question the way
[Ascore](https://doi.org/10.1038/nbt1240) does: it takes the candidate placements the
ProForma string allows, asks how much of each candidate's fragment evidence the spectrum
actually contains, and reports how much better the best placement is than the next one.

## One call

Write the ambiguity into the peptide and pass the spectrum:

```python
import numpy as np
import peptacular as pt
from spxtacular import Spectrum, localize

# A spectrum of the b and y ions of PEPTIDESK phosphorylated on T4.
fragments = pt.fragment("PEPT[Phospho]IDESK", ion_types=("b", "y"), charges=(1,))
spectrum = Spectrum(
    mz=np.array(sorted(f.mz for f in fragments)),
    intensity=np.full(len(fragments), 1000.0),
)

result = localize(spectrum, "PEPT[Phospho#g1]IDES[#g1]K", tolerance=0.05)
print(result.best)                 # PEPT[Phospho#g1]IDESK
print(round(result.ascore, 2))     # 215.92
print(result.is_ambiguous())       # False
```

`result.ascore` is in the paper's units, `-10 log10 P`. The paper's cut for a confident
site is 19, which is `ASCORE_CONFIDENT_CUT` and the default of `is_ambiguous()`; pass your
own cut if your field uses another.

## Saying which sites are candidates

The candidates come from the ProForma string alone, through peptacular's
`localization_isomers`, so the notation decides what is asked:

```python
import peptacular as pt

# A labelled group: exactly these two residues.
print([a.serialize() for a in pt.localization_isomers("PEPT[Phospho#g1]IDES[#g1]K")])
# A range: every residue inside it.
print([a.serialize() for a in pt.localization_isomers("PEP(ST)[Phospho]IDE")])
# An unknown position: any residue at all, whatever its letter.
print([a.serialize() for a in pt.localization_isomers("[Phospho]?PEST")])
```

A peptide with no ambiguity has one candidate: `localize` still returns, with `runner_up`
and `ascore` both `None`, so a loop over a result file needs no special case. Localization
scores already written into the string (`#g1(0.8)`) name the candidate residues and are
never read as evidence; the numbers that come back are this function's own.

## What comes back

```python
import numpy as np
import peptacular as pt
from spxtacular import Spectrum, localize

fragments = pt.fragment("PEPT[Phospho]IDESK", ion_types=("b", "y"), charges=(1,))
spectrum = Spectrum(
    mz=np.array(sorted(f.mz for f in fragments)),
    intensity=np.full(len(fragments), 1000.0),
)
result = localize(spectrum, "PEPT[Phospho#g1]IDES[#g1]K", tolerance=0.05)

print(result.isomers)           # every candidate, in peptacular's fixed order
print(result.peptide_scores)    # the weighted peptide score of each
print(result.peak_depth)        # the depth the AScore was computed at
print(result.site_determining)  # (matched, available) per candidate
print(result.provenance.to_dict()["method"])
```

`site_determining` counts, at `peak_depth`, the ions of each candidate that the best other
candidate cannot explain, and how many of them the spectrum contains. When the top two
candidates have no such ions, or none of them matched, `ascore` is `0.0` rather than an
error: a precursor-only annotation, or two placements inside one uncleaved stretch, carry
no site information and the result says so.

## How the score is built

1. For each peak depth 1 to 10, keep the most intense *depth* peaks per 100 Da window
   (`Spectrum.filter(top_n_per_window=...)`).
2. Match each candidate's ions and score the count with the cumulative binomial,
   `-10 log10 P(X >= k)` for `X ~ Binomial(n, p)`, where *n* is the candidate's ions, *k*
   how many matched, and `p = depth * 2 * tolerance_in_da / window`.
3. Rank candidates by the weighted mean of those ten scores, over the paper's peak-depth
   weights (0.5, 0.75, 1, 1, 1, 1, 0.75, 0.5, 0.25, 0.25).
4. Take the depth that separates the top two candidates best and recompute both scores
   from their site-determining ions alone. The AScore is the difference.

The paper writes `p = i / 100`, which is step 2 for a 1 Da window on the low-resolution
data it was developed on. Spelling the tolerance out keeps the statistic honest on
high-resolution data, and it is what the published Python implementation
([pyAscore](https://doi.org/10.1021/acs.jproteome.1c00766)) computes: on the example
above, pyAscore 1.0.1 returns the same winning residue, the same peptide score
(364.79053) and the same AScore (215.9176). `tests/test_localization.py` pins that
agreement and skips when pyascore is not installed.

Full-length ions are dropped: a 9-mer has b1 to b8 and y1 to y8, and b9 and y9 are the
whole peptide, the same mass for every placement.

## What it is not

`localize` scores one PSM against one spectrum. It does not infer protein-level sites or a
false-localization rate, does not rescore or re-search, and does not choose the
modification: the ProForma string says what is ambiguous. Neutral (decharged) spectra are
refused, because the peak-depth window and the random-match probability are defined in m/z.
