These are analytical fixtures, not experimental spectra or identification benchmarks.

`carbon_envelopes.json` freezes nominal isotope intensities for carbon-only compositions.
For `n` carbon atoms, intensity at nominal offset `k` is
`1e6 * binomial(n, k) * 0.0107**k * 0.9893**(n-k)`.
The fixture was generated using Python's `math.comb`, without importing spxtacular,
peptacular, or the BRAIN implementation. The isotope probabilities match the declared
natural-abundance convention being tested. Masses use 12 Da per light carbon atom,
1.00335483507 Da per heavy-carbon substitution, and the carrier masses recorded below.

- Proton attachment: 1.00727646688 Da.
- Proton loss: -1.00727646688 Da.
- Sodium attachment: 22.98922070209 Da.
- Charge magnitudes: 1, 2, or 5. The search tests magnitudes 1 through 6.
- Intensities below 1% of the apex are omitted in the missing-mono case.
- Complete envelopes retain the monoisotopic peak through the last isotope above 1%.
- The overlapping case superposes two envelopes with different mobilities and abundance.
- Isolated noise peaks have explicit expected singleton counts.

These cases check independent numerical expectations for charge, mass, false assignments,
missing monoisotopic recovery, mobility gating, unsorted inputs, and intensity conservation.
They do not validate the average-composition models against real samples.
The absolute mass tolerance is 1e-6 Da. It was 1e-4 Da while deconvolution spaced
isotopes by peptacular's rounded `C13_NEUTRON_MASS` (1.003350), which put masses
decharged from an A+n apex off by n * 4.8e-6 Da.

`independent_references.json` is written by `generate_independent_references.py`,
which imports pyteomics 5.0.1, scipy 1.18.1 and numpy but not spxtacular or peptacular.
Its header lists the sources and the command to regenerate it. It holds:

- isotope envelopes from pyteomics isotopologue enumeration (small molecules) and from
  direct polynomial convolution of IUPAC abundances (peptides, a lipid, a ubiquitin-size
  formula, a sulfur-rich formula, C2000); the two methods agree to 1e-13;
- Senko (1995) averagine envelopes from 500 to 20000 Da and exact envelopes of five
  tryptic-like peptides, with pyteomics monoisotopic masses;
- ion m/z for protonated, deprotonated, sodiated and ammoniated ions from CODATA 2018
  proton and electron masses and AME2020 atomic masses;
- pyteomics precursor m/z and b/y fragment m/z for PEPTIDE and SAMPLER;
- binomial log survival values from `scipy.stats.binom.logsf`.

`test_independent_references.py` checks spxtacular against it. Masses agree to 1e-6 Da
or better. Two expected differences are asserted rather than hidden: the peptide isotope
model uses its own atom rates, so it is close to Senko averagine (cosine > 0.9999) but not
equal, and pyteomics uses a less precise sulfur mass, so SAMPLER fragments differ from
peptacular's by about 1.6e-7 Da.

The existing Thermo RAW fixture remains the measured-data reader integration check.
It has not been relabeled as a ground-truth deconvolution benchmark. Any future experimental
benchmark needs documented independent assignments, source identifiers, redistribution
permission, and acquisition conditions before it can establish scientific accuracy.
