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
The absolute mass tolerance is 0.0001 Da, allowing the deconvolution algorithm's
nominal-envelope anchoring approximation.

The existing Thermo RAW fixture remains the measured-data reader integration check.
It has not been relabeled as a ground-truth deconvolution benchmark. Any future experimental
benchmark needs documented independent assignments, source identifiers, redistribution
permission, and acquisition conditions before it can establish scientific accuracy.
