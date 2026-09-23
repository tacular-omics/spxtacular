<p align="center">
  <img src="spxtacular_logo.svg" alt="spxtacular logo" width="400"/>
</p>

# spxtacular

spxtacular is a Python library for processing mass spectra in proteomics, metabolomics,
lipidomics, glycomics, and oligonucleotide analysis. A `Spectrum` holds parallel peak arrays, and
chainable methods denoise, filter, centroid, deconvolute, and convert them to neutral masses.

Readers load Bruker timsTOF `.d`, mzML, Thermo `.raw`, MGF, MS2, and MSP files. Fragment
matching and PSM scoring build on [peptacular](https://peptacular.readthedocs.io/), and
interactive plotly figures cover stick, mirror, annotated, mass-error, and sequence-coverage plots.
Spectra convert to and from matchms, spectrum_utils, and URL-safe spectrl tokens.

```bash
pip install spxtacular
```

Python 3.12 or newer. File readers are optional extras, listed in
[Getting started](getting-started.md#install).

## Where next

- [Getting started](getting-started.md): install, a first processing pipeline, reading a file
- Guides: [Spectrum](spectrum.md), [Deconvolution](deconvolution.md), [Readers](readers.md),
  [Matching & scoring](scoring.md), [Visualization](visualization.md)
- [API reference](api.md): every public name
- [Citation](citation.md): how to cite spxtacular

## Related packages

The tacular-omics mass spectrometry stack:

- [tdfpy](https://tacular-omics.github.io/tdfpy/) reads Bruker timsTOF `.d` data.
- [mzmlpy](https://tacular-omics.github.io/mzmlpy/) reads mzML files.
- **[spxtacular](https://tacular-omics.github.io/spxtacular/)** (this package) processes the spectra from both: centroiding, deconvolution, matching, scoring and plotting.
