---
title: 'spxtacular: A chainable Python library for mass spectrum processing'
tags:
  - 'Python'
  - 'mass spectrometry'
  - 'deconvolution'
  - 'visualization'
  - 'bioinformatics'
  - 'proteomics'
  - 'metabolomics'
authors:
  - name:
      given-names: 'Patrick T.'
      surname: 'Garrett'
    orcid: 0000-0002-8434-9693
    email: pgarrett@scripps.edu
    affiliation: 1
  - name:
      given-names: 'John R.'
      surname: 'Yates'
      suffix: 'III'
    orcid: 0000-0001-5267-1672
    email: jyates@scripps.edu
    corresponding: true
    affiliation: 1
affiliations:
  - name: 'The Scripps Research Institute, La Jolla, CA 92037, USA'
    index: 1
date: '15 August 2026'
bibliography: paper.bib
---

# Summary

Mass spectra record detected ions as peaks defined by a mass-to-charge ratio (m/z) and an intensity. Turning these peaks into chemical evidence can require noise removal, isotope deconvolution, charge assignment, neutral-mass calculation, fragment matching, and scoring. Reimplementing these steps in separate scripts can make workflows difficult to reproduce and intermediate results difficult to inspect.

**spxtacular** organizes this workflow around a `Spectrum` object whose processing methods return another `Spectrum`, allowing operations to be called in sequence. The object keeps charge assignments, ion-mobility values, isotope scores, and other peak-level metadata aligned with their peaks. Researchers can inspect each stage without managing detached arrays or intermediate files.

![Overview of the spxtacular workflow. A chainable `Spectrum` pipeline transforms raw centroids through denoising, deconvolution, neutral-mass conversion, fragment matching, and scoring. File readers, Numba-accelerated deconvolution, interactive plots, and compact spectrum sharing support the workflow.](figures/graphical_abstract.png){ #fig:abstract }

# Statement of need

During interactive method development, moving among specialized tools can require repeated data-model conversions and manual preservation of charge assignments, isotope scores, and other information associated with individual peaks. spxtacular addresses this coordination problem by representing deconvolution output as another spectrum that can continue through the same workflow. The intended users are mass spectrometry researchers, method developers, and software authors who need to inspect or transform spectra without adopting a complete analysis pipeline. The common interface supports exploratory analysis, reusable processing code, and integration into larger research workflows.

# State of the field

Several Python packages provide related capabilities. Pyteomics [@levitsky2019pyteomics] emphasizes file access and proteomics utilities, while pyOpenMS [@rost2014pyopenms] provides a broader data model and collection of mass spectrometry algorithms. `ms_deisotope` [@msdeisotope] supports averagine-based deisotoping and charge-state deconvolution, matchms [@huber2020matchms] supports composable spectrum processing and similarity calculations, and spectrum_utils [@bittremieux2023spectrumutils] focuses on spectrum annotation and plotting.

spxtacular differs by storing peak-level metadata and spectrum type directly in its data model. As a separate library, spxtacular keeps this model independent of the scope of any single existing package.

# Software design

spxtacular grew out of spectrum-processing functions originally implemented in `peptacular` [@peptacular], a Python package for peptide-sequence analysis and fragment generation. Moving those functions into a dedicated library allows spectrum file handling, processing, visualization, and interoperability to develop independently while `peptacular` remains focused on peptide chemistry.

The core data structure in spxtacular is the `Spectrum` object. It stores m/z values, intensity values, and optional per-peak metadata in parallel arrays, so entries at the same position describe the same peak. Processing methods retain per-peak metadata that remain aligned with the result and return a distinct spectrum by default. Returning new objects improves predictability and reproducibility but requires memory for both the input and result. Users can request in-place processing when memory use is a concern. A `SpectrumType` value records whether data are profile, centroid, or deconvoluted and rejects invalid operation sequences, such as neutral-mass conversion before deconvolution.

Visualizations are generated directly from the spectrum and its peak-level metadata. The object can produce stick, mirror, and fragment-annotated plots along with a peak table. spxtacular builds on NumPy [@harris2020numpy], pandas [@mckinney2010pandas], and Plotly [@plotly2015].

## Deconvolution and scoring

Isotope-cluster detection follows a greedy strategy related to MS-Deconv [@liu2010msdeconv] and THRASH [@horn2000thrash]. The software calculates aggregated nominal isotope distributions from an analyte-specific average elemental composition with a BRAIN-style Newton-Girard recurrence [@dittwald2014brain]. Built-in models cover peptides, glycans, lipids, DNA, and RNA, and users can define custom compositions and isotope abundances. Isotopes are added to the predicted envelope until the relative abundance of the next isotope falls below a user-selected threshold.

The most intense unused peak becomes the seed for a possible isotope cluster. For each candidate charge state, the algorithm aligns the seed with the predicted envelope apex or a neighboring isotope of similar predicted abundance. It then matches peaks in both directions using mass-error limits, relative-abundance limits, and optional ion-mobility limits. Each candidate is scored against the complete predicted envelope using the Bhattacharyya coefficient. Missing predicted peaks and observed peaks whose intensities exceed the prediction lower the score, while missing isotopes above a user-selected detection threshold receive an additional penalty. Matched peaks are assigned to the winning cluster only after all candidates have been evaluated. This apex-first procedure can infer the monoisotopic m/z when the A+0 peak is absent. If the best score is below `min_score`, the seed is retained as an unassigned peak (`charge=-1`, `iso_score=0.0`), while peaks tested as possible cluster members remain available as seeds for later searches.

## Readers and interoperability

`Reader` examines the suffix of an input path and selects the matching format-specific parser. It supports Bruker timsTOF `.d` data directories, mzML files [@martens2011mzml], Thermo `.raw` files, and MGF, MS2, or MSP peak-list files. Every parser returns the same `MsnSpectrum` type with available acquisition metadata, such as scan number, MS level, retention time, and precursor information. The peak-list parsers have no extra dependencies. The mzML and Bruker parsers use mzmlpy [@mzmlpy] and tdfpy [@tdfpy], respectively, while Thermo RAW support requires a .NET runtime.

Spectrum interpretation follows community standards from the HUPO Proteomics Standards Initiative. Fragment matching uses fragments that the companion `peptacular` package generates from ProForma peptidoform notation [@leduc2022proforma], including the [2.1 specification](https://www.psidev.info/proforma). The companion `paftacular` package [@paftacular] labels matched peaks in the mzPAF peak annotation format [@klein2024mzpaf]. Together with mzML input, these standards keep spxtacular inputs and annotations portable across other tools that support them. Matches can be evaluated with a base-10 hyperscore following the dot-product and factorial structure of X!Tandem [@craig2004tandem], with unit theoretical intensities and deduplicated peak contributions, the normalized spectral contrast angle [@toprak2014spectralangle], matched fraction, and mass-error statistics. The optional `spectrl` package [@spectrl] encodes spectra as URL-safe tokens for sharing a specific spectrum without attaching a file.

Bidirectional adapters connect spxtacular spectra with matchms `Spectrum` and spectrum_utils `MsmsSpectrum` objects. The matchms adapter stores spxtacular-specific fields under a dedicated metadata key so round trips preserve them. Because spectrum_utils cannot represent per-peak charge, ion mobility, isotope scores, or all acquisition metadata, its adapter warns when populated fields would be dropped. Adapter dependencies load only when used.

# Research impact statement

spxtacular provides the spectrum processing and visualization used by the public [Spectra web application](https://spectra.tacular.dev/), an interactive spectrum viewer. In ongoing unpublished work, it also supports a post-translational modification motif-ratio quantitation workflow for covalent protein painting data [@son2023cpp] in a study of transthyretin amyloidosis (ATTR). Preliminary results from that study have been presented at ASMS 2026 [@pankow2026asms]. spxtacular has also been demonstrated with `peptacular` in an ASMS 2026 poster [@garrett2026asms].

# AI usage disclosure

Anthropic Claude Code, using Opus and Fable 5, and OpenAI Codex, using Sol, assisted with code editing and manuscript review. The authors made the core design decisions and reviewed, edited, and validated all AI-assisted output. They retain responsibility for the software and manuscript.

# Author contributions

Patrick T. Garrett designed and implemented the library and drafted the manuscript. John R. Yates III directed the project and revised the manuscript.

# Conflicts of interest

The authors declare no competing financial interests.

# Acknowledgements

This work was supported by the National Institutes of Health under grants R01 AG077046, R01 MH132570, R01 MH100175, R01 HL165168, and U01 AG088679. The funders had no role in software design, manuscript preparation, or the decision to publish. The authors also acknowledge support from The Scripps Research Institute and thank Claire Delahunty for reviewing the manuscript.

# References
