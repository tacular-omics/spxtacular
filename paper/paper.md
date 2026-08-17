---
title: 'spxtacular: A Chainable Python Library for Mass Spectrum Processing'
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

Mass spectra record detected ions as peaks, each defined by a mass-to-charge ratio (m/z) and an intensity. Extracting chemical evidence from these peaks may require denoising, isotope deconvolution, charge assignment, neutral-mass calculation, fragment matching, and scoring. Reimplementing these steps in individual analysis scripts can make workflows difficult to reproduce and intermediate results difficult to inspect.

**spxtacular** organizes these operations as methods on a chainable `Spectrum` object. The object keeps m/z and intensity values aligned with optional per-peak charge, ion mobility, and isotopic profile scores. Metadata produced by each operation remains attached to the spectrum during subsequent filtering, plotting, matching, and scoring.

```python
neutral = (spec.denoise(method="mad")
               .deconvolute(charge_range=(1, 4), tolerance=15,
                            tolerance_type="ppm", min_score=0.5)
               .decharge())
```

\autoref{fig:abstract} summarizes how readers, processing methods, interpretation, and outputs operate through the same metadata-bearing spectrum object.

![Overview of the spxtacular workflow. A chainable `Spectrum` pipeline transforms raw centroids through denoising, deconvolution, neutral-mass conversion, fragment matching, and scoring. File readers, accelerated deconvolution, interactive plots, and compact spectrum sharing support the workflow.](figures/graphical_abstract.png){ #fig:abstract }

The core package depends on NumPy [@harris2020numpy], pandas [@mckinney2010pandas], Plotly [@plotly2015], peptacular [@peptacular], and paftacular [@paftacular]. An optional spectrl integration [@spectrl] provides URL-safe spectrum encoding.

# Statement of need

Existing Python tools cover important parts of this workflow. Pyteomics [@levitsky2019pyteomics] emphasizes file access and proteomics utilities, whereas pyOpenMS [@rost2014pyopenms] exposes the broader OpenMS data model and algorithms. During interactive method development, moving among tools can still require repeated data-model conversions and manual preservation of peak-level evidence.

spxtacular addresses this coordination problem by returning the result of deconvolution as another spectrum rather than a file or a set of detached arrays. The result can then be filtered, plotted, matched against fragments, or processed again in the same session. Charge and isotope scores remain attached to each peak as evidence for its assignment.

The intended users are mass spectrometry researchers, method developers, and software authors who need to inspect or transform spectra without committing to a complete analysis pipeline. Built-in isotope models support peptide, glycan, lipid, DNA, and RNA analytes, while custom average-composition models extend the same workflow to other chemical classes. A common interface supports exploratory analysis, reusable processing code, and integration into larger workflows.

# State of the field

Several Python packages offer complementary capabilities. `ms_deisotope` [@msdeisotope] supports averagine-based deisotoping and charge-state deconvolution, while matchms [@huber2020matchms] supports spectrum similarity and composable processing. spectrum_utils [@bittremieux2023spectrumutils] focuses on annotation and plotting, and pyOpenMS offers a broader collection of mass spectrometry algorithms.

spxtacular focuses on coordinating these tasks through a single spectrum model that retains reader metadata, deconvolution evidence, fragment assignments, scores, and visualization inputs. Adding this workflow to any one of the existing packages would have tied it to that package's data model and scope. Instead, adapters connect spxtacular to matchms and spectrum_utils while preserving its peak-level metadata and processing state.

# Software design

The spectrum-processing code originated in peptacular. It was separated into spxtacular so that spectrum input and output, processing, visualization, and interoperability could evolve independently while peptacular remained focused on peptide chemistry and fragment generation.

The central design choice is a `Spectrum` object that keeps m/z and intensity arrays aligned with optional per-peak metadata. Its methods preserve compatible metadata and return a distinct spectrum by default. This behavior improves predictability and reproducibility at the cost of additional memory. Users can select an explicit in-place option when memory use is a concern. A `SpectrumType` value records whether the spectrum contains profile, centroid, or deconvoluted data and prevents invalid operation sequences such as neutral-mass conversion before deconvolution.

Visualizations are generated directly from the spectrum and its per-peak metadata. A `Spectrum` can therefore be displayed as a stick, mirror, or fragment-annotated plot and inspected as an HTML peak table without an intermediate conversion.

## Deconvolution and scoring

Isotope cluster detection follows a greedy strategy related to MS-Deconv [@liu2010msdeconv] and THRASH [@horn2000thrash]. Aggregated nominal isotope distributions are calculated with a BRAIN-style Newton--Girard recurrence [@dittwald2014brain] from an average elemental composition. Built-in models cover peptides, glycans, lipids, DNA, and RNA, and users can define custom compositions and isotope abundances. Envelope length expands adaptively until the predicted tail falls below the selected abundance threshold.

Beginning with the most intense unused peak, the algorithm evaluates every candidate charge state and aligns the seed with the predicted envelope apex or a contiguous near-apex isotope. It then matches outward in both directions using mass-error, abundance-fold, and optional ion-mobility gates. Each candidate is compared with the complete predicted envelope using the Bhattacharyya coefficient, with missing detectable peaks reducing the score. The best charge and alignment consume their matched peaks only after all candidates have been evaluated. This apex-first procedure can infer the monoisotopic m/z when the A+0 peak is not observed. If the best score falls below `min_score`, the seed is retained as a singleton while the other candidate peaks remain available for subsequent searches.

Assigned peaks retain their charge and `iso_score`, allowing users to inspect and filter deconvolution results without discarding the evidence behind individual assignments.

## Readers and interoperability

`Reader` dispatches by path suffix to backends for Bruker timsTOF `.d` directories, mzML files [@martens2011mzml], Thermo `.raw` files, and MGF, MS2, or MSP peak lists. The mzML and Bruker backends depend on mzmlpy [@mzmlpy] and tdfpy [@tdfpy], respectively. Every backend yields the same `MsnSpectrum` type with available acquisition metadata attached. Peak-list support requires no optional dependencies, whereas the vendor backends are optional and Thermo RAW support requires a .NET runtime.

Fragment matching accepts fragments from the companion peptacular library, which supports ProForma notation [@leduc2022proforma] and the current [2.1 specification](https://www.psidev.info/proforma). paftacular labels matched peaks in mzPAF notation. The resulting matches can be evaluated with the hyperscore popularized by X!Tandem [@craig2004tandem], the normalized spectral contrast angle [@toprak2014spectralangle], matched fraction, and mass-error statistics. For collaboration and troubleshooting, spectrl tokens allow a specific spectrum to be shared in an issue or message without attaching a file.

Optional bidirectional adapters connect spxtacular spectra with matchms `Spectrum` and spectrum_utils `MsmsSpectrum` objects. The matchms adapter uses a namespaced payload to preserve the richer metadata available in spxtacular. Because the spectrum_utils representation is narrower, its adapter warns when fields cannot be represented. Dependencies are imported only when an adapter is called.

# Research impact statement

spxtacular powers the public [Spectra web application](https://spectra.tacular.dev/) for interactive spectrum viewing. It also supports laboratory workflows that analyze and visualize output from the Sage search engine [@lazear2023sage]. In ongoing, unpublished research, the library supports a post-translational modification motif ratio quantitation workflow for covalent protein painting data [@son2023cpp]. This work has identified candidate conformational changes associated with transthyretin amyloidosis (ATTR) and was presented at ASMS 2026 [@pankow2026asms]. spxtacular was also demonstrated alongside peptacular and related applications in an ASMS 2026 poster [@garrett2026asms]. Together, these applications span interactive viewing, search-result analysis, and quantitative structural proteomics.

# AI usage disclosure

Anthropic Claude Code with Opus and Fable 5, and OpenAI Codex with Sol, were used for code editing and manuscript review. The authors made the core design decisions and reviewed, edited, and validated all AI-assisted outputs. They retain responsibility for the software and manuscript.

# Author contributions

Patrick T. Garrett designed and implemented the library and drafted the manuscript. John R. Yates III directed the project and revised the manuscript.

# Funding

This work was supported by the National Institutes of Health under grants R01 AG077046 (Analysis of protein interactions in neurodegenerative disease), R01 MH132570 (Brain-wide mapping of neuronal inhibition by novel inverse activity markers), and R01 MH100175 (Proteogenetics of Autism Spectrum Disorders). Additional support came from grants R01 HL165168 (The CFTR Interactome) and U01 AG088679 (Understanding Gene-Environment Interactions in Brain Aging and Alzheimer's Disease (AD) and AD-Related Dementias (ADRD)). The funders had no role in software design, manuscript preparation, or the decision to publish.

# Conflicts of interest

The authors declare no competing financial interests.

# Acknowledgements

The authors acknowledge support from The Scripps Research Institute and thank Claire Delahunty for her review of the manuscript.

# References
