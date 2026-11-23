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
date: '24 September 2026'
bibliography: paper.bib
---

# Summary

A mass spectrometer reports a spectrum as a list of peaks, each with a mass-to-charge ratio (m/z) and an intensity. Working out what produced those peaks takes a sequence of steps, among them removing noise, grouping isotopes, assigning charge, matching fragments and scoring, and every intermediate result is itself a spectrum. **spxtacular** is a Python library built around that: a `Spectrum` object holds the peaks, every processing method returns another `Spectrum`, and what a step produces, such as charge assignments, isotope scores and ion mobilities, stays aligned with the peaks it describes. A chain can therefore be stopped at any point, inspected, plotted or written out, and continued. The same object reads Bruker, Thermo, mzML and peak-list data, carries it through processing and scoring, quantifies isobaric reporter ions, and draws figures for a screen or a page.

![The spxtacular workflow on one public spectrum (ProteomeXchange dataset PXD000561, scan 17555, peptide VLHPLEGAVVIIFK). Each panel is headed by the method call that produced it: the spectrum as fetched, after denoising, after deconvolution, after conversion to neutral masses and merging of the ions seen at two charge states, and after scoring against the b and y fragment ions from `peptacular`. The code block is the complete script.](figures/graphical_abstract.png){ #fig:abstract }

# Statement of need

Most mass spectrometry data are analysed by complete pipelines that take raw files and return identifications and quantities. The steps in between, such as deconvolution, fragment matching and scoring, decide whether a method works, yet their intermediate results are rarely seen. A researcher checking a doubtful identification, or a developer testing a new deconvolution setting, needs to look at one spectrum after each step and see what changed.

In Python this is harder than it should be. The steps are spread across libraries with different spectrum representations, so a workflow converts at every boundary. What each step learns about individual peaks, such as charge, isotope score and ion mobility, is carried by hand. Nothing checks that those results still describe the same peaks after filtering, or records which steps were applied, so a step run out of order yields a number rather than an error. Existing packages each cover part of this chain, and none carries per-peak results and processing state on one spectrum object through all of it.

spxtacular was written to fill that gap. The peaks, the per-peak results of every step and the processing state are one object, and every method returns another spectrum, so a workflow can be stopped after any step and the result inspected, plotted or passed to another library. The intended users are researchers who examine individual spectra, method developers who need to see the effect of each processing step, and software authors who need one stage of processing without a complete analysis pipeline.

# State of the field

Five Python packages overlap with spxtacular's scope. Pyteomics [@levitsky2019pyteomics] emphasizes file access and proteomics utilities, while pyOpenMS [@rost2014pyopenms] provides a broader data model and collection of mass spectrometry algorithms. `ms_deisotope` [@msdeisotope] supports averagine-based deisotoping and charge-state deconvolution, matchms [@huber2020matchms] supports composable spectrum processing and similarity calculations, and spectrum_utils [@bittremieux2023spectrumutils] focuses on spectrum annotation and plotting. spxtacular differs by storing peak-level metadata and spectrum type directly in its data model.

spxtacular began as spectrum-processing functions in `peptacular` [@peptacular], split out to keep that package focused on peptide chemistry. Building a separate package rather than contributing to pyOpenMS or matchms was a choice about data models, not features. pyOpenMS exposes a C++ library through generated Python bindings, so carrying an open set of aligned per-peak fields, such as charge, isotope score, and ion mobility, would mean changing C++ classes and the binding layer. matchms is built for spectral similarity networking in metabolomics, with spectrum-level metadata and no place for peak-level metadata or for a state recording whether data are profile, centroid, or deconvoluted.

spxtacular is designed for use alongside these packages. A matchms user can read vendor data, centroid and deconvolute it in spxtacular, then pass the spectra on with `to_matchms`, which preserves per-peak charge, isotope score, and ion mobility under a dedicated metadata key. A spectrum_utils user can convert in the other direction, with a warning for fields its model does not carry. Each conversion is one call, so spxtacular can be adopted for one stage without replacing an existing pipeline.

# Software design

The core data structure is the `Spectrum` object. It stores m/z values, intensity values, and optional peak-level metadata in parallel arrays, so entries at the same position describe the same peak. Processing methods return a new spectrum by default, which leaves the input unchanged for a later step or a re-run but requires memory for both input and result, so in-place processing can be requested. A `SpectrumType` value records whether data are profile, centroid, or deconvoluted and rejects invalid operation sequences, such as neutral-mass conversion before deconvolution.

Parallel arrays, rather than one object per peak, keep processing vectorized and interoperable with NumPy and pandas, but alignment of the metadata arrays is the library's responsibility, not a property of the type system. A single `Spectrum` class with a state value is simpler to chain than separate classes, but an invalid sequence fails at run time. Describing a figure once and letting either backend draw it keeps the layout of interactive and print output the same, but limits the description to what both backends support. Greedy apex-first deconvolution, rather than a global fit over all charge-state hypotheses, is fast and traces each assignment to one seed and score, but resolves a crowded region locally rather than optimally.

spxtacular builds on NumPy [@harris2020numpy], pandas [@mckinney2010pandas], and Plotly [@plotly2015]. Physical constants, element data, and isobaric-tag definitions come from `tacular` [@tacular], shared with `peptacular` and `paftacular`, so the three libraries agree on masses.

## Figures

Plots are generated from the spectrum and its peak-level metadata: stick, mirror, fragment-annotated, mass-error, sequence-coverage, faceted, extracted-ion chromatogram, and reporter-ion plots, along with a peak table for screen readers. A backend-neutral figure layer draws each plot with Plotly for interactive exploration or with Matplotlib [@hunter2007matplotlib] for print. `compose_figure` arranges several plots as one lettered multi-panel figure.

## Deconvolution and scoring

Isotope-cluster detection follows a greedy, apex-first algorithm related to MS-Deconv [@liu2010msdeconv] and THRASH [@horn2000thrash]. Predicted isotope distributions are calculated from an analyte-specific average elemental composition with a BRAIN-style recurrence [@dittwald2014brain]. Built-in models cover peptides, glycans, lipids, DNA, and RNA, and users can define their own.

The most intense unused peak seeds a candidate cluster. For each candidate charge state, the seed is aligned with the apex of the predicted envelope or a neighboring isotope of similar predicted abundance, and peaks are matched in both directions within mass-error, relative-abundance, and optional ion-mobility limits. Each candidate is scored against the complete predicted envelope with the Bhattacharyya coefficient. Missing predicted peaks and observed peaks whose intensities exceed the prediction lower the score, and missing isotopes above a user-selected detection threshold receive an additional penalty. This apex-first algorithm can infer the monoisotopic m/z when the monoisotopic peak is absent.

Fragment matches can be evaluated with a base-10 hyperscore following the dot-product and factorial structure of X!Tandem [@craig2004tandem], with unit theoretical intensities and deduplicated peak contributions, the normalized spectral contrast angle [@toprak2014spectralangle], matched fraction, and mass-error statistics. Spectra can be compared with cosine, modified-cosine [@watrous2012networking], and spectral-entropy [@li2021entropy] similarity.

## Reporter-ion quantification

Isobaric labelling with tandem mass tags (TMT and TMTpro) [@thompson2003tmt] or iTRAQ [@ross2004itraq] is quantified from reporter ions. `extract_reporter_ions` takes the most intense peak within a tolerance of each channel's reporter m/z, and `reporter_ion_table` returns one row per spectrum. Optional isotope-impurity correction builds a correction matrix from the reagent lot sheet and solves it by non-negative least squares, so corrected intensities cannot become negative.

## Readers and interoperability

`Reader` selects a parser from the suffix of an input path. It supports Bruker timsTOF `.d` directories, mzML files [@martens2011mzml], Thermo `.raw` files, and MGF, MS2, or MSP peak lists, and every parser returns the same `MsnSpectrum` type with the available acquisition metadata. A spectrum can be fetched by scan number, by native identifier, or by the `scannr` value in a Sage [@lazear2023sage] results table, which joins search results to their spectra. The mzML and Bruker parsers use the companion packages mzmlpy [@mzmlpy] and tdfpy [@tdfpy].

Spectrum interpretation follows standards from the HUPO Proteomics Standards Initiative. Fragments are generated by the companion package `peptacular` [@peptacular] from ProForma peptidoform notation [@leduc2022proforma], matched peaks are labelled by `paftacular` [@paftacular] in the mzPAF peak annotation format [@klein2024mzpaf], and spectral libraries are read and written in mzSpecLib 1.0 [@klein2024mzpaf]. Public spectra can be fetched by Universal Spectrum Identifier [@deutsch2021usi], and the optional `spectrl` package [@spectrl] encodes a spectrum as a URL-safe token. Bidirectional adapters connect spxtacular spectra with matchms and spectrum_utils objects.

# Research impact statement

spxtacular provides the spectrum processing and visualization used by the public [Spectra web application](https://spectra.tacular.dev/), an interactive spectrum viewer. The two connect through the URL-safe tokens described above: `to_spectrl_url` binds a spectrum into a link that the viewer decodes in the browser, so a spectrum can be sent to a colleague or cited in a manuscript without attaching a file and without the data reaching a server. In ongoing unpublished work, it also supports a post-translational modification motif-ratio quantitation workflow for covalent protein painting data [@son2023cpp] in a study of transthyretin amyloidosis (ATTR). Preliminary results from that study have been presented at ASMS 2026 [@pankow2026asms]. spxtacular has also been demonstrated with `peptacular` in an ASMS 2026 poster [@garrett2026asms].

# AI usage disclosure

During the preparation of this work the authors used Anthropic Claude large language models via the Claude Code interface for software-development assistance, including code, tests, and documentation, and for manuscript drafting and editing. The authors reviewed and edited all content and take full responsibility for the software and the publication.

# Author contributions

Patrick T. Garrett designed and implemented the library and drafted the manuscript. John R. Yates III directed the project and revised the manuscript.

# Conflicts of interest

The authors declare no competing financial interests.

# Acknowledgements

We thank Claire Delahunty, Ph.D., for a careful reading of the manuscript. This work was supported by the U.S. National Institutes of Health (grants R01 HL165168, R01 AG077046, R01 MH100175, and R01 AG075862 to J.R.Y.) and by the Skaggs Graduate School of Chemical and Biological Sciences at The Scripps Research Institute (P.T.G.).

# References
