---
title: 'spxtacular: A Chainable Python Library for Mass Spectrometry Spectrum Processing'
tags:
  - 'Python'
  - 'proteomics'
  - 'mass spectrometry'
  - 'deconvolution'
  - 'visualization'
  - 'bioinformatics'
authors:
  - name: 'Patrick Tyler Garrett'
    orcid: 0000-0002-8434-9693
    affiliation: 1
  - name: 'John R. Yates III'
    orcid: 0000-0001-5267-1672
    corresponding: true
    affiliation: 1
affiliations:
  - name: 'The Scripps Research Institute, La Jolla, CA 92037, USA'
    index: 1
date: 'August 2026'
bibliography: paper.bib
---

# Summary

Mass spectrometry generates spectra whose peaks are not the quantities an analyst reasons about. A tryptic peptide arrives as an isotopic envelope spread over several m/z values at an unknown charge, sitting on a noise floor, and the question asked of it is about one neutral mass. The steps between those two representations, denoising, isotope deconvolution, charge assignment, neutral mass conversion, fragment matching and scoring, are standard, and are reimplemented in nearly every analysis script that needs them.

**spxtacular** is a Python library that provides those steps as methods on a single chainable `Spectrum` object. That object carries m/z and intensity alongside optional per-peak charge, ion mobility, and isotopic profile score. What one step learns is therefore available to the next, rather than returned as a parallel array the caller has to keep in register.

```python
neutral = (spec.denoise(method="mad")
               .deconvolute(charge_range=(1, 4), tolerance=15,
                            tolerance_type="ppm", min_score=0.5)
               .decharge())
```

A `SpectrumType` tag records whether the object holds profile, centroid or deconvoluted data, and guards the operations that make sense in only one of those states. Calling `decharge()` on a spectrum that was never deconvoluted raises, rather than silently returning wrong masses. The library exports 58 public names and depends only on NumPy [@harris2020numpy], pandas, Plotly, and two companion packages for peptide chemistry and fragment notation.

# Statement of need

Existing Python tooling for proteomics divides along a line that leaves a gap in the middle. File format libraries such as pyteomics [@goloborodko2013pyteomics] and pyOpenMS [@rost2014pyopenms] read spectra and hand back arrays, and the processing is the user's problem. Search engines and deconvolution tools such as MS-Deconv [@liu2010msdeconv] and FLASHDeconv [@jeong2020flashdeconv] are complete programs rather than libraries: they are excellent at the one job they do and are difficult to call in the middle of an interactive analysis. Between the two sits the work of actually looking at a spectrum, which is where method development happens and where most bespoke scripts are written.

spxtacular targets that middle. Deconvolution is a method call that returns a spectrum, not a subprocess that returns a file, so its output can be filtered, plotted, matched against fragments and fed back into another deconvolution inside one session. Charge and isotope score travel with the peaks, so the quality of an assignment stays inspectable. A weakly scoring cluster is distinguishable from a convincing one in a filter and in a plot, rather than flattened into an accept or reject decision taken upstream.

The library also treats visualization as part of the analysis rather than a reporting step bolted on afterwards. Every spectrum plots itself, and all figures share one theme whose palettes were validated for protanopia and deuteranopia against both light and dark backgrounds. Charge state is encoded as an ordinal ramp rather than a categorical cycle, which is what stops a ten-color palette from drawing the first and the eleventh state identically. Intensities are shown relative to the base peak by default. A `table_view()` helper renders the same peak data as an HTML table for screen reader users.

# Deconvolution

Isotope cluster detection follows the greedy strategy that MS-Deconv and THRASH [@horn2000thrash] established, with scoring made explicit and per-peak. The algorithm seeds on the most intense unused peak. For each state in the requested range it extends a candidate cluster forward in steps of one neutron mass divided by that state, then scores the candidate against a theoretical isotope distribution using the Bhattacharyya coefficient. Detectable peaks that are missing from the envelope are penalized, and the best scoring state wins. Clusters falling below `min_score` are recorded as singletons rather than discarded, and their peaks stay available to later seeds.

The score survives into the result as `iso_score`, which is what makes the threshold a filter the user controls rather than a constant compiled into the algorithm. \autoref{fig:pipeline} shows the effect on one real spectrum: tandem mass spectrometry (MS2) scan 2 of the Bruker timsTOF example acquisition shipped with the library, deconvoluted over charges one to 4 at 15 ppm tolerance. Of 500 centroid peaks, 38 resolved into charge-assigned clusters carrying 22.5% of the total ion current at a mean isotope score of 0.86. States as high as 4 were assigned, and `decharge()` returned 38 neutral masses reaching 2,282 Da.

![One MS2 spectrum before and after deconvolution, drawn by the library's own `mirror_plot`. Above the axis, the deconvoluted spectrum colored by assigned charge on the ordinal ramp; unassigned singletons stay neutral gray. Below it, the vendor centroid peaks the clusters were built from, mirrored so a cluster can be traced back to the peaks that produced it.](figures/pipeline.png){ #fig:pipeline }

# Reading files and interoperating

`Reader` dispatches on the path suffix to Bruker timsTOF `.d` directories, mzML [@martens2011mzml], and Thermo `.raw` files. Every backend yields the same `MsnSpectrum` type, with scan number, retention time, precursors, collision energy and ion mobility filled in where the format carries them. MGF and MS2 peak lists are read and written with no optional dependency at all. The vendor-specific readers are optional extras, so a downstream package can depend on spxtacular without pulling in a .NET runtime it will never use.

Fragment matching accepts fragments from the companion peptacular library, which implements ProForma 2.0 [@leduc2018proforma], and labels matched peaks in mzPAF notation. Scoring covers the hyperscore that X!Tandem popularized [@craig2004tandem] and the normalized spectral contrast angle [@toprak2014spectralangle], alongside the matched fraction and error statistics a method paper usually reports. A spectrum can be serialized to a compact URL safe token, which makes a specific spectrum citable in an issue or a message without attaching a file.

# State of the field

matchms [@huber2020matchms] addresses a neighboring problem, spectrum to spectrum similarity for metabolomics, and shares the goal of a processing pipeline expressed as composable steps. spectrum_utils [@bittremieux2020spectrumutils] covers annotation and plotting of identified spectra with a similar emphasis on visualization quality. spxtacular differs from both in making charge state deconvolution a first class operation whose per-peak evidence is retained, and in reading vendor formats directly rather than starting from an already converted peak list. It complements rather than replaces them, and interoperates through the same community standards, mzML, ProForma, mzPAF and the Universal Spectrum Identifier [@deutsch2021usi].
