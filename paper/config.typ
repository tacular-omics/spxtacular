// =============================================================================
// FILL THIS IN. Everything project-specific about the manuscript lives here.
//
// This is the single source of truth for the manuscript's identity. paper.typ
// imports it for the PDF and the Word front matter, wordcount.typ counts the
// abstract out of it, and audio/config.py reads the title straight out of this
// file so the narration can never announce a title the paper no longer has.
//
// Nothing below this block should need editing to start a new paper.
// =============================================================================

#let paper-title = "spxtacular: A Chainable Python Library for Mass Spectrometry Spectrum Processing"

// Short form used on the audiobook cover art. Keep it to a couple of words.
#let paper-wordmark = "spxtacular"

// Shown under the wordmark on the cover; \n breaks the line.
#let paper-cover-subtitle = "Spectrum Processing,\nDeconvolution and Visualization"

// `orcid` and `corresponding` are read by tools/joss.py for the paper.md front
// matter; arkheion renders the ORCID badge from the same field and ignores the
// rest, so the two front matters cannot disagree about who wrote this.
#let paper-authors = (
  (
    name: "Patrick Tyler Garrett",
    email: "pgarrett@scripps.edu",
    orcid: "0000-0002-8434-9693",
    affiliation: "The Scripps Research Institute, La Jolla, CA 92037, USA",
  ),
  (
    name: "John R. Yates III",
    email: "jyates@scripps.edu",
    orcid: "0000-0001-5267-1672",
    corresponding: true,
    affiliation: "The Scripps Research Institute, La Jolla, CA 92037, USA",
  ),
)

#let paper-keywords = (
  "Python",
  "proteomics",
  "mass spectrometry",
  "deconvolution",
  "visualization",
  "bioinformatics",
)

#let paper-date = "August 2026"

// Shown on the audiobook cover under the author line.
#let paper-institution = "The Scripps Research Institute"

// The bibliography style. Typst ships CSL styles by name, e.g.
// "american-chemical-society", "ieee", "nature", "apa".
#let paper-bib-style = "apa"

// The abstract. Kept as its own binding, rather than inline in the template
// call, because three separate consumers slice it out of this file by name: the
// Word export path, the word counter (journals cap the abstract separately), and
// the audiobook narrator.
// JOSS papers carry no abstract. This one is for the PDF and Word renderings of
// the manuscript, which are read outside the JOSS submission; tools/joss.py
// deliberately does not emit it into paper.md.
#let paper-abstract = [
  spxtacular is a Python library for processing mass spectrometry spectra. It
  presents one chainable `Spectrum` object that carries m/z, intensity, charge,
  ion mobility and per-peak isotope scores together, and exposes the steps
  between a vendor centroid list and an interpreted spectrum as methods on it:
  denoising, scored isotope deconvolution, conversion to neutral masses,
  fragment matching, and peptide-spectrum match scoring. Readers for Bruker
  timsTOF, mzML and Thermo raw files, and for MGF and MS2 peak lists, produce
  the same object, so an analysis written against one instrument runs against
  another. Every processing step is plottable through a color-vision-safe Plotly
  theme shared by all of the library's figures.
]

// -----------------------------------------------------------------------------
// Derived values. Nothing to edit below here.
// -----------------------------------------------------------------------------

// Unique affiliations in first-appearance order, so the Word front matter can
// number them the way the PDF template does. Deriving this rather than typing a
// second author line by hand removes the drift that a duplicated list invites.
#let paper-affiliations = {
  let seen = ()
  for a in paper-authors {
    if a.affiliation not in seen { seen.push(a.affiliation) }
  }
  seen
}

#let affiliation-number(affil) = (
  paper-affiliations.position(x => x == affil) + 1
)

// "Ada Lovelace^1, Grace Hopper^2" for the Word front matter, which has no
// template to build an author line for it. Derived rather than retyped, so the
// superscript markers cannot drift out of step with the PDF.
// Wrapped in a code block because a method chain broken across lines after
// `#let x =` would otherwise end at the first newline.
#let paper-author-line = {
  paper-authors
    .map(a => a.name + super(str(affiliation-number(a.affiliation))))
    .join(", ")
}

// Generational and post-nominal suffixes, so a surname lookup does not return
// "III" for "John R. Yates III". Compared case- and period-insensitively.
#let name-suffixes = (
  "jr",
  "sr",
  "ii",
  "iii",
  "iv",
  "v",
  "phd",
  "md",
  "dphil",
  "dsc",
  "esq",
)

// The family name: the last token that is not a suffix. "John R. Yates III"
// gives "Yates". Used for the audiobook artist tag and the cover art.
#let surname-of(full) = {
  let parts = full.split(" ").filter(p => p.trim() != "")
  let i = parts.len() - 1
  while i > 0 and lower(parts.at(i).replace(".", "")) in name-suffixes {
    i -= 1
  }
  parts.at(i)
}

// "Lovelace, Hopper" -- used as the audiobook artist tag and on the cover.
#let paper-surnames = paper-authors.map(a => surname-of(a.name))

// Restated on the Supporting Information title page.
#let si-authors = paper-authors.map(a => a.name).join(", ")
