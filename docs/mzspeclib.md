# Spectral libraries (mzSpecLib)

[mzSpecLib](https://psidev.info/mzSpecLib) is the HUPO-PSI standard for spectral libraries.
Each library spectrum is a set of controlled-vocabulary (CV) attributes, one or more analytes
(usually a ProForma peptidoform), and a peak list whose peaks carry
[mzPAF](https://psidev.info/mzPAF) annotations. spxtacular reads and writes version 1.0 in
both of its forms, text and JSON.

```python
from spxtacular import read_mzspeclib, write_mzspeclib

library = read_mzspeclib("library.mzspeclib.txt")      # or .json, optionally .gz
for entry in library:
    print(entry.key, entry.peptidoform, entry.charge, entry.spectrum.rt)

write_mzspeclib(library, "copy.mzspeclib.json")         # JSON because of the suffix
```

`read_mzspeclib` detects JSON from the content and gzip from its magic bytes.
`write_mzspeclib` writes JSON when the file name ends in `.json` (before any `.gz`) and text
otherwise; pass `format="text"` or `format="json"` to choose.

## The model

| Class | Holds |
|---|---|
| `SpectralLibrary` | `entries`, library-level `attributes`, `clusters` (keyed by cluster key). Iterable, `len()`, indexable |
| `LibraryEntry` | `spectrum` (`MsnSpectrum`), `key`, `name`, `analytes`, `interpretations`, `peak_annotations`, `peak_attributes`, `attributes` |
| `Analyte` | `id`, `peptidoform` (peptacular `ProFormaAnnotation`), `charge`, `attributes` |
| `Interpretation` | `id`, `members` (analyte ids of a chimeric match), `score`, `attributes`, `member_attributes` |
| `CvParam` | `accession`, `name`, `value`, `value_accession`, `group` |

`LibraryEntry` has shortcuts for the common case: `entry.peptidoform` and `entry.charge`
(precursor charge, else the first analyte's), and `entry.score` (the first interpretation's
`MS:1002357` PSM-level probability).

Build a library from spectra you already have with `LibraryEntry.from_spectrum`:

```python
import peptacular as pt
from spxtacular import LibraryEntry, match_fragments, write_mzspeclib

frags = pt.fragment("PEPTIDE/2", ion_types=("b", "y"), charges=(1, 2))
matches = match_fragments(spectrum, frags, tolerance=10, tolerance_type="ppm")
entry = LibraryEntry.from_spectrum(
    spectrum,
    "PEPTIDE/2",
    score=0.99,
    peak_annotations=matches,           # or one mzPAF string / list per peak
)
write_mzspeclib([entry], "mine.mzspeclib.txt")
```

The ProForma charge and the analyte charge are the same thing: a charge given on one side is
copied to the other, and a conflict raises `SpxtacularError`.

## Field mapping

Attributes that have a place on `MsnSpectrum` are moved there on reading and written back
from there. Everything else stays in the element's `attributes` as `CvParam` objects, in file
order.

| mzSpecLib term | spxtacular |
|---|---|
| `<Spectrum=N>` / `MS:1003237` library spectrum key | `entry.key` (filled with the 1-based position when `None` on writing) |
| `MS:1003061` library spectrum name | `entry.name` |
| `MS:1003208` experimental precursor monoisotopic m/z | `precursor_mz` with `is_monoisotopic=True` |
| `MS:1000744` selected ion m/z | `precursor_mz` with `is_monoisotopic=None` |
| `MS:1000041` charge state (spectrum) | precursor `charge` |
| `MS:1003085` previous MSn-1 scan precursor intensity | precursor `intensity` |
| `MS:1002815` / `MS:1002476` / `MS:1002954` | precursor `im` with `im_type` `ook0` / `drift_time_ms` / `ccs` |
| `MS:1000894` retention time | `rt` in seconds (minutes are converted) |
| `MS:1000045` collision energy | `collision_energy` (eV) |
| `MS:1000044` dissociation method | `activation_type` |
| `MS:1000465` scan polarity | `polarity` |
| `MS:1000511` ms level | `ms_level` (2 when absent) |
| `MS:1003057` scan number, `MS:1000285` TIC, `MS:1000927` ion injection time | `scan_number`, `total_ion_current`, `injection_time` |
| `MS:1003270` proforma peptidoform ion notation, or `MS:1003169` + `MS:1000041` | `analyte.peptidoform`, `analyte.charge` |
| `MS:1002357` PSM-level probability | `interpretation.score` |
| `MS:1003163` analyte mixture members | `interpretation.members` |
| Peak column 3 (mzPAF) | `entry.peak_annotations`: a tuple of paftacular `PafAnnotation` per peak |
| Peak columns 4+ | `entry.peak_attributes` |

A term is only moved when it appears once and any group it is in holds just its unit. Two
retention times, or a collision energy grouped with something other than its unit, stay in
`attributes` untouched.

Attribute sets (`<AttributeSet ...>` in text, `*_attribute_sets` in JSON) are resolved on
reading: the `all` set applies to every element, then each set the element names, then its
own attributes, each replacing earlier values of the same term. The writer never emits
attribute sets, so a read/write cycle expands them.

## What is not stored

mzSpecLib has no term for these `MsnSpectrum` fields, so `write_mzspeclib` drops them:
`native_id`, `resolution`, `analyzer`, `ramp_time`, the m/z, ion-mobility and isolation
ranges, the precursor `iso_score`, the per-peak `charge`, `im` and `iso_score` arrays, and the
spectrum-level `im_type`. `ActivationType.PASEF` is written as `MS:1002481` (higher energy
beam-type CID) and reads back as `HCD`. EThcD and ETciD are written as their combined terms
(`MS:1002631`, `MS:1003182`); ETD plus a supplemental activation term reads back as the
combined type.

`write_mzspeclib` raises `SpxtacularError` for a spectrum it cannot represent: profile data,
neutral masses, more than one precursor, ion mobility of the generic `im` type, an
`activation_type` with no PSI-MS term, duplicate keys, or a user attribute that repeats a
field it writes.

## Limits

- JSON peak annotations are read as mzPAF strings (or lists of strings). The spec's
  structured annotation objects are not supported and raise `SpxtacularError`.
- Only mzPAF annotations are parsed. A peak column in another format raises, naming the
  file and line.
- Text values carry no type: a value that looks like a number is read as one, except for
  terms whose values are text by definition (names, versions, accessions, USIs, ProForma).
- The whole file is read into memory.

## mzPAF in MSP and MGF

`write_msp` and `write_mgf` take the same annotations with `annotations=`, one entry per
spectrum. Annotated peaks get a quoted last column; the default output is unchanged.

```python
write_msp([spectrum], "lib.msp", annotations=[matches])
# 175.119 100.0 "y1/0.3ppm"
```

`MgfReader` and `MspReader` read such files and ignore the annotation column.
