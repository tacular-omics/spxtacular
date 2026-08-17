# Polarity, adduct, and isotope-model roadmap

Status: exploration for `codex/polarity-adducts`

## Summary

Add a first-class `IonizationModel` before expanding isotope-model factories or
processing provenance. Keep per-peak charge values as positive magnitudes: the
existing `-1` value means “unknown/singleton,” so signed charges would be
ambiguous and would break filtering, plotting, and serialization assumptions.

An ionization model should own conversion between observed m/z and neutral mass.
Deconvolution still finds charge magnitude from isotope spacing, but it must use
the ionization model when estimating neutral mass for isotope-envelope scoring.
`decharge()` must reuse the model selected during deconvolution.

The source worktree has uncommitted isotope-model work (`IsotopeModel` plus
peptide, lipid, glycan, DNA, and RNA presets). Land or transplant it before the
later isotope work described here.

## Proposed mass model

```python
@dataclass(frozen=True, slots=True)
class IonizationModel:
    name: str
    polarity: Polarity
    carrier_mass: float  # signed ion-mass delta per unit charge

    def ion_mz(self, neutral_mass: float, charge: int) -> float:
        return (neutral_mass + charge * self.carrier_mass) / charge

    def neutral_mass(self, mz: float, charge: int) -> float:
        return mz * charge - charge * self.carrier_mass
```

`charge` remains a positive magnitude. `carrier_mass` is positive for
attachment and negative for loss.

| Preset | Polarity | Carrier delta | Neutral-mass equation |
| --- | --- | ---: | --- |
| `[M+H]+` | positive | `+PROTON_MASS` | `M = mz*z - z*PROTON_MASS` |
| `[M-H]-` | negative | `-PROTON_MASS` | `M = mz*z + z*PROTON_MASS` |
| `[M+Na]+` | positive | sodium-cation mass | `M = mz*z - z*NA_CATION_MASS` |
| `[M+NH4]+` | positive | ammonium-cation mass | `M = mz*z - z*NH4_CATION_MASS` |

The custom form should accept explicit `carrier_mass`, `polarity`, and `name`.
Validate positive integer charge, finite masses, and non-negative neutral mass.

Preset display notation must reflect charge. For example, the sodium preset is
`[M+Na]+` at charge 1 and `[M+2Na]2+` at charge 2; it must not label the latter
as `[M+Na]+`. The same applies to protonation, deprotonation, and ammonium.

This initial model describes one repeated carrier per charge. A future
`Adduct`/`IonComposition` can represent mixed ions such as `[M+2H+Na]3+`.
Implementation should call model methods, not inline equations, so that
extension remains possible.

## Public API and resolution

```python
decon = spec.deconvolute(
    isotope_model="lipid",
    ionization_model="[M+Na]+",
)
neutral = decon.decharge()  # reuses recorded ionization model

custom = IonizationModel(
    name="[M+K]+",
    polarity="positive",
    carrier_mass=38.963158,
)
```

1. `deconvolute()` defaults to `[M+H]+` for backward compatibility.
2. A named/custom model is authoritative.
3. Do not infer an adduct solely from polarity in the first release. If no model
   is passed and an `MsnSpectrum` is negative, warn or require an explicit model.
4. Reject an explicit model that conflicts with known spectrum polarity.
5. `decharge()` uses provenance. An explicit override should warn because it
   changes the interpretation of existing m/z values.
6. Legacy deconvoluted spectra without provenance use historical `[M+H]+`, with
   a warning for known negative polarity.

The low-level `deconvolve_spectrum()` must accept the resolved model (or carrier
mass) because it currently subtracts proton mass while selecting a template.

## Provenance

The current model has only `denoised` and `normalized` strings. Introduce a
focused immutable record rather than more parallel flags:

```python
@dataclass(frozen=True, slots=True)
class DeconvolutionProvenance:
    isotope_model: ModelReference
    ionization_model: IonizationModelReference
    charge_range: tuple[int, int]
    tolerance: float
    tolerance_type: str
    intensity_mode: str
    min_intensity: float
    min_score: float
```

Attach `deconvolution: DeconvolutionProvenance | None` to `Spectrum`. A general
chronological `processing_history` can follow, but should not block correct
decharging. Custom models need serializable value descriptions; built-ins may
use stable identifiers plus a schema version. Do not pickle Python objects.

Provenance must round-trip through native `.npz` JSON, the namespaced matchms
payload (bump its schema while reading v1), and spectrl namespaced user params.
It must also survive copy/update/masks. Lossy spectrum-utils conversion should
report it among dropped fields.

## Model-specific isotope mass offsets

The algorithm currently searches at `pt.C13_NEUTRON_MASS / charge`. This is
separate from isotope-envelope abundance modeling and can bias oxygen-, sulfur-,
nitrogen-, and phosphorus-rich analytes.

Extend isotope definitions from nominal offset plus abundance to include exact
mass shift. For every nominal A+k bin calculate total probability and its
probability-weighted centroid shift. Expose
`IsotopeModel.mass_offsets(neutral_mass, ...)` aligned with `distribution(...)`.
Matching then uses `mono_mz + offsets[k] / charge` instead of
`mono_mz + k*C13_NEUTRON_MASS/charge`.

The uncommitted BRAIN-style recurrence tracks abundance only, so it needs a
companion first-moment recurrence or polynomial propagation of
`(probability, probability*mass_shift)`. Exact fine-structure enumeration is
not required for the first implementation.

## Factories

Factories should produce `IsotopeModel` values, not new algorithm branches.

1. Nucleotide base composition: base counts/fractions plus DNA/RNA backbone and
   terminal composition.
2. Glycan composition: monosaccharide counts/types, with averagose as fallback.
3. Lipid classes: class/headgroup plus chain carbons and unsaturation, using
   exact formulas where the grammar is unambiguous and broad lipid averagine as
   fallback.

```python
IsotopeModel.for_nucleotides({"A": 3, "C": 2, "G": 4, "T": 1}, polymer="dna")
IsotopeModel.for_glycan({"Hex": 5, "HexNAc": 4, "Fuc": 1})
IsotopeModel.for_lipid("PC", carbons=34, double_bonds=1)
```

Factory outputs should retain a stable human-readable name and source
composition in their serializable reference.

## Hidden coupling

The proton/C13 assumptions occur beyond `deconvolute()` and `decharge()`:

- `decon/scored.py`: neutral-mass estimate for isotope-template lookup;
- `decon/greedy.py`: isotope spacing;
- `Spectrum.remove_precursor_peak()`: neutral conversion, charge-state targets,
  and isotope spacing;
- `peaklist.py`: precursor neutral-mass helper;
- documentation/examples describing positive protonation only.

All mass conversions should route through shared model methods. Otherwise a
negative-mode spectrum may deconvolute correctly while precursor removal or a
later decharge still applies the positive-protonated equation.

## Implementation order

1. Land/transplant the current isotope-model work and establish a green baseline.
2. Add `ionization.py`, immutable models, presets, resolver, and conversion tests.
3. Thread the model through low-level and `Spectrum.deconvolute()` scoring while
   retaining the current default.
4. Add deconvolution provenance and make `decharge()` consume it.
5. Update precursor removal and remaining mass helpers.
6. Add `.npz`, matchms, and spectrl round-trip tests and lossy warnings.
7. Add model-specific isotope offsets.
8. Add nucleotide, glycan, and lipid factories.

## Minimum test matrix

- Every preset round-trips `neutral -> m/z -> neutral` for charges 1–5.
- `[M-H]-` works at charges 1–5 while stored charge values stay positive.
- Sodium/ammonium models affect scoring and `decharge()` by the expected delta.
- Custom positive and negative carrier masses validate and round-trip.
- Existing no-argument positive-protonation behavior is unchanged.
- Known negative polarity plus implicit legacy protonation warns or errors.
- Provenance survives copy, filtering, `.npz`, matchms, and spectrl.
- Model-specific offsets differ among peptide, lipid, glycan, DNA, and RNA.
- Precursor removal uses the same ionization/isotope models as deconvolution.

## Decisions before implementation

- Infer `[M-H]-` from negative polarity or require an explicit model? Explicit is
  safer; inference is more convenient.
- Does custom `carrier_mass` mean ion-mass delta (recommended) or neutral mass
  that the library adjusts for electron mass?
- Start with focused deconvolution provenance (recommended) or a general history?
- Are mixed/multiple adduct compositions explicitly out of scope for v1?
- Do sodium/ammonium presets allow repeated carriers at higher charge or default
  to charge 1 unless the caller opts in?
