# Polarity, adduct, and isotope-model design record

Status: implemented in 0.5.0, with future extensions listed below.

## Current model

spxtacular stores per-peak charge as a magnitude. Positive values are assigned charge states,
`-1` means unknown or singleton, and `0` marks a neutral mass after `decharge()`. Scan polarity and
the mass of the repeated charge carrier are represented separately by `IonizationModel`.

```python
@dataclass(frozen=True, slots=True)
class IonizationModel:
    name: str
    polarity: Polarity
    carrier_mass: float
    carrier: str = "custom"

    def ion_mz(self, neutral_mass, charge): ...
    def neutral_mass(self, mz, charge): ...
```

`carrier_mass` is the signed ion-mass delta per unit charge. The built-in models are:

| Preset | Polarity | Neutral-mass equation |
| --- | --- | --- |
| `[M+H]+` | positive | `M = mz*z - z*PROTON_MASS` |
| `[M-H]-` | negative | `M = mz*z + z*PROTON_MASS` |
| `[M+Na]+` | positive | `M = mz*z - z*SODIUM_CATION_MASS` |
| `[M+NH4]+` | positive | `M = mz*z - z*AMMONIUM_CATION_MASS` |

The notation reflects charge magnitude. For example, sodiated charge 2 is displayed as
`[M+2Na]2+`.

## Resolution and provenance

`Spectrum.deconvolute()` accepts a preset name, alias, signed carrier mass, or custom
`IonizationModel`. An explicit model is authoritative and must agree with known scan polarity.
When no model is passed, a negative scan uses `[M-H]-` and other scans use `[M+H]+`.

The output stores a `DeconvolutionProvenance` record containing the resolved isotope and ionization
models plus every envelope and matching parameter. `decharge()` reuses that record. In its absence,
it applies the same polarity fallback. Native `.npz`, namespaced matchms metadata, and spectrl user
parameters preserve the record. Copying, filtering, masking, and updating a spectrum preserve it
unless an operation invalidates deconvolution state.

```python
decon = spec.deconvolute(
    isotope_model="lipid",
    ionization_model="[M+Na]+",
)
neutral = decon.decharge()
```

Automatic precursor removal also resolves the isotope and ionization models from explicit
arguments, deconvolution provenance, or scan polarity. Mass conversions route through model
methods instead of inline proton-only equations.

## Isotope models

`IsotopeModel` predicts nominal isotope-envelope abundance from an average elemental composition.
Built-in models cover peptides, glycans, lipids, DNA, and RNA. Callers can supply custom atoms per
Dalton, fixed composition, and isotope abundances. The exact custom model definition is stored in
deconvolution provenance.

The current deconvolution algorithm uses `peptacular.C13_NEUTRON_MASS / charge` for nominal isotope
spacing. This is separate from the analyte-specific abundance model.

## Future scope

The following extensions are not implemented:

1. Mixed charge carriers such as `[M+2H+Na]3+`. The current model repeats one carrier for every
   unit of charge.
2. Model-specific exact isotope mass offsets. A future isotope model could propagate
   probability-weighted exact mass shifts alongside nominal-bin abundance.
3. Composition factories such as `for_nucleotides()`, `for_glycan()`, and `for_lipid()`. Current
   callers construct `IsotopeModel` directly or select a broad built-in model.
4. A chronological processing history. The current focused provenance record covers
   deconvolution only.

These extensions should preserve the existing charge sentinel meanings and continue routing mass
conversion through `IonizationModel` methods.
