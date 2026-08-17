"""Fast aggregated isotope envelopes and biological average-composition models.

The distribution calculation uses the Newton-Girard power-series recurrence
described by the BRAIN algorithm.  It calculates nominal neutron-offset peaks,
which is the representation needed for isotope-cluster scoring.  It does not
calculate isotope fine structure.

References
----------
Dittwald et al. (2014), BRAIN 2.0, doi:10.1007/s13361-013-0796-5.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cache, lru_cache
from math import ceil, floor, sqrt
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

MAX_ISOTOPE_PEAKS: Final[int] = 32

# Monoisotopic masses and terrestrial natural abundances used by Tacular and
# Peptacular.  An abundance tuple is (nominal neutron offset, probability).
_MONOISOTOPIC_MASS: Final[dict[str, float]] = {
    "C": 12.0,
    "H": 1.00782503223,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "P": 30.97376199842,
    "S": 31.9720711744,
}

NATURAL_ISOTOPE_ABUNDANCES: Final[dict[str, tuple[tuple[int, float], ...]]] = {
    "C": ((0, 0.9893), (1, 0.0107)),
    "H": ((0, 0.999885), (1, 0.000115)),
    "N": ((0, 0.99636), (1, 0.00364)),
    "O": ((0, 0.99757), (1, 0.00038), (2, 0.00205)),
    "P": ((0, 1.0),),
    "S": ((0, 0.9499), (1, 0.0075), (2, 0.0425), (4, 0.0001)),
}


class IsotopeModelType(StrEnum):
    """Built-in average-composition models."""

    PEPTIDE = "peptide"
    GLYCAN = "glycan"
    LIPID = "lipid"
    DNA = "dna"
    RNA = "rna"


type IsotopeAbundances = Mapping[str, Mapping[int, float]]


def _canonical_abundances(
    abundances: IsotopeAbundances | None,
) -> tuple[tuple[str, tuple[tuple[int, float], ...]], ...]:
    if abundances is None:
        return ()

    canonical: list[tuple[str, tuple[tuple[int, float], ...]]] = []
    for raw_element, raw_pattern in abundances.items():
        element = str(raw_element)
        if not raw_pattern:
            raise ValueError(f"isotope_abundances[{element!r}] cannot be empty")

        pattern: list[tuple[int, float]] = []
        for raw_offset, raw_abundance in raw_pattern.items():
            offset = int(raw_offset)
            abundance = float(raw_abundance)
            if offset < 0:
                raise ValueError(f"isotope offsets must be non-negative, got {offset} for {element}")
            if abundance < 0.0 or not np.isfinite(abundance):
                raise ValueError(f"isotope abundances must be finite and non-negative, got {abundance} for {element}")
            if abundance > 0.0:
                pattern.append((offset, abundance))

        total = sum(abundance for _, abundance in pattern)
        if total <= 0.0:
            raise ValueError(f"isotope_abundances[{element!r}] must contain a positive abundance")
        merged: dict[int, float] = {}
        for offset, abundance in pattern:
            merged[offset] = merged.get(offset, 0.0) + abundance / total
        if merged.get(0, 0.0) <= 0.0:
            raise ValueError(f"isotope_abundances[{element!r}] must include a positive offset-0 isotope")
        canonical.append((element, tuple(sorted(merged.items()))))
    return tuple(sorted(canonical))


def _canonical_counts(values: Mapping[str, int | float], *, integer: bool) -> tuple[tuple[str, int | float], ...]:
    canonical: list[tuple[str, int | float]] = []
    for raw_element, raw_count in values.items():
        element = str(raw_element)
        count = int(raw_count) if integer else float(raw_count)
        if not np.isfinite(count) or count < 0:
            raise ValueError(f"element counts must be finite and non-negative, got {raw_count!r} for {element}")
        if integer and count != raw_count:
            raise ValueError(f"fixed composition counts must be integers, got {raw_count!r} for {element}")
        if count:
            canonical.append((element, count))
    return tuple(sorted(canonical))


def _patterns_for_signature(
    abundance_signature: tuple[tuple[str, tuple[tuple[int, float], ...]], ...],
) -> dict[str, tuple[tuple[int, float], ...]]:
    patterns = dict(NATURAL_ISOTOPE_ABUNDANCES)
    patterns.update(dict(abundance_signature))
    return patterns


@cache
def _element_log_coefficients(pattern: tuple[tuple[int, float], ...], max_isotopes: int) -> tuple[float, ...]:
    """Compile one element polynomial into BRAIN recurrence coefficients."""
    base = dict(pattern).get(0, 0.0)
    if base <= 0.0:
        raise ValueError("the offset-0 isotope abundance must be positive")

    ratio = np.zeros(max_isotopes, dtype=np.float64)
    ratio[0] = 1.0
    for offset, abundance in pattern:
        if 0 < offset < max_isotopes:
            ratio[offset] += abundance / base

    log_coeff = np.zeros(max_isotopes, dtype=np.float64)
    for n in range(1, max_isotopes):
        correction = 0.0
        for k in range(1, n):
            correction += k * log_coeff[k] * ratio[n - k]
        log_coeff[n] = ratio[n] - correction / n
    return tuple(float(value) for value in log_coeff)


def brain_isotopic_distribution(
    composition: Mapping[str, int],
    max_isotopes: int = MAX_ISOTOPE_PEAKS,
    isotope_abundances: IsotopeAbundances | None = None,
) -> NDArray[np.float64]:
    """Calculate an aggregated nominal isotope distribution with BRAIN.

    The returned probabilities start at the all-light-isotope composition and
    sum to one over the requested window.
    """
    if max_isotopes < 1:
        raise ValueError(f"max_isotopes must be positive, got {max_isotopes}")
    counts = _canonical_counts(composition, integer=True)
    abundance_signature = _canonical_abundances(isotope_abundances)
    return np.asarray(_brain_distribution_cached(counts, abundance_signature, max_isotopes), dtype=np.float64)


@lru_cache(maxsize=4096)
def _brain_distribution_cached(
    counts: tuple[tuple[str, int | float], ...],
    abundance_signature: tuple[tuple[str, tuple[tuple[int, float], ...]], ...],
    max_isotopes: int,
) -> tuple[float, ...]:
    patterns = _patterns_for_signature(abundance_signature)
    aggregate_log = np.zeros(max_isotopes, dtype=np.float64)
    for element, raw_count in counts:
        count = int(raw_count)
        try:
            pattern = patterns[element]
        except KeyError as exc:
            raise ValueError(
                f"no isotope abundances are available for {element!r}; provide isotope_abundances for that element"
            ) from exc
        aggregate_log += count * np.asarray(_element_log_coefficients(pattern, max_isotopes))

    distribution = np.zeros(max_isotopes, dtype=np.float64)
    distribution[0] = 1.0
    for n in range(1, max_isotopes):
        value = 0.0
        for k in range(1, n + 1):
            value += k * aggregate_log[k] * distribution[n - k]
        distribution[n] = value / n

    # Round-off can produce tiny negative values in the far tail.
    distribution[distribution < 0.0] = 0.0
    total = float(distribution.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ArithmeticError("isotope recurrence did not produce a finite distribution")
    distribution /= total
    return tuple(float(value) for value in distribution)


@dataclass(frozen=True, slots=True)
class IsotopeModel:
    """Average elemental composition used to predict isotope envelopes.

    ``atoms_per_da`` gives the expected count of each element per Dalton.
    ``fixed_composition`` contains atoms that occur once rather than scaling
    with mass, such as terminal water for a peptide or oligonucleotide.
    """

    atoms_per_da: Mapping[str, float]
    fixed_composition: Mapping[str, int] = field(default_factory=dict)
    isotope_abundances: IsotopeAbundances | None = None
    name: str = "custom"
    _signature: tuple[
        tuple[tuple[str, int | float], ...],
        tuple[tuple[str, int | float], ...],
        tuple[tuple[str, tuple[tuple[int, float], ...]], ...],
    ] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        rates = _canonical_counts(self.atoms_per_da, integer=False)
        fixed = _canonical_counts(self.fixed_composition, integer=True)
        abundances = _canonical_abundances(self.isotope_abundances)
        known = _patterns_for_signature(abundances)
        for element, _ in (*rates, *fixed):
            if element not in known:
                raise ValueError(
                    f"no isotope abundances are available for {element!r}; provide isotope_abundances for that element"
                )
        for element, _ in fixed:
            if element not in _MONOISOTOPIC_MASS:
                raise ValueError(f"no monoisotopic mass is available for fixed element {element!r}")

        object.__setattr__(self, "atoms_per_da", MappingProxyType(dict(rates)))
        object.__setattr__(self, "fixed_composition", MappingProxyType(dict(fixed)))
        object.__setattr__(
            self,
            "isotope_abundances",
            MappingProxyType({element: MappingProxyType(dict(pattern)) for element, pattern in abundances}),
        )
        object.__setattr__(self, "_signature", (rates, fixed, abundances))

    @property
    def fixed_mass(self) -> float:
        """Monoisotopic mass of the non-scaling composition."""
        return sum(_MONOISOTOPIC_MASS[element] * int(count) for element, count in self.fixed_composition.items())

    def estimate_composition(self, neutral_mass: float) -> dict[str, int]:
        """Estimate an integer elemental composition for ``neutral_mass``."""
        mass = float(neutral_mass)
        if not np.isfinite(mass) or mass < 0.0:
            raise ValueError(f"neutral_mass must be finite and non-negative, got {neutral_mass!r}")
        scalable_mass = max(0.0, mass - self.fixed_mass)
        composition = {element: floor(rate * scalable_mass + 0.5) for element, rate in self.atoms_per_da.items()}
        for element, count in self.fixed_composition.items():
            composition[element] = composition.get(element, 0) + int(count)
        return {element: count for element, count in composition.items() if count > 0}

    def distribution(self, neutral_mass: float, max_isotopes: int = MAX_ISOTOPE_PEAKS) -> NDArray[np.float64]:
        """Return the cached envelope for the nearest integer Dalton."""
        mass = float(neutral_mass)
        if not np.isfinite(mass) or mass < 0.0:
            raise ValueError(f"neutral_mass must be finite and non-negative, got {neutral_mass!r}")
        if max_isotopes < 1:
            raise ValueError(f"max_isotopes must be positive, got {max_isotopes}")
        nominal_mass = floor(mass + 0.5)
        return np.asarray(_model_distribution_cached(self._signature, nominal_mass, max_isotopes), dtype=np.float64)

    def adaptive_distribution(
        self,
        neutral_mass: float,
        min_relative_abundance: float = 0.01,
        max_isotopes: int | None = None,
    ) -> NDArray[np.float64]:
        """Return an envelope long enough to cover its significant tail.

        The envelope always starts at A+0, even when that peak is below
        ``min_relative_abundance``. When ``max_isotopes`` is ``None``, the
        length is selected from the isotope-count mean and variance and then
        expanded until the complete trailing window is below the threshold.
        """
        mass = float(neutral_mass)
        if not np.isfinite(mass) or mass < 0.0:
            raise ValueError(f"neutral_mass must be finite and non-negative, got {neutral_mass!r}")
        threshold = float(min_relative_abundance)
        if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
            raise ValueError(f"min_relative_abundance must be in (0, 1], got {min_relative_abundance!r}")
        if max_isotopes is not None and max_isotopes < 1:
            raise ValueError(f"max_isotopes must be positive or None, got {max_isotopes}")
        nominal_mass = floor(mass + 0.5)
        return np.asarray(
            _adaptive_model_distribution_cached(self._signature, nominal_mass, threshold, max_isotopes),
            dtype=np.float64,
        )

    def apex_index(self, neutral_mass: float, max_isotopes: int = MAX_ISOTOPE_PEAKS) -> int:
        """Index of the predicted most abundant isotope peak."""
        return int(np.argmax(self.distribution(neutral_mass, max_isotopes=max_isotopes)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable value description."""
        return {
            "name": self.name,
            "atoms_per_da": dict(self.atoms_per_da),
            "fixed_composition": dict(self.fixed_composition),
            "isotope_abundances": {
                element: dict(pattern) for element, pattern in (self.isotope_abundances or {}).items()
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> IsotopeModel:
        """Reconstruct a model from :meth:`to_dict` output."""
        raw_abundances = value.get("isotope_abundances")
        abundances = None
        if raw_abundances:
            abundances = {
                str(element): {int(offset): float(abundance) for offset, abundance in pattern.items()}
                for element, pattern in raw_abundances.items()
            }
        return cls(
            name=str(value.get("name", "custom")),
            atoms_per_da={str(element): float(count) for element, count in value["atoms_per_da"].items()},
            fixed_composition={
                str(element): int(count) for element, count in value.get("fixed_composition", {}).items()
            },
            isotope_abundances=abundances,
        )


@lru_cache(maxsize=32768)
def _model_distribution_cached(
    signature: tuple[
        tuple[tuple[str, int | float], ...],
        tuple[tuple[str, int | float], ...],
        tuple[tuple[str, tuple[tuple[int, float], ...]], ...],
    ],
    nominal_mass: int,
    max_isotopes: int,
) -> tuple[float, ...]:
    _, _, abundances = signature
    counts = _estimated_counts(signature, nominal_mass)
    return _brain_distribution_cached(counts, abundances, max_isotopes)


def _estimated_counts(
    signature: tuple[
        tuple[tuple[str, int | float], ...],
        tuple[tuple[str, int | float], ...],
        tuple[tuple[str, tuple[tuple[int, float], ...]], ...],
    ],
    nominal_mass: int,
) -> tuple[tuple[str, int], ...]:
    rates, fixed, _ = signature
    fixed_mass = sum(_MONOISOTOPIC_MASS[element] * int(count) for element, count in fixed)
    scalable_mass = max(0.0, nominal_mass - fixed_mass)
    composition = {element: floor(float(rate) * scalable_mass + 0.5) for element, rate in rates}
    for element, count in fixed:
        composition[element] = composition.get(element, 0) + int(count)
    return tuple(sorted((element, count) for element, count in composition.items() if count > 0))


@lru_cache(maxsize=32768)
def _adaptive_model_distribution_cached(
    signature: tuple[
        tuple[tuple[str, int | float], ...],
        tuple[tuple[str, int | float], ...],
        tuple[tuple[str, tuple[tuple[int, float], ...]], ...],
    ],
    nominal_mass: int,
    min_relative_abundance: float,
    max_isotopes: int | None,
) -> tuple[float, ...]:
    _, _, abundances = signature
    counts = _estimated_counts(signature, nominal_mass)
    patterns = _patterns_for_signature(abundances)

    mean = 0.0
    variance = 0.0
    max_offset = 0
    for element, count in counts:
        pattern = patterns[element]
        element_mean = sum(offset * abundance for offset, abundance in pattern)
        element_second = sum(offset * offset * abundance for offset, abundance in pattern)
        mean += count * element_mean
        variance += count * max(0.0, element_second - element_mean * element_mean)
        max_offset = max(max_offset, max(offset for offset, _ in pattern))

    estimated_length = max(8, ceil(mean + 8.0 * sqrt(variance) + 2 * max_offset + 4))
    length = min(estimated_length, max_isotopes) if max_isotopes is not None else estimated_length
    trailing_window = max(2, max_offset + 1)

    while True:
        distribution = np.asarray(_brain_distribution_cached(counts, abundances, length), dtype=np.float64)
        maximum = float(distribution.max())
        relative = distribution / maximum
        if max_isotopes is not None:
            break
        window = relative[-min(trailing_window, len(relative)) :]
        if int(np.argmax(relative)) < len(relative) - 1 and np.all(window < min_relative_abundance):
            break
        if length >= 4096:
            raise ArithmeticError("adaptive isotope envelope exceeded the 4096-peak safety ceiling")
        length = min(4096, length * 2)

    significant = np.flatnonzero(relative >= min_relative_abundance)
    end = int(significant[-1]) + 1 if len(significant) else int(np.argmax(relative)) + 1
    trimmed = distribution[:end].copy()
    trimmed /= float(trimmed.sum())
    return tuple(float(value) for value in trimmed)


def _rates_from_unit(composition: Mapping[str, float], unit_mass: float | None = None) -> dict[str, float]:
    mass = unit_mass
    if mass is None:
        mass = sum(_MONOISOTOPIC_MASS[element] * count for element, count in composition.items())
    return {element: count / mass for element, count in composition.items()}


# Standard peptide averagine ratios used by Peptacular, with terminal water
# represented explicitly as the fixed composition.
PEPTIDE_ISOTOPE_MODEL: Final[IsotopeModel] = IsotopeModel(
    name="peptide",
    atoms_per_da={"C": 0.044179, "H": 0.069749, "N": 0.012344, "O": 0.013352, "S": 0.0004},
    fixed_composition={"H": 2, "O": 1},
)

# Human-serum N-glycan averagose from Kronewitter et al., Proteomics 2012.
GLYCAN_ISOTOPE_MODEL: Final[IsotopeModel] = IsotopeModel(
    name="glycan",
    atoms_per_da=_rates_from_unit({"C": 6.0, "H": 9.8124, "N": 0.3733, "O": 4.3470}, 156.64662),
    fixed_composition={"H": 2, "O": 1},
)

# Equal-base average polymer residues.  Terminal water is separated from the
# repeat composition so it does not scale with oligonucleotide mass.
DNA_ISOTOPE_MODEL: Final[IsotopeModel] = IsotopeModel(
    name="dna",
    atoms_per_da=_rates_from_unit({"C": 9.75, "H": 12.25, "N": 3.75, "O": 6.0, "P": 1.0}),
    fixed_composition={"H": 2, "O": 1},
)
RNA_ISOTOPE_MODEL: Final[IsotopeModel] = IsotopeModel(
    name="rna",
    atoms_per_da=_rates_from_unit({"C": 9.5, "H": 11.75, "N": 3.75, "O": 7.0, "P": 1.0}),
    fixed_composition={"H": 2, "O": 1},
)

# A broad lipid estimate averaged from representative phosphatidylcholine,
# phosphatidylethanolamine, triacylglycerol, ceramide, sterol, and fatty-acid
# formulas.  Lipids are heterogeneous, so an exact or class-specific custom
# model is preferable when the class is known.
_LIPID_REFERENCE_FORMULAS: Final[tuple[dict[str, float], ...]] = (
    {"C": 42, "H": 82, "N": 1, "O": 8, "P": 1},  # PC 34:1
    {"C": 39, "H": 76, "N": 1, "O": 8, "P": 1},  # PE 34:1
    {"C": 55, "H": 100, "O": 6},  # TAG 52:2
    {"C": 34, "H": 67, "N": 1, "O": 3},  # ceramide 34:1
    {"C": 27, "H": 46, "O": 1},  # cholesterol
    {"C": 18, "H": 34, "O": 2},  # fatty acid 18:1
)


def _average_formula_rates(formulas: tuple[dict[str, float], ...]) -> dict[str, float]:
    rates: dict[str, float] = {}
    for formula in formulas:
        for element, rate in _rates_from_unit(formula).items():
            rates[element] = rates.get(element, 0.0) + rate / len(formulas)
    return rates


LIPID_ISOTOPE_MODEL: Final[IsotopeModel] = IsotopeModel(
    name="lipid",
    atoms_per_da=_average_formula_rates(_LIPID_REFERENCE_FORMULAS),
)

ISOTOPE_MODELS: Final[dict[IsotopeModelType, IsotopeModel]] = {
    IsotopeModelType.PEPTIDE: PEPTIDE_ISOTOPE_MODEL,
    IsotopeModelType.GLYCAN: GLYCAN_ISOTOPE_MODEL,
    IsotopeModelType.LIPID: LIPID_ISOTOPE_MODEL,
    IsotopeModelType.DNA: DNA_ISOTOPE_MODEL,
    IsotopeModelType.RNA: RNA_ISOTOPE_MODEL,
}

type IsotopeModelLike = IsotopeModel | IsotopeModelType | str


def resolve_isotope_model(model: IsotopeModelLike = IsotopeModelType.PEPTIDE) -> IsotopeModel:
    """Resolve a custom model, enum member, preset name, or supported alias."""
    if isinstance(model, IsotopeModel):
        return model
    value = str(model).lower()
    aliases = {"protein": "peptide", "averagine": "peptide", "averagose": "glycan"}
    value = aliases.get(value, value)
    try:
        return ISOTOPE_MODELS[IsotopeModelType(value)]
    except ValueError as exc:
        valid = ", ".join(item.value for item in IsotopeModelType)
        raise ValueError(f"unknown isotope model {model!r}; expected one of {valid}, or an IsotopeModel") from exc


__all__ = [
    "DNA_ISOTOPE_MODEL",
    "GLYCAN_ISOTOPE_MODEL",
    "ISOTOPE_MODELS",
    "LIPID_ISOTOPE_MODEL",
    "MAX_ISOTOPE_PEAKS",
    "NATURAL_ISOTOPE_ABUNDANCES",
    "PEPTIDE_ISOTOPE_MODEL",
    "RNA_ISOTOPE_MODEL",
    "IsotopeModel",
    "IsotopeModelLike",
    "IsotopeModelType",
    "brain_isotopic_distribution",
    "resolve_isotope_model",
]
