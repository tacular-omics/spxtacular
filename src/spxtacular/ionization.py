"""Ionization models for charged-ion and neutral-mass conversion."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np
import peptacular as pt
from numpy.typing import ArrayLike, NDArray

from .enums import Polarity

# Monoisotopic ion masses. Sodium is the neutral atom less one electron;
# ammonium is 14N + 4(1H) less one electron.
SODIUM_CATION_MASS = 22.9897692820 - pt.ELECTRON_MASS
AMMONIUM_CATION_MASS = 14.00307400443 + 4 * 1.00782503223 - pt.ELECTRON_MASS


@dataclass(frozen=True, slots=True)
class IonizationModel:
    """A repeated charge carrier used to form an observed ion.

    ``carrier_mass`` is the signed ion-mass delta per unit of positive charge
    magnitude. Attachment is positive; loss is negative. Per-peak charge arrays
    remain positive magnitudes regardless of polarity.
    """

    name: str
    polarity: Polarity | str
    carrier_mass: float
    carrier: str = "custom"

    def __post_init__(self) -> None:
        try:
            polarity = Polarity(str(self.polarity).lower())
        except ValueError as exc:
            raise ValueError(f"polarity must be 'positive' or 'negative', got {self.polarity!r}") from exc
        mass = float(self.carrier_mass)
        if not isfinite(mass):
            raise ValueError(f"carrier_mass must be finite, got {self.carrier_mass!r}")
        if not str(self.name).strip():
            raise ValueError("ionization model name cannot be empty")
        object.__setattr__(self, "polarity", polarity)
        object.__setattr__(self, "carrier_mass", mass)

    def ion_mz(self, neutral_mass: ArrayLike, charge: ArrayLike) -> float | NDArray[np.float64]:
        """Convert neutral mass to m/z for positive charge magnitudes."""
        mass, z, scalar = _validated_inputs(neutral_mass, charge, value_name="neutral_mass")
        result = (mass + z * self.carrier_mass) / z
        if np.any(result <= 0.0):
            raise ValueError("ionization model produced a non-positive m/z")
        return float(result) if scalar else result

    def neutral_mass(self, mz: ArrayLike, charge: ArrayLike) -> float | NDArray[np.float64]:
        """Convert m/z to neutral mass for positive charge magnitudes."""
        observed, z, scalar = _validated_inputs(mz, charge, value_name="mz")
        result = observed * z - z * self.carrier_mass
        if np.any(result < 0.0):
            raise ValueError("ionization model produced a negative neutral mass")
        return float(result) if scalar else result

    def notation(self, charge: int = 1) -> str:
        """Return charge-aware adduct notation, such as ``[M+2Na]2+``."""
        z = _validated_charge_scalar(charge)
        sign = "+" if self.carrier_mass >= 0 else "-"
        suffix = "+" if self.polarity == Polarity.POSITIVE else "-"
        count = "" if z == 1 else str(z)
        charge_suffix = suffix if z == 1 else f"{z}{suffix}"
        return f"[M{sign}{count}{self.carrier}]{charge_suffix}"

    def to_dict(self) -> dict[str, str | float]:
        """Return a JSON-serializable value description."""
        return {
            "name": self.name,
            "polarity": self.polarity.value,
            "carrier_mass": self.carrier_mass,
            "carrier": self.carrier,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> IonizationModel:
        """Reconstruct a model from :meth:`to_dict` output."""
        return cls(
            name=str(value["name"]),
            polarity=str(value["polarity"]),
            carrier_mass=float(value["carrier_mass"]),
            carrier=str(value.get("carrier", "custom")),
        )


def _validated_inputs(value: ArrayLike, charge: ArrayLike, *, value_name: str):
    values = np.asarray(value, dtype=np.float64)
    charges = np.asarray(charge)
    scalar = values.ndim == 0 and charges.ndim == 0
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{value_name} must be finite and non-negative")
    if np.any(charges != np.floor(charges)) or np.any(charges <= 0):
        raise ValueError("charge must contain positive integer magnitudes")
    return values, charges.astype(np.float64, copy=False), scalar


def _validated_charge_scalar(charge: int) -> int:
    if isinstance(charge, bool) or int(charge) != charge or charge < 1:
        raise ValueError(f"charge must be a positive integer, got {charge!r}")
    return int(charge)


PROTONATED = IonizationModel("protonated", Polarity.POSITIVE, pt.PROTON_MASS, "H")
DEPROTONATED = IonizationModel("deprotonated", Polarity.NEGATIVE, -pt.PROTON_MASS, "H")
SODIATED = IonizationModel("sodiated", Polarity.POSITIVE, SODIUM_CATION_MASS, "Na")
AMMONIATED = IonizationModel("ammoniated", Polarity.POSITIVE, AMMONIUM_CATION_MASS, "NH4")

IONIZATION_MODELS: dict[str, IonizationModel] = {
    "protonated": PROTONATED,
    "deprotonated": DEPROTONATED,
    "sodiated": SODIATED,
    "ammoniated": AMMONIATED,
}

_ALIASES = {
    "[m+h]+": "protonated",
    "[m-h]-": "deprotonated",
    "[m+na]+": "sodiated",
    "[m+nh4]+": "ammoniated",
    "h+": "protonated",
    "h-": "deprotonated",
    "na+": "sodiated",
    "nh4+": "ammoniated",
}

type IonizationModelLike = IonizationModel | str | float


def resolve_ionization_model(model: IonizationModelLike = PROTONATED) -> IonizationModel:
    """Resolve a model instance, preset/alias string, or custom carrier mass."""
    if isinstance(model, IonizationModel):
        return model
    if isinstance(model, (int, float)) and not isinstance(model, bool):
        mass = float(model)
        polarity = Polarity.POSITIVE if mass >= 0.0 else Polarity.NEGATIVE
        return IonizationModel("custom", polarity, mass)
    value = str(model).strip().lower().replace(" ", "")
    value = _ALIASES.get(value, value)
    try:
        return IONIZATION_MODELS[value]
    except KeyError as exc:
        valid = ", ".join(IONIZATION_MODELS)
        raise ValueError(
            f"unknown ionization model {model!r}; expected {valid}, an adduct alias, or a custom model"
        ) from exc


@dataclass(frozen=True, slots=True)
class DeconvolutionProvenance:
    """Models and parameters that determine deconvolution interpretation."""

    isotope_model: str
    ionization_model: IonizationModel
    charge_range: tuple[int, int]
    tolerance: float
    tolerance_type: str
    intensity_mode: str
    min_intensity: float
    min_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "isotope_model": self.isotope_model,
            "ionization_model": self.ionization_model.to_dict(),
            "charge_range": list(self.charge_range),
            "tolerance": self.tolerance,
            "tolerance_type": self.tolerance_type,
            "intensity_mode": self.intensity_mode,
            "min_intensity": self.min_intensity,
            "min_score": self.min_score,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DeconvolutionProvenance:
        if value.get("schema_version", 1) != 1:
            raise ValueError(f"unsupported deconvolution provenance schema: {value.get('schema_version')!r}")
        charge_range = value["charge_range"]
        return cls(
            isotope_model=str(value["isotope_model"]),
            ionization_model=IonizationModel.from_dict(value["ionization_model"]),
            charge_range=(int(charge_range[0]), int(charge_range[1])),
            tolerance=float(value["tolerance"]),
            tolerance_type=str(value["tolerance_type"]),
            intensity_mode=str(value["intensity_mode"]),
            min_intensity=float(value["min_intensity"]),
            min_score=float(value["min_score"]),
        )


__all__ = [
    "AMMONIATED",
    "AMMONIUM_CATION_MASS",
    "DEPROTONATED",
    "IONIZATION_MODELS",
    "PROTONATED",
    "SODIATED",
    "SODIUM_CATION_MASS",
    "DeconvolutionProvenance",
    "IonizationModel",
    "IonizationModelLike",
    "resolve_ionization_model",
]
