"""m/z recalibration from matched fragments, and the record of what was applied.

The matched fragments of a confident PSM are the only peaks in a spectrum whose true
m/z is known, so they are what a per-spectrum calibration can be fitted on; the
correction is then applied to every peak, matched or not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray
from tacular.types import ToleranceUnit

from .enums import (
    DEFAULT_FRAGMENT_TOLERANCE,
    DEFAULT_FRAGMENT_TOLERANCE_UNIT,
    check_tolerance_unit,
)
from .errors import SpxtacularError

if TYPE_CHECKING:
    from .core import Spectrum
    from .matching import MatchedFragment

CalibrationModel = Literal["offset", "linear"]
CalibrationEstimator = Literal["median", "least_squares"]

CALIBRATION_MODELS: tuple[CalibrationModel, ...] = ("offset", "linear")
CALIBRATION_ESTIMATORS: tuple[CalibrationEstimator, ...] = ("median", "least_squares")


@dataclass(frozen=True, slots=True)
class CalibrationProvenance:
    """What :func:`recalibrate` fitted, and how well it fitted.

    ``offset_ppm`` and ``slope_ppm_per_mz`` are the correction in ppm as a function
    of observed m/z: ``offset_ppm`` alone for ``model="offset"``, and
    ``offset_ppm + slope_ppm_per_mz * mz`` for ``model="linear"``.
    ``previous`` is the record of an earlier recalibration of the same spectrum, so
    the total correction stays auditable. Serialized with ``schema_version`` 1.

    Attributes
    ----------
    model, estimator
        As passed to :func:`recalibrate`.
    offset_ppm
        The fitted intercept, in ppm.
    slope_ppm_per_mz
        The fitted slope, in ppm per Th; ``None`` for ``model="offset"``.
    n_matches, n_unique_peaks
        How many matched fragments, and how many distinct peaks they occupy.
    mz_span
        ``(min, max)`` observed m/z of the matched fragments.
    residual_ppm_sd, residual_ppm_mad
        Spread of the matched fragments' signed ppm errors after the correction:
        the sample standard deviation, and the median absolute deviation from the
        median.
    tolerance, tolerance_unit
        The matching tolerance the matches came from, recorded for interpretation.
    previous
        The spectrum's prior calibration record, when it had one.
    """

    model: CalibrationModel
    estimator: CalibrationEstimator
    offset_ppm: float
    slope_ppm_per_mz: float | None
    n_matches: int
    n_unique_peaks: int
    mz_span: tuple[float, float]
    residual_ppm_sd: float
    residual_ppm_mad: float
    tolerance: float
    tolerance_unit: ToleranceUnit
    _: KW_ONLY
    previous: CalibrationProvenance | None = None

    def correction_ppm(self, mz: NDArray[np.float64] | float) -> Any:
        """Return the correction this record applies at ``mz``, in ppm."""
        slope = 0.0 if self.slope_ppm_per_mz is None else self.slope_ppm_per_mz
        return self.offset_ppm + slope * np.asarray(mz, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned JSON-compatible representation."""

        return {
            "schema_version": 1,
            "model": self.model,
            "estimator": self.estimator,
            "offset_ppm": self.offset_ppm,
            "slope_ppm_per_mz": self.slope_ppm_per_mz,
            "n_matches": self.n_matches,
            "n_unique_peaks": self.n_unique_peaks,
            "mz_span": list(self.mz_span),
            "residual_ppm_sd": self.residual_ppm_sd,
            "residual_ppm_mad": self.residual_ppm_mad,
            "tolerance": self.tolerance,
            "tolerance_unit": self.tolerance_unit,
            "previous": self.previous.to_dict() if self.previous is not None else None,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CalibrationProvenance:
        """Restore provenance written by schema version 1."""

        schema_version = value.get("schema_version", 1)
        if schema_version != 1:
            raise SpxtacularError(f"unsupported calibration provenance schema: {value.get('schema_version')!r}")
        model = _check_model(value["model"])
        estimator = _check_estimator(value["estimator"])
        span = value["mz_span"]
        if len(span) != 2:
            raise SpxtacularError(f"calibration provenance mz_span must hold two values; got {len(span)}")
        slope = value["slope_ppm_per_mz"]
        previous = value.get("previous")
        return cls(
            model=model,
            estimator=estimator,
            offset_ppm=float(value["offset_ppm"]),
            slope_ppm_per_mz=None if slope is None else float(slope),
            n_matches=int(value["n_matches"]),
            n_unique_peaks=int(value["n_unique_peaks"]),
            mz_span=(float(span[0]), float(span[1])),
            residual_ppm_sd=float(value["residual_ppm_sd"]),
            residual_ppm_mad=float(value["residual_ppm_mad"]),
            tolerance=float(value["tolerance"]),
            tolerance_unit=check_tolerance_unit(value["tolerance_unit"]),
            previous=None if previous is None else cls.from_dict(dict(previous)),
        )


def _check_model(value: Any) -> CalibrationModel:
    """Coerce a calibration model name, or raise."""
    if value in CALIBRATION_MODELS:
        return value
    valid = ", ".join(repr(name) for name in CALIBRATION_MODELS)
    raise SpxtacularError(f"unknown calibration model {value!r}; expected {valid}")


def _check_estimator(value: Any) -> CalibrationEstimator:
    """Coerce a calibration estimator name, or raise."""
    if value in CALIBRATION_ESTIMATORS:
        return value
    valid = ", ".join(repr(name) for name in CALIBRATION_ESTIMATORS)
    raise SpxtacularError(f"unknown calibration estimator {value!r}; expected {valid}")


def recalibrate(
    spectrum: Spectrum,
    matches: Sequence[MatchedFragment],
    *,
    model: CalibrationModel = "offset",
    estimator: CalibrationEstimator = "median",
    min_matches: int = 8,
    min_mz_span: float = 200.0,
    max_abs_ppm: float = 50.0,
    tolerance: float = DEFAULT_FRAGMENT_TOLERANCE,
    tolerance_unit: ToleranceUnit = DEFAULT_FRAGMENT_TOLERANCE_UNIT,
    inplace: bool = False,
) -> Spectrum:
    """Correct a spectrum's m/z axis from the mass errors of its matched fragments.

    The signed ppm error each :class:`~spxtacular.MatchedFragment` already carries is
    fitted against observed m/z, and the fitted correction is applied to **every**
    peak as ``mz / (1 + ppm(mz) * 1e-6)``. One pass only: there is no refit-and-rematch
    loop, which is where a recalibration starts agreeing with its own errors.

    Parameters
    ----------
    spectrum:
        The spectrum the matches came from. Must still carry m/z, not the neutral
        masses :meth:`~spxtacular.Spectrum.decharge` produces.
    matches:
        Matched fragments from :func:`~spxtacular.match_fragments`.
    model:
        ``"offset"`` fits one number, ``ppm = a``. ``"linear"`` fits
        ``ppm = a + b * mz``, for an instrument whose error fans out with m/z.
    estimator:
        ``"median"`` takes the median signed ppm error, which a minority of
        mis-assigned matches cannot move, and is why it is the default.
        ``"least_squares"`` takes the mean. ``model="linear"`` is fitted by least
        squares and requires ``estimator="least_squares"``.
    min_matches:
        Refuse with fewer matched fragments, or fewer distinct matched peaks, than
        this: two fragments matching one peak are one observation.
    min_mz_span:
        Refuse ``model="linear"`` when the matched m/z span is narrower than this,
        since a slope fitted over a narrow span is extrapolation across the rest of
        the spectrum. The offset model has no span requirement.
    max_abs_ppm:
        Refuse a correction larger than this anywhere in the spectrum's m/z range. A
        correction that large is a wrong peptide or the wrong tolerance unit, not a
        calibration drift.
    tolerance, tolerance_unit:
        The matching tolerance the matches came from. Recorded in the provenance for
        interpretation; it does not change the fit.
    inplace:
        Modify and return ``spectrum`` instead of a new one.

    Returns
    -------
    Spectrum
        The spectrum with corrected ``mz`` and a :class:`CalibrationProvenance` on
        its ``calibration`` attribute; ``intensity``, ``charge``, ``im`` and
        ``iso_score`` are carried over unchanged and are not shared with the input.

    Raises
    ------
    SpxtacularError
        If the spectrum is already decharged, if there are too few matches or too few
        distinct matched peaks, if ``model="linear"`` has too narrow an m/z span, if
        the correction exceeds ``max_abs_ppm`` anywhere in range, or if a fitted
        coefficient is not finite. Never a silent no-op and never a half-fit.

    Examples
    --------
    >>> import numpy as np
    >>> import peptacular as pt
    >>> from spxtacular import Spectrum, match_fragments, recalibrate
    >>> fragments = pt.fragment("PEPTIDESK", ion_types=("b", "y"), charges=(1,))
    >>> true_mz = np.array(sorted(f.mz for f in fragments))
    >>> drifted = true_mz * (1 + 4.0e-6)  # a +4 ppm instrument drift
    >>> spectrum = Spectrum(mz=drifted, intensity=np.full(len(drifted), 100.0))
    >>> matches = match_fragments(spectrum, fragments, tolerance=0.02)
    >>> fixed = recalibrate(spectrum, matches)
    >>> round(fixed.calibration.offset_ppm, 3)
    4.0
    >>> fixed.calibration.n_matches, fixed.calibration.n_unique_peaks
    (18, 18)
    >>> bool(np.allclose(fixed.mz, true_mz, rtol=1e-9))
    True
    """
    model = _check_model(model)
    estimator = _check_estimator(estimator)
    if model == "linear" and estimator != "least_squares":
        raise SpxtacularError(
            f"model='linear' is fitted by least squares; estimator={estimator!r} is not available for it. "
            "Pass estimator='least_squares', or keep the default model='offset'."
        )
    tolerance_unit = check_tolerance_unit(tolerance_unit)

    # Refusal 5: neutral masses are not m/z, and a ppm correction on them is a
    # different quantity.
    if spectrum.charge is not None and bool(np.any(spectrum.charge == 0)):
        raise SpxtacularError(
            "recalibrate() needs m/z, but the spectrum carries neutral masses "
            f"({int(np.count_nonzero(spectrum.charge == 0))} of {len(spectrum.charge)} peaks have charge 0). "
            "Recalibrate before decharge()."
        )

    # Refusal 1: with fewer points the fit is dominated by whichever ions matched.
    if len(matches) < min_matches:
        raise SpxtacularError(
            f"recalibrate() needs at least min_matches={min_matches} matched fragments; got {len(matches)}"
        )

    # Refusal 4: two fragments matching one peak are one observation.
    from .scoring import _unique_peak_indices  # local: scoring imports core, which imports this module

    n_unique_peaks = len(_unique_peak_indices(list(matches)))
    if n_unique_peaks < min_matches:
        raise SpxtacularError(
            f"recalibrate() needs at least min_matches={min_matches} distinct matched peaks; "
            f"got {n_unique_peaks} from {len(matches)} matched fragments"
        )

    observed = np.asarray([m.peak_mz for m in matches], dtype=np.float64)
    theoretical = np.asarray([m.peak_mz - m.da_error for m in matches], dtype=np.float64)
    errors = np.asarray([m.ppm_error for m in matches], dtype=np.float64)
    mz_span = (float(observed.min()), float(observed.max()))

    # Refusal 2: a slope fitted over a narrow span is extrapolation.
    span = mz_span[1] - mz_span[0]
    if model == "linear" and span < min_mz_span:
        raise SpxtacularError(
            f"recalibrate(model='linear') needs a matched m/z span of at least min_mz_span={min_mz_span}; "
            f"got {span:.4g} ({mz_span[0]:.4f} to {mz_span[1]:.4f})"
        )

    if model == "offset":
        offset_ppm = float(np.median(errors) if estimator == "median" else np.mean(errors))
        slope: float | None = None
    else:
        fitted_slope, fitted_offset = np.polyfit(observed, errors, 1)
        offset_ppm = float(fitted_offset)
        slope = float(fitted_slope)

    # Refusal 6.
    for name, value in (("offset_ppm", offset_ppm), ("slope_ppm_per_mz", slope)):
        if value is not None and not isfinite(value):
            raise SpxtacularError(f"recalibrate() fitted a non-finite {name} ({value!r}); the fit is unusable")

    # Refusal 3: evaluated at the ends of the spectrum's m/z range, since the
    # correction is linear in m/z and therefore extremal there.
    ends: tuple[float, ...] = ()
    if len(spectrum.mz):
        ends = (float(spectrum.mz.min()), float(spectrum.mz.max()))
    probe = np.asarray([*mz_span, *ends], dtype=np.float64)
    corrections = offset_ppm + (0.0 if slope is None else slope) * probe
    worst = float(np.max(np.abs(corrections)))
    if worst > max_abs_ppm:
        raise SpxtacularError(
            f"recalibrate() fitted a correction of {worst:.4g} ppm, above max_abs_ppm={max_abs_ppm}, "
            f"over m/z {float(probe.min()):.4f} to {float(probe.max()):.4f}; "
            "that is a wrong peptide or the wrong tolerance unit, not a calibration drift"
        )

    def _corrected(mz: NDArray[np.float64]) -> NDArray[np.float64]:
        factor = 1.0 + (offset_ppm + (0.0 if slope is None else slope) * mz) * 1e-6
        if mz.size and not bool(np.all(factor > 0.0)):
            raise SpxtacularError(
                f"recalibrate() fitted a correction that inverts the m/z axis (factor {float(factor.min()):.4g}); "
                "the fit is unusable"
            )
        return mz / factor

    residuals = (_corrected(observed) - theoretical) / np.abs(theoretical) * 1e6
    provenance = CalibrationProvenance(
        model=model,
        estimator=estimator,
        offset_ppm=offset_ppm,
        slope_ppm_per_mz=slope,
        n_matches=len(matches),
        n_unique_peaks=n_unique_peaks,
        mz_span=mz_span,
        residual_ppm_sd=float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0,
        residual_ppm_mad=float(np.median(np.abs(residuals - np.median(residuals)))),
        tolerance=float(tolerance),
        tolerance_unit=tolerance_unit,
        previous=spectrum.calibration,
    )
    return spectrum.update(mz=_corrected(spectrum.mz), calibration=provenance, inplace=inplace)
