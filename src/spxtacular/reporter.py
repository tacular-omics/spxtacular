"""Isobaric reporter-ion quantification: TMT, TMTpro and iTRAQ.

Reporter channels and their m/z come from tacular's :data:`~tacular.ISOBARIC_TAG_LOOKUP`
(computed from each reporter ion's isotopic composition); nothing here types in a mass.

- :func:`extract_reporter_ions` reads one spectrum into a :class:`ReporterIons`.
- :func:`reporter_ion_table` reads many spectra (any iterable, a reader view such as
  ``reader.ms2``, or a :class:`~spxtacular.Reader`) into a :class:`pandas.DataFrame`,
  one row per spectrum and one column per channel.
- :func:`isotope_correction_matrix` builds the reagent mixing matrix from a lot sheet's
  impurity table; :func:`correct_isotope_impurities` solves it with non-negative least
  squares. Both extractors take ``impurities=`` to do this for you.

Choices (all documented on the functions):

- Default tolerance 20 ppm. The closest reporter pair (the TMT/TMTpro N/C pairs,
  6.32 mDa apart) sits about 47-50 ppm apart, so +/-20 ppm windows never overlap, and it
  is the usual Orbitrap reporter tolerance. Windows that would overlap raise.
- Several peaks in a window: the most intense one is taken (ties: the lowest m/z).
- A channel with no peak has intensity ``0.0`` and observed m/z / error ``NaN``, so
  sums, ratios and the correction work while "not found" stays visible.
"""

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray
from tacular import ELEMENT_LOOKUP, ISOBARIC_TAG_LOOKUP, IsobaricTagInfo, tolerance_window
from tacular.types import ToleranceUnit

from .errors import SpxtacularError

if TYPE_CHECKING:
    import pandas as pd

    from .core import Spectrum

__all__ = [
    "DEFAULT_REPORTER_TOLERANCE",
    "DEFAULT_REPORTER_TOLERANCE_UNIT",
    "ImpurityTable",
    "ReporterIons",
    "correct_isotope_impurities",
    "extract_reporter_ions",
    "isotope_correction_matrix",
    "reporter_ion_table",
]

DEFAULT_REPORTER_TOLERANCE = 20.0
"""Default reporter m/z tolerance (with :data:`DEFAULT_REPORTER_TOLERANCE_UNIT`, 20 ppm)."""
DEFAULT_REPORTER_TOLERANCE_UNIT: ToleranceUnit = "ppm"

ReporterNormalize = Literal["sum", "max"]

ImpurityTable = Mapping[str, Mapping[str, float | None]]
"""A lot sheet's impurity table: ``{channel: {shift: percent}}``, e.g.
``{"126": {"-2": 0.0, "-1": 0.0, "+1": 7.1, "+2": 0.2}, ...}``. A
:class:`pandas.DataFrame` with channels as the index and shifts as columns works too."""

# Isotope substitution masses from tacular's element table (13C - 12C, 15N - 14N, 18O - 16O).
_ISOTOPE_SHIFT: dict[str, float] = {
    "13C": ELEMENT_LOOKUP.get_mass("13C") - ELEMENT_LOOKUP.get_mass("C"),
    "15N": ELEMENT_LOOKUP.get_mass("15N") - ELEMENT_LOOKUP.get_mass("N"),
    "18O": ELEMENT_LOOKUP.get_mass("18O") - ELEMENT_LOOKUP.get_mass("O"),
}
# An impurity lands on the channel nearest its shifted m/z, if one is within
# min(0.02 Da, half the plex's smallest channel spacing). For TMT6 and iTRAQ (channels
# ~1 Da apart) that is 0.02 Da, the nominal mapping of the lot sheets; for TMT10/11 and
# TMTpro (N/C pairs 6.32 mDa apart) it is ~3.2 mDa, so an impurity counts only on a
# channel it actually overlaps, as in OpenMS. A shift that lands 6.32 mDa beside a
# channel (e.g. TMT10 130C +13C at the 131C position) is outside that channel's picking
# window and is lost signal.
_MAX_TARGET_TOLERANCE = 0.02
_MAIN_PEAK_LABELS = frozenset({"0", "+0", "-0", "main", "reporter", "reporter ion", "monoisotopic"})
_SHIFT_TOKEN = re.compile(r"([+-])\s*(?:(\d+)\s*[xX*]\s*)?(13C|15N|18O)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Plex and parameter resolution
# ---------------------------------------------------------------------------


def _resolve_plex(plex: "str | IsobaricTagInfo") -> IsobaricTagInfo:
    if isinstance(plex, IsobaricTagInfo):
        return plex
    info = ISOBARIC_TAG_LOOKUP.query_name(plex) if isinstance(plex, str) else None
    if info is None:
        names = ", ".join(ISOBARIC_TAG_LOOKUP.keys())
        raise SpxtacularError(
            f"Unknown isobaric plex {plex!r}; expected one of {names} (or an alias such as 'TMTpro18')."
        )
    return info


def _windows(
    info: IsobaricTagInfo, tolerance: float, tolerance_unit: ToleranceUnit
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Reporter m/z and the (lo, hi) window bounds; rejects bad or overlapping windows."""
    if tolerance_unit not in ("da", "ppm"):
        raise SpxtacularError(f"tolerance_unit must be 'da' or 'ppm', got {tolerance_unit!r}.")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise SpxtacularError(f"tolerance must be a finite number >= 0, got {tolerance!r}.")
    ref = np.asarray(info.reporter_mzs, dtype=np.float64)
    bounds = np.array([tolerance_window(float(mz), float(tolerance), tolerance_unit=tolerance_unit) for mz in ref])
    lo, hi = bounds[:, 0].copy(), bounds[:, 1].copy()
    order = np.argsort(ref)
    overlap = np.flatnonzero(hi[order][:-1] >= lo[order][1:])
    if overlap.size:
        a, b = order[overlap[0]], order[overlap[0] + 1]
        sorted_ref = ref[order]
        gaps = np.diff(sorted_ref)
        if tolerance_unit == "ppm":
            limit = float(np.min(gaps / (sorted_ref[:-1] + sorted_ref[1:]))) * 1e6
        else:
            limit = float(np.min(gaps)) / 2
        raise SpxtacularError(
            f"{info.name} channels {info.channels[a]} and {info.channels[b]} are "
            f"{ref[b] - ref[a]:.5f} Da apart; a {tolerance:g} {tolerance_unit} tolerance makes their windows "
            f"overlap. Use a tolerance below {limit:.4g} {tolerance_unit}."
        )
    return ref, lo, hi


def _target_tolerance(ref: NDArray[np.float64]) -> float:
    """How close a shifted impurity must land to a channel to count as that channel."""
    gaps = np.diff(np.sort(ref))
    return min(_MAX_TARGET_TOLERANCE, float(gaps.min()) / 2) if gaps.size else _MAX_TARGET_TOLERANCE


def _check_normalize(normalize: object) -> None:
    if normalize is not None and normalize not in ("sum", "max"):
        raise SpxtacularError(f"normalize must be 'sum', 'max' or None, got {normalize!r}.")


def _normalize_rows(values: NDArray[np.float64], normalize: ReporterNormalize | None) -> NDArray[np.float64]:
    if normalize is None:
        return values
    scale = (
        values.sum(axis=-1, keepdims=True) if normalize == "sum" else values.max(axis=-1, keepdims=True, initial=0.0)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(scale > 0, values / np.where(scale > 0, scale, 1.0), 0.0)


# ---------------------------------------------------------------------------
# Isotope impurity correction
# ---------------------------------------------------------------------------


def _shift_mass(shift: object) -> float:
    """Mass offset of a lot-sheet column label.

    Numbers and ``"-2"``/``"-1"``/``"+1"``/``"+2"`` are that many 13C, never 15N;
    labels like ``"-13C"``, ``"+2x13C"``, ``"-15N"`` or ``"-13C-15N"`` name the
    substitutions explicitly. A main-peak column (``"0"``, ``"main"``, ``"reporter"``,
    ``"monoisotopic"``) is 0.0 and is ignored by the caller.
    """
    if isinstance(shift, bool):
        raise SpxtacularError(f"Unrecognised impurity shift {shift!r}.")
    if isinstance(shift, int | float):
        return float(shift) * _ISOTOPE_SHIFT["13C"]
    if not isinstance(shift, str):
        raise SpxtacularError(f"Unrecognised impurity shift {shift!r}.")
    text = shift.strip()
    if text.lower() in _MAIN_PEAK_LABELS:
        return 0.0
    try:
        return float(text) * _ISOTOPE_SHIFT["13C"]
    except ValueError:
        pass
    total, pos = 0.0, 0
    compact = text.replace(" ", "")
    for match in _SHIFT_TOKEN.finditer(compact):
        if match.start() != pos:
            break
        sign = -1.0 if match.group(1) == "-" else 1.0
        count = int(match.group(2) or 1)
        total += sign * count * _ISOTOPE_SHIFT[match.group(3).upper()]
        pos = match.end()
    if pos == 0 or pos != len(compact):
        raise SpxtacularError(
            f"Unrecognised impurity shift {shift!r}; use -2/-1/+1/+2 (13C) or labels like '-13C', '+2x13C', '-13C-15N'."
        )
    return total


def _impurity_rows(impurities: Any) -> dict[str, dict[Any, Any]]:
    if hasattr(impurities, "to_dict") and hasattr(impurities, "index"):  # pandas DataFrame
        return {str(k): dict(v) for k, v in impurities.to_dict(orient="index").items()}
    if isinstance(impurities, Mapping):
        rows: dict[str, dict[Any, Any]] = {}
        for channel, row in impurities.items():
            if not isinstance(row, Mapping):
                raise SpxtacularError(f"Impurities for channel {channel!r} must be a mapping of shift -> percent.")
            rows[str(channel)] = dict(row)
        return rows
    raise SpxtacularError("impurities must be a {channel: {shift: percent}} mapping, a DataFrame or a square matrix.")


def isotope_correction_matrix(
    plex: "str | IsobaricTagInfo", impurities: "ImpurityTable | pd.DataFrame"
) -> NDArray[np.float64]:
    """Mixing matrix ``M`` from a reagent lot sheet: ``observed = M @ true``.

    ``M[i, j]`` is the fraction of reagent ``j``'s reporter signal that appears in
    channel ``i`` (channels in :attr:`IsobaricTagInfo.channels` order).

    Parameters
    ----------
    plex:
        Plex name or alias (``"TMT10"``, ``"TMTpro18"``, ``"iTRAQ4"``, ...) or a tacular
        :class:`~tacular.IsobaricTagInfo`.
    impurities:
        Per-channel impurity percentages, the shape of a TMT/iTRAQ lot sheet:
        ``{"126": {"-2": 0, "-1": 0, "+1": 7.1, "+2": 0.2}, ...}`` or a DataFrame with
        channels as the index and shifts as columns. ``None``/NaN cells are 0. Channels
        left out are taken as pure.

        Column labels are either numbers (``-2``, ``-1``, ``+1``, ``+2``), which always
        mean that many **13C**, or explicit substitutions (``"-13C"``, ``"+2x13C"``,
        ``"-15N"``, ``"+13C+15N"``, ``"-18O"``). For TMT10/11 and TMTpro N channels a
        -1 impurity from a missing 15N is ``"-15N"``, not ``-1``: newer Thermo sheets
        that list 15N and 13C separately need the explicit labels, or those impurities
        are placed at the wrong m/z. A main-peak column (a zero shift, ``"0"``,
        ``"main"``, ``"reporter"``, ``"monoisotopic"``) is ignored: the main peak is
        always ``100 - sum(impurities)``.

    Returns
    -------
    numpy.ndarray
        Square ``(plex, plex)`` matrix. A reagent keeps ``100 - sum(its impurities)``
        percent in its own channel (the OpenMS convention). Each impurity is assigned to
        the channel nearest the shifted reporter m/z if it lies within
        ``min(0.02 Da, half the plex's smallest channel spacing)``: 0.02 Da for TMT6 and
        iTRAQ (nominal mapping), about 3.2 mDa for TMT10/11 and TMTpro, whose N/C pairs
        are 6.32 mDa apart. So a -1 (13C) of TMT 128C goes to 127C and of 128N to 127N,
        while a -1 (13C) of 127N or a +1 of TMT10's 130C (which lands on the 131C
        position) matches no channel. An impurity that matches no channel of this plex
        is lost signal and only lowers the diagonal, as in OpenMS.

    Raises
    ------
    SpxtacularError
        Unknown plex, channel or shift label; a negative or non-numeric percentage; or
        impurities that sum to more than 100 % for one channel.
    """
    info = _resolve_plex(plex)
    ref = np.asarray(info.reporter_mzs, dtype=np.float64)
    n = len(ref)
    target_tolerance = _target_tolerance(ref)
    matrix = np.eye(n)
    for channel, row in _impurity_rows(impurities).items():
        ion = info.query_reporter(channel)
        if ion is None:
            raise SpxtacularError(f"Channel {channel!r} is not in {info.name} (channels: {', '.join(info.channels)}).")
        j = info.channels.index(ion.channel)
        total = 0.0
        for shift, value in row.items():
            percent = _percent(value, channel, shift)
            shift_mass = _shift_mass(shift)  # rejects a bad label even when its value is 0
            if percent == 0.0 or shift_mass == 0.0:  # a main-peak column is not an impurity
                continue
            total += percent
            distance = np.abs(ref - (ref[j] + shift_mass))
            i = int(np.argmin(distance))
            if i != j and distance[i] <= target_tolerance:
                matrix[i, j] += percent / 100.0
        if total > 100.0:
            raise SpxtacularError(f"Impurities for channel {channel!r} sum to {total:g} %, more than 100 %.")
        matrix[j, j] = 1.0 - total / 100.0
    return matrix


def _percent(value: object, channel: str, shift: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, int | float | np.number):
        raise SpxtacularError(f"Impurity for channel {channel!r}, shift {shift!r} must be a number, got {value!r}.")
    number = float(value)
    if math.isnan(number):
        return 0.0
    if not math.isfinite(number) or number < 0:
        raise SpxtacularError(f"Impurity for channel {channel!r}, shift {shift!r} must be >= 0, got {value!r}.")
    return number


def _nnls(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Lawson-Hanson non-negative least squares: argmin ||a x - b|| subject to x >= 0."""
    n = a.shape[1]
    ata, atb = a.T @ a, a.T @ b
    tol = 10 * np.finfo(np.float64).eps * float(np.abs(a).sum(axis=0).max(initial=1.0)) * max(a.shape)
    x = np.zeros(n)
    passive = np.zeros(n, dtype=bool)
    w = atb - ata @ x
    for _ in range(3 * n + 3):
        if passive.all() or not np.any(w[~passive] > tol):
            break
        passive[int(np.argmax(np.where(passive, -np.inf, w)))] = True
        for _ in range(3 * n + 3):
            idx = np.flatnonzero(passive)
            s = np.zeros(n)
            s[idx] = np.linalg.lstsq(a[:, idx], b, rcond=None)[0]
            if np.all(s[idx] > 0):
                x = s
                break
            blocking = passive & (s <= 0)
            alpha = float(np.min(x[blocking] / (x[blocking] - s[blocking])))
            x = x + alpha * (s - x)
            passive &= x > tol
            x[~passive] = 0.0
        w = atb - ata @ x
    return x


def correct_isotope_impurities(
    intensities: "NDArray[np.float64] | Iterable[float] | Iterable[Iterable[float]]",
    correction: "NDArray[np.float64] | ImpurityTable | pd.DataFrame",
    *,
    plex: "str | IsobaricTagInfo | None" = None,
) -> NDArray[np.float64]:
    """Undo reagent isotope impurities: solve ``observed = M @ true`` for ``true >= 0``.

    Parameters
    ----------
    intensities:
        Observed reporter intensities, one spectrum ``(plex,)`` or many ``(n, plex)``,
        channels in plex order.
    correction:
        The mixing matrix ``M`` (square, e.g. from :func:`isotope_correction_matrix`) or a
        lot-sheet impurity table, which then needs ``plex``.
    plex:
        Plex name or :class:`~tacular.IsobaricTagInfo`; only needed for an impurity table.

    Returns
    -------
    numpy.ndarray
        Corrected intensities, same shape as ``intensities``, never negative.

    Notes
    -----
    Each spectrum is first solved exactly (``numpy.linalg.solve``). Noise can make that
    solution slightly negative in a channel next to a strong one; those spectra are
    re-solved by non-negative least squares (a small Lawson-Hanson NNLS in numpy, since
    scipy is not a dependency). That is the exact answer whenever it is already
    non-negative, and otherwise the closest non-negative fit, which, unlike clipping the
    negatives to zero, redistributes the remainder to the neighbouring channels.
    """
    observed = np.asarray(intensities, dtype=np.float64)
    single = observed.ndim == 1
    rows = np.atleast_2d(observed)
    matrix = _as_matrix(correction, plex, rows.shape[1])
    if rows.ndim != 2:
        raise SpxtacularError(f"intensities must be 1-D or 2-D, got shape {observed.shape}.")
    try:
        solved = np.linalg.solve(matrix, rows.T).T
    except np.linalg.LinAlgError as exc:
        raise SpxtacularError("The isotope correction matrix is singular.") from exc
    for r in np.flatnonzero(np.any(solved < 0, axis=1)):
        solved[r] = _nnls(matrix, rows[r])
    solved = np.maximum(solved, 0.0)
    return solved[0] if single else solved


def _as_matrix(correction: Any, plex: "str | IsobaricTagInfo | None", n: int) -> NDArray[np.float64]:
    if isinstance(correction, np.ndarray) or (
        isinstance(correction, list | tuple) and correction and not isinstance(correction[0], str)
    ):
        matrix = np.asarray(correction, dtype=np.float64)
    else:
        if plex is None:
            raise SpxtacularError("An impurity table needs plex= to know the channels; or pass a correction matrix.")
        matrix = isotope_correction_matrix(plex, correction)
    if matrix.shape != (n, n):
        raise SpxtacularError(f"Correction matrix has shape {matrix.shape}; expected ({n}, {n}) for this plex.")
    if not np.all(np.isfinite(matrix)):
        raise SpxtacularError("Correction matrix contains NaN or infinite values.")
    return matrix


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReporterIons:
    """Reporter-ion intensities of one spectrum for one plex.

    Arrays are read-only and in :attr:`channels` order. A channel with no peak in its
    window has ``intensity`` 0.0 and ``observed_mz`` NaN.
    """

    plex: str
    """Plex name from tacular, e.g. ``"TMT10"``, ``"TMT18"``, ``"iTRAQ4"``."""
    channels: tuple[str, ...]
    """Channel names, e.g. ``("126", "127N", ...)``."""
    reporter_mz: NDArray[np.float64]
    """Theoretical reporter m/z."""
    intensity: NDArray[np.float64]
    """Reported intensities: :attr:`raw_intensity` after impurity correction and
    normalization, when requested."""
    raw_intensity: NDArray[np.float64]
    """Intensity of the picked peak (0.0 when none), before correction/normalization."""
    observed_mz: NDArray[np.float64]
    """m/z of the picked peak, NaN when none."""
    corrected: bool = False
    """Whether an isotope impurity correction was applied."""
    normalize: ReporterNormalize | None = None
    """The normalization applied (``"sum"``, ``"max"``) or ``None``."""

    def __post_init__(self) -> None:
        """Freeze the arrays."""
        for name in ("reporter_mz", "intensity", "raw_intensity", "observed_mz"):
            array = np.array(getattr(self, name), dtype=np.float64)
            array.flags.writeable = False
            object.__setattr__(self, name, array)

    @property
    def found(self) -> NDArray[np.bool_]:
        """Whether a peak was found in each channel's window."""
        return ~np.isnan(self.observed_mz)

    @property
    def mz_error(self) -> NDArray[np.float64]:
        """Observed minus theoretical m/z in Da (NaN when not found)."""
        return self.observed_mz - self.reporter_mz

    @property
    def ppm_error(self) -> NDArray[np.float64]:
        """Observed minus theoretical m/z in ppm (NaN when not found)."""
        return self.mz_error / self.reporter_mz * 1e6

    def __len__(self) -> int:
        """Number of channels."""
        return len(self.channels)

    def __getitem__(self, channel: str) -> float:
        """Intensity of ``channel`` (case-insensitive, e.g. ``"127n"``)."""
        key = channel.strip().upper() if isinstance(channel, str) else channel
        if key not in self.channels:
            raise KeyError(channel)
        return float(self.intensity[self.channels.index(key)])

    def to_dict(self) -> dict[str, Any]:
        """Plain, JSON-serializable dict; NaN becomes ``None``."""

        def _list(array: NDArray[np.float64]) -> list[float | None]:
            return [None if math.isnan(v) else float(v) for v in array]

        return {
            "plex": self.plex,
            "channels": list(self.channels),
            "reporter_mz": _list(self.reporter_mz),
            "intensity": _list(self.intensity),
            "raw_intensity": _list(self.raw_intensity),
            "observed_mz": _list(self.observed_mz),
            "corrected": self.corrected,
            "normalize": self.normalize,
        }


def _pick(
    spectrum: "Spectrum", lo: NDArray[np.float64], hi: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Most intense peak in each window (ties: lowest m/z); (intensity, observed m/z)."""
    if spectrum.is_decharged:
        raise SpxtacularError("Reporter ions need m/z values; this spectrum is decharged to neutral masses.")
    mz, intensity = spectrum.mz, spectrum.intensity
    if not np.isfinite(intensity).all():
        raise SpxtacularError("Reporter ions need finite intensities; this spectrum has NaN or inf intensities.")
    order = np.argsort(mz, kind="stable")  # m/z is not assumed sorted (timsTOF)
    smz, sint = mz[order], intensity[order]
    left = np.searchsorted(smz, lo, side="left")
    right = np.searchsorted(smz, hi, side="right")
    found_int = np.zeros(len(lo))
    found_mz = np.full(len(lo), np.nan)
    for k, (a, b) in enumerate(zip(left, right, strict=True)):
        if b > a:
            best = a + int(np.argmax(sint[a:b]))
            found_int[k] = sint[best]
            found_mz[k] = smz[best]
    return found_int, found_mz


def extract_reporter_ions(
    spectrum: "Spectrum",
    plex: "str | IsobaricTagInfo",
    *,
    tolerance: float = DEFAULT_REPORTER_TOLERANCE,
    tolerance_unit: ToleranceUnit = DEFAULT_REPORTER_TOLERANCE_UNIT,
    impurities: "ImpurityTable | pd.DataFrame | NDArray[np.float64] | None" = None,
    normalize: ReporterNormalize | None = None,
) -> ReporterIons:
    """Reporter-ion intensities of one (centroided) spectrum.

    Parameters
    ----------
    spectrum:
        A centroided :class:`~spxtacular.Spectrum` in m/z (not decharged).
    plex:
        Plex name or alias from tacular's ``ISOBARIC_TAG_LOOKUP``, case-insensitive:
        ``TMT0``, ``TMT2``, ``TMT6``, ``TMT10``, ``TMT11``, ``TMTpro0``, ``TMT16``/``TMTpro16``,
        ``TMT18``/``TMTpro18``, ``iTRAQ4``, ``iTRAQ8``; or an :class:`~tacular.IsobaricTagInfo`.
    tolerance, tolerance_unit:
        Half-width of each channel's window around the reporter m/z, in ``"ppm"``
        (default 20) or ``"da"``. Windows that would overlap raise.
    impurities:
        Optional isotope impurity correction: a lot-sheet table (see
        :func:`isotope_correction_matrix`) or a precomputed square mixing matrix.
    normalize:
        ``"sum"`` divides by the channel sum, ``"max"`` by the largest channel (after
        correction); a spectrum with no reporter signal stays all zero.

    Returns
    -------
    ReporterIons
        In each window the most intense peak is taken (ties: lowest m/z). Missing
        channels have intensity 0.0 and observed m/z NaN.

    Raises
    ------
    SpxtacularError
        Unknown plex, bad tolerance or unit, overlapping windows, a bad impurity table, a
        decharged spectrum, or NaN/inf intensities.
    """
    info = _resolve_plex(plex)
    ref, lo, hi = _windows(info, tolerance, tolerance_unit)
    _check_normalize(normalize)
    matrix = None if impurities is None else _as_matrix(impurities, info, len(ref))
    raw, observed = _pick(spectrum, lo, hi)
    values = raw if matrix is None else correct_isotope_impurities(raw, matrix)
    return ReporterIons(
        plex=info.name,
        channels=info.channels,
        reporter_mz=ref,
        intensity=_normalize_rows(values, normalize),
        raw_intensity=raw,
        observed_mz=observed,
        corrected=matrix is not None,
        normalize=normalize,
    )


def reporter_ion_table(
    spectra: "Iterable[Spectrum] | Any",
    plex: "str | IsobaricTagInfo",
    *,
    tolerance: float = DEFAULT_REPORTER_TOLERANCE,
    tolerance_unit: ToleranceUnit = DEFAULT_REPORTER_TOLERANCE_UNIT,
    impurities: "ImpurityTable | pd.DataFrame | NDArray[np.float64] | None" = None,
    normalize: ReporterNormalize | None = None,
    ms_level: int | None = None,
    include_errors: bool = False,
) -> "pd.DataFrame":
    """Reporter-ion intensities of many spectra as a DataFrame, one row per spectrum.

    Parameters
    ----------
    spectra:
        Any iterable of spectra (a list, ``reader.ms2``, a generator). An object with an
        ``ms2`` attribute (:class:`~spxtacular.Reader` and the format readers) is always
        read through ``spectra.ms2``, even if it is itself iterable, so passing a reader
        gives its MS2 spectra, not every level. For SPS-MS3 data pass an iterable of the MS3
        spectra (e.g. ``iter(reader)``, which has no ``ms2`` attribute) and ``ms_level=3``.
    plex, tolerance, tolerance_unit, impurities, normalize:
        As in :func:`extract_reporter_ions`; correction and normalization are applied to
        every row at once.
    ms_level:
        Keep only :class:`~spxtacular.MsnSpectrum` rows with this ``ms_level``; ``None``
        keeps every spectrum.
    include_errors:
        Add a ``<channel>_ppm_error`` column per channel (NaN when not found).

    Returns
    -------
    pandas.DataFrame
        Columns ``spectrum_index`` (position in ``spectra``), ``scan_number``,
        ``native_id``, ``ms_level``, ``rt`` (``None`` for plain :class:`~spxtacular.Spectrum`),
        then one float column per channel named as in tacular (``"126"``, ``"127N"``, ...),
        then the error columns if requested.
    """
    import pandas as pd

    info = _resolve_plex(plex)
    ref, lo, hi = _windows(info, tolerance, tolerance_unit)
    _check_normalize(normalize)
    matrix = None if impurities is None else _as_matrix(impurities, info, len(ref))
    source: Iterable[Any] = getattr(spectra, "ms2", spectra)  # a reader: always its MS2 view

    meta: list[dict[str, Any]] = []
    raw_rows: list[NDArray[np.float64]] = []
    mz_rows: list[NDArray[np.float64]] = []
    for position, spectrum in enumerate(source):
        level = getattr(spectrum, "ms_level", None)
        if ms_level is not None and level != ms_level:
            continue
        raw, observed = _pick(spectrum, lo, hi)
        raw_rows.append(raw)
        mz_rows.append(observed)
        meta.append(
            {
                "spectrum_index": position,
                "scan_number": getattr(spectrum, "scan_number", None),
                "native_id": getattr(spectrum, "native_id", None),
                "ms_level": level,
                "rt": getattr(spectrum, "rt", None),
            }
        )

    n = len(ref)
    raw_all = np.vstack(raw_rows) if raw_rows else np.zeros((0, n))
    mz_all = np.vstack(mz_rows) if mz_rows else np.zeros((0, n))
    values = raw_all if matrix is None or not len(raw_all) else correct_isotope_impurities(raw_all, matrix)
    values = _normalize_rows(values, normalize)

    table = pd.DataFrame(meta, columns=["spectrum_index", "scan_number", "native_id", "ms_level", "rt"])
    channel_frame = pd.DataFrame(values, columns=list(info.channels), index=table.index)
    frames = [table, channel_frame]
    if include_errors:
        errors = (mz_all - ref) / ref * 1e6
        frames.append(pd.DataFrame(errors, columns=[f"{c}_ppm_error" for c in info.channels], index=table.index))
    return pd.concat(frames, axis=1)
