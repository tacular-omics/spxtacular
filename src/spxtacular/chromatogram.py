"""
Run-level extraction: chromatograms and extracted ion chromatograms.

Everything else in the library works on one spectrum. These functions work on a
*run* -- any iterable of spectra carrying retention times, which is exactly what
``reader.ms1`` yields.

    with spx.Reader("run.d") as reader:
        tic = extract_chromatogram(reader.ms1)
        xics = extract_xic(reader.ms1, [500.2649, 622.0290], tolerance=20)

Two properties shape the design:

**One pass.** A reader is expensive to iterate -- a 65-frame timsTOF run takes
several seconds to load -- and ``reader.ms1`` may be a generator that cannot be
replayed. So every function here consumes the iterable exactly once, and
``extract_xic`` takes a *list* of targets rather than one, so extracting twenty
traces costs one pass rather than twenty.

**Any m/z order.** timsTOF frames are ordered by ion-mobility scan and are only
sorted by m/z *within* each scan, so a ``DReader`` MS1 frame is not globally
sorted. Each frame is sorted once on arrival when needed, after which every
target is a binary search rather than a full scan -- which is what makes many
targets cheap.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import NDArray

from .core import Spectrum
from .enums import ToleranceLike, ToleranceType
from .serialization import (
    CHROMATOGRAM_SCHEMA,
    JSON_SCHEMA_VERSION,
    require_exact_keys,
    require_mapping,
    require_number,
    require_number_array_or_none,
    require_schema,
    require_string,
    strict_json_dumps,
    strict_json_loads,
    to_json_value,
)

Aggregate = Literal["sum", "max"]


@dataclass(slots=True)
class Chromatogram:
    """Intensity against retention time.

    Attributes
    ----------
    rt:
        Retention times, seconds, ascending.
    intensity:
        One value per retention time.
    label:
        Short name for the legend, e.g. ``"TIC"`` or ``"m/z 500.2649"``.
    mz:
        Target m/z, for an extracted ion chromatogram. ``None`` for a TIC/BPC.
    tolerance, tolerance_type:
        The extraction window, kept so a figure can say what it plotted.
    """

    rt: NDArray[np.float64]
    intensity: NDArray[np.float64]
    label: str = ""
    mz: float | None = None
    tolerance: float | None = None
    tolerance_type: str | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce array inputs and validate the one-row-per-time-point model."""
        self.rt = np.asarray(self.rt, dtype=np.float64)
        self.intensity = np.asarray(self.intensity, dtype=np.float64)
        if self.rt.ndim != 1:
            raise ValueError(f"rt array must be one-dimensional; got shape {self.rt.shape}")
        if self.intensity.ndim != 1:
            raise ValueError(f"intensity array must be one-dimensional; got shape {self.intensity.shape}")
        if len(self.rt) != len(self.intensity):
            raise ValueError("rt and intensity must have the same length")

    def __len__(self) -> int:
        return len(self.rt)

    def to_dict(self) -> dict[str, Any]:
        """Return a versioned, JSON-compatible representation."""
        return {
            "schema": CHROMATOGRAM_SCHEMA,
            "schema_version": JSON_SCHEMA_VERSION,
            "kind": "chromatogram",
            "arrays": to_json_value({"rt": self.rt, "intensity": self.intensity}, "arrays"),
            "metadata": to_json_value(
                {
                    "label": self.label,
                    "mz": self.mz,
                    "tolerance": self.tolerance,
                    "tolerance_type": self.tolerance_type,
                    "meta": self.meta,
                },
                "metadata",
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Reconstruct a chromatogram from :meth:`to_dict` output."""
        data = require_mapping(payload, "payload")
        require_schema(data, CHROMATOGRAM_SCHEMA, {"chromatogram"})

        arrays = require_mapping(data["arrays"], "payload.arrays")
        require_exact_keys(arrays, {"rt", "intensity"}, "payload.arrays")
        rt = require_number_array_or_none(arrays["rt"], "payload.arrays.rt")
        intensity = require_number_array_or_none(arrays["intensity"], "payload.arrays.intensity")
        if rt is None or intensity is None:
            raise ValueError("payload.arrays.rt and payload.arrays.intensity cannot be null")

        metadata = require_mapping(data["metadata"], "payload.metadata")
        require_exact_keys(
            metadata,
            {"label", "mz", "tolerance", "tolerance_type", "meta"},
            "payload.metadata",
        )
        normalized_metadata = to_json_value(metadata, "payload.metadata")
        meta = require_mapping(normalized_metadata["meta"], "payload.metadata.meta")

        return cls(
            rt=np.asarray(rt, dtype=np.float64),
            intensity=np.asarray(intensity, dtype=np.float64),
            label=require_string(normalized_metadata["label"], "payload.metadata.label"),
            mz=require_number(normalized_metadata["mz"], "payload.metadata.mz", nullable=True),
            tolerance=require_number(normalized_metadata["tolerance"], "payload.metadata.tolerance", nullable=True),
            tolerance_type=require_string(
                normalized_metadata["tolerance_type"], "payload.metadata.tolerance_type", nullable=True
            ),
            meta=dict(meta),
        )

    def to_json(self, *, indent: int | None = None) -> str:
        """Encode :meth:`to_dict` output as standards-compliant JSON."""
        return strict_json_dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> Self:
        """Reconstruct a chromatogram from a JSON string or UTF-8 byte sequence."""
        payload = require_mapping(strict_json_loads(value), "payload")
        return cls.from_dict(payload)

    @property
    def apex_rt(self) -> float | None:
        """Retention time of the most intense point, or ``None`` if empty."""
        if len(self.rt) == 0:
            return None
        return float(self.rt[int(np.argmax(self.intensity))])

    @property
    def total(self) -> float:
        """Summed intensity across the trace -- the usual peak-area proxy."""
        return float(self.intensity.sum())


def _rt_of(spectrum: Spectrum, index: int) -> float:
    """Retention time, falling back to the scan index when the reader gave none."""
    rt = getattr(spectrum, "rt", None)
    return float(rt) if rt is not None else float(index)


def _sorted_view(mz: NDArray[np.float64], *arrays: NDArray[np.float64] | None):
    """Return ``mz`` ascending plus the same permutation applied to ``arrays``.

    Sorting once per spectrum turns every subsequent target lookup into a binary
    search. On unsorted input (any timsTOF frame) the alternative is a full scan
    per target, which is what makes a many-target extraction expensive.
    """
    if mz.size > 1 and bool(np.any(mz[1:] < mz[:-1])):
        order = np.argsort(mz, kind="stable")
        return mz[order], [None if a is None else a[order] for a in arrays]
    return mz, list(arrays)


def extract_chromatogram(
    spectra: Iterable[Spectrum],
    mode: Literal["tic", "bpc"] = "tic",
    mz_range: tuple[float, float] | None = None,
) -> Chromatogram:
    """Total-ion or base-peak chromatogram over a run.

    Parameters
    ----------
    spectra:
        Any iterable of spectra, typically ``reader.ms1``. Consumed once.
    mode:
        ``"tic"`` sums each spectrum's intensity; ``"bpc"`` takes its maximum.
    mz_range:
        Optional ``(low, high)`` m/z window to restrict the sum to.

    Returns
    -------
    :class:`Chromatogram`
    """
    if mode not in ("tic", "bpc"):
        raise ValueError(f"mode must be 'tic' or 'bpc', got {mode!r}")

    rts: list[float] = []
    values: list[float] = []

    for i, spec in enumerate(spectra):
        intensity = np.asarray(spec.intensity, dtype=np.float64)
        if mz_range is not None:
            mz = np.asarray(spec.mz, dtype=np.float64)
            keep = (mz >= mz_range[0]) & (mz <= mz_range[1])
            intensity = intensity[keep]

        rts.append(_rt_of(spec, i))
        if intensity.size == 0:
            values.append(0.0)
        else:
            values.append(float(intensity.sum() if mode == "tic" else intensity.max()))

    rt = np.asarray(rts, dtype=np.float64)
    inten = np.asarray(values, dtype=np.float64)
    order = np.argsort(rt, kind="stable")

    label = "TIC" if mode == "tic" else "Base peak"
    if mz_range is not None:
        label += f" ({mz_range[0]:g}-{mz_range[1]:g} m/z)"
    return Chromatogram(rt=rt[order], intensity=inten[order], label=label)


def extract_xic(
    spectra: Iterable[Spectrum],
    targets: Sequence[float] | float,
    tolerance: float = 20.0,
    tolerance_type: ToleranceLike = ToleranceType.PPM,
    im_window: tuple[float, float] | None = None,
    aggregate: Aggregate = "sum",
) -> list[Chromatogram]:
    """Extracted ion chromatograms for one or more target m/z values.

    All targets are extracted in a **single pass** over ``spectra``, because the
    iterable is usually a reader that is expensive to walk and may not be
    replayable.

    Parameters
    ----------
    spectra:
        Any iterable of spectra, typically ``reader.ms1``. Consumed once.
    targets:
        One m/z, or a sequence of them.
    tolerance, tolerance_type:
        Extraction window, ``"ppm"`` (default) or ``"da"``.
    im_window:
        Optional ``(low, high)`` ion-mobility window. On timsTOF data this is
        what makes a trace selective -- two co-eluting species at the same m/z
        usually separate in mobility. Spectra carrying no ion mobility cannot be
        gated, so they pass through ungated and a :class:`UserWarning` is
        raised; ``meta["im_window_applied"]`` then says whether the gate ever
        ran and ``meta["im_window_skipped"]`` how many spectra escaped it.
    aggregate:
        ``"sum"`` of the peaks in the window (the quantification convention), or
        ``"max"``.

    Returns
    -------
    One :class:`Chromatogram` per target, in the order given.
    """
    tol_type = ToleranceType(str(tolerance_type).lower())
    if aggregate not in ("sum", "max"):
        raise ValueError(f"aggregate must be 'sum' or 'max', got {aggregate!r}")

    single = np.isscalar(targets)
    target_arr = np.atleast_1d(np.asarray(targets, dtype=np.float64))
    if target_arr.size == 0:
        return []

    if tol_type is ToleranceType.PPM:
        lo_targets = target_arr * (1.0 - tolerance / 1e6)
        hi_targets = target_arr * (1.0 + tolerance / 1e6)
    else:
        lo_targets = target_arr - tolerance
        hi_targets = target_arr + tolerance

    rts: list[float] = []
    rows: list[NDArray[np.float64]] = []
    # An IM-less spectrum cannot be gated. Dropping it would silently shorten a
    # trace, so it passes through ungated -- but the caller has to be told, or a
    # window that was never applied looks like one that found nothing.
    im_gated = 0
    im_ungated = 0

    for i, spec in enumerate(spectra):
        mz = np.asarray(spec.mz, dtype=np.float64)
        intensity = np.asarray(spec.intensity, dtype=np.float64)
        im = None if spec.im is None else np.asarray(spec.im, dtype=np.float64)

        if im_window is not None:
            if im is None:
                im_ungated += 1
            else:
                keep = (im >= im_window[0]) & (im <= im_window[1])
                mz, intensity = mz[keep], intensity[keep]
                im_gated += 1

        mz, (intensity,) = _sorted_view(mz, intensity)  # type: ignore[assignment]

        row = np.zeros(target_arr.size, dtype=np.float64)
        if mz.size:
            lo = np.searchsorted(mz, lo_targets, side="left")
            hi = np.searchsorted(mz, hi_targets, side="right")
            for k in range(target_arr.size):
                seg = intensity[lo[k] : hi[k]]
                if seg.size:
                    # Summed directly rather than differencing a cumulative sum:
                    # the window holds a handful of peaks, so this is both faster
                    # and free of the cancellation error a cumsum accumulates
                    # across tens of thousands of values.
                    row[k] = float(seg.sum() if aggregate == "sum" else seg.max())
        rows.append(row)
        rts.append(_rt_of(spec, i))

    if im_ungated:
        if im_gated:
            warnings.warn(
                f"im_window={im_window} was not applied to {im_ungated} of {im_ungated + im_gated} spectra: "
                "they carry no ion mobility, so their peaks were counted ungated.",
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"im_window={im_window} was requested but no spectrum carries ion mobility; "
                "the window was not applied.",
                stacklevel=2,
            )

    rt = np.asarray(rts, dtype=np.float64)
    order = np.argsort(rt, kind="stable")
    matrix = np.vstack(rows)[order] if rows else np.zeros((0, target_arr.size))

    unit = "ppm" if tol_type is ToleranceType.PPM else "Da"
    out = [
        Chromatogram(
            rt=rt[order],
            intensity=matrix[:, k] if matrix.size else np.zeros(0),
            label=f"m/z {target_arr[k]:.4f}",
            mz=float(target_arr[k]),
            tolerance=float(tolerance),
            tolerance_type=unit,
            meta={
                "im_window": im_window,
                "im_window_applied": im_gated > 0,
                "im_window_skipped": im_ungated,
                "aggregate": aggregate,
            },
        )
        for k in range(target_arr.size)
    ]
    return out[:1] if single else out
