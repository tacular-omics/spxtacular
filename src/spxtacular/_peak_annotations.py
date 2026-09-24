"""Normalise per-peak annotation input — standard library only.

Shared by the MSP/MGF writers (which must stay free of non-numpy imports) and
the mzSpecLib model. Parsing and validating mzPAF is the caller's business:
this module only lines annotations up with peaks.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

from .errors import SpxtacularError

# An annotation item is a string or an object with ``serialize()`` (a paftacular
# ``PafAnnotation``). A peak holds zero or more items.
type PeakAnnotations = tuple[object, ...]


def _is_item(value: object) -> bool:
    return isinstance(value, str) or callable(getattr(value, "serialize", None))


def _peak_items(value: object, *, where: str, index: int) -> PeakAnnotations:
    """The annotation items of one peak."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if _is_item(value):
        return (value,)
    if isinstance(value, Iterable):
        items = tuple(value)
        for item in items:
            if not _is_item(item):
                raise SpxtacularError(
                    f"{where}: peak {index} annotation must be a string or have serialize(), got {type(item).__name__}"
                )
        return tuple(item for item in items if not (isinstance(item, str) and not item))
    raise SpxtacularError(
        f"{where}: peak {index} annotation must be None, a string, an mzPAF annotation or a sequence of them, "
        f"got {type(value).__name__}"
    )


def _is_matched_fragment(value: object) -> bool:
    return hasattr(value, "peak_index") and hasattr(value, "annotation")


def align_peak_annotations(annotations: Any, n_peaks: int, *, where: str) -> tuple[PeakAnnotations, ...]:
    """Return one tuple of annotation items per peak.

    Parameters
    ----------
    annotations
        Either one entry per peak (each ``None``, a string, an object with
        ``serialize()``, or a sequence of those), or an iterable of matched
        fragments (objects with ``peak_index`` and ``annotation``, such as
        :class:`~spxtacular.matching.MatchedFragment`), grouped by peak.
    n_peaks
        Number of peaks in the spectrum.
    where
        Prefix for error messages.

    Raises
    ------
    SpxtacularError
        If the per-peak form has the wrong length, a matched fragment points
        outside the spectrum, or an item is neither a string nor serialisable.
    """
    if isinstance(annotations, str) or not isinstance(annotations, Iterable):
        raise SpxtacularError(
            f"{where}: annotations must be a sequence with one entry per peak or an iterable of matched fragments"
        )
    values = list(annotations)
    if values and all(_is_matched_fragment(value) for value in values):
        grouped: list[list[object]] = [[] for _ in range(n_peaks)]
        for match in values:
            index = int(match.peak_index)
            if not 0 <= index < n_peaks:
                raise SpxtacularError(f"{where}: matched fragment peak_index {index} is outside 0..{n_peaks - 1}")
            grouped[index].append(match.annotation)
        return tuple(tuple(items) for items in grouped)
    if not values and n_peaks:
        # An empty match list: nothing annotated.
        return tuple(() for _ in range(n_peaks))
    if len(values) != n_peaks:
        raise SpxtacularError(f"{where}: {len(values)} annotations for {n_peaks} peaks")
    return tuple(_peak_items(value, where=where, index=i) for i, value in enumerate(values))


def item_text(item: object) -> str:
    """The text of one annotation item."""
    if isinstance(item, str):
        return item
    return str(cast(Any, item).serialize())


_FORBIDDEN_IN_PEAK_LISTS = ('"', ";", "\t", "\n", "\r")


def peak_list_annotation_texts(annotations: Any, n_peaks: int, *, where: str) -> list[str | None]:
    """Per-peak annotation text for MSP/MGF, comma-joined, ``None`` for an unannotated peak.

    Raises
    ------
    SpxtacularError
        If an annotation contains a character that would break the peak line.
    """
    texts: list[str | None] = []
    for index, items in enumerate(align_peak_annotations(annotations, n_peaks, where=where)):
        if not items:
            texts.append(None)
            continue
        text = ",".join(item_text(item) for item in items)
        for bad in _FORBIDDEN_IN_PEAK_LISTS:
            if bad in text:
                raise SpxtacularError(f"{where}: peak {index} annotation {text!r} contains {bad!r}")
        texts.append(text)
    return texts


def per_spectrum(annotations: Iterable[Any]) -> Iterable[Any]:
    """Pass-through that rejects a bare string where one item per spectrum is expected."""
    if isinstance(annotations, str):
        raise SpxtacularError("annotations must hold one entry per spectrum, not a string")
    return annotations
