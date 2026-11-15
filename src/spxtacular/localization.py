"""PTM site localization by the Ascore method.

Ascore (Beausoleil et al., *Nature Biotechnology* 24(10):1285-1292, 2006,
doi:10.1038/nbt1240) asks which of several candidate placements of a modification the
fragment ions support. The pieces come from three packages: peptacular enumerates the
candidate placements and the ions that separate them, this module filters the spectrum to
each peak depth and matches, and :func:`spxtacular.scoring._binom_log10_survival` supplies
the cumulative binomial.

The per-depth peptide score is ``-10 log10 P(X >= k)`` for ``X ~ Binomial(n, p)``, with
``n`` the theoretical ions of the candidate, ``k`` how many were matched, and

    p = depth * (2 * tolerance in Da) / window

the chance one peak falls in one ion's match window. The paper writes ``p = i / 100``,
which is this expression for a 1 Da window on the low-resolution data it was developed on;
spelling the tolerance out keeps the statistic honest on high-resolution data. Candidates
are ranked by the weighted mean of the per-depth scores over the paper's peak-depth weights
(0.5, 0.75, 1, 1, 1, 1, 0.75, 0.5, 0.25, 0.25), the quantity the paper calls the peptide
score. The AScore itself is the difference, at the depth that separates the top two
candidates best, between their scores recomputed from the site-determining ions alone.
"""

from __future__ import annotations

import math
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from tacular.types import ToleranceUnit

from .core import Spectrum, SpectrumType
from .errors import SpxtacularError
from .matching import match_fragments
from .scoring import _binom_log10_survival

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = [
    "ASCORE_CONFIDENT_CUT",
    "ASCORE_PEAK_DEPTHS",
    "LocalizationProvenance",
    "LocalizationResult",
    "localize",
]

ASCORE_PEAK_DEPTHS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
"""The paper's ten peak depths: keep the top 1, then 2, ... peaks per window."""

ASCORE_CONFIDENT_CUT = 19.0
"""The paper's AScore cut for a confident site (p <= 0.01). Callers may use another."""

_ASCORE_DEPTH_WEIGHTS: tuple[float, ...] = (0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 0.75, 0.5, 0.25, 0.25)
"""Peak-depth weights from the paper, for :data:`ASCORE_PEAK_DEPTHS` only."""


def _require_peptacular() -> Any:
    try:
        import peptacular as pt
    except ImportError as error:  # pragma: no cover - peptacular is a runtime dependency
        raise SpxtacularError("Site localization requires peptacular.") from error
    return pt


@dataclass(frozen=True, slots=True)
class LocalizationProvenance:
    """Every argument that determines a :class:`LocalizationResult`.

    Serialized with ``schema_version`` 1.
    """

    ion_types: tuple[str, ...]
    charges: tuple[int, ...]
    tolerance: float
    tolerance_unit: ToleranceUnit
    peak_depths: tuple[int, ...]
    window: float
    _: KW_ONLY
    max_isomers: int | None = None
    method: str = "ascore"

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned JSON-compatible representation."""
        return {
            "schema_version": 1,
            "method": self.method,
            "ion_types": list(self.ion_types),
            "charges": list(self.charges),
            "tolerance": self.tolerance,
            "tolerance_unit": self.tolerance_unit,
            "peak_depths": list(self.peak_depths),
            "window": self.window,
            "max_isomers": self.max_isomers,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LocalizationProvenance:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            ion_types=tuple(payload["ion_types"]),
            charges=tuple(int(c) for c in payload["charges"]),
            tolerance=float(payload["tolerance"]),
            tolerance_unit=payload["tolerance_unit"],
            peak_depths=tuple(int(d) for d in payload["peak_depths"]),
            window=float(payload["window"]),
            max_isomers=payload.get("max_isomers"),
            method=payload.get("method", "ascore"),
        )


@dataclass(frozen=True, slots=True)
class LocalizationResult:
    """Which placement of an ambiguous modification the spectrum supports.

    ``peptide_scores`` is the weighted peptide score of each candidate, in
    :attr:`isomers` order. ``site_determining`` holds ``(matched, available)`` per
    candidate, counted at :attr:`peak_depth` against the best other candidate.
    """

    isomers: tuple[str, ...]
    best: str
    runner_up: str | None
    ascore: float | None
    peptide_scores: tuple[float, ...]
    peak_depth: int
    site_determining: tuple[tuple[int, int], ...]
    provenance: LocalizationProvenance

    def is_ambiguous(self, cut: float = ASCORE_CONFIDENT_CUT) -> bool:
        """Whether the AScore is below ``cut``. A lone candidate is never ambiguous."""
        return False if self.ascore is None else self.ascore < cut

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned JSON-compatible representation."""
        return {
            "schema_version": 1,
            "isomers": list(self.isomers),
            "best": self.best,
            "runner_up": self.runner_up,
            "ascore": self.ascore,
            "peptide_scores": list(self.peptide_scores),
            "peak_depth": self.peak_depth,
            "site_determining": [list(pair) for pair in self.site_determining],
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LocalizationResult:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            isomers=tuple(payload["isomers"]),
            best=payload["best"],
            runner_up=payload["runner_up"],
            ascore=payload["ascore"],
            peptide_scores=tuple(float(s) for s in payload["peptide_scores"]),
            peak_depth=int(payload["peak_depth"]),
            site_determining=tuple((int(a), int(b)) for a, b in payload["site_determining"]),
            provenance=LocalizationProvenance.from_dict(payload["provenance"]),
        )


def _tolerance_da(spectrum: Spectrum, tolerance: float, tolerance_unit: ToleranceUnit) -> float:
    """The match half-window in Da, as :func:`spxtacular.score` computes it."""
    if tolerance_unit == "ppm":
        if spectrum.mz.size == 0:
            return 0.0
        return float(tolerance) * float(np.median(spectrum.mz)) / 1e6
    return float(tolerance)


def _matched_ions(spectrum: Spectrum, fragments: list[Any], tolerance: float, tolerance_unit: ToleranceUnit) -> int:
    """How many distinct theoretical ions of ``fragments`` the spectrum explains."""
    if not fragments or spectrum.mz.size == 0:
        return 0
    matches = match_fragments(spectrum, fragments, tolerance=tolerance, tolerance_unit=tolerance_unit)
    return len({(str(m.fragment.ion_type), m.fragment.position, m.fragment.charge_state) for m in matches})


def _sequence_informative(fragments: list[Any]) -> list[Any]:
    """Drop the full-length fragment of each series: it carries no site information.

    A 9-mer has b1..b8 and y1..y8, not b9 and y9, which are the whole peptide and the same
    mass for every candidate placement. peptacular generates them; counting them would
    inflate ``n`` and the score. pyAscore excludes them too, reporting 16 fragments for a
    9-mer where peptacular returns 18.
    """
    kept = []
    for fragment in fragments:
        position = getattr(fragment, "position", None)
        length = getattr(fragment, "parent_sequence_length", None)
        if position is not None and length is not None and int(position) >= int(length):
            continue
        kept.append(fragment)
    return kept


def _depth_score(k: int, n: int, p: float) -> float:
    """``-10 log10 P(X >= k)`` for ``X ~ Binomial(n, p)``, the per-depth peptide score."""
    if n <= 0 or k <= 0:
        return 0.0
    return -10.0 * _binom_log10_survival(k, n, min(1.0, max(0.0, p)))


def _weights(peak_depths: tuple[int, ...]) -> tuple[float, ...]:
    """The paper's weights for its ten depths, uniform weights for any other set."""
    if peak_depths == ASCORE_PEAK_DEPTHS:
        return _ASCORE_DEPTH_WEIGHTS
    return (1.0,) * len(peak_depths)


def localize(
    spectrum: Spectrum,
    peptide: str | Any,
    *,
    ion_types: Sequence[Any] = ("b", "y"),
    charges: Sequence[int] = (1,),
    tolerance: float = 0.02,
    tolerance_unit: ToleranceUnit = "da",
    peak_depths: Sequence[int] = ASCORE_PEAK_DEPTHS,
    window: float = 100.0,
    max_isomers: int | None = 10_000,
) -> LocalizationResult:
    """Which placement of an ambiguous modification a spectrum supports (Ascore).

    Parameters
    ----------
    spectrum:
        A centroided :class:`~spxtacular.Spectrum` in m/z (not decharged).
    peptide:
        A ProForma string (or peptacular annotation) whose modification positions are
        ambiguous: a ``#label`` group (``PEPS[Phospho#g1]TIDE``), a range
        (``PEP(ST)[Phospho]IDE``) or an unknown position (``[Phospho]?PEPTIDE``). A
        peptide with no ambiguity gives one candidate and no AScore. Any localization
        scores already written into the string are used only to find the candidate
        residues, never read as evidence.
    ion_types, charges:
        Fragment ions to generate, as for :func:`~spxtacular.match_fragments`.
    tolerance, tolerance_unit:
        Match tolerance. It also sets the random-match probability, which is
        ``depth * 2 * tolerance_in_da / window``.
    peak_depths:
        Peak depths to score. The paper's ten by default; another set is scored with
        uniform weights instead of the paper's.
    window:
        Width in Da of the peak-depth window (the paper's 100 Da).
    max_isomers:
        Passed to peptacular; more candidates than this raises.

    Returns
    -------
    LocalizationResult
        ``ascore`` is ``None`` when the peptide has one candidate, and 0.0 when the top two
        candidates have no site-determining ions or none of them matched.

    Raises
    ------
    SpxtacularError
        A decharged or profile spectrum, an empty ``peak_depths``, a non-positive
        ``window``, a peptide peptacular cannot expand, or more candidates than
        ``max_isomers``.

    Examples
    --------
    >>> import numpy as np
    >>> from spxtacular import Spectrum, localize
    >>> import peptacular as pt
    >>> frags = pt.fragment("PEPT[Phospho]IDESK", ion_types=("b", "y"), charges=(1,))
    >>> spec = Spectrum(mz=np.array(sorted(f.mz for f in frags)),
    ...                 intensity=np.full(len(frags), 1000.0))
    >>> result = localize(spec, "PEPT[Phospho#g1]IDES[#g1]K", tolerance=0.05)
    >>> result.best
    'PEPT[Phospho#g1]IDESK'
    >>> round(result.ascore, 4)
    215.9176
    """
    pt = _require_peptacular()
    if spectrum.is_decharged:
        raise SpxtacularError("Site localization needs m/z values; this spectrum is decharged to neutral masses.")
    if spectrum.spectrum_type == SpectrumType.PROFILE:
        raise SpxtacularError("Site localization needs centroid data, not profile data. Call .centroid() first.")
    depths = tuple(int(d) for d in peak_depths)
    if not depths:
        raise SpxtacularError("peak_depths is empty; pass at least one peak depth.")
    if any(d <= 0 for d in depths):
        raise SpxtacularError(f"peak_depths must all be >= 1, got {depths!r}.")
    if not math.isfinite(window) or window <= 0:
        raise SpxtacularError(f"window must be a finite number > 0, got {window!r}.")

    try:
        isomers = pt.localization_isomers(peptide, max_isomers=max_isomers)
    except Exception as error:
        raise SpxtacularError(f"Cannot expand the candidate sites of {peptide!r}: {error}") from error
    if not isomers:
        raise SpxtacularError(f"{peptide!r} has no candidate placements.")
    names = tuple(isomer.serialize() for isomer in isomers)

    tol_da = _tolerance_da(spectrum, tolerance, tolerance_unit)
    fragments = [
        _sequence_informative(pt.fragment(isomer, ion_types=tuple(ion_types), charges=tuple(charges)))
        for isomer in isomers
    ]
    filtered = {d: spectrum.filter(top_n_per_window=(d, float(window))) for d in depths}

    # Per-depth peptide score for every candidate, then the paper's weighted mean.
    per_depth: list[list[float]] = []
    for frags in fragments:
        row = []
        for d in depths:
            p = d * 2.0 * tol_da / float(window)
            k = _matched_ions(filtered[d], frags, tolerance, tolerance_unit)
            row.append(_depth_score(k, len(frags), p))
        per_depth.append(row)
    weights = _weights(depths)
    total = float(sum(weights))
    scores = tuple(float(sum(w * s for w, s in zip(weights, row, strict=True)) / total) for row in per_depth)

    order = sorted(range(len(names)), key=lambda i: (-scores[i], i))
    best_i = order[0]
    if len(names) == 1:
        depth = depths[int(np.argmax(per_depth[best_i]))]
        return LocalizationResult(
            isomers=names,
            best=names[best_i],
            runner_up=None,
            ascore=None,
            peptide_scores=scores,
            peak_depth=depth,
            site_determining=((0, 0),),
            provenance=LocalizationProvenance(
                ion_types=tuple(str(t) for t in ion_types),
                charges=tuple(int(c) for c in charges),
                tolerance=float(tolerance),
                tolerance_unit=tolerance_unit,
                peak_depths=depths,
                window=float(window),
                max_isomers=max_isomers,
            ),
        )

    second_i = order[1]
    # The depth that separates the top two best is the one the AScore is computed at.
    gaps = [per_depth[best_i][j] - per_depth[second_i][j] for j in range(len(depths))]
    depth_index = int(np.argmax(gaps))
    depth = depths[depth_index]
    p_depth = depth * 2.0 * tol_da / float(window)
    at_depth = filtered[depth]

    pairs = pt.pairwise_site_determining_ions(
        isomers, ion_types=tuple(ion_types), charges=tuple(charges), tolerance=tolerance, tolerance_unit=tolerance_unit
    )

    def _sdi(i: int, j: int) -> tuple[int, int]:
        ions = list(pairs.get((i, j), ()))
        return _matched_ions(at_depth, ions, tolerance, tolerance_unit), len(ions)

    site_determining = tuple(_sdi(i, best_i if i != best_i else second_i) for i in range(len(names)))
    k_best, n_best = site_determining[best_i]
    k_second, n_second = site_determining[second_i]
    ascore = _depth_score(k_best, n_best, p_depth) - _depth_score(k_second, n_second, p_depth)

    return LocalizationResult(
        isomers=names,
        best=names[best_i],
        runner_up=names[second_i],
        ascore=float(ascore),
        peptide_scores=scores,
        peak_depth=depth,
        site_determining=site_determining,
        provenance=LocalizationProvenance(
            ion_types=tuple(str(t) for t in ion_types),
            charges=tuple(int(c) for c in charges),
            tolerance=float(tolerance),
            tolerance_unit=tolerance_unit,
            peak_depths=depths,
            window=float(window),
            max_isomers=max_isomers,
        ),
    )
