"""Exception types raised by spxtacular."""

from __future__ import annotations

__all__ = ["SpxtacularError"]


class SpxtacularError(ValueError):
    """Base class for errors spxtacular raises about its own inputs and state.

    It subclasses :class:`ValueError`, so ``except ValueError`` keeps catching
    everything spxtacular raised as a ``ValueError`` before. A reader used before
    it is opened also raises it (it raised ``RuntimeError`` before 0.9).
    """
