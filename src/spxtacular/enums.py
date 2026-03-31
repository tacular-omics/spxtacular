"""Enum types shared across spxtacular modules."""

from enum import StrEnum
from typing import Literal


class ToleranceType(StrEnum):
    DA = "da"
    PPM = "ppm"


class PeakSelection(StrEnum):
    CLOSEST = "closest"
    LARGEST = "largest"
    ALL = "all"


ToleranceLike = ToleranceType | Literal["da", "ppm"]
PeakSelectionLike = PeakSelection | Literal["closest", "largest", "all"]
