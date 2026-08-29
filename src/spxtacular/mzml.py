"""Thin mzML creation helpers backed by the optional mzMLPy dependency."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_indexed_mzml_gzip(
    source: str | Path,
    output: str | Path,
    *,
    compression_level: int = 6,
) -> Any:
    """Create a self-indexed mzML gzip file using mzMLPy's format implementation."""
    try:
        from mzmlpy import write_indexed_gzip
    except ImportError as error:
        raise ImportError(
            "write_indexed_mzml_gzip requires mzMLPy 0.7 or newer. Install it with: pip install spxtacular[mzml]"
        ) from error
    return write_indexed_gzip(source, output, compression_level=compression_level)


__all__ = ["write_indexed_mzml_gzip"]
