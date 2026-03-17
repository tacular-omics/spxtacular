"""
Visualization tools for mass spectrometry data.
"""

import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .core import Spectrum
    # from peptacular import Annotation # Assuming this might be available later


def requires_plotly(func):
    """Decorator to check if plotly is installed."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import plotly.graph_objects as go  # type: ignore[import]
        except ImportError:
            raise ImportError("plotly required for plotting: pip install plotly")
        return func(*args, **kwargs)

    return wrapper


@requires_plotly
def plot_spectrum(
    spectrum: "Spectrum",
    title: str | None = None,
    show_charges: bool = True,
    **layout_kwargs,
):
    """Plot spectrum using plotly."""
    # Implementation will go here
    pass


def plot_df(
    peptide: str,
    ion_types: Sequence[str] = ("b", "y"),
    charges: Sequence[int] = (1, 2),
    losses: Sequence[str] | None = None,
    isotopes: int = 0,
    mz_tol: float = 0.02,
    mz_tol_type: Literal["Da", "ppm"] = "Da",
):
    """Returns a df with all the data needed for plotting"""
    pass


def annotate_spectrum(
    spectrum: "Spectrum",
    peptide: str,
    ion_types: Sequence[str] = ("b", "y"),
    charges: Sequence[int] = (1, 2),
    losses: Sequence[str] | None = None,
    isotopes: int = 0,
    mz_tol: float = 0.02,
    mz_tol_type: Literal["Da", "ppm"] = "Da",
    **plot_kwargs,
):
    """Annotate spectrum peaks."""
    pass


def annotate_from_df(df):
    """Annotate from a dataframe generated from plot_df"""
    pass
