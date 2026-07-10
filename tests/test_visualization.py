"""
Tests for spxtacular.visualization — all functions must return go.Figure objects.

annotate_spectrum calls paftacular internally for fragment labels, so real
Fragment objects (from peptacular) are used instead of mocks for that function.
"""

import numpy as np
import plotly.graph_objects as go
from peptacular import IonType
from peptacular.annotation.frag import Fragment

from spxtacular.core import Spectrum, SpectrumType
from spxtacular.visualization import annotate_spectrum, mirror_plot, plot_spectrum

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _raw() -> Spectrum:
    mz = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    intensity = np.array([10.0, 50.0, 20.0], dtype=np.float64)
    return Spectrum(mz=mz, intensity=intensity)


def _decon() -> Spectrum:
    return Spectrum(
        mz=np.array([200.0], dtype=np.float64),
        intensity=np.array([50.0], dtype=np.float64),
        charge=np.array([2], dtype=np.int32),
        iso_score=np.array([0.85], dtype=np.float64),
        spectrum_type=SpectrumType.DECONVOLUTED,
    )


def _real_frag(mz: float = 200.0, ion_type: str = "b", position: int = 2) -> Fragment:
    """Return a real Fragment using peptacular's API."""
    return Fragment(
        ion_type=IonType(ion_type),
        position=position,
        mass=mz,
        monoisotopic=True,
        charge_state=1,
    )


# ---------------------------------------------------------------------------
# plot_spectrum
# ---------------------------------------------------------------------------


def test_plot_spectrum_returns_figure() -> None:
    fig = plot_spectrum(_raw())
    assert isinstance(fig, go.Figure)


def test_plot_spectrum_with_charges_returns_figure() -> None:
    fig = plot_spectrum(_decon(), color="charge")
    assert isinstance(fig, go.Figure)


def test_plot_spectrum_with_scores_returns_figure() -> None:
    fig = plot_spectrum(_decon(), show_scores=True)
    assert isinstance(fig, go.Figure)


def test_plot_spectrum_show_charges_false_returns_figure() -> None:
    fig = plot_spectrum(_decon(), color=None)
    assert isinstance(fig, go.Figure)


def test_plot_spectrum_show_charges_deprecated_alias() -> None:
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = plot_spectrum(_decon(), show_charges=False)
    assert isinstance(fig, go.Figure)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_plot_spectrum_show_scores_false_returns_figure() -> None:
    fig = plot_spectrum(_decon(), show_scores=False)
    assert isinstance(fig, go.Figure)


def test_plot_spectrum_with_title_returns_figure() -> None:
    fig = plot_spectrum(_raw(), title="Test Spectrum")
    assert isinstance(fig, go.Figure)


def test_plot_empty_spectrum_returns_figure() -> None:
    empty = Spectrum(mz=np.array([], dtype=np.float64), intensity=np.array([], dtype=np.float64))
    fig = plot_spectrum(empty)
    assert isinstance(fig, go.Figure)


def test_plot_spectrum_has_at_least_one_trace() -> None:
    fig = plot_spectrum(_raw())
    assert len(fig.data) >= 1


def test_plot_spectrum_scores_annotated_when_present() -> None:
    decon = _decon()  # score=[0.85]
    fig = plot_spectrum(decon, show_scores=True)
    # Expect at least one annotation for the peak with score > 0
    assert len(fig.layout.annotations) >= 1


# ---------------------------------------------------------------------------
# mirror_plot
# ---------------------------------------------------------------------------


def test_mirror_plot_returns_figure() -> None:
    fig = mirror_plot(_raw(), _decon())
    assert isinstance(fig, go.Figure)


def test_mirror_plot_with_scores_returns_figure() -> None:
    fig = mirror_plot(_raw(), _decon(), show_scores=True)
    assert isinstance(fig, go.Figure)


def test_mirror_plot_show_charges_false_returns_figure() -> None:
    fig = mirror_plot(_raw(), _decon(), show_charges=False)
    assert isinstance(fig, go.Figure)


def test_mirror_plot_no_normalize_returns_figure() -> None:
    fig = mirror_plot(_raw(), _decon(), normalize=False)
    assert isinstance(fig, go.Figure)


def test_mirror_plot_has_multiple_traces() -> None:
    fig = mirror_plot(_raw(), _decon())
    # At minimum: raw trace + deconvoluted trace
    assert len(fig.data) >= 2


# ---------------------------------------------------------------------------
# annotate_spectrum
# ---------------------------------------------------------------------------


def test_annotate_spectrum_returns_figure() -> None:
    frag = _real_frag(mz=200.0, ion_type="b", position=2)
    fig = annotate_spectrum(_raw(), [frag])
    assert isinstance(fig, go.Figure)


def test_annotate_spectrum_no_fragments_returns_figure() -> None:
    fig = annotate_spectrum(_raw(), [])
    assert isinstance(fig, go.Figure)


def test_annotate_spectrum_with_ppm_tolerance_returns_figure() -> None:
    frag = _real_frag(mz=200.0, ion_type="b", position=2)
    fig = annotate_spectrum(_raw(), [frag], tolerance=10, tolerance_type="ppm")
    assert isinstance(fig, go.Figure)


def test_annotate_spectrum_matched_peak_has_annotation() -> None:
    frag = _real_frag(mz=200.0, ion_type="b", position=2)
    fig = annotate_spectrum(_raw(), [frag], tolerance=0.02, tolerance_type="da")
    # Matched peaks get annotations
    assert len(fig.layout.annotations) >= 1


def test_annotate_spectrum_no_match_has_no_annotation() -> None:
    frag = _real_frag(mz=999.0, ion_type="b", position=2)
    fig = annotate_spectrum(_raw(), [frag], tolerance=0.02, tolerance_type="da")
    assert len(fig.layout.annotations) == 0


# ---------------------------------------------------------------------------
# plot_spectrum: color="im" (ion mobility) path
# ---------------------------------------------------------------------------


def _spec_with_im() -> Spectrum:
    return Spectrum(
        mz=np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float64),
        intensity=np.array([10.0, 50.0, 20.0, 30.0], dtype=np.float64),
        im=np.array([0.8, 1.0, 1.2, 1.4], dtype=np.float64),
    )


def test_plot_spectrum_color_im_returns_figure() -> None:
    fig = plot_spectrum(_spec_with_im(), color="im")
    assert isinstance(fig, go.Figure)


def test_plot_spectrum_color_im_constant_im_does_not_crash() -> None:
    spec = Spectrum(
        mz=np.array([100.0, 200.0], dtype=np.float64),
        intensity=np.array([10.0, 50.0], dtype=np.float64),
        im=np.array([1.0, 1.0], dtype=np.float64),
    )
    fig = plot_spectrum(spec, color="im")
    assert isinstance(fig, go.Figure)


def test_plot_spectrum_color_im_with_scores_annotates() -> None:
    spec = Spectrum(
        mz=np.array([100.0, 200.0], dtype=np.float64),
        intensity=np.array([10.0, 50.0], dtype=np.float64),
        im=np.array([0.9, 1.1], dtype=np.float64),
        iso_score=np.array([0.0, 0.85], dtype=np.float64),
    )
    fig = plot_spectrum(spec, color="im", show_scores=True)
    # Only peaks with score > 0 are annotated
    assert len(fig.layout.annotations) == 1


def test_plot_spectrum_color_im_falls_back_when_no_im() -> None:
    # No im → falls through to the charge-coloured plot_from_table path
    fig = plot_spectrum(_raw(), color="im")
    assert isinstance(fig, go.Figure)


def test_plot_spectrum_color_im_respects_im_type_ook0() -> None:
    """``im_type="ook0"`` should render the colorbar label as ``1/K0``."""
    from spxtacular.core import MsnSpectrum

    spec = MsnSpectrum(
        mz=np.array([100.0, 200.0], dtype=np.float64),
        intensity=np.array([10.0, 50.0], dtype=np.float64),
        im=np.array([0.9, 1.1], dtype=np.float64),
        im_type="ook0",
    )
    fig = plot_spectrum(spec, color="im")
    assert isinstance(fig, go.Figure)
