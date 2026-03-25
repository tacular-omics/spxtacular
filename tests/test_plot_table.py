"""
Tests for spxtacular.plot_table — build_plot_table, build_annot_plot_table,
and plot_from_table.

Fragments are mocked with MagicMock; build_annot_plot_table only reads
.mz, .ion_type, .position, and .charge_state from each fragment.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from peptacular.annotation.frag import Fragment
from tacular.ion_types.data import IonType

from spxtacular.core import Spectrum
from spxtacular.plot_table import (
    _SINGLETON_COLOR,
    build_annot_plot_table,
    build_plot_table,
    plot_from_table,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXPECTED_COLUMNS = [
    "mz",
    "intensity",
    "charge",
    "score",
    "im",
    "color",
    "linewidth",
    "opacity",
    "series",
    "label",
    "label_size",
    "label_font",
    "label_color",
    "label_yshift",
    "label_xanchor",
    "hover",
]


def _spectrum() -> Spectrum:
    return Spectrum(
        mz=np.array([100.0, 200.0, 300.0], dtype=np.float64),
        intensity=np.array([10.0, 50.0, 20.0], dtype=np.float64),
    )


def _decon_spectrum() -> Spectrum:
    return Spectrum(
        mz=np.array([200.0, 400.0], dtype=np.float64),
        intensity=np.array([100.0, 50.0], dtype=np.float64),
        charge=np.array([1, 2], dtype=np.int32),
        score=np.array([0.8, 0.3], dtype=np.float64),
    )


def _make_frag(
    mz: float,
    ion_type: str = "b",
    position: int = 1,
    charge_state: int = 1,
) -> MagicMock:
    f = MagicMock()
    f.mz = mz
    f.ion_type = ion_type
    f.position = position
    f.charge_state = charge_state
    return f


def _real_frag(
    mz: float,
    ion_type: str = "b",
    position: int = 1,
    charge_state: int = 1,
) -> Fragment:
    """Real Fragment needed for tests that exercise _fragment_label (paftacular)."""
    return Fragment(
        ion_type=IonType(ion_type),
        position=position,
        mass=mz,
        monoisotopic=True,
        charge_state=charge_state,
    )


# ---------------------------------------------------------------------------
# build_plot_table — schema
# ---------------------------------------------------------------------------


def test_build_plot_table_columns() -> None:
    table = build_plot_table(_spectrum())
    assert list(table.columns) == _EXPECTED_COLUMNS


def test_build_plot_table_row_count() -> None:
    spec = _spectrum()
    table = build_plot_table(spec)
    assert len(table) == len(spec.mz)


def test_build_plot_table_data_values() -> None:
    spec = _spectrum()
    table = build_plot_table(spec)
    np.testing.assert_array_equal(table["mz"].to_numpy(), spec.mz)
    np.testing.assert_array_equal(table["intensity"].to_numpy(), spec.intensity)


def test_build_plot_table_defaults() -> None:
    table = build_plot_table(_spectrum())
    assert (table["linewidth"] == 1.0).all()
    assert (table["opacity"] == 1.0).all()
    assert (table["label_size"] == 10.0).all()


# ---------------------------------------------------------------------------
# build_plot_table — plain spectrum (no charge/score)
# ---------------------------------------------------------------------------


def test_build_plot_table_no_charge_color() -> None:
    table = build_plot_table(_spectrum())
    assert (table["color"] == "steelblue").all()
    assert (table["series"] == "peaks").all()


def test_build_plot_table_no_charge_no_na() -> None:
    table = build_plot_table(_spectrum())
    assert table["charge"].isna().all()


def test_build_plot_table_no_score_label_empty() -> None:
    table = build_plot_table(_spectrum())
    assert (table["label"] == "").all()


# ---------------------------------------------------------------------------
# build_plot_table — deconvoluted spectrum (charge + score)
# ---------------------------------------------------------------------------


def test_build_plot_table_charge_series() -> None:
    table = build_plot_table(_decon_spectrum())
    assert table.loc[table["mz"] == 200.0, "series"].iloc[0] == "z=1"
    assert table.loc[table["mz"] == 400.0, "series"].iloc[0] == "z=2"


def test_build_plot_table_charge_colors_differ() -> None:
    table = build_plot_table(_decon_spectrum())
    color_z1 = table.loc[table["mz"] == 200.0, "color"].iloc[0]
    color_z2 = table.loc[table["mz"] == 400.0, "color"].iloc[0]
    assert color_z1 != color_z2


def test_build_plot_table_singleton_color() -> None:
    spec = Spectrum(
        mz=np.array([100.0, 200.0], dtype=np.float64),
        intensity=np.array([10.0, 50.0], dtype=np.float64),
        charge=np.array([-1, 1], dtype=np.int32),
        score=np.array([0.0, 0.9], dtype=np.float64),
    )
    table = build_plot_table(spec)
    singleton_row = table.loc[table["mz"] == 100.0]
    assert singleton_row["series"].iloc[0] == "singleton"
    assert singleton_row["color"].iloc[0] == _SINGLETON_COLOR


def test_build_plot_table_score_label() -> None:
    table = build_plot_table(_decon_spectrum())
    label_z1 = table.loc[table["mz"] == 200.0, "label"].iloc[0]
    label_z2 = table.loc[table["mz"] == 400.0, "label"].iloc[0]
    # score=0.8 → labelled; score=0.3 → also labelled (both > 0)
    assert label_z1 == "0.80"
    assert label_z2 == "0.30"


def test_build_plot_table_zero_score_label_empty() -> None:
    spec = Spectrum(
        mz=np.array([100.0], dtype=np.float64),
        intensity=np.array([10.0], dtype=np.float64),
        charge=np.array([-1], dtype=np.int32),
        score=np.array([0.0], dtype=np.float64),
    )
    table = build_plot_table(spec)
    assert table["label"].iloc[0] == ""


def test_build_plot_table_show_scores_false() -> None:
    table = build_plot_table(_decon_spectrum(), show_scores=False)
    assert (table["label"] == "").all()


def test_build_plot_table_show_charges_false() -> None:
    table = build_plot_table(_decon_spectrum(), show_charges=False)
    assert (table["color"] == "steelblue").all()
    assert (table["series"] == "peaks").all()


# ---------------------------------------------------------------------------
# build_plot_table — Spectrum methods
# ---------------------------------------------------------------------------


def test_spectrum_plot_table_method() -> None:
    spec = _decon_spectrum()
    via_method = spec.plot_table()
    via_function = build_plot_table(spec)
    pd.testing.assert_frame_equal(via_method, via_function)


# ---------------------------------------------------------------------------
# build_annot_plot_table — schema
# ---------------------------------------------------------------------------


def test_annot_plot_table_columns() -> None:
    table = build_annot_plot_table(_spectrum(), [])
    assert list(table.columns) == _EXPECTED_COLUMNS


# ---------------------------------------------------------------------------
# build_annot_plot_table — unmatched peaks
# ---------------------------------------------------------------------------


def test_annot_plot_table_unmatched_color() -> None:
    # Fragment at 999.0 will not match any peak in _spectrum()
    frag = _make_frag(999.0)
    table = build_annot_plot_table(_spectrum(), [frag], tolerance=0.02)
    assert (table["color"] == "#cccccc").all()
    assert (table["series"] == "unmatched").all()


def test_annot_plot_table_unmatched_label_empty() -> None:
    frag = _make_frag(999.0)
    table = build_annot_plot_table(_spectrum(), [frag], tolerance=0.02)
    assert (table["label"] == "").all()


# ---------------------------------------------------------------------------
# build_annot_plot_table — matched peaks
# ---------------------------------------------------------------------------


def test_annot_plot_table_matched_color() -> None:
    # Fragment at 200.005 matches peak at 200.0 within 0.02 Da
    frag = _real_frag(200.005, ion_type="b", position=2)
    table = build_annot_plot_table(_spectrum(), [frag], tolerance=0.02)
    matched_row = table.loc[table["mz"] == 200.0]
    assert matched_row["color"].iloc[0] == "#1f77b4"
    assert matched_row["series"].iloc[0] == "b"


def test_annot_plot_table_matched_label_nonempty() -> None:
    frag = _real_frag(200.005, ion_type="b", position=2)
    table = build_annot_plot_table(_spectrum(), [frag], tolerance=0.02)
    matched_label = table.loc[table["mz"] == 200.0, "label"].iloc[0]
    assert matched_label != ""


def test_annot_plot_table_y_ion_color() -> None:
    frag = _real_frag(200.005, ion_type="y", position=2)
    table = build_annot_plot_table(_spectrum(), [frag], tolerance=0.02)
    matched_row = table.loc[table["mz"] == 200.0]
    assert matched_row["color"].iloc[0] == "#d62728"
    assert matched_row["series"].iloc[0] == "y"


# ---------------------------------------------------------------------------
# plot_from_table — guarded by plotly availability
# ---------------------------------------------------------------------------

pytest.importorskip("plotly", reason="plotly is not installed")


def test_plot_from_table_returns_figure() -> None:
    import plotly.graph_objects as go

    table = build_plot_table(_spectrum())
    fig = plot_from_table(table)
    assert isinstance(fig, go.Figure)


def test_plot_from_table_title() -> None:
    table = build_plot_table(_spectrum())
    fig = plot_from_table(table, title="My Spectrum")
    assert fig.layout.title.text == "My Spectrum"


def test_plot_from_table_single_series_no_legend() -> None:
    table = build_plot_table(_spectrum())
    # _spectrum() has no charge data → single "peaks" series
    fig = plot_from_table(table)
    assert fig.layout.showlegend is False


def test_plot_from_table_multi_series_legend() -> None:
    # _decon_spectrum() has z=1 and z=2 → two series
    table = build_plot_table(_decon_spectrum())
    fig = plot_from_table(table)
    assert fig.layout.showlegend is True


def test_plot_from_table_label_creates_annotation() -> None:
    table = build_plot_table(_decon_spectrum(), show_scores=True)
    # Both peaks have score > 0, so both should be labelled
    fig = plot_from_table(table)
    assert len(fig.layout.annotations) > 0


def test_plot_from_table_no_label_no_annotation() -> None:
    table = build_plot_table(_spectrum())
    # No scores set → all labels are empty → no annotations
    fig = plot_from_table(table)
    assert len(fig.layout.annotations) == 0


def test_plot_from_table_custom_color() -> None:
    import plotly.graph_objects as go

    table = build_plot_table(_spectrum())
    table["color"] = "#ff0000"
    fig = plot_from_table(table)
    # Only one (series, color) group should exist, coloured red
    assert any(
        isinstance(trace, go.Scatter) and trace.line.color == "#ff0000"
        for trace in fig.data
    )


def test_plot_from_table_custom_linewidth() -> None:
    import plotly.graph_objects as go

    table = build_plot_table(_spectrum())
    table["linewidth"] = 3.5
    fig = plot_from_table(table)
    assert any(
        isinstance(trace, go.Scatter) and trace.line.width == 3.5
        for trace in fig.data
    )
