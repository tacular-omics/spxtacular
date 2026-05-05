"""
Optional interactive spectrum viewer — requires `pip install spxtacular[viewer]`.

Launch via CLI::

    spxtacular-view path/to/file.mzML
    spxtacular-view path/to/file.d --port 8051
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

if TYPE_CHECKING:
    from .reader import Reader

try:
    import dash as _dash_check  # noqa: F401

    _HAS_DASH = True
except ImportError:
    _HAS_DASH = False


def _build_index(reader: Reader) -> pd.DataFrame:
    """Build a metadata-only index without loading or centroiding any peak arrays.

    Bypasses the spxtacular reader wrappers and reads directly from the underlying
    tdfpy/mzmlpy objects so that centroiding (which happens on first peak access)
    is never triggered.
    """
    from .reader import AcquisitionType, DReader, MzmlReader

    rows: list[dict] = []
    underlying = reader._reader  # type: ignore[attr-defined]  # DReader | MzmlReader

    if isinstance(underlying, DReader):
        tdf: Any = cast(Any, underlying._reader)
        assert tdf is not None, "DReader must be open before building index"

        # MS1 frames — frame.time and frame.frame_id are metadata; frame.centroid() is NOT called
        for frame in tdf.ms1:
            rows.append(
                {
                    "scan_number": frame.frame_id,
                    "ms_level": 1,
                    "rt": round(frame.time, 3),
                    "precursor_mz": None,
                }
            )

        # MS2 — access only stored metadata fields, never .peaks or .centroid()
        if underlying.acquisition_type == AcquisitionType.DDA:
            for prec in tdf.precursors:
                mz = prec.monoisotopic_mz if prec.monoisotopic_mz is not None else prec.largest_peak_mz
                rows.append(
                    {
                        "scan_number": prec.precursor_id,
                        "ms_level": 2,
                        "rt": round(prec.rt, 3),
                        "precursor_mz": round(mz, 4) if mz is not None else None,
                    }
                )
        elif underlying.acquisition_type == AcquisitionType.DIA:
            for window in tdf.windows:
                mz_range = window.mz_range
                mid_mz = round((mz_range[0] + mz_range[1]) / 2, 4) if mz_range else None
                rows.append(
                    {
                        "scan_number": window.frame_id,
                        "ms_level": 2,
                        "rt": round(window.rt, 3),
                        "precursor_mz": mid_mz,
                    }
                )
        elif underlying.acquisition_type == AcquisitionType.PRM:
            for transition in tdf.transitions:
                target = transition.target
                rows.append(
                    {
                        "scan_number": transition.frame_id,
                        "ms_level": 2,
                        "rt": round(transition.rt, 3),
                        "precursor_mz": round(target.monoisotopic_mz, 4)
                        if target.monoisotopic_mz is not None
                        else None,
                    }
                )

    elif isinstance(underlying, MzmlReader):
        # Iterate mzmlpy spectrum objects but access only header metadata.
        # spec.mz / spec.intensity are NOT accessed so base64 decoding and
        # any profile→centroid conversion is never triggered.
        handle = underlying._mzml_handle
        if handle is None:
            raise RuntimeError("MzmlReader must be open before building index")
        for spec in handle.spectra:
            precursor_mz = None
            if spec.ms_level is not None and spec.ms_level > 1:
                precs = list(spec.precursors)
                if precs:
                    ions = precs[0].selected_ions
                    if ions:
                        precursor_mz = ions[0].selected_ion_mz
            rt_val = spec.scan_start_time.total_seconds() if spec.scan_start_time is not None else None
            rows.append(
                {
                    "scan_number": spec.index,
                    "ms_level": spec.ms_level,
                    "rt": round(rt_val, 3) if rt_val is not None else None,
                    "precursor_mz": round(precursor_mz, 4) if precursor_mz is not None else None,
                }
            )

    else:
        raise TypeError(f"Unsupported reader type: {type(underlying)!r}")

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["scan_number", "ms_level", "rt", "precursor_mz"])
    if not df.empty and df["rt"].notna().any():
        df.sort_values("rt", inplace=True, ignore_index=True)
    return df



def launch_viewer(
    path: str | Path,
    port: int = 8050,
    host: str = "127.0.0.1",
    centroid_config: Any = None,
) -> None:
    """Build a Dash web app for browsing spectra in *path* and start the server.

    Parameters
    ----------
    path:
        Path to a Bruker ``.d`` directory or ``.mzML`` file.
    port:
        TCP port to listen on (default 8050).
    host:
        Host to bind the server to (default ``127.0.0.1``).
    centroid_config:
        :class:`~spxtacular.reader.CentroidConfig` instance with tdfpy centroiding
        parameters for Bruker ``.d`` files.  Ignored for ``.mzML`` files.

    Raises
    ------
    ImportError
        If ``dash`` is not installed.
    """
    if not _HAS_DASH:
        raise ImportError(
            "The viewer requires the 'dash' package, which is not installed. "
            "Install it with: pip install spxtacular[viewer]"
        )

    import plotly.graph_objects as go
    from dash import Dash, Input, Output, State, dash_table, dcc, html

    from .reader import CentroidConfig, DReader, Reader
    from .visualization import plot_spectrum

    path = Path(path)
    reader = Reader(path, centroid_config=centroid_config)
    reader.open()
    _is_dreader = isinstance(reader._reader, DReader)  # type: ignore[attr-defined]

    print(f"Building scan index for {path.name}...", flush=True)
    index_df = _build_index(reader)
    n = len(index_df)
    print(f"  {n:,} scans indexed.", flush=True)
    if n > 50_000:
        print(f"  Note: {n:,} scans — initial page load may take a moment.", flush=True)

    ms_levels: list[int] = (
        sorted(index_df["ms_level"].dropna().unique().astype(int).tolist()) if not index_df.empty else []
    )

    app = Dash(__name__)

    app.layout = html.Div(
        style={"fontFamily": "sans-serif", "maxWidth": "1400px", "margin": "0 auto", "padding": "16px"},
        children=[
            html.H2(f"spxtacular — {path.name}", style={"marginBottom": "12px"}),
            # Filter bar
            html.Div(
                style={
                    "display": "flex",
                    "gap": "20px",
                    "alignItems": "center",
                    "marginBottom": "10px",
                    "flexWrap": "wrap",
                },
                children=[
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "6px"},
                        children=[
                            html.Label("MS level:"),
                            dcc.Dropdown(
                                id="ms-level-filter",
                                options=[{"label": "All", "value": "all"}]
                                + [{"label": f"MS{lv}", "value": str(lv)} for lv in ms_levels],
                                value="all",
                                clearable=False,
                                style={"width": "110px"},
                            ),
                        ],
                    ),
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "6px"},
                        children=[
                            html.Label("RT (s):"),
                            dcc.Input(
                                id="rt-min-input",
                                type="number",
                                placeholder="min",
                                debounce=True,
                                style={"width": "80px"},
                            ),
                            html.Span("–"),
                            dcc.Input(
                                id="rt-max-input",
                                type="number",
                                placeholder="max",
                                debounce=True,
                                style={"width": "80px"},
                            ),
                        ],
                    ),
                ],
            ),
            # Centroid settings (only shown for Bruker .d files)
            html.Div(
                id="centroid-panel",
                style={
                    "display": "flex" if _is_dreader else "none",
                    "gap": "16px",
                    "alignItems": "center",
                    "marginBottom": "10px",
                    "padding": "8px 12px",
                    "background": "#f0f4ff",
                    "border": "1px solid #c8d8f8",
                    "borderRadius": "4px",
                    "flexWrap": "wrap",
                    "fontSize": "13px",
                },
                children=[
                    html.Span("Centroid:", style={"fontWeight": "bold", "color": "#333"}),
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "6px"},
                        children=[
                            html.Label("m/z tolerance:"),
                            dcc.Input(
                                id="centroid-mz-tol",
                                type="number",
                                value=8.0,
                                min=0,
                                debounce=True,
                                style={"width": "70px"},
                            ),
                            dcc.Dropdown(
                                id="centroid-mz-tol-type",
                                options=[{"label": "ppm", "value": "ppm"}, {"label": "Da", "value": "da"}],
                                value="ppm",
                                clearable=False,
                                style={"width": "70px"},
                            ),
                        ],
                    ),
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "6px"},
                        children=[
                            html.Label("IM tolerance:"),
                            dcc.Input(
                                id="centroid-im-tol",
                                type="number",
                                value=0.1,
                                min=0,
                                debounce=True,
                                style={"width": "70px"},
                            ),
                            dcc.Dropdown(
                                id="centroid-im-tol-type",
                                options=[
                                    {"label": "relative", "value": "relative"},
                                    {"label": "absolute", "value": "absolute"},
                                ],
                                value="relative",
                                clearable=False,
                                style={"width": "100px"},
                            ),
                        ],
                    ),
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "6px"},
                        children=[
                            html.Label("Min peaks:"),
                            dcc.Input(
                                id="centroid-min-peaks",
                                type="number",
                                value=3,
                                min=1,
                                step=1,
                                debounce=True,
                                style={"width": "60px"},
                            ),
                        ],
                    ),
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "6px"},
                        children=[
                            html.Label("Noise filter:"),
                            dcc.Dropdown(
                                id="centroid-noise-filter",
                                options=[
                                    {"label": "Off", "value": "off"},
                                    {"label": "MAD", "value": "mad"},
                                    {"label": "Percentile", "value": "percentile"},
                                    {"label": "Histogram", "value": "histogram"},
                                    {"label": "Baseline", "value": "baseline"},
                                ],
                                value="off",
                                clearable=False,
                                style={"width": "130px"},
                            ),
                        ],
                    ),
                ],
            ),
            # Scan table
            dash_table.DataTable(
                id="scan-table",
                columns=[
                    {"name": "Scan #", "id": "scan_number"},
                    {"name": "MS", "id": "ms_level"},
                    {"name": "RT (s)", "id": "rt"},
                    {"name": "Precursor m/z", "id": "precursor_mz"},
                ],
                data=index_df.to_dict("records"),
                page_size=100,
                page_action="native",
                sort_action="native",
                filter_action="none",
                style_table={"height": "320px", "overflowY": "auto"},
                style_cell={"textAlign": "left", "padding": "4px 10px", "fontSize": "13px"},
                style_header={"fontWeight": "bold", "backgroundColor": "#f4f4f4"},
                style_data_conditional=cast(
                    Any,
                    [{"if": {"state": "active"}, "backgroundColor": "#d0e8ff", "border": "1px solid #1a73e8"}],
                ),
            ),
            # Processing options
            html.Div(
                style={
                    "display": "flex",
                    "gap": "20px",
                    "alignItems": "center",
                    "marginTop": "10px",
                    "padding": "8px 12px",
                    "background": "#f9f9f9",
                    "border": "1px solid #e0e0e0",
                    "borderRadius": "4px",
                    "flexWrap": "wrap",
                    "fontSize": "13px",
                },
                children=[
                    html.Span("Processing:", style={"fontWeight": "bold", "color": "#333"}),
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "6px"},
                        children=[
                            html.Label("Denoise:"),
                            dcc.Dropdown(
                                id="denoise-method",
                                options=[
                                    {"label": "Off", "value": "off"},
                                    {"label": "MAD", "value": "mad"},
                                    {"label": "Percentile", "value": "percentile"},
                                    {"label": "Histogram", "value": "histogram"},
                                    {"label": "Baseline", "value": "baseline"},
                                ],
                                value="off",
                                clearable=False,
                                style={"width": "140px"},
                            ),
                        ],
                    ),
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "6px"},
                        children=[
                            html.Label("Rel. threshold (%):"),
                            dcc.Input(
                                id="rel-threshold",
                                type="number",
                                value=0,
                                min=0,
                                max=100,
                                step=0.1,
                                debounce=True,
                                style={"width": "70px"},
                            ),
                        ],
                    ),
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "6px"},
                        children=[
                            html.Label("Top N:"),
                            dcc.Input(
                                id="top-n-peaks",
                                type="number",
                                value=None,
                                min=1,
                                step=1,
                                placeholder="all",
                                debounce=True,
                                style={"width": "70px"},
                            ),
                        ],
                    ),
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "6px"},
                        children=[
                            html.Label("Normalize:"),
                            dcc.Checklist(
                                id="normalize-check",
                                options=[{"label": "", "value": "on"}],
                                value=[],
                                style={"display": "inline"},
                            ),
                        ],
                    ),
                ],
            ),
            # Spectrum info line
            html.Div(id="spectrum-info", style={"marginTop": "8px", "color": "#555", "fontSize": "13px"}),
            # Spectrum plot
            dcc.Graph(id="spectrum-graph", style={"height": "480px"}),
        ],
    )

    @app.callback(
        Output("scan-table", "data"),
        Input("ms-level-filter", "value"),
        Input("rt-min-input", "value"),
        Input("rt-max-input", "value"),
    )
    def filter_table(ms_level_str: str, rt_min_val: float | None, rt_max_val: float | None) -> list[dict]:
        df = index_df
        if ms_level_str != "all":
            df = df[df["ms_level"] == int(ms_level_str)]
        if rt_min_val is not None:
            df = df[df["rt"] >= float(rt_min_val)]
        if rt_max_val is not None:
            df = df[df["rt"] <= float(rt_max_val)]
        return df.to_dict("records")  # type: ignore[return-value]

    @app.callback(
        Output("spectrum-graph", "figure"),
        Output("spectrum-info", "children"),
        Input("scan-table", "active_cell"),
        State("scan-table", "derived_virtual_data"),
        Input("denoise-method", "value"),
        Input("rel-threshold", "value"),
        Input("top-n-peaks", "value"),
        Input("normalize-check", "value"),
        Input("centroid-mz-tol", "value"),
        Input("centroid-mz-tol-type", "value"),
        Input("centroid-im-tol", "value"),
        Input("centroid-im-tol-type", "value"),
        Input("centroid-min-peaks", "value"),
        Input("centroid-noise-filter", "value"),
    )
    def update_spectrum(
        active_cell: dict | None,
        virtual_data: list[dict] | None,
        denoise_method: str,
        rel_threshold: float | None,
        top_n: int | None,
        normalize_vals: list[str] | None,
        c_mz_tol: float | None,
        c_mz_tol_type: str | None,
        c_im_tol: float | None,
        c_im_tol_type: str | None,
        c_min_peaks: int | None,
        c_noise_filter: str | None,
    ) -> tuple[go.Figure, str]:
        # Apply centroid config to DReader before loading (no-op for MzmlReader)
        if _is_dreader:
            cast(DReader, reader._reader)._centroid_config = CentroidConfig(  # type: ignore[attr-defined]
                mz_tolerance=float(c_mz_tol or 8.0),
                mz_tolerance_type=cast(Any, c_mz_tol_type or "ppm"),
                im_tolerance=float(c_im_tol or 0.1),
                im_tolerance_type=cast(Any, c_im_tol_type or "relative"),
                min_peaks=int(c_min_peaks or 3),
                noise_filter=None if (c_noise_filter or "off") == "off" else cast(Any, c_noise_filter),
            )

        if active_cell is None or not virtual_data:
            return go.Figure(), ""
        row = virtual_data[active_cell["row"]]
        scan_number = row.get("scan_number")
        ms_level = row.get("ms_level", 1)
        rt_val = row.get("rt")
        precursor_mz = row.get("precursor_mz")

        if scan_number is None:
            return go.Figure(), "No scan number available for this entry."

        try:
            spec = reader.ms1[scan_number] if ms_level == 1 else reader.ms2[scan_number]
        except NotImplementedError:
            msg = (
                f"Scan {scan_number} | MS{ms_level} — "
                "direct spectrum lookup is not supported for DIA/PRM acquisition types."
            )
            return go.Figure(), msg
        except (KeyError, Exception) as exc:
            return go.Figure(), f"Failed to load scan {scan_number}: {exc}"

        # Apply processing options in order: denoise → threshold → top-N → normalize
        if denoise_method and denoise_method != "off":
            spec = spec.denoise(method=cast(Any, denoise_method))
        if rel_threshold is not None and float(rel_threshold) > 0 and len(spec.intensity) > 0:
            threshold = float(spec.intensity.max()) * float(rel_threshold) / 100.0
            spec = spec.filter(min_intensity=threshold)
        if top_n is not None and int(top_n) > 0:
            spec = spec.filter(top_n=int(top_n))
        if "on" in (normalize_vals or []):
            spec = spec.normalize()

        rt_str = f" | RT={rt_val:.2f}s" if rt_val is not None else ""
        prec_str = f" | precursor={precursor_mz:.4f}" if precursor_mz is not None else ""
        title = f"Scan {scan_number} | MS{ms_level}{rt_str}{prec_str}"
        color = "im" if spec.im is not None and len(spec.im) == len(spec.mz) else "charge"
        fig = plot_spectrum(spec, title=title, color=color)
        return fig, title

    url = f"http://{host}:{port}/"
    print(f"Launching viewer at {url}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        app.run(host=host, port=port, debug=False, threaded=False)
    finally:
        reader.close()


def main() -> None:
    """CLI entry point for ``spxtacular-view``."""
    if not _HAS_DASH:
        print(
            "Error: The viewer requires the 'dash' package, which is not installed.\n"
            "Install it with: pip install spxtacular[viewer]",
            file=sys.stderr,
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="spxtacular spectrum viewer — browse .d and .mzML files interactively",
        prog="spxtacular-view",
    )
    parser.add_argument("file", help="Path to a Bruker .d directory or .mzML file")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the viewer on (default: 8050)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    args = parser.parse_args()

    launch_viewer(args.file, port=args.port, host=args.host)
