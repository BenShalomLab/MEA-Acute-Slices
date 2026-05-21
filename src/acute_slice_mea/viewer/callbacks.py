"""Dash callbacks wiring up the trace viewer.

The viewer is a thin client over the cache; all heavy data lives in
``WellData`` instances cached per ``cache_dir`` path. Callbacks read those
objects and emit Plotly figures or store updates.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback_context, ctx, html, no_update
from dash.exceptions import PreventUpdate

from acute_slice_mea.jobs import (
    DEFAULT_PARAMS,
    JobsBackend,
    PIPELINE_VERSION,
)
from acute_slice_mea.library import LibraryIndex, RecordingEntry, WellEntry
from acute_slice_mea.probe_geometry import (
    MAXWELL_COLS,
    MAXWELL_PITCH_UM,
    MAXWELL_ROWS,
    WIDTH_UM,
    HEIGHT_UM,
)
from acute_slice_mea.viewer.data_loader import WellData
from acute_slice_mea.viewer.layout import PLATE_COLS, PLATE_ROWS

DEFAULT_SELECTION_SIZE = 6
TRACE_COLOR = "#0a7d7f"
BURST_COLOR = "rgba(196, 114, 8, 0.18)"


@lru_cache(maxsize=32)
def _load_well_data(cache_dir: str, _cache_version: float) -> WellData:
    """Load a WellData bundle.

    ``_cache_version`` is the mtime of the cache_meta.json sentinel — including
    it in the cache key means a fresh ``cache_meta.json`` (written when a job
    completes or after Recompute) automatically invalidates the LRU entry.
    """
    return WellData.load(cache_dir)


def _cache_version(cache_dir: Path) -> float:
    """Stable float key changing whenever the cache is (re)written."""
    sentinel = Path(cache_dir) / "cache_meta.json"
    if sentinel.exists():
        try:
            return sentinel.stat().st_mtime
        except OSError:
            return 0.0
    manifest = Path(cache_dir) / "manifest.json"
    if manifest.exists():
        try:
            return manifest.stat().st_mtime
        except OSError:
            return 0.0
    return 0.0


def register_all(
    app,
    library: LibraryIndex,
    *,
    jobs_backend: JobsBackend | None = None,
) -> None:
    """Bind all callbacks against the given Dash app + library.

    When ``jobs_backend`` is provided the spawn/cache UI (library pills,
    params sheet, jobs drawer, workers slider) is wired up; otherwise the
    dashboard runs in view-only mode against pre-built caches.
    """

    # -- helpers ------------------------------------------------------

    def _resolve_cache_dir(recording_id: str, well_id: str) -> Path | None:
        """Return the cache dir for this (recording, well) if a bundle exists.

        Two lookup paths: the library may already know the cache_dir from
        startup (cache-only mode, or a well that was cached when the dashboard
        launched), or we derive the canonical path from
        ``jobs_backend.cache_root`` and check for ``manifest.json`` on disk.
        The latter is what lets a freshly-completed job show up without a
        library rebuild.
        """
        well = library.find(recording_id, well_id)
        if well is not None and well.cache_dir:
            return Path(well.cache_dir)
        if jobs_backend is None:
            return None
        candidate = Path(
            jobs_backend.cache_root, *recording_id.split("/"), well_id
        )
        if (candidate / "manifest.json").exists():
            # Memoize on the WellEntry so subsequent lookups are O(1).
            if well is not None:
                well.cache_dir = str(candidate)
            return candidate
        return None

    def _well_data(recording_id: str | None, well_id: str | None) -> WellData | None:
        if not recording_id or not well_id:
            return None
        cache_dir = _resolve_cache_dir(recording_id, well_id)
        if cache_dir is None:
            return None
        return _load_well_data(str(cache_dir), _cache_version(cache_dir))

    # -- library: click → select recording ----------------------------

    @app.callback(
        Output("selected-recording-id", "data"),
        Input({"type": "lib-item", "recording_id": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def on_library_click(_n_clicks):
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            raise PreventUpdate
        return triggered.get("recording_id")

    @app.callback(
        Output({"type": "lib-item", "recording_id": ALL}, "className"),
        Input("selected-recording-id", "data"),
        State({"type": "lib-item", "recording_id": ALL}, "id"),
    )
    def style_library_items(selected, ids):
        out = []
        for entry in ids:
            base = "lib-item"
            if entry["recording_id"] == selected:
                base += " is-active"
            out.append(base)
        return out

    # -- library: search filter (clientside, hides non-matching items) -

    app.clientside_callback(
        """
        function(query, ids) {
            const q = (query || "").trim().toLowerCase();
            const out = [];
            for (const entry of ids) {
                const target = (entry.recording_id || "").toLowerCase();
                const hide = q && target.indexOf(q) === -1;
                out.push({ display: hide ? "none" : "" });
            }
            return out;
        }
        """,
        Output({"type": "lib-item", "recording_id": ALL}, "style"),
        Input("library-search", "value"),
        State({"type": "lib-item", "recording_id": ALL}, "id"),
    )

    # -- recording change → plate, default well, reset channels -------

    @app.callback(
        Output({"type": "plate-cell", "well_id": ALL}, "className"),
        Output({"type": "plate-cell", "well_id": ALL}, "disabled"),
        Output({"type": "plate-cell", "well_id": ALL}, "title"),
        Output("selected-well-id", "data"),
        Input("selected-recording-id", "data"),
        State({"type": "plate-cell", "well_id": ALL}, "id"),
        State("selected-well-id", "data"),
        prevent_initial_call=False,
    )
    def update_plate_for_recording(recording_id, plate_ids, current_well):
        rec = library.find_recording(recording_id) if recording_id else None
        available_wells: dict[str, WellEntry] = {w.well_id: w for w in (rec.wells if rec else [])}
        # Default to first available well when recording changes.
        if rec and rec.wells:
            chosen_well = (
                current_well
                if current_well in available_wells
                else rec.wells[0].well_id
            )
        else:
            chosen_well = None

        classes, disabled, titles = [], [], []
        for entry in plate_ids:
            well_id = entry["well_id"]
            cls = "plate-cell"
            if well_id in available_wells:
                cls += " is-ok"
                disabled.append(False)
                well = available_wells[well_id]
                ne = well.num_recorded_electrodes
                titles.append(
                    f"{_well_id_to_name(well_id)} · {ne if ne is not None else '?'} electrodes routed"
                )
            else:
                cls += " is-missing"
                disabled.append(True)
                titles.append(f"{_well_id_to_name(well_id)} · not in this recording")
            if well_id == chosen_well:
                cls += " is-selected"
            classes.append(cls)
        return classes, disabled, titles, chosen_well

    # -- plate cell click → select well -------------------------------

    @app.callback(
        Output("selected-well-id", "data", allow_duplicate=True),
        Input({"type": "plate-cell", "well_id": ALL}, "n_clicks"),
        State({"type": "plate-cell", "well_id": ALL}, "disabled"),
        prevent_initial_call=True,
    )
    def on_plate_click(_clicks, disabled_states):
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            raise PreventUpdate
        # Ignore clicks on disabled cells (Dash still fires the event on some browsers).
        triggered_idx = None
        for idx, item in enumerate(callback_context.inputs_list[0]):
            if item.get("id") == triggered:
                triggered_idx = idx
                break
        if triggered_idx is not None and disabled_states[triggered_idx]:
            raise PreventUpdate
        return triggered["well_id"]

    # -- well change → bump version, reset window, seed channels ------

    @app.callback(
        Output("well-version", "data"),
        Output("time-window", "data"),
        Output("selected-channels", "data"),
        Input("selected-recording-id", "data"),
        Input("selected-well-id", "data"),
        State("well-version", "data"),
        prevent_initial_call=False,
    )
    def on_well_change(recording_id, well_id, version):
        wd = _well_data(recording_id, well_id)
        if wd is None:
            return (version or 0) + 1, [0.0, 5.0], []
        duration = wd.duration_sec or 60.0
        window = [0.0, float(min(5.0, duration))]
        # Seed with the top-RMS routed electrodes so the viewer is never empty.
        seed = _top_rms_electrodes(wd, DEFAULT_SELECTION_SIZE)
        return (version or 0) + 1, window, seed

    # -- probe map ----------------------------------------------------

    @app.callback(
        Output("probe-map", "figure"),
        Input("well-version", "data"),
        Input("selected-channels", "data"),
        State("selected-recording-id", "data"),
        State("selected-well-id", "data"),
    )
    def render_probe_map(_version, selected_channels, recording_id, well_id):
        wd = _well_data(recording_id, well_id)
        return _build_probe_map_figure(wd, selected_channels or [])

    @app.callback(
        Output("selected-channels", "data", allow_duplicate=True),
        Input("probe-map", "selectedData"),
        State("selected-recording-id", "data"),
        State("selected-well-id", "data"),
        prevent_initial_call=True,
    )
    def on_probe_lasso(selected_data, recording_id, well_id):
        if not selected_data or "points" not in selected_data:
            raise PreventUpdate
        wd = _well_data(recording_id, well_id)
        if wd is None:
            raise PreventUpdate
        chosen = []
        for point in selected_data["points"]:
            eid = point.get("customdata")
            if eid is None:
                continue
            chosen.append(int(eid))
        return sorted(set(chosen))

    # -- quick-select chips ------------------------------------------

    @app.callback(
        Output("selected-channels", "data", allow_duplicate=True),
        Input({"type": "quick-select", "kind": ALL}, "n_clicks"),
        State("selected-recording-id", "data"),
        State("selected-well-id", "data"),
        prevent_initial_call=True,
    )
    def on_quick_select(_clicks, recording_id, well_id):
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            raise PreventUpdate
        wd = _well_data(recording_id, well_id)
        if wd is None:
            raise PreventUpdate
        kind = triggered["kind"]
        if kind == "none":
            return []
        if kind == "all":
            return wd.routed_electrode_ids()
        if kind == "top":
            return _top_rms_electrodes(wd, 12)
        if kind == "stride":
            ids = wd.routed_electrode_ids()
            return ids[::8] if ids else []
        raise PreventUpdate

    # -- window preset / gain controls -------------------------------

    @app.callback(
        Output("time-window", "data", allow_duplicate=True),
        Input({"type": "window-preset", "seconds": ALL}, "n_clicks"),
        State("time-window", "data"),
        State("selected-recording-id", "data"),
        State("selected-well-id", "data"),
        prevent_initial_call=True,
    )
    def on_window_preset(_clicks, window, recording_id, well_id):
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            raise PreventUpdate
        span = float(triggered["seconds"])
        wd = _well_data(recording_id, well_id)
        duration = wd.duration_sec if wd else 60.0
        center = (window[0] + window[1]) / 2 if window else span / 2
        return _clamp_window(center - span / 2, center + span / 2, duration)

    @app.callback(
        Output("gain", "data"),
        Input("gain-up", "n_clicks"),
        Input("gain-down", "n_clicks"),
        State("gain", "data"),
        prevent_initial_call=True,
    )
    def on_gain(_up, _down, gain):
        gain = float(gain or 1.0)
        if ctx.triggered_id == "gain-up":
            return min(8.0, gain * 1.4)
        if ctx.triggered_id == "gain-down":
            return max(0.25, gain / 1.4)
        raise PreventUpdate

    # -- scrubber: rangeslider drag → window store --------------------

    @app.callback(
        Output("time-window", "data", allow_duplicate=True),
        Input("scrubber-graph", "relayoutData"),
        State("time-window", "data"),
        State("selected-recording-id", "data"),
        State("selected-well-id", "data"),
        prevent_initial_call=True,
    )
    def on_scrubber_drag(relayout_data, current_window, recording_id, well_id):
        if not relayout_data:
            raise PreventUpdate
        a = relayout_data.get("xaxis.range[0]")
        b = relayout_data.get("xaxis.range[1]")
        if a is None and "xaxis.range" in relayout_data:
            a, b = relayout_data["xaxis.range"]
        if a is None or b is None:
            raise PreventUpdate
        wd = _well_data(recording_id, well_id)
        duration = wd.duration_sec if wd else 60.0
        return _clamp_window(float(a), float(b), duration)

    # -- traces figure ------------------------------------------------

    @app.callback(
        Output("traces-graph", "figure"),
        Output("scrubber-graph", "figure"),
        Output("status-text", "children"),
        Output("trace-count-readout", "children"),
        Output("window-readout", "children"),
        Output("gain-readout", "children"),
        Output("electrode-count-readout", "children"),
        Input("well-version", "data"),
        Input("selected-channels", "data"),
        Input("time-window", "data"),
        Input("gain", "data"),
        State("selected-recording-id", "data"),
        State("selected-well-id", "data"),
    )
    def render_traces(_version, selected_channels, window, gain, recording_id, well_id):
        wd = _well_data(recording_id, well_id)
        selected_channels = selected_channels or []
        window = window or [0.0, 5.0]
        gain = float(gain or 1.0)

        traces_fig = _build_traces_figure(wd, selected_channels, window, gain)
        scrubber_fig = _build_scrubber_figure(wd, window)

        n_traces = len(selected_channels)
        n_routed = len(wd.routed_electrode_ids()) if wd is not None else 0
        return (
            traces_fig,
            scrubber_fig,
            f"{n_traces} {'trace' if n_traces == 1 else 'traces'} on screen",
            f"{n_traces} {'trace' if n_traces == 1 else 'traces'}",
            f"{_fmt_time(window[0])} → {_fmt_time(window[1])}",
            f"{gain:.2f}×",
            f"{n_traces} / {n_routed} routed selected",
        )

    # -- crumbs + meta strip + notes ---------------------------------

    @app.callback(
        Output("crumbs-area", "children"),
        Input("selected-recording-id", "data"),
        Input("selected-well-id", "data"),
    )
    def update_crumbs(recording_id, well_id):
        rec = library.find_recording(recording_id) if recording_id else None
        if rec is None:
            return html.Span("No recording selected", className="cr-dim")
        crumbs = [
            html.Span(rec.sample, className="cr cr-strong"),
            html.Span("/", className="cr-sep"),
            html.Span(rec.iso_date, className="cr"),
            html.Span("/", className="cr-sep"),
            html.Span(rec.plate, className="cr"),
            html.Span("/", className="cr-sep"),
            html.Span(rec.scan, className="cr cr-pill"),
            html.Span("/", className="cr-sep"),
            html.Span(f"run {rec.run}", className="cr"),
        ]
        if well_id:
            crumbs.extend(
                [
                    html.Span("·", className="cr-sep cr-sep-bold"),
                    html.Span(f"Well {_well_id_to_name(well_id)}", className="cr cr-well"),
                ]
            )
        return crumbs

    @app.callback(
        Output("meta-strip", "children"),
        Output("notes-area", "children"),
        Input("selected-recording-id", "data"),
        Input("selected-well-id", "data"),
    )
    def update_meta(recording_id, well_id):
        wd = _well_data(recording_id, well_id)
        rec = library.find_recording(recording_id) if recording_id else None
        if wd is None:
            placeholder = [_meta_item(label, "—") for label in (
                "Sample rate", "Duration", "Routed electrodes", "Filter", "Reference", "Bursts"
            )]
            return placeholder, []
        meta = [
            _meta_item("Sample rate", f"{wd.sample_rate_hz:.0f} Hz" if wd.sample_rate_hz else "—"),
            _meta_item("Duration", _fmt_time(wd.duration_sec)),
            _meta_item("Routed electrodes", str(len(wd.routed_electrode_ids()))),
            _meta_item("Filter", "0.5–300 Hz (LFP)", mono=True),
            _meta_item("Reference", "Common avg.", mono=True),
            _meta_item("Bursts", str(len(wd.bursts))),
        ]
        notes_children: list = []
        if rec and rec.notes:
            notes_children = [
                html.Div(
                    className="notes",
                    children=[
                        html.Div("Notes", className="notes-h"),
                        html.Div(rec.notes, className="notes-body"),
                    ],
                )
            ]
        return meta, notes_children

    # -------------------------------------------------------------------
    # Jobs / spawn-and-cache callbacks (only registered in spawn mode).
    # -------------------------------------------------------------------
    if jobs_backend is not None:
        _register_jobs_callbacks(app, library, jobs_backend, _load_well_data)


# =============================================================================
# Figure builders
# =============================================================================


def _build_probe_map_figure(wd: WellData | None, selected_channels: list[int]) -> go.Figure:
    fig = go.Figure()
    # Full chip grid as a faint backdrop. We draw it as a single shape rather
    # than 26k markers to keep render cost low.
    fig.add_shape(
        type="rect",
        x0=-MAXWELL_PITCH_UM, x1=WIDTH_UM,
        y0=-MAXWELL_PITCH_UM, y1=HEIGHT_UM,
        line=dict(color="#cfcabc", width=1),
        fillcolor="rgba(243, 241, 234, 0.4)",
        layer="below",
    )

    if wd is None:
        fig.update_layout(_probe_map_layout(empty=True))
        return fig

    routed = (wd.probe or {}).get("routed") or []
    if not routed:
        # Fall back to electrodes.csv if probe.json is missing.
        recorded = wd.electrodes[wd.electrodes["recorded"].astype(bool)]
        routed = [
            {
                "electrode_id": int(row.electrode_id),
                "x_um": float(row.x_um),
                "y_um": float(row.y_um),
                "rms_uv": float(getattr(row, "rms_uv", float("nan"))) if "rms_uv" in recorded.columns else None,
            }
            for row in recorded.itertuples(index=False)
        ]

    selected_set = set(int(eid) for eid in selected_channels)
    rms_values = [r.get("rms_uv") for r in routed if r.get("rms_uv") is not None]
    rms_max = max(rms_values) if rms_values else 1.0

    xs, ys, custom, colors, sizes = [], [], [], [], []
    for entry in routed:
        eid = int(entry["electrode_id"])
        xs.append(entry["x_um"])
        ys.append(entry["y_um"])
        custom.append(eid)
        rms = entry.get("rms_uv")
        intensity = 0.0 if rms is None or rms_max == 0 else min(1.0, float(rms) / rms_max)
        colors.append(_interp_color(intensity))
        # Pixel sizing. At chip-fit zoom the 17.5 μm pitch is ~3 px wide, so
        # markers must stay tiny or they drown the grid. Selected state below
        # bumps to 6 px with an outline so the highlight reads at any zoom.
        sizes.append(2 + 2 * intensity)

    fig.add_trace(
        go.Scattergl(
            x=xs, y=ys,
            customdata=custom,
            mode="markers",
            name="routed",
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(width=0),
            ),
            hovertemplate="E%{customdata}<br>x=%{x:.0f} µm<br>y=%{y:.0f} µm<extra></extra>",
            selectedpoints=[i for i, eid in enumerate(custom) if eid in selected_set] or None,
            # scattergl.selected.Marker only supports color/opacity/size — no
            # line/outline. Use color + a modest size bump for the highlight.
            selected=dict(marker=dict(color=TRACE_COLOR, size=6)),
            unselected=dict(marker=dict(opacity=0.55)),
        )
    )

    fig.update_layout(_probe_map_layout())
    return fig


def _probe_map_layout(*, empty: bool = False) -> dict:
    return dict(
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f3f1ea",
        xaxis=dict(
            range=[-MAXWELL_PITCH_UM, WIDTH_UM],
            showgrid=False,
            zeroline=False,
            visible=False,
            fixedrange=False,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            range=[HEIGHT_UM, -MAXWELL_PITCH_UM],  # invert so chip "row 0" is at top
            showgrid=False,
            zeroline=False,
            visible=False,
            fixedrange=False,
        ),
        showlegend=False,
        dragmode="lasso",
        annotations=[]
        if not empty
        else [
            dict(
                text="Pick a well to load the chip", showarrow=False,
                x=0.5, y=0.5, xref="paper", yref="paper",
                font=dict(family="IBM Plex Sans", size=14, color="#8a8472"),
            )
        ],
    )


def _build_traces_figure(
    wd: WellData | None,
    selected_channels: list[int],
    window: list[float],
    gain: float,
) -> go.Figure:
    fig = go.Figure()
    if wd is None or not selected_channels:
        fig.update_layout(
            margin=dict(l=20, r=20, t=10, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#faf9f5",
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            annotations=[
                dict(
                    text=("Pick a recording, well, then electrodes on the chip"
                          if wd is None else
                          "Lasso-select electrodes on the chip to view their traces"),
                    showarrow=False,
                    x=0.5, y=0.5, xref="paper", yref="paper",
                    font=dict(family="IBM Plex Serif", size=16, color="#8a8472"),
                )
            ],
        )
        return fig

    t0, t1 = float(window[0]), float(window[1])
    payloads = wd.traces_for(selected_channels, signal="lfp", t0=t0, t1=t1)
    if not payloads:
        fig.update_layout(
            annotations=[
                dict(
                    text="Selected electrodes have no cached LFP traces",
                    showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
                    font=dict(family="IBM Plex Sans", size=12, color="#8a8472"),
                )
            ],
        )
        return fig

    # Stack traces vertically with constant vertical offset.
    spacing = 250.0 / max(gain, 0.1)  # microvolts between rows
    for idx, payload in enumerate(payloads):
        offset = -idx * spacing
        fig.add_trace(
            go.Scattergl(
                x=payload["time_sec"],
                y=payload["value"] * gain + offset,
                mode="lines",
                line=dict(color=TRACE_COLOR, width=1),
                name=f"E{payload['electrode_id']}",
                hovertemplate="t=%{x:.3f}s<br>%{text}<extra></extra>",
                text=[f"E{payload['electrode_id']}"] * len(payload["time_sec"]),
            )
        )

    # Burst shading
    for burst in wd.bursts:
        if burst["t_end_s"] < t0 or burst["t_start_s"] > t1:
            continue
        fig.add_vrect(
            x0=burst["t_start_s"], x1=burst["t_end_s"],
            fillcolor=BURST_COLOR, line_width=0, layer="below",
        )

    y_min = -(len(payloads) - 0.5) * spacing
    y_max = 0.5 * spacing
    label_positions = [-i * spacing for i in range(len(payloads))]
    fig.update_layout(
        margin=dict(l=70, r=24, t=12, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#faf9f5",
        showlegend=False,
        xaxis=dict(
            range=[t0, t1],
            showgrid=True,
            gridcolor="rgba(26, 25, 22, 0.06)",
            zeroline=False,
            title=dict(text="Time (s)", font=dict(family="IBM Plex Mono", size=10, color="#8a8472")),
        ),
        yaxis=dict(
            range=[y_min, y_max],
            showgrid=False,
            zeroline=False,
            tickmode="array",
            tickvals=label_positions,
            ticktext=[f"E{p['electrode_id']}" for p in payloads],
            tickfont=dict(family="IBM Plex Mono", size=10, color="#1a1916"),
        ),
        hovermode="closest",
    )
    return fig


def _build_scrubber_figure(wd: WellData | None, window: list[float]) -> go.Figure:
    fig = go.Figure()
    duration = (wd.duration_sec if wd else 60.0) or 60.0
    fig.add_shape(
        type="rect",
        x0=0, x1=duration, y0=0, y1=1,
        fillcolor="#f3f1ea", line_width=0, layer="below",
    )
    if wd is not None:
        for burst in wd.bursts:
            fig.add_shape(
                type="line",
                x0=burst["center_s"], x1=burst["center_s"],
                y0=0, y1=1,
                line=dict(color="#c47208", width=1),
            )
    # Highlighted current window
    fig.add_shape(
        type="rect",
        x0=window[0], x1=window[1], y0=0, y1=1,
        fillcolor="rgba(10, 125, 127, 0.18)",
        line=dict(color="#0a7d7f", width=2),
    )
    fig.update_layout(
        margin=dict(l=12, r=12, t=2, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, duration],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            rangeslider=dict(visible=False),
            fixedrange=False,
        ),
        yaxis=dict(visible=False, range=[0, 1], fixedrange=True),
        dragmode="zoom",
        showlegend=False,
        height=44,
    )
    return fig


# =============================================================================
# Small helpers
# =============================================================================


def _top_rms_electrodes(wd: WellData, n: int) -> list[int]:
    rms = wd.rms_by_electrode()
    if not rms:
        routed = wd.routed_electrode_ids()
        return routed[:n]
    ordered = sorted(rms.items(), key=lambda kv: kv[1], reverse=True)
    return [eid for eid, _ in ordered[:n]]


def _clamp_window(a: float, b: float, duration: float | None) -> list[float]:
    duration = float(duration or 60.0)
    if b <= a:
        b = a + 0.2
    span = b - a
    a = max(0.0, a)
    b = min(duration, b)
    if b - a < 0.2:
        a = max(0.0, duration - 0.2)
        b = a + 0.2
    if b - a > duration:
        a = 0.0
        b = duration
    if a + span <= duration and span >= 0.2:
        b = a + span
    return [float(a), float(b)]


def _fmt_time(seconds: float | None) -> str:
    if seconds is None or seconds <= 0:
        return "0.00 s"
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}:{int(round(rest)):02d}"


def _well_id_to_name(well_id: str) -> str:
    """well000..well023 → A1..D6 (MaxWell row-major 6 cols).

    Falls back to the raw id when not in canonical form (e.g., "well042").
    """
    if not well_id.startswith("well"):
        return well_id
    try:
        idx = int(well_id[4:])
    except ValueError:
        return well_id
    if not 0 <= idx < len(PLATE_ROWS) * len(PLATE_COLS):
        return well_id
    return f"{PLATE_ROWS[idx // 6]}{PLATE_COLS[idx % 6]}"


def _meta_item(label: str, value: Any, mono: bool = False) -> html.Div:
    cls = "meta-value mono" if mono else "meta-value"
    return html.Div(
        className="meta-item",
        children=[
            html.Div(label, className="meta-label"),
            html.Div(str(value), className=cls),
        ],
    )


def _interp_color(intensity: float) -> str:
    """Interpolate paper-2 (#cfcabc) → accent (#0a7d7f) by ``intensity`` ∈ [0,1]."""
    c0 = (0xcf, 0xca, 0xbc)
    c1 = (0x0a, 0x7d, 0x7f)
    t = max(0.0, min(1.0, float(intensity)))
    rgb = tuple(int(round(c0[i] + (c1[i] - c0[i]) * t)) for i in range(3))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


# =============================================================================
# Jobs / spawn-and-cache callbacks
# =============================================================================


# Map backend status → (pill label, pill className, action label, action className).
_PILL_STYLES = {
    "idle":      ("not computed",  "pill idle",  "Run",       "btn small primary"),
    "queued":    ("queued",        "pill live",  "Cancel",    "btn small"),
    "running":   ("running",       "pill live",  "Cancel",    "btn small"),
    "cached":    ("cached",        "pill ok",    "Recompute", "btn small ghost"),
    "stale":     ("stale",         "pill warn",  "Recompute", "btn small"),
    "failed":    ("failed",        "pill err",   "Retry",     "btn small"),
    "cancelled": ("cancelled",     "pill idle",  "Run",       "btn small primary"),
}


def _format_bytes(n: float | int | None) -> str:
    if not n:
        return "—"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def _format_eta(job) -> str:
    if job.state != "running" or not job.started_at:
        return ""
    # We don't have a real ETA from run_analysis; estimate from progress.
    progress = max(job.progress or 0.0, 0.01)
    elapsed = max(1.0, (job.last_heartbeat or job.started_at) - job.started_at)
    remaining = elapsed * (1.0 - progress) / progress
    if remaining <= 0:
        return "finishing…"
    if remaining < 60:
        return f"{int(remaining)}s left"
    m, s = divmod(int(remaining), 60)
    return f"{m}m {s}s left"


def _register_jobs_callbacks(app, library, backend, load_well_data):
    """Bind callbacks that drive the library pills, jobs drawer, params sheet,
    workers slider, and empty-state CTA. Called only when ``--data-root`` was
    passed at startup.
    """

    # ── library row pills + progress bars + action buttons ────────────────
    @app.callback(
        Output({"type": "lib-status-pill", "recording_id": ALL, "well_id": ALL}, "children"),
        Output({"type": "lib-status-pill", "recording_id": ALL, "well_id": ALL}, "className"),
        Output({"type": "lib-progress", "recording_id": ALL, "well_id": ALL}, "children"),
        Output({"type": "lib-action", "recording_id": ALL, "well_id": ALL}, "children"),
        Output({"type": "lib-action", "recording_id": ALL, "well_id": ALL}, "className"),
        Input("jobs-poll", "n_intervals"),
        State({"type": "lib-status-pill", "recording_id": ALL, "well_id": ALL}, "id"),
    )
    def update_library_rows(_tick, pill_ids):
        backend.refresh()
        pill_children, pill_classes = [], []
        progress_children = []
        action_children, action_classes = [], []
        for entry in pill_ids:
            rid, wid = entry["recording_id"], entry["well_id"]
            status = backend.get_status_for(rid, wid)
            label, pill_class, action_label, action_class = _PILL_STYLES.get(
                status, _PILL_STYLES["idle"]
            )
            # If running, show the progress %.
            active_job = next(
                (j for j in backend.get_jobs_for(rid, wid) if j.state in ("queued", "running")),
                None,
            )
            if active_job and active_job.state == "running":
                pct = int((active_job.progress or 0.0) * 100)
                label = f"running {pct}%"
                width = f"{pct}%"
            elif active_job and active_job.state == "queued":
                width = "0%"
            elif status == "cached":
                width = "100%"
            else:
                width = "0%"

            pill_children.append([html.I(className="dot"), html.Span(label)])
            pill_classes.append(pill_class)
            progress_children.append(html.Span(style={"width": width}))
            action_children.append(action_label)
            action_classes.append(action_class)
        return pill_children, pill_classes, progress_children, action_children, action_classes

    # ── jobs badge + cache total readout ─────────────────────────────────
    @app.callback(
        Output("jobs-badge", "children"),
        Output("cache-total-readout", "children"),
        Input("jobs-poll", "n_intervals"),
    )
    def update_topbar_readout(_tick):
        active = backend.get_active_jobs()
        cache_total = backend.get_cache_total()
        cache_label = (
            f"{cache_total['count']} cached · {_format_bytes(cache_total['bytes'])}"
            if cache_total["count"]
            else "no caches"
        )
        return str(len(active)), cache_label

    # ── jobs drawer body ─────────────────────────────────────────────────
    @app.callback(
        Output("jobs-drawer-body", "children"),
        Input("jobs-poll", "n_intervals"),
        Input("jobs-drawer-open", "data"),
    )
    def update_drawer_body(_tick, open_):
        if not open_:
            raise PreventUpdate
        active = backend.get_active_jobs()
        recent = backend.get_recent_jobs()
        sections: list = []
        if not active and not recent:
            return [html.Div("No jobs yet. Submit one from the library.", className="ax-drawer-empty")]
        if active:
            sections.append(html.Div("Active", className="ax-drawer-sect"))
            sections.extend(_render_job_card(j, active=True) for j in active)
        if recent:
            sections.append(html.Div("Recent", className="ax-drawer-sect"))
            sections.extend(_render_job_card(j, active=False) for j in recent)
        return sections

    # ── drawer toggle ────────────────────────────────────────────────────
    @app.callback(
        Output("jobs-drawer-open", "data"),
        Output("jobs-drawer", "className"),
        Input("jobs-drawer-toggle", "n_clicks"),
        Input("jobs-drawer-close", "n_clicks"),
        State("jobs-drawer-open", "data"),
        prevent_initial_call=True,
    )
    def toggle_drawer(_open, _close, current):
        triggered = ctx.triggered_id
        if triggered == "jobs-drawer-close":
            new_open = False
        else:
            new_open = not bool(current)
        return new_open, "ax-drawer" if new_open else "ax-drawer hidden"

    # ── lib-action click: route to backend or open the sheet ─────────────
    @app.callback(
        Output("params-sheet-state", "data"),
        Output("params-sheet", "className"),
        Output("params-sheet-subtitle", "children"),
        Output("params-overwrite-row", "className"),
        Output("params-overwrite", "value"),
        Output("params-sheet-error", "children"),
        Output("params-sheet-error", "className"),
        Input({"type": "lib-action", "recording_id": ALL, "well_id": ALL}, "n_clicks"),
        Input("params-cancel", "n_clicks"),
        prevent_initial_call=True,
    )
    def on_lib_action(action_clicks, _cancel_clicks):
        triggered = ctx.triggered_id
        if triggered == "params-cancel":
            return (
                None,
                "ax-sheet hidden",
                "",
                "ax-sheet-row hidden",
                [],
                "",
                "ax-sheet-error hidden",
            )
        if not isinstance(triggered, dict):
            raise PreventUpdate
        if not any((c or 0) > 0 for c in (action_clicks or [])):
            raise PreventUpdate
        rid = triggered["recording_id"]
        wid = triggered["well_id"]
        status = backend.get_status_for(rid, wid)
        if status in ("queued", "running"):
            # Cancel the in-flight job.
            jobs = [j for j in backend.get_jobs_for(rid, wid) if j.state in ("queued", "running")]
            if jobs:
                backend.cancel(jobs[0].id)
            return (
                None,
                "ax-sheet hidden",
                "",
                "ax-sheet-row hidden",
                [],
                "",
                "ax-sheet-error hidden",
            )
        if status == "failed":
            # Retry directly.
            jobs = [j for j in backend.get_jobs_for(rid, wid) if j.state == "failed"]
            if jobs:
                backend.retry(jobs[0].id)
            return (
                None,
                "ax-sheet hidden",
                "",
                "ax-sheet-row hidden",
                [],
                "",
                "ax-sheet-error hidden",
            )
        # Open params sheet for run / recompute.
        rec = library.find_recording(rid)
        well_label = _well_id_to_name(wid)
        subtitle = f"{rec.sample if rec else rid} · well {well_label}"
        cache_exists = backend.get_cache_for(rid, wid) is not None
        overwrite_cls = "ax-sheet-row" if cache_exists else "ax-sheet-row hidden"
        state = {"recording_id": rid, "well_id": wid, "cache_exists": cache_exists}
        return state, "ax-sheet", subtitle, overwrite_cls, [], "", "ax-sheet-error hidden"

    # ── params sheet submit ──────────────────────────────────────────────
    @app.callback(
        Output("params-sheet", "className", allow_duplicate=True),
        Output("params-sheet-error", "children", allow_duplicate=True),
        Output("params-sheet-error", "className", allow_duplicate=True),
        Output("params-sheet-state", "data", allow_duplicate=True),
        Input("params-submit", "n_clicks"),
        State("params-sheet-state", "data"),
        State("params-band-low", "value"),
        State("params-band-high", "value"),
        State("params-reference", "value"),
        State("params-decimation", "value"),
        State("params-overwrite", "value"),
        prevent_initial_call=True,
    )
    def submit_params(_clicks, state, low, high, reference, decimation, overwrite):
        if not state:
            raise PreventUpdate
        rid = state["recording_id"]
        wid = state["well_id"]
        rec = library.find_recording(rid)
        well = library.find(rid, wid) if rec else None
        raw_path = (well.raw_path if well else None) or (rec.raw_path if rec else None)
        params = {
            "band": [float(low or 0.5), float(high or 300)],
            "reference": str(reference or "CAR"),
            "decimation": int(decimation or 1),
            "channels": "routed64",
        }
        result = backend.submit(
            recording_id=rid,
            well_id=wid,
            raw_path=raw_path,
            rec_name=well.rec_name if well else None,
            params=params,
            overwrite=bool(overwrite),
            recording_label=rec.label if rec else None,
            well_label=_well_id_to_name(wid),
        )
        if "error" in result:
            if result["error"] == "cache-exists":
                msg = "A cache already exists for this well — tick Overwrite to replace it."
            elif result["error"] == "already-in-flight":
                msg = "A job for this well is already in flight."
            else:
                msg = result["error"]
            return "ax-sheet", msg, "ax-sheet-error", state
        # Success: close the sheet, clear state.
        return "ax-sheet hidden", "", "ax-sheet-error hidden", None

    # ── per-job retry / cancel from drawer ───────────────────────────────
    @app.callback(
        Output("jobs-drawer-body", "children", allow_duplicate=True),
        Input({"type": "drawer-action", "job_id": ALL, "kind": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def on_drawer_action(_clicks):
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            raise PreventUpdate
        if not any((c or 0) > 0 for c in (_clicks or [])):
            raise PreventUpdate
        kind = triggered["kind"]
        if kind == "retry":
            backend.retry(triggered["job_id"])
        elif kind == "cancel":
            backend.cancel(triggered["job_id"])
        # Force a redraw on the next poll by returning no_update; the next
        # jobs-poll tick will refresh the drawer body.
        return no_update

    # ── workers slider ───────────────────────────────────────────────────
    @app.callback(
        Output("workers-slider", "value"),
        Input("workers-slider", "value"),
        prevent_initial_call=True,
    )
    def on_workers_slider(value):
        if value is None:
            raise PreventUpdate
        return backend.set_worker_cap(int(value))

    # ── debug reset ──────────────────────────────────────────────────────
    @app.callback(
        Output("jobs-badge", "children", allow_duplicate=True),
        Input("debug-reset-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def on_debug_reset(n):
        if not n:
            raise PreventUpdate
        backend.debug_reset()
        return "0"

    # ── empty-state CTA for the center stage ─────────────────────────────
    @app.callback(
        Output("center-stage-cta", "children"),
        Output("center-stage-cta", "className"),
        Input("jobs-poll", "n_intervals"),
        Input("selected-recording-id", "data"),
        Input("selected-well-id", "data"),
    )
    def update_cta(_tick, recording_id, well_id):
        if not recording_id or not well_id:
            return [], "ax-cta hidden"
        status = backend.get_status_for(recording_id, well_id)
        if status in ("cached", "stale"):
            return [], "ax-cta hidden"
        rec = library.find_recording(recording_id)
        sample = rec.sample if rec else recording_id
        well_label = _well_id_to_name(well_id)
        if status in ("queued", "running"):
            active = next(
                (j for j in backend.get_jobs_for(recording_id, well_id)
                 if j.state in ("queued", "running")),
                None,
            )
            pct = int((active.progress or 0.0) * 100) if active else 0
            stage = (active.stage or "running") if active else "queued"
            eta = _format_eta(active) if active else ""
            return _cta_running_card(sample, well_label, pct, stage, eta), "ax-cta"
        if status == "failed":
            failed = next(
                (j for j in backend.get_jobs_for(recording_id, well_id) if j.state == "failed"),
                None,
            )
            error = (failed.error if failed else "unknown error") or "unknown error"
            return _cta_failed_card(sample, well_label, error, recording_id, well_id), "ax-cta"
        # idle
        return _cta_idle_card(sample, well_label, recording_id, well_id), "ax-cta"


# ── small renderers ────────────────────────────────────────────────────────


def _cta_idle_card(sample: str, well_label: str, recording_id: str, well_id: str):
    return html.Div(
        className="ax-cta-card",
        children=[
            html.Div(
                className="ax-cta-main",
                children=[
                    html.Div("No LFP cache for this well yet.", className="ax-cta-title"),
                    html.Div(
                        f"{sample} · well {well_label}. Hit Run on the library row to populate the viewer.",
                        className="ax-cta-body",
                    ),
                    html.Div(
                        f"Defaults: 0.5 – 300 Hz · CAR reference · pipeline {PIPELINE_VERSION}",
                        className="ax-cta-defaults",
                    ),
                ],
            ),
        ],
    )


def _cta_running_card(sample: str, well_label: str, pct: int, stage: str, eta: str):
    return html.Div(
        className="ax-cta-card",
        children=[
            html.Div(
                className="ax-cta-main",
                children=[
                    html.Div(f"Running LFP analysis · {eta}".rstrip(" ·"), className="ax-cta-title"),
                    html.Div(
                        f"{sample} · well {well_label}. We'll auto-open the trace viewer once the cache lands. "
                        f"Closing the tab won't stop the job.",
                        className="ax-cta-body",
                    ),
                    html.Div(f"Stage: {stage}", className="ax-cta-defaults"),
                    html.Div(
                        className="pbar thick",
                        children=html.Span(style={"width": f"{pct}%"}),
                    ),
                ],
            ),
            html.Div(
                className="ax-cta-pct",
                children=[
                    html.Div(f"{pct}", className="ax-cta-pct-num"),
                    html.Div("%", className="ax-cta-pct-unit"),
                ],
            ),
        ],
    )


def _cta_failed_card(sample: str, well_label: str, error: str, recording_id: str, well_id: str):
    return html.Div(
        className="ax-cta-card",
        children=[
            html.Div(
                className="ax-cta-main",
                children=[
                    html.Div("Last analysis failed.", className="ax-cta-title"),
                    html.Div(
                        f"{sample} · well {well_label}. {error}",
                        className="ax-cta-body",
                    ),
                ],
            ),
        ],
    )


def _render_job_card(job, *, active: bool):
    state_class = {
        "running": "pill live",
        "queued": "pill live",
        "cached": "pill ok",
        "failed": "pill err",
        "cancelled": "pill idle",
    }.get(job.state, "pill idle")
    label = job.state
    if job.state == "running":
        label = f"running {int((job.progress or 0.0) * 100)}%"
    title = job.recording_label or job.recording_id
    well_label = job.well_label or job.well_id
    subtitle = f"{title} · {well_label}"
    actions: list = []
    if job.state in ("queued", "running"):
        actions.append(
            html.Button(
                "Cancel",
                id={"type": "drawer-action", "job_id": job.id, "kind": "cancel"},
                className="btn small",
                n_clicks=0,
            )
        )
    elif job.state == "failed":
        actions.append(
            html.Button(
                "Retry",
                id={"type": "drawer-action", "job_id": job.id, "kind": "retry"},
                className="btn small",
                n_clicks=0,
            )
        )
    return html.Div(
        className="ax-job",
        children=[
            html.Div(
                className="ax-job-h",
                children=[
                    html.Div(subtitle, className="ax-job-name"),
                    html.Span([html.I(className="dot"), html.Span(label)], className=state_class),
                ],
            ),
            html.Div(
                className="ax-job-meta",
                children=[
                    html.Span(job.hash, className="mono"),
                    html.Span(_format_eta(job) if job.state == "running" else "", className="mono"),
                ],
            ),
            html.Div(actions, className="ax-job-actions") if actions else None,
        ],
    )
