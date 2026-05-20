"""Dash layout for the Acute Slice trace viewer.

Class names mirror the design prototype at
``/tmp/design_pkg/acute-slice/project/styles.css`` so the CSS port in
``assets/styles.css`` applies directly. The pane structure follows the
prototype: library (left), center column (controls + traces + meta), right
pane (plate map + channel picker).
"""

from __future__ import annotations

from dataclasses import dataclass

from dash import dcc, html

from acute_slice_mea.library import LibraryIndex, RecordingEntry

PRESET_WINDOW_SECONDS = [1, 2, 5, 10, 30]
DEFAULT_WINDOW_SECONDS = 5

# 4 x 6 MaxWell 24-well plate. We render the *fixed* layout; missing wells
# are styled differently based on whether they appear in the recording.
PLATE_ROWS = ["A", "B", "C", "D"]
PLATE_COLS = [1, 2, 3, 4, 5, 6]


@dataclass
class TopBarState:
    """Returned by the breadcrumb callback for ``crumbs-area``."""

    recording: RecordingEntry | None
    well_id: str | None


def build_layout(library: LibraryIndex) -> html.Div:
    return html.Div(
        className="app",
        children=[
            _top_bar(library),
            html.Div(
                className="workspace",
                children=[
                    _library_pane(library),
                    _center_column(),
                    _right_pane(),
                ],
            ),
            # Hidden stores
            dcc.Store(id="selected-recording-id"),
            dcc.Store(id="selected-well-id"),
            dcc.Store(id="selected-channels", data=[]),
            dcc.Store(id="time-window", data=[0.0, float(DEFAULT_WINDOW_SECONDS)]),
            dcc.Store(id="gain", data=1.0),
            # Bumped whenever the active (recording, well) changes so downstream
            # callbacks can flush their per-well caches.
            dcc.Store(id="well-version", data=0),
        ],
    )


# =============================================================================
# Top bar
# =============================================================================


def _top_bar(library: LibraryIndex) -> html.Div:
    return html.Div(
        className="topbar",
        children=[
            html.Div(
                className="brand",
                children=[
                    html.Div(
                        className="brand-mark",
                        children=html.Div("≈", style={"fontSize": "16px", "fontWeight": 600}),
                        **{"aria-hidden": "true"},
                    ),
                    html.Div(
                        className="brand-text",
                        children=[
                            html.Div("Acute Slice · Trace Viewer", className="brand-title"),
                            html.Div("LFP · bandpass 0.5–300 Hz", className="brand-sub"),
                        ],
                    ),
                ],
            ),
            html.Div(id="crumbs-area", className="crumbs"),
            html.Div(
                className="topbar-right",
                children=[
                    html.Span(
                        id="status-chip",
                        className="status-chip",
                        children=[
                            html.I(className="dot dot-live"),
                            html.Span("0 traces on screen", id="status-text"),
                        ],
                    ),
                    html.Span(
                        f"{len(library.recordings)} recording(s)",
                        className="status-chip",
                    ),
                ],
            ),
        ],
    )


# =============================================================================
# Library pane (left)
# =============================================================================


def _library_pane(library: LibraryIndex) -> html.Aside:
    grouped = library.grouped_by_date()
    return html.Aside(
        className="pane pane-library",
        children=[
            html.Div(
                className="pane-head",
                children=[
                    html.Div("Recordings", className="pane-title"),
                    html.Span(str(len(library.recordings)), className="muted"),
                ],
            ),
            html.Label(
                className="searchbox",
                children=[
                    html.Span("⌕"),
                    dcc.Input(
                        id="library-search",
                        type="text",
                        placeholder="Search sample, plate, scan…",
                        debounce=False,
                    ),
                ],
            ),
            html.Div(
                id="library-list",
                className="lib-scroll",
                children=[_library_group(date, items) for date, items in grouped]
                or [html.Div("No recordings found in cache.", className="empty-state-s", style={"padding": "24px"})],
            ),
            html.Div(
                className="lib-foot",
                children=html.Div(
                    f"Cache root: {library.root}",
                    style={"fontFamily": "var(--font-mono)", "fontSize": "10.5px"},
                ),
            ),
        ],
    )


def _library_group(date: str, items: list[RecordingEntry]) -> html.Div:
    return html.Div(
        className="lib-group",
        children=[
            html.Div(
                className="lib-group-head",
                children=[html.Span(date), html.Span(str(len(items)), className="muted")],
            ),
            html.Ul(
                className="lib-list",
                children=[html.Li(_library_item(rec)) for rec in items],
            ),
        ],
    )


def _library_item(rec: RecordingEntry) -> html.Button:
    scan_class = "lib-scan lib-scan-network" if rec.scan.lower() == "network" else "lib-scan"
    return html.Button(
        id={"type": "lib-item", "recording_id": rec.recording_id},
        className="lib-item",
        n_clicks=0,
        children=[
            html.Div(
                className="lib-item-top",
                children=[
                    html.Span(rec.sample, className="lib-sample"),
                    html.Span(rec.scan, className=scan_class),
                ],
            ),
            html.Div(rec.label or f"{rec.plate} · run {rec.run}", className="lib-label"),
            html.Div(
                className="lib-meta",
                children=[
                    html.Span(rec.plate),
                    html.Span("·", className="dotsep"),
                    html.Span(f"run {rec.run}"),
                    html.Span("·", className="dotsep"),
                    html.Span(f"{len(rec.wells)} well(s)"),
                ],
            ),
        ],
    )


# =============================================================================
# Center column
# =============================================================================


def _center_column() -> html.Div:
    return html.Div(
        className="center-col",
        children=[
            _control_bar(),
            html.Div(
                id="center-stage",
                className="center-stage",
                children=dcc.Graph(
                    id="traces-graph",
                    config={"displaylogo": False, "displayModeBar": "hover"},
                    style={"height": "100%", "minHeight": "320px"},
                ),
            ),
            html.Div(
                className="scrubber",
                children=dcc.Graph(
                    id="scrubber-graph",
                    config={"displaylogo": False, "displayModeBar": False, "staticPlot": False},
                    style={"height": "44px"},
                ),
            ),
            html.Div(id="meta-strip", className="meta-strip"),
        ],
    )


def _control_bar() -> html.Div:
    return html.Div(
        className="controlbar",
        children=[
            html.Div(
                className="ctrl-group",
                children=[
                    html.Span("Bandpass", className="ctrl-label"),
                    html.Span("0.5 – 300 Hz", className="ctrl-readout ctrl-readout-locked"),
                    html.Span("LFP standard", className="ctrl-hint"),
                ],
            ),
            html.Div(className="ctrl-divider"),
            html.Div(
                className="ctrl-group",
                children=[
                    html.Span("Window", className="ctrl-label"),
                    html.Div(
                        className="seg",
                        id="window-presets",
                        children=[
                            html.Button(
                                f"{s}s",
                                id={"type": "window-preset", "seconds": s},
                                className="seg-btn",
                                n_clicks=0,
                            )
                            for s in PRESET_WINDOW_SECONDS
                        ],
                    ),
                    html.Span(id="window-readout", className="ctrl-readout"),
                ],
            ),
            html.Div(className="ctrl-divider"),
            html.Div(
                className="ctrl-group",
                children=[
                    html.Span("Gain", className="ctrl-label"),
                    html.Button("−", id="gain-down", className="step", n_clicks=0),
                    html.Span("1.00×", id="gain-readout", className="ctrl-readout ctrl-readout-num"),
                    html.Button("+", id="gain-up", className="step", n_clicks=0),
                ],
            ),
            html.Div(className="ctrl-spacer"),
            html.Div(
                className="ctrl-group",
                children=html.Span(id="trace-count-readout", className="muted"),
            ),
        ],
    )


# =============================================================================
# Right pane: plate map + channel picker
# =============================================================================


def _right_pane() -> html.Aside:
    return html.Aside(
        className="pane pane-right",
        children=[
            _plate_map(),
            html.Div(className="pane-divider"),
            _channel_picker(),
            html.Div(id="notes-area"),
        ],
    )


def _plate_map() -> html.Div:
    cells: list = [html.Div(className="plate-corner")]
    cells.extend(html.Div(str(c), className="plate-col-hd") for c in PLATE_COLS)
    for row in PLATE_ROWS:
        cells.append(html.Div(row, className="plate-row-hd"))
        for col in PLATE_COLS:
            name = f"{row}{col}"
            well_id = _well_name_to_id(row, col)
            cells.append(
                html.Button(
                    id={"type": "plate-cell", "well_id": well_id},
                    className="plate-cell is-missing",
                    n_clicks=0,
                    children=html.Span(name, className="plate-name"),
                    title=f"Well {name}",
                )
            )
    return html.Div(
        className="plate",
        children=[
            html.Div(
                className="plate-head",
                children=[
                    html.Div("Plate map", className="pane-title"),
                    html.Span("24 wells", className="muted"),
                ],
            ),
            html.Div(id="plate-grid", className="plate-grid", children=cells),
        ],
    )


def _well_name_to_id(row: str, col: int) -> str:
    """Map A1..D6 to MaxWell well000..well023 (row-major, 6 cols)."""
    row_idx = PLATE_ROWS.index(row)
    return f"well{row_idx * 6 + (col - 1):03d}"


def _channel_picker() -> html.Div:
    return html.Div(
        className="chgrid",
        children=[
            html.Div(
                className="chgrid-head",
                children=[
                    html.Div("Electrodes", className="pane-title"),
                    html.Span(id="electrode-count-readout", className="muted"),
                ],
            ),
            html.Div(
                className="chgrid-quick",
                children=[
                    html.Button("All routed", id={"type": "quick-select", "kind": "all"}, className="qbtn", n_clicks=0),
                    html.Button("None", id={"type": "quick-select", "kind": "none"}, className="qbtn", n_clicks=0),
                    html.Button("Top RMS", id={"type": "quick-select", "kind": "top"}, className="qbtn", n_clicks=0),
                    html.Button("Every 8th", id={"type": "quick-select", "kind": "stride"}, className="qbtn", n_clicks=0),
                ],
            ),
            dcc.Graph(
                id="probe-map",
                className="chgrid-graph",
                config={
                    "displaylogo": False,
                    "modeBarButtonsToAdd": ["lasso2d", "select2d"],
                    "modeBarButtonsToRemove": ["autoScale2d"],
                    "scrollZoom": True,
                },
                style={"height": "320px"},
            ),
            html.Div(
                "Box- or lasso-select electrodes on the chip. Scroll to zoom.",
                className="chgrid-hint",
            ),
        ],
    )
