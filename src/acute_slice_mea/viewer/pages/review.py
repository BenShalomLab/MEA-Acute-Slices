"""Review tab layout — the existing three-pane trace viewer.

Extracted from layout.py so it can be mounted as one tab in the multi-tab
shell. All component IDs are unchanged so existing callbacks still work.
"""

from __future__ import annotations

from dash import dcc, html

from acute_slice_mea.library import LibraryIndex, RecordingEntry

PLATE_ROWS = ["A", "B", "C", "D"]
PLATE_COLS = [1, 2, 3, 4, 5, 6]


def build_review_page(library: LibraryIndex) -> html.Div:
    return html.Div(
        className="workspace",
        children=[
            _library_pane(library),
            _center_column(),
            _right_pane(),
        ],
    )


# ── Library pane (left) ─────────────────────────────────────────────────


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


# ── Center column ───────────────────────────────────────────────────────


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
                    html.Span("Gain", className="ctrl-label"),
                    html.Button("−", id="gain-down", className="step", n_clicks=0),
                    html.Span("1.00×", id="gain-readout", className="ctrl-readout ctrl-readout-num"),
                    html.Button("+", id="gain-up", className="step", n_clicks=0),
                ],
            ),
            html.Div(className="ctrl-spacer"),
            html.Div(
                className="ctrl-group",
                children=[
                    html.Span(id="crumbs-area", className="crumbs"),
                ],
            ),
            html.Div(className="ctrl-spacer"),
            html.Div(
                className="ctrl-group",
                children=html.Span(id="trace-count-readout", className="muted"),
            ),
        ],
    )


# ── Right pane: plate map + channel picker ──────────────────────────────


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
                "Box- or lasso-select on the chip · click to toggle · scroll to zoom.",
                className="chgrid-hint",
            ),
            dcc.Input(
                id="electrode-id-entry",
                type="text",
                debounce=True,
                placeholder="IDs / ranges, e.g. 500-520, 550, 600-610  (Enter to apply)",
                className="chgrid-id-entry",
                style={"width": "100%", "marginTop": "6px"},
            ),
        ],
    )
