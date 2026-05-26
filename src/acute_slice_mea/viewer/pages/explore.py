"""Data Explore tab — directory scanner, expression filter, results table."""

from __future__ import annotations

from dash import dcc, html


FILE_SCHEMA = [
    {"field": "sample", "label": "Sample", "type": "text"},
    {"field": "date", "label": "Recording date", "type": "date"},
    {"field": "plate", "label": "Plate", "type": "text"},
    {"field": "scan", "label": "Scan type", "type": "enum", "values": ["Network", "Activity", "AxonTracking", "Spike"]},
    {"field": "run", "label": "Run", "type": "text"},
    {"field": "well_id", "label": "Well", "type": "text"},
    {"field": "duration_s", "label": "Duration", "type": "number", "unit": "s"},
    {"field": "sample_rate_hz", "label": "Sample rate", "type": "number", "unit": "Hz"},
    {"field": "num_electrodes", "label": "Electrodes", "type": "int"},
    {"field": "cached", "label": "Cached", "type": "bool"},
    {"field": "operator", "label": "Operator", "type": "text"},
]

TABLE_COLUMNS = [
    {"key": "sample", "label": "Sample"},
    {"key": "date", "label": "Date"},
    {"key": "plate", "label": "Plate"},
    {"key": "scan", "label": "Scan"},
    {"key": "run", "label": "Run"},
    {"key": "well_id", "label": "Well"},
    {"key": "duration_s", "label": "Dur (s)"},
    {"key": "sample_rate_hz", "label": "Fs (Hz)"},
    {"key": "num_electrodes", "label": "Elecs"},
    {"key": "cached", "label": "Status"},
]


def build_explore_page(data_root: str | None = None) -> html.Div:
    return html.Div(
        className="explorer",
        children=[
            _root_bar(data_root),
            _filter_editor(),
            _results_section(),
        ],
    )


def _root_bar(data_root: str | None) -> html.Div:
    return html.Div(
        className="rootbar",
        children=[
            html.Div(
                className="rootbar-icon",
                children=html.Div(
                    "📁", style={"fontSize": "16px"},
                ),
                **{"aria-hidden": "true"},
            ),
            html.Div(
                className="rootbar-input-wrap",
                children=[
                    html.Div("Root directory", className="rootbar-label"),
                    dcc.Input(
                        id="data-root-input",
                        type="text",
                        value=data_root or "",
                        placeholder="/data/Acute_Slice_MEA",
                        className="rootbar-input",
                        debounce=True,
                    ),
                ],
            ),
            html.Div(
                className="rootbar-actions",
                children=[
                    html.Div(
                        id="scan-progress-area",
                        className="scan-progress",
                        children=html.Span("Not scanned"),
                    ),
                    html.Button(
                        "Scan directory",
                        id="scan-btn",
                        className="btn btn-accent",
                        n_clicks=0,
                    ),
                ],
            ),
        ],
    )


def _filter_editor() -> html.Div:
    return html.Div(
        className="filter-bar",
        children=[
            html.Div(
                className="filter-bar-head",
                children=[
                    html.Div(
                        className="filter-bar-title",
                        children=[
                            html.Span("Filter expression", style={"fontWeight": 600}),
                            html.Span(
                                " · SQL-like · AND / OR / NOT · parentheses to group",
                                className="muted mono",
                                style={"fontSize": "11px", "marginLeft": "8px", "fontWeight": 400},
                            ),
                        ],
                    ),
                    html.Div(
                        className="filter-bar-actions",
                        children=[
                            html.Button(
                                "From selected",
                                id="expr-from-selection-btn",
                                className="btn btn-sm",
                                n_clicks=0,
                            ),
                            html.Button("Clear", id="expr-clear-btn", className="btn btn-sm btn-ghost", n_clicks=0),
                        ],
                    ),
                ],
            ),
            # Chip rail
            html.Div(
                className="chip-rail",
                children=[
                    html.Span(
                        className="chip-rail-group",
                        children=[
                            html.Span("connectives", className="chip-rail-lbl"),
                            html.Button("AND", id={"type": "kchip", "val": " AND "}, className="kchip kchip-and", n_clicks=0),
                            html.Button("OR", id={"type": "kchip", "val": " OR "}, className="kchip kchip-or", n_clicks=0),
                            html.Button("NOT", id={"type": "kchip", "val": "NOT "}, className="kchip kchip-not", n_clicks=0),
                            html.Button("( … )", id={"type": "kchip", "val": "(  )"}, className="kchip kchip-paren", n_clicks=0),
                        ],
                    ),
                    html.Span(
                        className="chip-rail-group",
                        children=[
                            html.Span("operators", className="chip-rail-lbl"),
                            *[
                                html.Button(op, id={"type": "kchip", "val": f" {op} "}, className="kchip", n_clicks=0)
                                for op in ["==", "!=", "<", "<=", ">", ">=", "contains", "in [", "between"]
                            ],
                        ],
                    ),
                ],
            ),
            # Editor + side panel
            html.Div(
                className="expr-editor-wrap",
                children=[
                    html.Div(
                        className="expr-editor-col",
                        children=[
                            html.Div(
                                className="expr-editor-frame",
                                children=[
                                    dcc.Textarea(
                                        id="expr-textarea",
                                        className="expr-editor",
                                        placeholder='scan == "Network" AND well_id == "well000"\nor click an example below…',
                                        spellCheck=False,
                                        style={"minHeight": "96px", "resize": "vertical"},
                                    ),
                                    html.Div(
                                        id="autocomplete-panel",
                                        className="autocomplete is-empty",
                                        children=html.Span(
                                            "Type a filter expression above.",
                                            className="muted mono",
                                            style={"fontSize": "10.5px"},
                                        ),
                                    ),
                                ],
                            ),
                            html.Div(id="expr-status", className="expr-status"),
                        ],
                    ),
                    html.Div(
                        className="expr-side",
                        children=[
                            html.Div(
                                className="expr-side-block",
                                children=[
                                    html.Div("Parses as", className="expr-side-h"),
                                    html.Div(
                                        id="expr-preview",
                                        className="fpreview",
                                        children=html.Span("(no filter — showing all rows)", style={"color": "var(--ink-3)"}),
                                    ),
                                ],
                            ),
                            html.Div(
                                className="expr-side-block",
                                children=[
                                    html.Div("Examples — click to load", className="expr-side-h"),
                                    html.Div(
                                        className="example-list",
                                        children=[
                                            _example_row("Network scans only", 'scan == "Network"'),
                                            _example_row("Long network, cached", 'scan == "Network" AND duration_s >= 300 AND cached is true'),
                                            _example_row("Specific well tuples", '(plate == "PlateA" AND well_id == "well000") OR (plate == "PlateC" AND well_id == "well006")'),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def _example_row(label: str, expr: str) -> html.Button:
    display = expr if len(expr) <= 60 else expr[:58] + "…"
    return html.Button(
        id={"type": "example-row", "expr": expr},
        className="example-row",
        n_clicks=0,
        children=[
            html.Div(label, className="example-label"),
            html.Code(display, className="example-code"),
        ],
    )


def _results_section() -> html.Div:
    return html.Div(
        className="results",
        children=[
            html.Div(
                className="results-summary",
                children=[
                    html.Span(id="results-match-count", className="chip-count", children="0"),
                    html.Span(id="results-total-text", children="of 0 recordings match"),
                    html.Span("·", className="muted"),
                    html.Span(id="results-sel-text", children="0 selected"),
                    html.Div(className="results-summary-spacer"),
                ],
            ),
            html.Div(
                id="results-table-wrap",
                className="tablewrap",
                children=html.Div(
                    className="empty",
                    children=[
                        html.Div("No files scanned", className="empty-h"),
                        html.Div("Set a root directory and click Scan to discover .h5 recordings.", className="empty-s"),
                    ],
                ),
            ),
            html.Div(
                className="results-footer",
                children=[
                    html.Div(
                        id="results-footer-text",
                        className="selection-text",
                        children="Tick rows to send to the analysis pipeline →",
                    ),
                    html.Div(
                        className="actions",
                        children=[
                            html.Button(
                                "Continue to parameters →",
                                id="nav-to-params-btn",
                                className="btn btn-sm btn-primary",
                                n_clicks=0,
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
