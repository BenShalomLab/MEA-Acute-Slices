"""Dash layout for the Acute Slice trace viewer.

Class names mirror the design prototype at
``/tmp/design_pkg/acute-slice/project/styles.css`` so the CSS port in
``assets/styles.css`` applies directly. The pane structure follows the
prototype: library (left), center column (controls + traces + meta), right
pane (plate map + channel picker).

When ``jobs_enabled=True`` (i.e., ``--data-root`` was passed), each library
row gains a status pill + inline action button, the center stage gains an
empty-state CTA overlay, and the top bar exposes a jobs drawer, params
sheet, and "Compute" controls (workers slider).
"""

from __future__ import annotations

from dataclasses import dataclass

from dash import dcc, html

from acute_slice_mea.library import LibraryIndex, RecordingEntry, WellEntry

PRESET_WINDOW_SECONDS = [1, 2, 5, 10, 30]
DEFAULT_WINDOW_SECONDS = 5

# Seed values for the editable power-band rows in the params sheet. The
# names match ``acute_slice_mea.spectral.DEFAULT_LFP_BANDS`` so changes here
# round-trip cleanly through the pipeline.
DEFAULT_BAND_ROWS: list[tuple[str, float, float]] = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 12.0),
    ("beta", 12.0, 30.0),
    ("low_gamma", 30.0, 80.0),
    ("high_gamma", 80.0, 150.0),
]


def _build_band_rows() -> list:
    """Render the editable band rows. Each row carries a pattern-matched
    id ``{"type": "param-band", "name": <band>, "edge": "low"|"high"}`` so
    a single ``State({"type": "param-band", ...}, "value")`` collects all
    six bands in one shot in the submit callback.
    """
    rows = []
    for name, low, high in DEFAULT_BAND_ROWS:
        rows.append(
            html.Div(
                className="ax-sheet-band-row",
                children=[
                    html.Span(name, className="ax-sheet-band-name"),
                    dcc.Input(
                        id={"type": "param-band", "name": name, "edge": "low"},
                        type="number",
                        value=low,
                        min=0.0,
                        step=0.5,
                        className="ax-sheet-input ax-sheet-input-narrow",
                    ),
                    dcc.Input(
                        id={"type": "param-band", "name": name, "edge": "high"},
                        type="number",
                        value=high,
                        min=0.0,
                        step=0.5,
                        className="ax-sheet-input ax-sheet-input-narrow",
                    ),
                ],
            )
        )
    return rows

# 4 x 6 MaxWell 24-well plate. We render the *fixed* layout; missing wells
# are styled differently based on whether they appear in the recording.
PLATE_ROWS = ["A", "B", "C", "D"]
PLATE_COLS = [1, 2, 3, 4, 5, 6]

JOBS_POLL_INTERVAL_MS = 2000


@dataclass
class TopBarState:
    """Returned by the breadcrumb callback for ``crumbs-area``."""

    recording: RecordingEntry | None
    well_id: str | None


def build_layout(library: LibraryIndex, *, jobs_enabled: bool = False) -> html.Div:
    children: list = [
        _top_bar(library, jobs_enabled=jobs_enabled),
        html.Div(
            className="workspace",
            children=[
                _library_pane(library, jobs_enabled=jobs_enabled),
                _center_column(jobs_enabled=jobs_enabled),
                _right_pane(),
            ],
        ),
        # Hidden stores
        dcc.Store(id="selected-recording-id"),
        dcc.Store(id="selected-well-id"),
        dcc.Store(id="selected-channels", data=[]),
        dcc.Store(id="time-window", data=[0.0, float(DEFAULT_WINDOW_SECONDS)]),
        dcc.Store(id="gain", data=1.0),
        dcc.Store(id="well-version", data=0),
        # jobs-poll is included even in view-only mode so the library-filter
        # callback (which lists this as an Input) is always satisfied. The
        # interval is harmless when there is no jobs backend to query.
        dcc.Interval(id="jobs-poll", interval=JOBS_POLL_INTERVAL_MS),
    ]
    if jobs_enabled:
        children.extend(
            [
                dcc.Store(id="jobs-drawer-open", data=False),
                dcc.Store(id="params-sheet-state", data=None),
                _jobs_drawer(),
                _params_sheet(),
            ]
        )
    return html.Div(className="app", children=children)


# =============================================================================
# Top bar
# =============================================================================


def _top_bar(library: LibraryIndex, *, jobs_enabled: bool) -> html.Div:
    right_children: list = [
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
    ]
    if jobs_enabled:
        right_children.extend(
            [
                html.Button(
                    id="jobs-drawer-toggle",
                    className="topbar-btn",
                    n_clicks=0,
                    children=[
                        html.Span("Jobs", className="topbar-btn-label"),
                        html.Span("0", id="jobs-badge", className="topbar-badge"),
                    ],
                ),
                html.Div(className="topbar-compute", children=_compute_controls()),
            ]
        )
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
            html.Div(className="topbar-right", children=right_children),
        ],
    )


def _compute_controls() -> list:
    """Workers slider + cache total readout. Visible only when jobs_enabled."""
    return [
        html.Span("Workers", className="compute-label"),
        dcc.Slider(
            id="workers-slider",
            min=1,
            max=1,
            step=1,
            value=1,
            marks={1: "1"},
            tooltip={"placement": "bottom"},
            className="compute-slider",
        ),
        html.Span(id="cache-total-readout", className="muted compute-readout", children="—"),
        html.Button(
            "Reset jobs",
            id="debug-reset-btn",
            className="topbar-btn topbar-btn-ghost",
            n_clicks=0,
            title="Cancel all jobs and wipe the jobs directory (debug).",
        ),
    ]


# =============================================================================
# Library pane (left)
# =============================================================================


def _library_pane(library: LibraryIndex, *, jobs_enabled: bool) -> html.Aside:
    grouped = library.grouped_by_date()
    sample_options = _unique_options([r.sample for r in library.recordings])
    scan_options = _unique_options([r.scan for r in library.recordings])
    plate_options = _unique_options([r.plate for r in library.recordings])
    # Per-well groupnames (post-metadata extraction).
    group_options = _unique_options(
        [w.group for r in library.recordings for w in r.wells if w.group]
    )
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
            _library_filter_bar(
                sample_options=sample_options,
                scan_options=scan_options,
                plate_options=plate_options,
                group_options=group_options,
                jobs_enabled=jobs_enabled,
            ),
            html.Div(
                id="library-list",
                className="lib-scroll",
                children=[_library_group(date, items, jobs_enabled=jobs_enabled) for date, items in grouped]
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


def _unique_options(values) -> list[dict]:
    """Return a sorted list of dropdown options from observed values."""
    uniq = sorted({str(v) for v in values if v})
    return [{"label": v, "value": v} for v in uniq]


def _library_filter_bar(
    *,
    sample_options: list[dict],
    scan_options: list[dict],
    plate_options: list[dict],
    group_options: list[dict],
    jobs_enabled: bool,
) -> html.Div:
    """Render the multi-axis filter row above the library list.

    Each Dropdown is multi-select; combining them is AND across axes, OR
    within an axis. Filters intersect (not subtract) — leaving a dropdown
    empty means "any value passes" for that axis.
    """
    cache_state_options = [
        {"label": "Cached", "value": "cached"},
        {"label": "Queued", "value": "queued"},
        {"label": "Running", "value": "running"},
        {"label": "Failed", "value": "failed"},
        {"label": "Not built", "value": "idle"},
    ]
    children = [
        dcc.Dropdown(
            id="lib-filter-sample",
            options=sample_options,
            multi=True,
            placeholder="Sample",
            className="lib-filter-dd",
        ),
        dcc.Dropdown(
            id="lib-filter-scan",
            options=scan_options,
            multi=True,
            placeholder="Scan type",
            className="lib-filter-dd",
        ),
        dcc.Dropdown(
            id="lib-filter-plate",
            options=plate_options,
            multi=True,
            placeholder="Plate",
            className="lib-filter-dd",
        ),
    ]
    if group_options:
        children.append(
            dcc.Dropdown(
                id="lib-filter-group",
                options=group_options,
                multi=True,
                placeholder="Group / condition",
                className="lib-filter-dd",
            )
        )
    else:
        # Still register the component so the filter callback's Input list
        # is stable across libraries that lack groupname metadata.
        children.append(
            dcc.Dropdown(
                id="lib-filter-group",
                options=[],
                multi=True,
                placeholder="Group / condition (none found)",
                className="lib-filter-dd",
                disabled=True,
            )
        )
    if jobs_enabled:
        children.extend(
            [
                dcc.Dropdown(
                    id="lib-filter-cache",
                    options=cache_state_options,
                    multi=True,
                    placeholder="Cache state",
                    className="lib-filter-dd",
                ),
                html.Button(
                    "Run analysis on filtered",
                    id="lib-batch-run-btn",
                    n_clicks=0,
                    className="btn small primary lib-batch-run",
                ),
            ]
        )
    else:
        # Placeholder cache-filter so the callback always has the input.
        children.append(
            dcc.Dropdown(
                id="lib-filter-cache",
                options=cache_state_options,
                multi=True,
                placeholder="Cache state",
                className="lib-filter-dd",
                disabled=True,
            )
        )
    return html.Div(className="lib-filter-bar", children=children)


def _library_group(date: str, items: list[RecordingEntry], *, jobs_enabled: bool) -> html.Div:
    return html.Div(
        className="lib-group",
        children=[
            html.Div(
                className="lib-group-head",
                children=[html.Span(date), html.Span(str(len(items)), className="muted")],
            ),
            html.Ul(
                className="lib-list",
                children=[html.Li(_library_item(rec, jobs_enabled=jobs_enabled)) for rec in items],
            ),
        ],
    )


def _library_item(rec: RecordingEntry, *, jobs_enabled: bool) -> html.Div:
    scan_class = "lib-scan lib-scan-network" if rec.scan.lower() == "network" else "lib-scan"
    select_btn = html.Button(
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
    if not jobs_enabled:
        return select_btn

    # Phase-1 scope: the row pill + Run button target the recording's first
    # well only. Multi-well per-recording batch is deferred — see the plan.
    target_well: WellEntry | None = rec.wells[0] if rec.wells else None
    well_id = target_well.well_id if target_well else "—"
    # Pattern-matched id carries the filter axes; the library-filter callback
    # reads these from State to decide each row's visibility.
    row_id = {
        "type": "lib-row",
        "recording_id": rec.recording_id,
        "well_id": well_id,
        "sample": rec.sample,
        "scan": rec.scan,
        "plate": rec.plate,
        "group": (target_well.group if target_well else "") or "",
    }
    return html.Div(
        id=row_id,
        className="lib-row-wrap",
        children=[
            select_btn,
            html.Div(
                className="lib-cache",
                children=[
                    html.Span(
                        id={
                            "type": "lib-status-pill",
                            "recording_id": rec.recording_id,
                            "well_id": well_id,
                        },
                        className="pill idle",
                        children=[html.I(className="dot"), html.Span("idle")],
                    ),
                    html.Div(
                        id={
                            "type": "lib-progress",
                            "recording_id": rec.recording_id,
                            "well_id": well_id,
                        },
                        className="pbar",
                        children=html.Span(style={"width": "0%"}),
                    ),
                    html.Button(
                        "Run",
                        id={
                            "type": "lib-action",
                            "recording_id": rec.recording_id,
                            "well_id": well_id,
                        },
                        n_clicks=0,
                        className="btn small primary",
                    ),
                ],
            ),
        ],
    )


# =============================================================================
# Center column
# =============================================================================


def _center_column(*, jobs_enabled: bool) -> html.Div:
    stage_children: list = [
        dcc.Graph(
            id="traces-graph",
            config={"displaylogo": False, "displayModeBar": "hover"},
            style={"height": "100%", "minHeight": "320px"},
        ),
    ]
    if jobs_enabled:
        stage_children.append(
            html.Div(id="center-stage-cta", className="ax-cta hidden", children=[]),
        )
    return html.Div(
        className="center-col",
        children=[
            _control_bar(),
            html.Div(
                id="center-stage",
                className="center-stage",
                children=stage_children,
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


# =============================================================================
# Jobs drawer (right slide-in)
# =============================================================================


def _jobs_drawer() -> html.Div:
    return html.Div(
        id="jobs-drawer",
        className="ax-drawer hidden",
        children=[
            html.Div(
                className="ax-drawer-head",
                children=[
                    html.Div("Jobs", className="ax-drawer-title"),
                    html.Button("×", id="jobs-drawer-close", className="ax-drawer-close", n_clicks=0),
                ],
            ),
            html.Div(id="jobs-drawer-body", className="ax-drawer-body", children=[]),
        ],
    )


# =============================================================================
# Params sheet (modal)
# =============================================================================


def _params_sheet() -> html.Div:
    return html.Div(
        id="params-sheet",
        className="ax-sheet hidden",
        children=[
            html.Div(
                className="ax-sheet-card",
                children=[
                    html.Div(
                        className="ax-sheet-head",
                        children=[
                            html.Div("Run LFP analysis", className="ax-sheet-title"),
                            html.Div(id="params-sheet-subtitle", className="ax-sheet-sub"),
                        ],
                    ),
                    html.Div(
                        className="ax-sheet-body",
                        children=[
                            html.Div(
                                className="ax-sheet-row",
                                children=[
                                    html.Label("Bandpass low (Hz)", className="ax-sheet-label"),
                                    dcc.Input(
                                        id="params-band-low",
                                        type="number",
                                        value=0.5,
                                        min=0.01,
                                        max=50,
                                        step=0.1,
                                        className="ax-sheet-input",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="ax-sheet-row",
                                children=[
                                    html.Label("Bandpass high (Hz)", className="ax-sheet-label"),
                                    dcc.Input(
                                        id="params-band-high",
                                        type="number",
                                        value=300,
                                        min=10,
                                        max=5000,
                                        step=1,
                                        className="ax-sheet-input",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="ax-sheet-row",
                                children=[
                                    html.Label("Reference", className="ax-sheet-label"),
                                    dcc.RadioItems(
                                        id="params-reference",
                                        options=[
                                            # CAR is kept as a label for backward
                                            # compatibility with prior caches.
                                            # The CMR (median) operator is the
                                            # one that's actually been in use.
                                            {"label": "CMR (median)", "value": "CMR"},
                                            {"label": "CAR (mean)", "value": "MEAN"},
                                            {"label": "None", "value": "none"},
                                        ],
                                        value="CMR",
                                        className="ax-sheet-radio",
                                        labelStyle={"marginRight": "12px"},
                                    ),
                                ],
                            ),
                            html.Div(
                                className="ax-sheet-row",
                                children=[
                                    html.Label("LFP target fs (Hz)", className="ax-sheet-label"),
                                    dcc.Input(
                                        id="params-lfp-fs",
                                        type="number",
                                        value=1000,
                                        min=100,
                                        max=10000,
                                        step=100,
                                        className="ax-sheet-input",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="ax-sheet-row",
                                children=[
                                    html.Label("Notch (Hz, comma-sep.)", className="ax-sheet-label"),
                                    dcc.Input(
                                        id="params-notch",
                                        type="text",
                                        value="",
                                        placeholder="e.g. 50,100  or  60,120",
                                        className="ax-sheet-input",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="ax-sheet-row",
                                children=[
                                    html.Label("Notch Q", className="ax-sheet-label"),
                                    dcc.Input(
                                        id="params-notch-q",
                                        type="number",
                                        value=30,
                                        min=1,
                                        max=200,
                                        step=1,
                                        className="ax-sheet-input",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="ax-sheet-row",
                                children=[
                                    html.Label("Band-power window / step (s)", className="ax-sheet-label"),
                                    html.Div(
                                        className="ax-sheet-pair",
                                        children=[
                                            dcc.Input(
                                                id="params-window-sec",
                                                type="number",
                                                value=10,
                                                min=0.5,
                                                max=120,
                                                step=0.5,
                                                className="ax-sheet-input",
                                            ),
                                            dcc.Input(
                                                id="params-step-sec",
                                                type="number",
                                                value=5,
                                                min=0.1,
                                                max=60,
                                                step=0.1,
                                                className="ax-sheet-input",
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                className="ax-sheet-row ax-sheet-bands",
                                children=[
                                    html.Label("Power bands (Hz)", className="ax-sheet-label"),
                                    html.Div(
                                        id="params-bands-rows",
                                        className="ax-sheet-bands-rows",
                                        children=_build_band_rows(),
                                    ),
                                ],
                            ),
                            html.Div(
                                id="params-overwrite-row",
                                className="ax-sheet-row hidden",
                                children=dcc.Checklist(
                                    id="params-overwrite",
                                    options=[{"label": "  Overwrite previous result", "value": "yes"}],
                                    value=[],
                                    className="ax-sheet-overwrite",
                                ),
                            ),
                            html.Div(id="params-sheet-error", className="ax-sheet-error hidden"),
                        ],
                    ),
                    html.Div(
                        className="ax-sheet-foot",
                        children=[
                            html.Button("Cancel", id="params-cancel", className="btn", n_clicks=0),
                            html.Button("Submit", id="params-submit", className="btn primary", n_clicks=0),
                        ],
                    ),
                ],
            ),
        ],
    )
