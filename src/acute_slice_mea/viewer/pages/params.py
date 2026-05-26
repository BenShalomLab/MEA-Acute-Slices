"""Analysis Parameters tab — schema-driven parameter form + pipeline runner."""

from __future__ import annotations

from dash import dcc, html


PARAM_SCHEMA = [
    {
        "id": "cache_root",
        "section": "Output",
        "label": "Result cache root",
        "hint": "Pipeline writes per-recording cache bundles under this directory.",
        "type": "path",
        "default": "/data/processed",
    },
    {
        "id": "cache_overwrite",
        "section": "Output",
        "label": "Overwrite existing",
        "hint": "If off, recordings whose cache already exists are skipped.",
        "type": "toggle",
        "default": False,
    },
    {
        "id": "bandpass_low",
        "section": "Filtering",
        "label": "Bandpass low cutoff",
        "hint": "Low frequency cutoff in Hz (LFP default: 0.5 Hz).",
        "type": "number",
        "unit": "Hz",
        "default": 0.5,
    },
    {
        "id": "bandpass_high",
        "section": "Filtering",
        "label": "Bandpass high cutoff",
        "hint": "High frequency cutoff in Hz (LFP default: 300 Hz).",
        "type": "number",
        "unit": "Hz",
        "default": 300,
    },
    {
        "id": "notch_freqs",
        "section": "Filtering",
        "label": "Notch filter(s)",
        "hint": "Comma-separated centers in Hz.",
        "type": "text",
        "unit": "Hz",
        "default": "60,120",
    },
    {
        "id": "notch_q",
        "section": "Filtering",
        "label": "Notch Q-factor",
        "hint": "Higher = narrower notch. 30 is a good default.",
        "type": "number",
        "default": 30,
    },
    {
        "id": "downsample",
        "section": "Resampling",
        "label": "Downsample target",
        "hint": "Resample LFP to this rate (Hz). Set to 0 to keep native.",
        "type": "number",
        "unit": "Hz",
        "default": 1000,
    },
    {
        "id": "reref_enabled",
        "section": "Re-referencing",
        "label": "Enable re-referencing",
        "hint": "Subtract a reference signal from each electrode.",
        "type": "toggle",
        "default": True,
    },
    {
        "id": "reref_method",
        "section": "Re-referencing",
        "label": "Reference method",
        "hint": "CAR = arithmetic mean. CMR = median (more robust to spikes).",
        "type": "radio",
        "options": [
            {"value": "median", "label": "CMR", "sub": "Common Median"},
            {"value": "average", "label": "CAR", "sub": "Common Average"},
        ],
        "default": "median",
    },
    {
        "id": "reref_scope",
        "section": "Re-referencing",
        "label": "Reference scope",
        "hint": "Global = all routed electrodes. Local = annulus per channel.",
        "type": "radio",
        "options": [
            {"value": "global", "label": "Global", "sub": "all electrodes"},
            {"value": "local", "label": "Local", "sub": "annulus per channel"},
        ],
        "default": "global",
    },
    {
        "id": "reref_inner_radius",
        "section": "Re-referencing",
        "label": "Inner radius",
        "hint": "Channels closer than this (µm) are excluded.",
        "type": "number",
        "unit": "µm",
        "default": 30,
    },
    {
        "id": "reref_outer_radius",
        "section": "Re-referencing",
        "label": "Outer radius",
        "hint": "Channels farther than this (µm) are excluded.",
        "type": "number",
        "unit": "µm",
        "default": 200,
    },
    {
        "id": "spectrogram_window",
        "section": "Spectral analysis",
        "label": "Spectrogram window",
        "hint": "STFT window length in seconds.",
        "type": "number",
        "unit": "s",
        "default": 2.0,
    },
    {
        "id": "spectrogram_overlap",
        "section": "Spectral analysis",
        "label": "Window overlap",
        "hint": "Fractional overlap between successive windows.",
        "type": "number",
        "default": 0.5,
    },
    {
        "id": "burst_threshold",
        "section": "Burst detection",
        "label": "Burst threshold",
        "hint": "Z-score above baseline RMS to flag as a burst.",
        "type": "number",
        "unit": "σ",
        "default": 3.0,
    },
    {
        "id": "burst_min_duration",
        "section": "Burst detection",
        "label": "Minimum burst duration",
        "hint": "Ignore burst events shorter than this.",
        "type": "number",
        "unit": "ms",
        "default": 30,
    },
]


def build_params_page() -> html.Div:
    sections = _group_by_section()
    return html.Div(
        className="params",
        children=[
            html.Div(
                className="params-main",
                children=[_build_section(title, params) for title, params in sections],
            ),
            _build_sidebar(),
            # Dedicated stores for toggle/radio params (not backed by dcc.Input)
            dcc.Store(id="param-store-cache-overwrite", data=False),
            dcc.Store(id="param-store-reref-enabled", data=True),
            dcc.Store(id="param-store-reref-method", data="median"),
            dcc.Store(id="param-store-reref-scope", data="global"),
        ],
    )


def _group_by_section() -> list[tuple[str, list[dict]]]:
    groups: dict[str, list[dict]] = {}
    for p in PARAM_SCHEMA:
        sec = p["section"]
        if sec not in groups:
            groups[sec] = []
        groups[sec].append(p)
    return list(groups.items())


def _build_section(title: str, params: list[dict]) -> html.Div:
    return html.Div(
        className="params-section",
        children=[
            html.Div(
                className="params-section-head",
                children=[
                    html.Div(title, className="params-section-title"),
                    html.Span(
                        f"{len(params)} parameter{'s' if len(params) != 1 else ''}",
                        className="params-section-sub",
                    ),
                ],
            ),
            html.Div(
                className="pgrid",
                children=[_build_field(p) for p in params],
            ),
        ],
    )


def _build_field(p: dict) -> html.Div:
    pid = p["id"]
    ptype = p["type"]
    default = p.get("default")
    unit = p.get("unit", "")
    hint = p.get("hint", "")
    wide_class = " pfield-wide" if ptype == "path" else ""

    children = []
    children.append(html.Div(
        className="pfield-label",
        children=[
            p["label"],
            html.Span(f" · {unit}", className="pfield-unit") if unit else None,
        ],
    ))

    if ptype == "number":
        children.append(html.Div(
            className="pfield-row",
            children=[
                dcc.Input(
                    id={"type": "param-input", "param": pid},
                    type="number",
                    value=default,
                    className="input is-mono",
                    style={"flex": "1"},
                ),
                html.Span(unit, className="pfield-unit") if unit else None,
            ],
        ))
    elif ptype == "path":
        children.append(html.Div(
            className="path-input-row",
            children=[
                html.Div("📁", className="path-input-icon", **{"aria-hidden": "true"}),
                dcc.Input(
                    id={"type": "param-input", "param": pid},
                    type="text",
                    value=default or "",
                    className="input is-mono path-input",
                    placeholder="/data/processed",
                    spellCheck=False,
                ),
            ],
        ))
    elif ptype == "text":
        children.append(dcc.Input(
            id={"type": "param-input", "param": pid},
            type="text",
            value=str(default) if default else "",
            className="input is-mono",
        ))
    elif ptype == "toggle":
        store_id = f"param-store-{pid.replace('_', '-')}"
        children.append(html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "10px", "paddingTop": "2px"},
            children=[
                html.Button(
                    id={"type": "toggle-btn", "param": pid},
                    className=f"toggle-btn {'is-on' if default else ''}",
                    n_clicks=0,
                    style={
                        "appearance": "none",
                        "width": "30px", "height": "18px",
                        "borderRadius": "999px",
                        "background": "var(--accent)" if default else "var(--ink-3)",
                        "cursor": "pointer",
                        "position": "relative",
                        "transition": "background .15s",
                        "border": "0",
                    },
                ),
                html.Span(
                    "enabled" if default else "disabled",
                    id={"type": "toggle-label", "param": pid},
                    className="muted mono",
                    style={"fontSize": "11px"},
                ),
            ],
        ))
    elif ptype == "radio":
        options = p.get("options", [])
        children.append(html.Div(
            className="radio-row",
            children=[
                html.Button(
                    id={"type": "param-radio-btn", "param": pid, "value": opt["value"]},
                    className=f"radio-pill {'is-active' if opt['value'] == default else ''}",
                    n_clicks=0,
                    children=[
                        html.Span(opt["label"], className="radio-pill-label"),
                        html.Span(opt.get("sub", ""), className="radio-pill-sub") if opt.get("sub") else None,
                    ],
                )
                for opt in options
            ],
        ))

    children.append(html.Div(hint, className="pfield-hint"))

    return html.Div(
        id={"type": "param-field", "param": pid},
        className=f"pfield{wide_class}",
        children=children,
    )


def _build_sidebar() -> html.Aside:
    return html.Aside(
        className="params-side",
        children=[
            html.Div("Run summary", className="params-side-title"),
            html.Div(
                className="run-panel",
                children=[
                    html.H4("Selection"),
                    html.Div(className="run-summary-row", children=[
                        html.Span("Files queued"),
                        html.Span(id="params-file-count", className="v mono", children="0"),
                    ]),
                    html.Div(className="run-summary-row", children=[
                        html.Span("Total parameters"),
                        html.Span(className="v mono", children=str(len(PARAM_SCHEMA))),
                    ]),
                ],
            ),
            html.Div(
                className="run-panel",
                children=[
                    html.H4("Pipeline preview"),
                    html.Div(id="params-preview", children=[
                        html.Div(className="run-summary-row", children=[
                            html.Span("Bandpass"), html.Span("0.5–300 Hz", className="v mono"),
                        ]),
                        html.Div(className="run-summary-row", children=[
                            html.Span("Notch"), html.Span("60, 120 Hz", className="v mono"),
                        ]),
                        html.Div(className="run-summary-row", children=[
                            html.Span("Downsample"), html.Span("1000 Hz", className="v mono"),
                        ]),
                        html.Div(className="run-summary-row", children=[
                            html.Span("Re-reference"), html.Span("CMR · global", className="v mono"),
                        ]),
                    ]),
                ],
            ),
            html.Div(
                id="params-validation",
                className="run-validation is-ok",
                children=[
                    html.Div("✓ ready to run", className="head"),
                    html.Div(
                        "all parameters valid",
                        className="mono",
                        style={"fontSize": "11.5px", "marginTop": "2px"},
                    ),
                ],
            ),
            html.Div(
                id="run-progress-area",
                style={"display": "none"},
                children=html.Div(
                    className="run-progress",
                    children=[
                        html.Div(
                            className="run-progress-bar",
                            children=html.Span(id="run-progress-fill", style={"width": "0%"}),
                        ),
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between"},
                            children=[
                                html.Span(id="run-progress-label", children="Idle"),
                                html.Span(id="run-progress-pct", children="0%"),
                            ],
                        ),
                        html.Div(
                            id="run-log",
                            style={"maxHeight": "150px", "overflow": "auto", "color": "var(--ink-2)", "marginTop": "6px"},
                        ),
                    ],
                ),
            ),
            html.Button(
                "Run pipeline ▶",
                id="run-pipeline-btn",
                className="btn btn-accent",
                style={"justifyContent": "center", "padding": "10px 14px"},
                n_clicks=0,
            ),
            html.Button(
                "Review LFP results →",
                id="nav-to-review-btn",
                className="btn btn-primary",
                style={"justifyContent": "center", "display": "none"},
                n_clicks=0,
            ),
        ],
    )
