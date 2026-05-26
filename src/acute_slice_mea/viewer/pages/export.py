"""Export tab — placeholder (on hold per spec)."""

from __future__ import annotations

from dash import html


def build_export_page() -> html.Div:
    return html.Div(
        className="export",
        children=[
            html.Div(
                className="export-card",
                children=[
                    html.Div(
                        className="export-card-glyph",
                        children=html.Div("↓", style={"fontSize": "22px"}),
                    ),
                    html.H3("Data Export"),
                    html.P(
                        "Slice and export a specific time window and electrode subset for "
                        "downstream analysis (MATLAB, Python, NWB). This page is on hold while "
                        "the upstream pipeline stabilises — we'll wire it up once the LFP "
                        "review workflow is finalised."
                    ),
                    html.Div(
                        className="export-card-features",
                        children=[
                            html.Div("Planned", className="export-card-features-h"),
                            html.Ul([
                                html.Li("Time-range selector with snap-to-burst and scrubber preview."),
                                html.Li("Electrode subset import from the Review tab's selection."),
                                html.Li("Formats: NWB 2.x, MAT (HDF5), NumPy npz, CSV, raw binary."),
                                html.Li("Optional re-export of derived features: bursts, spectra, FOOOF."),
                                html.Li("Batch mode: apply the same window to every selected recording."),
                            ]),
                        ],
                    ),
                    html.Div(
                        id="export-status",
                        style={"marginTop": "16px", "fontFamily": "var(--font-mono)", "fontSize": "11px", "color": "var(--ink-3)"},
                        children="0 files queued · 0 electrodes pending",
                    ),
                ],
            ),
        ],
    )
