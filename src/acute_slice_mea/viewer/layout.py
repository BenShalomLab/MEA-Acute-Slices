"""Dash layout for the Acute Slice LFP Dashboard.

Multi-tab shell: Explore → Parameters → Review → Export. Tab switching is
handled by a clientside callback toggling ``display`` on the four page Divs
so all callbacks remain mounted at all times.
"""

from __future__ import annotations

from dash import dcc, html

from acute_slice_mea.library import LibraryIndex
from acute_slice_mea.viewer.pages.explore import FILE_SCHEMA, build_explore_page
from acute_slice_mea.viewer.pages.export import build_export_page
from acute_slice_mea.viewer.pages.params import build_params_page
from acute_slice_mea.viewer.pages.review import PLATE_COLS, PLATE_ROWS, build_review_page

# Re-export for callbacks.py which imports these from layout
__all__ = ["PLATE_ROWS", "PLATE_COLS", "build_layout"]

TABS = [
    {"id": "explore", "num": "01", "label": "Data Explore"},
    {"id": "params", "num": "02", "label": "Analysis Parameters"},
    {"id": "review", "num": "03", "label": "LFP Review"},
    {"id": "export", "num": "04", "label": "Export"},
]


def build_layout(library: LibraryIndex, *, data_root: str | None = None) -> html.Div:
    default_tab = "review" if data_root is None else "explore"
    return html.Div(
        className="app",
        children=[
            _top_bar(library),
            _tab_nav(default_tab),
            # Pages — only one visible at a time
            html.Div(
                id="page-explore",
                className="page",
                style={"display": "grid" if default_tab == "explore" else "none"},
                children=build_explore_page(data_root),
            ),
            html.Div(
                id="page-params",
                className="page",
                style={"display": "none"},
                children=build_params_page(),
            ),
            html.Div(
                id="page-review",
                className="page",
                style={"display": "grid" if default_tab == "review" else "none"},
                children=build_review_page(library),
            ),
            html.Div(
                id="page-export",
                className="page",
                style={"display": "none"},
                children=build_export_page(),
            ),
            # Stores — existing
            dcc.Store(id="selected-recording-id"),
            dcc.Store(id="selected-well-id"),
            dcc.Store(id="selected-channels", data=[]),
            dcc.Store(id="gain", data=1.0),
            dcc.Store(id="well-version", data=0),
            # Stores — new (tab navigation + explore + params)
            dcc.Store(id="active-tab", data=default_tab),
            dcc.Store(id="scan-results", data=[]),
            dcc.Store(id="selected-file-ids", data=[]),
            dcc.Store(id="file-schema", data=FILE_SCHEMA),
            dcc.Store(id="run-state", data={"active": False, "done": False, "pct": 0, "log": []}),
            # Interval for pipeline progress polling
            dcc.Interval(id="run-poll-interval", interval=800, disabled=True),
        ],
    )


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
                            html.Div("Acute Slice · LFP Dashboard", className="brand-title"),
                            html.Div(
                                "HD-MEA · explore → analyse → review → export",
                                className="brand-sub",
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(),
            html.Div(
                className="topbar-right",
                children=[
                    html.Span(
                        className="status-chip",
                        children=[
                            html.I(id="scan-dot", className="dot"),
                            html.Span(id="scan-status-text", children="no scan yet"),
                        ],
                    ),
                    html.Span(
                        className="status-chip",
                        children=[
                            html.I(id="sel-dot", className="dot"),
                            html.Span(id="sel-status-text", children="0 selected"),
                        ],
                    ),
                    html.Span(
                        className="status-chip",
                        children=[
                            html.I(id="pipeline-dot", className="dot"),
                            html.Span(id="pipeline-status-text", children="pipeline idle"),
                        ],
                    ),
                    html.Span(
                        id="status-chip",
                        className="status-chip",
                        children=[
                            html.I(className="dot dot-live"),
                            html.Span(id="status-text", children="0 traces on screen"),
                        ],
                    ),
                ],
            ),
        ],
    )


def _tab_nav(default_tab: str) -> html.Div:
    children = []
    for t in TABS:
        active = " is-active" if t["id"] == default_tab else ""
        extras = []
        if t["id"] == "export":
            extras.append(html.Span(
                "on hold",
                style={
                    "marginLeft": "4px",
                    "fontFamily": "var(--font-mono)", "fontSize": "10px",
                    "background": "var(--paper-2)", "color": "var(--ink-3)",
                    "padding": "1px 6px", "borderRadius": "999px", "border": "1px solid var(--rule)",
                },
            ))
        children.append(
            html.Button(
                id={"type": "tab-btn", "tab": t["id"]},
                className=f"tabnav-item{active}",
                n_clicks=0,
                children=[
                    html.Span(t["num"], className="tabnav-idx"),
                    t["label"],
                    *extras,
                ],
            )
        )
    children.append(html.Div(className="tabnav-spacer"))
    children.append(html.Div("v0.5.0 · multi-tab", className="tabnav-meta"))
    return html.Div(className="tabnav", children=children)
