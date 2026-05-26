"""Dash app factory for the Acute Slice LFP Dashboard."""

from __future__ import annotations

from pathlib import Path

from dash import ALL, Input, Output, State, Dash, ctx, html, no_update
from dash.exceptions import PreventUpdate

from acute_slice_mea.library import LibraryIndex
from acute_slice_mea.viewer.callbacks import register_all
from acute_slice_mea.viewer.callbacks_explore import register_explore_callbacks
from acute_slice_mea.viewer.callbacks_params import register_params_callbacks
from acute_slice_mea.viewer.layout import build_layout


def create_app(
    cache_root: str | Path,
    *,
    data_root: str | Path | None = None,
    title: str = "Acute Slice · LFP Dashboard",
) -> Dash:
    """Build the Dash app from a directory of cached recordings."""
    library = LibraryIndex.from_cache_root(cache_root)

    app = Dash(
        __name__,
        title=title,
        update_title=None,
        suppress_callback_exceptions=True,
    )

    data_root_str = str(data_root) if data_root else None
    app.layout = build_layout(library, data_root=data_root_str)

    _register_tab_callbacks(app)
    _register_toggle_callbacks(app)
    register_all(app, library)
    register_explore_callbacks(app)
    register_params_callbacks(app, library)

    app.library = library  # type: ignore[attr-defined]
    return app


def _register_tab_callbacks(app: Dash) -> None:
    """Tab switching: button clicks → store update → page visibility toggle."""

    @app.callback(
        Output("active-tab", "data", allow_duplicate=True),
        Output({"type": "tab-btn", "tab": ALL}, "className"),
        Input({"type": "tab-btn", "tab": ALL}, "n_clicks"),
        State("active-tab", "data"),
        State({"type": "tab-btn", "tab": ALL}, "id"),
        prevent_initial_call=True,
    )
    def on_tab_click(_clicks, current_tab, tab_ids):
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            raise PreventUpdate
        new_tab = triggered["tab"]
        classes = [
            f"tabnav-item{' is-active' if tid['tab'] == new_tab else ''}"
            for tid in tab_ids
        ]
        return new_tab, classes

    app.clientside_callback(
        """
        function(activeTab) {
            var pages = ['explore', 'params', 'review', 'export'];
            return pages.map(function(p) {
                return {display: p === activeTab ? 'grid' : 'none'};
            });
        }
        """,
        Output("page-explore", "style"),
        Output("page-params", "style"),
        Output("page-review", "style"),
        Output("page-export", "style"),
        Input("active-tab", "data"),
    )


def _register_toggle_callbacks(app: Dash) -> None:
    """Toggle buttons (for params that use toggle type)."""

    @app.callback(
        Output("param-store-cache-overwrite", "data"),
        Output({"type": "toggle-btn", "param": "cache_overwrite"}, "style"),
        Output({"type": "toggle-label", "param": "cache_overwrite"}, "children"),
        Input({"type": "toggle-btn", "param": "cache_overwrite"}, "n_clicks"),
        State("param-store-cache-overwrite", "data"),
        prevent_initial_call=True,
    )
    def toggle_cache_overwrite(n, current):
        new_val = not current
        style = _toggle_style(new_val)
        return new_val, style, "enabled" if new_val else "disabled"

    @app.callback(
        Output("param-store-reref-enabled", "data"),
        Output({"type": "toggle-btn", "param": "reref_enabled"}, "style"),
        Output({"type": "toggle-label", "param": "reref_enabled"}, "children"),
        Input({"type": "toggle-btn", "param": "reref_enabled"}, "n_clicks"),
        State("param-store-reref-enabled", "data"),
        prevent_initial_call=True,
    )
    def toggle_reref_enabled(n, current):
        new_val = not current
        style = _toggle_style(new_val)
        return new_val, style, "enabled" if new_val else "disabled"


def _toggle_style(on: bool) -> dict:
    return {
        "appearance": "none",
        "width": "30px", "height": "18px",
        "borderRadius": "999px",
        "background": "var(--accent)" if on else "var(--ink-3)",
        "cursor": "pointer",
        "position": "relative",
        "transition": "background .15s",
        "border": "0",
    }
