"""Dash app factory for the Acute Slice trace viewer."""

from __future__ import annotations

from pathlib import Path

from dash import Dash

from acute_slice_mea.library import LibraryIndex
from acute_slice_mea.viewer.callbacks import register_all
from acute_slice_mea.viewer.layout import build_layout


def create_app(cache_root: str | Path, *, title: str = "Acute Slice · Trace Viewer") -> Dash:
    """Build the Dash app from a directory of cached recordings."""
    library = LibraryIndex.from_cache_root(cache_root)

    app = Dash(
        __name__,
        title=title,
        update_title=None,
        suppress_callback_exceptions=True,
    )
    app.layout = build_layout(library)
    register_all(app, library)
    # Expose the library on the app for easy debugging / tests.
    app.library = library  # type: ignore[attr-defined]
    return app
