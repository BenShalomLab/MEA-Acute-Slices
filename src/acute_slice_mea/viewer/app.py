"""Dash app factory for the Acute Slice trace viewer."""

from __future__ import annotations

from pathlib import Path

from dash import Dash

from acute_slice_mea.jobs import JobsBackend
from acute_slice_mea.library import LibraryIndex
from acute_slice_mea.viewer.callbacks import register_all
from acute_slice_mea.viewer.layout import build_layout


def create_app(
    cache_root: str | Path,
    *,
    data_root: str | Path | None = None,
    title: str = "Acute Slice · Trace Viewer",
) -> Dash:
    """Build the Dash app.

    Args:
        cache_root: Directory holding per-(recording, well) cache bundles.
        data_root: Optional raw ``.h5`` tree. When given, the library shows
            every recording (cached or not) and the dashboard can spawn
            analyses; without it the app stays view-only.
    """
    if data_root is not None:
        library = LibraryIndex.from_data_and_cache_root(data_root, cache_root)
    else:
        library = LibraryIndex.from_cache_root(cache_root)

    jobs_backend = JobsBackend(cache_root) if data_root is not None else None

    app = Dash(
        __name__,
        title=title,
        update_title=None,
        suppress_callback_exceptions=True,
    )
    app.layout = build_layout(library, jobs_enabled=jobs_backend is not None)
    register_all(app, library, jobs_backend=jobs_backend)
    app.library = library  # type: ignore[attr-defined]
    app.jobs_backend = jobs_backend  # type: ignore[attr-defined]
    return app
