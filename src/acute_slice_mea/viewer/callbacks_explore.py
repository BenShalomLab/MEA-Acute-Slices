"""Callbacks for the Data Explore tab."""

from __future__ import annotations

import logging

from dash import ALL, Input, Output, State, ctx, html, no_update
from dash.exceptions import PreventUpdate

from acute_slice_mea.library import LibraryIndex

logger = logging.getLogger(__name__)

TABLE_COLUMNS = [
    ("sample", "Sample"),
    ("date", "Date"),
    ("plate", "Plate"),
    ("scan", "Scan"),
    ("run", "Run"),
    ("well_id", "Well"),
    ("duration_s", "Dur (s)"),
    ("sample_rate_hz", "Fs (Hz)"),
    ("num_electrodes", "Elecs"),
    ("cached", "Status"),
]


def register_explore_callbacks(app) -> None:
    """Bind all Data-Explore tab callbacks."""

    # ── Scan directory ──────────────────────────────────────────────────

    @app.callback(
        Output("scan-results", "data"),
        Output("scan-progress-area", "children"),
        Output("scan-btn", "children"),
        Output("scan-dot", "className"),
        Output("scan-status-text", "children"),
        Input("scan-btn", "n_clicks"),
        State("data-root-input", "value"),
        prevent_initial_call=True,
    )
    def on_scan_click(n_clicks, data_root):
        if not data_root or not data_root.strip():
            raise PreventUpdate
        try:
            lib = LibraryIndex.from_data_root(data_root.strip())
        except Exception:
            logger.exception("Scan failed for %s", data_root)
            return (
                no_update,
                html.Span("Scan failed", style={"color": "var(--err)"}),
                "Scan directory",
                "dot",
                "scan error",
            )
        rows = _library_to_rows(lib)
        n_recs = len(lib.recordings)
        n_wells = sum(len(r.wells) for r in lib.recordings)
        return (
            rows,
            html.Span(
                f"✓ {n_wells} wells · {n_recs} recordings",
                style={"color": "var(--ok)"},
            ),
            "Rescan",
            "dot dot-live",
            f"{n_wells} wells indexed",
        )

    # ── Table rendering (server-side with Dash components) ──────────────

    @app.callback(
        Output("results-match-count", "children"),
        Output("results-total-text", "children"),
        Output("results-sel-text", "children"),
        Output("results-table-wrap", "children"),
        Output("expr-status", "children"),
        Output("results-footer-text", "children"),
        Input("scan-results", "data"),
        Input("selected-file-ids", "data"),
    )
    def render_table(scan_results, selected_ids):
        rows = scan_results or []
        sel_set = set(selected_ids or [])

        if not rows:
            return (
                "0",
                "of 0 recordings match",
                f"{len(sel_set)} selected",
                html.Div(className="empty", children=[
                    html.Div("No files scanned", className="empty-h"),
                    html.Div("Set a root directory and click Scan to discover .h5 recordings.", className="empty-s"),
                ]),
                "Empty — scan a directory first.",
                "Tick rows to send to the analysis pipeline →",
            )

        n_sel = len(sel_set)
        footer_text = (
            f"{n_sel} file{'s' if n_sel != 1 else ''} selected for analysis"
            if n_sel > 0 else "Tick rows to send to the analysis pipeline →"
        )

        # Build table header
        header_cells = [html.Th(
            html.Input(
                id="select-all-check",
                type="checkbox",
                className="check",
                n_clicks=0,
            ),
            className="checkcell",
        )]
        for key, label in TABLE_COLUMNS:
            header_cells.append(html.Th(label))

        # Build table rows
        body_rows = []
        for r in rows:
            rid = r.get("id", "")
            is_sel = rid in sel_set
            row_cls = "is-selected" if is_sel else ""

            cells = [html.Td(
                html.Input(
                    id={"type": "row-check", "rid": rid},
                    type="checkbox",
                    className="check",
                    checked=is_sel,
                    n_clicks=0,
                ),
                className="checkcell",
            )]

            for key, label in TABLE_COLUMNS:
                val = r.get(key, "")
                if key == "scan":
                    scan_cls = {
                        "Network": "scan-network", "Activity": "scan-activity",
                        "AxonTracking": "scan-axon", "Spike": "scan-spike",
                    }.get(val, "")
                    cells.append(html.Td(html.Span(val or "", className=f"scan-pill {scan_cls}")))
                elif key == "cached":
                    pill_cls = "status-pill is-cached" if val else "status-pill is-raw"
                    cells.append(html.Td(html.Span("cached" if val else "raw", className=pill_cls)))
                elif key == "sample":
                    cells.append(html.Td(val or "", className="cell-sample"))
                elif key == "sample_rate_hz":
                    display = f"{int(val/1000)}k" if val else "—"
                    cells.append(html.Td(display, className="cell-num"))
                elif key == "duration_s":
                    display = str(int(val)) if val else "—"
                    cells.append(html.Td(display, className="cell-num"))
                elif key == "num_electrodes":
                    display = f"{val:,}" if val else "—"
                    cells.append(html.Td(display, className="cell-num"))
                else:
                    cells.append(html.Td(str(val) if val is not None else ""))

            body_rows.append(html.Tr(cells, className=row_cls))

        table = html.Table(
            className="ftable",
            children=[
                html.Thead(html.Tr(header_cells)),
                html.Tbody(body_rows),
            ],
        )

        status = f"Showing all {len(rows)} rows."

        return (
            str(len(rows)),
            f"of {len(rows)} recordings match",
            f"{n_sel} selected",
            table,
            status,
            footer_text,
        )

    # ── Row checkbox toggle ──────────────────────────────────────────

    @app.callback(
        Output("selected-file-ids", "data"),
        Input({"type": "row-check", "rid": ALL}, "n_clicks"),
        State({"type": "row-check", "rid": ALL}, "id"),
        State({"type": "row-check", "rid": ALL}, "checked"),
        State("selected-file-ids", "data"),
        prevent_initial_call=True,
    )
    def on_row_check(_clicks, check_ids, checked_states, current_sel):
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            raise PreventUpdate
        rid = triggered["rid"]
        sel_set = set(current_sel or [])
        if rid in sel_set:
            sel_set.discard(rid)
        else:
            sel_set.add(rid)
        return sorted(sel_set)

    # ── Select all ───────────────────────────────────────────────────

    @app.callback(
        Output("selected-file-ids", "data", allow_duplicate=True),
        Input("select-all-check", "n_clicks"),
        State("scan-results", "data"),
        State("selected-file-ids", "data"),
        prevent_initial_call=True,
    )
    def on_select_all(n, scan_results, current_sel):
        if not n:
            raise PreventUpdate
        all_ids = [r["id"] for r in (scan_results or [])]
        sel_set = set(current_sel or [])
        all_selected = all(rid in sel_set for rid in all_ids)
        if all_selected:
            return []
        return all_ids

    # ── Selection count in status bar ────────────────────────────────

    @app.callback(
        Output("sel-dot", "className"),
        Output("sel-status-text", "children"),
        Input("selected-file-ids", "data"),
    )
    def update_sel_status(sel_ids):
        n = len(sel_ids) if sel_ids else 0
        dot_cls = "dot dot-amber" if n > 0 else "dot"
        return dot_cls, f"{n} selected"

    # ── Example presets ──────────────────────────────────────────────

    @app.callback(
        Output("expr-textarea", "value"),
        Input({"type": "example-row", "expr": ALL}, "n_clicks"),
        Input("expr-clear-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def on_example_or_clear(*_):
        triggered = ctx.triggered_id
        if triggered == "expr-clear-btn":
            return ""
        if isinstance(triggered, dict) and triggered.get("type") == "example-row":
            return triggered["expr"]
        raise PreventUpdate

    # ── Navigate to params tab ───────────────────────────────────────

    @app.callback(
        Output("active-tab", "data", allow_duplicate=True),
        Input("nav-to-params-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def nav_to_params(n):
        if not n:
            raise PreventUpdate
        return "params"


def _library_to_rows(lib: LibraryIndex) -> list[dict]:
    """Convert a LibraryIndex into flat rows for the table."""
    rows = []
    for rec in lib.recordings:
        for well in rec.wells:
            rows.append({
                "id": f"{rec.recording_id}/{well.well_id}",
                "sample": rec.sample,
                "date": rec.iso_date,
                "plate": rec.plate,
                "scan": rec.scan,
                "run": rec.run,
                "well_id": well.well_id,
                "duration_s": well.duration_sec,
                "sample_rate_hz": well.sample_rate_hz,
                "num_electrodes": well.num_recorded_electrodes,
                "cached": well.has_dashboard_data,
                "operator": rec.operator or "",
                "recording_id": rec.recording_id,
                "raw_path": well.raw_path or rec.raw_path or "",
                "rec_name": well.rec_name,
            })
    return rows
