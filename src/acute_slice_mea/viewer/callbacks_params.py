"""Callbacks for the Analysis Parameters tab.

Each parameter is backed by its own ``dcc.Store`` in the layout
(id = ``{"type": "param-val", "param": <id>}``). Radio buttons and toggles
update these stores; number/text inputs are read directly from their value
property. A single collector callback reads all stores to build the pipeline
config.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from dash import ALL, Input, Output, State, ctx, html, no_update
from dash.exceptions import PreventUpdate

from acute_slice_mea.library import LibraryIndex
from acute_slice_mea.pipeline import AnalysisConfig, run_analysis
from acute_slice_mea.viewer.pages.params import PARAM_SCHEMA

logger = logging.getLogger(__name__)

_run_state = {
    "active": False,
    "done": False,
    "pct": 0,
    "total": 0,
    "current": 0,
    "log": [],
}


def register_params_callbacks(app, library: LibraryIndex) -> None:
    """Bind all Analysis-Parameters tab callbacks."""

    # ── File count in sidebar ────────────────────────────────────────

    @app.callback(
        Output("params-file-count", "children"),
        Input("selected-file-ids", "data"),
    )
    def update_file_count(sel_ids):
        return str(len(sel_ids) if sel_ids else 0)

    # ── Radio button clicks → update store + button styles ───────────

    @app.callback(
        Output("param-store-reref-method", "data"),
        Output({"type": "param-radio-btn", "param": "reref_method", "value": ALL}, "className"),
        Input({"type": "param-radio-btn", "param": "reref_method", "value": ALL}, "n_clicks"),
        State("param-store-reref-method", "data"),
        State({"type": "param-radio-btn", "param": "reref_method", "value": ALL}, "id"),
        prevent_initial_call=True,
    )
    def on_reref_method_click(_clicks, current, radio_ids):
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            raise PreventUpdate
        new_val = triggered["value"]
        classes = [
            "radio-pill is-active" if rid["value"] == new_val else "radio-pill"
            for rid in radio_ids
        ]
        return new_val, classes

    @app.callback(
        Output("param-store-reref-scope", "data"),
        Output({"type": "param-radio-btn", "param": "reref_scope", "value": ALL}, "className"),
        Input({"type": "param-radio-btn", "param": "reref_scope", "value": ALL}, "n_clicks"),
        State("param-store-reref-scope", "data"),
        State({"type": "param-radio-btn", "param": "reref_scope", "value": ALL}, "id"),
        prevent_initial_call=True,
    )
    def on_reref_scope_click(_clicks, current, radio_ids):
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            raise PreventUpdate
        new_val = triggered["value"]
        classes = [
            "radio-pill is-active" if rid["value"] == new_val else "radio-pill"
            for rid in radio_ids
        ]
        return new_val, classes

    # ── Visibility: hide inner/outer radius when scope is global ─────

    @app.callback(
        Output({"type": "param-field", "param": "reref_inner_radius"}, "style"),
        Output({"type": "param-field", "param": "reref_outer_radius"}, "style"),
        Output({"type": "param-field", "param": "reref_method"}, "style"),
        Output({"type": "param-field", "param": "reref_scope"}, "style"),
        Input("param-store-reref-enabled", "data"),
        Input("param-store-reref-scope", "data"),
    )
    def toggle_reref_visibility(enabled, scope):
        if not enabled:
            hide = {"display": "none"}
            return hide, hide, hide, hide
        show = {}
        if scope == "local":
            return show, show, show, show
        return {"display": "none"}, {"display": "none"}, show, show

    # ── Run pipeline ─────────────────────────────────────────────────

    @app.callback(
        Output("run-progress-area", "style"),
        Output("run-pipeline-btn", "children"),
        Output("run-pipeline-btn", "disabled"),
        Output("run-poll-interval", "disabled"),
        Output("pipeline-dot", "className"),
        Output("pipeline-status-text", "children"),
        Input("run-pipeline-btn", "n_clicks"),
        State("selected-file-ids", "data"),
        State("scan-results", "data"),
        # Read all param values from their individual inputs/stores
        State({"type": "param-input", "param": "cache_root"}, "value"),
        State("param-store-cache-overwrite", "data"),
        State({"type": "param-input", "param": "bandpass_low"}, "value"),
        State({"type": "param-input", "param": "bandpass_high"}, "value"),
        State({"type": "param-input", "param": "notch_freqs"}, "value"),
        State({"type": "param-input", "param": "notch_q"}, "value"),
        State({"type": "param-input", "param": "downsample"}, "value"),
        State("param-store-reref-enabled", "data"),
        State("param-store-reref-method", "data"),
        State("param-store-reref-scope", "data"),
        State({"type": "param-input", "param": "reref_inner_radius"}, "value"),
        State({"type": "param-input", "param": "reref_outer_radius"}, "value"),
        State({"type": "param-input", "param": "spectrogram_window"}, "value"),
        State({"type": "param-input", "param": "spectrogram_overlap"}, "value"),
        State({"type": "param-input", "param": "burst_threshold"}, "value"),
        State({"type": "param-input", "param": "burst_min_duration"}, "value"),
        prevent_initial_call=True,
    )
    def on_run_pipeline(
        n_clicks, sel_ids, scan_results,
        cache_root, overwrite,
        bp_low, bp_high, notch_str, notch_q, downsample,
        reref_enabled, reref_method, reref_scope,
        inner_r, outer_r,
        spec_window, spec_overlap,
        burst_thresh, burst_min_dur,
    ):
        if not n_clicks or not sel_ids:
            raise PreventUpdate
        if _run_state["active"]:
            raise PreventUpdate

        cache_root = cache_root or "/data/processed"

        # Parse notch frequencies
        notch_freqs = []
        if notch_str:
            for tok in str(notch_str).replace(" ", "").split(","):
                try:
                    notch_freqs.append(float(tok))
                except ValueError:
                    pass

        # Build job list from selected file IDs
        row_map = {r["id"]: r for r in (scan_results or [])}
        jobs = []
        for fid in sel_ids:
            row = row_map.get(fid)
            if row and row.get("raw_path"):
                jobs.append({
                    "data_path": row["raw_path"],
                    "well_id": row["well_id"],
                    "rec_name": row.get("rec_name"),
                    "label": f"{row.get('sample', '?')}/{row.get('plate', '?')}/{row['well_id']}",
                })

        if not jobs:
            raise PreventUpdate

        config_kwargs = dict(
            lfp_low_hz=float(bp_low or 0.5),
            lfp_high_hz=float(bp_high or 300),
            notch_freqs_hz=notch_freqs,
            notch_q=float(notch_q or 30),
            downsample_hz=int(downsample or 0),
            apply_lfp_common_reference=bool(reref_enabled),
            lfp_reference_method=reref_method or "median",
            lfp_reference_scope=reref_scope or "global",
            lfp_reference_inner_radius=float(inner_r or 30),
            lfp_reference_outer_radius=float(outer_r or 200),
            welch_segment_sec=float(spec_window or 2.0),
            welch_overlap_frac=float(spec_overlap or 0.5),
            burst_zscore_threshold=float(burst_thresh or 3.0),
            burst_min_duration_ms=float(burst_min_dur or 30),
            verbose=True,
        )

        _run_state.update({
            "active": True, "done": False, "pct": 0,
            "total": len(jobs), "current": 0,
            "log": [f"[00:00] Starting pipeline for {len(jobs)} well(s)…"],
        })

        def _worker():
            t0 = time.time()
            for i, job in enumerate(jobs):
                elapsed = int(time.time() - t0)
                stamp = f"[{elapsed//60:02d}:{elapsed%60:02d}]"
                _run_state["current"] = i
                _run_state["pct"] = int(i / len(jobs) * 100)
                _run_state["log"].append(f"{stamp} Processing {job['label']}…")

                output_dir = str(Path(cache_root) / job["label"].replace("/", "_"))
                if not overwrite and Path(output_dir, "manifest.json").exists():
                    _run_state["log"].append(f"  ↳ cache exists, skipping")
                    continue

                try:
                    config = AnalysisConfig(
                        data_path=job["data_path"],
                        well_id=job["well_id"],
                        output_dir=output_dir,
                        rec_name=job.get("rec_name"),
                        **config_kwargs,
                    )
                    run_analysis(config)
                    _run_state["log"].append(f"  ↳ done")
                except Exception as exc:
                    _run_state["log"].append(f"  ↳ ERROR: {exc}")
                    logger.exception("Pipeline failed for %s", job["label"])

            elapsed = int(time.time() - t0)
            stamp = f"[{elapsed//60:02d}:{elapsed%60:02d}]"
            _run_state["pct"] = 100
            _run_state["current"] = len(jobs)
            _run_state["log"].append(f"{stamp} ✓ Pipeline complete.")
            _run_state["active"] = False
            _run_state["done"] = True

            try:
                new_lib = LibraryIndex.from_cache_root(cache_root)
                library.recordings = new_lib.recordings
                library.root = new_lib.root
            except Exception:
                logger.exception("Failed to refresh library after pipeline run")

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        return (
            {"display": "block"},
            [html.Span(className="spinner"), " Running pipeline…"],
            True,
            False,  # enable polling
            "dot dot-amber",
            "pipeline running…",
        )

    # ── Poll pipeline progress ───────────────────────────────────────

    @app.callback(
        Output("run-progress-fill", "style"),
        Output("run-progress-label", "children"),
        Output("run-progress-pct", "children"),
        Output("run-log", "children"),
        Output("run-pipeline-btn", "children", allow_duplicate=True),
        Output("run-pipeline-btn", "disabled", allow_duplicate=True),
        Output("run-poll-interval", "disabled", allow_duplicate=True),
        Output("nav-to-review-btn", "style"),
        Output("pipeline-dot", "className", allow_duplicate=True),
        Output("pipeline-status-text", "children", allow_duplicate=True),
        Input("run-poll-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def poll_progress(_n):
        pct = _run_state["pct"]
        active = _run_state["active"]
        done = _run_state["done"]
        log_lines = [html.Div(line) for line in _run_state["log"]]

        if done and not active:
            return (
                {"width": "100%"},
                "Complete",
                "100%",
                log_lines,
                "Run again ↻",
                False,
                True,
                {"display": "block", "justifyContent": "center"},
                "dot dot-live",
                "pipeline ready",
            )
        return (
            {"width": f"{pct}%"},
            f"Running… ({_run_state['current']}/{_run_state['total']})",
            f"{pct}%",
            log_lines,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
        )

    # ── Navigate to review tab ───────────────────────────────────────

    @app.callback(
        Output("active-tab", "data", allow_duplicate=True),
        Input("nav-to-review-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def nav_to_review(n):
        if not n:
            raise PreventUpdate
        return "review"
