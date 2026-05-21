"""Static dashboard data export and HTML generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from acute_slice_mea.cache import load_band_power, load_cache_manifest
from acute_slice_mea.spectral import DEFAULT_LFP_BANDS, get_traces_safe

SIGNAL_LABELS = {
    "raw": "No filter",
    "lfp": "LFP 0.5-300 Hz",
    "spike": "Spike 300-3000 Hz",
}


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _clean_records(df: pd.DataFrame) -> list[dict]:
    records = []
    for record in df.to_dict(orient="records"):
        cleaned = {}
        for key, value in record.items():
            if pd.isna(value):
                cleaned[key] = None
            elif isinstance(value, np.generic):
                cleaned[key] = value.item()
            else:
                cleaned[key] = value
        records.append(cleaned)
    return records


def _write_json(path: Path, payload: dict | list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, default=_json_default, separators=(",", ":")))
    return path


def _recorded_electrodes(electrodes: pd.DataFrame) -> pd.DataFrame:
    recorded = electrodes[electrodes["recorded"].astype(bool)].copy()
    if recorded.empty:
        raise ValueError("No recorded electrodes available for dashboard export.")
    recorded["electrode_id"] = recorded["electrode_id"].astype(int)
    return recorded.sort_values("electrode_id")


def export_dashboard_data(
    dashboard_dir,
    *,
    recordings: dict[str, object],
    electrodes: pd.DataFrame,
    band_power: pd.DataFrame,
    summary: dict,
    max_points_per_electrode=None,
    include_signals=("lfp",),
    progress_callback=None,
) -> dict:
    """Export dashboard data files for lazy loading by the trace viewer.

    Per-electrode traces are written as compressed NumPy ``.npz`` at the
    recording's full sample rate (1 kHz for the LFP path after resample).
    The dashboard's plotly-resampler integration aggregates from this
    full-resolution source at display time, so we keep all detail on disk
    and only decide point density inside the browser.

    Only the LFP signal is exported by default — this codebase analyses
    LFP only, and the trace viewer reads ``lfp`` from the materialized
    on-disk binary. Callers that need raw/spike previews can opt back in
    by passing ``include_signals=("raw", "lfp", "spike")``.

    ``max_points_per_electrode`` is kept for backward compatibility (older
    callers pass it explicitly). When given a positive value it caps the
    on-disk length to that many points via stride decimation, matching the
    legacy JSON behavior. Default ``None`` preserves full resolution.

    ``progress_callback(fraction)`` is invoked periodically during the
    per-electrode loop (``fraction`` in [0, 1]) so the job runner can drive
    a smooth progress bar through the saving phase.
    """
    dashboard_dir = Path(dashboard_dir)
    data_dir = dashboard_dir / "data"
    traces_dir = data_dir / "traces"
    recorded = _recorded_electrodes(electrodes)
    signals = [signal for signal in include_signals if signal in recordings]
    if not signals:
        raise ValueError("No requested dashboard signals are available in recordings.")

    trace_index: dict[str, dict[str, str]] = {signal: {} for signal in signals}
    signal_steps: dict[str, int] = {}
    # Progress is reported over the total electrode×signal write count so the
    # bar advances per electrode regardless of how many signals are exported.
    total_writes = int(len(recorded) * len(signals))
    writes_done = 0

    def _report_progress() -> None:
        if progress_callback is None or total_writes == 0:
            return
        try:
            progress_callback(writes_done / total_writes)
        except Exception:  # never let a UI hook break the export
            pass

    for signal in signals:
        recording = recordings[signal]
        fs = float(recording.get_sampling_frequency())
        num_samples = int(recording.get_num_samples())
        if max_points_per_electrode is not None and int(max_points_per_electrode) > 0:
            step = max(1, int(np.ceil(num_samples / int(max_points_per_electrode))))
        else:
            step = 1
        sample_frames = np.arange(0, num_samples, step)
        time_sec = (sample_frames / fs).astype(np.float32)
        signal_steps[signal] = int(step)
        signal_dir = traces_dir / signal
        signal_dir.mkdir(parents=True, exist_ok=True)
        for row in recorded.itertuples(index=False):
            traces = get_traces_safe(recording, 0, num_samples, [row.channel_id])
            values = np.asarray(traces[::step, 0], dtype=np.float32)
            # Per-electrode .npz at full resolution. Meta scalars are kept in
            # a small JSON header file alongside (avoids string encoding
            # awkwardness in numpy archives).
            trace_path = signal_dir / f"{int(row.electrode_id)}.npz"
            np.savez_compressed(trace_path, time_sec=time_sec, value=values)
            meta_path = signal_dir / f"{int(row.electrode_id)}.json"
            _write_json(
                meta_path,
                {
                    "signal": signal,
                    "signal_label": SIGNAL_LABELS.get(signal, signal),
                    "electrode_id": int(row.electrode_id),
                    "channel_id": row.channel_id,
                    "x_um": float(row.x_um),
                    "y_um": float(row.y_um),
                    "sample_step": int(step),
                    "sample_rate_hz": float(fs),
                    "num_samples": int(values.shape[0]),
                    "trace_path": str(trace_path.relative_to(dashboard_dir)),
                },
            )
            trace_index[signal][str(int(row.electrode_id))] = str(meta_path.relative_to(dashboard_dir))
            writes_done += 1
            _report_progress()

    electrodes_payload = _clean_records(recorded)
    band_power_payload = _clean_records(
        band_power[["electrode_id", "time_sec", "band", "power_db"]].copy()
    )
    _write_json(data_dir / "electrodes.json", electrodes_payload)
    _write_json(data_dir / "band_power.json", band_power_payload)

    data_manifest = {
        "summary": summary,
        "signals": signals,
        "signal_labels": {signal: SIGNAL_LABELS.get(signal, signal) for signal in signals},
        "bands": DEFAULT_LFP_BANDS,
        "max_points_per_electrode": int(max_points_per_electrode),
        "sample_step": signal_steps,
        "electrodes": electrodes_payload,
        "files": {
            "electrodes": "data/electrodes.json",
            "band_power": "data/band_power.json",
            "traces": trace_index,
        },
    }
    manifest_path = _write_json(data_dir / "manifest.json", data_manifest)
    return {
        "dashboard_dir": str(dashboard_dir),
        "data_manifest": str(manifest_path),
        "num_electrodes": int(len(recorded)),
        "signals": signals,
    }


def write_dashboard_html(dashboard_dir, *, title="Acute Slice MEA Dashboard") -> Path:
    """Write the static dashboard shell."""
    dashboard_dir = Path(dashboard_dir)
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    html = DASHBOARD_HTML.replace("__DASHBOARD_TITLE__", title)
    output_path = dashboard_dir / "index.html"
    output_path.write_text(html)
    return output_path


def write_dashboard_from_cache(cache_dir, *, title="Acute Slice MEA Dashboard") -> Path:
    """Create dashboard HTML for an existing cache directory."""
    cache_dir = Path(cache_dir)
    manifest = load_cache_manifest(cache_dir)
    # Validate key dashboard data created by notebook 01 exists.
    data_manifest = cache_dir / "dashboard" / "data" / "manifest.json"
    if not data_manifest.exists():
        raise FileNotFoundError(
            f"Dashboard data manifest not found: {data_manifest}. Rerun notebook 01 to export dashboard data."
        )
    return write_dashboard_html(cache_dir / "dashboard", title=title)


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__DASHBOARD_TITLE__</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root { color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f7f7f5; color: #1f2933; }
    header { padding: 18px 24px 12px; border-bottom: 1px solid #d7d7d2; background: #ffffff; }
    h1 { margin: 0 0 6px; font-size: 22px; font-weight: 650; letter-spacing: 0; }
    #meta { color: #5f6b76; font-size: 13px; }
    main { display: grid; grid-template-columns: 310px 1fr; min-height: calc(100vh - 74px); }
    aside { border-right: 1px solid #d7d7d2; background: #ffffff; padding: 16px; overflow: auto; }
    section { padding: 14px 18px 22px; overflow: hidden; }
    label { display: block; margin: 12px 0 6px; font-size: 13px; font-weight: 600; }
    select, input { width: 100%; box-sizing: border-box; border: 1px solid #b9c0c7; border-radius: 6px; padding: 7px 8px; font-size: 13px; background: white; }
    select[multiple] { height: 260px; }
    button { border: 1px solid #28666e; background: #2f7a83; color: white; border-radius: 6px; padding: 8px 10px; font-weight: 600; cursor: pointer; }
    button.secondary { background: white; color: #2f7a83; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .actions { display: flex; gap: 8px; margin-top: 12px; }
    .hint { color: #697681; font-size: 12px; line-height: 1.35; margin-top: 8px; }
    .plot { height: 43vh; min-height: 320px; margin-bottom: 14px; border: 1px solid #deded9; background: #ffffff; }
    #status { margin-top: 10px; color: #5f6b76; font-size: 12px; white-space: pre-line; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } aside { border-right: 0; border-bottom: 1px solid #d7d7d2; } }
  </style>
</head>
<body>
  <header>
    <h1>__DASHBOARD_TITLE__</h1>
    <div id="meta">Loading dashboard metadata...</div>
  </header>
  <main>
    <aside>
      <label for="signalSelect">Signal</label>
      <select id="signalSelect"></select>
      <label for="electrodeSelect">Electrodes</label>
      <select id="electrodeSelect" multiple></select>
      <div class="row">
        <div>
          <label for="startSec">Start (s)</label>
          <input id="startSec" type="number" value="0" min="0" step="1">
        </div>
        <div>
          <label for="endSec">End (s)</label>
          <input id="endSec" type="number" value="" min="0" step="1">
        </div>
      </div>
      <div class="actions">
        <button id="updateButton">Update</button>
        <button id="clearButton" class="secondary">Clear</button>
      </div>
      <div class="hint">Use Shift or Cmd/Ctrl to select multiple electrodes. Trace files are loaded only for selected electrodes.</div>
      <div id="status"></div>
    </aside>
    <section>
      <div id="tracePlot" class="plot"></div>
      <div id="bandPlot" class="plot"></div>
    </section>
  </main>
  <script>
    const state = { manifest: null, bandPower: [], traceCache: new Map() };
    const statusEl = document.getElementById("status");

    async function loadJson(path) {
      const response = await fetch(path);
      if (!response.ok) throw new Error(`Failed to load ${path}: ${response.status}`);
      return response.json();
    }

    function selectedElectrodes() {
      return Array.from(document.getElementById("electrodeSelect").selectedOptions).map(option => option.value);
    }

    function timeWindow() {
      const summary = state.manifest.summary || {};
      const duration = Number(summary.duration_sec || 0);
      const start = Number(document.getElementById("startSec").value || 0);
      const rawEnd = document.getElementById("endSec").value;
      const end = rawEnd === "" ? duration : Number(rawEnd);
      return { start, end };
    }

    function filterByTime(x, y, start, end) {
      const fx = [];
      const fy = [];
      for (let i = 0; i < x.length; i += 1) {
        if (x[i] >= start && x[i] <= end) {
          fx.push(x[i]);
          fy.push(y[i]);
        }
      }
      return [fx, fy];
    }

    async function loadTrace(signal, electrodeId) {
      const key = `${signal}:${electrodeId}`;
      if (state.traceCache.has(key)) return state.traceCache.get(key);
      const path = state.manifest.files.traces[signal][electrodeId];
      const payload = await loadJson(path);
      state.traceCache.set(key, payload);
      return payload;
    }

    async function updatePlots() {
      const signal = document.getElementById("signalSelect").value;
      const electrodes = selectedElectrodes();
      const { start, end } = timeWindow();
      if (electrodes.length === 0) {
        Plotly.react("tracePlot", [], { title: "Select at least one electrode", template: "plotly_white" });
        Plotly.react("bandPlot", [], { title: "Select at least one electrode", template: "plotly_white" });
        return;
      }
      statusEl.textContent = `Loading ${electrodes.length} electrode(s)...`;
      const traces = [];
      for (const electrodeId of electrodes) {
        const payload = await loadTrace(signal, electrodeId);
        const [x, y] = filterByTime(payload.time_sec, payload.value, start, end);
        traces.push({
          type: "scattergl",
          mode: "lines",
          x,
          y,
          name: `E${payload.electrode_id} / Ch ${payload.channel_id}`,
          line: { width: 1 }
        });
      }
      Plotly.react("tracePlot", traces, {
        title: `${state.manifest.signal_labels[signal]} traces`,
        template: "plotly_white",
        xaxis: { title: "Time (s)" },
        yaxis: { title: "Amplitude" },
        hovermode: "closest",
        margin: { l: 60, r: 20, t: 45, b: 50 }
      }, { responsive: true, displaylogo: false });

      const bandTraces = [];
      const bandOrder = Object.keys(state.manifest.bands || {});
      for (const band of bandOrder) {
        const grouped = new Map();
        for (const row of state.bandPower) {
          if (row.band !== band || !electrodes.includes(String(row.electrode_id))) continue;
          if (row.time_sec < start || row.time_sec > end) continue;
          if (!grouped.has(row.time_sec)) grouped.set(row.time_sec, []);
          grouped.get(row.time_sec).push(row.power_db);
        }
        const times = Array.from(grouped.keys()).sort((a, b) => a - b);
        if (times.length === 0) continue;
        bandTraces.push({
          type: "scattergl",
          mode: "lines+markers",
          x: times,
          y: times.map(t => grouped.get(t).reduce((a, b) => a + b, 0) / grouped.get(t).length),
          name: band.replace("_", " ")
        });
      }
      Plotly.react("bandPlot", bandTraces, {
        title: "Selected-electrode LFP band power",
        template: "plotly_white",
        xaxis: { title: "Time (s)" },
        yaxis: { title: "Power (dB)" },
        hovermode: "x unified",
        margin: { l: 60, r: 20, t: 45, b: 50 }
      }, { responsive: true, displaylogo: false });
      statusEl.textContent = `Loaded ${electrodes.length} electrode(s), ${start}-${end} s.`;
    }

    async function init() {
      state.manifest = await loadJson("data/manifest.json");
      state.bandPower = await loadJson(state.manifest.files.band_power);
      const summary = state.manifest.summary || {};
      document.getElementById("meta").textContent =
        `Well ${summary.well_id || ""} | ${Number(summary.duration_sec || 0).toFixed(1)} s | ` +
        `${summary.num_recorded_electrodes || state.manifest.electrodes.length} recorded electrodes`;
      document.getElementById("endSec").value = Number(summary.duration_sec || 0).toFixed(0);

      const signalSelect = document.getElementById("signalSelect");
      for (const signal of state.manifest.signals) {
        const option = document.createElement("option");
        option.value = signal;
        option.textContent = state.manifest.signal_labels[signal] || signal;
        signalSelect.appendChild(option);
      }

      const electrodeSelect = document.getElementById("electrodeSelect");
      for (const electrode of state.manifest.electrodes) {
        const option = document.createElement("option");
        option.value = String(electrode.electrode_id);
        option.textContent = `E${electrode.electrode_id} / Ch ${electrode.channel_id} (${electrode.x_um}, ${electrode.y_um})`;
        electrodeSelect.appendChild(option);
      }
      for (let i = 0; i < Math.min(3, electrodeSelect.options.length); i += 1) {
        electrodeSelect.options[i].selected = true;
      }
      Plotly.newPlot("tracePlot", [], { title: "Loading traces", template: "plotly_white" }, { responsive: true, displaylogo: false });
      Plotly.newPlot("bandPlot", [], { title: "Loading band power", template: "plotly_white" }, { responsive: true, displaylogo: false });
      document.getElementById("updateButton").addEventListener("click", updatePlots);
      document.getElementById("clearButton").addEventListener("click", () => {
        for (const option of electrodeSelect.options) option.selected = false;
        updatePlots();
      });
      signalSelect.addEventListener("change", updatePlots);
      await updatePlots();
    }

    init().catch(error => {
      statusEl.textContent = error.stack || String(error);
    });
  </script>
</body>
</html>
"""
