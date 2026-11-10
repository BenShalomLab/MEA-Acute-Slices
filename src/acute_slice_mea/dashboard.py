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


def _compute_downsampling_params(
    num_samples: int,
    fs: float,
    max_points_per_electrode: int,
    target_rate_hz: float | None = None,
) -> tuple[int, bool]:
    """Calculate bin size and aggregation mode balanced by recording length.

    Parameters
    ----------
    num_samples : int
        Total frames in the recording.
    fs : float
        Sampling frequency (Hz).
    max_points_per_electrode : int
        Maximum allowed display points per electrode.
    target_rate_hz : float | None
        Optional desired temporal resolution (e.g. 2000 Hz for spikes/raw or
        1000 Hz for LFP).

    Returns
    -------
    bin_size : int
        Number of raw frames per downsampling bin (>= 1).
    use_minmax : bool
        True if Min-Max aggregation is used (when downsampling); False if
        keeping un-decimated samples.
    """
    max_points = max(2, int(max_points_per_electrode))
    if num_samples <= max_points:
        return 1, False

    # Min-Max emits up to 2 points per bin (min and max chronologically)
    max_bins = max(1, max_points // 2)

    if target_rate_hz is not None and target_rate_hz > 0:
        # Balanced sampling: maintain desired target_rate_hz for shorter
        # recordings, but scale up bin_size gracefully when duration causes
        # bin count to exceed max_bins.
        ideal_bin_size = max(2, int(round(fs / float(target_rate_hz))))
        if num_samples // ideal_bin_size <= max_bins:
            return ideal_bin_size, True

    bin_size = max(2, int(np.ceil(num_samples / max_bins)))
    return bin_size, True


def _read_decimated_all_channels(
    recording,
    channel_ids: list[str],
    num_samples: int,
    fs: float | None = None,
    bin_size: int = 1,
    use_minmax: bool = True,
    chunk_frames_target: int | None = None,
    step: int | None = None,
    chunk_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read and aggregate all requested channels in one pass across time chunks.

    Applies Min-Max aggregation (preserving action potentials, burst envelopes,
    and extrema) or full-resolution slicing across chunk-aligned boundaries.
    The underlying recording applies a global common reference (median
    across ALL channels), so requesting even a single channel forces
    SpikeInterface to read/filter every channel internally anyway. Calling
    this once per signal (instead of once per electrode per signal) turns
    ~n_electrodes full-duration multi-channel reads into 1.

    Returns
    -------
    time_sec : np.ndarray
        1D array of timestamps (seconds).
    values : np.ndarray
        2D array of shape (n_points, len(channel_ids)) in physical units.
    """
    if fs is None:
        fs = float(recording.get_sampling_frequency())

    # Backwards compatibility: allow passing legacy step and chunk_frames
    if step is not None and bin_size == 1:
        bin_size = max(1, int(step))
        use_minmax = bin_size > 1

    if not channel_ids or num_samples <= 0:
        return np.empty(0, dtype=np.float64), np.empty((0, len(channel_ids)), dtype=np.float32)

    if not use_minmax or bin_size <= 1:
        # Full-resolution or unaggregated read
        chunk_target = chunk_frames or chunk_frames_target or max(1, int(round(5.0 * fs)))
        all_vals = []
        all_times = []
        for chunk_start in range(0, num_samples, chunk_target):
            chunk_end = min(num_samples, chunk_start + chunk_target)
            n_chunk = chunk_end - chunk_start
            traces = get_traces_safe(recording, chunk_start, chunk_end, channel_ids)
            arr = np.asarray(traces, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            all_vals.append(arr)
            all_times.append((np.arange(n_chunk, dtype=np.float64) + chunk_start) / fs)
        if all_vals:
            return np.concatenate(all_times), np.vstack(all_vals)
        return np.empty(0, dtype=np.float64), np.empty((0, len(channel_ids)), dtype=np.float32)

    # Min-Max aggregation: align chunk boundaries strictly to bin multiples
    chunk_target = chunk_frames or chunk_frames_target or max(bin_size, int(round(5.0 * fs)))
    aligned_chunk_frames = max(bin_size, (chunk_target // bin_size) * bin_size)

    all_times = []
    all_vals = []

    for chunk_start in range(0, num_samples, aligned_chunk_frames):
        chunk_end = min(num_samples, chunk_start + aligned_chunk_frames)
        n_chunk_samples = chunk_end - chunk_start
        traces = get_traces_safe(recording, chunk_start, chunk_end, channel_ids)
        arr = np.asarray(traces, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]

        n_full_bins = n_chunk_samples // bin_size
        rem = n_chunk_samples % bin_size

        if n_full_bins > 0:
            full_samples = n_full_bins * bin_size
            reshaped = arr[:full_samples].reshape(n_full_bins, bin_size, len(channel_ids))
            min_vals = np.min(reshaped, axis=1)
            max_vals = np.max(reshaped, axis=1)
            min_idx = np.argmin(reshaped, axis=1)
            max_idx = np.argmax(reshaped, axis=1)

            order = min_idx <= max_idx
            first_vals = np.where(order, min_vals, max_vals)
            second_vals = np.where(order, max_vals, min_vals)

            full_vals = np.empty((n_full_bins * 2, len(channel_ids)), dtype=np.float32)
            full_vals[0::2, :] = first_vals
            full_vals[1::2, :] = second_vals

            bin_starts = np.arange(n_full_bins, dtype=np.float64) * bin_size + chunk_start
            bin_ends = bin_starts + (bin_size - 1)
            full_times = np.empty(n_full_bins * 2, dtype=np.float64)
            full_times[0::2] = bin_starts / fs
            full_times[1::2] = bin_ends / fs
        else:
            full_vals = np.empty((0, len(channel_ids)), dtype=np.float32)
            full_times = np.empty(0, dtype=np.float64)
            full_samples = 0

        if rem == 1:
            rem_vals = np.asarray(arr[full_samples:], dtype=np.float32)
            rem_times = np.array([(chunk_start + full_samples) / fs], dtype=np.float64)
            chunk_vals = np.vstack([full_vals, rem_vals]) if full_vals.size else rem_vals
            chunk_times = np.concatenate([full_times, rem_times]) if full_times.size else rem_times
        elif rem > 1:
            rem_slice = arr[full_samples:]
            min_vals_rem = np.min(rem_slice, axis=0)
            max_vals_rem = np.max(rem_slice, axis=0)
            min_idx_rem = np.argmin(rem_slice, axis=0)
            max_idx_rem = np.argmax(rem_slice, axis=0)
            order_rem = min_idx_rem <= max_idx_rem
            first_rem = np.where(order_rem, min_vals_rem, max_vals_rem)
            second_rem = np.where(order_rem, max_vals_rem, min_vals_rem)
            rem_vals = np.vstack([first_rem, second_rem]).astype(np.float32)
            rem_times = np.array([
                (chunk_start + full_samples) / fs,
                (chunk_start + full_samples + rem - 1) / fs,
            ], dtype=np.float64)
            chunk_vals = np.vstack([full_vals, rem_vals]) if full_vals.size else rem_vals
            chunk_times = np.concatenate([full_times, rem_times]) if full_times.size else rem_times
        else:
            chunk_vals = full_vals
            chunk_times = full_times

        all_vals.append(chunk_vals)
        all_times.append(chunk_times)

    if all_vals:
        return np.concatenate(all_times), np.vstack(all_vals)
    return np.empty(0, dtype=np.float64), np.empty((0, len(channel_ids)), dtype=np.float32)


def export_dashboard_data(
    dashboard_dir,
    *,
    recordings: dict[str, object],
    electrodes: pd.DataFrame,
    band_power: pd.DataFrame,
    summary: dict,
    max_points_per_electrode=20000,
    include_signals=("raw", "lfp", "spike"),
    target_rate_hz: float | None = None,
) -> dict:
    """Export dashboard data files for lazy loading by a static HTML page."""
    dashboard_dir = Path(dashboard_dir)
    data_dir = dashboard_dir / "data"
    traces_dir = data_dir / "traces"
    recorded = _recorded_electrodes(electrodes)
    signals = [signal for signal in include_signals if signal in recordings]
    if not signals:
        raise ValueError("No requested dashboard signals are available in recordings.")

    first_recording = recordings[signals[0]]
    fs = float(first_recording.get_sampling_frequency())
    num_samples = int(first_recording.get_num_samples())
    bin_size, use_minmax = _compute_downsampling_params(
        num_samples=num_samples,
        fs=fs,
        max_points_per_electrode=max_points_per_electrode,
        target_rate_hz=target_rate_hz,
    )

    channel_ids_all = recorded["channel_id"].tolist()

    trace_index: dict[str, dict[str, str]] = {signal: {} for signal in signals}
    for signal in signals:
        recording = recordings[signal]
        signal_dir = traces_dir / signal
        time_sec, decimated = _read_decimated_all_channels(
            recording,
            channel_ids_all,
            num_samples=num_samples,
            fs=fs,
            bin_size=bin_size,
            use_minmax=use_minmax,
        )
        for col_idx, row in enumerate(recorded.itertuples(index=False)):
            values = np.asarray(decimated[:, col_idx], dtype=float)
            payload = {
                "signal": signal,
                "signal_label": SIGNAL_LABELS.get(signal, signal),
                "electrode_id": int(row.electrode_id),
                "channel_id": row.channel_id,
                "x_um": float(row.x_um),
                "y_um": float(row.y_um),
                "sample_step": int(bin_size),
                "time_sec": time_sec,
                "value": values,
            }
            trace_path = signal_dir / f"{int(row.electrode_id)}.json"
            _write_json(trace_path, payload)
            trace_index[signal][str(int(row.electrode_id))] = str(trace_path.relative_to(dashboard_dir))

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
        "sample_step": int(bin_size),
        "aggregation": "minmax" if use_minmax else "none",
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
