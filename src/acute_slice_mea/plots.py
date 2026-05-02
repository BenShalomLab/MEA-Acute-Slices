"""Plotly HTML output helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from acute_slice_mea.spectral import DEFAULT_LFP_BANDS

BAND_COLORS = {
    "delta": "#4c78a8",
    "theta": "#f58518",
    "alpha": "#54a24b",
    "beta": "#e45756",
    "low_gamma": "#b279a2",
    "high_gamma": "#9d755d",
}


def _write_html(fig, output_path, include_plotlyjs="cdn") -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        output_path,
        include_plotlyjs=include_plotlyjs,
        config={"responsive": True, "displaylogo": False},
    )
    return output_path


def write_band_power_html(band_power: pd.DataFrame, output_path, include_plotlyjs="cdn") -> Path:
    """Write mean LFP band power over time as an interactive HTML file."""
    fig = go.Figure()
    grouped = (
        band_power.groupby(["time_sec", "band"], as_index=False, observed=True)["power_db"]
        .mean()
        .sort_values(["band", "time_sec"])
    )
    for band_name in DEFAULT_LFP_BANDS:
        band_df = grouped[grouped["band"] == band_name]
        if band_df.empty:
            continue
        fig.add_trace(
            go.Scattergl(
                x=band_df["time_sec"],
                y=band_df["power_db"],
                mode="lines+markers",
                name=band_name.replace("_", " "),
                line={"color": BAND_COLORS.get(band_name), "width": 2},
                marker={"size": 5},
            )
        )
    fig.update_layout(
        title="Mean LFP band power over time",
        template="plotly_white",
        xaxis_title="Time (s)",
        yaxis_title="Band power (dB)",
        hovermode="x unified",
    )
    return _write_html(fig, output_path, include_plotlyjs)


def write_trace_preview_html(trace_preview: dict[str, np.ndarray], output_path, include_plotlyjs="cdn") -> Path:
    """Write raw/LFP/spike preview traces to HTML using WebGL traces."""
    signal_names = [name for name in ("raw", "lfp", "spike") if name in trace_preview]
    fig = make_subplots(
        rows=len(signal_names),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[name.upper() for name in signal_names],
        vertical_spacing=0.08,
    )
    time_sec = trace_preview["time_sec"]
    electrode_ids = trace_preview.get("electrode_ids", np.arange(trace_preview[signal_names[0]].shape[1]))
    channel_ids = trace_preview.get("channel_ids", electrode_ids)

    for row, name in enumerate(signal_names, start=1):
        traces = trace_preview[name]
        scale = np.nanpercentile(np.abs(traces), 95)
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        offsets = np.arange(traces.shape[1]) * scale * 2.5
        for idx in range(traces.shape[1]):
            fig.add_trace(
                go.Scattergl(
                    x=time_sec,
                    y=traces[:, idx] + offsets[idx],
                    mode="lines",
                    name=f"{name} E{electrode_ids[idx]} / Ch {channel_ids[idx]}",
                    legendgroup=f"{name}-{idx}",
                    line={"width": 1},
                ),
                row=row,
                col=1,
            )
        fig.update_yaxes(title_text=f"{name} + offset", row=row, col=1)
    fig.update_xaxes(title_text="Time (s)", row=len(signal_names), col=1)
    fig.update_layout(
        title="Raw, LFP, and spike-filtered trace preview",
        template="plotly_white",
        height=max(360, 280 * len(signal_names)),
        hovermode="closest",
    )
    return _write_html(fig, output_path, include_plotlyjs)


def write_spectrum_summary_html(spectrum: dict[str, np.ndarray], output_path, include_plotlyjs="cdn") -> Path:
    """Write all-electrode spectrum summary statistics to HTML."""
    freq = spectrum["freq_hz"]
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=freq, y=spectrum["mean_psd"], mode="lines", name="Mean PSD"))
    fig.add_trace(go.Scattergl(x=freq, y=spectrum["median_psd"], mode="lines", name="Median PSD"))
    fig.add_trace(
        go.Scattergl(
            x=np.concatenate([freq, freq[::-1]]),
            y=np.concatenate([spectrum["p95_psd"], spectrum["p05_psd"][::-1]]),
            fill="toself",
            mode="lines",
            name="5th-95th percentile",
            line={"width": 0.5},
            opacity=0.25,
        )
    )
    fig.update_layout(
        title="Welch spectrum summary across recorded electrodes",
        template="plotly_white",
        xaxis_title="Frequency (Hz)",
        yaxis_title="PSD",
        hovermode="x unified",
    )
    return _write_html(fig, output_path, include_plotlyjs)
