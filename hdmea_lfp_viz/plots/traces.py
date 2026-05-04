"""Representative trace figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hdmea_lfp_viz.style import add_caption, mmss, save_figure
from hdmea_lfp_viz.summaries import channel_ids_from_indices


def select_representative_channels(rms_per_channel: np.ndarray, n: int = 8) -> np.ndarray:
    """Pick high-, mid-, and low-RMS channels for contrast."""
    rms = np.asarray(rms_per_channel)
    valid = np.flatnonzero(np.isfinite(rms))
    if valid.size <= n:
        return valid
    order = valid[np.argsort(rms[valid])]
    picks = []
    picks.extend(order[-3:][::-1].tolist())
    quantiles = np.linspace(0.25, 0.75, 3)
    picks.extend(order[np.clip((quantiles * (order.size - 1)).astype(int), 0, order.size - 1)].tolist())
    picks.extend(order[:2].tolist())
    unique = []
    for ch in picks:
        if ch not in unique:
            unique.append(int(ch))
    for ch in order[::-1]:
        if len(unique) >= n:
            break
        if int(ch) not in unique:
            unique.append(int(ch))
    return np.asarray(unique[:n], dtype=int)


def _recording_traces(recording, start_s: float, duration_s: float, channels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sf = float(recording.get_sampling_frequency())
    start = max(0, int(round(start_s * sf)))
    stop = min(int(recording.get_num_samples()), start + int(round(duration_s * sf)))
    traces = recording.get_traces(
        start_frame=start,
        end_frame=stop,
        channel_ids=channel_ids_from_indices(recording, channels),
        return_scaled=True,
    ).astype(np.float32, copy=False)
    t = np.arange(traces.shape[0], dtype=np.float32) / sf
    return t, traces


def _minmax_decimate(t: np.ndarray, y: np.ndarray, max_points: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    if y.size <= max_points:
        return t, y
    bins = max(1, max_points // 2)
    edges = np.linspace(0, y.size, bins + 1, dtype=int)
    out_t = []
    out_y = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        segment = y[lo:hi]
        mid_t = 0.5 * (t[lo] + t[hi - 1])
        out_t.extend([mid_t, mid_t])
        out_y.extend([float(np.nanmin(segment)), float(np.nanmax(segment))])
    return np.asarray(out_t), np.asarray(out_y)


def plot_representative_traces(recording, summaries: dict, figures_dir: str | Path) -> None:
    """Figure 07: representative channels in early/mid/late 10-second windows."""
    figures_dir = Path(figures_dir)
    channels = select_representative_channels(summaries["rms_per_channel"], n=8)
    sf = float(recording.get_sampling_frequency())
    duration_s = recording.get_num_samples() / sf
    windows = [0.0, max(0.0, duration_s / 2.0 - 5.0), max(0.0, duration_s - 10.0)]
    titles = ["Early", "Mid", "Late"]

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    fig.suptitle("07 Representative LFP Traces")
    for ax, start_s, title in zip(axes, windows, titles):
        t, traces = _recording_traces(recording, start_s, 10.0, channels)
        scale = float(np.nanpercentile(np.abs(traces), 95))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        offsets = np.arange(channels.size)[::-1] * scale * 3.0
        for idx, ch in enumerate(channels):
            dt, dy = _minmax_decimate(t, traces[:, idx], max_points=5000)
            ax.plot(dt, dy + offsets[idx], lw=0.7, rasterized=dy.size > 5000)
            ax.text(10.05, offsets[idx], f"ch {ch}", va="center", fontsize=8)
        ax.set_title(f"{title} window starting {mmss(start_s)}")
        ax.set_xlim(0, 10)
        ax.set_yticks([])
        ax.set_ylabel("Offset traces")
        bar_x = 0.4
        bar_y = offsets[-1] - scale * 1.5
        ax.plot([bar_x, bar_x], [bar_y, bar_y + scale], color="black", lw=1.2)
        ax.text(bar_x + 0.12, bar_y + scale / 2, f"{scale:.1f} µV", va="center", fontsize=9)
    axes[-1].set_xlabel("Time within window (s)")
    add_caption(
        fig,
        "Eight channels selected from high, mid, and low RMS ranges; traces are vertically offset and peak-preserving downsampled for display.",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    save_figure(fig, figures_dir, "07_representative_traces")
