"""Spectral summary and spectrogram figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from hdmea_lfp_viz.plots.traces import select_representative_channels
from hdmea_lfp_viz.style import CB_SAFE_CMAP, add_caption, mmss_formatter, save_figure
from hdmea_lfp_viz.summaries import channel_ids_from_indices


BAND_BOUNDARIES = [0.5, 4, 8, 13, 30, 80, 150]


def plot_psd_grid(summaries: dict, figures_dir: str | Path) -> None:
    """Figure 03: median PSD with IQR."""
    freqs = summaries["freqs"]
    psd = summaries["psd"]
    median = np.nanmedian(psd, axis=0)
    q25, q75 = np.nanpercentile(psd, [25, 75], axis=0)
    keep = freqs > 0

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle("03 LFP Power Spectral Density")
    ax.fill_between(freqs[keep], q25[keep], q75[keep], color="#9ecae1", alpha=0.45, label="25-75% IQR")
    ax.loglog(freqs[keep], median[keep], color="black", lw=2.2, label="Median")
    y_top = np.nanmax(q75[keep])
    for f in BAND_BOUNDARIES:
        ax.axvline(f, color="0.45", ls="--", lw=0.8)
        ax.text(f, y_top, f"{f:g}", rotation=90, va="top", ha="right", fontsize=8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (µV²/Hz)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    add_caption(fig, "Welch PSD per channel, summarized by the channel median and interquartile range; dashed lines mark canonical band boundaries.")
    fig.tight_layout(rect=[0, 0.05, 0.86, 0.95])
    save_figure(fig, Path(figures_dir), "03_psd_grid")


def plot_spectrograms(recording, summaries: dict, figures_dir: str | Path) -> None:
    """Figure 08: Welch spectrograms for representative channels."""
    channels = select_representative_channels(summaries["rms_per_channel"], n=8)
    sf = float(recording.get_sampling_frequency())
    traces = recording.get_traces(channel_ids=channel_ids_from_indices(recording, channels), return_scaled=True).astype(np.float32, copy=False)
    nperseg = int(round(2.0 * sf))
    noverlap = nperseg // 2

    specs = []
    times = None
    freqs = None
    for i in range(channels.size):
        f, t, sxx = spectrogram(
            traces[:, i],
            fs=sf,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
            mode="psd",
        )
        keep = f <= 150
        freqs = f[keep]
        times = t
        specs.append(10.0 * np.log10(sxx[keep] + np.finfo(np.float32).eps))
    vmin, vmax = np.nanpercentile(np.concatenate([s.ravel() for s in specs]), [2, 98])

    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True, sharey=True)
    fig.suptitle("08 Representative Channel Spectrograms")
    last_im = None
    for ax, ch, spec in zip(axes.flat, channels, specs):
        last_im = ax.pcolormesh(times, freqs, spec, shading="auto", cmap=CB_SAFE_CMAP, vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_title(f"Channel {ch}")
        ax.set_ylabel("Frequency (Hz)")
        ax.xaxis.set_major_formatter(mmss_formatter())
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (mm:ss)")
    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), pad=0.01)
    cbar.set_label("Power (dB µV²/Hz)")
    add_caption(fig, "Two-second Welch spectrograms with 50% overlap, restricted to 0-150 Hz and shown on a shared log-power color scale.")
    fig.tight_layout(rect=[0, 0.05, 0.93, 0.95])
    save_figure(fig, Path(figures_dir), "08_spectrograms")
