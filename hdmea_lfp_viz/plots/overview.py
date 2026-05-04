"""Overview, channel quality, and combined summary figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from hdmea_lfp_viz.style import (
    CB_SAFE_CMAP,
    SEQUENTIAL_CMAP,
    add_caption,
    mmss_formatter,
    save_figure,
    set_equal_spatial_axes,
)


def plot_overview_heatmap(summaries: dict, figures_dir: str | Path) -> None:
    """Figure 01: channel x time RMS heatmap sorted by total RMS."""
    rms_timebins = np.asarray(summaries["rms_timebins"])
    total = np.nanmean(rms_timebins, axis=1)
    order = np.argsort(total)[::-1]
    sorted_bins = rms_timebins[order]
    n_bins = sorted_bins.shape[1]

    fig, ax = plt.subplots(figsize=(14.5, 8.2))
    fig.suptitle("01 LFP RMS Overview")
    extent = [0, n_bins, sorted_bins.shape[0], 0]
    im = ax.imshow(sorted_bins, cmap=SEQUENTIAL_CMAP, aspect=max(n_bins / max(sorted_bins.shape[0], 1) / 1.8, 0.03), extent=extent)
    ax.set_xlabel("Time (mm:ss)")
    ax.set_ylabel("Channels sorted by total RMS")
    ax.xaxis.set_major_formatter(mmss_formatter())
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("RMS (µV)")
    add_caption(fig, "One-second RMS bins reveal recording-wide drifts, bursts, and spatially broad activity changes.")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    save_figure(fig, Path(figures_dir), "01_overview_heatmap")


def plot_channel_quality(summaries: dict, locations: np.ndarray, figures_dir: str | Path) -> None:
    """Figure 02: RMS histogram and spatial RMS map."""
    rms = np.asarray(summaries["rms_per_channel"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle("02 Channel Quality by RMS")

    axes[0].hist(rms[np.isfinite(rms)], bins=50, color="#4c78a8", edgecolor="white")
    med = float(np.nanmedian(rms))
    axes[0].axvline(med, color="black", ls="--", lw=1.5, label=f"median {med:.2f} µV")
    axes[0].set_xlabel("RMS (µV)")
    axes[0].set_ylabel("Channel count")
    axes[0].legend(loc="upper right", frameon=False)

    sc = axes[1].scatter(locations[:, 0], locations[:, 1], c=rms, s=14, cmap=SEQUENTIAL_CMAP, linewidths=0)
    axes[1].set_title("Spatial RMS")
    set_equal_spatial_axes(axes[1], locations)
    cbar = fig.colorbar(sc, ax=axes[1], pad=0.01)
    cbar.set_label("RMS (µV)")

    add_caption(fig, "High-RMS channels may indicate strong signal or noise; spatial clustering helps separate biological activity from artifacts.")
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    save_figure(fig, Path(figures_dir), "02_channel_quality")


def plot_summary_panel(summaries: dict, locations: np.ndarray, figures_dir: str | Path) -> None:
    """Figure 10: single at-a-glance summary panel."""
    import matplotlib.gridspec as gridspec

    rms_timebins = np.asarray(summaries["rms_timebins"])
    order = np.argsort(np.nanmean(rms_timebins, axis=1))[::-1]
    freqs = summaries["freqs"]
    psd = summaries["psd"]
    median = np.nanmedian(psd, axis=0)
    pca = summaries["pca"]
    band_power = summaries["band_power"]

    fig = plt.figure(figsize=(15.5, 11))
    fig.suptitle("10 LFP Recording Summary")
    gs = gridspec.GridSpec(
        4,
        6,
        figure=fig,
        height_ratios=[1.4, 1.0, 0.95, 0.95],
        left=0.06,
        right=0.88,
        bottom=0.10,
        top=0.92,
        hspace=0.85,
        wspace=0.70,
    )

    ax_over = fig.add_subplot(gs[0, :])
    im = ax_over.imshow(rms_timebins[order], cmap=SEQUENTIAL_CMAP, aspect=max(rms_timebins.shape[1] / max(rms_timebins.shape[0], 1) / 2.2, 0.03))
    ax_over.set_title("One-second RMS overview")
    ax_over.set_xlabel("Time (mm:ss)")
    ax_over.set_ylabel("Channels by RMS")
    ax_over.xaxis.set_major_formatter(mmss_formatter())
    cax_over = fig.add_axes([0.90, 0.735, 0.014, 0.17])
    cbar = fig.colorbar(im, cax=cax_over)
    cbar.set_label("RMS (µV)")

    ax_psd = fig.add_subplot(gs[1:3, :3])
    ax_psd.loglog(freqs[1:], median[1:], color="black", lw=2)
    ax_psd.set_title("Median PSD")
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("PSD (µV²/Hz)")
    ax_psd.grid(True, which="both", alpha=0.2)

    ax_pc = fig.add_subplot(gs[1:3, 3:])
    sc = ax_pc.scatter(locations[:, 0], locations[:, 1], c=pca["components"][0], s=14, cmap="RdBu_r", linewidths=0)
    ax_pc.set_title(f"PC1 spatial loading ({pca['explained_variance_ratio'][0] * 100:.1f}% var.)")
    set_equal_spatial_axes(ax_pc, locations)
    cax_pc = fig.add_axes([0.90, 0.395, 0.014, 0.23])
    cbar = fig.colorbar(sc, cax=cax_pc)
    cbar.set_label("Loading")

    for i, (name, values) in enumerate(band_power.items()):
        ax = fig.add_subplot(gs[3, i])
        sc = ax.scatter(locations[:, 0], locations[:, 1], c=values, s=8, cmap=CB_SAFE_CMAP, linewidths=0)
        ax.set_title(name.replace("_", " "))
        set_equal_spatial_axes(ax, locations)
        ax.set_xticks([])
        ax.set_yticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.04)
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label("µV²", fontsize=8)
        cb.ax.tick_params(labelsize=8)

    add_caption(fig, "Combined view of temporal RMS structure, spectral content, dominant spatial mode, and band-limited spatial power.")
    save_figure(fig, Path(figures_dir), "10_summary_panel")
