"""Correlation matrix figure with hierarchical clustering."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform

from hdmea_lfp_viz.style import DIVERGING_CMAP, add_caption, save_figure

CORRELATION_LOG_TICKS = [-1.0, -0.3, -0.1, -0.03, 0.0, 0.03, 0.1, 0.3, 1.0]


def _style_dendrogram(ax) -> None:
    for collection in ax.collections:
        collection.set_color("0.05")
        collection.set_linewidth(0.9)
    ax.axis("off")


def plot_correlation_matrix(summaries: dict, figures_dir: str | Path) -> None:
    """Figure 05: clustered channel correlation matrix."""
    corr = np.asarray(summaries["corr_matrix"])
    distance = 1.0 - np.abs(corr)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    z = linkage(condensed, method="ward")
    order = leaves_list(z)
    clustered = corr[np.ix_(order, order)]

    fig = plt.figure(figsize=(10.5, 10))
    fig.suptitle("05 Channel Correlation Matrix")
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.25, 8.0],
        height_ratios=[1.25, 8.0],
        left=0.07,
        right=0.86,
        bottom=0.09,
        top=0.91,
        hspace=0.02,
        wspace=0.02,
    )
    ax_blank = fig.add_subplot(gs[0, 0])
    ax_blank.axis("off")
    ax_top = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_mat = fig.add_subplot(gs[1, 1])

    dendrogram(z, ax=ax_top, no_labels=True, color_threshold=0, above_threshold_color="0.05", link_color_func=lambda _: "0.05")
    _style_dendrogram(ax_top)
    dendrogram(
        z,
        ax=ax_left,
        orientation="left",
        no_labels=True,
        color_threshold=0,
        above_threshold_color="0.05",
        link_color_func=lambda _: "0.05",
    )
    _style_dendrogram(ax_left)

    im = ax_mat.imshow(
        clustered,
        cmap=DIVERGING_CMAP,
        norm=SymLogNorm(linthresh=0.02, vmin=-1.0, vmax=1.0),
        rasterized=True,
    )
    ax_mat.set_xlabel("Clustered channels")
    ax_mat.set_ylabel("Clustered channels")
    cax = fig.add_axes([0.89, 0.17, 0.02, 0.64])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks(CORRELATION_LOG_TICKS)
    cbar.set_label("Pearson r")
    add_caption(fig, "Channels are reordered by Ward clustering on 1 - |r|; matrix color uses signed-log-scaled Pearson r.")
    save_figure(fig, Path(figures_dir), "05_correlation_matrix")
