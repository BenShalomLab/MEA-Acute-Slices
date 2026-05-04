"""Correlation matrix figure with hierarchical clustering."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform

from hdmea_lfp_viz.style import DIVERGING_CMAP, add_caption, save_figure


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
        width_ratios=[1.0, 8.0],
        height_ratios=[1.0, 8.0],
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

    dendrogram(z, ax=ax_top, no_labels=True, color_threshold=0, above_threshold_color="0.25")
    ax_top.axis("off")
    dendrogram(z, ax=ax_left, orientation="left", no_labels=True, color_threshold=0, above_threshold_color="0.25")
    ax_left.axis("off")

    im = ax_mat.imshow(clustered, cmap=DIVERGING_CMAP, norm=TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0), rasterized=True)
    ax_mat.set_xlabel("Clustered channels")
    ax_mat.set_ylabel("Clustered channels")
    cax = fig.add_axes([0.89, 0.17, 0.02, 0.64])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Pearson r")
    add_caption(fig, "Channels are reordered by Ward clustering on 1 - |correlation| to reveal correlated spatial or noise-related channel groups.")
    save_figure(fig, Path(figures_dir), "05_correlation_matrix")
