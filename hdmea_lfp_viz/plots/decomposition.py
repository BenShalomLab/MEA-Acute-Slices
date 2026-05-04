"""PCA component figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hdmea_lfp_viz.style import DIVERGING_CMAP, add_caption, mmss_formatter, save_figure, set_equal_spatial_axes


def _stride_for_points(n: int, max_points: int = 5000) -> int:
    return max(1, int(np.ceil(n / max_points)))


def plot_pca_components(recording, summaries: dict, locations: np.ndarray, figures_dir: str | Path) -> None:
    """Figure 06: top 5 PC spatial loadings and time courses."""
    pca = summaries["pca"]
    components = pca["components"]
    time_courses = pca["time_courses"]
    var = pca["explained_variance_ratio"]
    sf = float(recording.get_sampling_frequency())
    top = min(5, components.shape[0])

    fig, axes = plt.subplots(top, 2, figsize=(14, 2.6 * top), gridspec_kw={"width_ratios": [1.0, 2.2]})
    if top == 1:
        axes = np.asarray([axes])
    fig.suptitle("06 PCA Spatial Components and Time Courses")
    for i in range(top):
        ax_map, ax_time = axes[i]
        vmax = float(np.nanmax(np.abs(components[i])))
        sc = ax_map.scatter(
            locations[:, 0],
            locations[:, 1],
            c=components[i],
            s=14,
            cmap=DIVERGING_CMAP,
            vmin=-vmax,
            vmax=vmax,
            linewidths=0,
        )
        ax_map.set_title(f"PC{i + 1} loading")
        set_equal_spatial_axes(ax_map, locations)
        cbar = fig.colorbar(sc, ax=ax_map, pad=0.01)
        cbar.set_label("Loading")

        stride = _stride_for_points(time_courses.shape[1])
        t = np.arange(0, time_courses.shape[1], stride) / sf
        ax_time.plot(t, time_courses[i, ::stride], lw=0.8, rasterized=t.size > 5000)
        ax_time.set_title(f"PC{i + 1} time course ({var[i] * 100:.1f}% variance)")
        ax_time.set_ylabel("Score")
        ax_time.xaxis.set_major_formatter(mmss_formatter())
    axes[-1, 1].set_xlabel("Time (mm:ss)")
    add_caption(fig, "Incremental PCA treats time samples as observations and channels as spatial features; traces are strided for display.")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    save_figure(fig, Path(figures_dir), "06_pca_components")
