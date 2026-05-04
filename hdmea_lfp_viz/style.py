"""Shared plotting style and formatters."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("cache") / "matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((Path("cache") / "xdg").resolve()))

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


SEQUENTIAL_CMAP = "viridis"
CB_SAFE_CMAP = "cividis"
DIVERGING_CMAP = "RdBu_r"


def apply_style() -> None:
    """Apply publication-oriented Matplotlib defaults."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.titlesize": 12,
            "figure.titlesize": 13,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )


def mmss(seconds: float, decimals: int = 0) -> str:
    """Format seconds as mm:ss or mm:ss.s."""
    sign = "-" if seconds < 0 else ""
    seconds = abs(float(seconds))
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    if decimals:
        width = 3 + decimals
        return f"{sign}{minutes:02d}:{rem:0{width}.{decimals}f}"
    return f"{sign}{minutes:02d}:{int(round(rem)):02d}"


def mmss_formatter(decimals: int = 0) -> FuncFormatter:
    """Return a Matplotlib formatter for seconds on an axis."""
    return FuncFormatter(lambda value, _pos: mmss(value, decimals=decimals))


def add_caption(fig, text: str, *, y: float = 0.01) -> None:
    """Add a consistent bottom caption."""
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=10, color="0.25")


def save_figure(fig, figures_dir: Path, stem: str) -> None:
    """Save a figure as 300-DPI PNG and SVG."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def set_equal_spatial_axes(ax, locations) -> None:
    """Set spatial scatter axes with equal aspect and micrometer labels."""
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    if locations is not None and len(locations):
        x = locations[:, 0]
        y = locations[:, 1]
        pad = max(float(x.max() - x.min()), float(y.max() - y.min()), 1.0) * 0.04
        ax.set_xlim(float(x.min() - pad), float(x.max() + pad))
        ax.set_ylim(float(y.min() - pad), float(y.max() + pad))
