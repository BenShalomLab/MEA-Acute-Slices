"""Authoritative ground truth extraction, visualization, and dashboard validator.

Provides lightweight, fast extraction from Maxwell raw HDF5 files via
SpikeInterface to verify that dashboard downsampling, scaling, and rendering
preserve exact signal fidelity.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration and HDF5 Plugin Management
# -----------------------------------------------------------------------------


def ensure_hdf5_plugin() -> str | None:
    """Ensure HDF5_PLUGIN_PATH is set for Maxwell compression.dll decompression.

    Checks os.environ first; if unset, looks in ~/.acute_slice_mea/paths.json.
    """
    import os

    current = os.environ.get("HDF5_PLUGIN_PATH")
    if current and Path(current).exists():
        return current

    config_path = Path.home() / ".acute_slice_mea" / "paths.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_paths = json.load(f)
            plugin_raw = user_paths.get("hdf5_plugin_path")
            if plugin_raw:
                p = Path(plugin_raw).expanduser()
                if p.exists():
                    os.environ["HDF5_PLUGIN_PATH"] = str(p)
                    return str(p)
        except Exception as exc:
            logger.warning("Could not read hdf5_plugin_path from %s: %s", config_path, exc)
    return None


# -----------------------------------------------------------------------------
# Recording Loader
# -----------------------------------------------------------------------------


def load_recording(
    raw_path: str | Path,
    *,
    well_id: str = "well000",
    rec_name: str | None = None,
    filter_mode: str = "raw",
    apply_cmr: bool = False,
    lfp_low_hz: float = 0.5,
    lfp_high_hz: float = 300.0,
    spike_low_hz: float = 300.0,
    spike_high_hz: float = 3000.0,
):
    """Load a Maxwell recording with optional filtering and common reference.

    Parameters
    ----------
    raw_path : str | Path
        Path to Maxwell .raw.h5 file.
    well_id : str
        Stream / well identifier (default 'well000').
    rec_name : str | None
        Optional sub-recording identifier in multi-recording files.
    filter_mode : str
        'raw' (unsigned-to-signed only), 'lfp' (bandpass 0.5-300 Hz), or
        'spike' (bandpass 300-3000 Hz).
    apply_cmr : bool
        If True, applies median common reference across all channels.

    Returns
    -------
    recording : BaseRecording
        SpikeInterface recording object.
    fs : float
        Sampling frequency in Hz.
    metadata : dict
        Dictionary containing channels, duration, and sampling rate.
    """
    ensure_hdf5_plugin()
    import spikeinterface.extractors as se
    import spikeinterface.preprocessing as spre

    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Recording file not found: {raw_path}")

    recording_raw = se.read_maxwell(str(raw_path), stream_id=str(well_id), rec_name=rec_name)
    signed = spre.unsigned_to_signed(recording_raw)

    mode = filter_mode.lower().strip()
    if mode == "raw":
        rec = signed
    elif mode == "lfp":
        rec = spre.bandpass_filter(
            signed,
            freq_min=lfp_low_hz,
            freq_max=lfp_high_hz,
            margin_ms=10000,
        )
    elif mode == "spike":
        rec = spre.bandpass_filter(
            signed,
            freq_min=spike_low_hz,
            freq_max=spike_high_hz,
        )
    else:
        raise ValueError(f"Unknown filter_mode '{filter_mode}'. Must be 'raw', 'lfp', or 'spike'.")

    if apply_cmr:
        rec = spre.common_reference(rec, reference="global", operator="median")

    fs = float(rec.get_sampling_frequency())
    num_frames = int(rec.get_num_frames())
    duration_sec = num_frames / fs if fs > 0 else 0.0

    metadata = {
        "file_name": raw_path.name,
        "well_id": str(well_id),
        "fs": fs,
        "num_channels": int(rec.get_num_channels()),
        "num_frames": num_frames,
        "duration_sec": duration_sec,
        "filter_mode": mode,
        "cmr_applied": apply_cmr,
        "channel_ids": list(rec.get_channel_ids()),
    }

    return rec, fs, metadata


# -----------------------------------------------------------------------------
# Timestamp to Index Slicing
# -----------------------------------------------------------------------------


def time_to_indices(
    t0: float,
    t1: float,
    fs: float,
    max_frames: int | None = None,
) -> tuple[int, int]:
    """Convert continuous timestamps in seconds to exact sample frame indices.

    Guarantees start_frame <= end_frame, clamps within [0, max_frames], and
    handles floating-point edge cases cleanly.

    Parameters
    ----------
    t0 : float
        Start timestamp in seconds.
    t1 : float
        End timestamp in seconds.
    fs : float
        Sampling frequency in Hz (> 0).
    max_frames : int | None
        Total number of frames in the recording for boundary clamping.

    Returns
    -------
    start_frame : int
    end_frame : int
    """
    if fs <= 0:
        raise ValueError(f"Sampling frequency fs must be positive, got {fs}")

    t_lo = min(float(t0), float(t1))
    t_hi = max(float(t0), float(t1))

    start_frame = max(0, int(round(t_lo * fs)))
    end_frame = max(start_frame, int(round(t_hi * fs)))

    if max_frames is not None:
        mf = max(0, int(max_frames))
        start_frame = min(start_frame, mf)
        end_frame = min(end_frame, mf)

    return start_frame, end_frame


# -----------------------------------------------------------------------------
# Stimulation Bookmark / Condition Block Detector
# -----------------------------------------------------------------------------


def find_stim_blocks(
    recording,
    *,
    stim_channel: str | int = "631",
    height: float = 500.0,
    min_interval_s: float = 0.005,
    gap_threshold_s: float = 4.0,
    prominence: float = 70.0,
) -> list[dict[str, Any]]:
    """Detect stimulation artifacts and group them into condition blocks.

    Provides exact time bookmarks (e.g. 1 Hz, 10 Hz, 20 Hz blocks) to navigate
    directly to active stimulation windows in the dashboard.

    Parameters
    ----------
    recording : BaseRecording
        SpikeInterface recording object.
    stim_channel : str | int
        Active stimulation channel ID (e.g. '631').
    height : float
        Minimum artifact amplitude peak threshold in µV.
    min_interval_s : float
        Minimum seconds between consecutive pulses within a block.
    gap_threshold_s : float
        Gaps longer than this threshold delineate a new condition block.
    prominence : float
        Peak prominence threshold for peak detection.

    Returns
    -------
    blocks : list[dict]
        List of detected condition blocks, each containing 'block', 't_start',
        't_end', 'n_pulses', and 'duration_s'.
    """
    cid_str = str(stim_channel)
    available_ids = set(recording.get_channel_ids())
    if cid_str not in available_ids:
        # Check integer matching
        matched = [c for c in available_ids if str(c) == cid_str or str(int(c)) == cid_str]
        if not matched:
            raise ValueError(
                f"Stimulation channel '{stim_channel}' not found in recording. "
                f"Available channel count: {len(available_ids)}"
            )
        cid_str = matched[0]

    fs = float(recording.get_sampling_frequency())
    num_frames = int(recording.get_num_frames())

    stim_trace = _safe_get_traces(
        recording, start_frame=0, end_frame=num_frames, channel_ids=[cid_str]
    ).squeeze()

    stim_centered = stim_trace - np.median(stim_trace)

    # Detect negative peaks (standard stimulation artifact)
    peak_idxs, _ = find_peaks(
        -stim_centered,
        height=height,
        distance=max(1, int(round(min_interval_s * fs))),
        prominence=prominence,
    )

    # If no negative peaks found, test positive polarity
    if len(peak_idxs) == 0:
        peak_idxs, _ = find_peaks(
            stim_centered,
            height=height,
            distance=max(1, int(round(min_interval_s * fs))),
            prominence=prominence,
        )

    if len(peak_idxs) == 0:
        logger.warning("No stimulation pulses detected on channel %s with height=%.1f", cid_str, height)
        return []

    peak_times_s = peak_idxs / fs
    inter_peak_gaps = np.diff(peak_times_s)
    gap_idxs = np.where(inter_peak_gaps > gap_threshold_s)[0]

    block_starts = np.concatenate([[peak_times_s[0]], peak_times_s[gap_idxs + 1]])
    block_ends = np.concatenate([peak_times_s[gap_idxs], [peak_times_s[-1]]])

    blocks = []
    for idx, (t0, t1) in enumerate(zip(block_starts, block_ends)):
        mask = (peak_times_s >= t0) & (peak_times_s <= t1)
        count = int(np.sum(mask))
        blocks.append(
            {
                "block": idx + 1,
                "t_start": float(t0),
                "t_end": float(t1),
                "n_pulses": count,
                "duration_s": float(t1 - t0),
            }
        )

    return blocks


# -----------------------------------------------------------------------------
# Channel & Electrode Resolution
# -----------------------------------------------------------------------------


def resolve_channel_and_electrode_pairs(
    recording,
    *,
    electrodes: Iterable[int | str] | None = None,
    channels: Iterable[str | int] | None = None,
    cache_dir: str | Path | None = None,
) -> list[tuple[int | None, str]]:
    """Map between physical MEA pad electrode_id and active recording channel_id.

    Returns a list of (electrode_id, channel_id) tuples.
    """
    all_cids = list(recording.get_channel_ids())
    cid_set = set(str(c) for c in all_cids)

    eid_to_cid: dict[int, str] = {}
    cid_to_eid: dict[str, int] = {}

    if cache_dir is not None:
        electrodes_csv = Path(cache_dir) / "electrodes.csv"
        if electrodes_csv.exists():
            df = pd.read_csv(electrodes_csv)
            if "electrode_id" in df.columns and "channel_id" in df.columns:
                rec_df = df[df["recorded"].astype(bool)] if "recorded" in df.columns else df
                for eid, cid in zip(rec_df["electrode_id"], rec_df["channel_id"]):
                    if pd.notna(cid):
                        try:
                            cid_clean = str(int(cid))
                        except (ValueError, TypeError):
                            cid_clean = str(cid)
                        eid_to_cid[int(eid)] = cid_clean
                        cid_to_eid[cid_clean] = int(eid)

    resolved: list[tuple[int | None, str]] = []

    # Priority 1: User specified channel_ids directly
    if channels is not None:
        for ch in channels:
            ch_str = str(ch)
            if ch_str not in cid_set:
                matched = [c for c in all_cids if str(c) == ch_str or str(int(c)) == ch_str]
                if not matched:
                    logger.warning("Channel %s not present in recording.", ch)
                    continue
                ch_str = matched[0]
            eid = cid_to_eid.get(ch_str)
            resolved.append((eid, ch_str))

    # Priority 2: User specified physical electrode_ids
    if electrodes is not None:
        for e in electrodes:
            eid_int = int(e)
            cid = eid_to_cid.get(eid_int)
            if cid is not None and cid in cid_set:
                resolved.append((eid_int, cid))
            elif str(eid_int) in cid_set:
                # Direct match when channel_id equals electrode_id number
                resolved.append((eid_int, str(eid_int)))
            else:
                logger.warning("Electrode %d not routed to an active recording channel.", eid_int)

    # Fallback: if neither specified, take first 5 routed channels
    if not resolved:
        for ch in all_cids[:5]:
            eid = cid_to_eid.get(str(ch))
            resolved.append((eid, str(ch)))

    return resolved


# -----------------------------------------------------------------------------
# Data Extraction
# -----------------------------------------------------------------------------


def extract_traces(
    recording,
    *,
    electrodes: Iterable[int | str] | None = None,
    channels: Iterable[str | int] | None = None,
    t0: float | None = None,
    t1: float | None = None,
    cache_dir: str | Path | None = None,
    baseline_subtract: bool = True,
) -> list[dict[str, Any]]:
    """Extract physical voltage traces in µV for specified electrodes/channels.

    Parameters
    ----------
    recording : BaseRecording
        SpikeInterface recording object.
    electrodes : Iterable[int | str] | None
        Physical electrode IDs (0-26399).
    channels : Iterable[str | int] | None
        Active recording channel IDs (0-1023).
    t0 : float | None
        Start time in seconds (defaults to 0.0).
    t1 : float | None
        End time in seconds (defaults to total recording duration).
    cache_dir : str | Path | None
        Optional pipeline cache dir containing electrodes.csv for ID mapping.
    baseline_subtract : bool
        If True, subtracts the median of each trace (zero-centering).

    Returns
    -------
    payloads : list[dict]
        List of extracted trace dicts with keys: 'electrode_id', 'channel_id',
        'time_sec', 'value', 'sampling_rate_hz'.
    """
    pairs = resolve_channel_and_electrode_pairs(
        recording, electrodes=electrodes, channels=channels, cache_dir=cache_dir
    )
    if not pairs:
        return []

    fs = float(recording.get_sampling_frequency())
    num_frames = int(recording.get_num_frames())
    dur_sec = num_frames / fs if fs > 0 else 0.0

    start_sec = 0.0 if t0 is None else max(0.0, float(t0))
    end_sec = dur_sec if t1 is None else min(dur_sec, float(t1))

    start_frame, end_frame = time_to_indices(start_sec, end_sec, fs, num_frames)
    if end_frame <= start_frame:
        return []

    time_sec = (np.arange(start_frame, end_frame, dtype=np.float64)) / fs
    channel_list = [p[1] for p in pairs]

    traces = _safe_get_traces(
        recording, start_frame=start_frame, end_frame=end_frame, channel_ids=channel_list
    )
    arr = np.asarray(traces, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]

    results = []
    for col_idx, (eid, cid) in enumerate(pairs):
        val = np.asarray(arr[:, col_idx], dtype=np.float64)
        if baseline_subtract and len(val) > 0:
            val = val - np.nanmedian(val)
        results.append(
            {
                "electrode_id": eid,
                "channel_id": cid,
                "time_sec": time_sec,
                "value": val,
                "sampling_rate_hz": fs,
            }
        )

    return results


# -----------------------------------------------------------------------------
# Static Matplotlib Visualization & Scale Bars
# -----------------------------------------------------------------------------


def add_scale_bars(
    ax,
    x_size: float,
    y_size: float,
    x_label: str,
    y_label: str,
    *,
    x_unit: str = "ms",
    y_unit: str = "µV",
    color: str = "#1a1916",
    lw: float = 1.5,
    fontsize: float = 8.0,
    pad_x_pct: float = 0.04,
    pad_y_pct: float = 0.06,
) -> None:
    """Draw clean L-shaped scale bars in the corner of ax in data coordinates."""
    ax.autoscale(False)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]

    x0 = xlim[0] + pad_x_pct * x_range
    y0 = ylim[0] + pad_y_pct * y_range

    # Horizontal time line
    ax.plot([x0, x0 + x_size], [y0, y0], color=color, lw=lw, clip_on=False, solid_capstyle="butt")
    # Vertical amplitude line
    ax.plot([x0, x0], [y0, y0 + y_size], color=color, lw=lw, clip_on=False, solid_capstyle="butt")

    # Time label (under horizontal bar)
    ax.text(
        x0 + x_size / 2.0,
        y0 - 0.03 * y_range,
        f"{x_label} {x_unit}".strip(),
        ha="center",
        va="top",
        fontsize=fontsize,
        color=color,
        fontweight="medium",
    )
    # Amplitude label (left of vertical bar)
    ax.text(
        x0 - 0.015 * x_range,
        y0 + y_size / 2.0,
        f"{y_label} {y_unit}".strip(),
        ha="right",
        va="center",
        fontsize=fontsize,
        color=color,
        rotation=90,
        fontweight="medium",
    )


def plot_ground_truth(
    payloads: Sequence[dict[str, Any]],
    *,
    spacing_uv: float = 250.0,
    gain: float = 1.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    scalebars: bool = True,
    scalebar_ms: float = 5.0,
    scalebar_uv: float = 100.0,
    trace_color: str = "#0a7d7f",
    title: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Render static, publication-ready ground truth traces with L-scalebars.

    Parameters
    ----------
    payloads : Sequence[dict]
        List of trace dictionaries produced by extract_traces.
    spacing_uv : float
        Vertical offset between adjacent stacked traces in µV (default 250).
    gain : float
        Visual display gain multiplier (matches dashboard gain control).
    xlim : tuple[float, float] | None
        Optional (t_start, t_end) limits in seconds.
    ylim : tuple[float, float] | None
        Optional (y_min, y_max) limits in µV.
    scalebars : bool
        If True, renders clean L-shaped scale bars and hides standard axis spines.
    scalebar_ms : float
        Time bar width in milliseconds.
    scalebar_uv : float
        Voltage bar height in microvolts.
    trace_color : str
        Hex or name color of plotted traces.
    title : str | None
        Optional figure title.
    save_path : str | Path | None
        Optional output path to save figure (PNG or PDF).
    figsize : tuple[float, float]
        Figure dimensions (width, height) in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if not payloads:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No traces available to plot", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=figsize)
    g = max(0.01, float(gain))
    effective_spacing = float(spacing_uv) / g

    y_tick_positions = []
    y_tick_labels = []

    for idx, p in enumerate(payloads):
        offset = -idx * effective_spacing
        time_sec = p["time_sec"]
        val = p["value"] * g + offset
        ax.plot(time_sec, val, color=trace_color, lw=1.0)

        eid = p.get("electrode_id")
        cid = p.get("channel_id")
        if eid is not None and cid is not None:
            label = f"E{eid} · ch{cid}"
        elif eid is not None:
            label = f"E{eid}"
        else:
            label = f"ch{cid}"

        y_tick_positions.append(offset)
        y_tick_labels.append(label)

    # Set ranges
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        all_times = [p["time_sec"] for p in payloads if len(p["time_sec"]) > 0]
        if all_times:
            t_min = min(t[0] for t in all_times)
            t_max = max(t[-1] for t in all_times)
            ax.set_xlim(t_min, t_max)

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        pad = effective_spacing * 0.6
        min_y = -len(payloads) * effective_spacing
        ax.set_ylim(min_y - pad, pad)

    if scalebars:
        # Hide top/right spines and standard ticks for electrophys style
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks(y_tick_positions)
        ax.set_yticklabels(y_tick_labels, fontsize=9, fontfamily="monospace")

        # Convert scalebar_ms to seconds for x coordinate
        x_size_sec = scalebar_ms / 1000.0
        y_size_val = scalebar_uv * g

        x_label = f"{scalebar_ms:.0f}" if scalebar_ms.is_integer() else f"{scalebar_ms:.1f}"
        y_label = f"{scalebar_uv:.0f}" if scalebar_uv.is_integer() else f"{scalebar_uv:.1f}"
        add_scale_bars(
            ax,
            x_size=x_size_sec,
            y_size=y_size_val,
            x_label=x_label,
            y_label=y_label,
            x_unit="ms",
            y_unit="µV",
        )
    else:
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Voltage (µV)", fontsize=10)
        ax.set_yticks(y_tick_positions)
        ax.set_yticklabels(y_tick_labels, fontsize=9, fontfamily="monospace")
        ax.grid(True, linestyle=":", alpha=0.5)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        logger.info("Saved ground truth figure to %s", out)

    return fig


# -----------------------------------------------------------------------------
# Optional Dashboard Validation / Comparison
# -----------------------------------------------------------------------------


def compare_with_dashboard(
    raw_path: str | Path,
    cache_dir: str | Path,
    *,
    electrodes: Sequence[int],
    t0: float,
    t1: float,
    signal: str = "lfp",
    gain: float = 1.0,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Compare raw ground truth against traces rendered by the dashboard.

    Plots a 3-panel validation figure:
    1. Ground Truth (Direct SpikeInterface uncompressed slice).
    2. Dashboard Rendered Trace (What WellData / data_loader serves).
    3. Residual Difference (Ground Truth - Dashboard) in µV.

    Parameters
    ----------
    raw_path : str | Path
        Path to raw Maxwell .raw.h5 file.
    cache_dir : str | Path
        Path to pipeline cache directory for the recording.
    electrodes : Sequence[int]
        Electrode IDs to compare (e.g. [821]).
    t0 : float
        Window start time in seconds.
    t1 : float
        Window end time in seconds.
    signal : str
        'lfp' or 'raw'.
    gain : float
        Display gain.
    save_path : str | Path | None
        Optional save path for validation plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    metrics : dict
        Calculated Max Absolute Error, RMS Error, and Pearson Correlation.
    """
    from acute_slice_mea.viewer.data_loader import WellData

    cache_dir = Path(cache_dir)
    raw_path = Path(raw_path)

    # 1. Load Ground Truth
    filter_mode = "lfp" if signal == "lfp" else "raw"
    apply_cmr = signal == "lfp"  # Dashboard LFP applies common median reference
    rec, fs, _ = load_recording(
        raw_path,
        filter_mode=filter_mode,
        apply_cmr=apply_cmr,
    )
    gt_payloads = extract_traces(
        rec, electrodes=electrodes, t0=t0, t1=t1, cache_dir=cache_dir, baseline_subtract=True
    )
    if not gt_payloads:
        raise ValueError(f"Could not extract ground truth for electrodes {electrodes} at [{t0}, {t1}]")

    # 2. Load Dashboard Traces via WellData
    wd = WellData.load(cache_dir)
    dash_payloads = wd.traces_for_window(
        electrodes, signal=signal, t0=t0, t1=t1, full_resolution=True
    )
    if not dash_payloads:
        # Fall back to regular traces_for
        dash_payloads = wd.traces_for(electrodes, signal=signal, t0=t0, t1=t1, decimate=False)

    if not dash_payloads:
        raise ValueError(f"Could not load dashboard data from {cache_dir} for electrodes {electrodes}")

    gt = gt_payloads[0]
    dash = dash_payloads[0]

    gt_t = gt["time_sec"]
    gt_v = gt["value"]

    dash_t = dash["time_sec"]
    dash_v = dash["value"]
    if signal == "raw" and len(dash_v) > 0:
        dash_v = dash_v - np.nanmedian(dash_v)

    # Align arrays for residual calculation
    if len(dash_t) != len(gt_t) or not np.allclose(dash_t, gt_t, atol=1e-5):
        # Interpolate dashboard values onto ground truth time grid
        dash_interp = np.interp(gt_t, dash_t, dash_v)
        residual = gt_v - dash_interp
    else:
        residual = gt_v - dash_v

    max_err = float(np.max(np.abs(residual)))
    rms_err = float(np.sqrt(np.mean(residual**2)))
    corr = float(np.corrcoef(gt_v, gt_v - residual)[0, 1]) if len(gt_v) > 1 else 1.0

    metrics = {
        "electrode_id": int(electrodes[0]),
        "max_absolute_error_uv": max_err,
        "rms_error_uv": rms_err,
        "pearson_correlation": corr,
        "gt_num_samples": len(gt_v),
        "dash_num_samples": len(dash_v),
    }

    # 3. Create 3-Panel Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    eid = electrodes[0]
    cid = gt.get("channel_id")
    title_str = f"Electrode E{eid} (ch{cid}) — Ground Truth vs. Dashboard Validation [{t0:.2f}s – {t1:.2f}s]"

    # Panel 1: Ground Truth
    axes[0].plot(gt_t, gt_v, color="#0a7d7f", lw=1.2, label="Ground Truth (Raw SpikeInterface)")
    axes[0].set_ylabel("GT (µV)", fontsize=9)
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    axes[0].set_title(title_str, fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.4)

    # Panel 2: Dashboard Rendered Trace
    axes[1].plot(dash_t, dash_v, color="#d95f02", lw=1.2, label="Dashboard Rendered (WellData)")
    axes[1].set_ylabel("Dashboard (µV)", fontsize=9)
    axes[1].legend(loc="upper right", frameon=False, fontsize=8)
    axes[1].grid(True, linestyle=":", alpha=0.4)

    # Panel 3: Residual Error
    axes[2].plot(gt_t, residual, color="#7570b3", lw=1.0, label="Residual (GT − Dashboard)")
    axes[2].axhline(0, color="gray", lw=0.8, linestyle="--")
    axes[2].set_ylabel("Δ Error (µV)", fontsize=9)
    axes[2].set_xlabel("Time (s)", fontsize=10)
    axes[2].legend(loc="upper right", frameon=False, fontsize=8)
    axes[2].grid(True, linestyle=":", alpha=0.4)

    metrics_text = (
        f"Max |Error|: {max_err:.3f} µV   |   RMS Error: {rms_err:.3f} µV   |   Correlation: r = {corr:.4f}"
    )
    axes[2].text(
        0.02,
        0.12,
        metrics_text,
        transform=axes[2].transAxes,
        fontsize=8.5,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f3f1ea", ec="#cfcabc", lw=0.8),
    )

    plt.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        logger.info("Saved comparison figure to %s", out)

    return fig, metrics


# -----------------------------------------------------------------------------
# Internal Helpers
# -----------------------------------------------------------------------------


def _safe_get_traces(recording, start_frame: int, end_frame: int, channel_ids: list[str]) -> np.ndarray:
    """Read traces with safe kwargs fallback across SpikeInterface versions."""
    kwargs = {
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "channel_ids": channel_ids,
    }
    try:
        return recording.get_traces(**kwargs, return_scaled=True)
    except (TypeError, ValueError):
        try:
            return recording.get_traces(**kwargs, return_in_uV=True)
        except TypeError:
            return recording.get_traces(**kwargs)
