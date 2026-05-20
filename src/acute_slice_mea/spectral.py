"""Spectral and LFP band-power analysis."""

from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs
from scipy import signal

from acute_slice_mea.electrodes import recorded_electrode_channels

DEFAULT_LFP_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "low_gamma": (30, 80),
    "high_gamma": (80, 150),
}


def get_traces_safe(recording, start_frame, end_frame, channel_ids=None, return_scaled=True):
    """Call SpikeInterface-style get_traces across versions."""
    kwargs = {
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "channel_ids": channel_ids,
    }
    try:
        return recording.get_traces(**kwargs, return_scaled=return_scaled)
    except TypeError:
        return recording.get_traces(**kwargs)


def _all_recording_channels(recording):
    channel_ids = list(recording.get_channel_ids())
    return list(range(len(channel_ids))), channel_ids


def _resolve_channels(recording, electrode_table=None, electrode_ids=None):
    if electrode_table is None:
        if electrode_ids is not None:
            channel_ids = list(recording.get_channel_ids())
            selected = [int(eid) for eid in electrode_ids]
            return selected, [channel_ids[eid] for eid in selected]
        return _all_recording_channels(recording)
    return recorded_electrode_channels(electrode_table, electrode_ids)


def _validate_parallel_options(n_jobs, channel_chunk_size):
    n_jobs = int(n_jobs)
    if n_jobs == 0:
        raise ValueError("n_jobs must be non-zero. Use 1 for serial or -1 for all available cores.")
    if channel_chunk_size is not None:
        channel_chunk_size = int(channel_chunk_size)
        if channel_chunk_size <= 0:
            raise ValueError("channel_chunk_size must be positive when provided.")
    return n_jobs, channel_chunk_size


def _resolve_channel_chunk_size(num_channels, n_jobs, channel_chunk_size):
    if channel_chunk_size is not None:
        return channel_chunk_size
    if n_jobs == 1:
        return max(1, num_channels)
    worker_count = max(1, effective_n_jobs(n_jobs))
    return max(1, ceil(num_channels / worker_count))


def _channel_chunks(electrode_ids, channel_ids, n_jobs, channel_chunk_size):
    chunk_size = _resolve_channel_chunk_size(len(channel_ids), n_jobs, channel_chunk_size)
    chunks = []
    for start in range(0, len(channel_ids), chunk_size):
        end = start + chunk_size
        chunks.append((list(electrode_ids[start:end]), list(channel_ids[start:end])))
    return chunks


def _band_masks(freqs, bands):
    masks = []
    for band_name, (freq_min, freq_max) in bands.items():
        mask = (freqs >= freq_min) & (freqs < freq_max)
        if np.any(mask):
            masks.append((band_name, float(freq_min), float(freq_max), mask))
    return masks


def _in_notebook():
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


def _progress_iter(iterable, *, total=None, desc=None):
    if _in_notebook():
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    return tqdm(iterable, total=total, desc=desc)


def compute_lfp_band_power_over_time(
    recording,
    electrode_table=None,
    electrode_ids=None,
    start_sec=0,
    end_sec=None,
    window_sec=10,
    step_sec=5,
    bands=None,
    welch_segment_sec=2,
    return_scaled=True,
    n_jobs=1,
    channel_chunk_size=None,
    progress=False,
) -> pd.DataFrame:
    """Compute Welch band power in sliding windows."""
    bands = bands or DEFAULT_LFP_BANDS
    n_jobs, channel_chunk_size = _validate_parallel_options(n_jobs, channel_chunk_size)
    fs = float(recording.get_sampling_frequency())
    total_sec = recording.get_num_samples() / fs
    start_sec = max(0.0, float(start_sec))
    end_sec = total_sec if end_sec is None else min(float(end_sec), total_sec)
    if end_sec <= start_sec:
        raise ValueError("Requested LFP analysis interval is empty.")

    window_frames = int(round(float(window_sec) * fs))
    step_frames = int(round(float(step_sec) * fs))
    if window_frames <= 0 or step_frames <= 0:
        raise ValueError("window_sec and step_sec must be positive.")

    electrode_ids_resolved, channel_ids = _resolve_channels(recording, electrode_table, electrode_ids)
    start_frame = int(round(start_sec * fs))
    end_frame = int(round(end_sec * fs))
    nperseg = max(1, min(int(round(float(welch_segment_sec) * fs)), window_frames))
    if hasattr(np, "trapezoid"):
        integrate = np.trapezoid
    else:
        integrate = np.trapz
    rows = []
    chunks = _channel_chunks(electrode_ids_resolved, channel_ids, n_jobs, channel_chunk_size)

    last_start = end_frame - window_frames
    if last_start < start_frame:
        return pd.DataFrame(
            columns=[
                "electrode_id",
                "channel_id",
                "time_sec",
                "window_start_sec",
                "window_end_sec",
                "band",
                "freq_min_hz",
                "freq_max_hz",
                "power",
                "power_db",
            ]
        )

    def compute_window_chunk(win_start, chunk_electrode_ids, chunk_channel_ids):
        win_end = win_start + window_frames
        traces = get_traces_safe(recording, win_start, win_end, chunk_channel_ids, return_scaled=return_scaled)
        freqs, psd = signal.welch(traces, fs=fs, nperseg=nperseg, axis=0)
        masks = _band_masks(freqs, bands)
        center_time_sec = (win_start + win_end) / (2 * fs)
        chunk_rows = []
        for channel_idx, (electrode_id, channel_id) in enumerate(zip(chunk_electrode_ids, chunk_channel_ids)):
            for band_name, freq_min, freq_max, mask in masks:
                power = float(integrate(psd[mask, channel_idx], freqs[mask]))
                chunk_rows.append(
                    {
                        "electrode_id": int(electrode_id),
                        "channel_id": channel_id,
                        "time_sec": float(center_time_sec),
                        "window_start_sec": win_start / fs,
                        "window_end_sec": win_end / fs,
                        "band": band_name,
                        "freq_min_hz": freq_min,
                        "freq_max_hz": freq_max,
                        "power": power,
                        "power_db": float(10 * np.log10(power + np.finfo(float).eps)),
                    }
                )
        return chunk_rows

    window_starts = range(start_frame, last_start + 1, step_frames)
    if progress:
        total_windows = ((last_start - start_frame) // step_frames) + 1
        window_starts = _progress_iter(window_starts, total=total_windows, desc="LFP band power")

    for win_start in window_starts:
        if n_jobs == 1:
            chunk_results = [
                compute_window_chunk(win_start, chunk_electrode_ids, chunk_channel_ids)
                for chunk_electrode_ids, chunk_channel_ids in chunks
            ]
        else:
            chunk_results = Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(compute_window_chunk)(win_start, chunk_electrode_ids, chunk_channel_ids)
                for chunk_electrode_ids, chunk_channel_ids in chunks
            )
        for chunk_rows in chunk_results:
            rows.extend(chunk_rows)
    return pd.DataFrame(rows)


def compute_welch_spectrum_summary(
    recording,
    channel_ids=None,
    duration_sec=10,
    max_freq_hz=None,
    welch_segment_sec=2,
    return_scaled=True,
    n_jobs=1,
    channel_chunk_size=None,
    progress=False,
) -> dict[str, np.ndarray]:
    """Compute all-channel Welch summary statistics for a recording segment."""
    n_jobs, channel_chunk_size = _validate_parallel_options(n_jobs, channel_chunk_size)
    fs = float(recording.get_sampling_frequency())
    end_frame = min(recording.get_num_samples(), int(round(float(duration_sec) * fs)))
    if end_frame <= 0:
        raise ValueError("duration_sec must select at least one sample.")
    if channel_ids is None:
        channel_ids = list(recording.get_channel_ids())
    channel_ids = list(channel_ids)
    nperseg = max(1, min(int(round(float(welch_segment_sec) * fs)), end_frame))
    chunks = _channel_chunks(range(len(channel_ids)), channel_ids, n_jobs, channel_chunk_size)

    def compute_spectrum_chunk(chunk_channel_ids):
        traces = get_traces_safe(recording, 0, end_frame, chunk_channel_ids, return_scaled=return_scaled)
        return signal.welch(traces, fs=fs, nperseg=nperseg, axis=0)

    chunk_iter = chunks
    if progress:
        chunk_iter = _progress_iter(chunk_iter, total=len(chunks), desc="Welch spectrum")

    if n_jobs == 1:
        chunk_results = [compute_spectrum_chunk(chunk_channel_ids) for _, chunk_channel_ids in chunk_iter]
    else:
        chunk_results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(compute_spectrum_chunk)(chunk_channel_ids) for _, chunk_channel_ids in chunk_iter
        )
    freqs = chunk_results[0][0]
    psd = np.concatenate([chunk_psd for _, chunk_psd in chunk_results], axis=1)
    if max_freq_hz is not None:
        mask = freqs <= float(max_freq_hz)
        freqs = freqs[mask]
        psd = psd[mask]
    return {
        "freq_hz": freqs,
        "mean_psd": np.nanmean(psd, axis=1),
        "median_psd": np.nanmedian(psd, axis=1),
        "p05_psd": np.nanpercentile(psd, 5, axis=1),
        "p95_psd": np.nanpercentile(psd, 95, axis=1),
        "dominant_freq_hz": freqs[np.nanargmax(psd, axis=0)],
        "channel_ids": np.asarray(channel_ids, dtype=object),
    }


def compute_lfp_rms_per_electrode(
    lfp_recording,
    electrode_table,
    *,
    electrode_ids=None,
    duration_sec: float = 10.0,
    return_scaled: bool = True,
) -> dict[int, float]:
    """Per-electrode RMS (µV) over the first ``duration_sec`` of the LFP view.

    Drives the heat coloring in the viewer's channel picker; cheap because the
    LFP is already a downsampled, common-referenced view.
    """
    fs = float(lfp_recording.get_sampling_frequency())
    num_samples = int(lfp_recording.get_num_samples())
    end_frame = min(num_samples, int(round(float(duration_sec) * fs)))
    if end_frame <= 0:
        return {}
    electrode_ids_resolved, channel_ids = recorded_electrode_channels(electrode_table, electrode_ids)
    traces = get_traces_safe(lfp_recording, 0, end_frame, channel_ids, return_scaled=return_scaled)
    arr = np.asarray(traces, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    rms = np.sqrt(np.mean(arr * arr, axis=0))
    return {int(eid): float(value) for eid, value in zip(electrode_ids_resolved, rms)}


def compute_trace_preview(
    recordings: dict[str, object],
    electrode_table=None,
    electrode_ids=None,
    start_sec=0,
    duration_sec=10,
    max_points=20000,
    return_scaled=True,
) -> dict[str, np.ndarray]:
    """Return decimated traces for quick raw/LFP/spike visualization.

    Each recording may have its own sampling rate (e.g. LFP resampled to
    1 kHz while raw/spike stay at the acquisition rate), so frame ranges
    and time grids are computed per-recording.
    """
    if not recordings:
        raise ValueError("At least one recording is required.")
    first = next(iter(recordings.values()))
    electrode_ids_resolved, channel_ids = _resolve_channels(first, electrode_table, electrode_ids)
    preview: dict[str, np.ndarray] = {
        "electrode_ids": np.asarray(electrode_ids_resolved),
        "channel_ids": np.asarray(channel_ids, dtype=object),
    }
    any_nonempty = False
    for name, recording in recordings.items():
        fs = float(recording.get_sampling_frequency())
        start_frame = max(0, int(round(float(start_sec) * fs)))
        end_frame = min(recording.get_num_samples(), start_frame + int(round(float(duration_sec) * fs)))
        if end_frame <= start_frame:
            continue
        any_nonempty = True
        step = max(1, int(np.ceil((end_frame - start_frame) / int(max_points))))
        frames = np.arange(start_frame, end_frame, step)
        traces = get_traces_safe(recording, start_frame, end_frame, channel_ids, return_scaled=return_scaled)
        preview[name] = traces[::step]
        preview[f"time_sec_{name}"] = frames / fs
        preview[f"sample_step_{name}"] = np.asarray(step)
    if not any_nonempty:
        raise ValueError("Requested trace preview window is empty.")
    return preview
