"""Spectral and LFP band-power analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
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
) -> pd.DataFrame:
    """Compute Welch band power in sliding windows."""
    bands = bands or DEFAULT_LFP_BANDS
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

    for win_start in range(start_frame, last_start + 1, step_frames):
        win_end = win_start + window_frames
        traces = get_traces_safe(recording, win_start, win_end, channel_ids, return_scaled=return_scaled)
        freqs, psd = signal.welch(traces, fs=fs, nperseg=nperseg, axis=0)
        center_time_sec = (win_start + win_end) / (2 * fs)
        for channel_idx, (electrode_id, channel_id) in enumerate(zip(electrode_ids_resolved, channel_ids)):
            for band_name, (freq_min, freq_max) in bands.items():
                mask = (freqs >= freq_min) & (freqs < freq_max)
                if not np.any(mask):
                    continue
                power = float(integrate(psd[mask, channel_idx], freqs[mask]))
                rows.append(
                    {
                        "electrode_id": int(electrode_id),
                        "channel_id": channel_id,
                        "time_sec": float(center_time_sec),
                        "window_start_sec": win_start / fs,
                        "window_end_sec": win_end / fs,
                        "band": band_name,
                        "freq_min_hz": float(freq_min),
                        "freq_max_hz": float(freq_max),
                        "power": power,
                        "power_db": float(10 * np.log10(power + np.finfo(float).eps)),
                    }
                )
    return pd.DataFrame(rows)


def compute_welch_spectrum_summary(
    recording,
    channel_ids=None,
    duration_sec=10,
    max_freq_hz=None,
    welch_segment_sec=2,
    return_scaled=True,
) -> dict[str, np.ndarray]:
    """Compute all-channel Welch summary statistics for a recording segment."""
    fs = float(recording.get_sampling_frequency())
    end_frame = min(recording.get_num_samples(), int(round(float(duration_sec) * fs)))
    if end_frame <= 0:
        raise ValueError("duration_sec must select at least one sample.")
    if channel_ids is None:
        channel_ids = list(recording.get_channel_ids())
    traces = get_traces_safe(recording, 0, end_frame, channel_ids, return_scaled=return_scaled)
    nperseg = max(1, min(int(round(float(welch_segment_sec) * fs)), traces.shape[0]))
    freqs, psd = signal.welch(traces, fs=fs, nperseg=nperseg, axis=0)
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


def compute_trace_preview(
    recordings: dict[str, object],
    electrode_table=None,
    electrode_ids=None,
    start_sec=0,
    duration_sec=10,
    max_points=20000,
    return_scaled=True,
) -> dict[str, np.ndarray]:
    """Return decimated traces for quick raw/LFP/spike visualization."""
    if not recordings:
        raise ValueError("At least one recording is required.")
    first = next(iter(recordings.values()))
    fs = float(first.get_sampling_frequency())
    electrode_ids_resolved, channel_ids = _resolve_channels(first, electrode_table, electrode_ids)
    start_frame = max(0, int(round(float(start_sec) * fs)))
    end_frame = min(first.get_num_samples(), start_frame + int(round(float(duration_sec) * fs)))
    if end_frame <= start_frame:
        raise ValueError("Requested trace preview window is empty.")

    step = max(1, int(np.ceil((end_frame - start_frame) / int(max_points))))
    frames = np.arange(start_frame, end_frame, step)
    preview = {
        "time_sec": frames / fs,
        "electrode_ids": np.asarray(electrode_ids_resolved),
        "channel_ids": np.asarray(channel_ids, dtype=object),
        "sample_step": np.asarray(step),
    }
    for name, recording in recordings.items():
        traces = get_traces_safe(recording, start_frame, end_frame, channel_ids, return_scaled=return_scaled)
        preview[name] = traces[::step]
    return preview
