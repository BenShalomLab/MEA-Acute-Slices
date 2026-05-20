"""Network-burst detection from LFP traces.

Designed for the trace-viewer dashboard, which displays bursts as amber-shaded
windows on top of stacked LFP traces. The core algorithm is pure NumPy so it
can be unit-tested with synthetic signals; a small wrapper adapts a
SpikeInterface recording for use from the analysis pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import signal as scipy_signal

from acute_slice_mea.electrodes import recorded_electrode_channels
from acute_slice_mea.spectral import get_traces_safe


@dataclass(frozen=True)
class BurstDetectionParams:
    envelope_smooth_ms: float = 50.0
    threshold_sd: float = 3.0
    min_duration_ms: float = 30.0
    max_duration_ms: float = 500.0
    min_gap_ms: float = 30.0


def _smooth_uniform(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.astype(float, copy=False)
    kernel = np.ones(int(window), dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def detect_bursts_from_envelope(
    population_envelope: np.ndarray,
    fs: float,
    params: BurstDetectionParams | None = None,
) -> list[dict]:
    """Find bursts from a 1D population envelope (mean |LFP| across channels)."""
    p = params or BurstDetectionParams()
    env = np.asarray(population_envelope, dtype=float)
    if env.ndim != 1:
        raise ValueError("population_envelope must be 1D")
    if env.size == 0:
        return []

    smooth_window = max(1, int(round(p.envelope_smooth_ms * 1e-3 * fs)))
    smoothed = _smooth_uniform(env, smooth_window)

    median = float(np.median(smoothed))
    mad = float(np.median(np.abs(smoothed - median)))
    sd = 1.4826 * mad if mad > 0 else float(np.std(smoothed))
    threshold = median + p.threshold_sd * sd

    above = smoothed > threshold
    if not above.any():
        return []

    # Find contiguous True runs.
    edges = np.diff(above.astype(np.int8))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if above[0]:
        starts = np.concatenate([[0], starts])
    if above[-1]:
        ends = np.concatenate([ends, [above.size]])

    min_samples = max(1, int(round(p.min_duration_ms * 1e-3 * fs)))
    max_samples = max(min_samples, int(round(p.max_duration_ms * 1e-3 * fs)))
    gap_samples = max(1, int(round(p.min_gap_ms * 1e-3 * fs)))

    # Merge runs separated by < min_gap.
    merged: list[tuple[int, int]] = []
    for s, e in zip(starts, ends):
        if merged and s - merged[-1][1] < gap_samples:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((int(s), int(e)))

    bursts: list[dict] = []
    for s, e in merged:
        length = e - s
        if length < min_samples:
            continue
        if length > max_samples:
            # Split overly long events by re-thresholding at a higher level.
            sub = smoothed[s:e]
            local_thresh = float(np.median(sub) + p.threshold_sd * (1.4826 * np.median(np.abs(sub - np.median(sub))) or np.std(sub)))
            mask = sub > local_thresh
            if not mask.any():
                continue
            mask_edges = np.diff(mask.astype(np.int8))
            sub_starts = np.where(mask_edges == 1)[0] + 1
            sub_ends = np.where(mask_edges == -1)[0] + 1
            if mask[0]:
                sub_starts = np.concatenate([[0], sub_starts])
            if mask[-1]:
                sub_ends = np.concatenate([sub_ends, [mask.size]])
            for ss, se in zip(sub_starts, sub_ends):
                if (se - ss) >= min_samples:
                    bursts.append(_burst_summary(smoothed, s + ss, s + se, fs))
            continue
        bursts.append(_burst_summary(smoothed, s, e, fs))
    return bursts


def _burst_summary(envelope: np.ndarray, start: int, end: int, fs: float) -> dict:
    segment = envelope[start:end]
    peak_offset = int(np.argmax(segment))
    return {
        "t_start_s": float(start / fs),
        "t_end_s": float(end / fs),
        "center_s": float((start + peak_offset) / fs),
        "peak_amp_uv": float(segment[peak_offset]),
        "duration_s": float((end - start) / fs),
    }


def detect_bursts_from_traces(
    traces: np.ndarray,
    fs: float,
    params: BurstDetectionParams | None = None,
) -> list[dict]:
    """Detect bursts given a (n_samples, n_channels) ndarray of LFP traces."""
    arr = np.asarray(traces, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.ndim != 2:
        raise ValueError("traces must be 1D or 2D (n_samples, n_channels)")
    envelope = np.abs(arr).mean(axis=1)
    return detect_bursts_from_envelope(envelope, fs, params)


def compute_bursts_from_recording(
    lfp_recording,
    electrode_table,
    *,
    electrode_ids: Iterable[int] | None = None,
    duration_sec: float | None = None,
    params: BurstDetectionParams | None = None,
    return_scaled: bool = True,
) -> list[dict]:
    """Run burst detection on a SpikeInterface LFP recording.

    Loads at most ``duration_sec`` of data (default: the full recording);
    callers should pass a bound (e.g. 120 s) when the recording is long.
    """
    fs = float(lfp_recording.get_sampling_frequency())
    num_samples = int(lfp_recording.get_num_samples())
    if duration_sec is not None:
        end_frame = min(num_samples, int(round(float(duration_sec) * fs)))
    else:
        end_frame = num_samples
    if end_frame <= 0:
        return []

    _, channel_ids = recorded_electrode_channels(electrode_table, electrode_ids)
    traces = get_traces_safe(lfp_recording, 0, end_frame, channel_ids, return_scaled=return_scaled)
    arr = np.asarray(traces, dtype=float)

    # Reduce to a population envelope cheaply.
    envelope = np.abs(arr).mean(axis=1)
    # If the underlying recording has a high sample rate, downsample the
    # envelope to ~1 kHz for cheap detection — bursts are slow events.
    target_fs = 1000.0
    if fs > 2 * target_fs:
        factor = int(round(fs / target_fs))
        # Anti-alias by mean-pooling rather than scipy.decimate to keep the
        # dependency surface small.
        usable = (envelope.size // factor) * factor
        envelope = envelope[:usable].reshape(-1, factor).mean(axis=1)
        fs_out = fs / factor
    else:
        fs_out = fs

    return detect_bursts_from_envelope(envelope, fs_out, params)
