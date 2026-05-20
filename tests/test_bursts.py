import numpy as np

from acute_slice_mea.bursts import (
    BurstDetectionParams,
    detect_bursts_from_envelope,
    detect_bursts_from_traces,
)


def _synthesize_lfp_with_bursts(fs=1000.0, duration_sec=10.0, burst_centers=(2.0, 5.0, 8.0), n_channels=8, seed=0):
    rng = np.random.default_rng(seed)
    n = int(duration_sec * fs)
    t = np.arange(n) / fs
    traces = rng.standard_normal((n, n_channels)) * 5.0  # ~5 µV baseline noise
    for center in burst_centers:
        width = 0.1  # 100 ms
        gauss = np.exp(-((t - center) / width) ** 2)
        traces += (gauss[:, None] * 80.0) * (1.0 + 0.05 * rng.standard_normal(n_channels))
    return traces, fs


def test_detect_bursts_recovers_injected_centers():
    traces, fs = _synthesize_lfp_with_bursts()
    bursts = detect_bursts_from_traces(traces, fs)
    centers = sorted(b["center_s"] for b in bursts)
    assert len(centers) == 3
    for expected, got in zip([2.0, 5.0, 8.0], centers):
        assert abs(got - expected) < 0.05


def test_detect_bursts_from_envelope_returns_empty_on_flat_signal():
    fs = 1000.0
    envelope = np.ones(1000) * 5.0
    assert detect_bursts_from_envelope(envelope, fs) == []


def test_burst_durations_within_bounds():
    traces, fs = _synthesize_lfp_with_bursts()
    bursts = detect_bursts_from_traces(traces, fs)
    for b in bursts:
        assert 0.03 <= b["duration_s"] <= 0.5


def test_min_duration_filter_drops_short_blips():
    traces, fs = _synthesize_lfp_with_bursts(burst_centers=(2.0,))
    # Demand 5+ second bursts; the injected 100 ms burst should be rejected.
    strict = detect_bursts_from_traces(traces, fs, BurstDetectionParams(min_duration_ms=5000.0, max_duration_ms=10000.0))
    assert strict == []
