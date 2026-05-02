import numpy as np
import pandas as pd
import pytest

from acute_slice_mea.spectral import (
    DEFAULT_LFP_BANDS,
    compute_lfp_band_power_over_time,
    compute_welch_spectrum_summary,
)


class FakeRecording:
    def __init__(self, traces, fs=1000.0, channel_ids=None):
        self._traces = np.asarray(traces)
        self._fs = float(fs)
        self._channel_ids = list(channel_ids or range(self._traces.shape[1]))

    def get_sampling_frequency(self):
        return self._fs

    def get_num_samples(self):
        return self._traces.shape[0]

    def get_channel_ids(self):
        return self._channel_ids

    def get_traces(self, start_frame=None, end_frame=None, channel_ids=None, return_scaled=True):
        start_frame = 0 if start_frame is None else start_frame
        end_frame = self._traces.shape[0] if end_frame is None else end_frame
        if channel_ids is None:
            indices = list(range(len(self._channel_ids)))
        else:
            indices = [self._channel_ids.index(ch) for ch in channel_ids]
        return self._traces[start_frame:end_frame, :][:, indices]


def test_lfp_band_power_uses_all_channels_when_no_electrodes_are_selected():
    fs = 100.0
    t = np.arange(0, 20, 1 / fs)
    traces = np.column_stack(
        [
            np.sin(2 * np.pi * 2 * t),
            np.sin(2 * np.pi * 10 * t),
        ]
    )
    recording = FakeRecording(traces, fs=fs, channel_ids=["ch0", "ch1"])

    result = compute_lfp_band_power_over_time(
        recording,
        start_sec=0,
        end_sec=20,
        window_sec=10,
        step_sec=5,
        bands=DEFAULT_LFP_BANDS,
    )

    assert set(result["electrode_id"]) == {0, 1}
    assert set(result["channel_id"]) == {"ch0", "ch1"}
    assert set(result["band"]) >= {"delta", "alpha"}
    assert result["power_db"].notna().all()
    assert len(result["time_sec"].unique()) == 3


def test_lfp_band_power_rejects_empty_windows():
    recording = FakeRecording(np.zeros((100, 2)), fs=100.0)

    try:
        compute_lfp_band_power_over_time(recording, start_sec=2, end_sec=1)
    except ValueError as exc:
        assert "interval is empty" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_lfp_band_power_does_not_touch_removed_numpy_trapz(monkeypatch):
    fs = 100.0
    t = np.arange(0, 10, 1 / fs)
    recording = FakeRecording(np.sin(2 * np.pi * 2 * t)[:, None], fs=fs)

    def fail_if_used(_name):
        raise AttributeError("module 'numpy' has no attribute 'trapz'")

    monkeypatch.setattr(np, "trapz", fail_if_used)

    result = compute_lfp_band_power_over_time(
        recording,
        start_sec=0,
        end_sec=10,
        window_sec=10,
        step_sec=5,
    )

    assert not result.empty


def test_welch_spectrum_summary_returns_frequency_axis_and_percentiles():
    fs = 100.0
    t = np.arange(0, 5, 1 / fs)
    traces = np.column_stack(
        [
            np.sin(2 * np.pi * 8 * t),
            np.sin(2 * np.pi * 20 * t),
            np.sin(2 * np.pi * 35 * t),
        ]
    )
    recording = FakeRecording(traces, fs=fs)

    summary = compute_welch_spectrum_summary(recording, duration_sec=5, max_freq_hz=40)

    assert summary["freq_hz"].ndim == 1
    assert summary["mean_psd"].shape == summary["freq_hz"].shape
    assert summary["p05_psd"].shape == summary["freq_hz"].shape
    assert summary["p95_psd"].shape == summary["freq_hz"].shape
    assert len(summary["dominant_freq_hz"]) == 3


def test_lfp_band_power_parallel_matches_serial_and_preserves_order():
    fs = 100.0
    t = np.arange(0, 20, 1 / fs)
    traces = np.column_stack(
        [
            np.sin(2 * np.pi * 2 * t),
            np.sin(2 * np.pi * 10 * t),
            np.sin(2 * np.pi * 35 * t),
        ]
    )
    recording = FakeRecording(traces, fs=fs, channel_ids=["ch0", "ch1", "ch2"])

    serial = compute_lfp_band_power_over_time(
        recording,
        start_sec=0,
        end_sec=20,
        window_sec=10,
        step_sec=5,
        bands=DEFAULT_LFP_BANDS,
        n_jobs=1,
    )
    parallel = compute_lfp_band_power_over_time(
        recording,
        start_sec=0,
        end_sec=20,
        window_sec=10,
        step_sec=5,
        bands=DEFAULT_LFP_BANDS,
        n_jobs=2,
        channel_chunk_size=1,
    )

    pd.testing.assert_frame_equal(parallel, serial)
    assert parallel["channel_id"].tolist() == serial["channel_id"].tolist()


def test_welch_spectrum_summary_parallel_matches_serial_and_preserves_channel_order():
    fs = 100.0
    t = np.arange(0, 5, 1 / fs)
    traces = np.column_stack(
        [
            np.sin(2 * np.pi * 8 * t),
            np.sin(2 * np.pi * 20 * t),
            np.sin(2 * np.pi * 35 * t),
        ]
    )
    recording = FakeRecording(traces, fs=fs, channel_ids=["ch0", "ch1", "ch2"])

    serial = compute_welch_spectrum_summary(recording, duration_sec=5, max_freq_hz=40, n_jobs=1)
    parallel = compute_welch_spectrum_summary(
        recording,
        duration_sec=5,
        max_freq_hz=40,
        n_jobs=2,
        channel_chunk_size=1,
    )

    assert list(parallel["channel_ids"]) == ["ch0", "ch1", "ch2"]
    assert parallel.keys() == serial.keys()
    for key in serial:
        np.testing.assert_equal(parallel[key], serial[key])


def test_parallel_options_reject_invalid_values():
    recording = FakeRecording(np.zeros((100, 2)), fs=100.0)

    with pytest.raises(ValueError, match="n_jobs"):
        compute_lfp_band_power_over_time(recording, n_jobs=0)

    with pytest.raises(ValueError, match="channel_chunk_size"):
        compute_welch_spectrum_summary(recording, channel_chunk_size=0)
