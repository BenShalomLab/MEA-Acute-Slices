import sys
import types
from unittest.mock import MagicMock
from pathlib import Path
import importlib

import numpy as np
import pytest

# Ensure acute_slice_mea can be imported in environments where heavy visualization
# or spike-sorting packages are not installed on the system runner.
if "acute_slice_mea" not in sys.modules:
    pkg = types.ModuleType("acute_slice_mea")
    pkg.__path__ = [str(Path(__file__).resolve().parent.parent / "src" / "acute_slice_mea")]
    sys.modules["acute_slice_mea"] = pkg

for m in ["pandas", "scipy", "scipy.signal", "joblib"]:
    if m not in sys.modules:
        sys.modules[m] = MagicMock()

dashboard_mod = importlib.import_module("acute_slice_mea.dashboard")
_compute_downsampling_params = dashboard_mod._compute_downsampling_params
_read_decimated_all_channels = dashboard_mod._read_decimated_all_channels


class MockRecording:
    """Minimal recording stub providing get_traces and get_sampling_frequency."""

    def __init__(self, traces: np.ndarray, fs: float = 20000.0, channel_ids: list[str] | None = None):
        self._traces = np.asarray(traces, dtype=np.float32)
        if self._traces.ndim == 1:
            self._traces = self._traces[:, np.newaxis]
        self._fs = float(fs)
        self._channel_ids = channel_ids or [str(i) for i in range(self._traces.shape[1])]

    def get_sampling_frequency(self) -> float:
        return self._fs

    def get_num_samples(self) -> int:
        return self._traces.shape[0]

    def get_num_channels(self) -> int:
        return self._traces.shape[1]

    def get_channel_ids(self) -> list[str]:
        return self._channel_ids

    def get_traces(self, start_frame: int = 0, end_frame: int | None = None, channel_ids: list[str] | None = None, return_scaled: bool = True):
        end = self._traces.shape[0] if end_frame is None else end_frame
        if channel_ids is None:
            return self._traces[start_frame:end]
        col_indices = [self._channel_ids.index(cid) for cid in channel_ids]
        return self._traces[start_frame:end, col_indices]


def test_compute_downsampling_params_short_recording_no_decimation():
    """Recordings shorter than max_points must not be downsampled."""
    fs = 20000.0
    num_samples = 5000
    max_points = 20000

    bin_size, use_minmax = _compute_downsampling_params(num_samples, fs, max_points)
    assert bin_size == 1
    assert not use_minmax


def test_compute_downsampling_params_respects_max_points_ceiling():
    """For any recording length, output points must not exceed max_points."""
    fs = 20000.0
    max_points = 20000

    for duration_sec in [2, 10, 60, 300, 1800]:
        num_samples = int(duration_sec * fs)
        bin_size, use_minmax = _compute_downsampling_params(num_samples, fs, max_points)
        assert use_minmax
        assert bin_size >= 2
        # Number of bins
        n_bins = int(np.ceil(num_samples / bin_size))
        max_output_points = n_bins * 2
        assert max_output_points <= max_points


def test_compute_downsampling_params_balanced_target_rate():
    """Short recordings should preserve target_rate_hz while long ones scale up."""
    fs = 20000.0
    max_points = 20000
    target_rate = 2000.0  # ideal: 10 samples per bin

    # 3 seconds: 60,000 samples -> ideal 6,000 bins = 12,000 points <= 20,000
    bin_size_short, use_minmax_short = _compute_downsampling_params(
        num_samples=60000, fs=fs, max_points_per_electrode=max_points, target_rate_hz=target_rate
    )
    assert use_minmax_short
    assert bin_size_short == 10  # 20000 / 2000

    # 60 seconds: 1,200,000 samples -> ideal 120,000 bins > 10,000 bins cap
    bin_size_long, use_minmax_long = _compute_downsampling_params(
        num_samples=1200000, fs=fs, max_points_per_electrode=max_points, target_rate_hz=target_rate
    )
    assert use_minmax_long
    assert bin_size_long == 120  # capped to stay within max_points


def test_minmax_preserves_spike_peaks_and_troughs():
    """Min-Max downsampling must retain 100% of action potential amplitudes."""
    fs = 20000.0
    duration_sec = 60.0
    num_samples = int(duration_sec * fs)
    t = np.arange(num_samples, dtype=np.float32) / fs

    # Baseline noise around 0 +/- 5 uV
    np.random.seed(42)
    signal = (np.random.randn(num_samples) * 5.0).astype(np.float32)

    # Inject 5 sharp spikes of -150 uV amplitude lasting 1 ms (20 frames)
    spike_times = [3.5, 15.2, 28.7, 44.1, 58.9]
    for st in spike_times:
        start_idx = int(st * fs)
        pulse = -150.0 * np.sin(np.linspace(0, np.pi, 20)).astype(np.float32)
        signal[start_idx : start_idx + 20] += pulse

    # Inject a positive excursion of +80 uV
    signal[int(20.0 * fs) : int(20.0 * fs) + 20] += 80.0

    true_min = float(np.min(signal))
    true_max = float(np.max(signal))
    assert true_min < -140.0
    assert true_max > 75.0

    rec = MockRecording(signal[:, np.newaxis], fs=fs, channel_ids=["0"])

    # Downsample using bin_size=120 (decimation from 1.2M down to 20,000 points)
    time_sec, decimated = _read_decimated_all_channels(
        rec, channel_ids=["0"], num_samples=num_samples, fs=fs, bin_size=120, use_minmax=True
    )

    dec_min = float(np.min(decimated[:, 0]))
    dec_max = float(np.max(decimated[:, 0]))

    # Min-Max must capture the exact extrema without attenuation
    assert np.isclose(dec_min, true_min, atol=1e-4)
    assert np.isclose(dec_max, true_max, atol=1e-4)

    # In contrast, show that naive strided decimation misses the spike amplitude:
    naive_strided = signal[::120]
    naive_min = float(np.min(naive_strided))
    assert naive_min > -50.0  # Naive striding lost > 65% of the spike!


def test_chunk_and_bin_boundary_alignment_across_odd_sizes():
    """Tests that chunks ending on odd boundaries don't lose samples or fail alignment."""
    fs = 1000.0
    for num_samples in [1, 2, 7, 50, 99, 100, 101, 250, 1357]:
        for bin_size in [2, 5, 20, 50]:
            rec_data = np.arange(num_samples, dtype=np.float32)[:, np.newaxis]
            rec = MockRecording(rec_data, fs=fs, channel_ids=["ch0"])

            time_sec, decimated = _read_decimated_all_channels(
                rec,
                channel_ids=["ch0"],
                num_samples=num_samples,
                fs=fs,
                bin_size=bin_size,
                use_minmax=True,
                chunk_frames_target=100,
            )

            # 1. Output time and value arrays must match in length
            assert len(time_sec) == decimated.shape[0]

            # 2. Timestamps must be monotonic non-decreasing
            if len(time_sec) > 1:
                assert np.all(np.diff(time_sec) >= 0)

            # 3. Global extrema must match the raw data
            assert np.isclose(np.min(decimated), np.min(rec_data))
            assert np.isclose(np.max(decimated), np.max(rec_data))


def test_full_resolution_when_unaggregated():
    """When bin_size=1 or use_minmax=False, all samples must be retained."""
    fs = 1000.0
    rec_data = np.arange(150, dtype=np.float32)[:, np.newaxis]
    rec = MockRecording(rec_data, fs=fs, channel_ids=["ch0"])

    time_sec, decimated = _read_decimated_all_channels(
        rec, channel_ids=["ch0"], num_samples=150, fs=fs, bin_size=1, use_minmax=False
    )
    assert len(time_sec) == 150
    assert decimated.shape == (150, 1)
    assert np.allclose(decimated, rec_data)


def test_export_dashboard_data_minmax_e2e(tmp_path):
    """Test export_dashboard_data end-to-end with min-max aggregation."""
    import json
    from collections import namedtuple

    export_dashboard_data = dashboard_mod.export_dashboard_data

    # Create synthetic recording with a spike
    fs = 1000.0
    num_samples = 500  # 0.5 s
    signal = np.sin(np.linspace(0, 10, num_samples)).astype(np.float32)
    signal[250] = -50.0  # sharp spike

    recordings = {
        "raw": MockRecording(signal[:, np.newaxis], fs=fs, channel_ids=["ch0"]),
        "spike": MockRecording(signal[:, np.newaxis], fs=fs, channel_ids=["ch0"]),
    }

    Row = namedtuple("Row", ["electrode_id", "channel_id", "x_um", "y_um", "recorded"])
    row = Row(101, "ch0", 0.0, 0.0, True)

    class FakeSeries(list):
        def tolist(self):
            return list(self)
        def astype(self, _t):
            return self

    class FakeDF:
        def __init__(self, rows):
            self._rows = rows
        def __getitem__(self, item):
            if isinstance(item, str):
                return FakeSeries([getattr(r, item) for r in self._rows])
            return self
        def __setitem__(self, key, value):
            pass
        def copy(self):
            return self
        def sort_values(self, _by):
            return self
        def to_dict(self, orient="records"):
            return [r._asdict() if hasattr(r, "_asdict") else r for r in self._rows]
        def itertuples(self, index=False):
            return iter(self._rows)
        @property
        def empty(self):
            return len(self._rows) == 0
        def __len__(self):
            return len(self._rows)

    electrodes = FakeDF([row])

    BandRow = namedtuple("BandRow", ["electrode_id", "time_sec", "band", "power_db"])
    band_power = FakeDF([BandRow(101, 0.25, "gamma", 12.0)])
    band_power.columns = ["electrode_id", "time_sec", "band", "power_db"]

    summary = {"duration_sec": 0.5, "sampling_frequency_hz": fs}

    result = export_dashboard_data(
        tmp_path / "dashboard",
        recordings=recordings,
        electrodes=electrodes,
        band_power=band_power,
        summary=summary,
        max_points_per_electrode=50,  # Forces bin_size = 500 // 25 = 20
    )

    data_manifest_path = Path(result["data_manifest"])
    assert data_manifest_path.exists()
    manifest_data = json.loads(data_manifest_path.read_text())
    assert manifest_data["aggregation"] == "minmax"
    assert manifest_data["sample_step"] == 20

    trace_file = tmp_path / "dashboard" / "data" / "traces" / "spike" / "101.json"
    assert trace_file.exists()
    payload = json.loads(trace_file.read_text())
    assert payload["electrode_id"] == 101
    assert payload["sample_step"] == 20
    assert len(payload["time_sec"]) <= 50
    assert len(payload["value"]) == len(payload["time_sec"])

    # Verify that the spike of -50.0 was preserved in the exported JSON
    min_val = min(payload["value"])
    assert np.isclose(min_val, -50.0, atol=1e-3)

