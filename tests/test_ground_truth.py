"""Tests for acute_slice_mea.ground_truth module."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path if running directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from acute_slice_mea.ground_truth import (
    add_scale_bars,
    find_stim_blocks,
    plot_ground_truth,
    resolve_channel_and_electrode_pairs,
    time_to_indices,
)


class DummyRecording:
    """Mock SpikeInterface BaseRecording for unit tests."""

    def __init__(self, channel_ids: list[str], fs: float = 20000.0, num_frames: int = 100000):
        self._channel_ids = [str(c) for c in channel_ids]
        self._fs = fs
        self._num_frames = num_frames

    def get_channel_ids(self) -> list[str]:
        return self._channel_ids

    def get_num_channels(self) -> int:
        return len(self._channel_ids)

    def get_sampling_frequency(self) -> float:
        return self._fs

    def get_num_frames(self) -> int:
        return self._num_frames

    def get_traces(self, start_frame: int, end_frame: int, channel_ids: list[str], **kwargs):
        n_frames = max(0, end_frame - start_frame)
        n_ch = len(channel_ids)
        t = (np.arange(start_frame, end_frame, dtype=np.float64) / self._fs)[:, np.newaxis]
        ch_factors = np.arange(1, n_ch + 1, dtype=np.float64)[np.newaxis, :]
        return (np.sin(2 * np.pi * 10.0 * t * ch_factors) * 50.0).astype(np.float32)


# -----------------------------------------------------------------------------
# Timestamp to Index Conversion Tests
# -----------------------------------------------------------------------------


def test_time_to_indices_standard():
    fs = 20000.0
    start, end = time_to_indices(1.0, 2.5, fs=fs, max_frames=100000)
    assert start == 20000
    assert end == 50000


def test_time_to_indices_clamping():
    fs = 1000.0
    # Clamping past max_frames
    start, end = time_to_indices(5.0, 15.0, fs=fs, max_frames=8000)
    assert start == 5000
    assert end == 8000

    # Negative timestamps clamped to 0
    start_neg, end_neg = time_to_indices(-2.0, 3.0, fs=fs, max_frames=8000)
    assert start_neg == 0
    assert end_neg == 3000


def test_time_to_indices_inverted_inputs():
    fs = 10000.0
    # User passes t0 > t1
    start, end = time_to_indices(3.0, 1.0, fs=fs)
    assert start == 10000
    assert end == 30000


def test_time_to_indices_invalid_fs():
    failed = False
    try:
        time_to_indices(0.0, 1.0, fs=0.0)
    except ValueError as exc:
        failed = True
        assert "Sampling frequency fs must be positive" in str(exc)
    assert failed, "Expected ValueError for fs=0.0"


# -----------------------------------------------------------------------------
# Scale Bar Helper Tests
# -----------------------------------------------------------------------------


def test_add_scale_bars():
    fig, ax = plt.subplots()
    ax.plot([0, 10], [-100, 100])

    initial_lines = len(ax.lines)
    initial_texts = len(ax.texts)

    add_scale_bars(
        ax,
        x_size=1.0,
        y_size=50.0,
        x_label="1",
        y_label="50",
        x_unit="s",
        y_unit="µV",
    )

    assert len(ax.lines) == initial_lines + 2
    assert len(ax.texts) == initial_texts + 2
    plt.close(fig)


# -----------------------------------------------------------------------------
# Channel & Electrode Resolution Tests
# -----------------------------------------------------------------------------


def test_resolve_channel_and_electrode_pairs(tmp_path: Path):
    electrodes_csv = tmp_path / "electrodes.csv"
    df = pd.DataFrame(
        {
            "electrode_id": [821, 148, 532],
            "channel_id": ["947", "566", "8"],
            "recorded": [True, True, True],
        }
    )
    df.to_csv(electrodes_csv, index=False)

    rec = DummyRecording(channel_ids=["947", "566", "8", "631"])

    # 1. Resolve by physical electrode ID
    pairs_by_eid = resolve_channel_and_electrode_pairs(
        rec, electrodes=[821, 148], cache_dir=tmp_path
    )
    assert pairs_by_eid == [(821, "947"), (148, "566")]

    # 2. Resolve by channel ID
    pairs_by_cid = resolve_channel_and_electrode_pairs(
        rec, channels=["947", "8"], cache_dir=tmp_path
    )
    assert pairs_by_cid == [(821, "947"), (532, "8")]

    # 3. Fallback when neither provided
    pairs_default = resolve_channel_and_electrode_pairs(rec, cache_dir=tmp_path)
    assert len(pairs_default) == 4


# -----------------------------------------------------------------------------
# Plot Ground Truth Tests
# -----------------------------------------------------------------------------


def test_plot_ground_truth_empty():
    fig = plot_ground_truth([])
    assert fig is not None
    plt.close(fig)


def test_plot_ground_truth_stacked(tmp_path: Path):
    t = np.linspace(0, 1.0, 1000)
    payloads = [
        {
            "electrode_id": 821,
            "channel_id": "947",
            "time_sec": t,
            "value": np.sin(2 * np.pi * 5 * t) * 40.0,
        },
        {
            "electrode_id": 148,
            "channel_id": "566",
            "time_sec": t,
            "value": np.cos(2 * np.pi * 5 * t) * 30.0,
        },
    ]

    out_file = tmp_path / "test_plot.png"
    fig = plot_ground_truth(
        payloads,
        spacing_uv=250.0,
        gain=1.0,
        scalebars=True,
        title="Test Ground Truth",
        save_path=out_file,
    )

    assert fig is not None
    assert out_file.exists()
    assert out_file.stat().st_size > 0
    plt.close(fig)


# -----------------------------------------------------------------------------
# Stimulation Bookmark / Condition Blocks Tests
# -----------------------------------------------------------------------------


class MockStimRecording:
    """Mock recording with artificial stim pulses grouped into 2 blocks."""

    def __init__(self, fs: float = 10000.0, total_duration_s: float = 30.0):
        self._fs = fs
        self._num_frames = int(total_duration_s * fs)
        self._channel_ids = ["631"]

    def get_channel_ids(self) -> list[str]:
        return self._channel_ids

    def get_sampling_frequency(self) -> float:
        return self._fs

    def get_num_frames(self) -> int:
        return self._num_frames

    def get_traces(self, start_frame: int, end_frame: int, channel_ids: list[str], **kwargs):
        n_frames = max(0, end_frame - start_frame)
        signal = np.zeros(n_frames, dtype=np.float32)

        # Block 1: 5 pulses spaced 0.5s apart from t=2.0s to t=4.0s
        block1_pulses = [2.0, 2.5, 3.0, 3.5, 4.0]
        # Block 2: 5 pulses spaced 0.1s apart from t=15.0s to t=15.4s (gap = 11s)
        block2_pulses = [15.0, 15.1, 15.2, 15.3, 15.4]

        for p_t in block1_pulses + block2_pulses:
            p_frame = int(p_t * self._fs)
            if start_frame <= p_frame < end_frame:
                idx = p_frame - start_frame
                signal[idx] = -800.0

        return signal[:, np.newaxis]


def test_find_stim_blocks():
    rec = MockStimRecording(fs=10000.0, total_duration_s=30.0)
    blocks = find_stim_blocks(rec, stim_channel="631", height=500.0, gap_threshold_s=4.0)

    assert len(blocks) == 2

    # Block 1 check
    assert blocks[0]["block"] == 1
    assert abs(blocks[0]["t_start"] - 2.0) < 0.01
    assert abs(blocks[0]["t_end"] - 4.0) < 0.01
    assert blocks[0]["n_pulses"] == 5

    # Block 2 check
    assert blocks[1]["block"] == 2
    assert abs(blocks[1]["t_start"] - 15.0) < 0.01
    assert abs(blocks[1]["t_end"] - 15.4) < 0.01
    assert blocks[1]["n_pulses"] == 5


if __name__ == "__main__":
    import tempfile

    print("Running tests in test_ground_truth.py...")
    test_time_to_indices_standard()
    test_time_to_indices_clamping()
    test_time_to_indices_inverted_inputs()
    test_time_to_indices_invalid_fs()
    test_add_scale_bars()

    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        test_resolve_channel_and_electrode_pairs(p)
        test_plot_ground_truth_stacked(p)

    test_plot_ground_truth_empty()
    test_find_stim_blocks()
    print("ALL TESTS PASSED SUCCESSFULLY! ✓")
