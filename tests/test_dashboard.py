import json
from pathlib import Path

import numpy as np
import pandas as pd

from acute_slice_mea.dashboard import export_dashboard_data, write_dashboard_html


class FakeRecording:
    def __init__(self, traces, fs=100.0, channel_ids=None):
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
        indices = [self._channel_ids.index(ch) for ch in channel_ids]
        return self._traces[start_frame:end_frame, :][:, indices]


def test_export_dashboard_data_writes_lazy_loaded_trace_files(tmp_path):
    t = np.arange(0, 10, 0.01)
    traces = np.column_stack([np.sin(t), np.cos(t)])
    recordings = {
        "raw": FakeRecording(traces, channel_ids=["a", "b"]),
        "lfp": FakeRecording(traces * 0.5, channel_ids=["a", "b"]),
        "spike": FakeRecording(traces * 2, channel_ids=["a", "b"]),
    }
    electrodes = pd.DataFrame(
        {
            "electrode_id": [10, 20],
            "channel_id": ["a", "b"],
            "x_um": [0.0, 17.5],
            "y_um": [0.0, 17.5],
            "recorded": [True, True],
        }
    )
    band_power = pd.DataFrame(
        {
            "electrode_id": [10, 10, 20],
            "time_sec": [5.0, 10.0, 5.0],
            "band": ["delta", "theta", "delta"],
            "power_db": [1.0, 2.0, 3.0],
        }
    )
    summary = {"well_id": "well004", "duration_sec": 10, "sampling_frequency_hz": 100}

    manifest = export_dashboard_data(
        tmp_path / "dashboard",
        recordings=recordings,
        electrodes=electrodes,
        band_power=band_power,
        summary=summary,
        max_points_per_electrode=100,
    )

    manifest_path = Path(manifest["data_manifest"])
    assert manifest_path.exists()
    loaded = json.loads(manifest_path.read_text())
    assert loaded["signals"] == ["raw", "lfp", "spike"]
    assert loaded["electrodes"][0]["electrode_id"] == 10

    trace_path = tmp_path / "dashboard" / "data" / "traces" / "raw" / "10.json"
    payload = json.loads(trace_path.read_text())
    assert payload["electrode_id"] == 10
    assert payload["channel_id"] == "a"
    assert len(payload["time_sec"]) <= 100
    assert len(payload["value"]) == len(payload["time_sec"])


def test_write_dashboard_html_references_lazy_data_manifest(tmp_path):
    dashboard_dir = tmp_path / "dashboard"
    data_dir = dashboard_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "manifest.json").write_text(json.dumps({"signals": ["raw"]}))

    output = write_dashboard_html(dashboard_dir)
    html = output.read_text()

    assert output.name == "index.html"
    assert "data/manifest.json" in html
    assert "Plotly.newPlot" in html
    assert "trace_preview" not in html
