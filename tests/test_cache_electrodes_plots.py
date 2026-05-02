import json
from pathlib import Path

import numpy as np
import pandas as pd

from acute_slice_mea.cache import load_cache_manifest, save_cache_bundle
from acute_slice_mea.electrodes import build_electrode_table
from acute_slice_mea.plots import write_band_power_html, write_trace_preview_html


class FakeProbe:
    def __init__(self):
        self.contact_positions = np.array(
            [
                [0.0, 0.0],
                [17.5, 0.0],
                [0.0, 17.5],
            ]
        )
        self.name = "fake"


class FakeRecording:
    def get_channel_ids(self):
        return ["a", "b"]

    def get_channel_locations(self):
        return np.array([[0.0, 0.0], [0.0, 17.5]])


def test_build_electrode_table_marks_unrecorded_contacts():
    table = build_electrode_table(FakeProbe(), FakeRecording())

    assert list(table.columns) == ["electrode_id", "channel_id", "x_um", "y_um", "recorded"]
    assert table.loc[0, "channel_id"] == "a"
    assert table.loc[1, "recorded"] is False
    assert table.loc[2, "channel_id"] == "b"


def test_cache_bundle_round_trips_manifest_tables_and_arrays(tmp_path):
    summary = {"data_path": "/tmp/data.raw.h5", "well_id": "well001"}
    electrodes = pd.DataFrame({"electrode_id": [0], "channel_id": ["a"], "recorded": [True]})
    band_power = pd.DataFrame({"electrode_id": [0], "band": ["delta"], "power_db": [1.2]})
    spectrum = {"freq_hz": np.array([1.0, 2.0]), "mean_psd": np.array([3.0, 4.0])}
    trace_preview = {"time_sec": np.array([0.0, 0.1]), "raw": np.ones((2, 1))}

    manifest = save_cache_bundle(
        tmp_path,
        summary=summary,
        electrodes=electrodes,
        band_power=band_power,
        spectrum=spectrum,
        trace_preview=trace_preview,
    )

    loaded = load_cache_manifest(tmp_path)
    assert loaded["summary"]["well_id"] == "well001"
    assert Path(manifest["files"]["electrodes"]).exists()
    assert Path(manifest["files"]["spectrum"]).exists()
    assert json.loads(Path(tmp_path / "summary.json").read_text())["data_path"] == "/tmp/data.raw.h5"


def test_plot_helpers_write_html_without_inline_display(tmp_path):
    band_power = pd.DataFrame(
        {
            "electrode_id": [0, 0, 0],
            "time_sec": [5.0, 10.0, 15.0],
            "band": ["delta", "delta", "delta"],
            "power_db": [1.0, 2.0, 1.5],
        }
    )
    trace_preview = {
        "time_sec": np.array([0.0, 0.1, 0.2]),
        "channel_ids": np.array(["a"]),
        "electrode_ids": np.array([0]),
        "raw": np.array([[0.0], [1.0], [0.0]]),
        "lfp": np.array([[0.0], [0.5], [0.0]]),
        "spike": np.array([[0.0], [2.0], [0.0]]),
    }

    band_path = write_band_power_html(band_power, tmp_path / "band.html")
    trace_path = write_trace_preview_html(trace_preview, tmp_path / "trace.html")

    assert band_path.exists()
    assert trace_path.exists()
    assert "scattergl" in trace_path.read_text().lower()
