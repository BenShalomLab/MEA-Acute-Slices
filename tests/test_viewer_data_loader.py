import json
from pathlib import Path

import numpy as np
import pandas as pd

from acute_slice_mea.viewer.data_loader import WellData


def _make_stub_cache(tmp_path: Path) -> Path:
    cache = tmp_path / "well000"
    cache.mkdir(parents=True)
    summary = {
        "well_id": "well000",
        "data_path": "/tmp/data.raw.h5",
        "sampling_frequency_hz": 20000.0,
        "duration_sec": 30.0,
        "num_recorded_electrodes": 3,
    }
    (cache / "summary.json").write_text(json.dumps(summary))
    electrodes = pd.DataFrame(
        [
            {"electrode_id": 0, "channel_id": "a", "x_um": 0.0, "y_um": 0.0, "recorded": True, "rms_uv": 12.5},
            {"electrode_id": 1, "channel_id": "b", "x_um": 17.5, "y_um": 0.0, "recorded": True, "rms_uv": 18.0},
            {"electrode_id": 2, "channel_id": None, "x_um": 0.0, "y_um": 17.5, "recorded": False, "rms_uv": None},
        ]
    )
    electrodes.to_csv(cache / "electrodes.csv", index=False)
    band_power = pd.DataFrame({"electrode_id": [0], "band": ["delta"], "power_db": [1.0], "time_sec": [5.0]})
    band_power.to_parquet(cache / "lfp_band_power.parquet", index=False)

    (cache / "bursts.json").write_text(json.dumps([
        {"t_start_s": 1.0, "t_end_s": 1.2, "center_s": 1.1, "peak_amp_uv": 100.0, "duration_s": 0.2}
    ]))
    (cache / "probe.json").write_text(json.dumps({"cols": 220, "rows": 120, "pitch_um": 17.5, "routed": [
        {"electrode_id": 0, "channel_id": "a", "col": 0, "row": 0, "x_um": 0.0, "y_um": 0.0, "rms_uv": 12.5},
        {"electrode_id": 1, "channel_id": "b", "col": 1, "row": 0, "x_um": 17.5, "y_um": 0.0, "rms_uv": 18.0},
    ]}))

    # Dashboard data with one trace file
    dash_data = cache / "dashboard" / "data" / "traces" / "lfp"
    dash_data.mkdir(parents=True)
    trace_payload = {
        "signal": "lfp",
        "electrode_id": 0,
        "channel_id": "a",
        "x_um": 0.0,
        "y_um": 0.0,
        "sample_step": 4,
        "time_sec": [0.0, 0.5, 1.0, 1.5, 2.0],
        "value": [0.0, 10.0, -5.0, 12.0, 0.0],
    }
    (dash_data / "0.json").write_text(json.dumps(trace_payload))
    (cache / "dashboard" / "data" / "manifest.json").write_text(json.dumps({
        "signals": ["lfp"],
        "files": {"traces": {"lfp": {"0": "data/traces/lfp/0.json"}}},
    }))

    manifest = {
        "summary": summary,
        "files": {
            "summary": str(cache / "summary.json"),
            "electrodes": str(cache / "electrodes.csv"),
            "band_power": str(cache / "lfp_band_power.parquet"),
            "bursts": str(cache / "bursts.json"),
            "probe": str(cache / "probe.json"),
        },
    }
    (cache / "manifest.json").write_text(json.dumps(manifest))
    return cache


def test_well_data_loads_summary_electrodes_bursts_and_probe(tmp_path):
    cache = _make_stub_cache(tmp_path)
    wd = WellData.load(cache)
    assert wd.duration_sec == 30.0
    assert wd.sample_rate_hz == 20000.0
    assert wd.routed_electrode_ids() == [0, 1]
    assert wd.rms_by_electrode() == {0: 12.5, 1: 18.0}
    assert wd.bursts[0]["center_s"] == 1.1
    assert wd.probe["routed"][0]["electrode_id"] == 0


def test_traces_for_clips_by_time_window(tmp_path):
    cache = _make_stub_cache(tmp_path)
    wd = WellData.load(cache)
    wd.clear_trace_cache()
    payloads = wd.traces_for([0, 99], t0=0.4, t1=1.6)
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["electrode_id"] == 0
    np.testing.assert_allclose(payload["time_sec"], [0.5, 1.0, 1.5])
    np.testing.assert_allclose(payload["value"], [10.0, -5.0, 12.0])


def test_traces_for_returns_empty_when_no_dashboard_data(tmp_path):
    bare = tmp_path / "bare"
    bare.mkdir()
    summary = {"well_id": "well000", "data_path": "/tmp/x", "sampling_frequency_hz": 1, "duration_sec": 1, "num_recorded_electrodes": 0}
    (bare / "summary.json").write_text(json.dumps(summary))
    pd.DataFrame({"electrode_id": [], "channel_id": [], "x_um": [], "y_um": [], "recorded": []}).to_csv(bare / "electrodes.csv", index=False)
    (bare / "manifest.json").write_text(json.dumps({
        "summary": summary,
        "files": {"summary": str(bare / "summary.json"), "electrodes": str(bare / "electrodes.csv")},
    }))
    wd = WellData.load(bare)
    assert wd.traces_for([0]) == []
    assert wd.routed_electrode_ids() == []


def test_traces_for_falls_back_to_recording_when_dashboard_missing(tmp_path, monkeypatch):
    """When no dashboard/ is present, traces_for should slice from raw_path via SpikeInterface."""
    from acute_slice_mea.viewer import data_loader as dl

    cache = tmp_path / "lazy_well"
    cache.mkdir()
    fake_raw = tmp_path / "data.raw.h5"
    fake_raw.write_bytes(b"")  # exists() is the only check
    summary = {
        "well_id": "well000",
        "data_path": str(fake_raw),
        "rec_name": None,
        "sampling_frequency_hz": 1000.0,
        "duration_sec": 5.0,
        "num_recorded_electrodes": 2,
    }
    (cache / "summary.json").write_text(json.dumps(summary))
    pd.DataFrame(
        [
            {"electrode_id": 10, "channel_id": 100, "x_um": 0.0, "y_um": 0.0, "recorded": True, "rms_uv": 12.0},
            {"electrode_id": 11, "channel_id": 101, "x_um": 17.5, "y_um": 0.0, "recorded": True, "rms_uv": 14.0},
        ]
    ).to_csv(cache / "electrodes.csv", index=False)
    (cache / "manifest.json").write_text(json.dumps({
        "summary": summary,
        "files": {"summary": str(cache / "summary.json"), "electrodes": str(cache / "electrodes.csv")},
    }))

    calls = {}

    class _FakeRec:
        def __init__(self):
            self.fs = 1000.0
            self.n = 5000

        def get_sampling_frequency(self):
            return self.fs

        def get_num_frames(self):
            return self.n

        def get_traces(self, *, start_frame, end_frame, channel_ids, return_scaled):
            calls["channel_ids"] = list(channel_ids)
            calls["start_frame"] = start_frame
            calls["end_frame"] = end_frame
            n = end_frame - start_frame
            return np.tile(np.arange(n, dtype=np.float32)[:, None], (1, len(channel_ids)))

    def fake_open(raw_path, well_id, rec_name):
        return {"lfp": _FakeRec(), "raw": _FakeRec(), "spike": _FakeRec()}

    monkeypatch.setattr(dl, "_open_prepared_recording", fake_open)
    dl._lazy_lfp_window.cache_clear()

    wd = dl.WellData.load(cache)
    payloads = wd.traces_for([10, 11], t0=1.0, t1=3.0)

    assert len(payloads) == 2
    assert {p["electrode_id"] for p in payloads} == {10, 11}
    assert calls["channel_ids"] == ["100", "101"]
    assert calls["start_frame"] == 1000
    assert calls["end_frame"] == 3000
    assert payloads[0]["time_sec"][0] == 1.0
    assert payloads[0]["value"].shape == payloads[0]["time_sec"].shape
