import numpy as np
import pandas as pd

from acute_slice_mea import pipeline
from acute_slice_mea.pipeline import AnalysisConfig, run_analysis


class FakeRecording:
    def get_sampling_frequency(self):
        return 100.0

    def get_num_samples(self):
        return 1000

    def get_probe(self):
        return object()


def test_run_analysis_writes_verbose_progress_to_stderr(monkeypatch, capsys, tmp_path):
    calls = {}
    recordings = {"raw": FakeRecording(), "lfp": FakeRecording(), "spike": FakeRecording()}
    electrodes = pd.DataFrame(
        {
            "electrode_id": [10, 20],
            "channel_id": ["ch0", "ch1"],
            "recorded": [True, True],
        }
    )

    monkeypatch.setattr(pipeline, "load_maxwell_recording", lambda data_path, well_id: FakeRecording())
    monkeypatch.setattr(pipeline, "prepare_recordings", lambda raw, **kwargs: recordings)
    monkeypatch.setattr(pipeline, "build_electrode_table", lambda probe, recording: electrodes)

    def fake_band_power(recording, **kwargs):
        calls["band_power_progress"] = kwargs["progress"]
        return pd.DataFrame({"electrode_id": [10], "band": ["delta"], "power_db": [1.0]})

    def fake_spectrum(recording, **kwargs):
        calls["spectrum_progress"] = kwargs["progress"]
        return {"freq_hz": np.array([1.0]), "mean_psd": np.array([2.0])}

    monkeypatch.setattr(pipeline, "compute_lfp_band_power_over_time", fake_band_power)
    monkeypatch.setattr(pipeline, "compute_welch_spectrum_summary", fake_spectrum)
    monkeypatch.setattr(pipeline, "compute_trace_preview", lambda *args, **kwargs: {"time_sec": np.array([0.0])})
    monkeypatch.setattr(
        pipeline,
        "save_cache_bundle",
        lambda output_dir, **kwargs: {"summary": kwargs["summary"], "files": {}},
    )
    monkeypatch.setattr(pipeline, "write_band_power_html", lambda band_power, path: path)
    monkeypatch.setattr(pipeline, "write_trace_preview_html", lambda trace_preview, path: path)
    monkeypatch.setattr(pipeline, "write_spectrum_summary_html", lambda spectrum, path: path)
    monkeypatch.setattr(pipeline, "export_dashboard_data", lambda *args, **kwargs: {"data_manifest": "manifest.json"})

    manifest = run_analysis(
        AnalysisConfig(
            data_path="data.raw.h5",
            well_id="well004",
            output_dir=str(tmp_path),
            spikeinterface_chunk_duration="",
            progress=True,
            verbose=True,
        )
    )
    captured = capsys.readouterr()

    assert calls == {"band_power_progress": True, "spectrum_progress": True}
    assert manifest["summary"]["progress"] is True
    assert manifest["summary"]["verbose"] is True
    assert captured.out == ""
    assert "Loading recording" in captured.err
    assert "Computing LFP band power" in captured.err
    assert "Writing manifest" in captured.err
