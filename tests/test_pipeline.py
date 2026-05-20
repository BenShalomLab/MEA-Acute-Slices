from pathlib import Path

import numpy as np
import pandas as pd

from acute_slice_mea import pipeline, recording as recording_module
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

    monkeypatch.setattr(pipeline, "load_maxwell_recording", lambda data_path, well_id, rec_name=None: FakeRecording())
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
    monkeypatch.setattr(pipeline, "compute_bursts_from_recording", lambda *args, **kwargs: [{"center_s": 1.0}])
    monkeypatch.setattr(pipeline, "compute_lfp_rms_per_electrode", lambda *args, **kwargs: {10: 12.5, 20: 8.0})
    monkeypatch.setattr(pipeline, "build_probe_geometry", lambda electrodes: {"cols": 220, "rows": 120, "routed": []})
    saved: dict = {}

    def fake_save(output_dir, **kwargs):
        saved.update(kwargs)
        return {"summary": kwargs["summary"], "files": {}}

    monkeypatch.setattr(pipeline, "save_cache_bundle", fake_save)
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
            cache_lfp_to_disk=False,
            progress=True,
            verbose=True,
        )
    )
    captured = capsys.readouterr()

    assert calls == {"band_power_progress": True, "spectrum_progress": True}
    assert manifest["summary"]["progress"] is True
    assert manifest["summary"]["verbose"] is True
    assert manifest["summary"]["num_bursts"] == 1
    assert saved["bursts"] == [{"center_s": 1.0}]
    assert saved["probe_geometry"] == {"cols": 220, "rows": 120, "routed": []}
    assert "rms_uv" in saved["electrodes"].columns
    assert captured.out == ""
    assert "Loading recording" in captured.err
    assert "Computing LFP band power" in captured.err
    assert "Detecting network bursts" in captured.err
    assert "Building probe geometry" in captured.err
    assert "Writing manifest" in captured.err


def _wire_minimal_pipeline_stubs(monkeypatch):
    """Stub out heavy pipeline collaborators with cheap fakes."""
    recordings = {"raw": FakeRecording(), "lfp": FakeRecording(), "spike": FakeRecording()}
    electrodes = pd.DataFrame(
        {
            "electrode_id": [10, 20],
            "channel_id": ["ch0", "ch1"],
            "recorded": [True, True],
        }
    )
    monkeypatch.setattr(pipeline, "load_maxwell_recording", lambda data_path, well_id, rec_name=None: FakeRecording())
    monkeypatch.setattr(pipeline, "prepare_recordings", lambda raw, **kwargs: recordings)
    monkeypatch.setattr(pipeline, "build_electrode_table", lambda probe, recording: electrodes)
    monkeypatch.setattr(
        pipeline,
        "compute_lfp_band_power_over_time",
        lambda recording, **kwargs: pd.DataFrame({"electrode_id": [10], "band": ["delta"], "power_db": [1.0]}),
    )
    monkeypatch.setattr(
        pipeline,
        "compute_welch_spectrum_summary",
        lambda recording, **kwargs: {"freq_hz": np.array([1.0]), "mean_psd": np.array([2.0])},
    )
    monkeypatch.setattr(pipeline, "compute_trace_preview", lambda *args, **kwargs: {"time_sec": np.array([0.0])})
    monkeypatch.setattr(pipeline, "compute_bursts_from_recording", lambda *args, **kwargs: [])
    monkeypatch.setattr(pipeline, "compute_lfp_rms_per_electrode", lambda *args, **kwargs: {})
    monkeypatch.setattr(pipeline, "build_probe_geometry", lambda electrodes: {"cols": 1, "rows": 1, "routed": []})
    monkeypatch.setattr(pipeline, "save_cache_bundle", lambda output_dir, **kwargs: {"summary": kwargs["summary"], "files": {}})
    monkeypatch.setattr(pipeline, "write_band_power_html", lambda band_power, path: path)
    monkeypatch.setattr(pipeline, "write_trace_preview_html", lambda trace_preview, path: path)
    monkeypatch.setattr(pipeline, "write_spectrum_summary_html", lambda spectrum, path: path)
    monkeypatch.setattr(pipeline, "export_dashboard_data", lambda *args, **kwargs: {})


def test_lfp_disk_cache_created_then_cleaned_up(monkeypatch, tmp_path):
    _wire_minimal_pipeline_stubs(monkeypatch)

    materialize_calls = {}

    def fake_materialize(lfp_recording, cache_dir, **kwargs):
        materialize_calls["cache_dir"] = Path(cache_dir)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        (Path(cache_dir) / "traces.raw").write_bytes(b"")
        return lfp_recording

    monkeypatch.setattr(pipeline, "materialize_lfp", fake_materialize)

    run_analysis(
        AnalysisConfig(
            data_path="data.raw.h5",
            well_id="well004",
            output_dir=str(tmp_path),
            spikeinterface_chunk_duration="",
            cache_lfp_to_disk=True,
            keep_lfp_cache=False,
        )
    )

    cache_dir = materialize_calls["cache_dir"]
    assert cache_dir == tmp_path / ".lfp_cache"
    assert not cache_dir.exists(), "cache dir should be removed when keep_lfp_cache=False"


def test_lfp_disk_cache_kept_when_requested(monkeypatch, tmp_path):
    _wire_minimal_pipeline_stubs(monkeypatch)

    def fake_materialize(lfp_recording, cache_dir, **kwargs):
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        (Path(cache_dir) / "traces.raw").write_bytes(b"")
        return lfp_recording

    monkeypatch.setattr(pipeline, "materialize_lfp", fake_materialize)

    run_analysis(
        AnalysisConfig(
            data_path="data.raw.h5",
            well_id="well004",
            output_dir=str(tmp_path),
            spikeinterface_chunk_duration="",
            cache_lfp_to_disk=True,
            keep_lfp_cache=True,
        )
    )

    assert (tmp_path / ".lfp_cache").exists()


def test_lfp_disk_cache_skipped_when_disabled(monkeypatch, tmp_path):
    _wire_minimal_pipeline_stubs(monkeypatch)

    def fail_materialize(*args, **kwargs):
        raise AssertionError("materialize_lfp should not be called when cache_lfp_to_disk=False")

    monkeypatch.setattr(pipeline, "materialize_lfp", fail_materialize)

    run_analysis(
        AnalysisConfig(
            data_path="data.raw.h5",
            well_id="well004",
            output_dir=str(tmp_path),
            spikeinterface_chunk_duration="",
            cache_lfp_to_disk=False,
        )
    )

    assert not (tmp_path / ".lfp_cache").exists()


class _ResamplingFakeRaw:
    """Stand-in for a Maxwell recording at 20 kHz."""

    def __init__(self):
        self._fs = 20_000.0

    def get_sampling_frequency(self):
        return self._fs


def test_prepare_recordings_resamples_lfp_path_to_target_fs(monkeypatch):
    calls = []

    def fake_unsigned_to_signed(rec):
        calls.append(("unsigned_to_signed",))
        return rec

    def fake_resample(rec, resample_rate, dtype="float32"):
        calls.append(("resample", resample_rate, dtype))
        out = _ResamplingFakeRaw()
        out._fs = float(resample_rate)
        return out

    def fake_bandpass(rec, **kwargs):
        calls.append(("bandpass", float(rec.get_sampling_frequency()), kwargs.get("freq_min"), kwargs.get("freq_max")))
        return rec

    def fake_common_reference(rec, **kwargs):
        calls.append(("common_reference", float(rec.get_sampling_frequency())))
        return rec

    import spikeinterface.preprocessing as spre

    monkeypatch.setattr(spre, "unsigned_to_signed", fake_unsigned_to_signed)
    monkeypatch.setattr(spre, "resample", fake_resample)
    monkeypatch.setattr(spre, "bandpass_filter", fake_bandpass)
    monkeypatch.setattr(spre, "common_reference", fake_common_reference)

    result = recording_module.prepare_recordings(_ResamplingFakeRaw())

    # Resample happens before LFP bandpass, with target 1000 Hz.
    op_names = [c[0] for c in calls]
    assert op_names[0] == "unsigned_to_signed"
    assert op_names[1] == "resample"
    resample_call = calls[1]
    assert resample_call[1] == 1000
    # LFP bandpass operates on the already-decimated signal.
    lfp_bp = next(c for c in calls if c[0] == "bandpass" and c[2] == 0.5)
    assert lfp_bp[1] == 1000.0
    # Spike bandpass still sees the 20 kHz signal.
    spike_bp = next(c for c in calls if c[0] == "bandpass" and c[2] == 300)
    assert spike_bp[1] == 20_000.0
    assert float(result["lfp"].get_sampling_frequency()) == 1000.0
    assert float(result["raw"].get_sampling_frequency()) == 20_000.0


def test_prepare_recordings_skips_resample_when_target_disabled(monkeypatch):
    calls = []
    import spikeinterface.preprocessing as spre

    monkeypatch.setattr(spre, "unsigned_to_signed", lambda rec: rec)
    monkeypatch.setattr(spre, "resample", lambda *a, **kw: calls.append("resample") or (_ for _ in ()).throw(AssertionError("resample should not be called")))
    monkeypatch.setattr(spre, "bandpass_filter", lambda rec, **kw: rec)
    monkeypatch.setattr(spre, "common_reference", lambda rec, **kw: rec)

    recording_module.prepare_recordings(_ResamplingFakeRaw(), lfp_target_fs_hz=None)
    assert calls == []
