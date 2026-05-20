import json
from pathlib import Path
import sys

import h5py
import numpy as np
import spikeinterface as si
from matplotlib.colors import SymLogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hdmea_lfp_viz import main, preprocess
from hdmea_lfp_viz.plots import connectivity


class FakeRawRecording:
    def get_sampling_frequency(self):
        return 10_000.0


class FakeLfpRecording:
    def get_num_channels(self):
        return 3

    def get_sampling_frequency(self):
        return 1_000.0

    def get_num_samples(self):
        return 20_000

    def save(self, **kwargs):
        folder = Path(kwargs["folder"])
        folder.mkdir(parents=True)
        (folder / ".zarray").write_text("{}")


class FakePreprocessingRecording:
    pass


def make_maxwell_wells_file(path, wells):
    with h5py.File(path, "w") as h5:
        wells_group = h5.create_group("wells")
        for well, rec_names in wells.items():
            well_group = wells_group.create_group(well)
            for rec_name in rec_names:
                well_group.create_group(rec_name)


def test_resolve_rec_name_from_single_well_recording(tmp_path):
    raw_path = tmp_path / "data.raw.h5"
    make_maxwell_wells_file(raw_path, {"well004": ["rec0000"]})

    assert preprocess.resolve_rec_name(raw_path, "well004") == "rec0000"


def test_resolve_rec_name_lists_available_wells_when_missing(tmp_path):
    raw_path = tmp_path / "data.raw.h5"
    make_maxwell_wells_file(raw_path, {"well001": ["rec0000"], "well002": ["rec0001"]})

    with np.testing.assert_raises_regex(ValueError, "Available wells: \\['well001', 'well002'\\]"):
        preprocess.resolve_rec_name(raw_path, "well004")


def test_resolve_rec_name_requires_override_for_multiple_recordings(tmp_path):
    raw_path = tmp_path / "data.raw.h5"
    make_maxwell_wells_file(raw_path, {"well004": ["rec0000", "rec0001"]})

    with np.testing.assert_raises_regex(ValueError, "Pass --rec-name"):
        preprocess.resolve_rec_name(raw_path, "well004")


def test_read_maxwell_passes_explicit_rec_name(monkeypatch, tmp_path):
    captured = {}
    raw_path = tmp_path / "data.raw.h5"
    make_maxwell_wells_file(raw_path, {"well004": ["rec0000", "rec0001"]})

    def fake_read_maxwell(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeRawRecording()

    monkeypatch.setattr("spikeinterface.extractors.read_maxwell", fake_read_maxwell)

    recording = preprocess.read_maxwell(raw_path, stream_id="well004", rec_name="rec0001")

    assert isinstance(recording, FakeRawRecording)
    assert captured["args"] == (str(raw_path),)
    assert captured["kwargs"] == {"stream_id": "well004", "rec_name": "rec0001"}


def test_build_lfp_preprocessing_defaults_to_current_notch_settings(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "spikeinterface.preprocessing.bandpass_filter",
        lambda recording, **kwargs: calls.append(("bandpass", kwargs)) or recording,
    )
    monkeypatch.setattr(
        "spikeinterface.preprocessing.notch_filter",
        lambda recording, **kwargs: calls.append(("notch", kwargs)) or recording,
    )
    monkeypatch.setattr(
        "spikeinterface.preprocessing.common_reference",
        lambda recording, **kwargs: calls.append(("reference", kwargs)) or recording,
    )
    monkeypatch.setattr(
        "spikeinterface.preprocessing.resample",
        lambda recording, **kwargs: calls.append(("resample", kwargs)) or recording,
    )

    preprocess.build_lfp_preprocessing(FakePreprocessingRecording())

    notch_calls = [kwargs for name, kwargs in calls if name == "notch"]
    assert [call["freq"] for call in notch_calls] == [60.0, 120.0, 180.0]
    assert [call["q"] for call in notch_calls] == [30.0, 30.0, 30.0]


def test_build_lfp_preprocessing_supports_custom_and_disabled_notches(monkeypatch):
    calls = []

    monkeypatch.setattr("spikeinterface.preprocessing.bandpass_filter", lambda recording, **kwargs: recording)
    monkeypatch.setattr(
        "spikeinterface.preprocessing.notch_filter",
        lambda recording, **kwargs: calls.append(kwargs) or recording,
    )
    monkeypatch.setattr("spikeinterface.preprocessing.common_reference", lambda recording, **kwargs: recording)
    monkeypatch.setattr("spikeinterface.preprocessing.resample", lambda recording, **kwargs: recording)

    preprocess.build_lfp_preprocessing(FakePreprocessingRecording(), notch_frequencies=(50, 100), notch_q=45)
    assert calls == [
        {"freq": 50.0, "q": 45, "dtype": "float32"},
        {"freq": 100.0, "q": 45, "dtype": "float32"},
    ]

    calls.clear()
    preprocess.build_lfp_preprocessing(FakePreprocessingRecording(), notch_frequencies=())
    assert calls == []


def test_save_lfp_cache_configures_spikeinterface_before_preprocessing(monkeypatch, tmp_path):
    calls = []
    raw_path = tmp_path / "data.raw.h5"
    make_maxwell_wells_file(raw_path, {"well004": ["rec0000"]})

    monkeypatch.setattr(preprocess, "read_maxwell", lambda *args, **kwargs: FakeRawRecording())
    monkeypatch.setattr(si, "set_global_job_kwargs", lambda **kwargs: calls.append(("configure", kwargs)))

    def fake_build_lfp_preprocessing(recording, **kwargs):
        calls.append(("build_lfp_preprocessing", recording, kwargs))
        return FakeLfpRecording()

    monkeypatch.setattr(preprocess, "build_lfp_preprocessing", fake_build_lfp_preprocessing)

    zarr_path = preprocess.save_lfp_cache(
        raw_path,
        cache_dir=tmp_path / "cache",
        stream_id="well004",
        n_jobs=4,
    )

    assert zarr_path == tmp_path / "cache" / "lfp_1khz.zarr"
    assert calls[0] == (
        "configure",
        {"chunk_duration": preprocess.DEFAULT_SPIKEINTERFACE_CHUNK_DURATION, "n_jobs": 4},
    )
    assert calls[1][0] == "build_lfp_preprocessing"
    assert calls[1][2] == {"notch_frequencies": (60.0, 120.0, 180.0), "notch_q": 30.0}

    metadata = json.loads((tmp_path / "cache" / "lfp_1khz_metadata.json").read_text())
    assert metadata["zarr_chunks_samples_channels"] == [10_000, 3]
    assert metadata["spikeinterface_chunk_duration"] == "60s"
    assert metadata["stream_id"] == "well004"
    assert metadata["rec_name"] == "rec0000"
    assert metadata["preprocessing"]["notch_hz"] == [60.0, 120.0, 180.0]
    assert metadata["preprocessing"]["notch_q"] == 30.0


def test_hdmea_lfp_viz_cli_passes_rec_name_and_spikeinterface_chunk_duration(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        "sys.argv",
        [
            "hdmea_lfp_viz",
            "data.raw.h5",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--figures-dir",
            str(tmp_path / "figures"),
            "--stream-id",
            "well004",
            "--rec-name",
            "rec0000",
            "--n-jobs",
            "4",
            "--spikeinterface-chunk-duration",
            "120s",
            "--notch-frequencies",
            "50,100",
            "--notch-q",
            "45",
        ],
    )

    def fake_save_lfp_cache(*args, **kwargs):
        captured["save_lfp_cache"] = kwargs
        return tmp_path / "cache" / "lfp_1khz.zarr"

    monkeypatch.setattr(main, "apply_style", lambda: None)
    monkeypatch.setattr(main.preprocess, "save_lfp_cache", fake_save_lfp_cache)
    monkeypatch.setattr(main.preprocess, "load_lfp_cache", lambda cache_dir: object())
    monkeypatch.setattr(main, "save_summaries", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "load_summaries", lambda cache_dir: {"rms_per_channel": np.array([1.0, 2.0])})
    monkeypatch.setattr(main, "generate_figures", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        main,
        "recording_metadata",
        lambda recording: {"n_channels": 2, "duration_s": 20.0, "sampling_frequency_hz": 1000.0},
    )

    main.main()

    assert captured["save_lfp_cache"]["spikeinterface_chunk_duration"] == "120s"
    assert captured["save_lfp_cache"]["rec_name"] == "rec0000"
    assert captured["save_lfp_cache"]["notch_frequencies"] == (50.0, 100.0)
    assert captured["save_lfp_cache"]["notch_q"] == 45
    assert captured["save_lfp_cache"]["n_jobs"] == 4


def test_hdmea_lfp_viz_cli_can_disable_notch_filter(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        "sys.argv",
        [
            "hdmea_lfp_viz",
            "data.raw.h5",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--figures-dir",
            str(tmp_path / "figures"),
            "--stream-id",
            "well004",
            "--disable-notch-filter",
        ],
    )

    monkeypatch.setattr(main, "apply_style", lambda: None)
    monkeypatch.setattr(
        main.preprocess,
        "save_lfp_cache",
        lambda *args, **kwargs: captured.setdefault("save_lfp_cache", kwargs) or tmp_path / "cache" / "lfp_1khz.zarr",
    )
    monkeypatch.setattr(main.preprocess, "load_lfp_cache", lambda cache_dir: object())
    monkeypatch.setattr(main, "save_summaries", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "load_summaries", lambda cache_dir: {"rms_per_channel": np.array([1.0, 2.0])})
    monkeypatch.setattr(main, "generate_figures", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        main,
        "recording_metadata",
        lambda recording: {"n_channels": 2, "duration_s": 20.0, "sampling_frequency_hz": 1000.0},
    )

    main.main()

    assert captured["save_lfp_cache"]["notch_frequencies"] == ()


def test_correlation_matrix_uses_signed_log_scale(monkeypatch, tmp_path):
    captured = {}

    def fake_save_figure(fig, figures_dir, stem):
        captured["fig"] = fig
        captured["figures_dir"] = figures_dir
        captured["stem"] = stem

    monkeypatch.setattr(connectivity, "save_figure", fake_save_figure)
    corr = np.array(
        [
            [1.0, 0.4, 0.02, -0.2],
            [0.4, 1.0, 0.03, -0.1],
            [0.02, 0.03, 1.0, -0.5],
            [-0.2, -0.1, -0.5, 1.0],
        ],
        dtype=np.float32,
    )

    connectivity.plot_correlation_matrix({"corr_matrix": corr}, tmp_path)

    image_axes = [ax for ax in captured["fig"].axes if ax.images]
    assert image_axes
    norm = image_axes[0].images[0].norm
    assert isinstance(norm, SymLogNorm)
    assert norm.linthresh == 0.02

    colorbar_axes = [ax for ax in captured["fig"].axes if ax.get_ylabel() == "Pearson r"]
    assert colorbar_axes
    np.testing.assert_allclose(colorbar_axes[0].get_yticks(), connectivity.CORRELATION_LOG_TICKS)
    assert captured["stem"] == "05_correlation_matrix"
