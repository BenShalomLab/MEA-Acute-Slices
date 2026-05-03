from acute_slice_mea import cli


def test_compute_cli_enables_progress_by_default_and_parses_verbose(monkeypatch):
    captured = {}

    def fake_run_analysis(config):
        captured["config"] = config
        return {"summary": {}, "files": {}}

    monkeypatch.setattr(cli, "run_analysis", fake_run_analysis)

    exit_code = cli.main(
        [
            "compute",
            "--data-path",
            "data.raw.h5",
            "--well-id",
            "well004",
            "--output-dir",
            "cache",
            "--n-jobs",
            "2",
            "--channel-chunk-size",
            "8",
            "--verbose",
        ]
    )

    assert exit_code == 0
    assert captured["config"].n_jobs == 2
    assert captured["config"].channel_chunk_size == 8
    assert captured["config"].progress is True
    assert captured["config"].verbose is True


def test_compute_cli_can_disable_progress(monkeypatch):
    captured = {}

    def fake_run_analysis(config):
        captured["config"] = config
        return {"summary": {}, "files": {}}

    monkeypatch.setattr(cli, "run_analysis", fake_run_analysis)

    exit_code = cli.main(
        [
            "compute",
            "--data-path",
            "data.raw.h5",
            "--well-id",
            "well004",
            "--output-dir",
            "cache",
            "--no-progress",
        ]
    )

    assert exit_code == 0
    assert captured["config"].progress is False
    assert captured["config"].verbose is False
