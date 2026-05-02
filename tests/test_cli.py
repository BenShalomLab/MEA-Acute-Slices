from acute_slice_mea import cli


def test_compute_cli_parses_parallel_options(monkeypatch):
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
        ]
    )

    assert exit_code == 0
    assert captured["config"].n_jobs == 2
    assert captured["config"].channel_chunk_size == 8
