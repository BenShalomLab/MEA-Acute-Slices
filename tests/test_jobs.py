"""Tests for :class:`acute_slice_mea.jobs.JobsBackend`.

The runner subprocess is replaced with a tiny shell script so we can drive the
state machine end-to-end (submit → run → cached / failed / cancelled / reaped)
without dragging in the spikeinterface pipeline.
"""

from __future__ import annotations

import json
import os
import textwrap
import time
from pathlib import Path

import pytest

from acute_slice_mea import jobs as jobs_mod
from acute_slice_mea.jobs import (
    CACHE_META_FILENAME,
    JobsBackend,
    PROGRESS_FILENAME,
    params_hash,
    recording_well_cache_dir,
    write_cache_meta,
)


RECORDING_ID = "CX138/260329/T003346/Network/000029"
WELL_ID = "well000"


def _make_fake_runner(tmp_path: Path, *, sleep_s: float = 0.3, succeed: bool = True) -> list[str]:
    """Build a shell command that pretends to be ``acute_slice_mea.job_runner``."""
    script = tmp_path / "fake_runner.py"
    cache_root = tmp_path  # we'll resolve relative inside the script
    body = textwrap.dedent(
        f"""
        import argparse, json, os, time, sys
        from pathlib import Path

        parser = argparse.ArgumentParser()
        parser.add_argument("--job-dir", required=True, type=Path)
        args = parser.parse_args()

        spec = json.loads((args.job_dir / "job.json").read_text())
        progress_path = args.job_dir / "progress.json"

        def write(payload):
            base = {{}}
            if progress_path.exists():
                try: base = json.loads(progress_path.read_text())
                except Exception: pass
            base.update(payload)
            tmp = progress_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(base))
            os.replace(tmp, progress_path)

        now = time.time()
        write({{"state": "running", "pid": os.getpid(), "started_at": now, "last_heartbeat": now, "progress": 0.1}})
        time.sleep({sleep_s})
        if not {succeed!r}:
            write({{"state": "failed", "error": "simulated failure", "ended_at": time.time()}})
            sys.exit(1)

        cache_root = args.job_dir.parent.parent
        rid = spec["recording_id"]
        wid = spec["well_id"]
        out = Path(cache_root, *rid.split("/"), wid)
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(json.dumps({{"summary": {{}}, "files": {{}}}}))
        meta = {{
            "params": spec.get("params"),
            "params_hash": spec.get("hash"),
            "pipeline": (spec.get("params") or {{}}).get("pipeline"),
            "created_at": "now",
        }}
        (out / "cache_meta.json").write_text(json.dumps(meta))
        write({{"state": "cached", "progress": 1.0, "ended_at": time.time(), "last_heartbeat": time.time()}})
        """
    )
    script.write_text(body)
    return ["python3", str(script)]


def _wait_for_state(backend: JobsBackend, recording_id: str, well_id: str, target: str, timeout_s: float = 5.0) -> str:
    """Poll until backend reports the desired status (or any terminal state)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        backend.refresh()
        status = backend.get_status_for(recording_id, well_id)
        if status == target:
            return status
        if status in ("failed", "cancelled") and target == "cached":
            return status  # surface unexpected terminal state for the assertion
        time.sleep(0.05)
    backend.refresh()
    return backend.get_status_for(recording_id, well_id)


def test_submit_runs_to_completion_writes_cache_meta(tmp_path):
    runner = _make_fake_runner(tmp_path, sleep_s=0.2, succeed=True)
    backend = JobsBackend(tmp_path, runner_command=runner)
    result = backend.submit(
        recording_id=RECORDING_ID,
        well_id=WELL_ID,
        raw_path="/dev/null",
        params={"band": [0.5, 300.0]},
    )
    assert "job_id" in result
    status = _wait_for_state(backend, RECORDING_ID, WELL_ID, "cached")
    assert status == "cached"
    cache_dir = recording_well_cache_dir(tmp_path, RECORDING_ID, WELL_ID)
    assert (cache_dir / CACHE_META_FILENAME).exists()


def test_submit_rejects_when_cache_exists_without_overwrite(tmp_path):
    backend = JobsBackend(tmp_path, runner_command=_make_fake_runner(tmp_path))
    cache_dir = recording_well_cache_dir(tmp_path, RECORDING_ID, WELL_ID)
    cache_dir.mkdir(parents=True)
    (cache_dir / "manifest.json").write_text("{}")
    write_cache_meta(cache_dir, params={"band": [0.5, 300]}, hash_=params_hash({"band": [0.5, 300]}))
    result = backend.submit(
        recording_id=RECORDING_ID,
        well_id=WELL_ID,
        raw_path="/dev/null",
    )
    assert result == {"error": "cache-exists", "cache_hash": result.get("cache_hash")}
    assert result["cache_hash"] is not None


def test_overwrite_drops_old_cache_before_running(tmp_path):
    backend = JobsBackend(tmp_path, runner_command=_make_fake_runner(tmp_path, sleep_s=0.1))
    cache_dir = recording_well_cache_dir(tmp_path, RECORDING_ID, WELL_ID)
    cache_dir.mkdir(parents=True)
    (cache_dir / "manifest.json").write_text("{}")
    write_cache_meta(cache_dir, params={"band": [1.0, 250]}, hash_="oldhsh")
    result = backend.submit(
        recording_id=RECORDING_ID,
        well_id=WELL_ID,
        raw_path="/dev/null",
        params={"band": [0.5, 300]},
        overwrite=True,
    )
    assert "job_id" in result
    _wait_for_state(backend, RECORDING_ID, WELL_ID, "cached")
    meta = json.loads((cache_dir / CACHE_META_FILENAME).read_text())
    assert meta["params"]["band"] == [0.5, 300]


def test_failed_runner_is_reported_as_failed(tmp_path):
    backend = JobsBackend(
        tmp_path,
        runner_command=_make_fake_runner(tmp_path, sleep_s=0.1, succeed=False),
    )
    backend.submit(recording_id=RECORDING_ID, well_id=WELL_ID, raw_path="/dev/null")
    status = _wait_for_state(backend, RECORDING_ID, WELL_ID, "failed")
    assert status == "failed"


def test_reconcile_reaps_stale_running_jobs(tmp_path):
    """Backend constructed with a running-but-stale job on disk reaps it."""
    job_id = "job_stale"
    job_dir = tmp_path / jobs_mod.JOBS_DIRNAME / job_id
    job_dir.mkdir(parents=True)
    spec = {
        "job_id": job_id,
        "recording_id": RECORDING_ID,
        "well_id": WELL_ID,
        "params": {"pipeline": jobs_mod.PIPELINE_VERSION},
        "hash": "abc123",
        "submitted_at": time.time() - 600,
        "raw_path": "/dev/null",
    }
    (job_dir / "job.json").write_text(json.dumps(spec))
    progress = {
        "state": "running",
        "progress": 0.5,
        "started_at": time.time() - 500,
        "last_heartbeat": time.time() - 200,  # well past HEARTBEAT_STALE_S
        "pid": 999_999,                       # unlikely-to-exist pid
    }
    (job_dir / "progress.json").write_text(json.dumps(progress))
    # Also stage a half-written cache dir to confirm it's cleaned.
    cache_dir = recording_well_cache_dir(tmp_path, RECORDING_ID, WELL_ID)
    cache_dir.mkdir(parents=True)
    (cache_dir / "manifest.json").write_text("{}")  # no cache_meta.json sidecar

    backend = JobsBackend(tmp_path, runner_command=_make_fake_runner(tmp_path))
    assert backend.get_status_for(RECORDING_ID, WELL_ID) == "failed"
    assert not cache_dir.exists()  # half-written cache wiped


def test_submit_rejects_when_already_in_flight(tmp_path):
    backend = JobsBackend(tmp_path, runner_command=_make_fake_runner(tmp_path, sleep_s=1.0))
    first = backend.submit(recording_id=RECORDING_ID, well_id=WELL_ID, raw_path="/dev/null")
    assert "job_id" in first
    second = backend.submit(recording_id=RECORDING_ID, well_id=WELL_ID, raw_path="/dev/null")
    assert second.get("error") == "already-in-flight"
    backend.cancel(first["job_id"])


def test_params_hash_is_canonical():
    a = params_hash({"band": [0.5, 300], "reference": "CAR"})
    b = params_hash({"reference": "CAR", "band": [0.5, 300]})
    assert a == b
    c = params_hash({"band": [0.5, 250], "reference": "CAR"})
    assert a != c
