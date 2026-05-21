"""Detached subprocess that runs one LFP analysis job.

Invoked by :class:`acute_slice_mea.jobs.JobsBackend` as::

    python -m acute_slice_mea.job_runner --job-dir <dir>

The directory is expected to contain ``job.json`` (immutable spec written by
the dashboard at submit time) and a ``progress.json`` that the dashboard has
already initialised to ``state="running"``. This process owns ``progress.json``
from spawn onward and writes coarse heartbeats at each pipeline phase
boundary. On success it writes ``cache_meta.json`` next to ``manifest.json``
*after* the analysis bundle is fully on disk — that sentinel file is what
``JobsBackend.get_cache_for`` treats as the atomic "this cache is ready"
marker.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback

from acute_slice_mea.jobs import (
    CACHE_META_FILENAME,
    JOB_SPEC_FILENAME,
    PROGRESS_FILENAME,
    PIPELINE_VERSION,
    recording_well_cache_dir,
    write_cache_meta,
)

logger = logging.getLogger(__name__)

# Phase boundaries match the stages inside run_analysis() — keeping the user
# informed without instrumenting deep into each numpy loop.
PHASES = [
    ("loading", 0.05),
    ("preparing", 0.12),
    ("band_power", 0.45),
    ("spectrum", 0.62),
    ("trace_preview", 0.72),
    ("bursts", 0.85),
    ("saving", 0.98),
    ("done", 1.0),
]

_cancel_requested = threading.Event()


def _install_signal_handlers() -> None:
    def _handler(signum, _frame):
        logger.warning("runner received signal %s — requesting cancel", signum)
        _cancel_requested.set()

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def _read_spec(job_dir: Path) -> dict:
    return json.loads((job_dir / JOB_SPEC_FILENAME).read_text())


def _write_progress(job_dir: Path, payload: dict) -> None:
    tmp = job_dir / (PROGRESS_FILENAME + ".tmp")
    final = job_dir / PROGRESS_FILENAME
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, final)


def _merge_existing_progress(job_dir: Path, updates: dict) -> dict:
    """Read whatever the dashboard wrote, then layer our updates on top."""
    path = job_dir / PROGRESS_FILENAME
    base: dict = {}
    if path.exists():
        try:
            base = json.loads(path.read_text())
        except Exception:
            base = {}
    base.update(updates)
    return base


class HeartbeatThread(threading.Thread):
    """Bumps last_heartbeat on a fixed interval so the dashboard can tell we're alive."""

    def __init__(self, job_dir: Path, interval_s: float = 3.0) -> None:
        super().__init__(daemon=True)
        self.job_dir = job_dir
        self.interval_s = interval_s
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            try:
                payload = _merge_existing_progress(
                    self.job_dir, {"last_heartbeat": time.time()}
                )
                _write_progress(self.job_dir, payload)
            except Exception:
                logger.exception("heartbeat write failed")


@contextmanager
def _heartbeat(job_dir: Path):
    hb = HeartbeatThread(job_dir)
    hb.start()
    try:
        yield
    finally:
        hb.stop()


def _phase(job_dir: Path, name: str, progress: float) -> None:
    if _cancel_requested.is_set():
        raise _Cancelled()
    payload = _merge_existing_progress(
        job_dir,
        {
            "state": "running",
            "stage": name,
            "progress": progress,
            "last_heartbeat": time.time(),
            "pid": os.getpid(),
        },
    )
    _write_progress(job_dir, payload)


class _Cancelled(Exception):
    """Raised internally when the runner observed a cancel signal."""


def run(job_dir: Path) -> int:
    spec = _read_spec(job_dir)
    recording_id: str = spec["recording_id"]
    well_id: str = spec["well_id"]
    raw_path: str | None = spec.get("raw_path")
    rec_name: str | None = spec.get("rec_name")
    params: dict = spec.get("params") or {}

    if not raw_path:
        _finalize_failed(job_dir, "Job spec is missing raw_path; cannot run analysis.")
        return 2

    # Resolve cache_root from the job_dir layout: <cache_root>/.jobs/<job_id>/
    cache_root = job_dir.parent.parent
    output_dir = recording_well_cache_dir(cache_root, recording_id, well_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    _install_signal_handlers()

    def _wipe_partial_cache() -> None:
        meta = output_dir / CACHE_META_FILENAME
        if output_dir.exists() and not meta.exists():
            import shutil

            try:
                shutil.rmtree(output_dir)
            except OSError:
                logger.exception("failed to clean partial cache: %s", output_dir)

    started_at = time.time()
    _write_progress(
        job_dir,
        _merge_existing_progress(
            job_dir,
            {
                "state": "running",
                "stage": "starting",
                "progress": 0.0,
                "started_at": started_at,
                "last_heartbeat": started_at,
                "pid": os.getpid(),
            },
        ),
    )

    try:
        with _heartbeat(job_dir):
            _phase(job_dir, "loading", PHASES[0][1])
            # Lazy import — keeps `python -m acute_slice_mea.job_runner --help`
            # cheap and means the heavy spikeinterface stack only loads in
            # the worker process.
            from acute_slice_mea.pipeline import AnalysisConfig, run_analysis

            band = params.get("band") or [0.5, 300.0]
            reference = (params.get("reference") or "CAR").upper()
            # "CAR" historically meant median; preserve that mapping so old
            # caches don't get invalidated. "CMR" is the new explicit median.
            if reference == "NONE":
                apply_lfp_common = False
                lfp_ref_op = "median"
            elif reference == "CMR":
                apply_lfp_common = True
                lfp_ref_op = "median"
            elif reference == "MEAN":
                apply_lfp_common = True
                lfp_ref_op = "mean"
            else:  # "CAR" — historic alias for median
                apply_lfp_common = True
                lfp_ref_op = "median"

            notch_freqs_raw = params.get("notch_freqs") or []
            notch_freqs = [float(f) for f in notch_freqs_raw if f is not None]

            bands_raw = params.get("bands") or {}
            lfp_bands = {
                str(name): (float(low_high[0]), float(low_high[1]))
                for name, low_high in bands_raw.items()
                if isinstance(low_high, (list, tuple)) and len(low_high) == 2
            } or None

            lfp_fs_hz_raw = params.get("lfp_fs_hz", 1000.0)
            lfp_fs_hz = float(lfp_fs_hz_raw) if lfp_fs_hz_raw not in (None, "") else None

            config = AnalysisConfig(
                data_path=str(raw_path),
                well_id=well_id,
                rec_name=rec_name,
                output_dir=str(output_dir),
                lfp_low_hz=float(band[0]),
                lfp_high_hz=float(band[1]),
                apply_lfp_common_reference=apply_lfp_common,
                lfp_reference_operator=lfp_ref_op,
                lfp_target_fs_hz=lfp_fs_hz,
                lfp_notch_freqs=notch_freqs or None,
                lfp_notch_q=float(params.get("notch_q", 30.0)),
                lfp_bands=lfp_bands,
                lfp_window_sec=float(params.get("window_sec", 10.0)),
                lfp_step_sec=float(params.get("step_sec", 5.0)),
                progress=False,
                verbose=False,
            )
            _phase(job_dir, "preparing", PHASES[1][1])

            # Pipeline now reports stage transitions directly via a callback.
            # We map each stage name to its phase progress, fall back to the
            # estimated baseline for unknown names, and clamp into [0, 1] so
            # bad numbers never break the bar.
            stage_progress = {name: pct for name, pct in PHASES}

            def _on_pipeline_stage(stage: str, fraction: float) -> None:
                target = stage_progress.get(stage, fraction)
                target = max(0.0, min(1.0, float(target)))
                _phase(job_dir, stage, target)

            run_analysis(config, progress_callback=_on_pipeline_stage)

            _phase(job_dir, "saving", PHASES[6][1])
            write_cache_meta(output_dir, params=params, hash_=spec.get("hash", ""))
            _finalize_success(job_dir)
            return 0
    except _Cancelled:
        _finalize_cancelled(job_dir, output_dir)
        return 130
    except Exception as exc:  # pragma: no cover - exercised through the runner
        logger.exception("analysis failed")
        _wipe_partial_cache()
        _finalize_failed(job_dir, f"{type(exc).__name__}: {exc}", tb=traceback.format_exc())
        return 1


def _finalize_success(job_dir: Path) -> None:
    now = time.time()
    payload = _merge_existing_progress(
        job_dir,
        {
            "state": "cached",
            "stage": "done",
            "progress": 1.0,
            "ended_at": now,
            "last_heartbeat": now,
            "pid": os.getpid(),
        },
    )
    _write_progress(job_dir, payload)


def _finalize_failed(job_dir: Path, message: str, tb: str | None = None) -> None:
    now = time.time()
    payload = _merge_existing_progress(
        job_dir,
        {
            "state": "failed",
            "error": message,
            "ended_at": now,
            "last_heartbeat": now,
            "pid": os.getpid(),
        },
    )
    _write_progress(job_dir, payload)
    if tb:
        try:
            (job_dir / "error.txt").write_text(tb)
        except OSError:
            pass


def _finalize_cancelled(job_dir: Path, output_dir: Path) -> None:
    now = time.time()
    payload = _merge_existing_progress(
        job_dir,
        {
            "state": "cancelled",
            "error": "Cancelled by user.",
            "ended_at": now,
            "last_heartbeat": now,
            "pid": os.getpid(),
        },
    )
    _write_progress(job_dir, payload)
    # Wipe the half-written cache dir if the sentinel isn't there.
    meta_path = output_dir / CACHE_META_FILENAME
    if output_dir.exists() and not meta_path.exists():
        import shutil

        try:
            shutil.rmtree(output_dir)
        except OSError:
            logger.exception("failed to clean cache dir after cancel: %s", output_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="acute-slice-job-runner",
        description="Run one LFP analysis job (spawned by the trace-viewer dashboard).",
    )
    parser.add_argument("--job-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(args.job_dir)


if __name__ == "__main__":
    raise SystemExit(main())
