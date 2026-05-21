"""Out-of-process LFP analysis job runner.

Submitting an analysis writes a small state directory under
``<cache_root>/.jobs/<job_id>/`` (``job.json`` immutable + ``progress.json``
heartbeat) and spawns a detached subprocess that imports
:func:`acute_slice_mea.pipeline.run_analysis`. The detached subprocess survives
the dashboard process going away (closing the browser tab, restarting Dash);
on the next dashboard startup we reconcile by checking each running job's PID
and heartbeat freshness.

Design rules ported from the feature plan (``LFP Cache · Feature Plan.html``):
    * One cache entry per (recording, well). Re-running with new params
      requires the user to tick "Overwrite previous result"; the old cache is
      removed before the new job starts.
    * Pipeline is versioned (``lfp@v1``); bumping the version invalidates
      caches via the params hash.
    * No "partial" state. A running job whose heartbeat goes stale is reaped
      to "failed" and its half-written cache directory is cleaned.
    * Concurrency is capped at 1 worker for v1. The slider in the UI ships now
      so the public API doesn't shift when the cap lifts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "lfp@v1"
HEARTBEAT_STALE_S = 30.0  # subprocess writes heartbeat ~every 2-5 s
JOBS_DIRNAME = ".jobs"
CACHE_META_FILENAME = "cache_meta.json"
RUNNER_LOG_FILENAME = "runner.log"
JOB_SPEC_FILENAME = "job.json"
PROGRESS_FILENAME = "progress.json"

# Terminal states stay on disk so the drawer can show recent history.
TERMINAL_STATES = {"cached", "failed", "cancelled"}
ACTIVE_STATES = {"queued", "running"}

DEFAULT_PARAMS: dict[str, Any] = {
    "band": [0.5, 300.0],
    "reference": "CAR",
    "channels": "routed64",
    "decimation": 1,
    "pipeline": PIPELINE_VERSION,
}


# =============================================================================
# Hashing
# =============================================================================


def _canonical(obj: Any) -> str:
    """Stable JSON serialization for hashing."""
    if isinstance(obj, list):
        return "[" + ",".join(_canonical(x) for x in obj) + "]"
    if isinstance(obj, dict):
        keys = sorted(obj.keys())
        return "{" + ",".join(json.dumps(k) + ":" + _canonical(obj[k]) for k in keys) + "}"
    return json.dumps(obj)


def params_hash(params: dict) -> str:
    """Truncated sha1 of canonical params JSON — matches the design's identity."""
    return hashlib.sha1(_canonical(params).encode("utf-8")).hexdigest()[:6]


def merge_params(params: dict | None) -> dict:
    """User params overlaid on defaults; pipeline version forced to current."""
    merged = dict(DEFAULT_PARAMS)
    if params:
        merged.update(params)
    merged["pipeline"] = PIPELINE_VERSION
    return merged


# =============================================================================
# On-disk cache helpers (used both by JobsBackend and job_runner)
# =============================================================================


def recording_well_cache_dir(cache_root: Path, recording_id: str, well_id: str) -> Path:
    """Cache directory for one (recording, well).

    ``recording_id`` is ``sample/date/plate/scan/run`` (matches LibraryIndex);
    we append ``well_id`` to land in the same layout that
    ``scripts/build_viewer_cache.py`` writes to. Multi-rec recordings encode
    ``rec_name`` as a trailing segment of ``recording_id``, so the split
    already lands us in the right sub-folder.
    """
    return Path(cache_root, *recording_id.split("/"), well_id)


def read_cache_meta(cache_dir: Path) -> dict | None:
    """Read cache_meta.json sidecar if present.

    Missing sidecar → cache is from a pre-spawn-feature run; we still report
    'cached' but freshness can't be compared.
    """
    meta_path = Path(cache_dir) / CACHE_META_FILENAME
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        logger.exception("failed to read cache meta: %s", meta_path)
        return None


def write_cache_meta(cache_dir: Path, *, params: dict, hash_: str) -> None:
    """Write the atomic 'cache is ready' sentinel, after manifest is on disk."""
    cache_dir = Path(cache_dir)
    payload = {
        "params": params,
        "params_hash": hash_,
        "pipeline": params.get("pipeline", PIPELINE_VERSION),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    tmp_path = cache_dir / (CACHE_META_FILENAME + ".tmp")
    final_path = cache_dir / CACHE_META_FILENAME
    tmp_path.write_text(json.dumps(payload, indent=2))
    os.replace(tmp_path, final_path)


def cache_size_bytes(cache_dir: Path) -> int:
    total = 0
    for path in Path(cache_dir).rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                pass
    return total


# =============================================================================
# Job model
# =============================================================================


@dataclass
class Job:
    id: str
    recording_id: str
    well_id: str
    params: dict
    hash: str
    state: str  # 'queued' | 'running' | 'cached' | 'failed' | 'cancelled'
    progress: float = 0.0
    submitted_at: float = 0.0
    started_at: float | None = None
    ended_at: float | None = None
    last_heartbeat: float | None = None
    pid: int | None = None
    stage: str | None = None
    error: str | None = None
    raw_path: str | None = None
    rec_name: str | None = None
    recording_label: str | None = None
    well_label: str | None = None

    def to_spec(self) -> dict:
        return {
            "job_id": self.id,
            "recording_id": self.recording_id,
            "well_id": self.well_id,
            "rec_name": self.rec_name,
            "params": self.params,
            "hash": self.hash,
            "submitted_at": self.submitted_at,
            "raw_path": self.raw_path,
            "recording_label": self.recording_label,
            "well_label": self.well_label,
        }

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "recording_id": self.recording_id,
            "well_id": self.well_id,
            "rec_name": self.rec_name,
            "params": self.params,
            "hash": self.hash,
            "state": self.state,
            "progress": self.progress,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "last_heartbeat": self.last_heartbeat,
            "pid": self.pid,
            "stage": self.stage,
            "error": self.error,
            "recording_label": self.recording_label,
            "well_label": self.well_label,
        }


# =============================================================================
# Backend
# =============================================================================


@dataclass
class WorkersInfo:
    cap: int
    max: int


class JobsBackend:
    """Owns the job state directory and (de)spawns runner subprocesses.

    There is exactly one of these per Dash app process. Methods are safe to
    call from Dash callback threads.
    """

    def __init__(
        self,
        cache_root: str | Path,
        *,
        runner_command: list[str] | None = None,
        worker_cap: int = 1,
        worker_max: int = 1,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.jobs_root = self.cache_root / JOBS_DIRNAME
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        # Allow tests to substitute a fake runner.
        self._runner_command = runner_command or [
            sys.executable,
            "-m",
            "acute_slice_mea.job_runner",
        ]
        self.workers = WorkersInfo(cap=max(1, worker_cap), max=max(1, worker_max))
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._hydrate()
        self._reconcile()
        self._maybe_start_next()

    # ─── hydration / reconciliation ─────────────────────────────────────────

    def _hydrate(self) -> None:
        """Load any job dirs already present on disk into memory."""
        if not self.jobs_root.exists():
            return
        for job_dir in self.jobs_root.iterdir():
            if not job_dir.is_dir():
                continue
            job = _load_job(job_dir)
            if job is not None:
                self._jobs[job.id] = job

    def _reconcile(self) -> None:
        """Reap orphaned running jobs.

        A running job is healthy iff its pid is alive AND its heartbeat is
        younger than ``HEARTBEAT_STALE_S``. Anything else gets promoted to
        ``failed`` and its half-written cache directory cleaned, per the
        "no partial state" rule.
        """
        now = time.time()
        for job in list(self._jobs.values()):
            if job.state != "running":
                continue
            alive = job.pid is not None and _pid_alive(job.pid)
            heartbeat_age = now - (job.last_heartbeat or job.started_at or now)
            if alive and heartbeat_age <= HEARTBEAT_STALE_S:
                continue  # still healthy

            # Reap.
            job.state = "failed"
            job.ended_at = now
            job.error = (
                "Worker did not check in (interrupted). "
                "Cache directory cleaned."
            )
            self._clean_partial_cache(job)
            _save_progress(self._job_dir(job.id), job)

    def _clean_partial_cache(self, job: Job) -> None:
        cache_dir = recording_well_cache_dir(
            self.cache_root, job.recording_id, job.well_id
        )
        meta_path = cache_dir / CACHE_META_FILENAME
        if cache_dir.exists() and not meta_path.exists():
            # No sentinel → half-written. Wipe the directory.
            try:
                shutil.rmtree(cache_dir)
            except OSError:
                logger.exception("failed to clean partial cache: %s", cache_dir)

    # ─── scheduling ─────────────────────────────────────────────────────────

    def _maybe_start_next(self) -> None:
        """Spawn the next queued job if we are below the worker cap."""
        running = sum(1 for j in self._jobs.values() if j.state == "running")
        if running >= self.workers.cap:
            return
        queued = sorted(
            (j for j in self._jobs.values() if j.state == "queued"),
            key=lambda j: j.submitted_at,
        )
        if not queued:
            return
        next_job = queued[0]
        self._spawn(next_job)

    def _spawn(self, job: Job) -> None:
        job_dir = self._job_dir(job.id)
        log_path = job_dir / RUNNER_LOG_FILENAME
        cmd = [*self._runner_command, "--job-dir", str(job_dir)]
        logger.info("spawning job %s: %s", job.id, " ".join(cmd))
        log_fh = open(log_path, "ab", buffering=0)
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,  # decouple from dashboard process group
                cwd=str(self.cache_root),
            )
        finally:
            log_fh.close()
        now = time.time()
        job.state = "running"
        job.pid = proc.pid
        job.started_at = now
        job.last_heartbeat = now
        job.progress = 0.0
        job.stage = "starting"
        _save_progress(job_dir, job)

    # ─── public API: submit / cancel / retry / cache ────────────────────────

    def submit(
        self,
        *,
        recording_id: str,
        well_id: str,
        raw_path: str | None,
        rec_name: str | None = None,
        params: dict | None = None,
        overwrite: bool = False,
        recording_label: str | None = None,
        well_label: str | None = None,
    ) -> dict:
        """POST /analyses — submit one job. Returns {job_id, hash} or {error}."""
        with self._lock:
            existing = self._find_active(recording_id, well_id)
            if existing is not None:
                return {"error": "already-in-flight", "job_id": existing.id}

            cache = self.get_cache_for(recording_id, well_id)
            if cache is not None and not overwrite:
                return {"error": "cache-exists", "cache_hash": cache.get("params_hash")}

            merged = merge_params(params)
            hash_ = params_hash(merged)

            if cache is not None and overwrite:
                cache_dir = recording_well_cache_dir(
                    self.cache_root, recording_id, well_id
                )
                try:
                    shutil.rmtree(cache_dir)
                except OSError:
                    logger.exception("overwrite: failed to drop cache %s", cache_dir)

            job_id = "job_" + uuid.uuid4().hex[:10]
            now = time.time()
            job = Job(
                id=job_id,
                recording_id=recording_id,
                well_id=well_id,
                params=merged,
                hash=hash_,
                state="queued",
                progress=0.0,
                submitted_at=now,
                raw_path=raw_path,
                rec_name=rec_name,
                recording_label=recording_label,
                well_label=well_label,
            )
            self._jobs[job_id] = job
            job_dir = self._job_dir(job_id)
            job_dir.mkdir(parents=True, exist_ok=True)
            (job_dir / JOB_SPEC_FILENAME).write_text(json.dumps(job.to_spec(), indent=2))
            _save_progress(job_dir, job)

            self._maybe_start_next()
            return {"job_id": job_id, "hash": hash_}

    def cancel(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state in TERMINAL_STATES:
                return
            was_running = job.state == "running"
            job.state = "cancelled"
            job.ended_at = time.time()
            job.error = "Cancelled by user."
            _save_progress(self._job_dir(job_id), job)
            if was_running and job.pid is not None:
                _terminate_process(job.pid)
                self._clean_partial_cache(job)
            self._maybe_start_next()

    def retry(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            # Re-submit with the same recording + params + overwrite (in case a
            # partial cache slipped through).
            return self.submit(
                recording_id=job.recording_id,
                well_id=job.well_id,
                raw_path=job.raw_path,
                rec_name=job.rec_name,
                params=job.params,
                overwrite=True,
                recording_label=job.recording_label,
                well_label=job.well_label,
            )

    def invalidate_cache(self, recording_id: str, well_id: str) -> None:
        with self._lock:
            cache_dir = recording_well_cache_dir(
                self.cache_root, recording_id, well_id
            )
            if cache_dir.exists():
                try:
                    shutil.rmtree(cache_dir)
                except OSError:
                    logger.exception("failed to invalidate cache: %s", cache_dir)

    def set_worker_cap(self, n: int) -> int:
        with self._lock:
            n = max(1, min(self.workers.max, int(n)))
            self.workers.cap = n
            self._maybe_start_next()
            return n

    # ─── public API: queries ────────────────────────────────────────────────

    def refresh(self) -> None:
        """Re-read every job's progress.json + re-check the worker cap.

        Cheap to call from the dashboard poll. Re-hydrating from disk is what
        lets us pick up writes from the runner subprocess.
        """
        with self._lock:
            # Re-load every job's progress file. We deliberately don't trust
            # the runner-side `state` if our side already moved on (e.g., a
            # cancel took priority).
            for job_id, job in list(self._jobs.items()):
                if job.state in TERMINAL_STATES:
                    continue
                fresh = _load_job(self._job_dir(job_id))
                if fresh is None:
                    continue
                # If our local state is more authoritative (we marked cancelled
                # while the runner was still writing 'running'), keep ours.
                if job.state == "cancelled":
                    continue
                self._jobs[job_id] = fresh
            self._reconcile()
            self._maybe_start_next()

    def get_jobs_for(self, recording_id: str, well_id: str | None = None) -> list[Job]:
        with self._lock:
            return [
                j for j in self._jobs.values()
                if j.recording_id == recording_id
                and (well_id is None or j.well_id == well_id)
            ]

    def get_active_jobs(self) -> list[Job]:
        with self._lock:
            return sorted(
                (j for j in self._jobs.values() if j.state in ACTIVE_STATES),
                key=lambda j: j.submitted_at,
            )

    def get_recent_jobs(self, max_age_s: float = 24 * 3600) -> list[Job]:
        cutoff = time.time() - max_age_s
        with self._lock:
            return sorted(
                (
                    j for j in self._jobs.values()
                    if j.state in TERMINAL_STATES
                    and (j.ended_at or j.submitted_at) >= cutoff
                ),
                key=lambda j: -(j.ended_at or 0),
            )

    def get_cache_for(self, recording_id: str, well_id: str) -> dict | None:
        """Return cache_meta.json contents (or a synthesized stub for legacy caches)."""
        cache_dir = recording_well_cache_dir(self.cache_root, recording_id, well_id)
        meta = read_cache_meta(cache_dir)
        if meta is not None:
            meta = dict(meta)
            meta["size_bytes"] = cache_size_bytes(cache_dir)
            return meta
        # Legacy cache (manifest.json from build_viewer_cache.py with no sidecar).
        manifest = cache_dir / "manifest.json"
        if manifest.exists():
            return {
                "params": None,
                "params_hash": None,
                "pipeline": None,
                "created_at": None,
                "legacy": True,
                "size_bytes": cache_size_bytes(cache_dir),
            }
        return None

    def get_cache_total(self) -> dict:
        bytes_ = 0
        count = 0
        # Walk only one level past the cache root to find well-leaf dirs cheaply.
        # Cache layout is sample/date/plate/scan/run/well — six levels. We
        # scan the union of (job-known) cache dirs + a generic walk for legacy.
        seen: set[Path] = set()
        for job in list(self._jobs.values()):
            cache_dir = recording_well_cache_dir(
                self.cache_root, job.recording_id, job.well_id
            )
            if cache_dir in seen:
                continue
            seen.add(cache_dir)
            if (cache_dir / CACHE_META_FILENAME).exists() or (cache_dir / "manifest.json").exists():
                bytes_ += cache_size_bytes(cache_dir)
                count += 1
        # Note: any cache discovered via LibraryIndex but never run through this
        # backend won't be counted here. That's fine for the v1 readout — we
        # just want a sense of "what this dashboard has produced".
        return {"bytes": bytes_, "count": count}

    def get_status_for(
        self,
        recording_id: str,
        well_id: str,
        current_params: dict | None = None,
    ) -> str:
        """Derive a single label per (recording, well) for the row pill.

        Returns one of: 'cached' | 'stale' | 'queued' | 'running' | 'failed' |
        'idle'.
        """
        with self._lock:
            active = self._find_active(recording_id, well_id)
            if active is not None:
                return active.state  # 'queued' or 'running'

            cache = self.get_cache_for(recording_id, well_id)
            if cache is not None:
                if cache.get("legacy"):
                    return "cached"  # legacy bundles can't be compared by hash
                if (cache.get("pipeline") or PIPELINE_VERSION) != PIPELINE_VERSION:
                    return "stale"
                if current_params is not None:
                    want = params_hash(merge_params(current_params))
                    if want != cache.get("params_hash"):
                        return "stale"
                return "cached"

            failed = next(
                (
                    j for j in self._jobs.values()
                    if j.recording_id == recording_id
                    and j.well_id == well_id
                    and j.state == "failed"
                ),
                None,
            )
            if failed is not None:
                return "failed"
            return "idle"

    # ─── internals ──────────────────────────────────────────────────────────

    def _job_dir(self, job_id: str) -> Path:
        return self.jobs_root / job_id

    def _find_active(self, recording_id: str, well_id: str) -> Job | None:
        return next(
            (
                j for j in self._jobs.values()
                if j.recording_id == recording_id
                and j.well_id == well_id
                and j.state in ACTIVE_STATES
            ),
            None,
        )

    # ─── debug ──────────────────────────────────────────────────────────────

    def debug_reset(self) -> None:
        """Wipe ALL job + cache state. Mirrors the design's debug button."""
        with self._lock:
            for job in list(self._jobs.values()):
                if job.state == "running" and job.pid is not None:
                    _terminate_process(job.pid)
            self._jobs.clear()
            if self.jobs_root.exists():
                try:
                    shutil.rmtree(self.jobs_root)
                except OSError:
                    logger.exception("failed to wipe jobs dir")
            self.jobs_root.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Module-private helpers
# =============================================================================


def _save_progress(job_dir: Path, job: Job) -> None:
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    tmp = job_dir / (PROGRESS_FILENAME + ".tmp")
    final = job_dir / PROGRESS_FILENAME
    tmp.write_text(json.dumps(job.to_dict(), indent=2))
    os.replace(tmp, final)


def _load_job(job_dir: Path) -> Job | None:
    spec_path = job_dir / JOB_SPEC_FILENAME
    progress_path = job_dir / PROGRESS_FILENAME
    if not spec_path.exists() or not progress_path.exists():
        return None
    try:
        spec = json.loads(spec_path.read_text())
        progress = json.loads(progress_path.read_text())
    except Exception:
        logger.exception("failed to load job dir: %s", job_dir)
        return None
    return Job(
        id=spec.get("job_id", job_dir.name),
        recording_id=spec["recording_id"],
        well_id=spec["well_id"],
        params=spec.get("params") or {},
        hash=spec.get("hash") or "",
        state=progress.get("state", "failed"),
        progress=float(progress.get("progress") or 0.0),
        submitted_at=float(spec.get("submitted_at") or progress.get("submitted_at") or 0.0),
        started_at=progress.get("started_at"),
        ended_at=progress.get("ended_at"),
        last_heartbeat=progress.get("last_heartbeat"),
        pid=progress.get("pid"),
        stage=progress.get("stage"),
        error=progress.get("error"),
        raw_path=spec.get("raw_path"),
        rec_name=spec.get("rec_name"),
        recording_label=spec.get("recording_label"),
        well_label=spec.get("well_label"),
    )


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Another user owns the process — it exists.
        return True
    return True


def _terminate_process(pid: int) -> None:
    if pid <= 0:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        logger.warning("no permission to terminate pid %s", pid)
        return
    # Don't block; the runner installs a SIGTERM handler that cleans up.
