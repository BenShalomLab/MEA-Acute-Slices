"""Recording / well discovery for the trace-viewer dashboard.

Two ingestion modes:

* ``LibraryIndex.from_cache_root(cache_root)`` — walks a directory tree of
  cache bundles produced by ``acute_slice_mea.pipeline.run_analysis``. Each
  leaf is one ``(recording, well)`` and exposes everything the viewer needs.
  This is the mode the Dash app runs in.

* ``LibraryIndex.from_data_root(data_root)`` — walks a tree of raw Maxwell
  ``.h5`` files following the ``[Sample/]Date/Plate/ScanType/Run`` layout used
  by the neighbouring ``Yuxin_MEA`` project. Used by
  ``scripts/build_viewer_cache.py`` to enumerate what to process; does not
  need spikeinterface installed (h5py only).

The shape of the public objects (RecordingEntry / WellEntry) is intentionally
shared across both modes so callers can branch on ``mode`` rather than type.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
import json
import logging
from pathlib import Path
import re
from typing import Iterable, Iterator

logger = logging.getLogger(__name__)

DATE_DIR_RE = re.compile(r"^\d{6}$")


@dataclass
class WellEntry:
    """One well within one recording."""

    well_id: str
    rec_name: str | None = None   # Maxwell sub-recording id (rec0000…); None when single-rec
    cache_dir: str | None = None  # populated in cache mode
    raw_path: str | None = None   # populated in raw mode
    sample_rate_hz: float | None = None
    duration_sec: float | None = None
    num_recorded_electrodes: int | None = None
    has_dashboard_data: bool = False
    group: str | None = None      # treatment / genotype label, if known


@dataclass
class RecordingEntry:
    """One Maxwell recording (== one .h5 file or its cached siblings)."""

    recording_id: str
    sample: str
    date: str
    plate: str
    scan: str
    run: str
    label: str | None = None
    operator: str | None = None
    notes: str | None = None
    raw_path: str | None = None
    cache_root: str | None = None
    wells: list[WellEntry] = field(default_factory=list)

    @property
    def iso_date(self) -> str:
        """YYMMDD → YYYY-MM-DD when parseable, otherwise the original string."""
        if len(self.date) == 6 and self.date.isdigit():
            try:
                d = date(2000 + int(self.date[:2]), int(self.date[2:4]), int(self.date[4:6]))
                return d.isoformat()
            except ValueError:
                pass
        return self.date


class LibraryIndex:
    """Indexed recordings grouped by sample/date/plate/scan/run."""

    def __init__(self, recordings: list[RecordingEntry], *, mode: str, root: str) -> None:
        self.recordings = sorted(
            recordings,
            key=lambda r: (r.date, r.sample, r.plate, r.scan, r.run),
            reverse=True,
        )
        self.mode = mode
        self.root = root

    # -- factories ----------------------------------------------------

    @classmethod
    def from_cache_root(cls, cache_root) -> "LibraryIndex":
        root = Path(cache_root)
        if not root.exists():
            return cls([], mode="cache", root=str(root))
        records: dict[str, RecordingEntry] = {}
        for manifest_path in root.rglob("manifest.json"):
            try:
                entry = _entry_from_cache_manifest(manifest_path, cache_root=root)
            except _SkipBundle as exc:
                logger.debug("skipping %s: %s", manifest_path, exc)
                continue
            except Exception:  # pragma: no cover - defensive
                logger.exception("failed to ingest cache bundle: %s", manifest_path)
                continue
            recording_id = entry["recording_id"]
            well_entry = entry["well"]
            if recording_id not in records:
                records[recording_id] = RecordingEntry(
                    recording_id=recording_id,
                    sample=entry["sample"],
                    date=entry["date"],
                    plate=entry["plate"],
                    scan=entry["scan"],
                    run=entry["run"],
                    label=entry.get("label"),
                    operator=entry.get("operator"),
                    notes=entry.get("notes"),
                    raw_path=entry.get("raw_path"),
                    cache_root=str(root),
                )
            records[recording_id].wells.append(well_entry)
        for rec in records.values():
            rec.wells.sort(key=lambda w: w.well_id)
        return cls(list(records.values()), mode="cache", root=str(root))

    @classmethod
    def from_data_and_cache_root(
        cls,
        data_root,
        cache_root,
        *,
        sample_override: str | None = None,
    ) -> "LibraryIndex":
        """Enumerate raw recordings and overlay cache state per well.

        Used by the spawn-capable dashboard: every recording shows up (even
        ones with no cache yet) so the user can hit Run; existing caches are
        recognised so the viewer can open them instantly. Multi-rec
        recordings already encode ``rec_name`` as a trailing segment of
        ``rec.run`` (see ``from_data_root``), so the path math here lands at
        the same cache directory the pipeline writes to.
        """
        raw = cls.from_data_root(data_root, sample_override=sample_override)
        cache_root = Path(cache_root)
        for rec in raw.recordings:
            for well in rec.wells:
                cache_dir = (
                    cache_root
                    / rec.sample
                    / rec.date
                    / rec.plate
                    / rec.scan
                    / rec.run
                    / well.well_id
                )
                manifest_path = cache_dir / "manifest.json"
                if not manifest_path.exists():
                    continue
                try:
                    manifest = json.loads(manifest_path.read_text())
                except Exception:
                    logger.exception("failed to read manifest: %s", manifest_path)
                    continue
                summary = manifest.get("summary") or {}
                well.cache_dir = str(cache_dir)
                well.sample_rate_hz = _safe_float(summary.get("sampling_frequency_hz"))
                well.duration_sec = _safe_float(summary.get("duration_sec"))
                well.num_recorded_electrodes = _safe_int(summary.get("num_recorded_electrodes"))
                well.has_dashboard_data = (cache_dir / "dashboard" / "data" / "manifest.json").exists()
            rec.cache_root = str(cache_root)
        return cls(raw.recordings, mode="data+cache", root=str(Path(data_root)))

    @classmethod
    def from_data_root(cls, data_root, *, sample_override: str | None = None) -> "LibraryIndex":
        root = Path(data_root)
        if not root.exists():
            return cls([], mode="raw", root=str(root))

        records: list[RecordingEntry] = []
        for run_dir in _iter_run_dirs(root, sample_override=sample_override):
            raw_path = run_dir.path / "data.raw.h5"
            if not raw_path.exists():
                continue
            wells_recs = _list_wells_and_recs_from_h5(raw_path)
            if not wells_recs:
                continue
            # Group (well_id, rec_name) pairs by rec_name. Single-rec files
            # collapse to one RecordingEntry with rec_name=None (legacy layout).
            # Multi-rec files expand to one RecordingEntry per rec_name with
            # /{rec_name} appended to run and recording_id so the viewer keys
            # them distinctly.
            by_rec: dict[str | None, list[str]] = {}
            for well_id, rec_name in wells_recs:
                by_rec.setdefault(rec_name, []).append(well_id)
            for rec_name, well_ids in by_rec.items():
                if rec_name is None:
                    actual_run = run_dir.run
                    recording_id = run_dir.recording_id
                else:
                    actual_run = f"{run_dir.run}/{rec_name}"
                    recording_id = f"{run_dir.recording_id}/{rec_name}"
                records.append(
                    RecordingEntry(
                        recording_id=recording_id,
                        sample=run_dir.sample,
                        date=run_dir.date,
                        plate=run_dir.plate,
                        scan=run_dir.scan,
                        run=actual_run,
                        raw_path=str(raw_path),
                        wells=[
                            WellEntry(well_id=w, rec_name=rec_name, raw_path=str(raw_path))
                            for w in sorted(well_ids)
                        ],
                    )
                )
        return cls(records, mode="raw", root=str(root))

    # -- queries ------------------------------------------------------

    def find(self, recording_id: str, well_id: str) -> WellEntry | None:
        for rec in self.recordings:
            if rec.recording_id != recording_id:
                continue
            for well in rec.wells:
                if well.well_id == well_id:
                    return well
        return None

    def find_recording(self, recording_id: str) -> RecordingEntry | None:
        for rec in self.recordings:
            if rec.recording_id == recording_id:
                return rec
        return None

    def grouped_by_date(self) -> list[tuple[str, list[RecordingEntry]]]:
        groups: dict[str, list[RecordingEntry]] = {}
        for rec in self.recordings:
            groups.setdefault(rec.iso_date, []).append(rec)
        return sorted(groups.items(), key=lambda kv: kv[0], reverse=True)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "root": self.root,
            "recordings": [
                {**asdict(rec), "wells": [asdict(w) for w in rec.wells]}
                for rec in self.recordings
            ],
        }


# =============================================================================
# Cache-mode helpers
# =============================================================================


class _SkipBundle(Exception):
    """Internal: raised when a manifest looks like it isn't ours."""


def _entry_from_cache_manifest(manifest_path: Path, *, cache_root: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    summary = manifest.get("summary")
    if not isinstance(summary, dict):
        raise _SkipBundle("missing summary")
    well_id = summary.get("well_id")
    data_path = summary.get("data_path")
    if not well_id or not data_path:
        raise _SkipBundle("summary missing well_id or data_path")

    parts = _parse_recording_id(Path(data_path))
    bundle_dir = manifest_path.parent

    # Multi-recording files: rec_name was persisted in the summary by
    # AnalysisConfig. Fold it into run/recording_id so the cache-mode index
    # matches the raw-mode layout (one logical recording per rec_name).
    rec_name = summary.get("rec_name") or None
    if rec_name:
        parts["run"] = f"{parts['run']}/{rec_name}"
        parts["recording_id"] = f"{parts['recording_id']}/{rec_name}"

    dashboard_manifest = bundle_dir / "dashboard" / "data" / "manifest.json"
    well = WellEntry(
        well_id=str(well_id),
        rec_name=rec_name,
        cache_dir=str(bundle_dir),
        raw_path=str(data_path),
        sample_rate_hz=_safe_float(summary.get("sampling_frequency_hz")),
        duration_sec=_safe_float(summary.get("duration_sec")),
        num_recorded_electrodes=_safe_int(summary.get("num_recorded_electrodes")),
        has_dashboard_data=dashboard_manifest.exists(),
    )

    return {
        "recording_id": parts["recording_id"],
        "sample": parts["sample"],
        "date": parts["date"],
        "plate": parts["plate"],
        "scan": parts["scan"],
        "run": parts["run"],
        "raw_path": str(data_path),
        "well": well,
    }


def _parse_recording_id(data_path: Path) -> dict:
    """Reconstruct (sample, date, plate, scan, run) from a Maxwell raw path.

    Expected layouts (matching Yuxin_MEA DatasetManager):
      [Sample]/Date/Plate/ScanType/Run/data.raw.h5
      Date/Plate/ScanType/Run/data.raw.h5
    """
    parts = list(data_path.resolve().parts)
    if data_path.name == "data.raw.h5":
        parts = parts[:-1]
    # Look for the last 6-digit date directory; that's the most reliable anchor.
    date_idx = None
    for idx in range(len(parts) - 3, -1, -1):
        if DATE_DIR_RE.match(parts[idx]):
            date_idx = idx
            break
    if date_idx is None or date_idx + 3 >= len(parts):
        # Fall back: assume the last 4 parts are Date/Plate/Scan/Run.
        date_idx = max(0, len(parts) - 4)
    sample = parts[date_idx - 1] if date_idx > 0 else "unknown"
    date_s = parts[date_idx] if date_idx < len(parts) else "unknown"
    plate = parts[date_idx + 1] if date_idx + 1 < len(parts) else "unknown"
    scan = parts[date_idx + 2] if date_idx + 2 < len(parts) else "unknown"
    run = parts[date_idx + 3] if date_idx + 3 < len(parts) else "unknown"
    recording_id = f"{sample}/{date_s}/{plate}/{scan}/{run}"
    return {
        "recording_id": recording_id,
        "sample": sample,
        "date": date_s,
        "plate": plate,
        "scan": scan,
        "run": run,
    }


def _safe_int(value) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


# =============================================================================
# Raw-mode helpers
# =============================================================================


@dataclass
class _RunDir:
    path: Path
    sample: str
    date: str
    plate: str
    scan: str
    run: str

    @property
    def recording_id(self) -> str:
        return f"{self.sample}/{self.date}/{self.plate}/{self.scan}/{self.run}"


def _iter_run_dirs(root: Path, *, sample_override: str | None) -> Iterator[_RunDir]:
    """Yield run directories under ``root`` matching the MEA layout."""
    if not root.is_dir():
        return
    # Determine whether root is "root level" (samples one level deep) or
    # "sample level" (dates directly under root).
    children = [p for p in root.iterdir() if p.is_dir()]
    if any(DATE_DIR_RE.match(child.name) for child in children):
        sample_dirs: Iterable[Path] = [root]
        default_sample = sample_override or root.name
    else:
        sample_dirs = children
        default_sample = None

    for sample_dir in sample_dirs:
        sample = default_sample or sample_dir.name
        for date_dir in sorted(p for p in sample_dir.iterdir() if p.is_dir()):
            if not DATE_DIR_RE.match(date_dir.name):
                continue
            for plate_dir in sorted(p for p in date_dir.iterdir() if p.is_dir()):
                for scan_dir in sorted(p for p in plate_dir.iterdir() if p.is_dir()):
                    for run_dir in sorted(p for p in scan_dir.iterdir() if p.is_dir()):
                        yield _RunDir(
                            path=run_dir,
                            sample=sample,
                            date=date_dir.name,
                            plate=plate_dir.name,
                            scan=scan_dir.name,
                            run=run_dir.name,
                        )


def _list_wells_and_recs_from_h5(raw_path: Path) -> list[tuple[str, str | None]]:
    """Enumerate (well_id, rec_name) pairs in a Maxwell h5 file.

    Mirrors ``neo.rawio.MaxwellRawIO._parse_header``: each well group under
    ``h5["wells"]`` has one or more ``rec*`` children. When the file contains
    only a single rec_name across all wells we collapse it to ``rec_name=None``
    so single-rec recordings (e.g., Network) keep the legacy cache path.
    Multi-rec files (ActivityScan, MaxTwo plates with per-row recording ids)
    yield one tuple per (well, rec) combination — callers expand them into
    distinct RecordingEntries.
    """
    try:
        import h5py
    except ImportError:  # pragma: no cover
        logger.warning("h5py not installed; cannot list wells from %s", raw_path)
        return []
    try:
        with h5py.File(raw_path, "r") as h5:
            wells = h5.get("wells")
            if wells is None:
                return []
            pairs: list[tuple[str, str]] = []
            all_rec_names: set[str] = set()
            for well_id in sorted(wells.keys()):
                rec_group = wells[well_id]
                rec_names = sorted(rec_group.keys()) if hasattr(rec_group, "keys") else []
                if not rec_names:
                    continue
                for rec_name in rec_names:
                    pairs.append((well_id, rec_name))
                    all_rec_names.add(rec_name)
    except (OSError, KeyError):
        logger.warning("failed to read wells from %s", raw_path, exc_info=True)
        return []
    if len(all_rec_names) <= 1:
        # Single rec_name across the whole file → preserve legacy layout.
        return [(well_id, None) for well_id, _ in pairs]
    return [(well_id, rec_name) for well_id, rec_name in pairs]
