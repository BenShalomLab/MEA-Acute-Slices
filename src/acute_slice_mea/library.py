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
    def from_data_root(cls, data_root, *, sample_override: str | None = None) -> "LibraryIndex":
        root = Path(data_root)
        if not root.exists():
            return cls([], mode="raw", root=str(root))

        records: list[RecordingEntry] = []
        for run_dir in _iter_run_dirs(root, sample_override=sample_override):
            raw_path = run_dir.path / "data.raw.h5"
            if not raw_path.exists():
                continue
            wells = _list_wells_from_h5(raw_path)
            records.append(
                RecordingEntry(
                    recording_id=run_dir.recording_id,
                    sample=run_dir.sample,
                    date=run_dir.date,
                    plate=run_dir.plate,
                    scan=run_dir.scan,
                    run=run_dir.run,
                    raw_path=str(raw_path),
                    wells=[WellEntry(well_id=w, raw_path=str(raw_path)) for w in wells],
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

    dashboard_manifest = bundle_dir / "dashboard" / "data" / "manifest.json"
    well = WellEntry(
        well_id=str(well_id),
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


def _list_wells_from_h5(raw_path: Path) -> list[str]:
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
            return sorted(wells.keys())
    except (OSError, KeyError):
        logger.warning("failed to read wells from %s", raw_path, exc_info=True)
        return []
