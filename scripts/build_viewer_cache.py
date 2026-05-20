"""Batch-build viewer caches from a raw data root.

Walks a directory tree of MaxWell ``.h5`` files following the
``[Sample/]Date/Plate/ScanType/Run`` layout, enumerates wells via h5py, and
runs ``acute_slice_mea.pipeline.run_analysis`` for any ``(recording, well)``
pair not already cached. Idempotent / resumable: existing bundles are skipped
unless ``--force`` is passed.

Example::

    python scripts/build_viewer_cache.py \
        --data-root /Volumes/SadeghYR/Yuxin/MEA/raw/MeaSlices_CarenPaula_04082026 \
        --cache-root data/processed \
        --only 260408/16719/ActivityScan/000001 \
        --wells well000

"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from time import perf_counter

from acute_slice_mea.library import LibraryIndex, RecordingEntry
from acute_slice_mea.pipeline import AnalysisConfig, run_analysis

logger = logging.getLogger("build_viewer_cache")


def build_cache_for(
    *,
    recording: RecordingEntry,
    well_id: str,
    cache_root: Path,
    force: bool = False,
    n_jobs: int = 1,
    progress: bool = True,
    compute_spectrum: bool = True,
    compute_trace_preview: bool = True,
    compute_bursts: bool = True,
    export_probe_geometry: bool = True,
    export_dashboard_data: bool = True,
) -> Path:
    """Run ``run_analysis`` for one (recording, well); return the cache dir."""
    out_dir = cache_root / recording.sample / recording.date / recording.plate / recording.scan / recording.run / well_id
    manifest = out_dir / "manifest.json"
    if manifest.exists() and not force:
        logger.info("skip (cached): %s", out_dir)
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    config = AnalysisConfig(
        data_path=recording.raw_path or "",
        well_id=well_id,
        output_dir=str(out_dir),
        n_jobs=n_jobs,
        progress=progress,
        verbose=True,
        compute_spectrum=compute_spectrum,
        compute_trace_preview=compute_trace_preview,
        compute_bursts=compute_bursts,
        export_probe_geometry=export_probe_geometry,
        export_dashboard_data=export_dashboard_data,
    )
    started = perf_counter()
    run_analysis(config)
    logger.info("done (%.1fs): %s", perf_counter() - started, out_dir)
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Restrict to recordings whose `recording_id` contains this substring; repeatable.",
    )
    parser.add_argument(
        "--wells",
        action="append",
        default=[],
        help="Restrict to specific well ids (e.g., well000); repeatable.",
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Re-run even when manifest.json exists.")
    parser.add_argument("--quiet", action="store_true", help="Disable per-window progress bars.")
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Compute only LFP band power and per-electrode RMS; skip spectrum, trace preview, bursts, probe geometry, and dashboard export.",
    )
    parser.add_argument("--no-spectrum", action="store_true", help="Skip Welch spectrum summary.")
    parser.add_argument("--no-trace-preview", action="store_true", help="Skip trace preview computation.")
    parser.add_argument("--no-bursts", action="store_true", help="Skip network-burst detection.")
    parser.add_argument("--no-probe-geometry", action="store_true", help="Skip probe geometry export.")
    parser.add_argument("--no-dashboard", action="store_true", help="Skip dashboard JSON export.")
    args = parser.parse_args(argv)

    compute_spectrum = not (args.no_spectrum or args.minimal)
    compute_trace_preview = not (args.no_trace_preview or args.minimal)
    compute_bursts = not (args.no_bursts or args.minimal)
    export_probe_geometry = not (args.no_probe_geometry or args.minimal)
    export_dashboard_data = not (args.no_dashboard or args.minimal)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    library = LibraryIndex.from_data_root(args.data_root)
    if not library.recordings:
        logger.warning("No recordings found under %s", args.data_root)
        return 1

    only_filters = args.only or []
    wells_filter = set(args.wells) if args.wells else None

    args.cache_root.mkdir(parents=True, exist_ok=True)
    total_failures = 0
    for rec in library.recordings:
        if only_filters and not any(needle in rec.recording_id for needle in only_filters):
            continue
        for well in rec.wells:
            if wells_filter and well.well_id not in wells_filter:
                continue
            try:
                build_cache_for(
                    recording=rec,
                    well_id=well.well_id,
                    cache_root=args.cache_root,
                    force=args.force,
                    n_jobs=args.n_jobs,
                    progress=not args.quiet,
                    compute_spectrum=compute_spectrum,
                    compute_trace_preview=compute_trace_preview,
                    compute_bursts=compute_bursts,
                    export_probe_geometry=export_probe_geometry,
                    export_dashboard_data=export_dashboard_data,
                )
            except Exception:
                total_failures += 1
                logger.exception("failed: %s / %s", rec.recording_id, well.well_id)
    return 0 if total_failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
