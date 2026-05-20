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
    rec_name: str | None = None,
    force: bool = False,
    n_jobs: int = 1,
    progress: bool = True,
    compute_spectrum: bool = True,
    compute_trace_preview: bool = True,
    compute_bursts: bool = True,
    export_probe_geometry: bool = True,
    export_dashboard_data: bool = True,
    cache_lfp_to_disk: bool = True,
    lfp_cache_dir: str | None = None,
    keep_lfp_cache: bool = False,
    lfp_target_fs_hz: float | None = 1000.0,
) -> Path:
    """Run ``run_analysis`` for one (recording, well); return the cache dir."""
    # recording.run already encodes /{rec_name} for multi-rec entries, so the
    # cache path naturally lands at .../scan/run/rec_name/well_id without any
    # extra joining here.
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
        rec_name=rec_name,
        n_jobs=n_jobs,
        progress=progress,
        verbose=True,
        compute_spectrum=compute_spectrum,
        compute_trace_preview=compute_trace_preview,
        compute_bursts=compute_bursts,
        export_probe_geometry=export_probe_geometry,
        export_dashboard_data=export_dashboard_data,
        cache_lfp_to_disk=cache_lfp_to_disk,
        lfp_cache_dir=lfp_cache_dir,
        keep_lfp_cache=keep_lfp_cache,
        lfp_target_fs_hz=lfp_target_fs_hz,
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
    parser.add_argument(
        "--no-lfp-cache",
        action="store_true",
        help="Disable on-disk LFP materialization (legacy path: filter+CR re-applied per window).",
    )
    parser.add_argument(
        "--keep-lfp-cache",
        action="store_true",
        help="Keep the .lfp_cache/ folder after the run (otherwise removed on completion).",
    )
    parser.add_argument(
        "--lfp-cache-dir",
        type=str,
        default=None,
        help="Override location of the on-disk LFP cache (default: <output_dir>/.lfp_cache).",
    )
    parser.add_argument(
        "--lfp-target-fs-hz",
        type=float,
        default=1000.0,
        help="Downsample the LFP path to this rate (Hz) before filtering. Default 1000. "
             "Use --no-resample to keep the native rate.",
    )
    parser.add_argument(
        "--no-resample",
        action="store_true",
        help="Disable the default 1 kHz LFP-path resample; run LFP analysis at the native sample rate.",
    )
    args = parser.parse_args(argv)

    compute_spectrum = not (args.no_spectrum or args.minimal)
    compute_trace_preview = not (args.no_trace_preview or args.minimal)
    compute_bursts = not (args.no_bursts or args.minimal)
    export_probe_geometry = not (args.no_probe_geometry or args.minimal)
    export_dashboard_data = not (args.no_dashboard or args.minimal)
    lfp_target_fs_hz: float | None = None if args.no_resample else float(args.lfp_target_fs_hz)

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
                    rec_name=well.rec_name,
                    force=args.force,
                    n_jobs=args.n_jobs,
                    progress=not args.quiet,
                    compute_spectrum=compute_spectrum,
                    compute_trace_preview=compute_trace_preview,
                    compute_bursts=compute_bursts,
                    export_probe_geometry=export_probe_geometry,
                    export_dashboard_data=export_dashboard_data,
                    cache_lfp_to_disk=not args.no_lfp_cache,
                    lfp_cache_dir=args.lfp_cache_dir,
                    keep_lfp_cache=args.keep_lfp_cache,
                    lfp_target_fs_hz=lfp_target_fs_hz,
                )
            except Exception:
                total_failures += 1
                logger.exception("failed: %s / %s", rec.recording_id, well.well_id)
    return 0 if total_failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
