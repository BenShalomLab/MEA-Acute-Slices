#!/usr/bin/env python3
"""
Brain Slice MEA Cache Computation

Configure one raw Maxwell data file and one well, then run the heavy analysis pipeline.
Outputs are written to the selected cache directory.

Usage:
    python compute_brain_slice_cache.py --data-path <path> --well-id <id> --output-dir <dir>

Example:
    python compute_brain_slice_cache.py \
        --data-path /path/to/data.raw.h5 \
        --well-id well004 \
        --output-dir ./data/processed/brain_slice_well004_cache
"""

import argparse
import json
import logging
from pathlib import Path
import sys


def setup_logging(log_file=None):
    """Setup logging to both file and stdout."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Stream handler for stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File handler if log file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_logging()

# Setup Python path for imports
project_root = Path.cwd().resolve()
src_path = project_root / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
    logger.info(f"Added {src_path} to Python path")

# Clear any previously loaded modules to ensure fresh import
loaded_package = sys.modules.get("acute_slice_mea")
loaded_package_file = getattr(loaded_package, "__file__", None)
if loaded_package_file and src_path.resolve() not in Path(loaded_package_file).resolve().parents:
    for module_name in [name for name in sys.modules if name == "acute_slice_mea" or name.startswith("acute_slice_mea.")]:
        del sys.modules[module_name]

from acute_slice_mea import AnalysisConfig, run_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Brain Slice MEA Cache Computation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--data-path",
        required=True,
        type=str,
        help="Path to the raw Maxwell data file (data.raw.h5)"
    )
    parser.add_argument(
        "--well-id",
        required=True,
        type=str,
        help="Well identifier (e.g., 'well004')"
    )
    parser.add_argument(
        "--rec-name",
        type=str,
        default=None,
        help="Maxwell sub-recording id (e.g., 'rec0000'). Required for multi-recording files such as ActivityScan or MaxTwo plates with per-row recording ids."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Output directory for cached files"
    )
    parser.add_argument(
        "--lfp-low-hz",
        type=float,
        default=0.5,
        help="Low frequency cutoff for LFP filter (default: 0.5)"
    )
    parser.add_argument(
        "--lfp-high-hz",
        type=float,
        default=300,
        help="High frequency cutoff for LFP filter (default: 300)"
    )
    parser.add_argument(
        "--spike-low-hz",
        type=float,
        default=300,
        help="Low frequency cutoff for spike filter (default: 300)"
    )
    parser.add_argument(
        "--spike-high-hz",
        type=float,
        default=3000,
        help="High frequency cutoff for spike filter (default: 3000)"
    )
    parser.add_argument(
        "--lfp-window-sec",
        type=float,
        default=10,
        help="LFP analysis window duration in seconds (default: 10)"
    )
    parser.add_argument(
        "--lfp-step-sec",
        type=float,
        default=5,
        help="LFP analysis step size in seconds (default: 5)"
    )
    parser.add_argument(
        "--welch-segment-sec",
        type=float,
        default=2,
        help="Welch segment duration in seconds (default: 2)"
    )
    parser.add_argument(
        "--spectrum-duration-sec",
        type=float,
        default=10,
        help="Spectrum computation duration in seconds (default: 10)"
    )
    parser.add_argument(
        "--spectrum-max-freq-hz",
        type=float,
        default=150,
        help="Maximum frequency for spectrum (default: 150)"
    )
    parser.add_argument(
        "--preview-start-sec",
        type=float,
        default=0,
        help="Start time for preview data in seconds (default: 0)"
    )
    parser.add_argument(
        "--preview-duration-sec",
        type=float,
        default=10,
        help="Duration of preview data in seconds (default: 10)"
    )
    parser.add_argument(
        "--preview-max-points",
        type=int,
        default=20000,
        help="Maximum number of preview points (default: 20000)"
    )
    parser.add_argument(
        "--preview-max-electrodes",
        type=int,
        default=16,
        help="Maximum number of electrodes in preview (default: 16)"
    )
    parser.add_argument(
        "--dashboard-max-points",
        type=int,
        default=20000,
        help="Maximum points per electrode for dashboard (default: 20000)"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-2,
        help="Number of parallel jobs (-1 for all, -2 for all but 1, default: -2)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (if not specified, logs only to stdout)"
    )

    args = parser.parse_args()

    # Setup logging with file if specified
    global logger
    logger = setup_logging(args.log_file)
    if args.log_file:
        logger.info(f"Logging to file: {args.log_file}")

    logger.info("Starting Brain Slice MEA Cache Computation")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Well ID: {args.well_id}")
    logger.info(f"Output directory: {args.output_dir}")

    # Validate inputs
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Data file not found: {args.data_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")

    # Create analysis configuration
    config = AnalysisConfig(
        data_path=args.data_path,
        well_id=args.well_id,
        output_dir=args.output_dir,
        rec_name=args.rec_name,
        lfp_low_hz=args.lfp_low_hz,
        lfp_high_hz=args.lfp_high_hz,
        spike_low_hz=args.spike_low_hz,
        spike_high_hz=args.spike_high_hz,
        lfp_window_sec=args.lfp_window_sec,
        lfp_step_sec=args.lfp_step_sec,
        welch_segment_sec=args.welch_segment_sec,
        spectrum_duration_sec=args.spectrum_duration_sec,
        spectrum_max_freq_hz=args.spectrum_max_freq_hz,
        preview_start_sec=args.preview_start_sec,
        preview_duration_sec=args.preview_duration_sec,
        preview_max_points=args.preview_max_points,
        preview_max_electrodes=args.preview_max_electrodes,
        dashboard_max_points_per_electrode=args.dashboard_max_points,
        export_dashboard_data=True,
        spikeinterface_chunk_duration="60s",
        preview_electrode_ids=None,
        n_jobs=args.n_jobs,
        channel_chunk_size=None,
        progress=True,
        verbose=args.verbose,
    )

    logger.info("Configuration created:")
    logger.info(json.dumps(str(config), indent=2))

    # Run analysis
    logger.info("Starting analysis pipeline...")
    try:
        manifest = run_analysis(config)

        logger.info("Analysis completed successfully!")
        logger.info("Summary:")
        logger.info(json.dumps(manifest["summary"], indent=2))

        logger.info("\nGenerated files:")
        for key, value in manifest["files"].items():
            logger.info(f"  {key}: {value}")

        return 0

    except Exception as e:
        logger.error(f"Analysis failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
