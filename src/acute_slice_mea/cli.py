"""Command-line entry points for MEA cache generation."""

from __future__ import annotations

import argparse
import json

from acute_slice_mea.pipeline import AnalysisConfig, run_analysis


def _parse_electrodes(value):
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_parser():
    parser = argparse.ArgumentParser(description="Acute-slice MEA analysis tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    compute = subparsers.add_parser("compute", help="compute cache for one raw file and well")
    compute.add_argument("--data-path", required=True)
    compute.add_argument("--well-id", required=True)
    compute.add_argument("--output-dir", required=True)
    compute.add_argument("--lfp-window-sec", type=float, default=10)
    compute.add_argument("--lfp-step-sec", type=float, default=5)
    compute.add_argument("--spectrum-duration-sec", type=float, default=10)
    compute.add_argument("--preview-start-sec", type=float, default=0)
    compute.add_argument("--preview-duration-sec", type=float, default=10)
    compute.add_argument("--preview-max-points", type=int, default=20000)
    compute.add_argument("--preview-max-electrodes", type=int, default=16)
    compute.add_argument("--preview-electrode-ids", default=None, help="comma-separated electrode IDs")
    compute.add_argument("--dashboard-max-points-per-electrode", type=int, default=20000)
    compute.add_argument("--no-dashboard-data", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "compute":
        config = AnalysisConfig(
            data_path=args.data_path,
            well_id=args.well_id,
            output_dir=args.output_dir,
            lfp_window_sec=args.lfp_window_sec,
            lfp_step_sec=args.lfp_step_sec,
            spectrum_duration_sec=args.spectrum_duration_sec,
            preview_start_sec=args.preview_start_sec,
            preview_duration_sec=args.preview_duration_sec,
            preview_max_points=args.preview_max_points,
            preview_max_electrodes=args.preview_max_electrodes,
            preview_electrode_ids=_parse_electrodes(args.preview_electrode_ids),
            dashboard_max_points_per_electrode=args.dashboard_max_points_per_electrode,
            export_dashboard_data=not args.no_dashboard_data,
        )
        manifest = run_analysis(config)
        print(json.dumps(manifest, indent=2, default=str))
        return 0
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
