#!/usr/bin/env python
"""Generate the published chip activity-map + RMS-electrode-overlay figure.

Defaults target the CarenPaula MEA-slices dataset (plate 16719):
  * background  = ActivityScan/000016 amplitudeMap.svg (Well1)
  * RMS source  = Network/000018 data.raw.h5, LFP RMS over [66, 76] s
  * ROI         = electrodes 75 76 87 (project electrode_id)

The computed RMS table is cached so re-running to tweak styling is cheap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from acute_slice_mea.published_figure import make_panel_a_figure

_PLATE = (
    "/mnt/benshalom-nas/raw_data/irc_maxone_desktop/home/mxwbio/Data2/"
    "MeaSlices_CarenPaula_04082026/MeaSlices_CarenPaula_04082026/260408/16719"
)
DEFAULT_SVG = f"{_PLATE}/ActivityScan/000021/analysis/ActivityAnalysis_v1/0001/Well1/amplitudeMap.svg"
DEFAULT_H5 = f"{_PLATE}/Network/000018/data.raw.h5"
DEFAULT_CACHE = "data/processed/16719_net000018_cache/electrodes_rms.csv"
DEFAULT_OUT = "figures/published_panel_a"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--svg", default=DEFAULT_SVG, help="activity-scan amplitudeMap.svg")
    p.add_argument(
        "--h5", default=DEFAULT_H5, help="network-scan data.raw.h5 (RMS source)"
    )
    p.add_argument(
        "--roi",
        type=int,
        nargs="*",
        default=[63, 91, 115, 272],
        help="ROI electrode_ids (bold border)",
    )
    p.add_argument("--start", type=float, default=66.0, help="RMS window start (s)")
    p.add_argument("--window", type=float, default=10.0, help="RMS window length (s)")
    p.add_argument(
        "--well", default=None, help="well/stream id (auto-detect if omitted)"
    )
    p.add_argument(
        "--crop",
        type=float,
        nargs=4,
        metavar=("X0", "X1", "Y0", "Y1"),
        default=None,
        help="crop region in microns (default: full chip)",
    )
    p.add_argument("--cache", default=DEFAULT_CACHE, help="RMS table cache CSV")
    p.add_argument("--force", action="store_true", help="recompute RMS, ignore cache")
    p.add_argument("--out", default=DEFAULT_OUT, help="output stem (.svg/.png)")
    p.add_argument(
        "--no-invert-y",
        dest="invert_y",
        action="store_false",
        help="place y=0 at the bottom (matching the source activity map's own "
        "orientation); default puts y=0 at the top (MaxWell chip convention, "
        "high-activity band at top, electrodes aligned)",
    )
    p.set_defaults(invert_y=True)
    p.add_argument(
        "--image-origin",
        default="lower",
        choices=["upper", "lower"],
        help="imshow origin for the background bitmap (default lower)",
    )
    args = p.parse_args()

    from acute_slice_mea.published_figure import compute_roi_rms

    rms_table = compute_roi_rms(
        args.h5,
        well_id=args.well,
        start_sec=args.start,
        window_sec=args.window,
        cache_path=args.cache,
        force=args.force,
    )

    make_panel_a_figure(
        args.svg,
        args.h5,
        args.roi,
        args.out,
        crop=tuple(args.crop) if args.crop else None,
        start_sec=args.start,
        window_sec=args.window,
        well_id=args.well,
        rms_table=rms_table,
        invert_y=args.invert_y,
        image_origin=args.image_origin,
    )
    print(f"Wrote {Path(args.out).with_suffix('.svg')} and .png")
    print(f"RMS cache: {args.cache}")


if __name__ == "__main__":
    main()
