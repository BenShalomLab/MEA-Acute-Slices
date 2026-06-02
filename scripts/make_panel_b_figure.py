#!/usr/bin/env python
"""Generate the published Panel B figure: ROI waveform snapshot + 3-MAD raster.

Defaults target the CarenPaula MEA-slices dataset (plate 16719):
  * source = Network/000018 data.raw.h5
  * ROI    = electrodes 63 91 115 272 (project electrode_id, same as Panel A)
  * window = 1 s LFP snapshot at [66, 67] s (a zoom into Panel A's RMS window)

Left panel  = stacked LFP traces (0.5-300 Hz + global-median CMR, the same chain
              as the viewer / build_viewer_cache).
Right panel = per-electrode raster marking |LFP - median| >= 3 robust-SD (MAD)
              crossings over the same window.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from acute_slice_mea.published_figure import make_panel_b_figure

_PLATE = (
    "/mnt/benshalom-nas/raw_data/irc_maxone_desktop/home/mxwbio/Data2/"
    "MeaSlices_CarenPaula_04082026/MeaSlices_CarenPaula_04082026/260408/16719"
)
DEFAULT_H5 = f"{_PLATE}/Network/000018/data.raw.h5"
DEFAULT_OUT = "figures/published_panel_b"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", default=DEFAULT_H5, help="network-scan data.raw.h5 (LFP source)")
    p.add_argument(
        "--roi",
        type=int,
        nargs="*",
        default=[63, 91, 115, 272],
        help="ROI electrode_ids (one waveform/raster row each)",
    )
    p.add_argument("--start", type=float, default=66.0, help="snapshot start (s)")
    p.add_argument("--window", type=float, default=1.0, help="snapshot length (s)")
    p.add_argument("--well", default=None, help="well/stream id (auto-detect if omitted)")
    p.add_argument(
        "--threshold-sd",
        type=float,
        default=3.0,
        help="raster threshold in robust SD (MAD-derived); default 3",
    )
    p.add_argument("--out", default=DEFAULT_OUT, help="output stem (.svg/.png)")
    args = p.parse_args()

    make_panel_b_figure(
        args.h5,
        args.roi,
        args.out,
        start_sec=args.start,
        window_sec=args.window,
        well_id=args.well,
        threshold_sd=args.threshold_sd,
    )
    print(f"Wrote {Path(args.out).with_suffix('.svg')} and .png")


if __name__ == "__main__":
    main()
