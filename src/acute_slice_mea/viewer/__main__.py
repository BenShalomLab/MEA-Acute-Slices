"""Run the LFP Dashboard Dash app: ``python -m acute_slice_mea.viewer``."""

from __future__ import annotations

import argparse
from pathlib import Path

from acute_slice_mea.viewer.app import create_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="acute-slice-viewer",
        description="Interactive LFP dashboard for cached Acute Slice MEA recordings.",
    )
    parser.add_argument(
        "--cache-root",
        required=True,
        type=Path,
        help="Directory containing one or more per-(recording, well) cache bundles.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Raw data root directory. Enables the Data Explore and Analysis Parameters tabs.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode (hot reload, error overlay).")
    args = parser.parse_args(argv)

    app = create_app(args.cache_root, data_root=args.data_root)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
