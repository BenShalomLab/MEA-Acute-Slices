"""MaxWell chip geometry for the trace-viewer dashboard.

The MaxWell MaxOne / MaxTwo chip exposes a fixed 220 (cols) x 120 (rows)
electrode array at 17.5 um pitch; only ~1000 electrodes are routed at a time.
The viewer needs the full grid (for the zoomable picker) plus a mask of which
electrodes are routed in the current recording.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MAXWELL_COLS = 220
MAXWELL_ROWS = 120
MAXWELL_PITCH_UM = 17.5

WIDTH_UM = MAXWELL_COLS * MAXWELL_PITCH_UM
HEIGHT_UM = MAXWELL_ROWS * MAXWELL_PITCH_UM


def _round_pitch(value: float) -> int:
    return int(round(float(value) / MAXWELL_PITCH_UM))


def position_to_grid(x_um: float, y_um: float) -> tuple[int, int]:
    """Map (x_um, y_um) to a (col, row) index on the 220x120 chip."""
    return _round_pitch(x_um), _round_pitch(y_um)


def grid_to_electrode_id(col: int, row: int) -> int:
    """Canonical row-major electrode id for a (col, row) MaxWell position."""
    return int(row) * MAXWELL_COLS + int(col)


def build_probe_geometry(electrode_table: pd.DataFrame) -> dict:
    """Return a dict describing the full MaxWell grid + routed electrodes.

    Output shape (suitable for JSON serialization):
    {
      "cols": 220, "rows": 120, "pitch_um": 17.5,
      "width_um": 3850.0, "height_um": 2100.0,
      "routed": [{electrode_id, channel_id, col, row, x_um, y_um}, ...],
    }

    Unrouted electrodes are not enumerated to keep the payload small; the
    viewer reconstructs them by drawing the full 220x120 grid client-side and
    overlaying the routed set on top.
    """
    if electrode_table is None or len(electrode_table) == 0:
        routed: list[dict] = []
    else:
        recorded = electrode_table[electrode_table["recorded"].astype(bool)].copy()
        routed = []
        for row in recorded.itertuples(index=False):
            col_idx, row_idx = position_to_grid(float(row.x_um), float(row.y_um))
            channel_id = getattr(row, "channel_id", None)
            entry = {
                "electrode_id": int(row.electrode_id),
                "channel_id": None if channel_id is None else _coerce_channel_id(channel_id),
                "col": int(col_idx),
                "row": int(row_idx),
                "x_um": float(row.x_um),
                "y_um": float(row.y_um),
            }
            rms = getattr(row, "rms_uv", None)
            if rms is not None and not _isnan(rms):
                entry["rms_uv"] = float(rms)
            routed.append(entry)
    return {
        "cols": MAXWELL_COLS,
        "rows": MAXWELL_ROWS,
        "pitch_um": MAXWELL_PITCH_UM,
        "width_um": WIDTH_UM,
        "height_um": HEIGHT_UM,
        "routed": routed,
    }


def _coerce_channel_id(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)


def _isnan(value) -> bool:
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def dump_probe_geometry_json(geometry: dict, path) -> Path:
    """Write the geometry dict as a compact JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(geometry, separators=(",", ":")))
    return path


def load_probe_geometry(path) -> dict:
    return json.loads(Path(path).read_text())
