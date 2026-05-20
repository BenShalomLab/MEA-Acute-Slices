import json

import pandas as pd

from acute_slice_mea.probe_geometry import (
    MAXWELL_COLS,
    MAXWELL_PITCH_UM,
    MAXWELL_ROWS,
    build_probe_geometry,
    dump_probe_geometry_json,
    grid_to_electrode_id,
    position_to_grid,
)


def test_constants_match_maxwell_chip():
    assert MAXWELL_COLS == 220
    assert MAXWELL_ROWS == 120
    assert MAXWELL_PITCH_UM == 17.5


def test_position_to_grid_inverts_grid_to_electrode_id():
    col, row = 12, 34
    x = col * MAXWELL_PITCH_UM
    y = row * MAXWELL_PITCH_UM
    assert position_to_grid(x, y) == (col, row)
    assert grid_to_electrode_id(col, row) == row * MAXWELL_COLS + col


def test_build_probe_geometry_marks_only_routed_electrodes():
    electrodes = pd.DataFrame(
        [
            {"electrode_id": 0, "channel_id": "a", "x_um": 0.0, "y_um": 0.0, "recorded": True},
            {"electrode_id": 1, "channel_id": "b", "x_um": 17.5, "y_um": 0.0, "recorded": True},
            {"electrode_id": 2, "channel_id": None, "x_um": 0.0, "y_um": 17.5, "recorded": False},
        ]
    )
    geom = build_probe_geometry(electrodes)
    assert geom["cols"] == 220
    assert geom["rows"] == 120
    assert geom["pitch_um"] == 17.5
    assert len(geom["routed"]) == 2
    assert {entry["channel_id"] for entry in geom["routed"]} == {"a", "b"}
    assert all("col" in entry and "row" in entry for entry in geom["routed"])


def test_dump_probe_geometry_json_is_compact_and_round_trips(tmp_path):
    electrodes = pd.DataFrame(
        [{"electrode_id": 0, "channel_id": "a", "x_um": 0.0, "y_um": 0.0, "recorded": True}]
    )
    geom = build_probe_geometry(electrodes)
    out = dump_probe_geometry_json(geom, tmp_path / "probe.json")
    loaded = json.loads(out.read_text())
    assert loaded["routed"][0]["channel_id"] == "a"
    # compact form: no leading whitespace newlines
    assert "\n" not in out.read_text()
