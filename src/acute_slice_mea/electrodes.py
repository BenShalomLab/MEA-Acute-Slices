"""Electrode/channel metadata helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_electrode_table(probe, recording) -> pd.DataFrame:
    """Return one row per probe contact with the matching recording channel."""
    contact_positions = np.asarray(probe.contact_positions)
    channel_locations = np.asarray(recording.get_channel_locations())
    channel_ids = list(recording.get_channel_ids())

    position_to_channel = {
        tuple(np.round(location, 1)): channel_id
        for channel_id, location in zip(channel_ids, channel_locations)
    }

    rows = []
    for electrode_id, position in enumerate(contact_positions):
        channel_id = position_to_channel.get(tuple(np.round(position, 1)))
        rows.append(
            {
                "electrode_id": int(electrode_id),
                "channel_id": channel_id,
                "x_um": float(position[0]),
                "y_um": float(position[1]),
                "recorded": bool(channel_id is not None),
            }
        )

    table = pd.DataFrame(rows, columns=["electrode_id", "channel_id", "x_um", "y_um", "recorded"])
    table["recorded"] = table["recorded"].astype(object)
    return table


def recorded_electrode_channels(electrode_table: pd.DataFrame, electrode_ids=None):
    """Return aligned electrode IDs and channel IDs for recorded electrodes."""
    table = electrode_table[electrode_table["recorded"].astype(bool)].copy()
    if electrode_ids is not None:
        requested = {int(eid) for eid in electrode_ids}
        table = table[table["electrode_id"].astype(int).isin(requested)]
    if table.empty:
        raise ValueError("No requested electrodes are recorded.")
    return table["electrode_id"].astype(int).tolist(), table["channel_id"].tolist()
