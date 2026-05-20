"""Read-only cache reader for the trace viewer.

Loads the per-(recording, well) cache bundle produced by
``acute_slice_mea.pipeline.run_analysis``. Trace JSONs are loaded lazily, on
demand, with a small in-process LRU cache so repeated panning/zooming does
not re-read files.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from acute_slice_mea.cache import load_band_power, load_cache_manifest


@dataclass
class WellData:
    """Everything the dashboard needs for one (recording, well) bundle."""

    cache_dir: Path
    summary: dict
    electrodes: pd.DataFrame
    bursts: list[dict]
    probe: dict | None
    band_power: pd.DataFrame | None
    dashboard_manifest: dict | None
    _trace_index: dict[str, dict[str, str]]

    @classmethod
    def load(cls, cache_dir) -> "WellData":
        cache_dir = Path(cache_dir)
        manifest = load_cache_manifest(cache_dir)
        files = manifest.get("files", {})
        summary = manifest.get("summary", {})
        electrodes_path = files.get("electrodes") or str(cache_dir / "electrodes.csv")
        electrodes = pd.read_csv(electrodes_path)

        bursts: list[dict] = []
        bursts_path = files.get("bursts") or (cache_dir / "bursts.json")
        if bursts_path and Path(bursts_path).exists():
            bursts = json.loads(Path(bursts_path).read_text())

        probe: dict | None = None
        probe_path = files.get("probe") or (cache_dir / "probe.json")
        if probe_path and Path(probe_path).exists():
            probe = json.loads(Path(probe_path).read_text())

        band_power: pd.DataFrame | None = None
        bp_path = files.get("band_power")
        if bp_path and Path(bp_path).exists():
            try:
                band_power = load_band_power(bp_path)
            except Exception:  # pragma: no cover - defensive
                band_power = None

        dashboard_manifest: dict | None = None
        trace_index: dict[str, dict[str, str]] = {}
        dash_manifest_path = cache_dir / "dashboard" / "data" / "manifest.json"
        if dash_manifest_path.exists():
            dashboard_manifest = json.loads(dash_manifest_path.read_text())
            trace_index = dashboard_manifest.get("files", {}).get("traces", {}) or {}

        return cls(
            cache_dir=cache_dir,
            summary=summary,
            electrodes=electrodes,
            bursts=bursts,
            probe=probe,
            band_power=band_power,
            dashboard_manifest=dashboard_manifest,
            _trace_index=trace_index,
        )

    # -- queries ------------------------------------------------------

    @property
    def duration_sec(self) -> float:
        return float(self.summary.get("duration_sec") or 0.0)

    @property
    def sample_rate_hz(self) -> float:
        return float(self.summary.get("sampling_frequency_hz") or 0.0)

    def rms_by_electrode(self) -> dict[int, float]:
        if "rms_uv" not in self.electrodes.columns:
            return {}
        rows = self.electrodes.dropna(subset=["rms_uv"])
        return {int(eid): float(rms) for eid, rms in zip(rows["electrode_id"], rows["rms_uv"])}

    def routed_electrode_ids(self) -> list[int]:
        recorded = self.electrodes[self.electrodes["recorded"].astype(bool)]
        return recorded["electrode_id"].astype(int).tolist()

    def signals(self) -> list[str]:
        if self.dashboard_manifest is None:
            return []
        return list(self.dashboard_manifest.get("signals", []))

    def traces_for(
        self,
        electrode_ids: Iterable[int],
        *,
        signal: str = "lfp",
        t0: float | None = None,
        t1: float | None = None,
    ) -> list[dict]:
        """Return raw payloads for the requested electrode trace JSON files.

        Each payload contains ``time_sec``, ``value``, ``electrode_id``,
        ``channel_id`` and (when ``t0``/``t1`` are given) is already clipped to
        the window. Missing electrode ids are silently skipped.
        """
        signal_index = self._trace_index.get(signal, {})
        if not signal_index:
            return []
        results: list[dict] = []
        for eid in electrode_ids:
            relative = signal_index.get(str(int(eid)))
            if relative is None:
                continue
            payload = _read_trace_json_cached(str(self.cache_dir / "dashboard" / relative))
            time_arr = np.asarray(payload["time_sec"], dtype=float)
            value_arr = np.asarray(payload["value"], dtype=float)
            if t0 is not None or t1 is not None:
                lo = -np.inf if t0 is None else float(t0)
                hi = np.inf if t1 is None else float(t1)
                mask = (time_arr >= lo) & (time_arr <= hi)
                time_arr = time_arr[mask]
                value_arr = value_arr[mask]
            results.append(
                {
                    "electrode_id": int(payload["electrode_id"]),
                    "channel_id": payload.get("channel_id"),
                    "time_sec": time_arr,
                    "value": value_arr,
                }
            )
        return results

    def clear_trace_cache(self) -> None:
        _read_trace_json_cached.cache_clear()


@lru_cache(maxsize=512)
def _read_trace_json_cached(path: str) -> dict:
    return json.loads(Path(path).read_text())
