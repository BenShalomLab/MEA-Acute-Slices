"""Read-only cache reader for the trace viewer.

Loads the per-(recording, well) cache bundle produced by
``acute_slice_mea.pipeline.run_analysis``. Trace JSONs are loaded lazily, on
demand, with a small in-process LRU cache so repeated panning/zooming does
not re-read files.

When a cache bundle was built without ``--dashboard`` (no per-electrode JSON
traces on disk), ``traces_for`` falls back to slicing LFP from the source
``.h5`` via SpikeInterface, using ``summary.data_path`` from the manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from acute_slice_mea.cache import load_band_power, load_cache_manifest


# Target points per electrode per window for the lazy reader. ~2k is enough for
# smooth WebGL line rendering at any window size the UI exposes (≤ 60 s).
_LAZY_TARGET_POINTS = 2000


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
    raw_path: Path | None = None
    well_id: str | None = None
    rec_name: str | None = None
    _eid_to_channel: dict[int, str] = field(default_factory=dict)

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

        raw_path_str = summary.get("data_path")
        raw_path = Path(raw_path_str) if raw_path_str else None
        well_id = summary.get("well_id")
        rec_name = summary.get("rec_name")

        eid_to_channel: dict[int, str] = {}
        if "electrode_id" in electrodes.columns and "channel_id" in electrodes.columns:
            recorded = (
                electrodes[electrodes["recorded"].astype(bool)]
                if "recorded" in electrodes.columns
                else electrodes
            )
            for eid, cid in zip(recorded["electrode_id"], recorded["channel_id"]):
                if pd.isna(cid):
                    continue
                # SpikeInterface's MaxWell extractor returns channel_ids as
                # zero-padded strings (e.g. "0", "1", ...); coerce here so
                # downstream get_traces(channel_ids=...) matches by value.
                try:
                    cid_str = str(int(cid))
                except (TypeError, ValueError):
                    cid_str = str(cid)
                eid_to_channel[int(eid)] = cid_str

        return cls(
            cache_dir=cache_dir,
            summary=summary,
            electrodes=electrodes,
            bursts=bursts,
            probe=probe,
            band_power=band_power,
            dashboard_manifest=dashboard_manifest,
            _trace_index=trace_index,
            raw_path=raw_path,
            well_id=well_id,
            rec_name=rec_name,
            _eid_to_channel=eid_to_channel,
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
        """Return raw payloads for the requested electrode traces.

        Each payload contains ``time_sec``, ``value``, ``electrode_id``,
        ``channel_id`` and (when ``t0``/``t1`` are given) is already clipped to
        the window. Missing electrode ids are silently skipped.

        Source order:
        1. Per-electrode JSON files from the dashboard export (if present).
        2. Lazy SpikeInterface slice from ``summary.data_path`` (fallback when
           the dashboard export was skipped, e.g. ``--minimal`` builds).
        """
        eid_list = [int(e) for e in electrode_ids]
        if not eid_list:
            return []

        signal_index = self._trace_index.get(signal, {})
        if signal_index:
            return self._traces_from_json(eid_list, signal_index, t0=t0, t1=t1)

        if self.raw_path is not None and self.well_id is not None:
            return self._traces_from_recording(
                eid_list, signal=signal, t0=t0, t1=t1
            )
        return []

    def _traces_from_json(
        self,
        eid_list: list[int],
        signal_index: dict[str, str],
        *,
        t0: float | None,
        t1: float | None,
    ) -> list[dict]:
        results: list[dict] = []
        for eid in eid_list:
            relative = signal_index.get(str(eid))
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

    def _traces_from_recording(
        self,
        eid_list: list[int],
        *,
        signal: str,
        t0: float | None,
        t1: float | None,
    ) -> list[dict]:
        if self.raw_path is None or self.well_id is None:
            return []
        if not self.raw_path.exists():
            return []
        duration = self.duration_sec or 0.0
        fs = self.sample_rate_hz or 0.0
        if duration <= 0 or fs <= 0:
            return []
        t_lo = 0.0 if t0 is None else max(0.0, float(t0))
        t_hi = duration if t1 is None else min(duration, float(t1))
        if t_hi <= t_lo:
            return []

        channel_ids: list[str] = []
        kept_eids: list[int] = []
        for eid in eid_list:
            cid = self._eid_to_channel.get(eid)
            if cid is None:
                continue
            channel_ids.append(cid)
            kept_eids.append(eid)
        if not channel_ids:
            return []

        time_arr, traces = _lazy_lfp_window(
            raw_path=str(self.raw_path),
            well_id=str(self.well_id),
            rec_name=self.rec_name,
            signal=signal,
            channel_ids=tuple(channel_ids),
            t_start=float(t_lo),
            t_end=float(t_hi),
        )
        results: list[dict] = []
        for idx, (eid, cid) in enumerate(zip(kept_eids, channel_ids)):
            results.append(
                {
                    "electrode_id": int(eid),
                    "channel_id": cid,
                    "time_sec": time_arr,
                    "value": traces[:, idx],
                }
            )
        return results

    def clear_trace_cache(self) -> None:
        _read_trace_json_cached.cache_clear()
        _lazy_lfp_window.cache_clear()


@lru_cache(maxsize=512)
def _read_trace_json_cached(path: str) -> dict:
    return json.loads(Path(path).read_text())


# -- Lazy SpikeInterface fallback ---------------------------------------


@lru_cache(maxsize=4)
def _open_prepared_recording(raw_path: str, well_id: str, rec_name: str | None):
    """Cache the (signed → bandpass → common-ref) chain per well."""
    from acute_slice_mea.recording import load_maxwell_recording, prepare_recordings

    raw = load_maxwell_recording(raw_path, well_id, rec_name=rec_name)
    return prepare_recordings(raw)


@lru_cache(maxsize=64)
def _lazy_lfp_window(
    *,
    raw_path: str,
    well_id: str,
    rec_name: str | None,
    signal: str,
    channel_ids: tuple[str, ...],
    t_start: float,
    t_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a (decimated) window of LFP/raw/spike traces lazily.

    Returns ``(time_sec, values)`` where ``values`` has shape
    ``(n_samples, len(channel_ids))`` in microvolts (or the recording's
    physical units — SpikeInterface returns float32 in chip units after
    bandpass+common-reference).
    """
    recordings = _open_prepared_recording(raw_path, well_id, rec_name)
    recording = recordings.get(signal) or recordings["lfp"]
    fs = float(recording.get_sampling_frequency())
    start_frame = max(0, int(round(t_start * fs)))
    end_frame = max(start_frame + 1, int(round(t_end * fs)))
    end_frame = min(end_frame, int(recording.get_num_frames()))

    traces = recording.get_traces(
        start_frame=start_frame,
        end_frame=end_frame,
        channel_ids=list(channel_ids),
        return_scaled=False,
    )
    traces = np.asarray(traces, dtype=np.float32)

    n_samples = traces.shape[0]
    # Decimate to ~_LAZY_TARGET_POINTS so plotly stays snappy.
    step = max(1, n_samples // _LAZY_TARGET_POINTS)
    if step > 1:
        traces = traces[::step]
    time_arr = (np.arange(traces.shape[0], dtype=np.float64) * step + start_frame) / fs
    return time_arr, traces
