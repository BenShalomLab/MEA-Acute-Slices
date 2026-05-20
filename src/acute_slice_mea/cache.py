"""Cache persistence for computed MEA analysis outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _write_table(df: pd.DataFrame, path_base: Path) -> Path:
    parquet_path = path_base.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        csv_path = path_base.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def save_cache_bundle(
    output_dir,
    *,
    summary: dict,
    electrodes: pd.DataFrame,
    band_power: pd.DataFrame,
    spectrum: dict[str, np.ndarray] | None = None,
    trace_preview: dict[str, np.ndarray] | None = None,
    bursts: list[dict] | None = None,
    probe_geometry: dict | None = None,
) -> dict:
    """Save analysis outputs and return a manifest dictionary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    electrodes_path = output_dir / "electrodes.csv"
    electrodes.to_csv(electrodes_path, index=False)
    band_power_path = _write_table(band_power, output_dir / "lfp_band_power")

    files = {
        "summary": str(summary_path),
        "electrodes": str(electrodes_path),
        "band_power": str(band_power_path),
    }
    if spectrum is not None:
        spectrum_path = output_dir / "spectrum_summary.npz"
        np.savez_compressed(spectrum_path, **spectrum)
        files["spectrum"] = str(spectrum_path)
    if trace_preview is not None:
        trace_preview_path = output_dir / "trace_preview.npz"
        np.savez_compressed(trace_preview_path, **trace_preview)
        files["trace_preview"] = str(trace_preview_path)
    if bursts is not None:
        bursts_path = output_dir / "bursts.json"
        bursts_path.write_text(json.dumps(bursts, separators=(",", ":")))
        files["bursts"] = str(bursts_path)
    if probe_geometry is not None:
        probe_path = output_dir / "probe.json"
        probe_path.write_text(json.dumps(probe_geometry, separators=(",", ":")))
        files["probe"] = str(probe_path)

    manifest = {
        "summary": summary,
        "files": files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def load_cache_manifest(cache_dir) -> dict:
    """Load manifest and summary metadata from a cache directory."""
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    summary_path = Path(manifest["files"].get("summary", cache_dir / "summary.json"))
    manifest["summary"] = json.loads(summary_path.read_text())
    return manifest


def load_band_power(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_npz_dict(path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}
