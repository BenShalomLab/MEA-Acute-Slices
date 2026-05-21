"""Stage 1: Maxwell loading, lazy preprocessing, and Zarr caching."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


RAW_PATH = None
DEFAULT_SPIKEINTERFACE_CHUNK_DURATION = "60s"
DEFAULT_NOTCH_FREQUENCIES = (60.0, 120.0, 180.0)
DEFAULT_NOTCH_Q = 30.0


def _path_mtime(path: Path) -> float:
    if path.is_dir():
        mtimes = [p.stat().st_mtime for p in path.rglob("*") if p.is_file()]
        return max(mtimes) if mtimes else path.stat().st_mtime
    return path.stat().st_mtime


def cache_is_fresh(raw_path: Path, zarr_path: Path) -> bool:
    """Return True when the Zarr cache exists and is newer than the raw file."""
    return zarr_path.exists() and _path_mtime(zarr_path) >= raw_path.stat().st_mtime


def rec_names_for_well(raw_path: str | Path, stream_id: str) -> list[str]:
    """Return recording IDs available under one Maxwell well."""
    import h5py

    with h5py.File(raw_path, "r") as h5:
        wells = h5.get("wells")
        if wells is None:
            raise ValueError(f"{raw_path} does not contain a /wells group.")
        if stream_id not in wells:
            available = sorted(wells.keys())
            raise ValueError(
                f"Well {stream_id!r} was not found in {raw_path}. "
                f"Available wells: {available}"
            )
        return sorted(wells[stream_id].keys())


def resolve_rec_name(raw_path: str | Path, stream_id: str, rec_name: str | None = None) -> str:
    """Resolve the Maxwell rec_name for a selected well."""
    rec_names = rec_names_for_well(raw_path, stream_id)
    if rec_name is not None:
        if rec_name not in rec_names:
            raise ValueError(
                f"Recording {rec_name!r} was not found under well {stream_id!r}. "
                f"Available rec_names: {rec_names}"
            )
        return rec_name
    if len(rec_names) == 1:
        return rec_names[0]
    raise ValueError(
        f"Well {stream_id!r} contains multiple recording IDs. "
        f"Pass --rec-name with one of: {rec_names}"
    )


def read_maxwell(raw_path: str | Path, stream_id: str | None = None, rec_name: str | None = None):
    """Read a Maxwell HDF5 recording through SpikeInterface."""
    import spikeinterface.extractors as se

    kwargs = {}
    if stream_id is not None:
        stream_id = str(stream_id)
        rec_name = resolve_rec_name(raw_path, stream_id, rec_name)
        kwargs["stream_id"] = stream_id
        kwargs["rec_name"] = rec_name
    elif rec_name is not None:
        kwargs["rec_name"] = str(rec_name)
    return se.read_maxwell(str(raw_path), **kwargs)


def normalize_notch_frequencies(notch_frequencies) -> tuple[float, ...]:
    """Normalize notch frequency configuration to a tuple of Hz values."""
    if notch_frequencies is None:
        return ()
    if isinstance(notch_frequencies, str):
        text = notch_frequencies.strip()
        if not text:
            return ()
        return tuple(float(part.strip()) for part in text.split(",") if part.strip())
    return tuple(float(freq) for freq in notch_frequencies)


def build_lfp_preprocessing(
    recording,
    *,
    notch_frequencies=DEFAULT_NOTCH_FREQUENCIES,
    notch_q: float = DEFAULT_NOTCH_Q,
):
    """Build the requested lazy LFP preprocessing chain."""
    import spikeinterface.preprocessing as spre

    from acute_slice_mea.recording import safe_bandpass_filter

    notch_frequencies = normalize_notch_frequencies(notch_frequencies)
    rec = safe_bandpass_filter(
        recording,
        freq_min=0.5,
        freq_max=300.0,
        margin_ms="auto",
        dtype="float32",
        ignore_low_freq_error=True,
        filter_order=4,
        filter_mode="sos",
    )
    for freq in notch_frequencies:
        rec = spre.notch_filter(rec, freq=freq, q=notch_q, dtype="float32")
    rec = spre.common_reference(rec, reference="global", operator="median", dtype="float32")
    rec = spre.resample(rec, resample_rate=1000, dtype="float32")
    return rec


def configure_spikeinterface_jobs(*, chunk_duration: str | None, n_jobs: int) -> None:
    """Set SpikeInterface chunking before constructing preprocessing wrappers."""
    if not chunk_duration:
        return

    import spikeinterface as si

    si.set_global_job_kwargs(chunk_duration=chunk_duration, n_jobs=n_jobs)


def save_lfp_cache(
    raw_path: str | Path,
    *,
    cache_dir: str | Path = "cache",
    stream_id: str | None = None,
    rec_name: str | None = None,
    overwrite: bool = False,
    n_jobs: int = 1,
    spikeinterface_chunk_duration: str | None = DEFAULT_SPIKEINTERFACE_CHUNK_DURATION,
    notch_frequencies=DEFAULT_NOTCH_FREQUENCIES,
    notch_q: float = DEFAULT_NOTCH_Q,
) -> Path:
    """Preprocess a Maxwell recording and save the 1 kHz LFP Zarr cache."""
    raw_path = Path(raw_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = cache_dir / "lfp_1khz.zarr"
    metadata_path = cache_dir / "lfp_1khz_metadata.json"

    if not overwrite and cache_is_fresh(raw_path, zarr_path):
        print(f"[preprocess] Reusing fresh cache: {zarr_path}")
        return zarr_path

    resolved_rec_name = resolve_rec_name(raw_path, str(stream_id), rec_name) if stream_id is not None else rec_name
    recording_raw = read_maxwell(raw_path, stream_id=stream_id, rec_name=resolved_rec_name)
    raw_sampling_frequency = float(recording_raw.get_sampling_frequency())
    configure_spikeinterface_jobs(chunk_duration=spikeinterface_chunk_duration, n_jobs=n_jobs)
    notch_frequencies = normalize_notch_frequencies(notch_frequencies)
    notch_q = float(notch_q)
    lfp = build_lfp_preprocessing(
        recording_raw,
        notch_frequencies=notch_frequencies,
        notch_q=notch_q,
    )
    n_channels = int(lfp.get_num_channels())

    if zarr_path.exists() and overwrite:
        shutil.rmtree(zarr_path)
    if zarr_path.exists():
        raise FileExistsError(
            f"{zarr_path} exists but is stale. Remove it or rerun with --overwrite-cache."
        )

    lfp.save(
        format="zarr",
        folder=zarr_path,
        channel_chunk_size=n_channels,
        chunk_size=10_000,
        n_jobs=n_jobs,
        progress_bar=True,
        verbose=True,
    )
    metadata = {
        "raw_path": str(raw_path.resolve()),
        "stream_id": stream_id,
        "rec_name": resolved_rec_name,
        "raw_sampling_frequency_hz": raw_sampling_frequency,
        "sampling_frequency_hz": float(lfp.get_sampling_frequency()),
        "n_channels": n_channels,
        "n_samples": int(lfp.get_num_samples()),
        "duration_s": float(lfp.get_num_samples() / lfp.get_sampling_frequency()),
        "zarr_chunks_samples_channels": [10_000, n_channels],
        "logical_chunking_channels_samples": [n_channels, 10_000],
        "spikeinterface_chunk_duration": spikeinterface_chunk_duration,
        "preprocessing": {
            "bandpass_hz": [0.5, 300.0],
            "butterworth_order": 4,
            "filter_mode": "sos",
            "notch_hz": list(notch_frequencies),
            "notch_q": notch_q,
            "reference": "global median",
            "resample_hz": 1000.0,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"[preprocess] Wrote {zarr_path}")
    return zarr_path


def load_lfp_cache(cache_dir: str | Path = "cache"):
    """Load the saved 1 kHz LFP Zarr recording."""
    import spikeinterface.extractors as se

    return se.read_zarr(Path(cache_dir) / "lfp_1khz.zarr")
