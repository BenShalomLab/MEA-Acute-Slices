"""Stage 2: cached summary statistics for LFP visualization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import welch
from sklearn.decomposition import IncrementalPCA
from tqdm.auto import tqdm


BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 80.0),
    "high_gamma": (80.0, 150.0),
}


def _sampling_frequency(recording) -> float:
    return float(recording.get_sampling_frequency())


def channel_ids_from_indices(recording, channel_indices=None):
    """Map positional channel indices to SpikeInterface channel IDs."""
    if channel_indices is None:
        return None
    channel_ids = np.asarray(recording.get_channel_ids())
    return channel_ids[np.asarray(channel_indices, dtype=int)]


def _scaled_traces(recording, start_frame: int, end_frame: int, channel_indices=None) -> np.ndarray:
    return recording.get_traces(
        start_frame=int(start_frame),
        end_frame=int(end_frame),
        channel_ids=channel_ids_from_indices(recording, channel_indices),
        return_scaled=True,
    ).astype(np.float32, copy=False)


def recording_metadata(recording) -> dict:
    """Return basic recording metadata used by CLI reports and figures."""
    sf = _sampling_frequency(recording)
    return {
        "n_channels": int(recording.get_num_channels()),
        "n_samples": int(recording.get_num_samples()),
        "sampling_frequency_hz": sf,
        "duration_s": float(recording.get_num_samples() / sf),
    }


def compute_rms_timebins(recording, *, bin_seconds: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel RMS and per-channel 1-second RMS bins."""
    sf = _sampling_frequency(recording)
    n_samples = int(recording.get_num_samples())
    n_channels = int(recording.get_num_channels())
    bin_size = max(1, int(round(bin_seconds * sf)))
    n_bins = int(np.ceil(n_samples / bin_size))
    rms_timebins = np.empty((n_channels, n_bins), dtype=np.float32)
    total_sumsq = np.zeros(n_channels, dtype=np.float64)
    total_count = 0

    for i in tqdm(range(n_bins), desc="RMS time bins"):
        start = i * bin_size
        stop = min(n_samples, start + bin_size)
        traces = _scaled_traces(recording, start, stop)
        sumsq = np.sum(np.square(traces, dtype=np.float64), axis=0)
        rms_timebins[:, i] = np.sqrt(sumsq / max(stop - start, 1)).astype(np.float32)
        total_sumsq += sumsq
        total_count += stop - start

    rms_per_channel = np.sqrt(total_sumsq / max(total_count, 1)).astype(np.float32)
    return rms_per_channel, rms_timebins


def compute_psd(recording, *, batch_channels: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel Welch PSD up to 300 Hz in channel batches."""
    sf = _sampling_frequency(recording)
    n_channels = int(recording.get_num_channels())
    psd_parts = []
    freqs = None
    for start_ch in tqdm(range(0, n_channels, batch_channels), desc="Welch PSD"):
        stop_ch = min(n_channels, start_ch + batch_channels)
        traces = _scaled_traces(recording, 0, recording.get_num_samples(), np.arange(start_ch, stop_ch))
        f, pxx = welch(
            traces.T,
            fs=sf,
            nperseg=2048,
            noverlap=1024,
            axis=-1,
            detrend="constant",
            scaling="density",
        )
        keep = f <= 300.0
        freqs = f[keep].astype(np.float32)
        psd_parts.append(pxx[:, keep].astype(np.float32))
    return freqs, np.vstack(psd_parts)


def compute_band_power(freqs: np.ndarray, psd: np.ndarray) -> dict[str, np.ndarray]:
    """Integrate PSD in canonical LFP bands."""
    band_power = {}
    for name, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs < high)
        if mask.sum() < 2:
            band_power[name] = np.full(psd.shape[0], np.nan, dtype=np.float32)
        else:
            band_power[name] = np.trapz(psd[:, mask], freqs[mask], axis=1).astype(np.float32)
    return band_power


def compute_corr_matrix(recording, *, chunk_seconds: float = 10.0) -> np.ndarray:
    """Compute channel Pearson correlation by streaming covariance chunks."""
    sf = _sampling_frequency(recording)
    n_samples = int(recording.get_num_samples())
    n_channels = int(recording.get_num_channels())
    chunk_size = max(1, int(round(chunk_seconds * sf)))
    sums = np.zeros(n_channels, dtype=np.float64)
    cross = np.zeros((n_channels, n_channels), dtype=np.float64)
    count = 0
    for start in tqdm(range(0, n_samples, chunk_size), desc="Correlation"):
        stop = min(n_samples, start + chunk_size)
        x = _scaled_traces(recording, start, stop).astype(np.float64, copy=False)
        sums += x.sum(axis=0)
        cross += x.T @ x
        count += x.shape[0]
    mean = sums / max(count, 1)
    cov = (cross - count * np.outer(mean, mean)) / max(count - 1, 1)
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    denom = np.outer(std, std)
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0).astype(np.float32)


def compute_pca(recording, *, n_components: int = 10, chunk_seconds: float = 10.0) -> dict[str, np.ndarray]:
    """Run IncrementalPCA on samples x channels and store spatial PCs plus time courses."""
    sf = _sampling_frequency(recording)
    n_samples = int(recording.get_num_samples())
    n_components = min(int(n_components), int(recording.get_num_channels()), max(1, n_samples - 1))
    chunk_size = max(n_components + 1, int(round(chunk_seconds * sf)))
    pca = IncrementalPCA(n_components=n_components)

    for start in tqdm(range(0, n_samples, chunk_size), desc="PCA fit"):
        stop = min(n_samples, start + chunk_size)
        if stop - start <= n_components:
            continue
        pca.partial_fit(_scaled_traces(recording, start, stop))

    time_courses = np.empty((n_components, n_samples), dtype=np.float32)
    for start in tqdm(range(0, n_samples, chunk_size), desc="PCA transform"):
        stop = min(n_samples, start + chunk_size)
        scores = pca.transform(_scaled_traces(recording, start, stop)).astype(np.float32)
        time_courses[:, start:stop] = scores.T

    return {
        "components": pca.components_.astype(np.float32),
        "time_courses": time_courses,
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
        "mean": pca.mean_.astype(np.float32),
    }


def save_summaries(
    recording,
    *,
    cache_dir: str | Path = "cache",
    force: bool = False,
    batch_channels: int = 100,
) -> Path:
    """Compute and cache all Stage 2 summaries."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "summaries.npz"
    meta_path = cache_dir / "summaries_metadata.json"
    if path.exists() and not force:
        print(f"[summaries] Reusing cached summaries: {path}")
        return path

    rms_per_channel, rms_timebins = compute_rms_timebins(recording)
    freqs, psd = compute_psd(recording, batch_channels=batch_channels)
    band_power = compute_band_power(freqs, psd)
    corr_matrix = compute_corr_matrix(recording)
    pca = compute_pca(recording)

    arrays = {
        "rms_per_channel": rms_per_channel,
        "rms_timebins": rms_timebins,
        "freqs": freqs,
        "psd": psd,
        "corr_matrix": corr_matrix,
        "pca_components": pca["components"],
        "pca_time_courses": pca["time_courses"],
        "pca_explained_variance_ratio": pca["explained_variance_ratio"],
        "pca_mean": pca["mean"],
        "band_names": np.asarray(list(BANDS), dtype=object),
    }
    arrays.update({f"band_power_{name}": values for name, values in band_power.items()})
    np.savez_compressed(path, **arrays)
    meta_path.write_text(json.dumps(recording_metadata(recording), indent=2))
    print(f"[summaries] Wrote {path}")
    return path


def load_summaries(cache_dir: str | Path = "cache") -> dict:
    """Load Stage 2 summary arrays and reconstruct grouped dictionaries."""
    path = Path(cache_dir) / "summaries.npz"
    data = np.load(path, allow_pickle=True)
    band_names = [str(x) for x in data["band_names"].tolist()]
    return {
        "rms_per_channel": data["rms_per_channel"],
        "rms_timebins": data["rms_timebins"],
        "freqs": data["freqs"],
        "psd": data["psd"],
        "band_power": {name: data[f"band_power_{name}"] for name in band_names},
        "corr_matrix": data["corr_matrix"],
        "pca": {
            "components": data["pca_components"],
            "time_courses": data["pca_time_courses"],
            "explained_variance_ratio": data["pca_explained_variance_ratio"],
            "mean": data["pca_mean"],
        },
    }


def bad_channel_indices(rms_per_channel: np.ndarray) -> np.ndarray:
    """Flag channels by robust high RMS or very low RMS thresholds."""
    rms = np.asarray(rms_per_channel)
    median = float(np.nanmedian(rms))
    mad = float(np.nanmedian(np.abs(rms - median)))
    high = rms > median + 5.0 * mad
    low = rms < 0.1 * median
    return np.flatnonzero(high | low)
