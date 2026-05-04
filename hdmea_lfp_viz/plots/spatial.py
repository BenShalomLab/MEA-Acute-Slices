"""Spatial band-power maps and band-envelope movie outputs."""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt
from tqdm.auto import tqdm

from hdmea_lfp_viz.summaries import BANDS
from hdmea_lfp_viz.style import CB_SAFE_CMAP, SEQUENTIAL_CMAP, add_caption, mmss, save_figure, set_equal_spatial_axes


def plot_band_power_maps(summaries: dict, locations: np.ndarray, figures_dir: str | Path) -> None:
    """Figure 04: spatial scatter plots for all band powers."""
    band_power = summaries["band_power"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("04 Spatial Band Power Maps")
    for ax, (name, values) in zip(axes.flat, band_power.items()):
        sc = ax.scatter(locations[:, 0], locations[:, 1], c=values, s=14, cmap=CB_SAFE_CMAP, linewidths=0)
        ax.set_title(name.replace("_", " "))
        set_equal_spatial_axes(ax, locations)
        cbar = fig.colorbar(sc, ax=ax, pad=0.01)
        cbar.set_label("Band power (µV²)")
    add_caption(fig, "Band powers are integrated from each channel's Welch PSD and plotted on the measured electrode coordinates.")
    save_figure(fig, Path(figures_dir), "04_band_power_maps")


def _band_envelope(
    recording,
    start_s: float,
    stop_s: float,
    *,
    pad_s: float = 1.0,
    band: tuple[float, float] = (30.0, 80.0),
    smooth_ms: float = 100.0,
) -> tuple[np.ndarray, float, int]:
    """Read a padded window and return a smoothed band-limited envelope."""
    sf = float(recording.get_sampling_frequency())
    n_samples = int(recording.get_num_samples())
    read_start = max(0, int(round((start_s - pad_s) * sf)))
    read_stop = min(n_samples, int(round((stop_s + pad_s) * sf)))
    traces = recording.get_traces(start_frame=read_start, end_frame=read_stop, return_scaled=True).astype(np.float32, copy=False)
    sos = butter(4, band, btype="bandpass", fs=sf, output="sos")
    filtered = sosfiltfilt(sos, traces, axis=0)
    env = np.abs(hilbert(filtered, axis=0)).astype(np.float32)
    smooth = max(1, int(round(smooth_ms * sf / 1000.0)))
    env = uniform_filter1d(env, size=smooth, axis=0, mode="nearest")
    return env, sf, read_start


def _envelope_at_times(recording, times_s: np.ndarray, *, band: tuple[float, float] = (30.0, 80.0)) -> np.ndarray:
    """Compute band-envelope vectors at requested timestamps."""
    if times_s.size == 0:
        return np.empty((0, recording.get_num_channels()), dtype=np.float32)
    duration_s = float(recording.get_num_samples() / recording.get_sampling_frequency())
    if float(times_s.max() - times_s.min()) > 30.0:
        frames = []
        for timestamp in tqdm(times_s, desc="Envelope snapshots", leave=False):
            start_s = max(0.0, float(timestamp) - 0.2)
            stop_s = min(duration_s, float(timestamp) + 0.2)
            env, sf, read_start = _band_envelope(recording, start_s, stop_s, band=band)
            idx = int(np.clip(round(timestamp * sf) - read_start, 0, env.shape[0] - 1))
            frames.append(env[idx])
        return np.asarray(frames, dtype=np.float32)
    start_s = max(0.0, float(times_s.min()) - 0.2)
    stop_s = min(duration_s, float(times_s.max()) + 0.2)
    env, sf, read_start = _band_envelope(recording, start_s, stop_s, band=band)
    frame_indices = np.clip((times_s * sf).round().astype(int) - read_start, 0, env.shape[0] - 1)
    return env[frame_indices]


def plot_band_envelope_frames(recording, locations: np.ndarray, figures_dir: str | Path) -> None:
    """Figure 09: 20 evenly spaced low-gamma envelope snapshots."""
    duration_s = float(recording.get_num_samples() / recording.get_sampling_frequency())
    times_s = np.linspace(0, max(0.0, duration_s), 20)
    frames = _envelope_at_times(recording, times_s)
    vmin, vmax = np.nanpercentile(frames, [2, 98])

    fig, axes = plt.subplots(4, 5, figsize=(15.5, 10))
    fig.suptitle("09 Low-Gamma Envelope Movie Frames")
    last_sc = None
    for ax, timestamp, values in zip(axes.flat, times_s, frames):
        last_sc = ax.scatter(
            locations[:, 0],
            locations[:, 1],
            c=values,
            s=10,
            cmap=SEQUENTIAL_CMAP,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
        )
        ax.set_title(mmss(timestamp, decimals=1))
        set_equal_spatial_axes(ax, locations)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(left=0.045, right=0.88, bottom=0.09, top=0.90, hspace=0.45, wspace=0.18)
    cax = fig.add_axes([0.91, 0.18, 0.018, 0.64])
    cbar = fig.colorbar(last_sc, cax=cax)
    cbar.set_label("Low-gamma envelope (µV)")
    add_caption(fig, "Twenty evenly spaced frames show the 30-80 Hz Hilbert-envelope amplitude smoothed with a 100 ms window.")
    save_figure(fig, Path(figures_dir), "09_band_envelope_movie_frames")


def _estimate_movie_scale(recording, *, band: tuple[float, float], samples: int = 40) -> tuple[float, float]:
    duration_s = float(recording.get_num_samples() / recording.get_sampling_frequency())
    times = np.linspace(0, duration_s, min(samples, max(2, int(duration_s))))
    frames = _envelope_at_times(recording, times, band=band)
    return tuple(np.nanpercentile(frames, [2, 98]))


def save_band_envelope_movie(
    recording,
    locations: np.ndarray,
    figures_dir: str | Path,
    *,
    band_name: str = "low_gamma",
    band: tuple[float, float] = (30.0, 80.0),
    fps: int = 30,
    playback_speed: float = 10.0,
    chunk_seconds: float = 10.0,
    write_legacy_low_gamma_alias: bool = False,
) -> Path:
    """Write one band-limited spatial envelope animation as MP4."""
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError("Matplotlib ffmpeg writer is not available. Install ffmpeg or rerun with --skip-movie.")
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    movie_path = figures_dir / f"band_envelope_{band_name}.mp4"
    duration_s = float(recording.get_num_samples() / recording.get_sampling_frequency())
    vmin, vmax = _estimate_movie_scale(recording, band=band)
    label = band_name.replace("_", " ").title()

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(f"{label} Envelope")
    initial = np.zeros(recording.get_num_channels(), dtype=np.float32)
    sc = ax.scatter(locations[:, 0], locations[:, 1], c=initial, s=16, cmap=SEQUENTIAL_CMAP, vmin=vmin, vmax=vmax, linewidths=0)
    set_equal_spatial_axes(ax, locations)
    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label(f"{label} envelope (µV)")
    title = ax.set_title("00:00.0")

    writer = FFMpegWriter(fps=fps, metadata={"title": f"HD-MEA {label} envelope ({playback_speed:g}x)"})
    frame_times = np.arange(0, duration_s, playback_speed / fps)
    with writer.saving(fig, str(movie_path), dpi=150):
        for chunk_start in tqdm(np.arange(0, duration_s, chunk_seconds), desc=f"{label} movie"):
            chunk_stop = min(duration_s, float(chunk_start + chunk_seconds))
            mask = (frame_times >= chunk_start) & (frame_times < chunk_stop)
            times = frame_times[mask]
            if times.size == 0:
                continue
            env, sf, read_start = _band_envelope(recording, float(chunk_start), chunk_stop, band=band)
            indices = np.clip((times * sf).round().astype(int) - read_start, 0, env.shape[0] - 1)
            for timestamp, idx in zip(times, indices):
                sc.set_array(env[idx])
                title.set_text(mmss(timestamp, decimals=1))
                writer.grab_frame()
    plt.close(fig)
    print(f"[figures] Wrote {movie_path}")
    if write_legacy_low_gamma_alias and band_name == "low_gamma":
        legacy_path = figures_dir / "band_envelope.mp4"
        copyfile(movie_path, legacy_path)
        print(f"[figures] Wrote {legacy_path}")
    return movie_path


def save_band_envelope_movies(
    recording,
    locations: np.ndarray,
    figures_dir: str | Path,
    *,
    fps: int = 30,
    playback_speed: float = 10.0,
    chunk_seconds: float = 10.0,
) -> list[Path]:
    """Write one spatial envelope MP4 for every canonical LFP band."""
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError("Matplotlib ffmpeg writer is not available. Install ffmpeg or rerun with --skip-movie.")
    paths = []
    for band_name, band in BANDS.items():
        paths.append(
            save_band_envelope_movie(
                recording,
                locations,
                figures_dir,
                band_name=band_name,
                band=band,
                fps=fps,
                playback_speed=playback_speed,
                chunk_seconds=chunk_seconds,
                write_legacy_low_gamma_alias=True,
            )
        )
    return paths
