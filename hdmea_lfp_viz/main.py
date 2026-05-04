"""CLI entry point for the HD-MEA LFP exploratory visualization pipeline."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("cache") / "matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((Path("cache") / "xdg").resolve()))

import numpy as np

from hdmea_lfp_viz import preprocess
from hdmea_lfp_viz.plots.connectivity import plot_correlation_matrix
from hdmea_lfp_viz.plots.decomposition import plot_pca_components
from hdmea_lfp_viz.plots.overview import plot_channel_quality, plot_overview_heatmap, plot_summary_panel
from hdmea_lfp_viz.plots.spatial import plot_band_envelope_frames, plot_band_power_maps, save_band_envelope_movies
from hdmea_lfp_viz.plots.spectral import plot_psd_grid, plot_spectrograms
from hdmea_lfp_viz.plots.traces import plot_representative_traces
from hdmea_lfp_viz.style import apply_style
from hdmea_lfp_viz.summaries import bad_channel_indices, load_summaries, recording_metadata, save_summaries


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def _load_cache_metadata(cache_dir: Path) -> dict:
    metadata_path = cache_dir / "lfp_1khz_metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text())
    except Exception as exc:
        return {"cache_metadata_error": str(exc)}


def get_locations(recording) -> np.ndarray:
    """Read electrode positions from the recording."""
    locations = np.asarray(recording.get_channel_locations(), dtype=float)
    if locations.ndim != 2 or locations.shape[1] < 2:
        raise ValueError("Recording did not provide channel locations with x/y coordinates.")
    return locations[:, :2]


def generate_figures(recording, summaries: dict, figures_dir: Path, *, skip_movie: bool = False) -> None:
    """Run all Stage 3 figure functions."""
    locations = get_locations(recording)
    plot_overview_heatmap(summaries, figures_dir)
    print(f"[figures] Wrote {figures_dir / '01_overview_heatmap.png'}")
    plot_channel_quality(summaries, locations, figures_dir)
    print(f"[figures] Wrote {figures_dir / '02_channel_quality.png'}")
    plot_psd_grid(summaries, figures_dir)
    print(f"[figures] Wrote {figures_dir / '03_psd_grid.png'}")
    plot_band_power_maps(summaries, locations, figures_dir)
    print(f"[figures] Wrote {figures_dir / '04_band_power_maps.png'}")
    plot_correlation_matrix(summaries, figures_dir)
    print(f"[figures] Wrote {figures_dir / '05_correlation_matrix.png'}")
    plot_pca_components(recording, summaries, locations, figures_dir)
    print(f"[figures] Wrote {figures_dir / '06_pca_components.png'}")
    plot_representative_traces(recording, summaries, figures_dir)
    print(f"[figures] Wrote {figures_dir / '07_representative_traces.png'}")
    plot_spectrograms(recording, summaries, figures_dir)
    print(f"[figures] Wrote {figures_dir / '08_spectrograms.png'}")
    plot_band_envelope_frames(recording, locations, figures_dir)
    print(f"[figures] Wrote {figures_dir / '09_band_envelope_movie_frames.png'}")
    plot_summary_panel(summaries, locations, figures_dir)
    print(f"[figures] Wrote {figures_dir / '10_summary_panel.png'}")
    if not skip_movie:
        save_band_envelope_movies(recording, locations, figures_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_path", nargs="?", help="Path to Maxwell .h5 recording. Falls back to RAW_PATH in preprocess.py.")
    parser.add_argument("--stream-id", default=None, help="Optional Maxwell stream_id/well id for read_maxwell.")
    parser.add_argument("--cache-dir", default="cache", help="Cache directory. Default: ./cache")
    parser.add_argument("--figures-dir", default="figures", help="Figure output directory. Default: ./figures")
    parser.add_argument("--skip-preprocess", action="store_true", help="Use existing ./cache/lfp_1khz.zarr.")
    parser.add_argument("--skip-summaries", action="store_true", help="Use existing ./cache/summaries.npz.")
    parser.add_argument("--figures-only", action="store_true", help="Equivalent to --skip-preprocess --skip-summaries.")
    parser.add_argument("--skip-movie", action="store_true", help="Skip the slow MP4 band-envelope animations.")
    parser.add_argument("--overwrite-cache", action="store_true", help="Overwrite existing Zarr and summary caches.")
    parser.add_argument("--n-jobs", type=int, default=1, help="SpikeInterface jobs for Zarr saving.")
    parser.add_argument(
        "--spikeinterface-chunk-duration",
        default=preprocess.DEFAULT_SPIKEINTERFACE_CHUNK_DURATION,
        help="SpikeInterface global chunk duration used while constructing preprocessing. Default: 60s",
    )
    parser.add_argument("--psd-batch-channels", type=int, default=100, help="Channels per Welch PSD batch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_style()
    cache_dir = Path(args.cache_dir)
    figures_dir = Path(args.figures_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw_value = args.raw_path if args.raw_path is not None else preprocess.RAW_PATH
    raw_path = Path(raw_value) if raw_value else None
    if args.figures_only:
        args.skip_preprocess = True
        args.skip_summaries = True
    if not args.skip_preprocess and raw_path is None:
        raise SystemExit("A raw Maxwell .h5 path is required unless --skip-preprocess or --figures-only is used.")

    timings = {}
    stage_start = time.perf_counter()
    if args.skip_preprocess:
        zarr_path = cache_dir / "lfp_1khz.zarr"
        if not zarr_path.exists():
            raise FileNotFoundError(f"Missing cache: {zarr_path}")
        print(f"[preprocess] Skipped; using {zarr_path}")
    else:
        zarr_path = preprocess.save_lfp_cache(
            raw_path,
            cache_dir=cache_dir,
            stream_id=args.stream_id,
            overwrite=args.overwrite_cache,
            n_jobs=args.n_jobs,
            spikeinterface_chunk_duration=args.spikeinterface_chunk_duration,
        )
    timings["preprocess_s"] = time.perf_counter() - stage_start

    recording = preprocess.load_lfp_cache(cache_dir)

    stage_start = time.perf_counter()
    if args.skip_summaries:
        summaries_path = cache_dir / "summaries.npz"
        if not summaries_path.exists():
            raise FileNotFoundError(f"Missing summaries: {summaries_path}")
        print(f"[summaries] Skipped; using {summaries_path}")
    else:
        save_summaries(
            recording,
            cache_dir=cache_dir,
            force=args.overwrite_cache,
            batch_channels=args.psd_batch_channels,
        )
    summaries = load_summaries(cache_dir)
    timings["summaries_s"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    generate_figures(recording, summaries, figures_dir, skip_movie=args.skip_movie)
    timings["figures_s"] = time.perf_counter() - stage_start

    metadata = recording_metadata(recording)
    metadata.update(_load_cache_metadata(cache_dir))
    bad = bad_channel_indices(summaries["rms_per_channel"])
    report = {
        "n_channels": metadata["n_channels"],
        "duration_s": metadata["duration_s"],
        "sampling_frequency_hz_decimated": metadata["sampling_frequency_hz"],
        "sampling_frequency_hz_raw": metadata.get("raw_sampling_frequency_hz", "unknown"),
        "bad_channel_count": int(bad.size),
        "bad_channel_indices": bad.tolist(),
        "runtime_s": timings,
        "cache_size": _human_bytes(_dir_size(cache_dir)),
        "figures_size": _human_bytes(_dir_size(figures_dir)),
    }
    (figures_dir / "run_report.json").write_text(json.dumps(report, indent=2))
    print("\nQuality report")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
