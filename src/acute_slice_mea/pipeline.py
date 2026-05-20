"""End-to-end cache generation pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

from acute_slice_mea.bursts import compute_bursts_from_recording
from acute_slice_mea.cache import save_cache_bundle
from acute_slice_mea.dashboard import export_dashboard_data
from acute_slice_mea.electrodes import build_electrode_table
from acute_slice_mea.plots import (
    write_band_power_html,
    write_spectrum_summary_html,
    write_trace_preview_html,
)
from acute_slice_mea.probe_geometry import build_probe_geometry
from acute_slice_mea.recording import (
    load_maxwell_recording,
    materialize_lfp,
    prepare_recordings,
)
from acute_slice_mea.spectral import (
    DEFAULT_LFP_BANDS,
    compute_lfp_band_power_over_time,
    compute_lfp_rms_per_electrode,
    compute_trace_preview,
    compute_welch_spectrum_summary,
)


@dataclass
class AnalysisConfig:
    data_path: str
    well_id: str
    output_dir: str
    rec_name: str | None = None
    lfp_low_hz: float = 0.5
    lfp_high_hz: float = 300
    spike_low_hz: float = 300
    spike_high_hz: float = 3000
    lfp_window_sec: float = 10
    lfp_step_sec: float = 5
    welch_segment_sec: float = 2
    spectrum_duration_sec: float = 10
    spectrum_max_freq_hz: float = 150
    preview_start_sec: float = 0
    preview_duration_sec: float = 10
    preview_max_points: int = 20000
    preview_max_electrodes: int | None = 16
    preview_electrode_ids: list[int] | None = field(default=None)
    dashboard_max_points_per_electrode: int = 20000
    export_dashboard_data: bool = True
    compute_spectrum: bool = True
    compute_trace_preview: bool = True
    compute_bursts: bool = True
    burst_detection_max_sec: float = 120.0
    export_probe_geometry: bool = True
    compute_rms_per_electrode: bool = True
    rms_window_sec: float = 10.0
    spikeinterface_chunk_duration: str = "60s"
    apply_lfp_common_reference: bool = True
    apply_spike_common_reference: bool = True
    lfp_filter_margin_ms: int = 10000
    lfp_ignore_low_freq_error: bool = True
    lfp_target_fs_hz: float | None = 1000.0
    lfp_target_fs_hz: float | None = 1000
    n_jobs: int = 1
    channel_chunk_size: int | None = None
    cache_lfp_to_disk: bool = True
    lfp_cache_dir: str | None = None
    keep_lfp_cache: bool = False
    lfp_save_n_jobs: int = 1
    lfp_save_chunk_duration: str = "10s"
    progress: bool = True
    verbose: bool = False


def _log_verbose(config: AnalysisConfig, message: str) -> None:
    if config.verbose:
        print(message, file=sys.stderr)


def run_analysis(config: AnalysisConfig) -> dict:
    """Run the heavy analysis pipeline and write cache outputs."""
    started = perf_counter()
    output_dir = Path(config.output_dir)
    figures_dir = output_dir / "figures"

    if config.spikeinterface_chunk_duration:
        _log_verbose(config, "Configuring SpikeInterface jobs")
        import spikeinterface as si

        si.set_global_job_kwargs(chunk_duration=config.spikeinterface_chunk_duration)

    _log_verbose(config, "Loading recording")
    raw = load_maxwell_recording(config.data_path, config.well_id, rec_name=config.rec_name)
    _log_verbose(config, "Preparing recordings")
    recordings = prepare_recordings(
        raw,
        lfp_low_hz=config.lfp_low_hz,
        lfp_high_hz=config.lfp_high_hz,
        spike_low_hz=config.spike_low_hz,
        spike_high_hz=config.spike_high_hz,
        apply_lfp_common_reference=config.apply_lfp_common_reference,
        apply_spike_common_reference=config.apply_spike_common_reference,
        lfp_filter_margin_ms=config.lfp_filter_margin_ms,
        lfp_ignore_low_freq_error=config.lfp_ignore_low_freq_error,
        lfp_target_fs_hz=config.lfp_target_fs_hz,
    )
    metadata_recording = recordings["lfp"]
    fs = float(metadata_recording.get_sampling_frequency())
    num_samples = int(metadata_recording.get_num_samples())
    _log_verbose(config, "Building electrode table")
    electrodes = build_electrode_table(metadata_recording.get_probe(), metadata_recording)
    recorded = electrodes[electrodes["recorded"].astype(bool)]

    lfp_cache_dir: Path | None = None
    if config.cache_lfp_to_disk:
        lfp_cache_dir = Path(config.lfp_cache_dir or (output_dir / ".lfp_cache"))
        _log_verbose(config, f"Materializing LFP to {lfp_cache_dir}")
        recordings["lfp"] = materialize_lfp(
            recordings["lfp"],
            lfp_cache_dir,
            save_n_jobs=config.lfp_save_n_jobs,
            save_chunk_duration=config.lfp_save_chunk_duration,
            progress=config.progress,
        )

    _log_verbose(config, "Computing LFP band power")
    band_power = compute_lfp_band_power_over_time(
        recordings["lfp"],
        electrode_table=electrodes,
        start_sec=0,
        end_sec=None,
        window_sec=config.lfp_window_sec,
        step_sec=config.lfp_step_sec,
        bands=DEFAULT_LFP_BANDS,
        welch_segment_sec=config.welch_segment_sec,
        n_jobs=config.n_jobs,
        channel_chunk_size=config.channel_chunk_size,
        progress=config.progress,
    )
    preview_electrode_ids = config.preview_electrode_ids
    if preview_electrode_ids is None and config.preview_max_electrodes is not None:
        preview_electrode_ids = recorded["electrode_id"].astype(int).head(config.preview_max_electrodes).tolist()

    spectrum: dict | None = None
    if config.compute_spectrum:
        _log_verbose(config, "Computing spectrum summary")
        spectrum = compute_welch_spectrum_summary(
            recordings["lfp"],
            channel_ids=recorded["channel_id"].tolist(),
            duration_sec=config.spectrum_duration_sec,
            max_freq_hz=config.spectrum_max_freq_hz,
            welch_segment_sec=config.welch_segment_sec,
            n_jobs=config.n_jobs,
            channel_chunk_size=config.channel_chunk_size,
            progress=config.progress,
        )

    trace_preview: dict | None = None
    if config.compute_trace_preview:
        _log_verbose(config, "Computing trace preview")
        trace_preview = compute_trace_preview(
            recordings,
            electrode_table=electrodes,
            electrode_ids=preview_electrode_ids,
            start_sec=config.preview_start_sec,
            duration_sec=config.preview_duration_sec,
            max_points=config.preview_max_points,
        )
    else:
        preview_electrode_ids = None

    bursts: list[dict] | None = None
    if config.compute_bursts:
        _log_verbose(config, "Detecting network bursts")
        bursts = compute_bursts_from_recording(
            recordings["lfp"],
            electrode_table=electrodes,
            duration_sec=config.burst_detection_max_sec,
        )

    rms_by_electrode: dict[int, float] = {}
    if config.compute_rms_per_electrode:
        _log_verbose(config, "Computing per-electrode RMS")
        rms_by_electrode = compute_lfp_rms_per_electrode(
            recordings["lfp"],
            electrode_table=electrodes,
            duration_sec=config.rms_window_sec,
        )
    if rms_by_electrode:
        electrodes = electrodes.assign(
            rms_uv=electrodes["electrode_id"].astype(int).map(rms_by_electrode)
        )

    probe_geometry: dict | None = None
    if config.export_probe_geometry:
        _log_verbose(config, "Building probe geometry")
        probe_geometry = build_probe_geometry(electrodes)

    elapsed_sec = perf_counter() - started
    summary = {
        **asdict(config),
        "sampling_frequency_hz": fs,
        "num_samples": num_samples,
        "duration_sec": num_samples / fs,
        "num_recorded_electrodes": int(len(recorded)),
        "num_probe_contacts": int(len(electrodes)),
        "elapsed_sec": elapsed_sec,
    }
    summary["preview_electrode_ids_resolved"] = preview_electrode_ids
    summary["num_bursts"] = None if bursts is None else len(bursts)
    _log_verbose(config, "Saving cache bundle")
    manifest = save_cache_bundle(
        output_dir,
        summary=summary,
        electrodes=electrodes,
        band_power=band_power,
        spectrum=spectrum,
        trace_preview=trace_preview,
        bursts=bursts,
        probe_geometry=probe_geometry,
    )
    _log_verbose(config, "Writing figures")
    figure_paths = {
        "band_power": str(write_band_power_html(band_power, figures_dir / "lfp_band_power.html")),
    }
    if trace_preview is not None:
        figure_paths["trace_preview"] = str(
            write_trace_preview_html(trace_preview, figures_dir / "trace_preview_raw_lfp_spike.html")
        )
    if spectrum is not None:
        figure_paths["spectrum"] = str(
            write_spectrum_summary_html(spectrum, figures_dir / "lfp_spectrum_summary.html")
        )
    manifest["files"]["figures"] = figure_paths
    if config.export_dashboard_data:
        _log_verbose(config, "Exporting dashboard data")
        manifest["files"]["dashboard"] = export_dashboard_data(
            output_dir / "dashboard",
            recordings=recordings,
            electrodes=electrodes,
            band_power=band_power,
            summary=summary,
            max_points_per_electrode=config.dashboard_max_points_per_electrode,
        )
    _log_verbose(config, "Writing manifest")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if lfp_cache_dir is not None and not config.keep_lfp_cache:
        _log_verbose(config, f"Removing LFP cache {lfp_cache_dir}")
        shutil.rmtree(lfp_cache_dir, ignore_errors=True)
    return manifest
