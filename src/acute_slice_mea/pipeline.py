"""End-to-end cache generation pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from time import perf_counter

from acute_slice_mea.cache import save_cache_bundle
from acute_slice_mea.dashboard import export_dashboard_data
from acute_slice_mea.electrodes import build_electrode_table
from acute_slice_mea.plots import (
    write_band_power_html,
    write_spectrum_summary_html,
    write_trace_preview_html,
)
from acute_slice_mea.recording import load_maxwell_recording, prepare_recordings
from acute_slice_mea.spectral import (
    DEFAULT_LFP_BANDS,
    compute_lfp_band_power_over_time,
    compute_trace_preview,
    compute_welch_spectrum_summary,
)


@dataclass
class AnalysisConfig:
    data_path: str
    well_id: str
    output_dir: str
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
    spikeinterface_chunk_duration: str = "60s"
    apply_lfp_common_reference: bool = True
    apply_spike_common_reference: bool = True
    lfp_filter_margin_ms: int = 10000
    lfp_ignore_low_freq_error: bool = True


def run_analysis(config: AnalysisConfig) -> dict:
    """Run the heavy analysis pipeline and write cache outputs."""
    started = perf_counter()
    output_dir = Path(config.output_dir)
    figures_dir = output_dir / "figures"

    if config.spikeinterface_chunk_duration:
        import spikeinterface as si

        si.set_global_job_kwargs(chunk_duration=config.spikeinterface_chunk_duration)

    raw = load_maxwell_recording(config.data_path, config.well_id)
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
    )
    metadata_recording = recordings["raw"]
    fs = float(metadata_recording.get_sampling_frequency())
    num_samples = int(metadata_recording.get_num_samples())
    electrodes = build_electrode_table(metadata_recording.get_probe(), metadata_recording)
    recorded = electrodes[electrodes["recorded"].astype(bool)]

    band_power = compute_lfp_band_power_over_time(
        recordings["lfp"],
        electrode_table=electrodes,
        start_sec=0,
        end_sec=None,
        window_sec=config.lfp_window_sec,
        step_sec=config.lfp_step_sec,
        bands=DEFAULT_LFP_BANDS,
        welch_segment_sec=config.welch_segment_sec,
    )
    preview_electrode_ids = config.preview_electrode_ids
    if preview_electrode_ids is None and config.preview_max_electrodes is not None:
        preview_electrode_ids = recorded["electrode_id"].astype(int).head(config.preview_max_electrodes).tolist()
    spectrum = compute_welch_spectrum_summary(
        recordings["lfp"],
        channel_ids=recorded["channel_id"].tolist(),
        duration_sec=config.spectrum_duration_sec,
        max_freq_hz=config.spectrum_max_freq_hz,
        welch_segment_sec=config.welch_segment_sec,
    )
    trace_preview = compute_trace_preview(
        recordings,
        electrode_table=electrodes,
        electrode_ids=preview_electrode_ids,
        start_sec=config.preview_start_sec,
        duration_sec=config.preview_duration_sec,
        max_points=config.preview_max_points,
    )

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
    manifest = save_cache_bundle(
        output_dir,
        summary=summary,
        electrodes=electrodes,
        band_power=band_power,
        spectrum=spectrum,
        trace_preview=trace_preview,
    )
    figure_paths = {
        "band_power": str(write_band_power_html(band_power, figures_dir / "lfp_band_power.html")),
        "trace_preview": str(write_trace_preview_html(trace_preview, figures_dir / "trace_preview_raw_lfp_spike.html")),
        "spectrum": str(write_spectrum_summary_html(spectrum, figures_dir / "lfp_spectrum_summary.html")),
    }
    manifest["files"]["figures"] = figure_paths
    if config.export_dashboard_data:
        manifest["files"]["dashboard"] = export_dashboard_data(
            output_dir / "dashboard",
            recordings=recordings,
            electrodes=electrodes,
            band_power=band_power,
            summary=summary,
            max_points_per_electrode=config.dashboard_max_points_per_electrode,
        )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest
