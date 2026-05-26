"""SpikeInterface recording loading and preprocessing."""

from __future__ import annotations

import logging
import os

import psutil

_proc = psutil.Process(os.getpid())
_dbg = logging.getLogger("mem_debug")


def _mem(label: str) -> float:
    rss_gb = _proc.memory_info().rss / (1024**3)
    _dbg.warning("MEM %-50s  RSS=%.2f GB", label, rss_gb)
    return rss_gb


def load_maxwell_recording(data_path, well_id, rec_name=None):
    """Load a Maxwell raw HDF5 recording for one (well, rec_name)."""
    import spikeinterface.extractors as se

    return se.read_maxwell(str(data_path), stream_id=str(well_id), rec_name=rec_name)


def prepare_recordings(
    recording_raw,
    *,
    lfp_low_hz=0.5,
    lfp_high_hz=300,
    spike_low_hz=300,
    spike_high_hz=3000,
    apply_lfp_common_reference=True,
    apply_spike_common_reference=True,
    lfp_filter_margin_ms=10000,
    lfp_ignore_low_freq_error=True,
):
    """Prepare signed raw, LFP-filtered, and spike-filtered recording views."""
    import spikeinterface.preprocessing as spre

    _mem(f"prepare_rec  START  n_ch={recording_raw.get_num_channels()} margin_ms={lfp_filter_margin_ms}")
    signed = spre.unsigned_to_signed(recording_raw)
    _mem("prepare_rec  AFTER unsigned_to_signed (lazy)")
    lfp = spre.bandpass_filter(
        signed,
        freq_min=lfp_low_hz,
        freq_max=lfp_high_hz,
        margin_ms=lfp_filter_margin_ms,
        ignore_low_freq_error=lfp_ignore_low_freq_error,
    )
    _mem("prepare_rec  AFTER bandpass LFP (lazy)")
    spike = spre.bandpass_filter(signed, freq_min=spike_low_hz, freq_max=spike_high_hz)
    _mem("prepare_rec  AFTER bandpass spike (lazy)")
    if apply_lfp_common_reference:
        lfp = spre.common_reference(lfp, reference="global", operator="median")
        _mem("prepare_rec  AFTER common_ref LFP (lazy)")
    if apply_spike_common_reference:
        spike = spre.common_reference(spike, reference="global", operator="median")
        _mem("prepare_rec  AFTER common_ref spike (lazy)")
    return {"raw": signed, "lfp": lfp, "spike": spike}
