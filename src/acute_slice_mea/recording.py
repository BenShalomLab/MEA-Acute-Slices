"""SpikeInterface recording loading and preprocessing."""

from __future__ import annotations


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
    lfp_reference_scope="global",
    lfp_reference_local_radius=(30.0, 70.0),
    lfp_filter_margin_ms=10000,
    lfp_ignore_low_freq_error=True,
):
    """Prepare signed raw, LFP-filtered, and spike-filtered recording views."""
    import spikeinterface.preprocessing as spre

    signed = spre.unsigned_to_signed(recording_raw)
    lfp = spre.bandpass_filter(
        signed,
        freq_min=lfp_low_hz,
        freq_max=lfp_high_hz,
        margin_ms=lfp_filter_margin_ms,
        ignore_low_freq_error=lfp_ignore_low_freq_error,
    )
    spike = spre.bandpass_filter(signed, freq_min=spike_low_hz, freq_max=spike_high_hz)
    if apply_lfp_common_reference:
        if lfp_reference_scope == "local":
            lfp = spre.common_reference(
                lfp, reference="local", operator="median",
                local_radius=lfp_reference_local_radius,
            )
        else:
            lfp = spre.common_reference(lfp, reference="global", operator="median")
    if apply_spike_common_reference:
        spike = spre.common_reference(spike, reference="global", operator="median")
    return {"raw": signed, "lfp": lfp, "spike": spike}
