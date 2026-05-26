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
    notch_freqs_hz=None,
    notch_q=30.0,
    downsample_hz=0,
    apply_lfp_common_reference=True,
    apply_spike_common_reference=True,
    lfp_reference_method="median",
    lfp_reference_scope="global",
    lfp_reference_inner_radius=30.0,
    lfp_reference_outer_radius=200.0,
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

    if notch_freqs_hz:
        for freq in notch_freqs_hz:
            lfp = spre.notch_filter(lfp, freq=float(freq), q=float(notch_q))

    if downsample_hz and downsample_hz > 0:
        lfp = spre.resample(lfp, resample_rate=int(downsample_hz))

    if apply_lfp_common_reference:
        operator = lfp_reference_method if lfp_reference_method in ("median", "average") else "median"
        if lfp_reference_scope == "local":
            lfp = spre.common_reference(
                lfp,
                reference="local",
                operator=operator,
                local_radius=(float(lfp_reference_inner_radius), float(lfp_reference_outer_radius)),
            )
        else:
            lfp = spre.common_reference(lfp, reference="global", operator=operator)
    if apply_spike_common_reference:
        spike = spre.common_reference(spike, reference="global", operator="median")
    return {"raw": signed, "lfp": lfp, "spike": spike}
