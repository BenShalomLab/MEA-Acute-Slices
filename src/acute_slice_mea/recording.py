"""SpikeInterface recording loading and preprocessing."""

from __future__ import annotations

import functools
import inspect
import os
import shutil
from pathlib import Path


def _default_save_n_jobs() -> int:
    # Cap at 4 workers. Each materialize_lfp worker holds the bandpass
    # margin's worth of upstream (20 kHz × n_channels) input data; at
    # margin=10 s + 1024 ch that's ~1.7 GB/worker. 16 workers (cpu//2 on
    # a 32-core box) tipped a 62 GB host into OOM. 4 workers keeps peak
    # transient around 8 GB while still amortizing the filter cost.
    return max(1, min((os.cpu_count() or 2) // 2, 4))


def load_maxwell_recording(data_path, well_id, rec_name=None):
    """Load a Maxwell raw HDF5 recording for one (well, rec_name)."""
    import spikeinterface.extractors as se

    return se.read_maxwell(str(data_path), stream_id=str(well_id), rec_name=rec_name)


@functools.lru_cache(maxsize=1)
def _bandpass_accepts_ignore_low_freq_error() -> bool:
    # Older SI 0.103.x builds don't declare `ignore_low_freq_error`; the kwarg
    # then falls through `**filter_kwargs` and crashes inside FilterRecording.
    # Those builds also lack the highpass_check, so omitting it is a no-op.
    import spikeinterface.preprocessing as spre

    return "ignore_low_freq_error" in inspect.signature(spre.bandpass_filter).parameters


def safe_bandpass_filter(recording, **kwargs):
    """`spre.bandpass_filter` that drops `ignore_low_freq_error` on older SI."""
    import spikeinterface.preprocessing as spre

    if "ignore_low_freq_error" in kwargs and not _bandpass_accepts_ignore_low_freq_error():
        kwargs.pop("ignore_low_freq_error")
    return spre.bandpass_filter(recording, **kwargs)


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
    lfp_target_fs_hz: float | None = 1000,
    lfp_notch_freqs: list[float] | None = None,
    lfp_notch_q: float = 30.0,
    lfp_reference_operator: str = "median",
    spike_reference_operator: str = "median",
):
    """Prepare signed raw, LFP-filtered, and spike-filtered recording views.

    The LFP path is resampled to ``lfp_target_fs_hz`` before bandpass/CMR so
    that every downstream operation runs on the decimated signal. Spike path
    stays at the raw rate (needs >6 kHz Nyquist for 300–3000 Hz content).

    Notch filters (when given) are applied to the LFP path immediately after
    the bandpass, so power-line frequencies (50/60 Hz) and their harmonics
    can be suppressed before any band-power analysis. ``lfp_reference_operator``
    selects ``"mean"`` (true CAR) or ``"median"`` (CMR) for the common
    reference; the median is more robust to single-channel outliers.
    """
    import spikeinterface.preprocessing as spre

    signed = spre.unsigned_to_signed(recording_raw)

    lfp_source = signed
    raw_fs = float(signed.get_sampling_frequency())
    if lfp_target_fs_hz is not None and float(lfp_target_fs_hz) < raw_fs:
        lfp_source = spre.resample(
            signed,
            resample_rate=int(lfp_target_fs_hz),
            dtype="float32",
        )

    lfp = safe_bandpass_filter(
        lfp_source,
        freq_min=lfp_low_hz,
        freq_max=lfp_high_hz,
        margin_ms=lfp_filter_margin_ms,
        ignore_low_freq_error=lfp_ignore_low_freq_error,
    )
    # Notch each requested line frequency. We chain narrowband notches rather
    # than computing a multi-frequency filter so the user can pick arbitrary
    # frequencies (50, 60, 100, 120 …) and the filter stays well-conditioned.
    for f in (lfp_notch_freqs or []):
        if f is None:
            continue
        lfp = spre.notch_filter(lfp, freq=float(f), q=float(lfp_notch_q))
    spike = spre.bandpass_filter(signed, freq_min=spike_low_hz, freq_max=spike_high_hz)
    if apply_lfp_common_reference:
        lfp = spre.common_reference(lfp, reference="global", operator=lfp_reference_operator)
    if apply_spike_common_reference:
        spike = spre.common_reference(spike, reference="global", operator=spike_reference_operator)
    return {"raw": signed, "lfp": lfp, "spike": spike}


def materialize_lfp(
    lfp_recording,
    cache_dir,
    *,
    save_n_jobs: int | None = None,
    save_chunk_duration: str = "10s",
    progress: bool = True,
):
    """Stream a preprocessed LFP recording to a flat binary on disk.

    The returned recording reads directly from the binary file with no
    preprocessing chain, so get_traces() calls avoid re-running the
    bandpass filter and common-reference median on every window.

    ``save_n_jobs`` defaults to ``min(cpu_count()//2, 4)`` — parallel
    enough to amortize the bandpass+CMR cost, conservative enough to avoid
    the multiplicative memory blowup that comes from each worker holding
    the filter-margin's worth of upstream 20 kHz data. ``save_chunk_duration``
    defaults to ``"10s"`` so the 10 s filter margin only doubles the
    per-chunk input read (instead of the 21× amplification a 1 s chunk
    would impose on top of the 20× resample upstream).
    """
    if save_n_jobs is None:
        save_n_jobs = _default_save_n_jobs()
    cache_dir = Path(cache_dir)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    return lfp_recording.save(
        folder=cache_dir,
        format="binary",
        n_jobs=save_n_jobs,
        chunk_duration=save_chunk_duration,
        progress_bar=progress,
        overwrite=True,
    )
