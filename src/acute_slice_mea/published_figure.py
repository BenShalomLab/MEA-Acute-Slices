"""Publication figure: chip activity-scan map with RMS-colored electrode overlay.

Panel 1 places a brain slice in spatial context:

* background  -- an activity-scan *amplitude map* (max amplitude per electrode),
  which exposes brain regions by their differing activity, covering the full
  MaxOne chip (3850 um wide x 2100 um high);
* overlay     -- the network-scan electrodes drawn as points coloured by their
  per-electrode LFP RMS (Viridis);
* ROI         -- a user-supplied set of electrodes highlighted with a bold edge.

The amplitude map is consumed as a rendered matplotlib SVG (the heatmap is an
embedded base64 PNG); we lift that bitmap out and place it in micrometre
coordinates so the electrode positions overlay directly.

A second panel is reserved (created blank) for content to be defined later.

Panel B (a separate figure, :func:`make_panel_b_figure`) zooms into the ROI
electrodes over a 1 s window: a stacked LFP **waveform snapshot** on the left and
a per-electrode **3-robust-SD raster** on the right, reusing the same LFP
processing chain (0.5-300 Hz + global-median common reference) and the burst
detector's robust threshold (median + 3 x 1.4826 x MAD, i.e. 3 robust SD).
"""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from acute_slice_mea.bursts import BurstDetectionParams
from acute_slice_mea.electrodes import build_electrode_table, recorded_electrode_channels
from acute_slice_mea.probe_geometry import HEIGHT_UM, MAXWELL_PITCH_UM, WIDTH_UM
from acute_slice_mea.recording import load_maxwell_recording, prepare_recordings
from acute_slice_mea.spectral import compute_lfp_rms_per_electrode, get_traces_safe

# Full-chip micrometre extent, matplotlib imshow convention [left, right, bottom, top].
FULL_CHIP_EXTENT = (0.0, WIDTH_UM, 0.0, HEIGHT_UM)

_BASE64_IMG_RE = re.compile(r"data:image/png;base64,\s*([A-Za-z0-9+/=\s]+?)[\"\)]")


# --------------------------------------------------------------------------- #
# Activity-scan background
# --------------------------------------------------------------------------- #
def extract_activity_map(svg_path) -> np.ndarray:
    """Return the activity-scan heatmap from a matplotlib amplitude-map SVG.

    The SVG embeds two base64 PNGs: the heatmap and a thin colorbar. We decode
    every embedded image and return the one with the largest pixel area (the
    heatmap) as an RGBA ``uint8`` array of shape ``(rows, cols, 4)``.
    """
    from PIL import Image

    text = Path(svg_path).read_text()
    matches = _BASE64_IMG_RE.findall(text)
    if not matches:
        raise ValueError(f"No embedded base64 PNG found in {svg_path}")

    best: np.ndarray | None = None
    best_area = -1
    for payload in matches:
        raw = base64.b64decode(re.sub(r"\s", "", payload))
        image = Image.open(io.BytesIO(raw)).convert("RGBA")
        arr = np.asarray(image)
        area = arr.shape[0] * arr.shape[1]
        if area > best_area:
            best, best_area = arr, area
    assert best is not None
    return best


# --------------------------------------------------------------------------- #
# Per-electrode LFP RMS
# --------------------------------------------------------------------------- #
def _discover_well_id(h5_path) -> str:
    """Return the single MaxWell stream/well id for a (MaxOne) recording."""
    import spikeinterface.extractors as se

    _, stream_ids = se.MaxwellRecordingExtractor.get_streams(str(h5_path))
    if not stream_ids:
        raise ValueError(f"No MaxWell streams found in {h5_path}")
    if len(stream_ids) > 1:
        raise ValueError(
            f"Expected a single well, found {stream_ids!r}; pass well_id explicitly."
        )
    return str(stream_ids[0])


def compute_roi_rms(
    h5_path,
    *,
    well_id: str | None = None,
    start_sec: float = 66.0,
    window_sec: float = 10.0,
    cache_path=None,
    force: bool = False,
) -> pd.DataFrame:
    """Electrode table for ``h5_path`` with a per-electrode LFP ``rms_uv`` column.

    RMS is the root-mean-square of the LFP-filtered (0.5-300 Hz, global-median
    common reference) signal over ``[start_sec, start_sec + window_sec]``. The
    underlying :func:`compute_lfp_rms_per_electrode` only reads from frame 0, so
    we frame-slice the LFP view to start at ``start_sec``.

    Result is cached to ``cache_path`` (CSV). Only the LFP-derived table is
    persisted -- no raw/spike traces (LFP-only project scope).
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force:
            return pd.read_csv(cache_path)

    if well_id is None:
        well_id = _discover_well_id(h5_path)

    raw = load_maxwell_recording(h5_path, well_id)
    recordings = prepare_recordings(raw)
    probe = raw.get_probe()
    electrodes = build_electrode_table(probe, raw)

    lfp = recordings["lfp"]
    fs = float(lfp.get_sampling_frequency())
    num_samples = int(lfp.get_num_samples())
    start_frame = max(0, min(num_samples, int(round(start_sec * fs))))
    end_frame = min(num_samples, start_frame + int(round(window_sec * fs)))
    if end_frame <= start_frame:
        raise ValueError(
            f"Empty RMS window: start={start_sec}s window={window_sec}s "
            f"but recording is only {num_samples / fs:.1f}s long."
        )
    lfp_window = lfp.frame_slice(start_frame=start_frame, end_frame=end_frame)

    rms_by_eid = compute_lfp_rms_per_electrode(
        lfp_window, electrodes, duration_sec=window_sec
    )
    electrodes = electrodes.assign(
        rms_uv=electrodes["electrode_id"].astype(int).map(rms_by_eid)
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        electrodes.to_csv(cache_path, index=False)
    return electrodes


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _muted_grayscale(image: np.ndarray, floor: float = 0.55) -> np.ndarray:
    """Mute an RGBA bitmap to a faint, lightened grayscale underlay.

    Converts to luminance (``0.299R + 0.587G + 0.114B``) normalised to ``[0, 1]``
    then remaps into a light band ``[floor, 1.0]`` so the background keeps its
    *regional* light/dark structure (e.g. the high-activity band) but loses the
    full-saturation colour that competes with the Viridis electrode overlay. The
    lighten step is what lets dark-end Viridis markers stay legible even over
    originally-dark regions. The original alpha is preserved so off-chip
    transparent areas stay transparent over the (white) axes facecolor.

    Returns an RGBA float array in ``[0, 1]`` of shape ``(rows, cols, 4)``.
    """
    arr = np.asarray(image)
    rgb = arr[..., :3].astype(np.float64) / 255.0
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    gray = floor + (1.0 - floor) * lum
    out = np.empty(arr.shape[:2] + (4,), dtype=np.float64)
    out[..., 0] = out[..., 1] = out[..., 2] = gray
    if arr.shape[-1] == 4:
        out[..., 3] = arr[..., 3].astype(np.float64) / 255.0
    else:
        out[..., 3] = 1.0
    return out


def _remap_red(
    image: np.ndarray,
    r_min: int = 250,
    g_max: int = 100,
    b_max: int = 100,
    target: tuple[int, int, int] = (255, 100, 100),
) -> np.ndarray:
    """Recolor near-pure-red background pixels to a muted ``target`` colour.

    Any pixel with ``R > r_min`` and ``G < g_max`` and ``B < b_max`` is set to
    ``target``. This tones down the saturated-red speckle in the activity-map
    background so it stops camouflaging the red ROI electrode borders, while a
    light-red ``target`` (rather than black) keeps it distinct from the black
    electrode marker edges. The alpha channel is left untouched.
    """
    arr = np.array(image)  # copy of the RGBA uint8 bitmap
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mask = (r > r_min) & (g < g_max) & (b < b_max)
    arr[mask, 0:3] = target
    return arr


def _rms_clip_range(values: np.ndarray, clip_pct=(1.0, 99.0)) -> tuple[float, float]:
    """Robust colour limits: percentile clip over positive RMS values."""
    finite = values[np.isfinite(values) & (values > 0)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, clip_pct)
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def plot_chip_panel(
    ax,
    image: np.ndarray,
    electrodes: pd.DataFrame,
    *,
    roi_ids=(),
    extent=FULL_CHIP_EXTENT,
    image_origin: str = "lower",
    invert_y: bool = True,
    cmap: str = "viridis",
    clip_pct=(1.0, 99.0),
    crop=None,
    marker_size: float = 26.0,
    roi_edgewidth: float = 5.0,
    bg_grayscale: bool = False,
    bg_floor: float = 0.55,
    suppress_bg_red: bool = True,
    bg_red_thresh: tuple[int, int, int] = (250, 100, 100),
    bg_red_target: tuple[int, int, int] = (255, 100, 100),
    marker_edgecolor: str = "black",
    marker_edgewidth: float = 0.4,
):
    """Draw the activity-map background and the RMS-coloured electrode overlay.

    ``crop`` is ``(x0, x1, y0, y1)`` in microns; default shows the full chip.
    Returns the scatter (for colorbar wiring).

    Readability: every electrode square gets a black edge
    (``marker_edgecolor`` / ``marker_edgewidth``) so it is delineated from the
    colour activity-map background and its neighbours, while ROI electrodes carry
    a thicker red edge (``roi_edgewidth``) on top to stand out. To keep red
    reserved for those ROI borders, ``suppress_bg_red`` (default true) recolors
    near-pure-red background pixels (``R > bg_red_thresh[0]`` and G/B below the
    remaining thresholds) to the muted ``bg_red_target`` colour. Optionally set
    ``bg_grayscale=True`` to instead mute the background to a faint, lightened
    grayscale underlay (see :func:`_muted_grayscale`, ``bg_floor`` tunes how
    faint) so the Viridis electrodes become the only colour in the panel.

    Orientation: the activity-map bitmap is flipped upside down (``np.flipud``)
    before it is drawn, then placed over the full-chip ``extent`` in micrometre
    coordinates. ``image_origin`` and ``invert_y`` only affect axis *display*
    direction (``invert_y=True`` puts y=0 at the top, the MaxWell chip
    convention matching the interactive viewer); the electrode overlay is drawn
    at ``(x_um, y_um)`` regardless.
    """
    x0, x1, y0, y1 = extent
    img_extent = (x0, x1, y0, y1)
    image = np.flipud(image)
    if suppress_bg_red:
        r_min, g_max, b_max = bg_red_thresh
        image = _remap_red(
            image, r_min=r_min, g_max=g_max, b_max=b_max, target=bg_red_target
        )
    if bg_grayscale:
        image = _muted_grayscale(image, floor=bg_floor)
    ax.imshow(
        image,
        extent=img_extent,
        origin=image_origin,
        aspect="equal",
        interpolation="nearest",
        zorder=0,
    )

    recorded = electrodes[electrodes["recorded"].astype(bool)].copy()
    rms = recorded["rms_uv"].to_numpy(dtype=float)
    vmin, vmax = _rms_clip_range(rms, clip_pct)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Draw each electrode as a true-size square (17.5 um pitch) in data
    # coordinates, so apparent size is physically accurate regardless of figure
    # size / crop / DPI. ``s=marker_size`` (points^2) is no longer used.
    side = float(MAXWELL_PITCH_UM)

    def _squares(rows: pd.DataFrame) -> list[Rectangle]:
        return [
            Rectangle((x - side / 2.0, y - side / 2.0), side, side)
            for x, y in zip(rows["x_um"], rows["y_um"])
        ]

    scatter = PatchCollection(
        _squares(recorded), cmap=cmap, norm=norm,
        edgecolors=marker_edgecolor, linewidths=marker_edgewidth, zorder=2,
    )
    scatter.set_array(rms)
    ax.add_collection(scatter)

    roi_set = {int(e) for e in roi_ids}
    if roi_set:
        roi = recorded[recorded["electrode_id"].astype(int).isin(roi_set)]
        roi_pc = PatchCollection(
            _squares(roi), cmap=cmap, norm=norm,
            edgecolors="red", linewidths=roi_edgewidth, zorder=3,
        )
        roi_pc.set_array(roi["rms_uv"].to_numpy(dtype=float))
        ax.add_collection(roi_pc)

    if crop is not None:
        x0, x1, y0, y1 = crop
    else:
        x0, x1, y0, y1 = extent
    ax.set_xlim(x0, x1)
    ax.set_ylim((y1, y0) if invert_y else (y0, y1))
    ax.set_aspect("equal")
    ax.set_xlabel("X Coordinate (um)")
    ax.set_ylabel("Y Coordinate (um)")
    return scatter


def make_panel_a_figure(
    svg_path,
    h5_path,
    roi_ids,
    out_stem,
    *,
    crop=None,
    start_sec: float = 66.0,
    window_sec: float = 10.0,
    well_id: str | None = None,
    cache_path=None,
    rms_table: pd.DataFrame | None = None,
    invert_y: bool = True,
    image_origin: str = "lower",
    dpi: int = 300,
) -> Figure:
    """Build and save the two-panel publication figure (panel 2 reserved blank).

    Saves ``<out_stem>.svg`` and ``<out_stem>.png``. ``rms_table`` lets callers
    inject a precomputed electrode/RMS table (skipping the heavy h5 read).
    """
    image = extract_activity_map(svg_path)
    if rms_table is None:
        rms_table = compute_roi_rms(
            h5_path, well_id=well_id, start_sec=start_sec,
            window_sec=window_sec, cache_path=cache_path,
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6), width_ratios=[1.0, 1.0])
    scatter = plot_chip_panel(
        axes[0], image, rms_table, roi_ids=roi_ids, crop=crop,
        invert_y=invert_y, image_origin=image_origin,
    )
    cbar = fig.colorbar(
        scatter, ax=axes[0], orientation="horizontal",
        fraction=0.05, pad=0.18, aspect=40,
    )
    cbar.set_label("RMS (µV)")
    axes[0].set_title("Activity map + network electrodes (RMS)")

    # Panel 2 reserved -- content to be defined later.
    axes[1].axis("off")
    axes[1].text(0.5, 0.5, "panel 2 (TBD)", ha="center", va="center",
                 transform=axes[1].transAxes, color="#999")

    fig.tight_layout()
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".svg"))
    fig.savefig(out_stem.with_suffix(".png"), dpi=dpi)
    return fig


# --------------------------------------------------------------------------- #
# Panel 2: ROI waveform snapshot + 3-MAD raster
# --------------------------------------------------------------------------- #
def _load_roi_lfp_window(
    h5_path,
    roi_ids,
    *,
    well_id: str | None = None,
    start_sec: float = 66.0,
    window_sec: float = 1.0,
    return_scaled: bool = True,
):
    """Load LFP traces for the ROI electrodes over ``[start, start+window]`` s.

    Uses the *same* processing chain as the viewer / build_viewer_cache: the
    default :func:`prepare_recordings` LFP view (0.5-300 Hz bandpass +
    global-median common reference). Returns ``(eids, channel_ids, traces, fs,
    t)`` where ``traces`` is ``(n_samples, n_roi)`` in micro-volts, ``fs`` the
    LFP sampling rate, and ``t`` the wall-clock time vector in seconds. ``eids``
    preserve the requested ROI order. Traces are returned in memory only -- not
    persisted (LFP-only project scope).
    """
    if well_id is None:
        well_id = _discover_well_id(h5_path)

    raw = load_maxwell_recording(h5_path, well_id)
    recordings = prepare_recordings(raw)
    probe = raw.get_probe()
    electrodes = build_electrode_table(probe, raw)

    lfp = recordings["lfp"]
    fs = float(lfp.get_sampling_frequency())
    num_samples = int(lfp.get_num_samples())
    start_frame = max(0, min(num_samples, int(round(start_sec * fs))))
    end_frame = min(num_samples, start_frame + int(round(window_sec * fs)))
    if end_frame <= start_frame:
        raise ValueError(
            f"Empty waveform window: start={start_sec}s window={window_sec}s "
            f"but recording is only {num_samples / fs:.1f}s long."
        )

    eids, channel_ids = recorded_electrode_channels(electrodes, roi_ids)
    traces = get_traces_safe(
        lfp, start_frame, end_frame, channel_ids, return_scaled=return_scaled
    )
    traces = np.asarray(traces, dtype=float)
    if traces.ndim == 1:
        traces = traces[:, np.newaxis]
    t = start_frame / fs + np.arange(traces.shape[0], dtype=np.float64) / fs
    return eids, channel_ids, traces, fs, t


def detect_mad_crossings(
    traces: np.ndarray,
    t: np.ndarray,
    *,
    threshold_sd: float = 3.0,
    min_gap_ms: float = 30.0,
) -> list[np.ndarray]:
    """Per-electrode robust threshold-crossing onset times.

    For each column (electrode) of ``traces`` we compute a robust baseline --
    ``median`` and ``mad = median(|x - median|)`` -- and convert MAD to a
    Gaussian-equivalent SD with the standard ``1.4826`` factor (matching
    :mod:`acute_slice_mea.bursts`). A sample is "deviant" when
    ``|x - median| > threshold_sd * 1.4826 * mad``; we return the **onset time**
    (in seconds, via ``t``) of each contiguous deviant run, merging runs closer
    than ``min_gap_ms``. Returns a list aligned to the trace columns.
    """
    arr = np.asarray(traces, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    t = np.asarray(t, dtype=float)
    fs = 1.0 / float(np.median(np.diff(t))) if t.size > 1 else 1.0
    gap_samples = max(1, int(round(min_gap_ms * 1e-3 * fs)))

    out: list[np.ndarray] = []
    for col in range(arr.shape[1]):
        x = arr[:, col]
        median = float(np.median(x))
        mad = float(np.median(np.abs(x - median)))
        sd = 1.4826 * mad if mad > 0 else float(np.std(x))
        if sd <= 0:
            out.append(np.empty(0, dtype=float))
            continue
        above = np.abs(x - median) > threshold_sd * sd
        if not above.any():
            out.append(np.empty(0, dtype=float))
            continue
        edges = np.diff(above.astype(np.int8))
        starts = np.where(edges == 1)[0] + 1
        ends = np.where(edges == -1)[0] + 1
        if above[0]:
            starts = np.concatenate([[0], starts])
        if above[-1]:
            ends = np.concatenate([ends, [above.size]])
        # Merge runs separated by < min_gap so a single excursion yields one tick.
        merged: list[int] = []
        for s, e in zip(starts, ends):
            if merged and s - merged[-1] < gap_samples:
                continue
            merged.append(int(s))
        out.append(t[np.asarray(merged, dtype=int)])
    return out


def _minmax_decimate(t: np.ndarray, y: np.ndarray, target_points: int = 2000):
    """Stride-decimate a single trace for display when it has many samples."""
    n = y.shape[0]
    if n <= target_points:
        return t, y
    step = max(1, n // target_points)
    return t[::step], y[::step]


def plot_waveform_panel(
    ax,
    t: np.ndarray,
    traces: np.ndarray,
    eids,
    *,
    gap: float | None = None,
    target_points: int = 2000,
    scale_uv: float = 100.0,
):
    """Stacked LFP waveform snapshot, one row per ROI electrode.

    Rows are offset by ``gap`` micro-volts (default: 6x the median robust SD
    across electrodes), top electrode first. A fixed ``scale_uv``-µV vertical
    scale bar is drawn in the left margin, *outside* the plotting area. Returns
    the computed ``gap``.
    """
    arr = np.asarray(traces, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    n_roi = arr.shape[1]

    sds = [1.4826 * np.median(np.abs(arr[:, i] - np.median(arr[:, i]))) for i in range(n_roi)]
    robust_sd = float(np.median([s for s in sds if s > 0]) or np.std(arr))
    if gap is None:
        gap = 6.0 * robust_sd if robust_sd > 0 else float(np.ptp(arr) or 1.0)

    offsets = []
    for i in range(n_roi):
        offset = (n_roi - 1 - i) * gap  # row 0 (first ROI) on top
        offsets.append(offset)
        ts, ys = _minmax_decimate(t, arr[:, i], target_points)
        ax.plot(ts, ys + offset, lw=0.6, color="#1f2933")

    ax.set_yticks(offsets)
    ax.set_yticklabels([f"E{int(e)}" for e in eids])
    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("")  # E## tick labels already identify rows; free the left margin

    # Fixed-length vertical scale bar (``scale_uv`` µV) drawn in the left margin,
    # *outside* the plotting area. ``get_yaxis_transform`` maps x to axes
    # fractions (negative = left of the spine) and y to data units (µV);
    # ``clip_on=False`` lets it render outside the axes box.
    trans = ax.get_yaxis_transform()
    ymin, ymax = ax.get_ylim()
    yc = 0.5 * (ymin + ymax)
    y0, y1 = yc - scale_uv / 2.0, yc + scale_uv / 2.0
    x_bar = -0.09
    ax.plot(
        [x_bar, x_bar], [y0, y1], transform=trans,
        color="black", lw=1.5, clip_on=False,
    )
    ax.text(
        x_bar - 0.015, yc, f"{scale_uv:.0f} µV", transform=trans,
        rotation=90, va="center", ha="right", fontsize=8, clip_on=False,
    )
    return gap


def plot_raster_panel(ax, crossings: list[np.ndarray], eids, *, window):
    """Raster of per-electrode robust-SD crossing onsets, one row per ROI electrode."""
    n_roi = len(crossings)
    # Row 0 (first ROI) on top to match the waveform panel's stacking order.
    positions = [crossings[i] for i in range(n_roi)]
    lineoffsets = list(range(n_roi - 1, -1, -1))
    ax.eventplot(
        positions,
        lineoffsets=lineoffsets,
        colors="#b91c1c",
        linelengths=0.7,
        linewidths=1.2,
    )
    ax.set_yticks(lineoffsets)
    ax.set_yticklabels([f"E{int(e)}" for e in eids])
    ax.set_ylim(-0.6, n_roi - 0.4)
    ax.set_xlim(window[0], window[1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Electrode")


def make_panel_b_figure(
    h5_path,
    roi_ids,
    out_stem,
    *,
    start_sec: float = 66.0,
    window_sec: float = 1.0,
    well_id: str | None = None,
    threshold_sd: float = 3.0,
    dpi: int = 300,
) -> Figure:
    """Build and save the Panel B figure: ROI waveform snapshot + 3-MAD raster.

    Left  -- stacked LFP traces (same 0.5-300 Hz + global-median CMR chain as the
    viewer / build_viewer_cache) for the ROI electrodes over
    ``[start_sec, start_sec + window_sec]``.
    Right -- a raster marking each electrode's |LFP - median| > ``threshold_sd``
    robust-SD crossing onsets over the same window.

    Saves ``<out_stem>.svg`` and ``<out_stem>.png``. Traces are held in memory
    only (LFP-only project scope); nothing trace-derived is persisted.
    """
    eids, _channel_ids, traces, fs, t = _load_roi_lfp_window(
        h5_path, roi_ids, well_id=well_id, start_sec=start_sec, window_sec=window_sec
    )
    crossings = detect_mad_crossings(
        traces, t, threshold_sd=threshold_sd,
        min_gap_ms=BurstDetectionParams().min_gap_ms,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6), width_ratios=[1.0, 1.0])
    plot_waveform_panel(axes[0], t, traces, eids)
    axes[0].set_title(
        f"ROI LFP snapshot ({start_sec:.0f}–{start_sec + window_sec:.0f} s)"
    )

    window = (float(t[0]), float(t[-1]))
    plot_raster_panel(axes[1], crossings, eids, window=window)
    axes[1].set_title(f"Threshold crossings (≥{threshold_sd:g} robust SD)")

    fig.tight_layout()
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".svg"))
    fig.savefig(out_stem.with_suffix(".png"), dpi=dpi)
    return fig


# --------------------------------------------------------------------------- #
# Combined figure: Panel A (chip map) stacked over Panel B (snapshot + raster)
# --------------------------------------------------------------------------- #
def make_combined_figure(
    svg_path,
    h5_path,
    roi_ids,
    out_stem,
    *,
    crop=None,
    rms_start_sec: float = 66.0,
    rms_window_sec: float = 10.0,
    snap_start_sec: float = 66.0,
    snap_window_sec: float = 1.0,
    threshold_sd: float = 3.0,
    well_id: str | None = None,
    cache_path=None,
    rms_table: pd.DataFrame | None = None,
    invert_y: bool = True,
    image_origin: str = "lower",
    dpi: int = 300,
) -> Figure:
    """Build and save the combined figure: Panel A stacked over Panel B.

    Top row (full width) -- the activity-scan map + RMS-coloured electrode
    overlay (Panel A). Bottom row -- the ROI LFP waveform snapshot (left) and the
    per-electrode robust-SD raster (right) (Panel B). Pure composition of the
    existing Panel A/B helpers; no new plotting logic.

    Saves ``<out_stem>.svg`` and ``<out_stem>.png``. ``rms_table`` lets callers
    inject a precomputed electrode/RMS table (skipping that h5 read). Traces stay
    in memory only (LFP-only project scope).
    """
    image = extract_activity_map(svg_path)
    if rms_table is None:
        rms_table = compute_roi_rms(
            h5_path, well_id=well_id, start_sec=rms_start_sec,
            window_sec=rms_window_sec, cache_path=cache_path,
        )

    eids, _channel_ids, traces, fs, t = _load_roi_lfp_window(
        h5_path, roi_ids, well_id=well_id,
        start_sec=snap_start_sec, window_sec=snap_window_sec,
    )
    crossings = detect_mad_crossings(
        traces, t, threshold_sd=threshold_sd,
        min_gap_ms=BurstDetectionParams().min_gap_ms,
    )

    # Size the top row from the chip's display aspect so the equal-aspect map
    # (set by plot_chip_panel) becomes width-limited and spans the full figure
    # width -- matching Panel B's waveform+raster row. Keeping the chip aspect
    # fixed means a wider Panel A is also taller, so the figure grows downward.
    fig_w = 12.0
    cx0, cx1, cy0, cy1 = crop if crop is not None else FULL_CHIP_EXTENT
    chip_aspect = (cx1 - cx0) / (cy1 - cy0)
    panelA_box_h = fig_w / chip_aspect          # chip box height at full width
    top_h = panelA_box_h + 1.2                  # + title + h-colorbar + xlabel
    bottom_h = 3.2                              # Panel B row height
    fig, axd = plt.subplot_mosaic(
        [["A", "A"], ["wave", "raster"]],
        figsize=(fig_w, top_h + bottom_h),
        height_ratios=[top_h, bottom_h], layout="constrained",
    )

    scatter = plot_chip_panel(
        axd["A"], image, rms_table, roi_ids=roi_ids, crop=crop,
        invert_y=invert_y, image_origin=image_origin,
    )
    cbar = fig.colorbar(
        scatter, ax=axd["A"], orientation="horizontal",
        fraction=0.046, pad=0.08, aspect=40, shrink=0.5,
    )
    cbar.set_label("RMS (µV)")
    axd["A"].set_title("Activity map + network electrodes (RMS)")

    plot_waveform_panel(axd["wave"], t, traces, eids)
    axd["wave"].set_title(
        f"ROI LFP snapshot ({snap_start_sec:.0f}–{snap_start_sec + snap_window_sec:.0f} s)"
    )

    window = (float(t[0]), float(t[-1]))
    plot_raster_panel(axd["raster"], crossings, eids, window=window)
    axd["raster"].set_title(f"Threshold crossings (≥{threshold_sd:g} robust SD)")

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".svg"))
    fig.savefig(out_stem.with_suffix(".png"), dpi=dpi)
    return fig
