# Acute Slice MEA

Analysis platform for multi-electrode array (MEA) recordings from acute brain slices (Maxwell MaxOne/MaxTwo). Provides batch signal processing, spectral analysis, burst detection, and an interactive web-based trace viewer.

## Overview

Two main workflows:

1. **CLI Pipeline** (`acute-slice-mea compute`) — one-time heavy computation on raw Maxwell HDF5 recordings. Produces cached outputs (JSON, CSV, Parquet, NPZ).
2. **Interactive Viewer** (`acute-slice-viewer`) — Dash web app for browsing LFP traces with dynamic resampling. Reads from cache; no reprocessing on each view.

```
Maxwell HDF5 (.h5)
    │
    ▼
┌─────────────────────────────────────┐
│  CLI Pipeline (one-time compute)    │
│  recording.py → spectral.py        │
│  → bursts.py → cache.py            │
└─────────────┬───────────────────────┘
              │ writes
              ▼
┌─────────────────────────────────────┐
│  Cache Bundle (per recording+well)  │
│  JSON / CSV / Parquet / NPZ         │
└─────────────┬───────────────────────┘
              │ reads
              ▼
┌─────────────────────────────────────┐
│  Dash Viewer (many-time query)      │
│  library.py → data_loader.py        │
│  → callbacks.py → browser           │
└─────────────────────────────────────┘
```

---

## Architecture

### Package Layout

| Module | Role |
|--------|------|
| `src/acute_slice_mea/recording.py` | Load Maxwell .h5 via SpikeInterface; apply bandpass + common reference |
| `src/acute_slice_mea/spectral.py` | Welch PSD, band power over time, RMS, trace preview |
| `src/acute_slice_mea/bursts.py` | Network burst detection (envelope threshold) |
| `src/acute_slice_mea/electrodes.py` | Build electrode table (position-to-channel mapping) |
| `src/acute_slice_mea/probe_geometry.py` | MaxWell chip constants: 220x120 grid, 17.5 um pitch |
| `src/acute_slice_mea/cache.py` | Serialize results to JSON/CSV/Parquet/NPZ |
| `src/acute_slice_mea/pipeline.py` | Orchestrator: `run_analysis(config)` end-to-end |
| `src/acute_slice_mea/library.py` | Discover recordings (from cache or raw .h5 tree) |
| `src/acute_slice_mea/viewer/` | Dash app: layout, callbacks, data_loader |
| `hdmea_lfp_viz/` | Separate publication-figure pipeline (Zarr, PNG/SVG/MP4) |

### Entry Points

| Command | Purpose |
|---------|---------|
| `acute-slice-mea compute --data-path X --well-id Y` | Run full pipeline, write cache |
| `acute-slice-viewer --cache-root DIR` | Launch Dash viewer on port 8050 |
| `python -m hdmea_lfp_viz.main` | Generate publication figures (PNG/SVG/MP4) |

---

## Signal Processing Pipeline

### 1. Data Loading

- **Format:** Maxwell HDF5 (`.h5`), read via `spikeinterface.extractors.read_maxwell()`
- **Organization:** `/wells/{well_id}/{rec_name}/` within the HDF5 file
- **Conversion:** Unsigned-to-signed (2's complement via `spre.unsigned_to_signed`)

### 2. Filtering

Three signal paths are produced from the signed recording:

| Signal | Band (Hz) | Method | Common Reference |
|--------|-----------|--------|------------------|
| LFP | 0.5–300 | `spre.bandpass_filter` (SpikeInterface default: scipy Butterworth), margin=10 s | Global median (all channels, per timepoint) |
| Spike | 300–3000 | `spre.bandpass_filter` (SpikeInterface default) | Global median |
| Raw | — | None (signed only) | None |

The common reference subtracts the per-timepoint median across all channels, reducing correlated noise.

Parameters are configurable in `prepare_recordings()`:
- `lfp_low_hz`, `lfp_high_hz`, `spike_low_hz`, `spike_high_hz`
- `lfp_filter_margin_ms` (default 10000 ms — prevents edge artifacts)
- `apply_lfp_common_reference`, `apply_spike_common_reference`

### 3. Spectral Analysis

**Band Power Over Time** (`compute_lfp_band_power_over_time`):

1. Sliding window: 10 s window, 5 s step (configurable via `window_sec`, `step_sec`)
2. Per window: `scipy.signal.welch(traces, fs=fs, nperseg=welch_segment_sec*fs)` — default `welch_segment_sec=2`
3. Band integration: `numpy.trapezoid(psd[band_mask], freqs[band_mask])`
4. Convert to dB: `10 * log10(power + machine_epsilon)`

**Default frequency bands:**

| Band | Range (Hz) |
|------|-----------|
| delta | 0.5–4 |
| theta | 4–8 |
| alpha | 8–12 |
| beta | 12–30 |
| low_gamma | 30–80 |
| high_gamma | 80–150 |

**Welch Spectrum Summary** (`compute_welch_spectrum_summary`):
- Computes mean, median, 5th/95th percentile PSD across channels
- Reports dominant frequency per channel (`freqs[argmax(psd)]`)
- Default: first 10 seconds of recording, `welch_segment_sec=2`

**RMS Per Electrode** (`compute_lfp_rms_per_electrode`):
- `sqrt(mean(traces^2, axis=time))` over first 10 s of LFP
- Used for electrode quality coloring in the probe map

**Trace Preview** (`compute_trace_preview`):
- Decimates raw/LFP/spike to ~20,000 points via uniform step
- `step = ceil(n_frames / max_points)`

### 4. Burst Detection

Algorithm in `bursts.py` (pure NumPy, no external spike sorter dependency):

1. **Population envelope:** `mean(|LFP_traces|)` across all channels
2. **Downsample** to ~1 kHz if `fs > 2000`: mean-pooling (anti-alias)
3. **Smooth:** uniform (boxcar) convolution, window = `envelope_smooth_ms` (default 50 ms)
4. **Threshold:** `median + threshold_sd * robust_SD`
   - `robust_SD = 1.4826 * MAD` (MAD = median absolute deviation)
   - Fallback: `np.std(smoothed)` if MAD = 0
   - Default `threshold_sd = 3.0`
5. **Edge detection:** find contiguous above-threshold runs
6. **Merge:** combine runs separated by < `min_gap_ms` (default 30 ms)
7. **Duration filter:** keep bursts with duration in [`min_duration_ms`, `max_duration_ms`] (default 30–500 ms)
8. **Long-event splitting:** events exceeding `max_duration_ms` are re-thresholded internally at a higher level

Output per burst: `{t_start_s, t_end_s, center_s, peak_amp_uv, duration_s}`

All parameters bundled in `BurstDetectionParams` dataclass.

### 5. Electrode Geometry

- Maxwell MaxOne/MaxTwo: 220 columns x 120 rows at 17.5 um pitch
- Total array: 3850 x 2100 um
- ~1000 routed electrodes per well (out of 26,400 possible positions)
- Electrode ID = `row * 220 + col` (row-major)
- Position matching: SpikeInterface `get_channel_locations()` mapped to electrode table

---

## Cache Structure

Each `(recording, well)` pair produces one cache bundle:

```
cache_dir/
├── manifest.json          # Index + config snapshot
├── summary.json           # Metadata (sample_rate, duration, well_id, rec_name)
├── electrodes.csv         # electrode_id, channel_id, x_um, y_um, rms_uv
├── probe.json             # Geometry + routed electrode list
├── lfp_band_power.parquet # Band power time series (6 bands x N windows x M electrodes)
├── spectrum_summary.npz   # Welch PSD statistics
├── trace_preview.npz      # Decimated raw/lfp/spike (~20k points)
├── bursts.json            # Detected burst events
└── dashboard/data/        # (optional) Pre-computed JSON traces per electrode
    ├── manifest.json
    └── traces/lfp/{electrode_id}.json
```

---

## Viewer Architecture

### Layout (3-pane CSS Grid)

- **Left (280px):** Recording library — searchable, grouped by date
- **Center:** LFP trace viewer (Scattergl + plotly-resampler)
- **Right (360px):** 24-well plate map + electrode picker (SVG probe map with lasso/box select)

### Data Flow

```
User selects recording → selects well → auto-seeds top-RMS electrodes
    → loads WellData from cache bundle (LRU cached, maxsize=16)
    → FigureResampler wraps full-resolution LFP traces
    → Browser shows ~1000 points per trace (MinMaxLTTB aggregated)

User zooms/pans → relayoutData fires
    → FigureResampler re-aggregates visible window via MinMaxLTTB
    → Patch update sent to client (no full figure rebuild)
```

### Key Design Patterns

- **plotly-resampler:** Full-resolution data stays server-side. MinMaxLTTB algorithm downsamples to ~1000 points per visible trace. Zoom triggers re-aggregation.
- **Patch updates:** Probe map selection uses `Patch()` (updates marker line widths only). Preserves lasso UI state without rebuilding figure.
- **LRU caching:** `@lru_cache(maxsize=16)` on WellData; `@lru_cache(maxsize=4)` on prepared SpikeInterface recordings.
- **Fallback path:** If pre-computed JSON traces don't exist, viewer loads directly from .h5 via SpikeInterface (lazy, windowed reads).
- **Module-level FigureResampler:** Stored in `_current_resampler` global — zoom callback needs the object; Dash Stores can't hold Python objects.

### Electrode Selection

- Lasso/box select on SVG probe map
- Quick-select chips: "All routed", "None", "Top RMS", "Every 8th"
- Text entry: parse ranges like "500-520, 550, 600-610"
- Single-click toggle

---

## For Continued Development

### Adding a new analysis step:
1. Write computation in a new module under `src/acute_slice_mea/`
2. Add output serialization in `cache.py`
3. Wire into `pipeline.py:run_analysis()`
4. Add viewer support: load in `data_loader.py`, display via new callback in `callbacks.py`

### Adding a new viewer feature:
1. Add UI component in `viewer/layout.py`
2. Add callback in `viewer/callbacks.py` (follow Store-driven, Patch-where-possible pattern)
3. If new data needed: extend `WellData` in `viewer/data_loader.py`

### The hdmea_lfp_viz/ pipeline:
Separate from the main package. Generates static publication figures (PNG/SVG) and band-envelope movies (MP4). Shares no code with the viewer — same .h5 format, similar filtering concept. Could be unified in future.

---

## Verification Checklist

To verify computations produce correct results:

| Computation | Verification approach |
|-------------|----------------------|
| Bandpass filter | Confirm passband/stopband behavior with known sinusoid; check SpikeInterface `bandpass_filter` default order for your SI version |
| Common reference | Verify noise floor reduction; check median is per-timepoint across all channels |
| Welch PSD | Validate against known sinusoid (peak at correct frequency, correct amplitude in dB) |
| Band integration | Compare trapezoidal integral vs Parseval's theorem on synthetic signal |
| Burst detection | Run on synthetic signal with known bursts; confirm MAD threshold logic (`1.4826 * MAD`) |
| RMS | Check units (uV); verify `sqrt(mean(x^2))` on known amplitude signal |
| Electrode mapping | Confirm `electrode_id = row * 220 + col` matches Maxwell documentation |

---

## Setup

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate acute-slice-mea
```

Install the package in editable mode:

```bash
python -m pip install -e .
```

## RTX 5090 PyTorch

Keep PyTorch out of the Conda solve so the target machine can install the correct CUDA wheel. For an NVIDIA RTX 5090 workstation, install PyTorch with CUDA 12.8 after activating the environment:

```bash
uv pip install torch torchvision torchaudio --torch-backend=cu128
```

If uv is unavailable, use pip with the PyTorch CUDA 12.8 wheel index:

```bash
python -m pip install -r requirements-pytorch-cu128.txt
```

Verify GPU access on the RTX 5090 machine:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The `uv pip --torch-backend` option is specific to uv's pip-compatible interface. Use `cu128` for CUDA 12.8.

## Spike Sorting Tools

`environment.yml` installs SpikeInterface and the Conda-available GUI/runtime dependencies used by phy. If the `phy` command is unavailable or broken after environment creation, follow phy's current upstream install path inside the active Conda environment:

```bash
python -m pip install git+https://github.com/cortex-lab/phy.git
```

## Project Layout

```text
src/acute_slice_mea/       Main analysis package
src/acute_slice_mea/viewer/ Dash web viewer
hdmea_lfp_viz/             Publication figure pipeline
tests/                     Unit tests (57 tests)
notebooks/                 Exploratory and report notebooks
scripts/                   build_viewer_cache.py, strip_notebook_outputs.py
data/raw/                  Original local data files
data/processed/            Derived local data / cache bundles
data/external/             External reference data
temp/                      Scratch files
reports/figures/           Generated images
reports/pdfs/              Generated PDF exports
```

Data, image, and PDF files are ignored by git. Keep raw recordings and generated outputs local.

## Development

Run tests:

```bash
python -m pytest
```

Launch Jupyter:

```bash
jupyter lab
```

Start new analyses by copying or editing `notebooks/00_mea_analysis_template.ipynb`.

## Interactive Trace Viewer

```bash
# 1. Build per-(recording, well) caches once. Resumable; skips cached bundles.
python scripts/build_viewer_cache.py \
    --data-root /Volumes/SadeghYR/Yuxin/MEA/raw/MeaSlices_CarenPaula_04082026 \
    --cache-root data/processed

# 2. Launch the viewer pointing at the same cache root.
python -m acute_slice_mea.viewer --cache-root data/processed
# Open http://localhost:8050
```

The viewer is a thin client over the cache; all heavy preprocessing (bandpass, burst detection, RMS, probe geometry) happens once in `scripts/build_viewer_cache.py`.

## HD-MEA LFP Exploratory Figures

Reproducible, non-interactive LFP visualization pipeline in `hdmea_lfp_viz/`. Reads a Maxwell MaxTwo `.h5` file, saves a 1 kHz LFP Zarr cache, computes summary statistics, and writes publication-style PNG/SVG figures.

Install the plotting pipeline dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the full pipeline on a new Maxwell recording:

```bash
python -m hdmea_lfp_viz.main /path/to/recording.h5 --stream-id well000 --skip-movie
```

Outputs:

```text
cache/lfp_1khz.zarr
cache/summaries.npz
figures/01_overview_heatmap.{png,svg}
...
figures/10_summary_panel.{png,svg}
figures/band_envelope_{delta,theta,alpha,beta,low_gamma,high_gamma}.mp4
```

After the cache exists, iterate on figures without recomputing:

```bash
python -m hdmea_lfp_viz.main --figures-only --skip-movie
```

To generate MP4 files, install a system `ffmpeg` binary and omit `--skip-movie`. Movies play at 10x real time by default. Use `--overwrite-cache` when intentionally rebuilding the Zarr and summary caches.

## Key Dependencies

| Library | Purpose |
|---------|---------|
| SpikeInterface | Maxwell decoder, filtering, common reference |
| scipy.signal | Welch PSD computation |
| plotly + dash | Web-based interactive visualization |
| plotly-resampler | Dynamic MinMaxLTTB trace aggregation on zoom |
| h5py | Raw HDF5 access (through SpikeInterface) |
| numpy / pandas | Numerical computation + tabular data |
| joblib | Parallel channel-chunk processing |
| zarr | HD-MEA LFP cache (hdmea_lfp_viz only) |
