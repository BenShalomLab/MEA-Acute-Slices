# Acute Slice MEA

Template repository for Python and Jupyter Notebook analysis of acute-slice multi-electrode array data.

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
src/                   Reusable analysis code you add later
tests/                 Unit tests you add later
notebooks/             Exploratory and report notebooks
data/raw/              Original local data files
data/processed/        Derived local data files
data/external/         External reference data
temp/                  Scratch files
reports/figures/       Generated images
reports/pdfs/          Generated PDF exports
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

The dashboard is the operations center for the project: scan a data root, filter
recordings, tune analysis parameters, run jobs out-of-process with live progress,
and review LFP signals at adaptive resolution.

### Launching

```bash
# Run-everything mode: scan raw recordings, run new analyses, review caches.
python -m acute_slice_mea.viewer \
    --data-root  /Volumes/SadeghYR/Yuxin/MEA/raw/MeaSlices_CarenPaula_04082026 \
    --cache-root data/processed
# Open http://localhost:8050

# View-only mode: no spawn UI, just browse existing caches.
python -m acute_slice_mea.viewer --cache-root data/processed
```

For batch headless builds without the dashboard, the CLI script still works:

```bash
python scripts/build_viewer_cache.py \
    --data-root /…/MeaSlices_CarenPaula_04082026 \
    --cache-root data/processed
```

### Five-step user flow

1. **Scan.** Point the viewer at a `--data-root`. The library pane lists every
   recording in `[Sample/]Date/Plate/Scan/Run/data.raw.h5` layout, with a
   per-well cache-state pill (idle / running / cached / failed).
2. **Enrich.** When a sibling `mxassay.metadata` file is present, the library
   surfaces `groupname`, chip id, started timestamp, and per-well annotations
   (plating date, density, media, …). See
   `src/acute_slice_mea/maxwell_metadata.py`.
3. **Filter.** The filter bar above the library narrows by sample, scan type,
   plate, group / condition, and cache state — combined as AND across axes,
   OR within an axis.
4. **Stage + Run.** Click `Run` on a single row, or `Run analysis on filtered`
   to batch-submit the filtered set. The params sheet then exposes:
   - Bandpass low / high (Hz)
   - LFP target fs (default 1000 Hz)
   - Reference: CMR (median) / CAR (mean) / None
   - Notch frequencies (comma-separated, e.g. `50,100`) and Q
   - Band-power window / step (s)
   - Six editable power-band rows (delta…high-gamma)
   - Overwrite checkbox (visible when a cache already exists)

   Jobs spawn as detached subprocesses (`subprocess.Popen(..., start_new_session=True)`)
   and survive dashboard restarts. Progress is reported at six pipeline stages
   (`loading → preparing → band_power → spectrum → trace_preview → bursts → saving`).
5. **Review.** Pick a well → use lasso, box, or click to select electrodes on
   the probe map → the trace pane renders adaptive-resolution LFP. The
   `plotly-resampler` integration re-aggregates from the 1 kHz cache on every
   zoom event, so a 10-min window shows ~2 K representative points but zooming
   to 100 ms shows the full underlying detail.

### Adaptive resolution

LFP traces use `plotly_resampler.FigureResampler` with the `MinMaxLTTB`
aggregator. The full 1 kHz cached trace is held server-side; only ~2000
points are sent to the browser at a time. When the user zooms, the
resampler's `relayoutData` callback fires and re-aggregates within the new
x-range. No client-side downsampling is needed — and the cache stays at full
resolution on disk.

### Params hash and cache identity

Every entry in `acute_slice_mea.jobs.DEFAULT_PARAMS` participates in the
`params_hash` computed at submit time. Changing any of `band`, `reference`,
`lfp_fs_hz`, `notch_freqs`, `notch_q`, `bands`, `window_sec`, `step_sec`, or
`decimation` produces a different hash and therefore a separate cache. If
you add a new tunable to the pipeline, also add it to `DEFAULT_PARAMS` so
old caches aren't silently reused with the new code. Bumping
`PIPELINE_VERSION` (the `pipeline` key) invalidates every cache.

See `src/acute_slice_mea/{viewer,library,jobs,pipeline,recording}.py` for
implementation details.

## HD-MEA LFP Exploratory Figures

This repository also includes a reproducible, non-interactive LFP visualization pipeline in `hdmea_lfp_viz/`.
It reads a Maxwell MaxTwo `.h5` file with SpikeInterface, saves a 1 kHz LFP Zarr cache, computes summary statistics, and writes publication-style PNG/SVG figures.

Install the plotting pipeline dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the full pipeline on a new Maxwell recording:

```bash
python -m hdmea_lfp_viz.main /path/to/recording.h5 --stream-id well000 --skip-movie
```

Outputs are written to:

```text
cache/lfp_1khz.zarr
cache/summaries.npz
figures/01_overview_heatmap.png
figures/01_overview_heatmap.svg
...
figures/10_summary_panel.png
figures/10_summary_panel.svg
figures/band_envelope_delta.mp4
figures/band_envelope_theta.mp4
figures/band_envelope_alpha.mp4
figures/band_envelope_beta.mp4
figures/band_envelope_low_gamma.mp4
figures/band_envelope_high_gamma.mp4
figures/band_envelope.mp4
figures/run_report.json
```

After the cache exists, iterate on figures without recomputing preprocessing or summaries:

```bash
python -m hdmea_lfp_viz.main --figures-only --skip-movie
```

To generate the band-envelope MP4 files, install a system `ffmpeg` binary and omit `--skip-movie`.
Movies play at 10x real time by default.
The legacy `figures/band_envelope.mp4` path is also written as a copy of `figures/band_envelope_low_gamma.mp4`.
Use `--overwrite-cache` when intentionally rebuilding the Zarr and summary caches from the raw file.
