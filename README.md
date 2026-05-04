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
figures/run_report.json
```

After the cache exists, iterate on figures without recomputing preprocessing or summaries:

```bash
python -m hdmea_lfp_viz.main --figures-only --skip-movie
```

To generate `figures/band_envelope.mp4`, install a system `ffmpeg` binary and omit `--skip-movie`.
Use `--overwrite-cache` when intentionally rebuilding the Zarr and summary caches from the raw file.
