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
