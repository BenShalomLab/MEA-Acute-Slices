"""Exploratory LFP visualization pipeline for Maxwell HD-MEA recordings."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("cache") / "matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((Path("cache") / "xdg").resolve()))

__all__ = ["__version__"]

__version__ = "0.1.0"
