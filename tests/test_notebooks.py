import json
import sys
import types
from pathlib import Path


def test_compute_cache_notebook_bootstraps_local_src_import(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "01_compute_brain_slice_cache.ipynb"
    notebook = json.loads(notebook_path.read_text())
    first_code_cell = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    source = "".join(first_code_cell["source"])
    src_path = str(repo_root / "src")
    original_sys_path = list(sys.path)

    monkeypatch.chdir(repo_root / "notebooks")
    monkeypatch.setattr(sys, "path", [path for path in sys.path if path != src_path])
    stale_module = types.ModuleType("acute_slice_mea")
    stale_module.__file__ = "/tmp/site-packages/acute_slice_mea/__init__.py"
    stale_module.AnalysisConfig = object
    stale_module.run_analysis = object()
    monkeypatch.setitem(sys.modules, "acute_slice_mea", stale_module)

    namespace = {}
    exec(compile(source, str(notebook_path), "exec"), namespace)

    assert namespace["AnalysisConfig"].__module__ == "acute_slice_mea.pipeline"
    assert namespace["config"].n_jobs == 1
    assert namespace["config"].channel_chunk_size is None
    sys.path[:] = original_sys_path
