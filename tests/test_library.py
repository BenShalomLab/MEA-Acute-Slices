import json
from pathlib import Path

from acute_slice_mea.library import LibraryIndex


def _write_stub_bundle(root: Path, *, sample, date, plate, scan, run, well_id, with_dashboard=True):
    raw_path = root / sample / date / plate / scan / run / "data.raw.h5"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(b"")
    bundle = root / "cache" / sample / date / plate / scan / run / well_id
    bundle.mkdir(parents=True, exist_ok=True)
    summary = {
        "well_id": well_id,
        "data_path": str(raw_path),
        "sampling_frequency_hz": 20000.0,
        "duration_sec": 600.0,
        "num_recorded_electrodes": 1024,
    }
    (bundle / "manifest.json").write_text(json.dumps({"summary": summary, "files": {}}))
    if with_dashboard:
        dash = bundle / "dashboard" / "data"
        dash.mkdir(parents=True)
        (dash / "manifest.json").write_text("{}")
    return bundle


def test_cache_mode_groups_wells_under_one_recording(tmp_path):
    _write_stub_bundle(tmp_path, sample="CX138", date="260329", plate="T003346", scan="Network", run="000029", well_id="well000")
    _write_stub_bundle(tmp_path, sample="CX138", date="260329", plate="T003346", scan="Network", run="000029", well_id="well001")
    _write_stub_bundle(tmp_path, sample="CX138", date="260415", plate="T003351", scan="Network", run="000017", well_id="well000", with_dashboard=False)

    index = LibraryIndex.from_cache_root(tmp_path / "cache")
    assert index.mode == "cache"
    assert len(index.recordings) == 2

    rec_29 = index.find_recording("CX138/260329/T003346/Network/000029")
    assert rec_29 is not None
    assert [w.well_id for w in rec_29.wells] == ["well000", "well001"]
    assert rec_29.sample == "CX138"
    assert rec_29.iso_date == "2026-03-29"

    rec_15 = index.find_recording("CX138/260415/T003351/Network/000017")
    assert rec_15 is not None
    assert rec_15.wells[0].has_dashboard_data is False


def test_grouped_by_date_sorts_recent_first(tmp_path):
    _write_stub_bundle(tmp_path, sample="CX138", date="260329", plate="P", scan="Network", run="000001", well_id="well000")
    _write_stub_bundle(tmp_path, sample="CX138", date="260502", plate="P", scan="Network", run="000002", well_id="well000")
    index = LibraryIndex.from_cache_root(tmp_path / "cache")
    grouped = index.grouped_by_date()
    assert [date for date, _ in grouped] == ["2026-05-02", "2026-03-29"]


def test_raw_mode_lists_recordings_when_h5_files_present(tmp_path, monkeypatch):
    # Build the directory structure of a sample-level layout.
    sample_root = tmp_path / "MeaSlices_Example"
    raw_path = sample_root / "260408" / "16719" / "ActivityScan" / "000001" / "data.raw.h5"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(b"")

    # Stub the wells listing to avoid touching real HDF5.
    import acute_slice_mea.library as lib

    monkeypatch.setattr(lib, "_list_wells_from_h5", lambda p: ["well000", "well001"])
    index = LibraryIndex.from_data_root(sample_root)
    assert index.mode == "raw"
    assert len(index.recordings) == 1
    rec = index.recordings[0]
    assert rec.sample == "MeaSlices_Example"
    assert rec.date == "260408"
    assert rec.scan == "ActivityScan"
    assert [w.well_id for w in rec.wells] == ["well000", "well001"]


def test_find_returns_none_for_unknown_well(tmp_path):
    index = LibraryIndex([], mode="cache", root=str(tmp_path))
    assert index.find("missing", "well000") is None
    assert index.find_recording("missing") is None
