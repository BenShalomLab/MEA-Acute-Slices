"""Decode MaxWell/MaxLab mxassay .metadata files into a Python dict.

Vendored from the sibling Yuxin_MEA project
(``yuxin_mea.dataset._mxassay_decoder``). Kept here as a stand-alone module so
the dashboard does not need Yuxin_MEA on its Python path.

Tested against two observed formats:
1. MaxTwo-style 24-well metadata with Qt ``@Variant(...)`` int64 timestamps/ratings.
2. Single-well metadata with plain Unix timestamps and a Qt ``@Variant(...)``
   selected-well list.

Usage::

    from acute_slice_mea.maxwell_metadata import decode_mxassay_metadata
    meta = decode_mxassay_metadata("/path/to/mxassay.metadata")
    # meta["properties"]["chipid"], meta["wells"]["wells"][0]["groupname"], …

A higher-level helper :func:`extract_recording_metadata` returns the small,
flat per-recording / per-well dict that :class:`acute_slice_mea.library.LibraryIndex`
overlays onto its entries.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TIME_KEYS = {"selected_time", "started", "finished"}


def _qt_escaped_to_bytes(s: str) -> bytes:
    """Convert Qt settings escape text into bytes.

    Handles the escape subset used in MaxWell metadata: ``\\0``, ``\\xNN``,
    and ordinary characters.
    """
    out = bytearray()
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s):
            nxt = s[i + 1]
            if nxt == "0":
                out.append(0)
                i += 2
                continue
            if nxt == "t":
                out.append(9)
                i += 2
                continue
            if nxt == "n":
                out.append(10)
                i += 2
                continue
            if nxt == "r":
                out.append(13)
                i += 2
                continue
            if nxt == "x":
                j = i + 2
                hx_chars = []
                while j < len(s) and len(hx_chars) < 2 and s[j] in "0123456789abcdefABCDEF":
                    hx_chars.append(s[j])
                    j += 1
                if hx_chars:
                    out.append(int("".join(hx_chars), 16))
                    i = j
                    continue
            out.append(ord(nxt))
            i += 2
            continue
        out.append(ord(ch))
        i += 1
    return bytes(out)


def _decode_qt_variant(value: str) -> Any:
    """Decode the small subset of Qt @Variant(...) values seen in MaxWell metadata.

    Supported types: 0x81 (int64 — timestamps, rating, progress), 0x02 (int32),
    0x09 (list — selected wells), 0x43 (QColor — returned as hex). Unknown
    variants are returned as ``{"_qt_variant_type": ..., "raw_hex": ...}``.
    """
    if not (value.startswith("@Variant(") and value.endswith(")")):
        return value

    inner = value[len("@Variant(") : -1]
    b = _qt_escaped_to_bytes(inner)

    if len(b) < 4:
        return {"_qt_variant_type": None, "raw_hex": b.hex()}

    type_code = int.from_bytes(b[0:4], "big", signed=False)

    if type_code == 0x81 and len(b) >= 12:
        return int.from_bytes(b[4:12], "big", signed=True)

    if type_code == 0x02 and len(b) >= 8:
        return int.from_bytes(b[4:8], "big", signed=True)

    if type_code == 0x09 and len(b) >= 8:
        count = int.from_bytes(b[4:8], "big", signed=False)
        pos = 8
        items: list[Any] = []
        for _ in range(count):
            if pos + 4 > len(b):
                break
            item_type = int.from_bytes(b[pos : pos + 4], "big", signed=False)
            pos += 4

            if item_type == 0x02 and pos + 4 <= len(b):
                items.append(int.from_bytes(b[pos : pos + 4], "big", signed=True))
                pos += 4
            elif item_type == 0x81 and pos + 8 <= len(b):
                items.append(int.from_bytes(b[pos : pos + 8], "big", signed=True))
                pos += 8
            else:
                items.append(
                    {
                        "_qt_variant_type": item_type,
                        "remaining_raw_hex": b[pos:].hex(),
                    }
                )
                break
        return items

    if type_code == 0x43:
        return {"_qt_variant_type": "QColor", "raw_hex": b.hex()}

    return {"_qt_variant_type": type_code, "raw_hex": b.hex()}


def _coerce_scalar(value: str) -> Any:
    """Convert a raw string value to bool/int/float/decoded Qt variant when possible."""
    value = value.strip()

    if value.startswith("@Variant("):
        return _decode_qt_variant(value)

    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False

    if re.fullmatch(r"[+-]?\d+", value):
        # Preserve leading-zero run IDs as strings (e.g. "000079").
        if len(value) > 1 and value[0] == "0":
            return value
        return int(value)

    if re.fullmatch(r"[+-]?\d+\.\d+", value):
        return float(value)

    return value


def _unix_to_iso(value: Any) -> str | None:
    """Return UTC ISO string for plausible Unix timestamps."""
    if isinstance(value, int) and 946684800 <= value <= 4102444800:  # 2000–2100
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    return None


def _read_ini_like(path: str | Path) -> dict[str, dict[str, Any]]:
    """Minimal INI parser that preserves case and backslash-containing keys."""
    sections: dict[str, dict[str, Any]] = {}
    current: str | None = None

    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue

        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections.setdefault(current, {})
            continue

        if current is None or "=" not in line:
            continue

        key, raw_value = line.split("=", 1)
        sections[current][key.strip()] = _coerce_scalar(raw_value)

    return sections


def _well_name_from_id(well_id: int, columns: int) -> str:
    """Convert row-major well ID to plate-style name (0 -> A1, 1 -> A2, …)."""
    row = well_id // columns
    col = well_id % columns + 1
    name = ""
    n = row
    while True:
        name = chr(ord("A") + (n % 26)) + name
        n = n // 26 - 1
        if n < 0:
            break
    return f"{name}{col}"


def _parse_wells(wells_section: dict[str, Any]) -> dict[str, Any]:
    rows = wells_section.get("rows")
    columns = wells_section.get("columns")

    selected_raw = wells_section.get("selected")
    if isinstance(selected_raw, str) and re.fullmatch(r"\s*\d+(\s*,\s*\d+)*\s*", selected_raw):
        selected_raw = [int(x.strip()) for x in selected_raw.split(",")]

    result: dict[str, Any] = {
        "rows": rows,
        "columns": columns,
        "selected": selected_raw,
        "info_size": wells_section.get("info\\size"),
        "wells": {},
    }

    info: dict[int, dict[str, Any]] = {}

    for key, value in wells_section.items():
        parts = key.split("\\")
        if len(parts) < 3 or parts[0] != "info" or not parts[1].isdigit():
            continue

        info_index = int(parts[1])
        entry = info.setdefault(info_index, {"annotations": {}})

        if len(parts) >= 5 and parts[2] == "annotations" and parts[3] == "property":
            if parts[4] == "size":
                entry["annotation_size"] = value
                continue
            if parts[4].isdigit() and len(parts) >= 6:
                ann_index = int(parts[4])
                ann_key = parts[5]
                ann = entry["annotations"].setdefault(ann_index, {})
                ann[ann_key] = value
                continue

        entry["\\".join(parts[2:])] = value

    parsed_wells: dict[int, dict[str, Any]] = {}
    for _, entry in sorted(info.items()):
        well_id = entry.get("id")
        if not isinstance(well_id, int):
            continue

        clean_annotations: dict[str, Any] = {}
        raw_annotations: list[dict[str, Any]] = []
        for ann_index, ann in sorted(entry.get("annotations", {}).items()):
            raw_annotations.append({"index": ann_index, **ann})
            name = ann.get("propertyName")
            if name:
                clean_annotations[str(name)] = ann.get("propertyValue")

        entry["annotations_raw"] = raw_annotations
        entry["annotations"] = clean_annotations
        if isinstance(columns, int):
            entry["well_name"] = _well_name_from_id(well_id, columns)

        parsed_wells[well_id] = entry

    result["wells"] = parsed_wells

    selected = result.get("selected")
    if isinstance(selected, list):
        result["selected_wells"] = selected
        if isinstance(columns, int):
            result["selected_well_names"] = [
                _well_name_from_id(x, columns) for x in selected if isinstance(x, int)
            ]

    return result


def decode_mxassay_metadata(path: str | Path, add_iso_times: bool = True) -> dict[str, Any]:
    """Decode a MaxWell/MaxLab mxassay .metadata file into a nested dictionary.

    Returns a dict with top-level sections: General, properties, recordings,
    runtime, wells. See module docstring for examples.
    """
    sections = _read_ini_like(path)

    metadata: dict[str, Any] = {
        "source_file": str(Path(path)),
        "General": sections.get("General", {}),
        "properties": sections.get("properties", {}),
        "recordings": sections.get("recordings", {}),
        "runtime": sections.get("runtime", {}),
    }

    if "wells" in sections:
        metadata["wells"] = _parse_wells(sections["wells"])

    if add_iso_times:
        for section_name in ("properties", "runtime"):
            section = metadata.get(section_name, {})
            if not isinstance(section, dict):
                continue
            for key in list(section.keys()):
                if key in _TIME_KEYS:
                    iso = _unix_to_iso(section[key])
                    if iso:
                        section[f"{key}_iso_utc"] = iso

    runtime = metadata.get("runtime", {})
    if isinstance(runtime, dict):
        started = runtime.get("started")
        finished = runtime.get("finished")
        if isinstance(started, int) and isinstance(finished, int):
            runtime["actual_runtime_seconds"] = finished - started

    return metadata


# =============================================================================
# Library-facing helper
# =============================================================================


def extract_recording_metadata(run_dir: str | Path) -> dict[str, Any] | None:
    """Return a flat metadata dict for one recording run directory.

    Looks for ``mxassay.metadata`` adjacent to ``data.raw.h5`` and pulls out
    the fields the dashboard library + filter UI uses. Returns ``None`` if no
    metadata file is present so the caller can fall back to bare HDF5 attrs.

    Output shape (all optional, may be ``None``):

    ::

        {
            "chipid":              <str>,
            "project_title":       <str>,
            "tag":                 <str>,
            "operator":            <str>,
            "started_iso_utc":     <iso str>,
            "finished_iso_utc":    <iso str>,
            "rating":              <int>,
            "runtime_sec":         <int>,
            "wells": {
                "well000": {"groupname": ..., "control": ..., "annotations": {...}},
                "well001": {...},
                ...
            },
        }
    """
    run_path = Path(run_dir)
    candidates = [run_path / "mxassay.metadata", run_path / "metadata"]
    metadata_path = next((p for p in candidates if p.exists()), None)
    if metadata_path is None:
        return None

    try:
        meta = decode_mxassay_metadata(metadata_path)
    except Exception as exc:  # malformed metadata shouldn't break library scan
        logger.warning("Failed to decode %s: %s", metadata_path, exc)
        return None

    properties = meta.get("properties", {}) or {}
    runtime = meta.get("runtime", {}) or {}
    wells_section = meta.get("wells", {}) or {}

    wells_out: dict[str, dict[str, Any]] = {}
    for well_id_int, well in (wells_section.get("wells") or {}).items():
        well_id = f"well{int(well_id_int):03d}"
        wells_out[well_id] = {
            "groupname": well.get("groupname"),
            "control": bool(well.get("control")) if "control" in well else None,
            "well_name": well.get("well_name"),
            "annotations": dict(well.get("annotations") or {}),
        }

    return {
        "chipid": properties.get("chipid") or properties.get("chip_id"),
        "project_title": properties.get("project_title"),
        "tag": properties.get("tag"),
        "operator": properties.get("operator"),
        "started_iso_utc": runtime.get("started_iso_utc"),
        "finished_iso_utc": runtime.get("finished_iso_utc"),
        "rating": properties.get("rating") if isinstance(properties.get("rating"), int) else runtime.get("rating"),
        "runtime_sec": runtime.get("actual_runtime_seconds"),
        "wells": wells_out,
    }


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode MaxWell mxassay .metadata files.")
    parser.add_argument("metadata_file", help="Path to mxassay .metadata file")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args()

    meta = decode_mxassay_metadata(args.metadata_file)
    print(json.dumps(meta, indent=2 if args.pretty else None, default=_json_default))


if __name__ == "__main__":
    main()
