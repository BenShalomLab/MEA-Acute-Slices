"""Unit tests for pure helpers used by the viewer callbacks.

The Dash callbacks themselves are integration code (they need a running
server to exercise meaningfully). The ID-parsing helper is extracted so it
can be tested directly.
"""

from __future__ import annotations

import pytest

from acute_slice_mea.viewer.callbacks import _parse_electrode_id_input, _rms_clip_range


VALID = set(range(1, 1001))  # routed-electrode set used in tests


class TestParseElectrodeIdInput:
    def test_single_int(self):
        assert _parse_electrode_id_input("500", VALID) == [500]

    def test_comma_separated(self):
        assert _parse_electrode_id_input("1, 2, 3", VALID) == [1, 2, 3]

    def test_range(self):
        assert _parse_electrode_id_input("1-5", VALID) == [1, 2, 3, 4, 5]

    def test_mixed(self):
        assert _parse_electrode_id_input(
            "1-5, 7, 10-12", VALID
        ) == [1, 2, 3, 4, 5, 7, 10, 11, 12]

    def test_whitespace_separators(self):
        # Spaces, tabs, newlines all delimit (commas are normalised first).
        assert _parse_electrode_id_input("1 2\n3\t4   5", VALID) == [1, 2, 3, 4, 5]

    def test_dedup(self):
        assert _parse_electrode_id_input("3, 3, 1-3", VALID) == [1, 2, 3]

    def test_reversed_range(self):
        assert _parse_electrode_id_input("5-1", VALID) == [1, 2, 3, 4, 5]

    def test_junk_dropped(self):
        assert _parse_electrode_id_input("abc, 7, 10-foo, 12", VALID) == [7, 12]

    def test_empty(self):
        assert _parse_electrode_id_input("", VALID) == []
        assert _parse_electrode_id_input("   ", VALID) == []

    def test_filters_against_valid_set(self):
        # Routed set is {100, 200, 300}; over-broad inputs are silently
        # filtered down so users can paste "everything" without errors.
        valid = {100, 200, 300}
        assert _parse_electrode_id_input("100-300, 500, 999", valid) == [100, 200, 300]

    def test_range_with_some_invalid_members(self):
        # Range expands then filters element-by-element against valid.
        valid = {2, 4, 6}
        assert _parse_electrode_id_input("1-6", valid) == [2, 4, 6]

    def test_only_invalid_ids(self):
        # All inputs outside the routed set → empty result (callback uses this
        # to PreventUpdate rather than wiping the current selection).
        assert _parse_electrode_id_input("9999, 8888-8890", VALID) == []


class TestRmsClipRange:
    def test_normal_range(self):
        values = list(range(1, 101))
        lo, hi = _rms_clip_range(values)
        assert lo == pytest.approx(2.98, abs=0.5)
        assert hi == pytest.approx(98.02, abs=0.5)

    def test_zeros_filtered(self):
        values = [0.0] * 50 + list(range(1, 51))
        lo, hi = _rms_clip_range(values)
        assert lo > 0

    def test_few_values_fallback(self):
        values = [1.0, 2.0, 100.0]
        lo, hi = _rms_clip_range(values)
        assert lo == 0.0
        assert hi == 100.0

    def test_all_zeros(self):
        lo, hi = _rms_clip_range([0.0, 0.0, 0.0])
        assert lo == 0.0
        assert hi == 0.0

    def test_uniform_values(self):
        values = [5.0] * 20
        lo, hi = _rms_clip_range(values)
        assert hi > lo
