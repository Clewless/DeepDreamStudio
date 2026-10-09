"""Property-based / fuzzing tests using Hypothesis to verify stability against junk/random data."""

from __future__ import annotations

import math
from typing import Any

import hypothesis.strategies as st
from hypothesis import given, settings

from app.gui_helpers import (
    clamp01,
    clamp_zoom,
    extract_output_path,
    parse_dropped_paths,
    resolve_out_dir,
    slider_t_to_val,
    slider_val_to_t,
    timeline_index_to_t,
    timeline_t_to_index,
)
from app.protocol import EVENT_PREFIX, format_event, parse_event


def _run_cli(args: list[str]) -> int:
    import contextlib
    import io

    from app import deepdream

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            deepdream.main(args)
            return 0
        except SystemExit as exc:
            return exc.code if isinstance(exc.code, int) else (1 if exc.code else 0)


class TestProtocolFuzzing:
    """Property tests for IPC event protocol serialization and parsing."""

    @given(st.text())
    def test_parse_arbitrary_junk_string_never_crashes(self, junk: str):
        result = parse_event(junk)
        if result is not None:
            assert isinstance(result, dict)
            assert "event" in result

    @given(st.binary())
    def test_parse_binary_decoded_junk_never_crashes(self, raw_bytes: bytes):
        text = raw_bytes.decode("latin-1")
        result = parse_event(text)
        if result is not None:
            assert isinstance(result, dict)

    @given(
        st.text(min_size=1).filter(lambda s: "\n" not in s and "\r" not in s),
        st.dictionaries(
            keys=st.text(min_size=1, max_size=20).filter(lambda s: s != "event"),
            values=st.one_of(
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.text(max_size=50),
                st.booleans(),
            ),
            max_size=5,
        ),
    )
    def test_roundtrip_valid_event_payloads(self, event_name: str, fields: dict[str, Any]):
        formatted = format_event(event_name, **fields)
        parsed = parse_event(formatted)
        assert parsed is not None
        assert parsed["event"] == event_name
        for k, v in fields.items():
            assert parsed.get(k) == v

    @given(st.text())
    def test_prefixed_corrupted_json_returns_none(self, junk_payload: str):
        corrupted = EVENT_PREFIX + junk_payload
        result = parse_event(corrupted)
        if result is not None:
            assert isinstance(result, dict)
            assert "event" in result


class TestSliderMathFuzzing:
    """Property tests for slider value <-> normalized position t conversions."""

    @given(
        t=st.floats(min_value=-1e6, max_value=1e6),
        from_=st.floats(min_value=-1000.0, max_value=1000.0),
        to=st.floats(min_value=-1000.0, max_value=1000.0),
        curve=st.floats(min_value=0.01, max_value=10.0),
    )
    def test_slider_t_to_val_always_bounded_and_finite(
        self, t: float, from_: float, to: float, curve: float
    ):
        val = slider_t_to_val(t, from_, to, curve=curve)
        assert math.isfinite(val)
        min_expected = min(from_, to)
        max_expected = max(from_, to)
        assert min_expected - 1e-4 <= val <= max_expected + 1e-4

    @given(
        t=st.floats(min_value=-1e6, max_value=1e6),
        from_=st.integers(min_value=-1000, max_value=1000),
        to=st.integers(min_value=-1000, max_value=1000),
        curve=st.floats(min_value=0.01, max_value=10.0),
    )
    def test_slider_t_to_val_is_int_produces_int_bounded(
        self, t: float, from_: int, to: int, curve: float
    ):
        val = slider_t_to_val(t, float(from_), float(to), curve=curve, is_int=True)
        assert isinstance(val, int)
        min_expected = min(from_, to)
        max_expected = max(from_, to)
        assert min_expected <= val <= max_expected

    @given(
        val=st.floats(min_value=-1e6, max_value=1e6),
        from_=st.floats(min_value=-1000.0, max_value=1000.0),
        to=st.floats(min_value=-1000.0, max_value=1000.0),
        curve=st.floats(min_value=0.01, max_value=10.0),
    )
    def test_slider_val_to_t_always_in_zero_one(
        self, val: float, from_: float, to: float, curve: float
    ):
        t = slider_val_to_t(val, from_, to, curve=curve)
        assert 0.0 <= t <= 1.0

    @given(
        t=st.floats(min_value=0.0, max_value=1.0),
        from_=st.floats(min_value=1.0, max_value=100.0),
        to=st.floats(min_value=101.0, max_value=500.0),
        curve=st.floats(min_value=0.5, max_value=3.0),
    )
    def test_slider_roundtrip_invertibility(self, t: float, from_: float, to: float, curve: float):
        val = float(slider_t_to_val(t, from_, to, curve=curve))
        t_recovered = slider_val_to_t(val, from_, to, curve=curve)
        val_recovered = float(slider_t_to_val(t_recovered, from_, to, curve=curve))
        assert abs(val_recovered - val) <= 1e-4
        half_quantum = 5e-5
        quantization_error = (half_quantum / abs(to - from_)) ** (1.0 / curve)
        assert abs(t - t_recovered) < 0.02 + quantization_error


class TestGuiHelpersFuzzing:
    """Property tests for path resolution and zoom clamping."""

    @given(st.text())
    def test_resolve_out_dir_junk_never_crashes(self, junk: str):
        resolved = resolve_out_dir(junk)
        assert isinstance(resolved, str)
        assert len(resolved) > 0

    @given(st.floats(allow_nan=True, allow_infinity=True))
    def test_clamp_zoom_safety(self, z: float):
        result = clamp_zoom(z)
        assert 0.05 <= result <= 32.0

    @given(st.floats(allow_nan=True, allow_infinity=True))
    def test_clamp01_safety(self, v: float):
        result = clamp01(v)
        assert 0.0 <= result <= 1.0

    @given(st.text())
    def test_parse_dropped_paths_junk_never_crashes(self, junk: str):
        result = parse_dropped_paths(junk)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, str)

    @given(st.text())
    def test_extract_output_path_junk_never_crashes(self, junk: str):
        result = extract_output_path(junk)
        assert result is None or isinstance(result, str)

    @given(
        index=st.integers(min_value=-1000, max_value=10000),
        total=st.integers(min_value=-1000, max_value=10000),
    )
    def test_timeline_mapping_safety(self, index: int, total: int):
        t = timeline_index_to_t(index, total)
        assert 0.0 <= t <= 1.0
        recovered_idx = timeline_t_to_index(t, total)
        if total <= 0:
            assert recovered_idx == 0
        else:
            assert 0 <= recovered_idx < total


class TestCliArgumentFuzzing:
    """Property tests passing random argument permutations to CLI main()."""

    @settings(max_examples=25, deadline=None)
    @given(st.lists(st.text(max_size=30), max_size=6))
    def test_cli_junk_arguments_safely_exit_without_uncaught_exceptions(self, args: list[str]):
        returncode = _run_cli(args)
        assert returncode in (0, 1, 2)
