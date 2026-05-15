from datetime import datetime, timedelta, timezone

import pytest
import rich_click as click

from truss.cli.logs.utils import MAX_LOG_RANGE, parse_since, resolve_log_time_range


def test_parse_since_minutes():
    assert parse_since("30m") == timedelta(minutes=30)


def test_parse_since_hours():
    assert parse_since("2h") == timedelta(hours=2)


def test_parse_since_days():
    assert parse_since("3d") == timedelta(days=3)


def test_parse_since_boundary_7d_ok():
    assert parse_since("7d") == MAX_LOG_RANGE


def test_parse_since_boundary_in_hours_ok():
    assert parse_since("168h") == MAX_LOG_RANGE


@pytest.mark.parametrize(
    "value",
    [
        "0m",
        "0h",
        "0d",
        "8d",
        "169h",
        "5x",
        "abc",
        "",
        "30",
        " 30m ",  # outer whitespace ok via strip, but no decimals/units mismatch
        "-1h",
        "1.5h",
        "1h30m",
    ],
)
def test_parse_since_invalid(value):
    # ' 30m ' is actually valid because we strip
    if value == " 30m ":
        assert parse_since(value) == timedelta(minutes=30)
        return
    with pytest.raises(click.BadParameter):
        parse_since(value)


def test_resolve_returns_none_when_no_inputs():
    assert resolve_log_time_range(None, None, None) == (None, None)


def test_resolve_since_sets_window_ending_now():
    start_ms, end_ms = resolve_log_time_range(None, None, "1h")
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    # End should be ~now (within a few seconds).
    assert abs(end_ms - now_ms) < 5_000
    # Start should be ~1h before end.
    assert abs((end_ms - start_ms) - 3_600_000) < 5_000


def test_resolve_only_start_defaults_end_to_now():
    start = datetime(2026, 5, 14, 0, 0, 0, tzinfo=timezone.utc)
    start_ms, end_ms = resolve_log_time_range(start, None, None)
    assert start_ms == int(start.timestamp() * 1000)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    assert abs(end_ms - now_ms) < 5_000


def test_resolve_only_start_at_exact_7d_boundary_accepted():
    # Regression: --start exactly 7d ago at second precision must not be
    # rejected because datetime.now() has microsecond precision.
    start = (datetime.now(timezone.utc) - MAX_LOG_RANGE).replace(microsecond=0)
    start_ms, end_ms = resolve_log_time_range(start, None, None)
    assert end_ms - start_ms <= int(MAX_LOG_RANGE.total_seconds() * 1000)


def test_resolve_only_end_defaults_start_to_end_minus_7d():
    end = datetime(2026, 5, 14, 0, 0, 0, tzinfo=timezone.utc)
    start_ms, end_ms = resolve_log_time_range(None, end, None)
    assert end_ms == int(end.timestamp() * 1000)
    assert end_ms - start_ms == int(MAX_LOG_RANGE.total_seconds() * 1000)


def test_resolve_both_honored():
    start = datetime(2026, 5, 13, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 14, 0, 0, 0, tzinfo=timezone.utc)
    start_ms, end_ms = resolve_log_time_range(start, end, None)
    assert start_ms == int(start.timestamp() * 1000)
    assert end_ms == int(end.timestamp() * 1000)


def test_resolve_naive_datetime_treated_as_utc():
    naive = datetime(2026, 5, 14, 0, 0, 0)
    aware = naive.replace(tzinfo=timezone.utc)
    start_ms_naive, _ = resolve_log_time_range(naive, None, None)
    start_ms_aware, _ = resolve_log_time_range(aware, None, None)
    assert start_ms_naive == start_ms_aware


def test_resolve_start_after_end_errors():
    start = datetime(2026, 5, 14, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 13, 0, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(click.UsageError):
        resolve_log_time_range(start, end, None)


def test_resolve_start_equals_end_errors():
    dt = datetime(2026, 5, 14, 0, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(click.UsageError):
        resolve_log_time_range(dt, dt, None)


def test_resolve_window_over_7d_errors():
    start = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 14, 0, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(click.UsageError):
        resolve_log_time_range(start, end, None)


def test_resolve_since_with_start_errors():
    start = datetime(2026, 5, 14, 0, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(click.UsageError):
        resolve_log_time_range(start, None, "1h")


def test_resolve_since_with_end_errors():
    end = datetime(2026, 5, 14, 0, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(click.UsageError):
        resolve_log_time_range(None, end, "1h")
