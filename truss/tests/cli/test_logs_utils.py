from datetime import datetime, timedelta, timezone

import pytest
import rich_click as click

from truss.cli.logs.utils import (
    LOG_RESUME_FORMAT,
    MAX_LOG_RANGE,
    format_flag_datetime,
    parse_since,
    resolve_log_time_range,
)


def test_parse_since_minutes():
    assert parse_since("30m") == timedelta(minutes=30)


def test_parse_since_hours():
    assert parse_since("2h") == timedelta(hours=2)


def test_parse_since_days():
    assert parse_since("3d") == timedelta(days=3)


def test_parse_since_seconds():
    assert parse_since("90s") == timedelta(seconds=90)


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


def test_resolve_only_start_leaves_end_none():
    # Omitted --end is deferred to the server (sent as None), not backfilled.
    # Paginated fetches pin their own end internally (see
    # get_loops_deployment_logs), so the resolver stays a pure translation.
    start = datetime(2026, 5, 14, 0, 0, 0, tzinfo=timezone.utc)
    start_ms, end_ms = resolve_log_time_range(start, None, None)
    assert start_ms == int(start.timestamp() * 1000)
    assert end_ms is None


def test_resolve_only_start_skips_window_check():
    # With end deferred, a far-past start is not rejected client-side; the
    # server enforces the 7-day window.
    start = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    start_ms, end_ms = resolve_log_time_range(start, None, None)
    assert start_ms == int(start.timestamp() * 1000)
    assert end_ms is None


def test_format_flag_datetime_round_trip_error_is_bounded_downward():
    # Hint callers bias by +/-1ms in their safe direction; that only works
    # if the render->parse round trip loses at most 1ms and never gains.
    # Samples include a known float-truncation case (2150399568319) and a
    # DST fall-back-hour epoch (1762065000000), which the %z-aware format
    # must round-trip without the 1-hour ambiguity.
    for epoch_ms in (1782968142946, 2150399568319, 1762065000000):
        rendered = format_flag_datetime(epoch_ms)
        parsed = datetime.strptime(rendered, LOG_RESUME_FORMAT)
        delta = int(parsed.timestamp() * 1000) - epoch_ms
        assert -1 <= delta <= 0


def test_resolve_only_end_leaves_start_none():
    # Omitted --start is deferred to the server (sent as None), not backfilled.
    end = datetime(2026, 5, 14, 0, 0, 0, tzinfo=timezone.utc)
    start_ms, end_ms = resolve_log_time_range(None, end, None)
    assert start_ms is None
    assert end_ms == int(end.timestamp() * 1000)


def test_resolve_both_honored():
    start = datetime(2026, 5, 13, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 14, 0, 0, 0, tzinfo=timezone.utc)
    start_ms, end_ms = resolve_log_time_range(start, end, None)
    assert start_ms == int(start.timestamp() * 1000)
    assert end_ms == int(end.timestamp() * 1000)


def test_resolve_naive_datetime_treated_as_local():
    # Naive input is interpreted in the local timezone (astimezone() on a naive
    # value assumes local), matching how logs are displayed.
    naive = datetime(2026, 5, 14, 12, 0, 0)
    start_ms, _ = resolve_log_time_range(naive, None, None)
    assert start_ms == int(naive.astimezone().timestamp() * 1000)


def test_resolve_aware_datetime_respects_offset():
    aware = datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone(timedelta(hours=5)))
    start_ms, _ = resolve_log_time_range(aware, None, None)
    assert start_ms == int(aware.timestamp() * 1000)


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
