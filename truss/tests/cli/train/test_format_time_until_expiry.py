from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import patch

import pytest

from truss.cli.train_commands import _format_time_until_expiry


def _utc_iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _fixed_now(target: datetime):
    """Return a patcher that freezes datetime.now to *target*."""
    return patch(
        "truss.cli.train_commands.datetime",
        wraps=datetime,
        **{"now.return_value": target},
    )


NOW = datetime(2026, 2, 17, 12, 0, 0, tzinfo=timezone.utc)


@dataclass
class FormatTimeTillExpiryTestCase:
    desc: str
    input: Optional[str]
    expected: str
    freeze_now: bool = True


CASES = [
    FormatTimeTillExpiryTestCase(
        desc="empty string returns empty", input="", expected="", freeze_now=False
    ),
    FormatTimeTillExpiryTestCase(
        desc="None returns empty", input=None, expected="", freeze_now=False
    ),
    FormatTimeTillExpiryTestCase(
        desc="hours and minutes",
        input=_utc_iso(NOW + timedelta(hours=2, minutes=30)),
        expected="2h 30m",
    ),
    FormatTimeTillExpiryTestCase(
        desc="hours only (zero minutes)",
        input=_utc_iso(NOW + timedelta(hours=4)),
        expected="4h 0m",
    ),
    FormatTimeTillExpiryTestCase(
        desc="minutes only", input=_utc_iso(NOW + timedelta(minutes=45)), expected="45m"
    ),
    FormatTimeTillExpiryTestCase(
        desc="less than one minute remaining",
        input=_utc_iso(NOW + timedelta(seconds=30)),
        expected="< 1m",
    ),
    FormatTimeTillExpiryTestCase(
        desc="already expired (past)",
        input=_utc_iso(NOW - timedelta(minutes=5)),
        expected="Expired",
    ),
    FormatTimeTillExpiryTestCase(
        desc="exactly expired (boundary)", input=_utc_iso(NOW), expected="Expired"
    ),
    FormatTimeTillExpiryTestCase(
        desc="invalid timestamp returns original",
        input="not-a-date",
        expected="not-a-date",
        freeze_now=False,
    ),
    FormatTimeTillExpiryTestCase(
        desc="ISO with Z suffix", input="2026-02-17T13:15:00Z", expected="1h 15m"
    ),
    FormatTimeTillExpiryTestCase(
        desc="ISO with +00:00 offset",
        input="2026-02-17T13:00:00+00:00",
        expected="1h 0m",
    ),
    FormatTimeTillExpiryTestCase(
        desc="infinite timeout (10 years out) shows No timeout",
        input=_utc_iso(NOW + timedelta(days=365 * 10)),
        expected="No timeout",
    ),
    FormatTimeTillExpiryTestCase(
        desc="just over one year still shows No timeout",
        input=_utc_iso(NOW + timedelta(days=366)),
        expected="No timeout",
    ),
    FormatTimeTillExpiryTestCase(
        desc="just under one year shows hours/minutes",
        input=_utc_iso(NOW + timedelta(days=364)),
        expected="8736h 0m",
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[c.desc for c in CASES])
def test_format_time_until_expiry(case: FormatTimeTillExpiryTestCase):
    if case.freeze_now:
        with _fixed_now(NOW):
            assert _format_time_until_expiry(case.input) == case.expected
    else:
        assert _format_time_until_expiry(case.input) == case.expected
