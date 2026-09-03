from datetime import datetime, timedelta, timezone

import pytest

from truss.templates.shared import serialization


@pytest.mark.parametrize(
    "dt",
    [
        datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone(timedelta(hours=2))),
        datetime(2024, 1, 2, 3, 4, 5, 123456),
    ],
)
def test_msgpack_datetime_roundtrip(dt):
    payload = serialization.truss_msgpack_serialize({"ts": dt})
    assert serialization.truss_msgpack_deserialize(payload) == {"ts": dt}


def test_msgpack_decodes_legacy_utc_z_suffix():
    # Older truss servers spell a UTC offset as "Z", which
    # `datetime.fromisoformat` only accepts from Python 3.11 on.
    decoded = serialization._truss_msgpack_decoder(
        {b"__dt_datetime_iso__": True, b"data": "2024-01-02T03:04:05Z"}
    )
    assert decoded == datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
