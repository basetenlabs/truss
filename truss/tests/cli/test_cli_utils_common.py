from truss.cli.utils import common


def test_normalize_iso_timestamp_handles_nanoseconds():
    normalized = common._normalize_iso_timestamp("2025-11-17 05:05:06.000000000 +0000")
    assert normalized == "2025-11-17 05:05:06.000000+00:00"


def test_normalize_iso_timestamp_handles_z_suffix_and_short_fraction():
    normalized = common._normalize_iso_timestamp("2025-11-17T05:05:06.123456Z")
    assert normalized == "2025-11-17T05:05:06.123456+00:00"
