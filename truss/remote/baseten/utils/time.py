from dateutil import parser


def iso_to_millis(ts: str) -> int:
    """
    Convert ISO 8601 timestamp string to milliseconds since epoch.

    Args:
        ts: ISO 8601 timestamp string (handles Zulu/UTC (Z) automatically)

    Returns:
        Milliseconds since epoch as integer
    """
    dt = parser.isoparse(ts)  # handles Zulu/UTC (Z) automatically
    return int(dt.timestamp() * 1000)
