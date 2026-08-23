import click

from truss.cli.cannery.errors import (
    ErrorCategory,
    command_failure,
    error_category,
    retry_info,
)


def test_throttled_error_preserves_retry_info(tmp_path):
    event = {
        "category": "throttled",
        "reason": "RATE_LIMITED",
        "message": "Too many operations",
        "retryInfo": {"retryDelay": {"seconds": "3", "nanos": 500_000_000}},
    }

    error = command_failure(
        event,
        return_code=1,
        correlation_id="corr-123",
        diagnostic_path=tmp_path / "diagnostic.jsonl",
    )

    assert isinstance(error, click.ClickException)
    assert not isinstance(error, click.UsageError)
    assert error_category(event, 1) == ErrorCategory.THROTTLED
    assert retry_info(event).delay_sec == 3.5
    assert "Retry after 3.5 seconds" in str(error)
    assert "corr-123" in str(error)
    assert "diagnostic.jsonl" in str(error)


def test_quota_error_is_not_presented_as_retryable(tmp_path):
    error = command_failure(
        {
            "category": "quota",
            "reason": "ORG_QUOTA_EXCEEDED",
            "message": "Capacity exhausted",
            "retryInfo": {"retryAfterSeconds": 2},
        },
        return_code=1,
        correlation_id="corr-123",
        diagnostic_path=tmp_path / "diagnostic.jsonl",
    )

    assert "Retrying will not help" in str(error)
    assert "Retry after" not in str(error)


def test_protojson_retry_duration_is_preserved():
    assert retry_info({"retryInfo": {"retryDelay": "1.25s"}}).delay_sec == 1.25


def test_machine_error_omits_unstructured_external_text(tmp_path):
    error = command_failure(
        {
            "category": "authentication",
            "reason": "UNAUTHENTICATED",
            "message": "bare-credential-value-9f4c",
            "hint": "another-bare-secret",
        },
        return_code=1,
        correlation_id="corr-123",
        diagnostic_path=tmp_path / "diagnostic.jsonl",
    )

    assert "bare-credential-value-9f4c" not in str(error)
    assert "another-bare-secret" not in str(error)
    assert "UNAUTHENTICATED" in str(error)
