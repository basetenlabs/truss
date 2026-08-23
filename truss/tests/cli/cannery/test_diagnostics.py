import json

import click
import pytest

from truss.cli.cannery.diagnostics import DiagnosticLog, endpoint_hostname, redact_text


def test_redaction_removes_common_credential_forms():
    value = (
        "Authorization: Bearer abc.def api_key=top-secret refresh-token: refresh-secret"
    )

    redacted = redact_text(value)

    assert "abc.def" not in redacted
    assert "top-secret" not in redacted
    assert "refresh-secret" not in redacted
    assert redacted.count("[REDACTED]") >= 3


def test_endpoint_diagnostics_exclude_credentials_path_and_query():
    assert (
        endpoint_hostname("https://user:password@bdn.baseten.co/private?token=secret")
        == "bdn.baseten.co"
    )


def test_diagnostic_log_is_private_structured_and_allowlisted(monkeypatch, tmp_path):
    monkeypatch.setenv("TRUSS_CANNERY_DIAGNOSTIC_DIR", str(tmp_path / "logs"))
    diagnostic = DiagnosticLog.create("corr-123")

    diagnostic.record(
        "failed",
        operation="push",
        message="Bearer token-value",
        authorization="must-never-be-written",
        environment={"TOKEN": "must-never-be-written"},
    )

    assert diagnostic.path.stat().st_mode & 0o777 == 0o600
    assert diagnostic.path.parent.stat().st_mode & 0o777 == 0o700
    raw = diagnostic.path.read_text()
    assert "token-value" not in raw
    assert "must-never-be-written" not in raw
    payload = json.loads(raw)
    assert payload["correlation_id"] == "corr-123"
    assert payload["operation"] == "push"


def test_diagnostic_log_rejects_symlink_replacement(monkeypatch, tmp_path):
    monkeypatch.setenv("TRUSS_CANNERY_DIAGNOSTIC_DIR", str(tmp_path / "logs"))
    diagnostic = DiagnosticLog.create("corr-123")
    target = tmp_path / "target"
    target.write_text("unchanged")
    diagnostic.path.unlink()
    diagnostic.path.symlink_to(target)

    with pytest.raises(click.ClickException, match="symlink"):
        diagnostic.record("failed", message="safe")

    assert target.read_text() == "unchanged"
