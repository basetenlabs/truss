import logging
from unittest import mock

import pytest
import requests
import tenacity

from truss.tests.templates.control.control.conftest import setup_control_imports

setup_control_imports()

from helpers import inference_server_starter  # noqa: E402


@pytest.fixture
def fast_retry(monkeypatch):
    monkeypatch.setattr(inference_server_starter, "PATCH_PING_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(
        inference_server_starter, "wait_exponential", lambda **_: tenacity.wait_none()
    )


@pytest.fixture
def patch_ping_env(monkeypatch):
    monkeypatch.setenv("PATCH_PING_URL_TRUSS", "http://patch-ping.example/ping")


@pytest.fixture
def controller():
    c = mock.Mock()
    c.truss_hash.return_value = "abc123"
    return c


def test_logs_classified_summary_after_retry_exhaustion_does_not_raise(
    fast_retry, patch_ping_env, controller, caplog
):
    err = requests.exceptions.ProxyError("Unable to connect to proxy")
    with mock.patch.object(requests, "post", side_effect=err):
        with caplog.at_level(logging.INFO):
            inference_server_starter.inference_server_startup_flow(
                controller, logging.getLogger("test")
            )

    messages = [r.getMessage() for r in caplog.records]
    levels = [r.levelname for r in caplog.records]

    # First failure logged at WARNING with full type+message and network classification.
    assert any(
        m.startswith("Patch ping network error: ProxyError:") for m in messages
    ), messages
    # Subsequent attempts logged terse at INFO.
    assert any("Patch ping retry 2/3: ProxyError" == m for m in messages), messages
    # Final summary at ERROR mentioning the URL.
    assert "ERROR" in levels
    assert any(
        "Patch ping failed after 3 attempts" in m
        and "http://patch-ping.example/ping" in m
        for m in messages
    ), messages
    controller.start.assert_not_called()


def test_unclassified_error_logged_as_generic(
    fast_retry, patch_ping_env, controller, caplog
):
    with mock.patch.object(requests, "post", side_effect=ValueError("boom")):
        with caplog.at_level(logging.WARNING):
            inference_server_starter.inference_server_startup_flow(
                controller, logging.getLogger("test")
            )

    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("Patch ping error: ValueError:") for m in messages), (
        messages
    )


def test_no_patch_ping_url_starts_immediately(monkeypatch, controller):
    monkeypatch.delenv("PATCH_PING_URL_TRUSS", raising=False)
    inference_server_starter.inference_server_startup_flow(
        controller, logging.getLogger("test")
    )
    controller.start.assert_called_once()
