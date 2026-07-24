from unittest.mock import MagicMock, patch

import pytest

from truss.api.definitions import ModelDeployment
from truss.remote.baseten.core import TERMINAL_FAILURE_STATUSES
from truss.remote.baseten.service import BasetenService


def _make_deployment(statuses):
    mock_service = MagicMock(spec=BasetenService)
    mock_service.model_id = "model_id"
    mock_service.model_version_id = "version_id"
    mock_service.poll_deployment_status.return_value = iter(statuses)
    return ModelDeployment(mock_service)


def test_wait_for_active_returns_true_on_active():
    deployment = _make_deployment(["BUILDING", "DEPLOYING", "ACTIVE"])

    assert deployment.wait_for_active() is True


def test_wait_for_active_returns_true_on_scaled_to_zero():
    deployment = _make_deployment(["BUILDING", "DEPLOYING", "SCALED_TO_ZERO"])

    assert deployment.wait_for_active() is True


def test_wait_for_active_keeps_polling_while_waking_up():
    deployment = _make_deployment(["WAKING_UP", "WAKING_UP", "ACTIVE"])

    assert deployment.wait_for_active() is True


def test_wait_for_active_keeps_polling_on_unknown_status():
    deployment = _make_deployment(["SOME_NEW_BACKEND_STATUS", "ACTIVE"])

    assert deployment.wait_for_active() is True


@pytest.mark.parametrize("status", TERMINAL_FAILURE_STATUSES)
def test_wait_for_active_raises_on_terminal_failure(status):
    deployment = _make_deployment(["BUILDING", status])

    with pytest.raises(ValueError, match=f"Deployment failed with status: {status}"):
        deployment.wait_for_active()


def test_wait_for_active_raises_on_timeout():
    deployment = _make_deployment(["SOME_NEW_BACKEND_STATUS", "ACTIVE"])

    with patch("truss.api.definitions.time.time", side_effect=[0, 100]):
        with pytest.raises(TimeoutError, match="Deployment timed out."):
            deployment.wait_for_active(timeout_seconds=10)
