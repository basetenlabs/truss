from unittest.mock import MagicMock

import pytest

from truss.cli.logs.model_log_watcher import ModelDeploymentLogWatcher
from truss.remote.baseten.api import BasetenApi


def _make_watcher(status):
    api = MagicMock(spec=BasetenApi)
    api.get_deployment.return_value = {"status": status, "is_development": False}
    watcher = ModelDeploymentLogWatcher(api, "model_id", "deployment_id")
    watcher.before_polling()
    return watcher


@pytest.mark.parametrize(
    "status",
    ["BUILDING", "DEPLOYING", "LOADING_MODEL", "ACTIVE", "UPDATING", "WAKING_UP"],
)
def test_should_poll_again_while_running(status):
    assert _make_watcher(status).should_poll_again() is True


def test_should_poll_again_while_scaled_to_zero():
    assert _make_watcher("SCALED_TO_ZERO").should_poll_again() is True


@pytest.mark.parametrize(
    "status", ["BUILD_FAILED", "DEPLOY_FAILED", "INACTIVE", "DEACTIVATING"]
)
def test_should_not_poll_again_once_stopped(status):
    assert _make_watcher(status).should_poll_again() is False


def test_post_poll_refreshes_status():
    api = MagicMock(spec=BasetenApi)
    api.get_deployment.side_effect = [
        {"status": "SCALED_TO_ZERO", "is_development": False},
        {"status": "INACTIVE", "is_development": False},
    ]
    watcher = ModelDeploymentLogWatcher(api, "model_id", "deployment_id")

    watcher.before_polling()
    assert watcher.should_poll_again() is True

    watcher.post_poll()
    assert watcher.should_poll_again() is False
