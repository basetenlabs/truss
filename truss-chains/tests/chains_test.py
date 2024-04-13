import json
from pathlib import Path

import pytest
import requests
from truss.tests.test_testing_utilities_for_other_tests import ensure_kill_all
from truss_chains import definitions, deploy, framework

DEFAULT_LOG_ERROR = "Internal Server Error"


def _log_contains_error(line: dict, error: str, message: str):
    return (
        line["levelname"] == "ERROR"
        and line["message"] == message
        and error in line["exc_info"]
    )


def assert_logs_contain_error(logs: str, error: str, message=DEFAULT_LOG_ERROR):
    loglines = logs.splitlines()
    assert any(
        _log_contains_error(json.loads(line), error, message) for line in loglines
    )


@pytest.mark.integration
def test_chain():
    with ensure_kill_all():
        root = Path(__file__).parent.resolve()
        chain_root = root / "itest_chain" / "itest_chain.py"
        entrypoint = framework.import_target(chain_root, "ItestChain")
        options = deploy.DeploymentOptionsLocalDocker(chain_name="integration-test")
        service = deploy.deploy_remotely(entrypoint, options)

        response = requests.post(
            service.run_url, json={"length": 30, "num_partitions": 3}, stream=True
        )
        print(response.content)
        assert response.status_code == 200
        assert response.json() == [6280, "erodfderodfderodfderodfderodfd", 123]

        # Test with errors.
        response = requests.post(
            service.run_url, json={"length": 300, "num_partitions": 3}, stream=True
        )
        print(response)
        error = definitions.RemoteErrorDetail.parse_obj(response.json()["error"])
        print(error.format())
        assert response.status_code == 500
