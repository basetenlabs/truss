import json
import sys
from pathlib import Path

import pytest
import requests
from slay import definitions, framework
from truss.tests.test_testing_utilities_for_other_tests import ensure_kill_all

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
def test_workflow():
    with ensure_kill_all():
        root = Path(__file__).parent.parent.resolve()
        workflow_root = root / "test_data" / "workflow_text_to_num"

        sys.path.append(str(workflow_root))
        from workflow import Workflow

        sys.path.pop()

        options = definitions.DeploymentOptionsLocalDocker(
            workflow_name="integration-test"
        )
        entrypoint_service = framework.deploy_remotely(Workflow, options)
        predict_url = entrypoint_service.predict_url.replace(
            "host.docker.internal", "localhost"
        )

        response = requests.post(
            predict_url, json={"length": 30, "num_partitions": 3}, stream=True
        )
        print(response.content)
        assert response.status_code == 200
        assert response.json() == [6280, "erodfderodfderodfderodfderodfd", 123]

        # Test with errors.
        response = requests.post(
            predict_url, json={"length": 300, "num_partitions": 3}, stream=True
        )
        print(response)
        error = definitions.RemoteErrorDetail.parse_obj(response.json()["error"])
        print(error.format())
        assert response.status_code == 500
