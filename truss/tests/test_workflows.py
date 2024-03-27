import json
import logging
import sys
import time
from pathlib import Path

import pytest
import requests
from slay import framework
from truss.tests.test_testing_utilities_for_other_tests import ensure_kill_all

logger = logging.getLogger(__name__)

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
        root = Path(__file__).parent.parent.parent.resolve()

        workflow_root = root / "slay-examples" / "text_to_num"

        sys.path.append(str(workflow_root))

        from workflow import Workflow

        service_descr = framework.deploy_remotely(
            Workflow, "integrationtest", baseten_url="", local_docker=True
        )
        print(service_descr)

        # TODO: get back containers and check their logs.

        url = service_descr.b10_model_url.replace("host.docker.internal", "localhost")
        predict_url = f"{url}/predict"
        print(predict_url)
        response = requests.post(
            predict_url, json={"length": 30, "num_partitions": 4}, stream=True
        )
        print(response)
        print(response.content)
        assert response.status_code == 200
        assert response.json() == ["0 modified", "1 modified"]
        time.sleep(3000)
        sys.path.pop()
