from pathlib import Path

import pytest
import requests

from truss.local.local_config_handler import LocalConfigHandler
from truss.tests.test_testing_utilities_for_other_tests import ensure_kill_all
from truss.truss_handle import TrussHandle


@pytest.mark.integration
def test_custom_server_truss():
    with ensure_kill_all():
        truss_root = Path(__file__).parent.parent.parent.resolve() / "truss"

        truss_dir = truss_root / "test_data" / "test_custom_server_truss"

        tr = TrussHandle(truss_dir)
        LocalConfigHandler.set_secret("hf_access_token", "123")
        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)
        truss_server_addr = "http://localhost:8090"
        full_url = f"{truss_server_addr}/v1/models/model:predict"

        try:
            response = requests.post(full_url, json={})
            if response.status_code != 200:
                print(response.text)
        except:
            raise Exception(f"Failed to reach {full_url}")
        assert response.status_code == 200
        assert response.json() == {
            "message": "Hello World",
            "is_env_var_passed": True,
            "is_secret_mounted": True,
        }
