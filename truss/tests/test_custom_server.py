import pytest
import requests
from tenacity import stop_after_attempt

from truss.local.local_config_handler import LocalConfigHandler
from truss.tests.test_testing_utilities_for_other_tests import ensure_kill_all
from truss.truss_handle.truss_handle import TrussHandle


@pytest.mark.integration
def test_custom_server_truss(test_data_path):
    with ensure_kill_all():
        print("Running test_custom_server_truss")
        truss_dir = test_data_path / "test_custom_server_truss"
        print(f"truss_dir: {truss_dir}")
        tr = TrussHandle(truss_dir)
        print("Setting secret")
        LocalConfigHandler.set_secret("hf_access_token", "123")
        try:
            print("Starting container")
            _ = tr.docker_run(
                local_port=8090,
                detach=True,
                wait_for_server_ready=True,
                model_server_stop_retry_override=stop_after_attempt(3),
            )
        except Exception as e:
            raise Exception(f"Failed to start container: {e}")
        truss_server_addr = "http://localhost:8090"
        full_url = f"{truss_server_addr}/v1/models/model:predict"

        response = requests.post(full_url, json={})
        assert response.status_code == 200
        assert response.json() == {
            "message": "Hello World",
            "is_env_var_passed": True,
            "is_secret_mounted": True,
        }
