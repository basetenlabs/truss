import logging
from pathlib import Path

import pytest
import requests
from truss.tests.test_testing_utilities_for_other_tests import ensure_kill_all
from truss_chains import definitions, deploy, framework, public_api, utils

utils.setup_dev_logging(logging.DEBUG)


@pytest.mark.integration
def test_chain():
    with ensure_kill_all():
        root = Path(__file__).parent.resolve()
        chain_root = root / "itest_chain" / "itest_chain.py"
        with framework.import_target(chain_root, "ItestChain") as entrypoint:
            options = definitions.DeploymentOptionsLocalDocker(
                chain_name="integration-test"
            )
            service = deploy.deploy_remotely(entrypoint, options)

        url = service.run_url.replace("host.docker.internal", "localhost")

        response = requests.post(
            url, json={"length": 30, "num_partitions": 3}, stream=True
        )
        print(response.content)
        assert response.status_code == 200
        assert response.json() == [6280, "erodfderodfderodfderodfderodfd", 123]

        # Test with errors.
        response = requests.post(
            url, json={"length": 300, "num_partitions": 3}, stream=True
        )
        print(response)
        error = definitions.RemoteErrorDetail.parse_obj(response.json()["error"])
        error_str = error.format()
        print(error_str)
        assert "ValueError: This input is too long: 100." in error_str
        assert response.status_code == 500


@pytest.mark.asyncio
async def test_chain_local():
    root = Path(__file__).parent.resolve()
    chain_root = root / "itest_chain" / "itest_chain.py"
    with framework.import_target(chain_root, "ItestChain") as entrypoint:

        with pytest.raises(definitions.ChainsRuntimeError):
            entrypoint().run(length=20, num_partitions=5)

        with public_api.run_local():
            with pytest.raises(ValueError):
                # First time `SplitTextFailOnce` raises an error and
                # currently local mode does not have retries.
                result = await entrypoint().run(length=20, num_partitions=5)

            result = await entrypoint().run(length=20, num_partitions=5)
            assert result == (4198, "erodfderodfderodfder", 123)
            print(result)
