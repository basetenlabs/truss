import logging
from pathlib import Path

import pytest
import requests
from truss.tests.test_testing_utilities_for_other_tests import ensure_kill_all

from truss_chains import definitions, framework, public_api, remote, utils

utils.setup_dev_logging(logging.DEBUG)


@pytest.mark.integration
def test_chain():
    with ensure_kill_all():
        root = Path(__file__).parent.resolve()
        chain_root = root / "itest_chain" / "itest_chain.py"
        with framework.import_target(chain_root, "ItestChain") as entrypoint:
            options = definitions.PushOptionsLocalDocker(
                chain_name="integration-test",
                user_env={"test_env_key": "test_env_value"},
            )
            service = remote.push(entrypoint, options)

        url = service.run_remote_url.replace("host.docker.internal", "localhost")

        # Call without providing values for default arguments.
        response = requests.post(
            url,
            json={"length": 30, "num_partitions": 3},
            headers={"traceparent": "TEST TEST TEST"},
        )
        print(response.content)
        assert response.status_code == 200
        assert response.json() == [
            6280,
            "erodfderodfderodfderodfderodfd",
            123,
            {
                "parts": [],
                "part_lens": [10],
            },
            ["a", "b"],
            "test_env_value",
        ]
        # Call with values for default arguments.
        response = requests.post(
            url,
            json={
                "length": 30,
                "num_partitions": 3,
                "pydantic_default_arg": {
                    "parts": ["marius"],
                    "part_lens": [3],
                },
                "simple_default_arg": ["bola"],
            },
        )
        print(response.content)
        assert response.status_code == 200
        assert response.json() == [
            6280,
            "erodfderodfderodfderodfderodfd",
            123,
            {
                "parts": ["marius"],
                "part_lens": [3],
            },
            ["bola"],
            "test_env_value",
        ]

        # Test with errors.
        response = requests.post(
            url, json={"length": 300, "num_partitions": 3}, stream=True
        )
        print(response)
        error = definitions.RemoteErrorDetail.model_validate(response.json()["error"])
        error_str = error.format()
        print(error_str)
        assert "ValueError: This input is too long: 100." in error_str
        assert response.status_code == 500


@pytest.mark.asyncio
async def test_chain_local():
    root = Path(__file__).parent.resolve()
    chain_root = root / "itest_chain" / "itest_chain.py"
    with framework.import_target(chain_root, "ItestChain") as entrypoint:
        with public_api.run_local(user_env={"test_env_key": "test_env_value"}):
            with pytest.raises(ValueError):
                # First time `SplitTextFailOnce` raises an error and
                # currently local mode does not have retries.
                await entrypoint().run_remote(length=20, num_partitions=5)

            result = await entrypoint().run_remote(length=20, num_partitions=5)
            print(result)
            expected = (
                4198,
                "erodfderodfderodfder",
                123,
                {
                    "parts": [],
                    "part_lens": [10],
                },
                ["a", "b"],
                "test_env_value",
            )

            # Convert the pydantic model to a dict for comparison
            result_dict = (
                result[0],
                result[1],
                result[2],
                result[3].dict(),
                result[4],
                result[5],
            )

            assert result_dict == expected

        with pytest.raises(
            definitions.ChainsRuntimeError,
            match="Chainlets cannot be naively instantiated",
        ):
            await entrypoint().run_remote(length=20, num_partitions=5)
