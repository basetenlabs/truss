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
        tests_root = Path(__file__).parent.resolve()
        chain_root = tests_root / "itest_chain" / "itest_chain.py"
        with framework.import_target(chain_root, "ItestChain") as entrypoint:
            options = definitions.PushOptionsLocalDocker(
                chain_name="integration-test", use_local_chains_src=True
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
    tests_root = Path(__file__).parent.resolve()
    chain_root = tests_root / "itest_chain" / "itest_chain.py"
    with framework.import_target(chain_root, "ItestChain") as entrypoint:
        with public_api.run_local():
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
            )

            # Convert the pydantic model to a dict for comparison
            result_dict = (
                result[0],
                result[1],
                result[2],
                result[3].dict(),
                result[4],
            )

            assert result_dict == expected

        with pytest.raises(
            definitions.ChainsRuntimeError,
            match="Chainlets cannot be naively instantiated",
        ):
            await entrypoint().run_remote(length=20, num_partitions=5)


@pytest.mark.integration
def test_streaming_chain():
    examples_root = Path(__file__).parent.parent.resolve() / "examples"
    chain_root = examples_root / "streaming" / "streaming_chain.py"
    with framework.import_target(chain_root, "Consumer") as entrypoint:
        service = remote.push(
            entrypoint,
            options=definitions.PushOptionsLocalDocker(
                chain_name="stream",
                only_generate_trusses=False,
                use_local_chains_src=True,
            ),
        )
        assert service is not None
        response = service.run_remote({})
        assert response.status_code == 200
        print(response.json())
        result = response.json()
        print(result)
        assert result["header"]["msg"] == "Start."
        assert result["chunks"][0]["words"] == ["G"]
        assert result["chunks"][1]["words"] == ["G", "HH"]
        assert result["chunks"][2]["words"] == ["G", "HH", "III"]
        assert result["chunks"][3]["words"] == ["G", "HH", "III", "JJJJ"]
        assert result["footer"]["duration_sec"] > 0
        assert result["strings"] == ["First second last."]


@pytest.mark.asyncio
async def test_streaming_chain_local():
    examples_root = Path(__file__).parent.parent.resolve() / "examples"
    chain_root = examples_root / "streaming" / "streaming_chain.py"
    with framework.import_target(chain_root, "Consumer") as entrypoint:
        with public_api.run_local():
            result = await entrypoint().run_remote()
            print(result)
            assert result.header.msg == "Start."
            assert result.chunks[0].words == ["G"]
            assert result.chunks[1].words == ["G", "HH"]
            assert result.chunks[2].words == ["G", "HH", "III"]
            assert result.chunks[3].words == ["G", "HH", "III", "JJJJ"]
            assert result.footer.duration_sec > 0
            assert result.strings == ["First second last."]
