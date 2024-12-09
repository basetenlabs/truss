import logging
import re
from pathlib import Path

import pytest
import requests
from truss.tests.test_testing_utilities_for_other_tests import ensure_kill_all

from truss_chains import definitions, framework, public_api, utils
from truss_chains.deployment import deployment_client

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
            service = deployment_client.push(entrypoint, options)

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
        assert response.status_code == 500

        error = definitions.RemoteErrorDetail.model_validate(response.json()["error"])
        error_str = error.format()
        print(error_str)

        error_regex = r"""
Chainlet-Traceback \(most recent call last\):
  File \".*?/itest_chain\.py\", line \d+, in run_remote
    value = self\._accumulate_parts\(text_parts\.parts\)
  File \".*?/itest_chain\.py\", line \d+, in _accumulate_parts
    value \+= self\._text_to_num\.run_remote\(part\)
ValueError: \(showing chained remote errors, root error at the bottom\)
├─ Error in dependency Chainlet `TextToNum`:
│   Chainlet-Traceback \(most recent call last\):
│     File \".*?/itest_chain\.py\", line \d+, in run_remote
│       generated_text = self\._replicator\.run_remote\(data\)
│   ValueError: \(showing chained remote errors, root error at the bottom\)
│   ├─ Error in dependency Chainlet `TextReplicator`:
│   │   Chainlet-Traceback \(most recent call last\):
│   │     File \".*?/itest_chain\.py\", line \d+, in run_remote
│   │       validate_data\(data\)
│   │     File \".*?/itest_chain\.py\", line \d+, in validate_data
│   │       raise ValueError\(f\"This input is too long: \{len\(data\)\}\.\"\)
╰   ╰   ValueError: This input is too long: \d+\.
                """
        assert re.match(error_regex.strip(), error_str.strip(), re.MULTILINE)


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
    with ensure_kill_all():
        examples_root = Path(__file__).parent.parent.resolve() / "examples"
        chain_root = examples_root / "streaming" / "streaming_chain.py"
        with framework.import_target(chain_root, "Consumer") as entrypoint:
            service = deployment_client.push(
                entrypoint,
                options=definitions.PushOptionsLocalDocker(
                    chain_name="integration-test-stream",
                    only_generate_trusses=False,
                    use_local_chains_src=True,
                ),
            )
            assert service is not None

            response = service.run_remote({"cause_error": False})
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
            assert result["strings"] == "First second last."

            # TODO: build error handling for stream reader.
            # response = service.run_remote({"cause_error": True})
            # assert response.status_code == 200
            # print(response.json())
            # result = response.json()
            # print(result)


@pytest.mark.asyncio
async def test_streaming_chain_local():
    examples_root = Path(__file__).parent.parent.resolve() / "examples"
    chain_root = examples_root / "streaming" / "streaming_chain.py"
    with framework.import_target(chain_root, "Consumer") as entrypoint:
        with public_api.run_local():
            result = await entrypoint().run_remote(cause_error=False)
            print(result)
            assert result.header.msg == "Start."
            assert result.chunks[0].words == ["G"]
            assert result.chunks[1].words == ["G", "HH"]
            assert result.chunks[2].words == ["G", "HH", "III"]
            assert result.chunks[3].words == ["G", "HH", "III", "JJJJ"]
            assert result.footer.duration_sec > 0
            assert result.strings == "First second last."


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["json", "binary"])
def test_numpy_chain(mode):
    if mode == "json":
        target = "HostJSON"
    else:
        target = "HostBinary"
    with ensure_kill_all():
        examples_root = Path(__file__).parent.parent.resolve() / "examples"
        chain_root = examples_root / "numpy_and_binary" / "chain.py"
        with framework.import_target(chain_root, target) as entrypoint:
            service = deployment_client.push(
                entrypoint,
                options=definitions.PushOptionsLocalDocker(
                    chain_name=f"integration-test-numpy-{mode}",
                    only_generate_trusses=False,
                    use_local_chains_src=True,
                ),
            )
            assert service is not None
            response = service.run_remote({})
            assert response.status_code == 200
            print(response.json())
