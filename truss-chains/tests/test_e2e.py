import logging
import re
import time
from pathlib import Path

import pytest
import requests
import websockets

from truss.tests.test_testing_utilities_for_other_tests import (
    ensure_kill_all,
    get_container_logs_from_prefix,
)
from truss.truss_handle.build import load
from truss_chains import framework, public_api, public_types, utils
from truss_chains.deployment import deployment_client
from truss_chains.deployment.code_gen import gen_truss_model_from_source

utils.setup_dev_logging(logging.DEBUG)

TEST_ROOT = Path(__file__).parent.resolve()


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.integration
def test_chain():
    with ensure_kill_all():
        chain_root = TEST_ROOT / "itest_chain" / "itest_chain.py"
        with framework.ChainletImporter.import_target(
            chain_root, "ItestChain"
        ) as entrypoint:
            service = deployment_client.push_debug_docker(entrypoint)
        assert service is not None
        time.sleep(1.0)  # Wait for models to be ready.
        url = service.run_remote_url.replace("host.docker.internal", "localhost")

        # Call without providing values for default arguments.
        response = requests.post(
            url,
            json={"length": 30, "num_partitions": 3},
            headers={"traceparent": "TRACE_ID"},
        )
        print(response.content)
        assert response.status_code == 200
        assert response.json() == [
            6280,
            "erodfderodfderodfderodfderodfd",
            123,
            {"parts": [], "part_lens": [10]},
            ["a", "b"],
        ]
        # Call with values for default arguments.
        response = requests.post(
            url,
            json={
                "length": 30,
                "num_partitions": 3,
                "pydantic_default_arg": {"parts": ["marius"], "part_lens": [3]},
                "simple_default_arg": ["bola"],
            },
        )
        print(response.content)
        assert response.status_code == 200
        assert response.json() == [
            6280,
            "erodfderodfderodfderodfderodfd",
            123,
            {"parts": ["marius"], "part_lens": [3]},
            ["bola"],
        ]

        # Test with errors.
        response = requests.post(
            url,
            json={"length": 300, "num_partitions": 3},
            stream=True,
            headers={"traceparent": "TRACE_ID"},
        )
        print(response)
        assert response.status_code == 500

        error = public_types.RemoteErrorDetail.model_validate(response.json()["error"])
        error_str = error.format()
        print(error_str)

        error_regex = r"""
Chainlet-Traceback \(most recent call last\):
  File \".*?/itest_chain\.py\", line \d+, in run_remote
    value = self\._accumulate_parts\(text_parts\.parts\)
  File \".*?/itest_chain\.py\", line \d+, in _accumulate_parts
    value \+= self\._text_to_num\.run_remote\(part\)
ValueError: \(showing chained remote errors, root error at the bottom\)
├─ Error calling dependency Chainlet `TextToNum`, HTTP status=500, trace ID=`TRACE_ID`\.
│   Chainlet-Traceback \(most recent call last\):
│     File \".*?/itest_chain\.py\", line \d+, in run_remote
│       generated_text = self\._replicator\.run_remote\(data\)
│   ValueError: \(showing chained remote errors, root error at the bottom\)
│   ├─ Error calling dependency Chainlet `TextReplicator`, HTTP status=500, trace ID=`TRACE_ID`\.
│   │   Chainlet-Traceback \(most recent call last\):
│   │     File \".*?/itest_chain\.py\", line \d+, in run_remote
│   │       validate_data\(data\)
│   │     File \".*?/itest_chain\.py\", line \d+, in validate_data
│   │       raise ValueError\(f\"This input is too long: \{len\(data\)\}\.\"\)
╰   ╰   ValueError: This input is too long: \d+\.
                """

        assert re.match(error_regex.strip(), error_str.strip(), re.MULTILINE), error_str


@pytest.mark.asyncio
async def test_chain_local():
    chain_root = TEST_ROOT / "itest_chain" / "itest_chain.py"
    with framework.ChainletImporter.import_target(
        chain_root, "ItestChain"
    ) as entrypoint:
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
                {"parts": [], "part_lens": [10]},
                ["a", "b"],
            )

            # Convert the pydantic model to a dict for comparison
            result_dict = (
                result[0],
                result[1],
                result[2],
                result[3].model_dump(),
                result[4],
            )

            assert result_dict == expected

        with pytest.raises(
            public_types.ChainsRuntimeError,
            match="Chainlets cannot be naively instantiated",
        ):
            await entrypoint().run_remote(length=20, num_partitions=5)


@pytest.mark.integration
def test_streaming_chain():
    with ensure_kill_all():
        examples_root = Path(__file__).parent.parent.resolve() / "examples"
        chain_root = examples_root / "streaming" / "streaming_chain.py"
        with framework.ChainletImporter.import_target(
            chain_root, "Consumer"
        ) as entrypoint:
            service = deployment_client.push_debug_docker(entrypoint)
            assert service is not None
            time.sleep(1.0)  # Wait for models to be ready.

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
    with framework.ChainletImporter.import_target(chain_root, "Consumer") as entrypoint:
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
        chain_root = TEST_ROOT / "numpy_and_binary" / "chain.py"
        with framework.ChainletImporter.import_target(chain_root, target) as entrypoint:
            service = deployment_client.push_debug_docker(
                entrypoint, f"numpy-chain-{mode}"
            )
            assert service is not None
            time.sleep(1.0)  # Wait for models to be ready.

            response = service.run_remote({})
            assert response.status_code == 200
            print(response.json())


@pytest.mark.integration
@pytest.mark.asyncio
async def test_timeout():
    with ensure_kill_all():
        chain_root = TEST_ROOT / "timeout" / "timeout_chain.py"
        with framework.ChainletImporter.import_target(
            chain_root, "TimeoutChain"
        ) as entrypoint:
            service = deployment_client.push_debug_docker(entrypoint)

        url = service.run_remote_url.replace("host.docker.internal", "localhost")
        time.sleep(1.0)  # Wait for models to be ready.

        # Async.
        response = requests.post(url, json={"use_sync": False})
        # print(response.content)

        assert response.status_code == 500
        error = public_types.RemoteErrorDetail.model_validate(response.json()["error"])
        error_str = error.format()
        error_regex = r"""
Chainlet-Traceback \(most recent call last\):
  File \".*?/timeout_chain\.py\", line \d+, in run_remote
    result = await self\._dep.run_remote\(\)
TimeoutError: Timeout calling remote Chainlet `Dependency` \(0.5 seconds limit\)\.
        """
        assert re.match(error_regex.strip(), error_str.strip(), re.MULTILINE), error_str

        # Sync:
        sync_response = requests.post(url, json={"use_sync": True})
        assert sync_response.status_code == 500
        sync_error = public_types.RemoteErrorDetail.model_validate(
            sync_response.json()["error"]
        )
        sync_error_str = sync_error.format()
        sync_error_regex = r"""
Chainlet-Traceback \(most recent call last\):
  File \".*?/timeout_chain\.py\", line \d+, in run_remote
    result = self\._dep_sync.run_remote\(\)
TimeoutError: Timeout calling remote Chainlet `DependencySync` \(0.5 seconds limit\)\.
        """
        assert re.match(
            sync_error_regex.strip(), sync_error_str.strip(), re.MULTILINE
        ), sync_error_str


@pytest.mark.integration
def test_traditional_truss():
    with ensure_kill_all():
        chain_root = TEST_ROOT / "traditional_truss" / "truss_model.py"
        truss_dir = gen_truss_model_from_source(chain_root, use_local_src=True)
        truss_handle = load(truss_dir)

        assert truss_handle.spec.config.resources.cpu == "4"
        assert truss_handle.spec.config.model_name == "OverridePassthroughModelName"

        port = utils.get_free_port()
        truss_handle.docker_run(local_port=port, detach=True, network="host")

        response = requests.post(
            f"http://localhost:{port}/v1/models/model:predict",
            json={"call_count_increment": 5},
        )
        assert response.status_code == 200
        assert response.json() == 5


@pytest.mark.integration
def test_custom_health_checks_chain():
    with ensure_kill_all():
        chain_root = TEST_ROOT / "custom_health_checks" / "custom_health_checks.py"
        with framework.ChainletImporter.import_target(
            chain_root, "CustomHealthChecks"
        ) as entrypoint:
            service = deployment_client.push_debug_docker(entrypoint)
            assert service is not None
            time.sleep(1.0)  # Wait for models to be ready.

            health_check_url = service.run_remote_url.split(":predict")[0]

            response = service.run_remote({"fail": False})
            assert response.status_code == 200
            response = requests.get(health_check_url)
            response.status_code == 200
            container_logs = get_container_logs_from_prefix(entrypoint.name)
            assert "Health check failed." not in container_logs

            # Start failing health checks
            _ = service.run_remote({"fail": True})
            response = requests.get(health_check_url)
            assert response.status_code == 503
            container_logs = get_container_logs_from_prefix(entrypoint.name)
            assert container_logs.count("Health check failed.") == 1
            response = requests.get(health_check_url)
            assert response.status_code == 503
            container_logs = get_container_logs_from_prefix(entrypoint.name)
            assert container_logs.count("Health check failed.") == 2


@pytest.mark.integration
async def test_websocket_chain(anyio_backend):
    with ensure_kill_all():
        chain_name = "websocket_chain"
        chain_root = TEST_ROOT / chain_name / f"{chain_name}.py"
        with framework.ChainletImporter.import_target(chain_root) as entrypoint:
            service = deployment_client.push_debug_docker(entrypoint, chain_name)
            assert service is not None
            time.sleep(1.0)  # Wait for models to be ready.

            # Get something like `ws://localhost:38605/v1/websocket`.
            url = service.run_remote_url.replace("http", "ws").replace(
                "v1/models/model:predict", "v1/websocket"
            )
            async with websockets.connect(url) as websocket:
                await websocket.send("Test")
                response = await websocket.recv()
                assert response == "You said: Test."

                await websocket.send("dep")
                response = await websocket.recv()
                assert response == "Hello from dependency, Head."


@pytest.mark.integration
async def test_http_status_propagation(anyio_backend):
    with ensure_kill_all():
        chain_name = "http_errors"
        chain_root = TEST_ROOT / chain_name / f"{chain_name}.py"
        with framework.ChainletImporter.import_target(chain_root) as entrypoint:
            service = deployment_client.push_debug_docker(entrypoint, chain_name)
            assert service is not None
            time.sleep(1.0)  # Wait for models to be ready.

            url = service.run_remote_url.replace("host.docker.internal", "localhost")
            response = requests.post(url, json={"max_value": 12})
            assert response.status_code == 422

            error = public_types.RemoteErrorDetail.model_validate(
                response.json()["error"]
            )
            error_str = error.format()
            error_regex = r"""
Chainlet-Traceback \(most recent call last\):
  File \".*?/http_errors\.py\", line \d+, in run_remote
    num_repetitions = self\._rand_int\.run_remote\(max_value\)
HTTPException: 422: \(showing chained remote errors, root error at the bottom\)
├─ Error calling dependency Chainlet `FailingHelper`, HTTP status=422, trace ID=``\.
│   Chainlet-Traceback \(most recent call last\):
│     File \".*?/http_errors\.py\", line \d+, in run_remote
│       raise fastapi\.HTTPException\(status_code=422, detail="This is a test error."\)
│   HTTPException: 422: This is a test error\.
╰   \(Exception class defined in `fastapi\.exceptions`\.\)
\(Exception class defined in `fastapi\.exceptions`\.\)
            """
            assert re.match(error_regex.strip(), error_str.strip(), re.MULTILINE), (
                error_str
            )

            container_logs = get_container_logs_from_prefix(entrypoint.name)
            assert (
                container_logs.count(
                    "Handled request - POST /v1/models/model:predict HTTP/1.1 422"
                )
                == 1
            )


@pytest.mark.integration
async def test_throttling_chain(anyio_backend):
    with ensure_kill_all():
        chain_name = "throttling_chain"
        chain_root = TEST_ROOT / chain_name / f"{chain_name}.py"
        with framework.ChainletImporter.import_target(chain_root) as entrypoint:
            service = deployment_client.push_debug_docker(entrypoint, chain_name)
            assert service is not None
            time.sleep(1.0)  # Wait for models to be ready.

            # Call dependency below load limit.
            response = service.run_remote({"num_requests": 2})
            assert response.status_code == 200
            runtime = response.json()
            assert 0.5 <= runtime <= 0.53

            response = service.run_remote({"num_requests": 4})
            assert response.status_code == 200
            runtime = response.json()
            assert 0.5 <= runtime <= 0.53

            time.sleep(0.5)  # Wait for log propagation.
            container_logs = get_container_logs_from_prefix(entrypoint.name)
            assert "No queueing" in container_logs
            assert "Queueing calls to `Dependency`" not in container_logs

            # Now "overload" dependency.
            response = service.run_remote({"num_requests": 8})
            assert response.status_code == 200
            runtime = response.json()
            assert 1.0 <= runtime <= 1.3

            time.sleep(0.5)  # Wait for log propagation.
            container_logs = get_container_logs_from_prefix(entrypoint.name)
            assert "Queueing calls to `Dependency`" in container_logs
