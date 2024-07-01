import logging
import re
from pathlib import Path

import pytest
import requests
from truss.tests.test_testing_utilities_for_other_tests import ensure_kill_all

import truss_chains as chains
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

        # Call without providing values for default arguments.
        response = requests.post(url, json={"length": 30, "num_partitions": 3})
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
    root = Path(__file__).parent.resolve()
    chain_root = root / "itest_chain" / "itest_chain.py"
    with framework.import_target(chain_root, "ItestChain") as entrypoint:
        with public_api.run_local():
            with pytest.raises(ValueError):
                # First time `SplitTextFailOnce` raises an error and
                # currently local mode does not have retries.
                await entrypoint().run_remote(length=20, num_partitions=5)

            result = await entrypoint().run_remote(length=20, num_partitions=5)
            assert result == (4198, "erodfderodfderodfder", 123)
            print(result)

        with pytest.raises(
            definitions.ChainsRuntimeError,
            match="Chainlets cannot be naively instantiated",
        ):
            await entrypoint().run_remote(length=20, num_partitions=5)


# Chainlet Initialization Guarding #####################################################


def test_raises_without_depends():
    with pytest.raises(definitions.ChainsUsageError, match="chains.provide"):

        class WithoutDepends(chains.ChainletBase):
            def __init__(self, chainlet1):
                self.chainlet1 = chainlet1

            def run_remote(self) -> str:
                return self.chainlet1.run_remote()


class Chainlet1(chains.ChainletBase):
    def run_remote(self) -> str:
        return self.__class__.__name__


class Chainlet2(chains.ChainletBase):
    def run_remote(self) -> str:
        return self.__class__.__name__


class InitInInit(chains.ChainletBase):
    def __init__(self, chainlet2=chains.depends(Chainlet2)):
        self.chainlet1 = Chainlet1()
        self.chainlet2 = chainlet2

    def run_remote(self) -> str:
        return self.chainlet1.run_remote()


class InitInRun(chains.ChainletBase):
    def run_remote(self) -> str:
        Chainlet1()
        return "abc"


def foo():
    return Chainlet1()


class InitWithFn(chains.ChainletBase):
    def __init__(self):
        foo()

    def run_remote(self) -> str:
        return self.__class__.__name__


def test_raises_init_in_init():
    match = "Chainlets cannot be naively instantiated"
    with pytest.raises(definitions.ChainsRuntimeError, match=match):
        with chains.run_local():
            InitInInit()


def test_raises_init_in_run():
    match = "Chainlets cannot be naively instantiated"
    with pytest.raises(definitions.ChainsRuntimeError, match=match):
        with chains.run_local():
            chain = InitInRun()
            chain.run_remote()


def test_raises_init_in_function():
    match = "Chainlets cannot be naively instantiated"
    with pytest.raises(definitions.ChainsRuntimeError, match=match):
        with chains.run_local():
            InitWithFn()


def test_raises_depends_usage():
    class InlinedDepends(chains.ChainletBase):
        def __init__(self):
            self.chainlet1 = chains.depends(Chainlet1)

        def run_remote(self) -> str:
            return self.chainlet1.run_remote()

    match = (
        "`chains.depends(Chainlet1)` was used, but not as "
        "an argument to the `__init__`"
    )
    with pytest.raises(definitions.ChainsRuntimeError, match=re.escape(match)):
        with chains.run_local():
            chain = InlinedDepends()
            chain.run_remote()
