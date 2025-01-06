import logging
import os
import pathlib
import re
import subprocess
import tempfile
import time
import uuid
from typing import Tuple

import pytest
import pytest_check
from truss.remote.baseten import core
from truss.remote.baseten import remote as b10_remote
from truss.remote.baseten.utils import status as status_utils

from truss_chains import definitions
from truss_chains.remote_chainlet import stub

backend_env_domain = "staging.baseten.co"
BASETEN_API_KEY = os.environ["BASETEN_API_KEY_STAGING"]

BASETEN_REMOTE_URL = f"https://app.{backend_env_domain}"
VENV_PATH = pathlib.Path(os.environ["TRUSS_ENV_PATH"])
CHAINS_ROOT = pathlib.Path(__file__).parent.parent.resolve() / "truss-chains"
URL_RE = re.compile(
    rf"https://chain-([a-zA-Z0-9]+)\.api\.{re.escape(backend_env_domain)}/deployment/([a-zA-Z0-9]+)/run_remote"
)
DEPLOY_TIMEOUT_SEC = 500


def make_stub(url: str, options: definitions.RPCOptions) -> stub.StubBase:
    context = definitions.DeploymentContext(
        chainlet_to_service={},
        secrets={definitions.BASETEN_API_SECRET_NAME: BASETEN_API_KEY},
    )
    return stub.StubBase.from_url(url, context, options)


def write_trussrc(api_key: str, dir_path: pathlib.Path) -> pathlib.Path:
    config = rf"""
        [staging]
        remote_provider = baseten
        api_key = {api_key}
        remote_url = {BASETEN_REMOTE_URL}
        """
    truss_rc_path = dir_path / ".trussrc"
    truss_rc_path.write_text(config)
    return truss_rc_path


@pytest.fixture
def prepare(request):
    temp_dir = pathlib.Path(tempfile.mkdtemp())
    truss_rc_path = write_trussrc(BASETEN_API_KEY, temp_dir)
    remote = b10_remote.BasetenRemote(BASETEN_REMOTE_URL, BASETEN_API_KEY)
    mutable_chain_deployment_id = [None]

    yield temp_dir, truss_rc_path, remote, mutable_chain_deployment_id
    # if not test_failed:
    #     shutil.rmtree(temp_dir, ignore_errors=True)


def generate_traceparent():
    trace_id = uuid.uuid4().hex
    span_id = uuid.uuid4().hex[:16]
    trace_flags = "01"
    traceparent = f"00-{trace_id}-{span_id}-{trace_flags}"
    return traceparent


def run_command(truss_rc_path: pathlib.Path, command: str) -> Tuple[str, str]:
    logging.info(f"Running command `{command}` in VENV `{VENV_PATH}` (subprocess).")
    activate_script = VENV_PATH / "bin" / "activate"
    env = os.environ.copy()
    env["USER_TRUSSRC_PATH"] = str(truss_rc_path)
    full_command = f"bash -c 'source {activate_script} && {command}'"
    result = subprocess.run(
        full_command,
        shell=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    logging.info("Command subprocess finished.")
    return stdout, stderr


def wait_ready(
    remote: b10_remote.BasetenRemote, chain_id: str, chain_deployment_id: str
) -> Tuple[bool, float]:
    logging.info(f"Waiting for chain deployment `{chain_deployment_id}` to be ready.")
    t0 = time.perf_counter()
    success = False
    wait_time_sec = 0.0
    while True:
        chainlets = remote.get_chainlets(chain_deployment_id)
        statuses = [
            status_utils.get_displayable_status(chainlet.status)
            for chainlet in chainlets
        ]
        num_services = len(statuses)
        num_ok = sum(s in [core.ACTIVE_STATUS, "SCALED_TO_ZERO"] for s in statuses)
        num_deploying = sum(s in core.DEPLOYING_STATUSES for s in statuses)
        if num_ok == num_services:
            success = True
            break
        elif num_services - num_ok - num_deploying:
            break
        if (wait_time_sec := time.perf_counter() - t0) > DEPLOY_TIMEOUT_SEC:
            break

        time.sleep(10)

    if success:
        logging.info(f"Deployed ready in {wait_time_sec} sec.")
    else:
        overview_url = f"{BASETEN_REMOTE_URL}/chains/{chain_id}/overview"
        raise Exception(
            f"Could not be invoked within {DEPLOY_TIMEOUT_SEC} sec.\n{chainlets}\n"
            f"Check deployment `{chain_deployment_id}` on {overview_url}."
        )

    return success, wait_time_sec


# Actual tests #########################################################################

# def test_truss_version(prepare):
#     _, truss_rc_path = prepare
#     result = run_command(truss_rc_path, "truss --version")
#     assert result.stdout.strip() == "truss, version 0.9.57"


def test_itest_chain_publish(prepare) -> None:
    remote: b10_remote.BasetenRemote
    tmpdir, truss_rc_path, remote, mutable_chain_deployment_id = prepare

    chain_src = CHAINS_ROOT / "tests" / "itest_chain" / "itest_chain.py"
    command = f"truss chains push {chain_src} --publish --name=itest_publish --no-wait"
    # stdout = (
    #     "https://chain-1lqzvkw4.api.staging.baseten.co/deployment/nwx4d0qy/run_remote"
    # )
    stdout, stderr = run_command(truss_rc_path, command)
    # Warning: Input is not a terminal (fd=0).
    # assert not stderr

    matches = URL_RE.search(stdout)
    assert matches, stdout
    url = matches.group(0)
    chain_id = matches.group(1)
    chain_deployment_id = matches.group(2)
    mutable_chain_deployment_id[0] = chain_deployment_id

    success, wait_time_sec = wait_ready(remote, chain_id, chain_deployment_id)
    pytest_check.less(wait_time_sec, 220, "Deployment took too long.")

    # Test regular invocation.
    chain_stub = make_stub(url, definitions.RPCOptions(timeout_sec=10))
    trace_parent = generate_traceparent()
    with stub.trace_parent_raw(trace_parent):
        result = chain_stub.predict_sync({"length": 30, "num_partitions": 3})

    expected = [
        6280,
        "erodfderodfderodfderodfderodfd",
        123,
        {"parts": [], "part_lens": [10]},
        ["a", "b"],
    ]
    pytest_check.equal(result, expected)

    # Test speed
    invocation_times_sec = []
    for i in range(10):
        t0 = time.perf_counter()
        with stub.trace_parent_raw(trace_parent):
            chain_stub.predict_sync({"length": 30, "num_partitions": 3})
        invocation_times_sec.append(time.perf_counter() - t0)

    invocation_times_sec.sort()
    logging.info(f"Invocation times(sec): {invocation_times_sec}.")
    pytest_check.less(invocation_times_sec[0], 0.32)  # Best of 10, could be <0.30....

    # Test binary invocation.
    chain_stub_binary = make_stub(
        url, definitions.RPCOptions(timeout_sec=10, use_binary=True)
    )
    trace_parent = generate_traceparent()
    with stub.trace_parent_raw(trace_parent):
        result = chain_stub_binary.predict_sync({"length": 30, "num_partitions": 3})

    expected = [
        6280,
        "erodfderodfderodfderodfderodfd",
        123,
        {"parts": [], "part_lens": [10]},
        ["a", "b"],
    ]
    pytest_check.equal(result, expected)

    # Test speed
    invocation_times_sec = []
    for i in range(10):
        t0 = time.perf_counter()
        with stub.trace_parent_raw(trace_parent):
            chain_stub_binary.predict_sync({"length": 30, "num_partitions": 3})
        invocation_times_sec.append(time.perf_counter() - t0)

    invocation_times_sec.sort()
    logging.info(f"Invocation times(sec): {invocation_times_sec}.")
    pytest_check.less(invocation_times_sec[0], 0.32)  # Best of 10, could be <0.30...

    if pytest_check.any_failures():
        logging.info(
            f"There were failures, leaving deployment `{chain_deployment_id}` "
            "undeleted for inspection."
        )
    else:
        logging.info(f"No failures. Deleting deployment `{chain_deployment_id}`.")
        remote.api.delete_chain_deployment(chain_id, chain_deployment_id)


@pytest.mark.skip("Not Implemented.")
def test_itest_chain_development(prepare):
    # 1. Push with watch.
    # 2. Invoke.
    # 3. Edit code.
    # 4. Verify invocation is updated.
    # 5. Start watch and edit code again.
    # 6. Verify invocation is updated.
    # 7. Delete.
    ...
