import logging
from threading import Thread
from time import sleep

import pytest
import requests
from tornado.ioloop import IOLoop
from truss.constants import PYTORCH
from truss.model_inference import (
    infer_model_information,
    map_to_supported_python_version,
    validate_provided_parameters_with_model,
)

logger = logging.getLogger(__name__)


def test_pytorch_init_arg_validation(
    pytorch_model_with_init_args, pytorch_model_init_args
):
    pytorch_model_with_init_args, _ = pytorch_model_with_init_args
    # Validates with args and kwargs
    validate_provided_parameters_with_model(
        pytorch_model_with_init_args.__class__, pytorch_model_init_args
    )

    # Errors if bad args
    with pytest.raises(ValueError):
        validate_provided_parameters_with_model(
            pytorch_model_with_init_args.__class__, {"foo": "bar"}
        )

    # Validates with only args
    copied_args = pytorch_model_init_args.copy()
    copied_args.pop("kwarg1")
    copied_args.pop("kwarg2")
    validate_provided_parameters_with_model(pytorch_model_with_init_args, copied_args)

    # Requires all args
    with pytest.raises(ValueError):
        validate_provided_parameters_with_model(pytorch_model_with_init_args, {})


@pytest.mark.skip(
    reason="package resolution works inconsistently between unit and integration tests"
)
def test_slow_load_model():
    # todo fix module resolution to be consistent in all environments
    # config_path = f"{pathlib.Path(__file__).parent.resolve()}/../test_data/models/slow_load/{CONFIG_FILE}"
    port = 8998
    # server = ConfiguredTrussServer(config_path, port)
    url = f"http://localhost:{port}"
    loop = IOLoop()

    def start():
        loop.make_current()
        # server.start()

    server_thread = Thread(target=start)
    server_thread.start()

    #  wait for startup
    sleep(1)

    try:
        #  liveness should be good right away
        resp = requests.get(url)
        assert resp.status_code == 200

        #  readiness should not be ready due to a long load
        resp = requests.get(f"{url}/v1/models/model")
        assert resp.status_code == 503

        #  wait for our long load to complete
        sleep(3)

        #  now we should be ready to serve traffic, post load
        resp = requests.get(f"{url}/v1/models/model")
        assert resp.status_code == 200
    finally:
        # stop the asyncio loop to kill truss server
        loop.stop()

        # wait for server to fully exit
        server_thread.join(10)


def test_infer_model_information(pytorch_model_with_init_args):
    model_info = infer_model_information(pytorch_model_with_init_args[0])
    assert model_info.model_framework == PYTORCH
    assert model_info.model_type == "MyModel"


@pytest.mark.parametrize(
    "python_version, expected_python_version",
    [
        ("py37", "py37"),
        ("py38", "py38"),
        ("py39", "py39"),
        ("py310", "py39"),
        ("py311", "py39"),
        ("py36", "py37"),
    ],
)
def test_map_to_supported_python_version(python_version, expected_python_version):
    out_python_version = map_to_supported_python_version(python_version)
    assert out_python_version == expected_python_version
