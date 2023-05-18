import logging
import tempfile
import time
from pathlib import Path
from threading import Thread

import numpy as np
import pytest
import requests
from truss.constants import PYTORCH
from truss.model_frameworks import SKLearn
from truss.model_inference import (
    infer_model_information,
    map_to_supported_python_version,
    validate_provided_parameters_with_model,
)
from truss.tests.test_testing_utilities_for_other_tests import ensure_kill_all
from truss.truss_handle import TrussHandle

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


def test_infer_model_information(pytorch_model_with_init_args):
    model_info = infer_model_information(pytorch_model_with_init_args[0])
    assert model_info.model_framework == PYTORCH
    assert model_info.model_type == "MyModel"


@pytest.mark.parametrize(
    "python_version, expected_python_version",
    [
        ("py37", "py38"),
        ("py38", "py38"),
        ("py39", "py39"),
        ("py310", "py310"),
        ("py311", "py311"),
        ("py312", "py311"),
        ("py36", "py38"),
    ],
)
def test_map_to_supported_python_version(python_version, expected_python_version):
    out_python_version = map_to_supported_python_version(python_version)
    assert out_python_version == expected_python_version


@pytest.mark.integration
def test_binary_request(sklearn_rfc_model):
    with ensure_kill_all(), tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        truss_dir = Path(tmp_work_dir, "truss")
        sklearn_framework = SKLearn()
        sklearn_framework.to_truss(sklearn_rfc_model, truss_dir)
        tr = TrussHandle(truss_dir)
        predictions = tr.docker_predict([[0, 0, 0, 0]], local_port=8090, binary=True)
        assert len(predictions["probabilities"]) == 1
        assert np.shape(predictions["probabilities"]) == (1, 3)


@pytest.mark.integration
def test_slow_truss():
    with ensure_kill_all():
        truss_root = Path(__file__).parent.parent.parent.resolve() / "truss"
        truss_dir = truss_root / "test_data" / "server_conformance_test_truss"
        tr = TrussHandle(truss_dir)

        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=False)

        truss_server_addr = "http://localhost:8090"

        def _test_liveness_probe(expected_code):
            live = requests.get(f"{truss_server_addr}/")
            assert live.status_code == expected_code

        def _test_readiness_probe(expected_code):
            ready = requests.get(f"{truss_server_addr}/v1/models/model")
            assert ready.status_code == expected_code

        SERVER_WARMUP_TIME = 3
        LOAD_TEST_TIME = 12
        LOAD_BUFFER_TIME = 7
        PREDICT_TEST_TIME = 15

        # Sleep a few seconds to get the server some time to wake up
        time.sleep(SERVER_WARMUP_TIME)

        # The truss takes about 30 seconds to load.
        # We want to make sure that it's not ready for that time.
        for _ in range(LOAD_TEST_TIME):
            _test_liveness_probe(200)
            _test_readiness_probe(503)
            time.sleep(1)

        time.sleep(LOAD_BUFFER_TIME)
        _test_liveness_probe(200)
        _test_readiness_probe(200)

        predict_call = Thread(
            target=lambda: requests.post(
                f"{truss_server_addr}/v1/models/model:predict", json={}
            )
        )
        predict_call.start()

        for _ in range(PREDICT_TEST_TIME):
            _test_liveness_probe(200)
            _test_readiness_probe(200)
            time.sleep(1)

        predict_call.join()
