import logging
import tempfile
import time
from pathlib import Path
from threading import Thread

import numpy as np
import pytest
import requests
from requests.exceptions import RequestException
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


class PropagatingThread(Thread):
    """
    PropagatingThread allows us to run threads and keep track of exceptions
    thrown.
    """

    def run(self):
        self.exc = None
        try:
            self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self, timeout=None):
        super(PropagatingThread, self).join(timeout)
        if self.exc:
            raise self.exc
        return self.ret


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
def test_model_load_failure_truss():
    with ensure_kill_all():
        truss_root = Path(__file__).parent.parent.parent.resolve() / "truss"
        truss_dir = truss_root / "test_data" / "model_load_failure_test"
        tr = TrussHandle(truss_dir)

        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=False)

        # Sleep a few seconds to get the server some time to  wake up
        time.sleep(10)

        truss_server_addr = "http://localhost:8090"

        def handle_request_exception(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except RequestException:
                    return False

            return wrapper

        @handle_request_exception
        def _test_liveness_probe(expected_code):
            live = requests.get(f"{truss_server_addr}/")
            assert live.status_code == expected_code
            return True

        @handle_request_exception
        def _test_readiness_probe(expected_code):
            ready = requests.get(f"{truss_server_addr}/v1/models/model")
            assert ready.status_code == expected_code
            return True

        @handle_request_exception
        def _test_ping(expected_code):
            ping = requests.get(f"{truss_server_addr}/ping")
            assert ping.status_code == expected_code
            return True

        @handle_request_exception
        def _test_invocations(expected_code):
            invocations = requests.post(f"{truss_server_addr}/invocations", json={})
            assert invocations.status_code == expected_code
            return True

        # The server should be completely down so all requests should result in a RequestException.
        # The decorator handle_request_exception catches the RequestException and returns False.
        assert not _test_readiness_probe(expected_code=200)
        assert not _test_liveness_probe(expected_code=200)
        assert not _test_ping(expected_code=200)
        assert not _test_invocations(expected_code=200)


@pytest.mark.integration
def test_concurrency_truss():
    # Tests that concurrency limits work correctly
    with ensure_kill_all():
        truss_root = Path(__file__).parent.parent.parent.resolve() / "truss"

        truss_dir = truss_root / "test_data" / "test_concurrency_truss"

        tr = TrussHandle(truss_dir)

        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)

        truss_server_addr = "http://localhost:8090"
        full_url = f"{truss_server_addr}/v1/models/model:predict"

        # Each request takes 2 seconds, for this thread, we allow
        # a concurrency of 2. This means the first two requests will
        # succeed within the 2 seconds, and the third will fail, since
        # it cannot start until the first two have completed.
        def make_request():
            requests.post(full_url, json={}, timeout=3)

        successful_thread_1 = PropagatingThread(target=make_request)
        successful_thread_2 = PropagatingThread(target=make_request)
        failed_thread = PropagatingThread(target=make_request)

        successful_thread_1.start()
        successful_thread_2.start()
        # Ensure that the thread to fail starts a little after the others
        time.sleep(0.2)
        failed_thread.start()

        successful_thread_1.join()
        successful_thread_2.join()
        with pytest.raises(requests.exceptions.ReadTimeout):
            failed_thread.join()


@pytest.mark.integration
def test_async_truss():
    with ensure_kill_all():
        truss_root = Path(__file__).parent.parent.parent.resolve() / "truss"

        truss_dir = truss_root / "test_data" / "test_async_truss"

        tr = TrussHandle(truss_dir)

        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)
        truss_server_addr = "http://localhost:8090"
        full_url = f"{truss_server_addr}/v1/models/model:predict"

        response = requests.post(full_url, json={})
        assert response.json() == {
            "preprocess_value": "value",
            "postprocess_value": "value",
        }


@pytest.mark.integration
def test_async_streaming():
    with ensure_kill_all():
        truss_root = Path(__file__).parent.parent.parent.resolve() / "truss"

        truss_dir = truss_root / "test_data" / "test_streaming_async_generator_truss"

        tr = TrussHandle(truss_dir)

        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)
        truss_server_addr = "http://localhost:8090"
        full_url = f"{truss_server_addr}/v1/models/model:predict"

        response = requests.post(full_url, json={}, stream=True)
        assert response.headers.get("transfer-encoding") == "chunked"
        assert [
            byte_string.decode() for byte_string in list(response.iter_content())
        ] == ["0", "1", "2", "3", "4"]

        predict_non_stream_response = requests.post(
            full_url,
            json={},
            stream=True,
            headers={"accept": "application/json"},
        )
        assert "transfer-encoding" not in predict_non_stream_response.headers
        assert predict_non_stream_response.json() == "01234"


@pytest.mark.integration
def test_streaming_truss():
    with ensure_kill_all():
        truss_root = Path(__file__).parent.parent.parent.resolve() / "truss"
        truss_dir = truss_root / "test_data" / "test_streaming_truss"
        tr = TrussHandle(truss_dir)

        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)

        truss_server_addr = "http://localhost:8090"

        # A request for which response is not completely read
        predict_response = requests.post(
            f"{truss_server_addr}/v1/models/model:predict", json={}, stream=True
        )
        # We just read the first part and leave it hanging here
        next(predict_response.iter_content())

        predict_response = requests.post(
            f"{truss_server_addr}/v1/models/model:predict", json={}, stream=True
        )

        assert predict_response.headers.get("transfer-encoding") == "chunked"
        assert [
            byte_string.decode()
            for byte_string in list(predict_response.iter_content())
        ] == ["0", "1", "2", "3", "4"]

        # When accept is set to application/json, the response is not streamed.
        predict_non_stream_response = requests.post(
            f"{truss_server_addr}/v1/models/model:predict",
            json={},
            stream=True,
            headers={"accept": "application/json"},
        )
        assert "transfer-encoding" not in predict_non_stream_response.headers
        assert predict_non_stream_response.json() == "01234"


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

        def _test_ping(expected_code):
            ping = requests.get(f"{truss_server_addr}/ping")
            assert ping.status_code == expected_code

        def _test_invocations(expected_code):
            invocations = requests.post(f"{truss_server_addr}/invocations", json={})
            assert invocations.status_code == expected_code

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
            _test_ping(503)
            _test_invocations(503)
            time.sleep(1)

        time.sleep(LOAD_BUFFER_TIME)
        _test_liveness_probe(200)
        _test_readiness_probe(200)
        _test_ping(200)

        predict_call = Thread(
            target=lambda: requests.post(
                f"{truss_server_addr}/v1/models/model:predict", json={}
            )
        )
        predict_call.start()

        for _ in range(PREDICT_TEST_TIME):
            _test_liveness_probe(200)
            _test_readiness_probe(200)
            _test_ping(200)
            time.sleep(1)

        predict_call.join()

        _test_invocations(200)
