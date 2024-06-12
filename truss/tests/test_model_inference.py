import concurrent
import inspect
import json
import logging
import tempfile
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Thread

import pytest
import requests
from requests.exceptions import RequestException

from truss.local.local_config_handler import LocalConfigHandler
from truss.model_inference import map_to_supported_python_version
from truss.tests.helpers import create_truss
from truss.tests.test_testing_utilities_for_other_tests import ensure_kill_all
from truss.truss_handle import TrussHandle

logger = logging.getLogger(__name__)

DEFAULT_LOG_ERROR = "Internal Server Error"


def _log_contains_error(line: dict, error: str, message: str):
    return (
        line["levelname"] == "ERROR"
        and line["message"] == message
        and error in line["exc_info"]
    )


def assert_logs_contain_error(logs: str, error: str, message=DEFAULT_LOG_ERROR):
    loglines = logs.splitlines()
    assert any(
        _log_contains_error(json.loads(line), error, message) for line in loglines
    )


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
def test_requirements_file_truss():
    with ensure_kill_all():
        truss_root = Path(__file__).parent.parent.parent.resolve() / "truss"

        truss_dir = truss_root / "test_data" / "test_requirements_file_truss"

        tr = TrussHandle(truss_dir)

        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)
        truss_server_addr = "http://localhost:8090"
        full_url = f"{truss_server_addr}/v1/models/model:predict"

        # The prediction imports torch which is specified in a requirements.txt and returns if GPU is available.
        response = requests.post(full_url, json={})
        assert response.status_code == 200
        assert response.json() is False


@pytest.mark.integration
@pytest.mark.parametrize("pydantic_major_version", ["1", "2"])
def test_requirements_pydantic(pydantic_major_version):
    with ensure_kill_all():
        truss_root = Path(__file__).parent.parent.parent.resolve() / "truss"

        truss_dir = truss_root / "test_data" / f"test_pyantic_v{pydantic_major_version}"

        tr = TrussHandle(truss_dir)

        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)
        truss_server_addr = "http://localhost:8090"
        full_url = f"{truss_server_addr}/v1/models/model:predict"

        response = requests.post(full_url, json={})
        assert response.status_code == 200
        assert response.json() == '{\n    "foo": "bla",\n    "bar": 123\n}'


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
def test_async_streaming_timeout():
    with ensure_kill_all():
        truss_root = Path(__file__).parent.parent.parent.resolve() / "truss"

        truss_dir = truss_root / "test_data" / "test_streaming_read_timeout"

        tr = TrussHandle(truss_dir)

        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=True
        )
        truss_server_addr = "http://localhost:8090"
        predict_url = f"{truss_server_addr}/v1/models/model:predict"

        # ChunkedEncodingError is raised when the chunk does not get processed due to streaming read timeout
        with pytest.raises(requests.exceptions.ChunkedEncodingError):
            response = requests.post(predict_url, json={}, stream=True)

            for chunk in response.iter_content():
                pass

        # Check to ensure the Timeout error is in the container logs
        assert_logs_contain_error(
            container.logs(),
            error="raise exceptions.TimeoutError()",
            message="Exception in ASGI application\n",
        )


@pytest.mark.integration
def test_streaming_with_error():
    with ensure_kill_all():
        truss_root = Path(__file__).parent.parent.parent.resolve() / "truss"

        truss_dir = truss_root / "test_data" / "test_streaming_truss_with_error"

        tr = TrussHandle(truss_dir)

        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)
        truss_server_addr = "http://localhost:8090"
        predict_url = f"{truss_server_addr}/v1/models/model:predict"

        predict_error_response = requests.post(
            predict_url, json={"throw_error": True}, stream=True, timeout=2
        )

        # In error cases, the response will return whatever the stream returned,
        # in this case, the first 3 items. We timeout after 2 seconds to ensure that
        # stream finishes reading and releases the predict semaphore.
        assert [
            byte_string.decode()
            for byte_string in predict_error_response.iter_content()
        ] == ["0", "1", "2"]

        # Test that we are able to continue to make requests successfully
        predict_non_error_response = requests.post(
            predict_url, json={"throw_error": False}, stream=True, timeout=2
        )

        assert [
            byte_string.decode()
            for byte_string in predict_non_error_response.iter_content()
        ] == [
            "0",
            "1",
            "2",
            "3",
            "4",
        ]


@pytest.mark.integration
def test_streaming_truss():
    with ensure_kill_all():
        truss_root = Path(__file__).parent.parent.parent.resolve() / "truss"
        truss_dir = truss_root / "test_data" / "test_streaming_truss"
        tr = TrussHandle(truss_dir)

        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)

        truss_server_addr = "http://localhost:8090"
        predict_url = f"{truss_server_addr}/v1/models/model:predict"

        # A request for which response is not completely read
        predict_response = requests.post(predict_url, json={}, stream=True)
        # We just read the first part and leave it hanging here
        next(predict_response.iter_content())

        predict_response = requests.post(predict_url, json={}, stream=True)

        assert predict_response.headers.get("transfer-encoding") == "chunked"
        assert [
            byte_string.decode()
            for byte_string in list(predict_response.iter_content())
        ] == [
            "0",
            "1",
            "2",
            "3",
            "4",
        ]

        # When accept is set to application/json, the response is not streamed.
        predict_non_stream_response = requests.post(
            predict_url,
            json={},
            stream=True,
            headers={"accept": "application/json"},
        )
        assert "transfer-encoding" not in predict_non_stream_response.headers
        assert predict_non_stream_response.json() == "01234"

        # Test that concurrency work correctly. The streaming Truss has a configured
        # concurrency of 1, so only one request can be in flight at a time. Each request
        # takes 2 seconds, so with a timeout of 3 seconds, we expect the first request to
        # succeed and for the second to timeout.
        #
        # Note that with streamed requests, requests.post raises a ReadTimeout exception if
        # `timeout` seconds has passed since receiving any data from the server.
        def make_request(delay: int):
            # For streamed responses, requests does not start receiving content from server until
            # `iter_content` is called, so we must call this in order to get an actual timeout.
            time.sleep(delay)
            list(requests.post(predict_url, json={}, stream=True).iter_content())

        with ThreadPoolExecutor() as e:
            # We use concurrent.futures.wait instead of the timeout property
            # on requests, since requests timeout property has a complex interaction
            # with streaming.
            first_request = e.submit(make_request, 0)
            second_request = e.submit(make_request, 0.2)
            futures = [first_request, second_request]
            done, not_done = concurrent.futures.wait(futures, timeout=3)
            assert first_request in done
            assert second_request in not_done


@pytest.mark.integration
def test_secrets_truss():
    class Model:
        def __init__(self, **kwargs):
            self._secrets = kwargs["secrets"]

        def predict(self, request):
            return self._secrets["secret"]

    config = """model_name: secrets-truss
secrets:
    secret: null
    """

    config_with_no_secret = "model_name: secrets-truss"
    missing_secret_error_message = """Secret 'secret' not found. Please ensure that:
  * Secret 'secret' is defined in the 'secrets' section of the Truss config file
  * The model was pushed with the --trusted flag"""

    with ensure_kill_all(), tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        truss_dir = Path(tmp_work_dir, "truss")

        create_truss(truss_dir, config, textwrap.dedent(inspect.getsource(Model)))

        tr = TrussHandle(truss_dir)
        LocalConfigHandler.set_secret("secret", "secret_value")
        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)
        truss_server_addr = "http://localhost:8090"
        full_url = f"{truss_server_addr}/v1/models/model:predict"

        response = requests.post(full_url, json={})

        assert response.json() == "secret_value"

    with ensure_kill_all(), tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        # Case where the secret is not specified in the config
        truss_dir = Path(tmp_work_dir, "truss")

        create_truss(
            truss_dir, config_with_no_secret, textwrap.dedent(inspect.getsource(Model))
        )
        tr = TrussHandle(truss_dir)
        LocalConfigHandler.set_secret("secret", "secret_value")
        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=True
        )
        truss_server_addr = "http://localhost:8090"
        full_url = f"{truss_server_addr}/v1/models/model:predict"

        response = requests.post(full_url, json={})

        assert "error" in response.json()

        assert_logs_contain_error(container.logs(), missing_secret_error_message)
        assert "Internal Server Error" in response.json()["error"]

    with ensure_kill_all(), tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        # Case where the secret is not mounted
        truss_dir = Path(tmp_work_dir, "truss")

        create_truss(truss_dir, config, textwrap.dedent(inspect.getsource(Model)))
        tr = TrussHandle(truss_dir)
        LocalConfigHandler.remove_secret("secret")
        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=True
        )
        truss_server_addr = "http://localhost:8090"
        full_url = f"{truss_server_addr}/v1/models/model:predict"

        response = requests.post(full_url, json={})
        assert response.status_code == 500

        assert_logs_contain_error(container.logs(), missing_secret_error_message)
        assert "Internal Server Error" in response.json()["error"]


@pytest.mark.integration
def test_postprocess_with_streaming_predict():
    """
    Test a Truss that has streaming response from both predict and postprocess.
    In this case, the postprocess step continues to happen within the predict lock,
    so we don't bother testing the lock scenario, just the behavior that the postprocess
    function is applied.
    """
    model = """
    import time

    class Model:
        def postprocess(self, response):
            for item in response:
                time.sleep(1)
                yield item + " modified"

        def predict(self, request):
            for i in range(2):
                yield str(i)
    """

    config = "model_name: error-truss"
    with ensure_kill_all(), tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        truss_dir = Path(tmp_work_dir, "truss")

        create_truss(truss_dir, config, textwrap.dedent(model))

        tr = TrussHandle(truss_dir)
        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)
        truss_server_addr = "http://localhost:8090"
        full_url = f"{truss_server_addr}/v1/models/model:predict"
        response = requests.post(full_url, json={}, stream=True)
        # Note that the postprocess function is applied to the
        # streamed response.
        assert response.content == b"0 modified1 modified"


@pytest.mark.integration
def test_streaming_postprocess():
    """
    Tests a Truss where predict returns non-streaming, but postprocess is streamd, and
    ensures that the postprocess step does not happen within the predict lock. To do this,
    we sleep for two seconds during the postprocess streaming process, and fire off two
    requests with a total timeout of 3 seconds, ensuring that if they were serialized
    the test would fail.
    """
    model = """
    import time

    class Model:
        def postprocess(self, response):
            for item in response:
                time.sleep(1)
                yield item + " modified"

        def predict(self, request):
            return ["0", "1"]
    """

    config = "model_name: error-truss"
    with ensure_kill_all(), tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        truss_dir = Path(tmp_work_dir, "truss")

        create_truss(truss_dir, config, textwrap.dedent(model))

        tr = TrussHandle(truss_dir)
        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)
        truss_server_addr = "http://localhost:8090"
        full_url = f"{truss_server_addr}/v1/models/model:predict"

        def make_request(delay: int):
            # For streamed responses, requests does not start receiving content from server until
            # `iter_content` is called, so we must call this in order to get an actual timeout.
            time.sleep(delay)
            response = requests.post(full_url, json={}, stream=True)

            assert response.status_code == 200
            assert response.content == b"0 modified1 modified"

        with ThreadPoolExecutor() as e:
            # We use concurrent.futures.wait instead of the timeout property
            # on requests, since requests timeout property has a complex interaction
            # with streaming.
            first_request = e.submit(make_request, 0)
            second_request = e.submit(make_request, 0.2)
            futures = [first_request, second_request]
            done, _ = concurrent.futures.wait(futures, timeout=3)
            # Ensure that both requests complete within the 3 second timeout,
            # as the predict lock is not held through the postprocess step
            assert first_request in done
            assert second_request in done

            for future in done:
                # Ensure that both futures completed without error
                future.result()


@pytest.mark.integration
def test_postprocess():
    """
    Tests a Truss that has a postprocess step defined, and ensures that the
    postprocess does not happen within the predict lock. To do this, we sleep
    for two seconds during the postprocess, and fire off two requests with a total
    timeout of 3 seconds, ensureing that if they were serialized the test would fail.
    """

    model = """
    import time

    class Model:
        def postprocess(self, response):
            updated_items = []
            for item in response:
                time.sleep(1)
                updated_items.append(item + " modified")
            return updated_items

        def predict(self, request):
            return ["0", "1"]

    """

    config = "model_name: error-truss"
    with ensure_kill_all(), tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        truss_dir = Path(tmp_work_dir, "truss")

        create_truss(truss_dir, config, textwrap.dedent(model))

        tr = TrussHandle(truss_dir)
        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)
        truss_server_addr = "http://localhost:8090"
        full_url = f"{truss_server_addr}/v1/models/model:predict"

        def make_request(delay: int):
            time.sleep(delay)
            response = requests.post(full_url, json={})
            assert response.status_code == 200
            assert response.json() == ["0 modified", "1 modified"]

        with ThreadPoolExecutor() as e:
            # We use concurrent.futures.wait instead of the timeout property
            # on requests, since requests timeout property has a complex interaction
            # with streaming.
            first_request = e.submit(make_request, 0)
            second_request = e.submit(make_request, 0.2)
            futures = [first_request, second_request]
            done, _ = concurrent.futures.wait(futures, timeout=3)
            # Ensure that both requests complete within the 3 second timeout,
            # as the predict lock is not held through the postprocess step
            assert first_request in done
            assert second_request in done

            for future in done:
                # Ensure that both futures completed without error
                future.result()


@pytest.mark.integration
def test_truss_with_errors():
    model = """
    class Model:
        def predict(self, request):
            raise ValueError("error")
    """

    config = "model_name: error-truss"

    with ensure_kill_all(), tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        truss_dir = Path(tmp_work_dir, "truss")

        create_truss(truss_dir, config, textwrap.dedent(model))

        tr = TrussHandle(truss_dir)
        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=True
        )
        truss_server_addr = "http://localhost:8090"
        full_url = f"{truss_server_addr}/v1/models/model:predict"

        response = requests.post(full_url, json={})
        assert response.status_code == 500
        assert "error" in response.json()

        assert_logs_contain_error(container.logs(), "ValueError: error")

        assert "Internal Server Error" in response.json()["error"]

    model_preprocess_error = """
    class Model:
        def preprocess(self, request):
            raise ValueError("error")

        def predict(self, request):
            return {"a": "b"}
    """

    with ensure_kill_all(), tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        truss_dir = Path(tmp_work_dir, "truss")

        create_truss(truss_dir, config, textwrap.dedent(model_preprocess_error))

        tr = TrussHandle(truss_dir)
        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=True
        )
        truss_server_addr = "http://localhost:8090"
        full_url = f"{truss_server_addr}/v1/models/model:predict"

        response = requests.post(full_url, json={})
        assert response.status_code == 500
        assert "error" in response.json()

        assert_logs_contain_error(container.logs(), "ValueError: error")
        assert "Internal Server Error" in response.json()["error"]

    model_postprocess_error = """
    class Model:
        def predict(self, request):
            return {"a": "b"}

        def postprocess(self, response):
            raise ValueError("error")
    """

    with ensure_kill_all(), tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        truss_dir = Path(tmp_work_dir, "truss")

        create_truss(truss_dir, config, textwrap.dedent(model_postprocess_error))

        tr = TrussHandle(truss_dir)
        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=True
        )
        truss_server_addr = "http://localhost:8090"
        full_url = f"{truss_server_addr}/v1/models/model:predict"

        response = requests.post(full_url, json={})
        assert response.status_code == 500
        assert "error" in response.json()
        assert_logs_contain_error(container.logs(), "ValueError: error")
        assert "Internal Server Error" in response.json()["error"]

    model_async = """
    class Model:
        async def predict(self, request):
            raise ValueError("error")
    """

    with ensure_kill_all(), tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        truss_dir = Path(tmp_work_dir, "truss")

        create_truss(truss_dir, config, textwrap.dedent(model_async))

        tr = TrussHandle(truss_dir)
        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=True
        )
        truss_server_addr = "http://localhost:8090"
        full_url = f"{truss_server_addr}/v1/models/model:predict"

        response = requests.post(full_url, json={})
        assert response.status_code == 500
        assert "error" in response.json()

        assert_logs_contain_error(container.logs(), "ValueError: error")

        assert "Internal Server Error" in response.json()["error"]


@pytest.mark.integration
def test_truss_with_user_errors():
    """Test that user-code raised `fastapi.HTTPExceptions` are passed through as is."""
    model = """
    import fastapi

    class Model:
        def predict(self, request):
            raise fastapi.HTTPException(status_code=500, detail="My custom message.")
    """

    config = "model_name: error-truss"

    with ensure_kill_all(), tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        truss_dir = Path(tmp_work_dir, "truss")

        create_truss(truss_dir, config, textwrap.dedent(model))

        tr = TrussHandle(truss_dir)
        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=True
        )
        truss_server_addr = "http://localhost:8090"
        full_url = f"{truss_server_addr}/v1/models/model:predict"

        response = requests.post(full_url, json={})
        assert response.status_code == 500
        assert "error" in response.json()

        assert_logs_contain_error(
            container.logs(),
            "HTTPException: 500: My custom message.",
            "Model raised HTTPException",
        )

        assert "My custom message." in response.json()["error"]


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
