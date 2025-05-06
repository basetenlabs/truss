import asyncio
import concurrent
import contextlib
import inspect
import json
import logging
import pathlib
import sys
import tempfile
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Thread
from typing import Iterator, Mapping, Optional

import httpx
import opentelemetry.trace.propagation.tracecontext as tracecontext
import pytest
import requests
import websockets
from opentelemetry import context, trace
from prometheus_client.parser import text_string_to_metric_families
from python_on_whales import Container
from requests.exceptions import RequestException

from truss.local.local_config_handler import LocalConfigHandler
from truss.tests.helpers import create_truss
from truss.tests.test_testing_utilities_for_other_tests import ensure_kill_all
from truss.truss_handle.truss_handle import TrussHandle, get_docker_urls, wait_for_truss

logger = logging.getLogger(__name__)

DEFAULT_LOG_ERROR = "Internal Server Error"


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _log_contains_line(
    line: dict, message: str, level: str, error: Optional[str] = None
):
    return (
        line["levelname"] == level
        and message in line["message"]
        and (error is None or error in line["exc_info"])
    )


def _assert_logs_contain_error(
    logs: str, error: Optional[str], message=DEFAULT_LOG_ERROR
):
    loglines = [json.loads(line) for line in logs.splitlines()]
    assert any(
        _log_contains_line(line, message, "ERROR", error) for line in loglines
    ), (
        f"Did not find expected error in logs.\nExpected error: {error}\n"
        f"Expected message: {message}\nActual logs:\n{loglines}"
    )


def _assert_logs_contain(logs: str, message: str, level: str = "INFO"):
    loglines = [json.loads(line) for line in logs.splitlines()]
    assert any(_log_contains_line(line, message, level) for line in loglines), (
        f"Did not find expected  logs.\n"
        f"Expected message: {message}\nActual logs:\n{loglines}"
    )


class _PropagatingThread(Thread):
    """
    _PropagatingThread allows us to run threads and keep track of exceptions
    thrown.
    """

    def run(self):
        self.exc = None
        try:
            self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self, timeout=None):
        super(_PropagatingThread, self).join(timeout)
        if self.exc:
            raise self.exc
        return self.ret


@contextlib.contextmanager
def _temp_truss(model_src: str, config_src: str = "") -> Iterator[TrussHandle]:
    with ensure_kill_all(), tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        truss_dir = Path(tmp_work_dir, "truss")
        create_truss(truss_dir, config_src, textwrap.dedent(model_src))
        yield TrussHandle(truss_dir)


# Test Cases ###########################################################################


@pytest.mark.integration
@pytest.mark.parametrize(
    "config_python_version, inspected_python_version",
    [
        ("py38", "3.8"),
        ("py39", "3.9"),
        ("py310", "3.10"),
        ("py311", "3.11"),
        ("py312", "3.12"),
        ("py313", "3.13"),
    ],
)
def test_predict_python_versions(config_python_version, inspected_python_version):
    model = """
    import sys
    class Model:
        def predict(self, data):
            version = sys.version_info
            return f"{version.major}.{version.minor}"
    """
    # config = """base_image:
    #                   image: baseten/truss-server-base:3.13-marius"""
    config = f"python_version: {config_python_version}"
    with ensure_kill_all(), _temp_truss(model, config) as tr:
        container, urls = tr.docker_run_for_test()
        response = requests.post(urls.predict_url, json={})
        assert inspected_python_version == response.json()


@pytest.mark.integration
def test_model_load_logs(test_data_path):
    model = """
    from typing import Optional
    import logging
    class Model:
        def load(self):
            logging.info(f"User Load Message")

        def predict(self, model_input):
            return self.environment_name
    """
    config = "model_name: init-environment-truss"
    with ensure_kill_all(), _temp_truss(model, config) as tr:
        container, urls = tr.docker_run_for_test()
        logs = container.logs()
        _assert_logs_contain(logs, message="Executing model.load()")
        _assert_logs_contain(logs, message="Loading truss model from file")
        _assert_logs_contain(logs, message="Completed model.load()")
        _assert_logs_contain(logs, message="User Load Message")


@pytest.mark.integration
def test_model_load_failure_truss(test_data_path):
    with ensure_kill_all():
        truss_dir = test_data_path / "model_load_failure_test"
        tr = TrussHandle(truss_dir)

        _, urls = tr.docker_run_for_test(wait_for_server_ready=False)

        # Sleep a few seconds to get the server some time to  wake up
        time.sleep(10)

        truss_server_addr = urls.base_url

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
        def _test_is_loaded(expected_code):
            ready = requests.get(f"{truss_server_addr}/v1/models/model/loaded")
            assert ready.status_code == expected_code
            return True

        @handle_request_exception
        def _test_ping(expected_code):
            ping = requests.get(f"{truss_server_addr}/ping")
            assert ping.status_code == expected_code
            return True

        @handle_request_exception
        def _test_invocations(expected_code):
            invocations = requests.post(
                f"{truss_server_addr}/v1/models/model:predict", json={}
            )
            assert invocations.status_code == expected_code
            return True

        # The server should be completely down so all requests should result in a RequestException.
        # The decorator handle_request_exception catches the RequestException and returns False.
        assert not _test_liveness_probe(expected_code=200)
        assert not _test_readiness_probe(expected_code=200)
        assert not _test_is_loaded(expected_code=200)
        assert not _test_ping(expected_code=200)
        assert not _test_invocations(expected_code=200)


@pytest.mark.integration
def test_concurrency_truss(test_data_path):
    # Tests that concurrency limits work correctly
    with ensure_kill_all():
        truss_dir = test_data_path / "test_concurrency_truss"
        tr = TrussHandle(truss_dir)
        container, urls = tr.docker_run_for_test()

        # Each request takes 2 seconds, for this thread, we allow
        # a concurrency of 2. This means the first two requests will
        # succeed within the 2 seconds, and the third will fail, since
        # it cannot start until the first two have completed.
        def make_request():
            requests.post(urls.predict_url, json={}, timeout=3)

        successful_thread_1 = _PropagatingThread(target=make_request)
        successful_thread_2 = _PropagatingThread(target=make_request)
        failed_thread = _PropagatingThread(target=make_request)

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
def test_requirements_file_truss(test_data_path):
    with ensure_kill_all():
        truss_dir = test_data_path / "test_requirements_file_truss"
        tr = TrussHandle(truss_dir)
        container, urls = tr.docker_run_for_test()
        time.sleep(3)  # Sleeping to allow the load to finish

        # The prediction imports torch which is specified in a requirements.txt and returns if GPU is available.
        response = requests.post(urls.predict_url, json={})
        assert response.status_code == 200
        assert response.json() is False


@pytest.mark.integration
@pytest.mark.parametrize("pydantic_major_version", ["1", "2"])
def test_requirements_pydantic(test_data_path, pydantic_major_version):
    with ensure_kill_all():
        truss_dir = test_data_path / f"test_pyantic_v{pydantic_major_version}"
        tr = TrussHandle(truss_dir)
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={})
        assert response.status_code == 200
        assert response.json() == '{\n    "foo": "bla",\n    "bar": 123\n}'


@pytest.mark.integration
def test_async_truss(test_data_path):
    with ensure_kill_all():
        truss_dir = test_data_path / "test_async_truss"
        tr = TrussHandle(truss_dir)
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={})
        assert response.json() == {
            "preprocess_value": "value",
            "postprocess_value": "value",
        }


@pytest.mark.integration
def test_async_streaming(test_data_path):
    with ensure_kill_all():
        truss_dir = test_data_path / "test_streaming_async_generator_truss"
        tr = TrussHandle(truss_dir)
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={}, stream=True)
        assert response.headers.get("transfer-encoding") == "chunked"
        assert [
            byte_string.decode() for byte_string in list(response.iter_content())
        ] == ["0", "1", "2", "3", "4"]

        predict_non_stream_response = requests.post(
            urls.predict_url,
            json={},
            stream=True,
            headers={"accept": "application/json"},
        )
        assert "transfer-encoding" not in predict_non_stream_response.headers
        assert predict_non_stream_response.json() == "01234"


@pytest.mark.integration
def test_async_streaming_timeout(test_data_path):
    with ensure_kill_all():
        truss_dir = test_data_path / "test_streaming_read_timeout"
        tr = TrussHandle(truss_dir)
        container, urls = tr.docker_run_for_test()
        # ChunkedEncodingError is raised when the chunk does not get processed due to streaming read timeout
        with pytest.raises(requests.exceptions.ChunkedEncodingError):
            response = requests.post(urls.predict_url, json={}, stream=True)

            for chunk in response.iter_content():
                pass

        # Check to ensure the Timeout error is in the container logs
        # TODO: maybe intercept this error better?
        _assert_logs_contain_error(
            container.logs(),
            error="raise exceptions.TimeoutError()",
            message="Exception in ASGI application\n",
        )


@pytest.mark.integration
def test_streaming_with_error_and_stacktrace(test_data_path):
    with ensure_kill_all():
        truss_dir = test_data_path / "test_streaming_truss_with_error"
        tr = TrussHandle(truss_dir)
        container, urls = tr.docker_run_for_test()

        predict_error_response = requests.post(
            urls.predict_url, json={"throw_error": True}, stream=True, timeout=2
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
            urls.predict_url, json={"throw_error": False}, stream=True, timeout=2
        )

        assert [
            byte_string.decode()
            for byte_string in predict_non_error_response.iter_content()
        ] == ["0", "1", "2", "3", "4"]
        expected_stack_trace = (
            "Traceback (most recent call last):\n"
            '  File "/app/model/model.py", line 12, in inner\n'
            "    helpers_1.foo(123)\n"
            '  File "/packages/helpers_1.py", line 5, in foo\n'
            "    return helpers_2.bar(x)\n"
            '  File "/packages/helpers_2.py", line 2, in bar\n'
            '    raise Exception("Crashed in `bar`.")\n'
            "Exception: Crashed in `bar`."
        )
        _assert_logs_contain_error(
            container.logs(),
            error=expected_stack_trace,
            message="Exception while generating streamed response: Crashed in `bar`.",
        )


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
  * Secret 'secret' is defined in the 'secrets' section of the Truss config file"""

    with ensure_kill_all(), _temp_truss(inspect.getsource(Model), config) as tr:
        LocalConfigHandler.set_secret("secret", "secret_value")
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={})

        assert response.json() == "secret_value"

    # Case where the secret is not specified in the config
    with (
        ensure_kill_all(),
        _temp_truss(inspect.getsource(Model), config_with_no_secret) as tr,
    ):
        LocalConfigHandler.set_secret("secret", "secret_value")
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={})
        assert "error" in response.json()
        _assert_logs_contain_error(container.logs(), missing_secret_error_message)
        assert "Internal Server Error" in response.json()["error"]
        assert response.headers["x-baseten-error-source"] == "04"
        assert response.headers["x-baseten-error-code"] == "600"

    # Case where the secret is not mounted
    with ensure_kill_all(), _temp_truss(inspect.getsource(Model), config) as tr:
        LocalConfigHandler.remove_secret("secret")
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={})
        assert response.status_code == 500
        _assert_logs_contain_error(container.logs(), missing_secret_error_message)
        assert "Internal Server Error" in response.json()["error"]
        assert response.headers["x-baseten-error-source"] == "04"
        assert response.headers["x-baseten-error-code"] == "600"


@pytest.mark.integration
def test_postprocess_with_streaming_predict():
    # TODO: revisit the decision to forbid this. If so remove below comment.
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

    with ensure_kill_all(), _temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={}, stream=True)
        logging.info(response.content)
        _assert_logs_contain_error(
            container.logs(),
            "ModelDefinitionError: If the predict function returns a generator (streaming), you cannot use postprocessing.",
        )
        assert "Internal Server Error" in response.json()["error"]
        assert response.headers["x-baseten-error-source"] == "04"
        assert response.headers["x-baseten-error-code"] == "600"


@pytest.mark.integration
def test_streaming_postprocess():
    """
    Tests a Truss where predict returns non-streaming, but postprocess is streamed, and
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

    with ensure_kill_all(), _temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test()

        def make_request(delay: int):
            # For streamed responses, requests does not start receiving content from server until
            # `iter_content` is called, so we must call this in order to get an actual timeout.
            time.sleep(delay)
            response = requests.post(urls.predict_url, json={}, stream=True)

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

    with ensure_kill_all(), _temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test()

        def make_request(delay: int):
            time.sleep(delay)
            response = requests.post(urls.predict_url, json={})
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

    with ensure_kill_all(), _temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={})
        assert response.status_code == 500
        assert "error" in response.json()

        _assert_logs_contain_error(container.logs(), "ValueError: error")

        assert "Internal Server Error" in response.json()["error"]
        assert response.headers["x-baseten-error-source"] == "04"
        assert response.headers["x-baseten-error-code"] == "600"

    model_preprocess_error = """
    class Model:
        def preprocess(self, request):
            raise ValueError("error")

        def predict(self, request):
            return {"a": "b"}
    """

    with ensure_kill_all(), _temp_truss(model_preprocess_error) as tr:
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={})
        assert response.status_code == 500
        assert "error" in response.json()

        _assert_logs_contain_error(container.logs(), "ValueError: error")
        assert "Internal Server Error" in response.json()["error"]
        assert response.headers["x-baseten-error-source"] == "04"
        assert response.headers["x-baseten-error-code"] == "600"

    model_postprocess_error = """
    class Model:
        def predict(self, request):
            return {"a": "b"}

        def postprocess(self, response):
            raise ValueError("error")
    """

    with ensure_kill_all(), _temp_truss(model_postprocess_error) as tr:
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={})
        assert response.status_code == 500
        assert "error" in response.json()
        _assert_logs_contain_error(container.logs(), "ValueError: error")
        assert "Internal Server Error" in response.json()["error"]
        assert response.headers["x-baseten-error-source"] == "04"
        assert response.headers["x-baseten-error-code"] == "600"

    model_async = """
    class Model:
        async def predict(self, request):
            raise ValueError("error")
    """

    with ensure_kill_all(), _temp_truss(model_async) as tr:
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={})
        assert response.status_code == 500
        assert "error" in response.json()

        _assert_logs_contain_error(container.logs(), "ValueError: error")

        assert "Internal Server Error" in response.json()["error"]
        assert response.headers["x-baseten-error-source"] == "04"
        assert response.headers["x-baseten-error-code"] == "600"


@pytest.mark.integration
def test_truss_with_user_errors():
    """Test that user-code raised `fastapi.HTTPExceptions` are passed through as is."""
    model = """
    import fastapi

    class Model:
        def predict(self, request):
            raise fastapi.HTTPException(status_code=500, detail="My custom message.")
    """

    with ensure_kill_all(), _temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={})
        assert response.status_code == 500
        assert "error" in response.json()
        assert response.headers["x-baseten-error-source"] == "04"
        assert response.headers["x-baseten-error-code"] == "600"

        _assert_logs_contain_error(
            container.logs(),
            "HTTPException: 500: My custom message.",
            "Model raised HTTPException",
        )

        assert "My custom message." in response.json()["error"]
        assert response.headers["x-baseten-error-source"] == "04"
        assert response.headers["x-baseten-error-code"] == "600"


@pytest.mark.integration
def test_truss_with_error_stacktrace(test_data_path):
    with ensure_kill_all():
        truss_dir = test_data_path / "test_truss_with_error"
        tr = TrussHandle(truss_dir)
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={})
        assert response.status_code == 500
        assert "error" in response.json()

        assert "Internal Server Error" in response.json()["error"]
        assert response.headers["x-baseten-error-source"] == "04"
        assert response.headers["x-baseten-error-code"] == "600"

        expected_stack_trace = (
            "Traceback (most recent call last):\n"
            '  File "/app/model/model.py", line 8, in predict\n'
            "    return helpers_1.foo(123)\n"
            '  File "/packages/helpers_1.py", line 5, in foo\n'
            "    return helpers_2.bar(x)\n"
            '  File "/packages/helpers_2.py", line 2, in bar\n'
            '    raise Exception("Crashed in `bar`.")\n'
            "Exception: Crashed in `bar`."
        )
        _assert_logs_contain_error(
            container.logs(),
            error=expected_stack_trace,
            message="Internal Server Error",
        )


@pytest.mark.integration
def test_slow_truss(test_data_path):
    with ensure_kill_all():
        truss_dir = test_data_path / "server_conformance_test_truss"
        tr = TrussHandle(truss_dir)

        _, urls = tr.docker_run_for_test(wait_for_server_ready=False)
        truss_server_addr = urls.base_url

        def _test_liveness_probe(expected_code):
            live = requests.get(f"{truss_server_addr}/")
            assert live.status_code == expected_code

        def _test_readiness_probe(expected_code):
            ready = requests.get(f"{truss_server_addr}/v1/models/model")
            assert ready.status_code == expected_code

        def _test_is_loaded(expected_code):
            ready = requests.get(f"{truss_server_addr}/v1/models/model/loaded")
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
            _test_is_loaded(503)
            _test_ping(503)
            _test_invocations(503)
            time.sleep(1)

        time.sleep(LOAD_BUFFER_TIME)
        _test_liveness_probe(200)
        _test_readiness_probe(200)
        _test_is_loaded(200)
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
            _test_is_loaded(200)
            _test_ping(200)
            time.sleep(1)

        predict_call.join()

        _test_invocations(200)


@pytest.mark.integration
def test_init_environment_parameter():
    # Test a truss deployment that is associated with an environment
    model = """
    from typing import Optional
    class Model:
        def __init__(self, **kwargs):
            self._config = kwargs["config"]
            self._environment = kwargs["environment"]
            self.environment_name = self._environment.get("name") if self._environment else None

        def load(self):
            print(f"Executing model.load with environment: {self.environment_name}")

        def predict(self, model_input):
            return self.environment_name
    """
    config = "model_name: init-environment-truss"
    with ensure_kill_all(), _temp_truss(model, config) as tr:
        # Mimic environment changing to staging
        staging_env = {"name": "staging"}
        staging_env_str = json.dumps(staging_env)
        LocalConfigHandler.set_dynamic_config("environment", staging_env_str)
        container, urls = tr.docker_run_for_test()
        assert "Executing model.load with environment: staging" in container.logs()
        response = requests.post(urls.predict_url, json={})
        assert response.json() == "staging"
        assert response.status_code == 200
        container.execute(["bash", "-c", "rm -f /etc/b10_dynamic_config/environment"])

    # Test a truss deployment with no associated environment
    config = "model_name: init-no-environment-truss"
    with ensure_kill_all(), _temp_truss(model, config) as tr:
        container, urls = tr.docker_run_for_test()
        assert "Executing model.load with environment: None" in container.logs()
        response = requests.post(urls.predict_url, json={})
        assert response.json() is None
        assert response.status_code == 200


@pytest.mark.integration
def test_setup_environment():
    # Test truss that uses setup_environment() without load()
    model = """
    from typing import Optional
    class Model:
        def setup_environment(self, environment: Optional[dict]):
            print("setup_environment called with", environment)
            self.environment_name = environment.get("name") if environment else None
            print(f"in {self.environment_name} environment")

        def predict(self, model_input):
            return model_input
    """
    with ensure_kill_all(), _temp_truss(model, "") as tr:
        container, urls = tr.docker_run_for_test()
        # Mimic environment changing to beta
        beta_env = {"name": "beta"}
        beta_env_str = json.dumps(beta_env)
        container.execute(
            [
                "bash",
                "-c",
                f"echo '{beta_env_str}' > /etc/b10_dynamic_config/environment",
            ]
        )
        time.sleep(30)
        assert (
            f"Executing model.setup_environment with environment: {beta_env}"
            in container.logs()
        )
        single_quote_beta_env_str = beta_env_str.replace('"', "'")
        assert (
            f"setup_environment called with {single_quote_beta_env_str}"
            in container.logs()
        )
        assert "in beta environment" in container.logs()
        container.execute(["bash", "-c", "rm -f /etc/b10_dynamic_config/environment"])

    # Test a truss that uses the environment in load()
    model = """
    from typing import Optional
    class Model:
        def setup_environment(self, environment: Optional[dict]):
            print("setup_environment called with", environment)
            self.environment_name = environment.get("name") if environment else None
            print(f"in {self.environment_name} environment")

        def load(self):
            print("loading in environment", self.environment_name)

        def predict(self, model_input):
            return model_input
    """
    with ensure_kill_all(), _temp_truss(model, "") as tr:
        # Mimic environment changing to staging
        staging_env = {"name": "staging"}
        staging_env_str = json.dumps(staging_env)
        LocalConfigHandler.set_dynamic_config("environment", staging_env_str)
        container, urls = tr.docker_run_for_test()
        # Don't need to wait here because we explicitly grab the environment from dynamic_config_resolver before calling user's load()
        assert (
            f"Executing model.setup_environment with environment: {staging_env}"
            in container.logs()
        )
        single_quote_staging_env_str = staging_env_str.replace('"', "'")
        assert (
            f"setup_environment called with {single_quote_staging_env_str}"
            in container.logs()
        )
        assert "in staging environment" in container.logs()
        assert "loading in environment staging" in container.logs()
        # Set environment to None
        no_env = None
        no_env_str = json.dumps(no_env)
        container.execute(
            ["bash", "-c", f"echo '{no_env_str}' > /etc/b10_dynamic_config/environment"]
        )
        time.sleep(30)
        assert (
            f"Executing model.setup_environment with environment: {no_env}"
            in container.logs()
        )
        assert "setup_environment called with None" in container.logs()
        container.execute(["bash", "-c", "rm -f /etc/b10_dynamic_config/environment"])


@pytest.mark.integration
def test_health_check_configuration():
    model = """
    class Model:
        def predict(self, model_input):
            return model_input
    """

    config = """runtime:
    health_checks:
        restart_check_delay_seconds: 100
        restart_threshold_seconds: 1700
    """

    with ensure_kill_all(), _temp_truss(model, config) as tr:
        container, urls = tr.docker_run_for_test()

        assert tr.spec.config.runtime.health_checks.restart_check_delay_seconds == 100
        assert tr.spec.config.runtime.health_checks.restart_threshold_seconds == 1700
        assert (
            tr.spec.config.runtime.health_checks.stop_traffic_threshold_seconds is None
        )

    config = """runtime:
    health_checks:
        restart_check_delay_seconds: 1200
        restart_threshold_seconds: 90
        stop_traffic_threshold_seconds: 50
    """

    with ensure_kill_all(), _temp_truss(model, config) as tr:
        container, urls = tr.docker_run_for_test()

        assert tr.spec.config.runtime.health_checks.restart_check_delay_seconds == 1200
        assert tr.spec.config.runtime.health_checks.restart_threshold_seconds == 90
        assert tr.spec.config.runtime.health_checks.stop_traffic_threshold_seconds == 50

    with ensure_kill_all(), _temp_truss(model, "") as tr:
        container, urls = tr.docker_run_for_test()

        assert tr.spec.config.runtime.health_checks.restart_check_delay_seconds is None
        assert tr.spec.config.runtime.health_checks.restart_threshold_seconds is None
        assert (
            tr.spec.config.runtime.health_checks.stop_traffic_threshold_seconds is None
        )


@pytest.mark.integration
def test_is_healthy():
    model = """
    class Model:
        def load(self):
            raise Exception("not loaded")

        def is_healthy(self) -> bool:
            return True

        def predict(self, model_input):
            return model_input
    """
    with ensure_kill_all(), _temp_truss(model, "") as tr:
        container, urls = tr.docker_run_for_test(wait_for_server_ready=False)
        for _ in range(5):
            time.sleep(1)
            healthy = requests.get(f"{urls.base_url}/v1/models/model")
            if healthy.status_code == 503:
                break
            assert healthy.status_code == 200
        assert healthy.status_code == 503
        diff = container.diff()
        assert "/root/inference_server_crashed.txt" in diff
        assert diff["/root/inference_server_crashed.txt"] == "A"

    model = """
    class Model:
        def is_healthy(self, argument) -> bool:
            pass

        def predict(self, model_input):
            return model_input
    """
    with ensure_kill_all(), _temp_truss(model, "") as tr:
        container, urls = tr.docker_run_for_test(wait_for_server_ready=False)
        time.sleep(1)
        _assert_logs_contain_error(
            container.logs(),
            message="Exception while loading model",
            error="`is_healthy` must have only one argument: `self`",
        )

    model = """
    class Model:
        def is_healthy(self) -> bool:
            raise Exception("not healthy")

        def predict(self, model_input):
            return model_input
    """
    with ensure_kill_all(), _temp_truss(model, "") as tr:
        container, urls = tr.docker_run_for_test(wait_for_server_ready=False)

        # Sleep a few seconds to get the server some time to wake up
        time.sleep(10)
        healthy = requests.get(f"{urls.base_url}/v1/models/model")
        assert healthy.status_code == 503
        assert (
            "Exception while checking if model is healthy: not healthy"
            in container.logs()
        )
        assert "Health check failed." in container.logs()

    model = """
    import time

    class Model:
        def load(self):
            time.sleep(10)

        def is_healthy(self) -> bool:
            return False

        def predict(self, model_input):
            return model_input
    """
    with ensure_kill_all(), _temp_truss(model, "") as tr:
        container, urls = tr.docker_run_for_test(wait_for_server_ready=False)
        time.sleep(5)
        healthy = requests.get(f"{urls.base_url}/v1/models/model")
        assert healthy.status_code == 503
        # Ensure we only log after model.load is complete
        assert "Health check failed." not in container.logs()

        # Sleep a few seconds to get the server some time to wake up
        time.sleep(10)

        healthy = requests.get(f"{urls.base_url}/v1/models/model")
        assert healthy.status_code == 503
        assert container.logs().count("Health check failed.") == 1
        healthy = requests.get(f"{urls.base_url}/v1/models/model")
        assert healthy.status_code == 503
        assert container.logs().count("Health check failed.") == 2

    model = """
    import time

    class Model:
        def __init__(self, **kwargs):
            self._healthy = False

        def load(self):
            time.sleep(10)
            self._healthy = True

        def is_healthy(self):
            return self._healthy

        def predict(self, model_input):
            self._healthy = model_input["healthy"]
            return model_input
    """
    with ensure_kill_all(), _temp_truss(model, "") as tr:
        container, urls = tr.docker_run_for_test(wait_for_server_ready=False)
        time.sleep(5)
        healthy = requests.get(f"{urls.base_url}/v1/models/model")
        assert healthy.status_code == 503
        time.sleep(10)
        healthy = requests.get(f"{urls.base_url}/v1/models/model")
        assert healthy.status_code == 200

        healthy_responses = [True, "yessss", 34, {"woo": "hoo"}]
        for response in healthy_responses:
            predict_response = requests.post(
                urls.predict_url, json={"healthy": response}
            )
            assert predict_response.status_code == 200
            healthy = requests.get(f"{urls.base_url}/v1/models/model")
            assert healthy.status_code == 200

        not_healthy_responses = [False, "", 0, {}]
        for response in not_healthy_responses:
            predict_response = requests.post(
                urls.predict_url, json={"healthy": response}
            )
            assert predict_response.status_code == 200
            healthy = requests.get(f"{urls.base_url}/v1/models/model")
            assert healthy.status_code == 503

    model = """
    class Model:
        def is_healthy(self) -> bool:
            return True

        def predict(self, model_input):
            return model_input
    """
    with ensure_kill_all(), _temp_truss(model, "") as tr:
        container, urls = tr.docker_run_for_test()
        healthy = requests.get(f"{urls.base_url}/v1/models/model")
        assert healthy.status_code == 200


@pytest.mark.integration
def test_instrument_metrics():
    model = """
    from prometheus_client import Counter
    class Model:
        def __init__(self):
            self.counter = Counter('my_really_cool_metric', 'my really cool metric description')
        def predict(self, model_input):
            self.counter.inc(10)
            return model_input
    """
    with ensure_kill_all(), _temp_truss(model, "") as tr:
        container, urls = tr.docker_run_for_test()
        requests.post(urls.predict_url, json={})
        resp = requests.get(urls.metrics_url)
        assert resp.status_code == 200
        metric_names = [
            family.name for family in text_string_to_metric_families(resp.text)
        ]
        assert metric_names == ["my_really_cool_metric"]
        assert "my_really_cool_metric_total 10.0" in resp.text
        assert "/metrics" not in container.logs()

    # Test otel metrics
    model = """
    from opentelemetry import metrics
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.metrics import MeterProvider
    class Model:
        def __init__(self):
            meter_provider = MeterProvider(metric_readers=[PrometheusMetricReader()])
            metrics.set_meter_provider(meter_provider)
            meter = metrics.get_meter(__name__)
            self.counter = meter.create_counter('my_really_cool_metric', description='my really cool metric description')
        def predict(self, model_input):
            self.counter.add(10)
            return model_input
    """
    config = """
    requirements:
    - opentelemetry-exporter-prometheus>=0.52b0
    """
    with ensure_kill_all(), _temp_truss(model, config) as tr:
        _, urls = tr.docker_run_for_test()
        requests.post(urls.predict_url, json={})
        resp = requests.get(urls.metrics_url)
        assert resp.status_code == 200
        metric_names = {
            family.name for family in text_string_to_metric_families(resp.text)
        }
        expected_metric_names = {"target_info", "my_really_cool_metric"}
        assert metric_names == expected_metric_names
        assert "my_really_cool_metric_total 10.0" in resp.text
        assert "/metrics" not in container.logs()


def _patch_termination_timeout(container: Container, seconds: int, truss_container_fs):
    app_path = truss_container_fs / "app"
    sys.path.append(str(app_path))
    import truss_server

    local_server_source = pathlib.Path(truss_server.__file__)
    container_server_source = "/app/truss_server.py"
    modified_content = local_server_source.read_text().replace(
        "TIMEOUT_GRACEFUL_SHUTDOWN = 120", f"TIMEOUT_GRACEFUL_SHUTDOWN = {seconds}"
    )
    with tempfile.NamedTemporaryFile() as patched_file:
        patched_file.write(modified_content.encode("utf-8"))
        patched_file.flush()
        container.copy_to(patched_file.name, container_server_source)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_graceful_shutdown(truss_container_fs):
    model = """
    import time
    class Model:
        def predict(self, request):
            print(f"Received {request}")
            time.sleep(request["seconds"])
            print(f"Done {request}")
            return request
    """
    with ensure_kill_all(), _temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test()

        async def predict_request(data: dict):
            async with httpx.AsyncClient() as client:
                response = await client.post(urls.predict_url, json=data)
                response.raise_for_status()
                return response.json()

        await predict_request({"seconds": 0, "task": 0})  # Warm up server.

        # Test starting two requests, each taking 2 seconds, then terminating server.
        # They should both finish successfully since the server grace period is 120 s.
        task_0 = asyncio.create_task(predict_request({"seconds": 2, "task": 0}))
        await asyncio.sleep(0.1)  # Yield to event loop to make above task run.
        task_1 = asyncio.create_task(predict_request({"seconds": 2, "task": 1}))
        await asyncio.sleep(0.1)  # Yield to event loop to make above task run.

        t0 = time.perf_counter()
        # Even though the server has 120s grace period, we expect to finish much
        # faster in the test here, so use 10s.
        container.stop(10)
        stop_time = time.perf_counter() - t0
        print(f"Stopped in {stop_time} seconds,")

        assert 3 < stop_time < 5
        assert (await task_0) == {"seconds": 2, "task": 0}
        assert (await task_1) == {"seconds": 2, "task": 1}

        # Now mess around in the docker container to reduce the grace period to 3 s.
        # (There's not nice way to patch this...)
        _patch_termination_timeout(container, 3, truss_container_fs)
        # Now only one request should complete.
        container.restart()
        del predict_request  # The restarted container has a different port.
        new_urls = get_docker_urls(container)
        wait_for_truss(container, True)

        async def new_predict_request(data: dict):
            async with httpx.AsyncClient() as client:
                response = await client.post(new_urls.predict_url, json=data)
                response.raise_for_status()
                return response.json()

        await new_predict_request({"seconds": 0, "task": 0})  # Warm up server.

        task_2 = asyncio.create_task(new_predict_request({"seconds": 2, "task": 2}))
        await asyncio.sleep(0.1)  # Yield to event loop to make above task run.
        task_3 = asyncio.create_task(new_predict_request({"seconds": 2, "task": 3}))
        await asyncio.sleep(0.1)  # Yield to event loop to make above task run.
        t0 = time.perf_counter()
        container.stop(10)
        stop_time = time.perf_counter() - t0
        print(f"Stopped in {stop_time} seconds,")
        assert 3 < stop_time < 5
        assert (await task_2) == {"seconds": 2, "task": 2}
        with pytest.raises(httpx.HTTPStatusError):
            await task_3


# Tracing ##############################################################################


def _make_otel_headers() -> Mapping[str, str]:
    """
    Create and return a mapping with OpenTelemetry trace context headers.

    This function starts a new span and injects the trace context into the headers,
    which can be used to propagate tracing information in outgoing HTTP requests.

    Returns:
        Mapping[str, str]: A mapping containing the trace context headers.
    """
    # Initialize a tracer
    tracer = trace.get_tracer(__name__)

    # Create a dictionary to hold the headers
    headers: dict[str, str] = {}

    # Start a new span
    with tracer.start_as_current_span("outgoing-request-span"):
        # Use the TraceContextTextMapPropagator to inject the trace context into the headers
        propagator = tracecontext.TraceContextTextMapPropagator()
        propagator.inject(headers, context=context.get_current())

    return headers


@pytest.mark.integration
@pytest.mark.parametrize("enable_tracing_data", [True, False])
def test_streaming_truss_with_user_tracing(test_data_path, enable_tracing_data):
    with ensure_kill_all():
        truss_dir = test_data_path / "test_streaming_truss_with_tracing"
        tr = TrussHandle(truss_dir)
        tr._update_config(
            runtime=tr._spec.config.runtime.model_copy(
                update={"enable_tracing_data": enable_tracing_data}
            )
        )

        container, urls = tr.docker_run_for_test()

        # A request for which response is not completely read
        headers_0 = _make_otel_headers()
        predict_response = requests.post(
            urls.predict_url, json={}, stream=True, headers=headers_0
        )
        # We just read the first part and leave it hanging here
        next(predict_response.iter_content())

        headers_1 = _make_otel_headers()
        predict_response = requests.post(
            urls.predict_url, json={}, stream=True, headers=headers_1
        )
        assert predict_response.headers.get("transfer-encoding") == "chunked"

        # When accept is set to application/json, the response is not streamed.
        headers_2 = _make_otel_headers()
        predict_non_stream_response = requests.post(
            urls.predict_url,
            json={},
            stream=True,
            headers={**headers_2, "accept": "application/json"},
        )
        assert "transfer-encoding" not in predict_non_stream_response.headers
        assert predict_non_stream_response.json() == "01234"

        with tempfile.TemporaryDirectory() as tmp_dir:
            truss_traces_file = pathlib.Path(tmp_dir) / "otel_traces.ndjson"
            container.copy_from("/tmp/otel_traces.ndjson", truss_traces_file)
            truss_traces = [
                json.loads(s) for s in truss_traces_file.read_text().splitlines()
            ]

            user_traces_file = pathlib.Path(tmp_dir) / "otel_user_traces.ndjson"
            container.copy_from("/tmp/otel_user_traces.ndjson", user_traces_file)
            user_traces = [
                json.loads(s) for s in user_traces_file.read_text().splitlines()
            ]

        if not enable_tracing_data:
            assert len(truss_traces) == 0
            assert len(user_traces) > 0
            return

        print("***")
        print(truss_traces)
        print("***")
        assert sum(1 for x in truss_traces if x["name"] == "predict-endpoint") == 3
        assert sum(1 for x in user_traces if x["name"] == "load_model") == 1
        assert sum(1 for x in user_traces if x["name"] == "predict") == 3

        user_parents = set(x["parent_id"] for x in user_traces)
        truss_spans = set(x["context"]["span_id"] for x in truss_traces)
        truss_parents = set(x["parent_id"] for x in truss_traces)
        # Make sure there is no context creep into user traces. No user trace should
        # have a truss trace as parent.
        assert user_parents & truss_spans == set()
        # But make sure traces have parents at all.
        assert len(user_parents) > 3
        assert len(truss_parents) > 3


# Returning Response Objects ###########################################################


@pytest.mark.integration
def test_truss_with_response():
    """Test that user-code can set a custom status code."""
    model = """
    from fastapi.responses import Response

    class Model:
        def predict(self, inputs):
            return Response(status_code=inputs["code"])
    """
    from fastapi import status

    with ensure_kill_all(), _temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test()

        response = requests.post(
            urls.predict_url, json={"code": status.HTTP_204_NO_CONTENT}
        )
        assert response.status_code == 204
        assert "x-baseten-error-source" not in response.headers
        assert "x-baseten-error-code" not in response.headers

        response = requests.post(
            urls.predict_url, json={"code": status.HTTP_500_INTERNAL_SERVER_ERROR}
        )
        assert response.status_code == 500
        assert response.headers["x-baseten-error-source"] == "04"
        assert response.headers["x-baseten-error-code"] == "700"


@pytest.mark.integration
def test_truss_with_streaming_response():
    # TODO: one issue with this is that (unlike our "builtin" streaming), this keeps
    #  the semaphore claimed potentially longer if the client drops.

    model = """from starlette.responses import StreamingResponse
class Model:
    def predict(self, model_input):
        def text_generator():
            for i in range(3):
                yield f"data: {i}\\n\\n"
        return StreamingResponse(text_generator(), media_type="text/event-stream")
    """

    with ensure_kill_all(), _temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test()

        # A request for which response is not completely read.
        predict_response = requests.post(urls.predict_url, json={}, stream=True)
        assert (
            predict_response.headers["Content-Type"]
            == "text/event-stream; charset=utf-8"
        )

        lines = predict_response.text.strip().split("\n")
        assert lines == ["data: 0", "", "data: 1", "", "data: 2"]


# Using Request in Model ###############################################################


@pytest.mark.integration
def test_truss_with_request():
    model = """
    import fastapi
    class Model:
        async def preprocess(self, request: fastapi.Request):
            return await request.json()

        async def predict(self, inputs, request: fastapi.Request):
            inputs["request_size"] = len(await request.body())
            return inputs

        def postprocess(self, inputs):
             return {**inputs, "postprocess": "was here"}
    """
    with ensure_kill_all(), _temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={"test": 123})
        assert response.status_code == 200
        assert response.json() == {
            "test": 123,
            "request_size": 13,
            "postprocess": "was here",
        }


@pytest.mark.integration
def test_truss_with_requests_and_invalid_signatures():
    model = """
    class Model:
        def predict(self, inputs, invalid_arg): ...
    """
    with ensure_kill_all(), _temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test(wait_for_server_ready=False)
        time.sleep(1.5)  # Wait for logs.
        _assert_logs_contain_error(
            container.logs(),
            "`predict` method with two arguments must have request as second argument",
            "Exception while loading model",
        )

    model = """
    import fastapi

    class Model:
        def predict(self, request: fastapi.Request, invalid_arg): ...
    """
    with ensure_kill_all(), _temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test(wait_for_server_ready=False)
        time.sleep(1.5)  # Wait for logs.
        _assert_logs_contain_error(
            container.logs(),
            "`predict` method with two arguments is not allowed to have request as "
            "first argument",
            "Exception while loading model",
        )

    model = """
    import fastapi

    class Model:
        def predict(self, inputs, request: fastapi.Request, something): ...
    """
    with ensure_kill_all(), _temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test(wait_for_server_ready=False)
        time.sleep(1.5)  # Wait for logs.
        _assert_logs_contain_error(
            container.logs(),
            "`predict` method cannot have more than two arguments",
            "Exception while loading model",
        )


@pytest.mark.integration
def test_truss_with_requests_and_invalid_argument_combinations():
    model = """
    import fastapi
    class Model:
        async def preprocess(self, inputs): ...

        def predict(self, request: fastapi.Request): ...
    """
    with ensure_kill_all(), _temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test(wait_for_server_ready=False)
        time.sleep(1.0)  # Wait for logs.
        _assert_logs_contain_error(
            container.logs(),
            "When using `preprocess`, the predict method cannot only have the request argument",
            "Exception while loading model",
        )

    model = """
    import fastapi
    class Model:
        def preprocess(self, inputs): ...

        async def predict(self, inputs, request: fastapi.Request): ...

        def postprocess(self, request: fastapi.Request): ...
    """
    with ensure_kill_all(), _temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test(wait_for_server_ready=False)
        time.sleep(1.0)  # Wait for logs.
        _assert_logs_contain_error(
            container.logs(),
            "The `postprocess` method cannot only have the request argument",
            "Exception while loading model",
        )

    model = """
    import fastapi
    class Model:
        def preprocess(self, inputs): ...
    """
    with ensure_kill_all(), _temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test(wait_for_server_ready=False)
        time.sleep(1.0)  # Wait for logs.
        _assert_logs_contain_error(
            container.logs(),
            "Truss model must have a `predict` or `websocket` method",
            "Exception while loading model",
        )


@pytest.mark.integration
def test_truss_forbid_postprocessing_with_response():
    model = """
    import fastapi, json
    class Model:
        def predict(self, inputs):
            return fastapi.Response(content=json.dumps(inputs), status_code=200)

        def postprocess(self, inputs):
             return inputs
    """
    with ensure_kill_all(), _temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={})
        assert response.status_code == 500
        assert response.headers["x-baseten-error-source"] == "04"
        assert response.headers["x-baseten-error-code"] == "600"
        _assert_logs_contain_error(
            container.logs(),
            "If the predict function returns a response object, you cannot "
            "use postprocessing.",
        )


@pytest.mark.integration
def test_async_streaming_with_cancellation():
    model = """
    import fastapi, asyncio, logging

    class Model:
        async def predict(self, inputs, request: fastapi.Request):
            await asyncio.sleep(1)
            if await request.is_disconnected():
                logging.warning("Cancelled (before gen).")
                return

            for i in range(5):
                await asyncio.sleep(1.0)
                logging.warning(i)
                yield str(i)
                if await request.is_disconnected():
                    logging.warning("Cancelled (during gen).")
                    return
    """
    with ensure_kill_all(), _temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test()
        # For hard cancellation we need to use httpx, requests' timeouts don't work.
        with pytest.raises(httpx.ReadTimeout):
            with httpx.Client(
                timeout=httpx.Timeout(1.0, connect=1.0, read=1.0)
            ) as client:
                response = client.post(urls.predict_url, json={}, timeout=1.0)
                response.raise_for_status()

        time.sleep(2)  # Wait a bit to get all logs.
        assert "Cancelled (during gen)." in container.logs()


@pytest.mark.integration
def test_async_non_streaming_with_cancellation():
    model = """
    import fastapi, asyncio, logging

    class Model:
        async def predict(self, inputs, request: fastapi.Request):
            logging.info("Start sleep")
            await asyncio.sleep(2)
            logging.info("done sleep, check request.")
            if await request.is_disconnected():
                logging.warning("Cancelled (before gen).")
                return
            logging.info("Not cancelled.")
            return "Done"
    """
    with ensure_kill_all(), _temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test()
        # For hard cancellation we need to use httpx, requests' timeouts don't work.
        with pytest.raises(httpx.ReadTimeout):
            with httpx.Client(
                timeout=httpx.Timeout(1.0, connect=1.0, read=1.0)
            ) as client:
                response = client.post(urls.predict_url, json={}, timeout=1.0)
                response.raise_for_status()

        time.sleep(2)  # Wait a bit to get all logs.
        assert "Cancelled (before gen)." in container.logs()


@pytest.mark.integration
def test_limit_concurrency_with_sse():
    # It seems that the "builtin" functionality of the FastAPI server already buffers
    # the generator, so that it doesn't keep hanging around if the client doesn't
    # consume data. `_buffered_response_generator` might be redundant.
    # This can be observed by waiting for a long time in `make_request`: the server will
    # print `Done` for the tasks, while we still wait and hold the unconsumed response.
    # For testing we need to have actually slow generation to keep the server busy.
    model = """
    import asyncio

    class Model:
        async def predict(self, request):
            print(f"Starting {request}")
            for i in range(5):
                await asyncio.sleep(0.1)
                yield str(i)
            print(f"Done {request}")

    """

    config = """runtime:
  predict_concurrency: 2"""

    def make_request(consume_chunks, timeout, task_id):
        t0 = time.time()
        with httpx.Client() as client:
            with client.stream(
                "POST", urls.predict_url, json={"task_id": task_id}
            ) as response:
                assert response.status_code == 200
                if consume_chunks:
                    chunks = [chunk for chunk in response.iter_text()]
                    print(f"consumed chunks ({task_id}): {chunks}")
                    assert len(chunks) > 0
                    t1 = time.time()
                    if t1 - t0 > timeout:
                        raise httpx.ReadTimeout("Timeout")
                    return chunks
                else:
                    print(f"waiting ({task_id})")
                    time.sleep(0.5)  # Hold the connection.
                    print(f"waiting done ({task_id})")

    with ensure_kill_all(), _temp_truss(model, config) as tr:
        container, urls = tr.docker_run_for_test()
        # Processing full request takes 0.5s.
        print("Make warmup request")
        make_request(consume_chunks=True, timeout=0.55, task_id=0)

        with ThreadPoolExecutor() as executor:
            # Start two requests and hold them without consuming all chunks
            # Each takes for 0.5 s. Semaphore should be claimed, with 0 remaining.
            print("Start two tasks.")
            task1 = executor.submit(make_request, False, 0.55, 1)
            task2 = executor.submit(make_request, False, 0.55, 2)
            print("Wait for tasks to start.")
            time.sleep(0.05)
            print("Make a request while server is busy.")
            with pytest.raises(httpx.ReadTimeout):
                make_request(True, timeout=0.55, task_id=3)

            task1.result()
            task2.result()
            print("Task 1 and 2 completed. Server should be free again.")

        result = make_request(True, timeout=0.55, task_id=4)
        print(f"Final chunks: {result}")


@pytest.mark.integration
def test_custom_openai_endpoints():
    """
    Test a Truss that exposes an OpenAI compatible endpoint.
    """
    model = """
    class Model:
        def load(self):
            self._predict_count = 0
            self._completions_count = 0

        async def predict(self, inputs) -> int:
            self._predict_count += inputs["increment"]
            return self._predict_count

        async def completions(self, inputs) -> int:
            self._completions_count += inputs["increment"]
            return self._completions_count
    """
    with ensure_kill_all(), _temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={"increment": 1})
        assert response.status_code == 200
        assert response.json() == 1

        response = requests.post(urls.completions_url, json={"increment": 2})
        assert response.status_code == 200
        assert response.json() == 2

        response = requests.post(urls.chat_completions_url, json={"increment": 3})
        assert response.status_code == 404


@pytest.mark.integration
def test_postprocess_async_generator_streaming():
    """
    Test a Truss that exposes an OpenAI compatible endpoint.
    """
    model = """
    from typing import List, Generator

    class Model:
        async def predict(self, inputs) -> List[str]:
            nums: List[int] = inputs["nums"]
            return nums

        async def postprocess(self, nums: List[str]) -> Generator[str, None, None]:
            for num in nums:
                yield num
    """
    with ensure_kill_all(), _temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test()

        response = requests.post(
            urls.predict_url, json={"nums": ["1", "2"]}, stream=True
        )
        assert response.headers.get("transfer-encoding") == "chunked"
        assert [
            byte_string.decode() for byte_string in list(response.iter_content())
        ] == ["1", "2"]


@pytest.mark.integration
def test_preprocess_async_generator():
    """
    Test a Truss that exposes an OpenAI compatible endpoint.
    """
    model = """
    from typing import List, AsyncGenerator

    class Model:
        async def preprocess(self, inputs) -> AsyncGenerator[str, None]:
            for num in inputs["nums"]:
                yield num

        async def predict(self, nums: AsyncGenerator[str, None]) -> List[str]:
            return [num async for num in nums]
    """
    with ensure_kill_all(), _temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={"nums": ["1", "2"]})
        assert response.status_code == 200
        assert response.json() == ["1", "2"]


@pytest.mark.integration
def test_openai_client_streaming():
    """
    Test a Truss that exposes an OpenAI compatible endpoint.
    """
    model = """
    from typing import AsyncGenerator

    class Model:
        async def chat_completions(self, inputs) -> AsyncGenerator[str, None]:
            for num in inputs["nums"]:
                yield num

        async def predict(self, inputs):
            pass
    """
    with ensure_kill_all(), _temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test()

        response = requests.post(
            urls.chat_completions_url,
            json={"nums": ["1", "2"]},
            stream=True,
            # Despite requesting json, we should still stream results back.
            headers={
                "accept": "application/json",
                "user-agent": "OpenAI/Python 1.61.0",
            },
        )
        assert response.headers.get("transfer-encoding") == "chunked"
        assert [
            byte_string.decode() for byte_string in list(response.iter_content())
        ] == ["1", "2"]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_raise_predict_and_websocket_endpoint():
    model = """
    class Model:
        async def websocket(self, websocket):
            pass

        async def predict(self, inputs):
            pass
    """
    with ensure_kill_all(), _temp_truss(model, "") as tr:
        container, urls = tr.docker_run_for_test(wait_for_server_ready=False)
        time.sleep(3)
        _assert_logs_contain_error(
            container.logs(),
            message="Exception while loading model",
            error="cannot have both",
        )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_raise_preprocess_and_websocket_endpoint():
    model = """
    class Model:
        async def websocket(self, websocket):
            pass

        async def preprocess(self, inputs):
            pass
    """
    with ensure_kill_all(), _temp_truss(model, "") as tr:
        container, urls = tr.docker_run_for_test(wait_for_server_ready=False)
        time.sleep(3)
        _assert_logs_contain_error(
            container.logs(),
            message="Exception while loading model",
            error="cannot have both",
        )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_raise_no_endpoint():
    model = """
    class Model:
       pass
    """
    with ensure_kill_all(), _temp_truss(model, "") as tr:
        container, urls = tr.docker_run_for_test(wait_for_server_ready=False)
        time.sleep(1)
        _assert_logs_contain_error(
            container.logs(),
            message="Exception while loading model",
            error="must have a `predict` or `websocket` method",
        )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_websocket_endpoint():
    model = """
    import fastapi

    class Model:
        async def websocket(self, websocket: fastapi.WebSocket):
            try:
                while True:
                    text = await websocket.receive_text()
                    if text == "done":
                        print("done")
                        return

                    await websocket.send_text(text + " pong")
            except fastapi.WebSocketDisconnect:
                pass
    """
    with ensure_kill_all(), _temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test()
        async with websockets.connect(urls.websockets_url) as websocket:
            # Send "hello" and verify response
            await websocket.send("hello")
            response = await websocket.recv()
            assert response == "hello pong"

            # Send "world" and verify response
            await websocket.send("world")
            response = await websocket.recv()
            assert response == "world pong"

            await websocket.send("done")

            with pytest.raises(websockets.exceptions.ConnectionClosed) as exc_info:
                await websocket.recv()

            assert exc_info.value.rcvd.code == 1000
            assert exc_info.value.rcvd.reason == ""


@pytest.mark.asyncio
@pytest.mark.integration
async def test_websocket_endpoint_error_logs():
    model = """
    import fastapi

    class Model:
        async def websocket(self, websocket: fastapi.WebSocket):
            try:
                while True:
                    text = await websocket.receive_text()
                    if text == "raise":
                        raise ValueError("This is test error.")
                    await websocket.send_text(text + " pong")
            except fastapi.WebSocketDisconnect:
                pass
    """
    with ensure_kill_all(), _temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test()
        async with websockets.connect(urls.websockets_url) as websocket:
            # Send "hello" and verify response
            await websocket.send("hello")
            response = await websocket.recv()
            assert response == "hello pong"

            # Send "raise" to raise a test error.
            await websocket.send("raise")
            with pytest.raises(websockets.exceptions.ConnectionClosedError) as exc_info:
                await websocket.recv()

            assert exc_info.value.rcvd.code == 1011
            assert (
                exc_info.value.rcvd.reason
                == "Internal Server Error (in model/chainlet)."
            )

            expected_stack_trace = (
                "Traceback (most recent call last):\n"
                '  File "/app/model/model.py", line 10, in websocket\n'
                '    raise ValueError("This is test error.")\n'
                "ValueError: This is test error."
            )
            _assert_logs_contain_error(
                container.logs(),
                error=expected_stack_trace,
                message="Internal Server Error (in model/chainlet).",
            )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_nonexistent_websocket_endpoint():
    model = """
    class Model:

        async def predict(self, inputs):
            pass
    """
    with ensure_kill_all(), _temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test()
        with pytest.raises(websockets.ConnectionClosedError) as exc_info:
            async with websockets.connect(urls.websockets_url) as ws:
                await ws.recv()

        assert exc_info.value.rcvd.code == 1003
        assert (
            exc_info.value.rcvd.reason
            == "WebSocket is not implemented on this deployment."
        )
        _assert_logs_contain_error(
            container.logs(),
            error=None,
            message="WebSocket is not implemented on this deployment.",
        )
