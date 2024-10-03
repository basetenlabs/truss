import concurrent
import contextlib
import dataclasses
import inspect
import json
import logging
import pathlib
import tempfile
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Thread
from typing import Iterator, Mapping

import httpx
import opentelemetry.trace.propagation.tracecontext as tracecontext
import pytest
import requests
from opentelemetry import context, trace
from requests.exceptions import RequestException

from truss.local.local_config_handler import LocalConfigHandler
from truss.model_inference import map_to_supported_python_version
from truss.tests.helpers import create_truss
from truss.tests.test_testing_utilities_for_other_tests import ensure_kill_all
from truss.truss_handle import TrussHandle

logger = logging.getLogger(__name__)

DEFAULT_LOG_ERROR = "Internal Server Error"
PREDICT_URL = "http://localhost:8090/v1/models/model:predict"


def _log_contains_error(line: dict, error: str, message: str):
    return (
        line["levelname"] == "ERROR"
        and line["message"] == message
        and error in line["exc_info"]
    )


def assert_logs_contain_error(logs: str, error: str, message=DEFAULT_LOG_ERROR):
    loglines = [json.loads(line) for line in logs.splitlines()]
    assert any(_log_contains_error(line, error, message) for line in loglines), (
        f"Did not find expected error in logs.\nExpected error: {error}\n"
        f"Expected message: {message}\nActual logs:\n{loglines}"
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


@contextlib.contextmanager
def temp_truss(model_src: str, config_src: str) -> Iterator[TrussHandle]:
    with ensure_kill_all(), tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        truss_dir = Path(tmp_work_dir, "truss")
        create_truss(truss_dir, config_src, textwrap.dedent(model_src))
        yield TrussHandle(truss_dir)


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

        # Each request takes 2 seconds, for this thread, we allow
        # a concurrency of 2. This means the first two requests will
        # succeed within the 2 seconds, and the third will fail, since
        # it cannot start until the first two have completed.
        def make_request():
            requests.post(PREDICT_URL, json={}, timeout=3)

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

        # The prediction imports torch which is specified in a requirements.txt and returns if GPU is available.
        response = requests.post(PREDICT_URL, json={})
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

        response = requests.post(PREDICT_URL, json={})
        assert response.status_code == 200
        assert response.json() == '{\n    "foo": "bla",\n    "bar": 123\n}'


@pytest.mark.integration
def test_async_truss():
    with ensure_kill_all():
        truss_root = Path(__file__).parent.parent.parent.resolve() / "truss"
        truss_dir = truss_root / "test_data" / "test_async_truss"
        tr = TrussHandle(truss_dir)
        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)

        response = requests.post(PREDICT_URL, json={})
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

        response = requests.post(PREDICT_URL, json={}, stream=True)
        assert response.headers.get("transfer-encoding") == "chunked"
        assert [
            byte_string.decode() for byte_string in list(response.iter_content())
        ] == ["0", "1", "2", "3", "4"]

        predict_non_stream_response = requests.post(
            PREDICT_URL,
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

        # ChunkedEncodingError is raised when the chunk does not get processed due to streaming read timeout
        with pytest.raises(requests.exceptions.ChunkedEncodingError):
            response = requests.post(PREDICT_URL, json={}, stream=True)

            for chunk in response.iter_content():
                pass

        # Check to ensure the Timeout error is in the container logs
        # TODO: maybe intercept this error better?
        assert_logs_contain_error(
            container.logs(),
            error="raise exceptions.TimeoutError()",
            message="Exception in ASGI application\n",
        )


@pytest.mark.integration
def test_streaming_with_error_and_stacktrace():
    with ensure_kill_all():
        truss_root = Path(__file__).parent.parent.parent.resolve() / "truss"
        truss_dir = truss_root / "test_data" / "test_streaming_truss_with_error"
        tr = TrussHandle(truss_dir)
        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=True
        )

        predict_error_response = requests.post(
            PREDICT_URL, json={"throw_error": True}, stream=True, timeout=2
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
            PREDICT_URL, json={"throw_error": False}, stream=True, timeout=2
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
        assert_logs_contain_error(
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
  * Secret 'secret' is defined in the 'secrets' section of the Truss config file
  * The model was pushed with the --trusted flag"""

    with ensure_kill_all(), temp_truss(inspect.getsource(Model), config) as tr:
        LocalConfigHandler.set_secret("secret", "secret_value")
        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)

        response = requests.post(PREDICT_URL, json={})

        assert response.json() == "secret_value"

    # Case where the secret is not specified in the config
    with ensure_kill_all(), temp_truss(
        inspect.getsource(Model), config_with_no_secret
    ) as tr:
        LocalConfigHandler.set_secret("secret", "secret_value")
        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=True
        )

        response = requests.post(PREDICT_URL, json={})
        assert "error" in response.json()
        assert_logs_contain_error(container.logs(), missing_secret_error_message)
        assert "Internal Server Error" in response.json()["error"]
        assert response.headers["x-baseten-error-source"] == "04"
        assert response.headers["x-baseten-error-code"] == "600"

    # Case where the secret is not mounted
    with ensure_kill_all(), temp_truss(inspect.getsource(Model), config) as tr:
        LocalConfigHandler.remove_secret("secret")
        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=True
        )

        response = requests.post(PREDICT_URL, json={})
        assert response.status_code == 500
        assert_logs_contain_error(container.logs(), missing_secret_error_message)
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

    config = "model_name: error-truss"
    with ensure_kill_all(), temp_truss(model, config) as tr:
        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=True
        )

        response = requests.post(PREDICT_URL, json={}, stream=True)
        logging.info(response.content)
        assert_logs_contain_error(
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

    config = "model_name: streaming-truss"
    with ensure_kill_all(), temp_truss(model, config) as tr:
        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)

        def make_request(delay: int):
            # For streamed responses, requests does not start receiving content from server until
            # `iter_content` is called, so we must call this in order to get an actual timeout.
            time.sleep(delay)
            response = requests.post(PREDICT_URL, json={}, stream=True)

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

    config = "model_name: postprocess-truss"
    with ensure_kill_all(), temp_truss(model, config) as tr:
        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)

        def make_request(delay: int):
            time.sleep(delay)
            response = requests.post(PREDICT_URL, json={})
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

    with ensure_kill_all(), temp_truss(model, config) as tr:
        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=True
        )

        response = requests.post(PREDICT_URL, json={})
        assert response.status_code == 500
        assert "error" in response.json()

        assert_logs_contain_error(container.logs(), "ValueError: error")

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

    with ensure_kill_all(), temp_truss(model_preprocess_error, config) as tr:
        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=True
        )

        response = requests.post(PREDICT_URL, json={})
        assert response.status_code == 500
        assert "error" in response.json()

        assert_logs_contain_error(container.logs(), "ValueError: error")
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

    with ensure_kill_all(), temp_truss(model_postprocess_error, config) as tr:
        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=True
        )

        response = requests.post(PREDICT_URL, json={})
        assert response.status_code == 500
        assert "error" in response.json()
        assert_logs_contain_error(container.logs(), "ValueError: error")
        assert "Internal Server Error" in response.json()["error"]
        assert response.headers["x-baseten-error-source"] == "04"
        assert response.headers["x-baseten-error-code"] == "600"

    model_async = """
    class Model:
        async def predict(self, request):
            raise ValueError("error")
    """

    with ensure_kill_all(), temp_truss(model_async, config) as tr:
        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=True
        )

        response = requests.post(PREDICT_URL, json={})
        assert response.status_code == 500
        assert "error" in response.json()

        assert_logs_contain_error(container.logs(), "ValueError: error")

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

    config = "model_name: error-truss"

    with ensure_kill_all(), temp_truss(model, config) as tr:
        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=True
        )

        response = requests.post(PREDICT_URL, json={})
        assert response.status_code == 500
        assert "error" in response.json()
        assert response.headers["x-baseten-error-source"] == "04"
        assert response.headers["x-baseten-error-code"] == "600"

        assert_logs_contain_error(
            container.logs(),
            "HTTPException: 500: My custom message.",
            "Model raised HTTPException",
        )

        assert "My custom message." in response.json()["error"]
        assert response.headers["x-baseten-error-source"] == "04"
        assert response.headers["x-baseten-error-code"] == "600"


@pytest.mark.integration
def test_truss_with_error_stacktrace():
    with ensure_kill_all():
        truss_root = Path(__file__).parent.parent.parent.resolve() / "truss"
        truss_dir = truss_root / "test_data" / "test_truss_with_error"
        tr = TrussHandle(truss_dir)
        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=True
        )

        response = requests.post(PREDICT_URL, json={})
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
        assert_logs_contain_error(
            container.logs(),
            error=expected_stack_trace,
            message="Internal Server Error",
        )


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
def test_streaming_truss_with_user_tracing(enable_tracing_data):
    with ensure_kill_all():
        truss_root = Path(__file__).parent.parent.parent.resolve() / "truss"
        truss_dir = truss_root / "test_data" / "test_streaming_truss_with_tracing"
        tr = TrussHandle(truss_dir)

        def enable_gpu_fn(conf):
            new_runtime = dataclasses.replace(
                conf.runtime, enable_tracing_data=enable_tracing_data
            )
            return dataclasses.replace(conf, runtime=new_runtime)

        tr._update_config(enable_gpu_fn)

        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=True
        )

        # A request for which response is not completely read
        headers_0 = _make_otel_headers()
        predict_response = requests.post(
            PREDICT_URL, json={}, stream=True, headers=headers_0
        )
        # We just read the first part and leave it hanging here
        next(predict_response.iter_content())

        headers_1 = _make_otel_headers()
        predict_response = requests.post(
            PREDICT_URL, json={}, stream=True, headers=headers_1
        )
        assert predict_response.headers.get("transfer-encoding") == "chunked"

        # When accept is set to application/json, the response is not streamed.
        headers_2 = _make_otel_headers()
        predict_non_stream_response = requests.post(
            PREDICT_URL,
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

    config = "model_name: custom-status-code-truss"

    with ensure_kill_all(), temp_truss(model, config) as tr:
        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)

        response = requests.post(PREDICT_URL, json={"code": status.HTTP_204_NO_CONTENT})
        assert response.status_code == 204
        assert "x-baseten-error-source" not in response.headers
        assert "x-baseten-error-code" not in response.headers

        response = requests.post(
            PREDICT_URL, json={"code": status.HTTP_500_INTERNAL_SERVER_ERROR}
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

    config = "model_name: sse-truss"

    with ensure_kill_all(), temp_truss(model, config) as tr:
        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)

        # A request for which response is not completely read.
        predict_response = requests.post(PREDICT_URL, json={}, stream=True)
        assert (
            predict_response.headers["Content-Type"]
            == "text/event-stream; charset=utf-8"
        )

        lines = predict_response.text.strip().split("\n")
        assert lines == [
            "data: 0",
            "",
            "data: 1",
            "",
            "data: 2",
        ]


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
    with ensure_kill_all(), temp_truss(model, "") as tr:
        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)

        response = requests.post(PREDICT_URL, json={"test": 123})
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
    with ensure_kill_all(), temp_truss(model, "") as tr:
        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=False
        )
        time.sleep(1.0)  # Wait for logs.
        assert_logs_contain_error(
            container.logs(),
            "`predict` method with two arguments must have request as second argument",
            "Exception while loading model",
        )

    model = """
    import fastapi

    class Model:
        def predict(self, request: fastapi.Request, invalid_arg): ...
    """
    with ensure_kill_all(), temp_truss(model, "") as tr:
        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=False
        )
        time.sleep(1.0)  # Wait for logs.
        assert_logs_contain_error(
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
    with ensure_kill_all(), temp_truss(model, "") as tr:
        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=False
        )
        time.sleep(1.0)  # Wait for logs.
        assert_logs_contain_error(
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
    with ensure_kill_all(), temp_truss(model, "") as tr:
        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=False
        )
        time.sleep(1.0)  # Wait for logs.
        assert_logs_contain_error(
            container.logs(),
            "When using preprocessing, the predict method cannot only have the request argument",
            "Exception while loading model",
        )

    model = """
    import fastapi
    class Model:
        def preprocess(self, inputs): ...

        async def predict(self, inputs, request: fastapi.Request): ...

        def postprocess(self, request: fastapi.Request): ...
    """
    with ensure_kill_all(), temp_truss(model, "") as tr:
        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=False
        )
        time.sleep(1.0)  # Wait for logs.
        assert_logs_contain_error(
            container.logs(),
            "The postprocessing method cannot only have the request argument",
            "Exception while loading model",
        )

    model = """
    import fastapi
    class Model:
        def preprocess(self, inputs): ...
    """
    with ensure_kill_all(), temp_truss(model, "") as tr:
        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=False
        )
        time.sleep(1.0)  # Wait for logs.
        assert_logs_contain_error(
            container.logs(),
            "Truss model must have a `predict` method.",
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
    with ensure_kill_all(), temp_truss(model, "") as tr:
        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=True
        )

        response = requests.post(PREDICT_URL, json={})
        assert response.status_code == 500
        assert response.headers["x-baseten-error-source"] == "04"
        assert response.headers["x-baseten-error-code"] == "600"
        assert_logs_contain_error(
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
    with ensure_kill_all(), temp_truss(model, "") as tr:
        container = tr.docker_run(
            local_port=8090, detach=True, wait_for_server_ready=True
        )
        # For hard cancellation we need to use httpx, requests' timeouts don't work.
        with pytest.raises(httpx.ReadTimeout):
            with httpx.Client(
                timeout=httpx.Timeout(1.0, connect=1.0, read=1.0)
            ) as client:
                response = client.post(PREDICT_URL, json={}, timeout=1.0)
                response.raise_for_status()

        time.sleep(2)  # Wait a bit to get all logs.
        assert "Cancelled (during gen)." in container.logs()


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
                "POST", PREDICT_URL, json={"task_id": task_id}
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

    with ensure_kill_all(), temp_truss(model, config) as tr:
        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)
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
