import inspect
import logging
import time

import pytest
import requests
import websockets

from truss.local.local_config_handler import LocalConfigHandler
from truss.tests.helpers import assert_logs_contain_error, temp_truss
from truss.tests.test_testing_utilities_for_other_tests import ensure_kill_all
from truss.truss_handle.truss_handle import TrussHandle

logger = logging.getLogger(__name__)


@pytest.fixture
def anyio_backend():
    return "asyncio"


### Inference Runtime Errors ###


@pytest.mark.integration
def test_truss_with_errors():
    model = """
    import traceback, inspect

    class Model:
        async def predict(self, request):
            stack_lines = traceback.format_stack()
            print("".join(stack_lines))
            for frame_info in inspect.stack():
                print(f"{frame_info.filename}:{frame_info.lineno} in {frame_info.function}")
            raise ValueError("error")
    """

    with ensure_kill_all(), temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={})
        assert response.status_code == 500
        assert "error" in response.json()

        assert_logs_contain_error(container.logs(), "ValueError: error")

        assert "Internal Server Error" in response.json()["error"]
        assert response.headers["x-baseten-error-source"] == "04"
        assert response.headers["x-baseten-error-code"] == "600"

    model_preprocess_error = """
    class Model:
        async def preprocess(self, request):
            raise ValueError("error")

        async def predict(self, request):
            return {"a": "b"}
    """

    with ensure_kill_all(), temp_truss(model_preprocess_error) as tr:
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={})
        assert response.status_code == 500
        assert "error" in response.json()

        assert_logs_contain_error(container.logs(), "ValueError: error")
        assert "Internal Server Error" in response.json()["error"]
        assert response.headers["x-baseten-error-source"] == "04"
        assert response.headers["x-baseten-error-code"] == "600"

    model_postprocess_error = """
    class Model:
        async def predict(self, request):
            return {"a": "b"}

        async def postprocess(self, response):
            raise ValueError("error")
    """

    with ensure_kill_all(), temp_truss(model_postprocess_error) as tr:
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={})
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

    with ensure_kill_all(), temp_truss(model_async) as tr:
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={})
        assert response.status_code == 500
        assert "error" in response.json()

        assert_logs_contain_error(container.logs(), "ValueError: error")

        assert "Internal Server Error" in response.json()["error"]
        assert response.headers["x-baseten-error-source"] == "04"
        assert response.headers["x-baseten-error-code"] == "600"


@pytest.mark.integration
def test_truss_with_user_http_status():
    """Test that user-code raised `fastapi.HTTPExceptions` are passed through as is."""
    model = """
    import fastapi

    class Model:
        async def predict(self, request):
            raise fastapi.HTTPException(status_code=500, detail="My custom message.")
    """

    with ensure_kill_all(), temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={})
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
def test_truss_with_schema_validation_errors():
    """Test that user-code raised `fastapi.HTTPExceptions` are passed through as is."""
    model = """
    import fastapi
    import pydantic

    class Input(pydantic.BaseModel):
        field: int

    class Output(pydantic.BaseModel):
        field: str

    class Model:
        async def predict(self, data: Input) -> Output:
            return Output(field=str(data.field))
    """

    with ensure_kill_all(), temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={})
        print(response.content)
        # assert response.status_code == 500
        # assert "error" in response.json()
        # assert response.headers["x-baseten-error-source"] == "04"
        # assert response.headers["x-baseten-error-code"] == "600"
        #
        # assert_logs_contain_error(
        #     container.logs(),
        #     "HTTPException: 500: My custom message.",
        #     "Model raised HTTPException",
        # )
        #
        # assert "My custom message." in response.json()["error"]
        # assert response.headers["x-baseten-error-source"] == "04"
        # assert response.headers["x-baseten-error-code"] == "600"


@pytest.mark.integration
def test_truss_with_error_stacktrace(test_data_path):
    with ensure_kill_all():
        # Raises an error from a nested helper function.
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
        assert_logs_contain_error(
            container.logs(),
            error=expected_stack_trace,
            message="Internal Server Error",
        )


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
    with ensure_kill_all(), temp_truss(model) as tr:
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
            assert_logs_contain_error(
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
    with ensure_kill_all(), temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test()
        with pytest.raises(websockets.ConnectionClosedError) as exc_info:
            async with websockets.connect(urls.websockets_url) as ws:
                await ws.recv()

        assert exc_info.value.rcvd.code == 1003
        assert (
            exc_info.value.rcvd.reason
            == "WebSocket is not implemented on this deployment."
        )
        assert_logs_contain_error(
            container.logs(),
            error=None,
            message="WebSocket is not implemented on this deployment.",
        )


### Stream Streaming Errors ###


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
        assert_logs_contain_error(
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
        assert_logs_contain_error(
            container.logs(),
            error=expected_stack_trace,
            message="Exception while generating streamed response: Crashed in `bar`.",
        )


### Model / Method Definition Errors ###


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
        async def postprocess(self, response):
            for item in response:
                time.sleep(1)
                yield item + " modified"

        async def predict(self, request):
            for i in range(2):
                yield str(i)
    """

    with ensure_kill_all(), temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={}, stream=True)
        logging.info(response.content)
        assert_logs_contain_error(
            container.logs(),
            "ModelDefinitionError: If the predict function returns a generator (streaming), you cannot use postprocessing.",
        )
        assert response.status_code == 412  # Precondition Failed.
        assert "Internal Server Error" in response.json()["error"]
        assert response.headers["x-baseten-error-source"] == "04"
        assert response.headers["x-baseten-error-code"] == "600"


@pytest.mark.integration
def test_truss_with_requests_and_invalid_signatures():
    model = """
    class Model:
        async def predict(self, inputs, invalid_arg): ...
    """
    with ensure_kill_all(), temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test(wait_for_server_ready=False)
        time.sleep(1.5)  # Wait for logs.
        assert_logs_contain_error(
            container.logs(),
            "`predict` method with two arguments must have request as second argument",
            "Exception while loading model",
        )

    model = """
    import fastapi

    class Model:
        async def predict(self, request: fastapi.Request, invalid_arg): ...
    """
    with ensure_kill_all(), temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test(wait_for_server_ready=False)
        time.sleep(1.5)  # Wait for logs.
        assert_logs_contain_error(
            container.logs(),
            "`predict` method with two arguments is not allowed to have request as "
            "first argument",
            "Exception while loading model",
        )

    model = """
    import fastapi

    class Model:
        async def predict(self, inputs, request: fastapi.Request, something): ...
    """
    with ensure_kill_all(), temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test(wait_for_server_ready=False)
        time.sleep(1.5)  # Wait for logs.
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

        async def predict(self, request: fastapi.Request): ...
    """
    with ensure_kill_all(), temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test(wait_for_server_ready=False)
        time.sleep(1.0)  # Wait for logs.
        assert_logs_contain_error(
            container.logs(),
            "When using `preprocess`, the predict method cannot only have the request argument",
            "Exception while loading model",
        )

    model = """
    import fastapi
    class Model:
        async def preprocess(self, inputs): ...

        async def predict(self, inputs, request: fastapi.Request): ...

        async def postprocess(self, request: fastapi.Request): ...
    """
    with ensure_kill_all(), temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test(wait_for_server_ready=False)
        time.sleep(1.0)  # Wait for logs.
        assert_logs_contain_error(
            container.logs(),
            "The `postprocess` method cannot only have the request argument",
            "Exception while loading model",
        )

    model = """
    import fastapi
    class Model:
        async def preprocess(self, inputs): ...
    """
    with ensure_kill_all(), temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test(wait_for_server_ready=False)
        time.sleep(1.0)  # Wait for logs.
        assert_logs_contain_error(
            container.logs(),
            "Truss model must have a `predict` or `websocket` method",
            "Exception while loading model",
        )


@pytest.mark.integration
def test_truss_forbid_postprocessing_with_response():
    model = """
    import fastapi, json
    class Model:
        async def predict(self, inputs):
            return fastapi.Response(content=json.dumps(inputs), status_code=200)

        async def postprocess(self, inputs):
             return inputs
    """
    with ensure_kill_all(), temp_truss(model) as tr:
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={})
        assert response.status_code == 500
        assert response.headers["x-baseten-error-source"] == "04"
        assert response.headers["x-baseten-error-code"] == "600"
        assert_logs_contain_error(
            container.logs(),
            "If the predict function returns a response object, you cannot "
            "use postprocessing.",
        )


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
    with ensure_kill_all(), temp_truss(model, "") as tr:
        container, urls = tr.docker_run_for_test(wait_for_server_ready=False)
        time.sleep(3)
        assert_logs_contain_error(
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
    with ensure_kill_all(), temp_truss(model, "") as tr:
        container, urls = tr.docker_run_for_test(wait_for_server_ready=False)
        time.sleep(3)
        assert_logs_contain_error(
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
    with ensure_kill_all(), temp_truss(model, "") as tr:
        container, urls = tr.docker_run_for_test(wait_for_server_ready=False)
        time.sleep(1)
        assert_logs_contain_error(
            container.logs(),
            message="Exception while loading model",
            error="must have a `predict` or `websocket` method",
        )


@pytest.mark.integration
def test_is_healthy_cannot_have_args():
    model = """
    class Model:
        async def is_healthy(self, argument) -> bool:
            pass

        async def predict(self, model_input):
            return model_input
    """
    with ensure_kill_all(), temp_truss(model, "") as tr:
        container, urls = tr.docker_run_for_test(wait_for_server_ready=False)
        time.sleep(1)
        assert_logs_contain_error(
            container.logs(),
            message="Exception while loading model",
            error="`is_healthy` must have only one argument: `self`",
        )


### Other Errors ###


@pytest.mark.integration
def test_secrets_errors():
    class Model:
        def __init__(self, **kwargs):
            self._secrets = kwargs["secrets"]

        async def predict(self, request):
            return self._secrets["secret"]

    config = """model_name: secrets-truss
secrets:
    secret: null
    """

    config_with_no_secret = "model_name: secrets-truss"
    missing_secret_error_message = """Secret 'secret' not found. Please ensure that:
  * Secret 'secret' is defined in the 'secrets' section of the Truss config file"""

    # Case where the secret is not specified in the config
    with (
        ensure_kill_all(),
        temp_truss(inspect.getsource(Model), config_with_no_secret) as tr,
    ):
        LocalConfigHandler.set_secret("secret", "secret_value")
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={})
        assert "error" in response.json()
        assert_logs_contain_error(container.logs(), missing_secret_error_message)
        assert "Internal Server Error" in response.json()["error"]
        assert response.headers["x-baseten-error-source"] == "04"
        assert response.headers["x-baseten-error-code"] == "600"

    # Case where the secret is not mounted
    with ensure_kill_all(), temp_truss(inspect.getsource(Model), config) as tr:
        LocalConfigHandler.remove_secret("secret")
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={})
        assert response.status_code == 500
        assert_logs_contain_error(container.logs(), missing_secret_error_message)
        assert "Internal Server Error" in response.json()["error"]
        assert response.headers["x-baseten-error-source"] == "04"
        assert response.headers["x-baseten-error-code"] == "600"
