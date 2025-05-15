import tempfile
import time
from pathlib import Path

import pytest
import requests

from truss.templates.shared import serialization
from truss.tests.helpers import create_truss
from truss.tests.test_testing_utilities_for_other_tests import ensure_kill_all
from truss.truss_handle.truss_handle import TrussHandle

DEFAULT_CONFIG = """model_name: test-truss"""


@pytest.mark.integration
def test_truss_with_no_annotations(test_data_path):
    truss_dir = test_data_path / "test_basic_truss"

    tr = TrussHandle(truss_dir)

    with ensure_kill_all():
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={"prompt": "value"})
        assert response.json() == {"prompt": "value"}

        schema_response = requests.get(urls.schema_url)
        assert schema_response.status_code == 404

        assert schema_response.json()["error"] == "No schema found"
        assert schema_response.headers["x-baseten-error-source"] == "04"
        assert schema_response.headers["x-baseten-error-code"] == "600"


@pytest.mark.integration
def test_truss_with_non_pydantic_annotations():
    truss_non_pydantic_annotations = """
class Model:
    def predict(self, request: str) -> list[str]:
        return ["hello"]
"""

    with ensure_kill_all(), tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        truss_dir = Path(tmp_work_dir, "truss")

        create_truss(truss_dir, DEFAULT_CONFIG, truss_non_pydantic_annotations)

        tr = TrussHandle(truss_dir)
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={"prompt": "value"})
        assert response.json() == ["hello"]

        schema_response = requests.get(urls.schema_url)
        assert schema_response.status_code == 404
        assert schema_response.json()["error"] == "No schema found"
        assert schema_response.headers["x-baseten-error-source"] == "04"
        assert schema_response.headers["x-baseten-error-code"] == "600"


@pytest.mark.integration
def test_truss_with_no_annotations_failed_load():
    truss_long_load = """
class Model:
    def load(self):
        raise Exception("Failed load")

    def predict(self, request):
        return "hello"
"""

    with ensure_kill_all(), tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        truss_dir = Path(tmp_work_dir, "truss")

        create_truss(truss_dir, DEFAULT_CONFIG, truss_long_load)

        tr = TrussHandle(truss_dir)
        container, urls = tr.docker_run_for_test(wait_for_server_ready=False)

        # Wait a bit for the server to start
        time.sleep(2)

        schema_response = requests.get(urls.schema_url)

        # If the load has not successfully completed,
        # we return a 503 instead of 404, so that clients
        # know that this can be retried.
        assert schema_response.status_code == 503


@pytest.mark.integration
def test_truss_with_annotated_inputs_outputs(test_data_path):
    truss_dir = test_data_path / "annotated_types_truss"

    tr = TrussHandle(truss_dir)

    with ensure_kill_all():
        container, urls = tr.docker_run_for_test()
        # Valid JSON input.
        json_input = {"prompt": "value"}
        response = requests.post(urls.predict_url, json=json_input)
        assert response.json() == {"generated_text": "value"}

        # Valid binary input.
        byte_input = serialization.truss_msgpack_serialize(json_input)
        print(byte_input)
        response = requests.post(
            urls.predict_url,
            data=byte_input,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert response.content == b"\x81\xaegenerated_text\xa5value"

        # An invalid input
        response = requests.post(urls.predict_url, json={"bad_key": "value"})
        assert response.status_code == 400
        assert "error" in response.json()
        assert (
            "Input Parsing Error:\n  `prompt`: Field required."
            in response.json()["error"]
        )
        assert response.headers["x-baseten-error-source"] == "04"
        assert response.headers["x-baseten-error-code"] == "700"

        # Schema response.
        schema_response = requests.get(urls.schema_url)
        schema = schema_response.json()
        assert schema["input_schema"] == {
            "properties": {"prompt": {"title": "Prompt", "type": "string"}},
            "required": ["prompt"],
            "title": "ModelInput",
            "type": "object",
        }
        assert schema["output_schema"] == {
            "properties": {
                "generated_text": {"title": "Generated Text", "type": "string"}
            },
            "required": ["generated_text"],
            "title": "ModelOutput",
            "type": "object",
        }
        assert not schema["supports_streaming"]


@pytest.mark.integration
def test_truss_annotated_inputs_streaming_and_output():
    streaming_truss = """
from pydantic import BaseModel
from typing import Union, Generator

class ModelInput(BaseModel):
    prompt: str

class ModelOutput(BaseModel):
    generated_text: str

class Model:
    def predict(self, request: ModelInput) -> Union[ModelOutput, Generator[str, None, None]]:
        return ModelOutput(generated_text=request.prompt)
"""

    with ensure_kill_all(), tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        truss_dir = Path(tmp_work_dir, "truss")

        create_truss(truss_dir, DEFAULT_CONFIG, streaming_truss)

        tr = TrussHandle(truss_dir)
        container, urls = tr.docker_run_for_test()
        response = requests.post(urls.predict_url, json={"prompt": "value"})

        assert response.json() == {"generated_text": "value"}

        schema_response = requests.get(urls.schema_url)

        schema = schema_response.json()

        assert schema["input_schema"] == {
            "properties": {"prompt": {"title": "Prompt", "type": "string"}},
            "required": ["prompt"],
            "title": "ModelInput",
            "type": "object",
        }

        assert schema["output_schema"] == {
            "properties": {
                "generated_text": {"title": "Generated Text", "type": "string"}
            },
            "required": ["generated_text"],
            "title": "ModelOutput",
            "type": "object",
        }
        assert schema["supports_streaming"]


@pytest.mark.integration
def test_truss_annotated_inputs_just_streaming():
    streaming_truss = """
from pydantic import BaseModel
from typing import Generator

class ModelInput(BaseModel):
    prompt: str

class ModelOutput(BaseModel):
    generated_text: str

class Model:
    def predict(self, prompt: ModelInput) -> Generator[str, None, None]:
        def inner():
            for i in range(2):
                yield str(i)
        return inner()
"""

    with ensure_kill_all(), tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        truss_dir = Path(tmp_work_dir, "truss")

        create_truss(truss_dir, DEFAULT_CONFIG, streaming_truss)
        tr = TrussHandle(truss_dir)
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={"prompt": "value"})

        assert response.content == b"01"

        schema_response = requests.get(urls.schema_url)

        schema = schema_response.json()

        assert schema["input_schema"] == {
            "properties": {"prompt": {"title": "Prompt", "type": "string"}},
            "required": ["prompt"],
            "title": "ModelInput",
            "type": "object",
        }

        assert schema["output_schema"] is None
        assert schema["supports_streaming"]


@pytest.mark.integration
def test_async_truss():
    truss_contents = """
from pydantic import BaseModel
from typing import AsyncGenerator
from typing import Awaitable


class ModelInput(BaseModel):
    prompt: str

class ModelOutput(BaseModel):
    generated_text: str

class Model:
    async def predict(self, request: ModelInput) -> Awaitable[ModelOutput]:
        return ModelOutput(generated_text=request.prompt)
"""

    with ensure_kill_all(), tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        truss_dir = Path(tmp_work_dir, "truss")

        create_truss(truss_dir, DEFAULT_CONFIG, truss_contents)
        tr = TrussHandle(truss_dir)
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={"prompt": "value"})

        assert response.json() == {"generated_text": "value"}

        schema_response = requests.get(urls.schema_url)

        schema = schema_response.json()

        assert schema["input_schema"] == {
            "properties": {"prompt": {"title": "Prompt", "type": "string"}},
            "required": ["prompt"],
            "title": "ModelInput",
            "type": "object",
        }

        assert schema["output_schema"] == {
            "properties": {
                "generated_text": {"title": "Generated Text", "type": "string"}
            },
            "required": ["generated_text"],
            "title": "ModelOutput",
            "type": "object",
        }
        assert not schema["supports_streaming"]


@pytest.mark.integration
def test_async_truss_streaming():
    streaming_truss = """
from pydantic import BaseModel
from typing import AsyncGenerator

class ModelInput(BaseModel):
    prompt: str

class ModelOutput(BaseModel):
    generated_text: str

class Model:
    async def predict(self, prompt: ModelInput) -> AsyncGenerator[str, None]:
        for i in range(2):
            yield str(i)
"""

    with ensure_kill_all(), tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        truss_dir = Path(tmp_work_dir, "truss")

        create_truss(truss_dir, DEFAULT_CONFIG, streaming_truss)
        tr = TrussHandle(truss_dir)
        container, urls = tr.docker_run_for_test()

        response = requests.post(urls.predict_url, json={"prompt": "value"})

        assert response.content == b"01"

        schema_response = requests.get(urls.schema_url)

        schema = schema_response.json()

        assert schema["input_schema"] == {
            "properties": {"prompt": {"title": "Prompt", "type": "string"}},
            "required": ["prompt"],
            "title": "ModelInput",
            "type": "object",
        }

        assert schema["output_schema"] is None
        assert schema["supports_streaming"]


@pytest.mark.integration
def test_async_truss_streaming_and_non_streaming():
    streaming_truss = """
from pydantic import BaseModel
from typing import Union, AsyncGenerator, Awaitable

class ModelInput(BaseModel):
    prompt: str
    stream: bool

class ModelOutput(BaseModel):
    generated_text: str

class Model:
    async def predict(self, request: ModelInput) -> Union[Awaitable[ModelOutput], AsyncGenerator[str, None]]:
        if request.stream:
            def inner():
                for i in range(2):
                    yield str(i)
            return inner()
        else:
            return ModelOutput(generated_text=request.prompt)
"""

    with ensure_kill_all(), tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        truss_dir = Path(tmp_work_dir, "truss")

        create_truss(truss_dir, DEFAULT_CONFIG, streaming_truss)

        tr = TrussHandle(truss_dir)
        container, urls = tr.docker_run_for_test()

        response = requests.post(
            urls.predict_url, json={"prompt": "value", "stream": False}
        )

        assert response.json() == {"generated_text": "value"}
        assert response.status_code == 200

        response = requests.post(
            urls.predict_url, json={"prompt": "value", "stream": True}
        )

        assert response.status_code == 200
        assert response.content == b"01"

        schema_response = requests.get(urls.schema_url)

        schema = schema_response.json()

        assert schema["input_schema"] == {
            "properties": {
                "prompt": {"title": "Prompt", "type": "string"},
                "stream": {"title": "Stream", "type": "boolean"},
            },
            "required": ["prompt", "stream"],
            "title": "ModelInput",
            "type": "object",
        }

        assert schema["output_schema"] == {
            "properties": {
                "generated_text": {"title": "Generated Text", "type": "string"}
            },
            "required": ["generated_text"],
            "title": "ModelOutput",
            "type": "object",
        }
        assert schema["supports_streaming"]


@pytest.mark.integration
def test_preprocess_postprocess():
    streaming_truss = """
from pydantic import BaseModel

class ModelInput(BaseModel):
    prompt: str

class ModelOutput(BaseModel):
    generated_text: str

class Model:
    def preprocess(self, request: ModelInput) -> str:
        return request.prompt

    def predict(self, request: str) -> str:
        return request

    def postprocess(self, request: str) -> ModelOutput:
        return ModelOutput(generated_text=request)
"""

    with ensure_kill_all(), tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        truss_dir = Path(tmp_work_dir, "truss")

        create_truss(truss_dir, DEFAULT_CONFIG, streaming_truss)

        tr = TrussHandle(truss_dir)
        container, urls = tr.docker_run_for_test()

        response = requests.post(
            urls.predict_url, json={"prompt": "value", "stream": False}
        )

        assert response.json() == {"generated_text": "value"}
        assert response.status_code == 200

        schema_response = requests.get(urls.schema_url)

        schema = schema_response.json()

        assert schema["input_schema"] == {
            "properties": {"prompt": {"title": "Prompt", "type": "string"}},
            "required": ["prompt"],
            "title": "ModelInput",
            "type": "object",
        }

        assert schema["output_schema"] == {
            "properties": {
                "generated_text": {"title": "Generated Text", "type": "string"}
            },
            "required": ["generated_text"],
            "title": "ModelOutput",
            "type": "object",
        }
        assert not schema["supports_streaming"]
