import tempfile
from pathlib import Path

import pytest
import requests
from truss.tests.helpers import create_truss
from truss.tests.test_testing_utilities_for_other_tests import ensure_kill_all
from truss.truss_handle import TrussHandle

# Test cases to add:
# * [done] Normal truss with no annotations
# * [done] Streaming
# * [done] Truss that supports both streaming & non-streaming
# * [done] Async
# * [done] Async Streaming
# * [done] Async w/ both streaming and non-streaming
# * preproces / postprocess
# * [done] Invalid json input

DEFAULT_CONFIG = """model_name: test-truss"""


@pytest.mark.integration
def test_truss_with_no_annotations():
    truss_root = Path(__file__).parent.parent.parent.resolve()

    truss_dir = truss_root / "truss" / "test_data" / "test_basic_truss"

    tr = TrussHandle(truss_dir)

    with ensure_kill_all():
        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)
        truss_server_addr = "http://localhost:8090"
        full_url = f"{truss_server_addr}/v1/models/model:predict"

        response = requests.post(full_url, json={"prompt": "value"})
        assert response.json() == {
            "prompt": "value",
        }

        schema_response = requests.get(f"{truss_server_addr}/v1/models/model/schema")
        assert schema_response.status_code == 404

        schema_response.json()["error"] == "No schema found"


@pytest.mark.integration
def test_truss_with_annotated_inputs_outputs():
    truss_root = Path(__file__).parent.parent.parent.resolve()

    truss_dir = truss_root / "examples" / "annotated_types"

    tr = TrussHandle(truss_dir)

    with ensure_kill_all():
        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)
        truss_server_addr = "http://localhost:8090"
        full_url = f"{truss_server_addr}/v1/models/model:predict"

        response = requests.post(full_url, json={"prompt": "value"})
        assert response.json() == {
            "generated_text": "value",
        }

        # An invalid input

        response = requests.post(full_url, json={"bad_key": "value"})

        assert response.status_code == 400
        assert "error" in response.json()
        assert "validation error" in response.json()["error"]

        schema_response = requests.get(f"{truss_server_addr}/v1/models/model/schema")

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
        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)
        truss_server_addr = "http://localhost:8090"
        full_url = f"{truss_server_addr}/v1/models/model:predict"

        response = requests.post(full_url, json={"prompt": "value"})

        assert response.json() == {"generated_text": "value"}

        schema_response = requests.get(f"{truss_server_addr}/v1/models/model/schema")

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
        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)
        truss_server_addr = "http://localhost:8090"
        full_url = f"{truss_server_addr}/v1/models/model:predict"

        response = requests.post(full_url, json={"prompt": "value"})

        assert response.content == b"01"

        schema_response = requests.get(f"{truss_server_addr}/v1/models/model/schema")

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
        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)
        truss_server_addr = "http://localhost:8090"
        full_url = f"{truss_server_addr}/v1/models/model:predict"

        response = requests.post(full_url, json={"prompt": "value"})

        assert response.json() == {"generated_text": "value"}

        schema_response = requests.get(f"{truss_server_addr}/v1/models/model/schema")

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
    # TODO: Write the test
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
        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)
        truss_server_addr = "http://localhost:8090"
        full_url = f"{truss_server_addr}/v1/models/model:predict"

        response = requests.post(full_url, json={"prompt": "value"})

        assert response.content == b"01"

        schema_response = requests.get(f"{truss_server_addr}/v1/models/model/schema")

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
        _ = tr.docker_run(local_port=8090, detach=True, wait_for_server_ready=True)
        truss_server_addr = "http://localhost:8090"
        full_url = f"{truss_server_addr}/v1/models/model:predict"

        response = requests.post(full_url, json={"prompt": "value", "stream": False})

        assert response.json() == {"generated_text": "value"}
        assert response.status_code == 200

        response = requests.post(full_url, json={"prompt": "value", "stream": True})

        assert response.status_code == 200
        assert response.content == b"01"

        schema_response = requests.get(f"{truss_server_addr}/v1/models/model/schema")

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
