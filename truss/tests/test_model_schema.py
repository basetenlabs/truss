import tempfile
from pathlib import Path

import pytest
import requests
from truss.tests.helpers import create_truss
from truss.tests.test_testing_utilities_for_other_tests import ensure_kill_all
from truss.truss_handle import TrussHandle

# Test cases to add:
# * Normal truss with no annotations
# * [done] Streaming
# * [done] Truss that supports both streaming & non-streaming
# * Async
# * Async Streaming
# * Async w/ both streaming and non-streaming
# * preproces / postprocess
# * Invalid json input

DEFAULT_CONFIG = """model_name: test-truss"""


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
    def predict(self, request: ModelInput) -> Union[ModelOutput, Generator]:
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
    def predict(self, prompt: ModelInput) -> Generator:
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
