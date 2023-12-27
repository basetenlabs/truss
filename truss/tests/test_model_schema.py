from pathlib import Path

import pytest
import requests
from truss.tests.test_testing_utilities_for_other_tests import ensure_kill_all
from truss.truss_handle import TrussHandle

# Test cases to add:
# * Normal truss with no annotations
# * Streaming
# * Truss that supports both streaming & non-streaming
# * Async
# * Async Streaming
# * Async w/ both streaming and non-streaming
# * Invalid json input


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
