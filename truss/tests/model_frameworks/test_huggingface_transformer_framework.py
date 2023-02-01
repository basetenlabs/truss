import tempfile
from pathlib import Path

import pytest
import requests
from python_on_whales import docker
from truss.constants import CONFIG_FILE, TRUSS
from truss.contexts.image_builder.serving_image_builder import (
    ServingImageBuilderContext,
)
from truss.contexts.local_loader.load_model_local import LoadModelLocal
from truss.model_frameworks.huggingface_transformer import HuggingfaceTransformer
from truss.tests.test_testing_utilities_for_other_tests import ensure_kill_all
from truss.truss_config import TrussConfig
from truss.truss_handle import wait_for_truss


def test_to_truss(huggingface_transformer_t5_small_pipeline):
    with tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        model = huggingface_transformer_t5_small_pipeline
        truss_dir = Path(tmp_work_dir, "truss")
        framework = HuggingfaceTransformer()
        framework.to_truss(model, truss_dir)

        # Assertions
        config = TrussConfig.from_yaml(truss_dir / CONFIG_FILE)
        assert config.model_class_filename == "model.py"
        assert config.model_class_name == "Model"
        assert config.model_framework == framework.typ()
        assert config.model_type == "text2text-generation"
        assert len(config.requirements) == 0

        assert config.python_version.startswith("py3")
        model_metadata = config.model_metadata
        assert not model_metadata["has_named_args"]
        assert not model_metadata["has_hybrid_args"]

        assert (truss_dir / "data" / "model").exists()
        assert (truss_dir / "model" / "model.py").exists()


def test_run_truss(huggingface_transformer_t5_small_pipeline):
    with tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        model = huggingface_transformer_t5_small_pipeline
        truss_dir = Path(tmp_work_dir, "truss")
        framework = HuggingfaceTransformer()
        framework.to_truss(model, truss_dir)
        model = LoadModelLocal.run(truss_dir)
        result = model.predict({"inputs": "My name is Sarah and I live in London"})
        predictions = result["predictions"]
        assert len(predictions) == 1
        prediction = predictions[0]
        assert prediction["generated_text"].startswith("Mein Name")


@pytest.mark.integration
def test_run_image(huggingface_transformer_t5_small_pipeline):
    with ensure_kill_all(), tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        model = huggingface_transformer_t5_small_pipeline
        truss_dir = Path(tmp_work_dir, "truss")
        framework = HuggingfaceTransformer()
        framework.to_truss(model, truss_dir)
        tag = f"test-{framework.typ().value}-model:latest"
        image = ServingImageBuilderContext.run(truss_dir).build_image(
            tag=tag, labels={TRUSS: True}
        )
        assert image.repo_tags == [tag]
        container = docker.run(tag, publish=[[8080, 8080]], detach=True)

        wait_for_truss("http://localhost:8080/v1/models/model", container)

        resp = requests.post(
            "http://localhost:8080/v1/models/model:predict",
            json={
                "inputs": ["My name is Sarah and I live in London"],
            },
        )

        predictions = resp.json()["predictions"]
        assert len(predictions) == 1
        prediction = predictions[0]
        assert prediction["generated_text"].startswith("Mein Name")
