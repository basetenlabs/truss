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
from truss.model_frameworks.keras import Keras
from truss.tests.test_testing_utilities_for_other_tests import ensure_kill_all
from truss.truss_config import TrussConfig
from truss.truss_handle import wait_for_truss


def test_to_truss(keras_mpg_model):
    with tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        truss_dir = Path(tmp_work_dir, "truss")
        framework = Keras()
        framework.to_truss(keras_mpg_model, truss_dir)

        # Assertions
        config = TrussConfig.from_yaml(truss_dir / CONFIG_FILE)
        assert config.model_class_filename == "model.py"
        assert config.model_class_name == "Model"
        assert config.model_framework == framework.typ()
        assert config.model_type == "Model"
        assert config.python_version.startswith("py3")
        assert len(config.requirements) != 0

        model_metadata = config.model_metadata
        assert model_metadata["model_binary_dir"] == "model"

        assert (truss_dir / "data" / "model").exists()
        assert (truss_dir / "model" / "model.py").exists()


def test_run_truss(keras_mpg_model):
    with tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        truss_dir = Path(tmp_work_dir, "truss")
        sklearn_framework = Keras()
        sklearn_framework.to_truss(keras_mpg_model, truss_dir)
        model = LoadModelLocal.run(truss_dir)
        result = model.predict({"inputs": [0, 0, 0, 0, 0, 0, 0, 0, 0]})
        predictions = result["predictions"]
        assert len(predictions) == 1
        assert len(predictions[0]) == 1


@pytest.mark.integration
def test_run_image(keras_mpg_model):
    with ensure_kill_all(), tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        truss_dir = Path(tmp_work_dir, "truss")
        framework = Keras()
        framework.to_truss(keras_mpg_model, truss_dir)
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
                "inputs": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
        )
        predictions = resp.json()["predictions"]
        assert len(predictions) == 1
        assert len(predictions[0]) == 1
