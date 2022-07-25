import tempfile
from pathlib import Path

import numpy as np
import pytest
from truss.constants import CONFIG_FILE
from truss.contexts.local_loader.load_local import LoadLocal
from truss.model_frameworks.xgboost import XGBoost
from truss.tests.test_testing_utilities_for_other_tests import ensure_kill_all
from truss.truss_config import TrussConfig
from truss.truss_handle import TrussHandle
from truss.types import ModelFrameworkType


def test_to_truss(xgboost_pima_model):
    with tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        truss_dir = Path(tmp_work_dir, "truss")
        xgboost_framework = XGBoost()
        xgboost_framework.to_truss(xgboost_pima_model, truss_dir)

        # Assertions
        config = TrussConfig.from_yaml(truss_dir / CONFIG_FILE)
        assert config.model_class_filename == "model.py"
        assert config.model_class_name == "Model"
        assert config.model_framework == ModelFrameworkType.XGBOOST
        assert config.model_type == "Model"
        assert config.python_version.startswith("py3")
        assert len(config.requirements) != 0

        model_metadata = config.model_metadata
        assert model_metadata["model_binary_dir"] == "model"
        assert not model_metadata["supports_predict_proba"]

        assert (truss_dir / "data" / "model" / "model.ubj").exists()
        assert (truss_dir / "model" / "model.py").exists()


def test_run_truss(xgboost_pima_model):
    with tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        truss_dir = Path(tmp_work_dir, "truss")
        xgboost_framework = XGBoost()
        xgboost_framework.to_truss(xgboost_pima_model, truss_dir)
        model = LoadLocal.run(truss_dir)
        predictions = model.predict({"inputs": [[0, 0, 0, 0, 0, 0, 0, 0]]})
        assert "predictions" in predictions
        assert len(predictions["predictions"]) == 1


@pytest.mark.integration
def test_run_image(xgboost_pima_model):
    with ensure_kill_all(), tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        truss_dir = Path(tmp_work_dir, "truss")
        xgboost_framework = XGBoost()
        xgboost_framework.to_truss(xgboost_pima_model, truss_dir)
        tr = TrussHandle(truss_dir)
        predictions = tr.docker_predict(
            {"inputs": [[0, 0, 0, 0, 0, 0, 0, 0]]},
            local_port=8080,
        )
        assert len(predictions["predictions"]) == 1
        assert np.shape(predictions["predictions"]) == (1,)
