import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest
from truss.constants import PYTORCH
from truss.model_frameworks import SKLearn
from truss.model_inference import (
    infer_model_information,
    map_to_supported_python_version,
    validate_provided_parameters_with_model,
)
from truss.tests.test_testing_utilities_for_other_tests import ensure_kill_all
from truss.truss_handle import TrussHandle

logger = logging.getLogger(__name__)


def test_pytorch_init_arg_validation(
    pytorch_model_with_init_args, pytorch_model_init_args
):
    pytorch_model_with_init_args, _ = pytorch_model_with_init_args
    # Validates with args and kwargs
    validate_provided_parameters_with_model(
        pytorch_model_with_init_args.__class__, pytorch_model_init_args
    )

    # Errors if bad args
    with pytest.raises(ValueError):
        validate_provided_parameters_with_model(
            pytorch_model_with_init_args.__class__, {"foo": "bar"}
        )

    # Validates with only args
    copied_args = pytorch_model_init_args.copy()
    copied_args.pop("kwarg1")
    copied_args.pop("kwarg2")
    validate_provided_parameters_with_model(pytorch_model_with_init_args, copied_args)

    # Requires all args
    with pytest.raises(ValueError):
        validate_provided_parameters_with_model(pytorch_model_with_init_args, {})


def test_infer_model_information(pytorch_model_with_init_args):
    model_info = infer_model_information(pytorch_model_with_init_args[0])
    assert model_info.model_framework == PYTORCH
    assert model_info.model_type == "MyModel"


@pytest.mark.parametrize(
    "python_version, expected_python_version",
    [
        ("py37", "py37"),
        ("py38", "py38"),
        ("py39", "py39"),
        ("py310", "py39"),
        ("py311", "py39"),
        ("py36", "py37"),
    ],
)
def test_map_to_supported_python_version(python_version, expected_python_version):
    out_python_version = map_to_supported_python_version(python_version)
    assert out_python_version == expected_python_version


@pytest.mark.integration
def test_binary_request(sklearn_rfc_model):
    with ensure_kill_all(), tempfile.TemporaryDirectory(dir=".") as tmp_work_dir:
        truss_dir = Path(tmp_work_dir, "truss")
        sklearn_framework = SKLearn()
        sklearn_framework.to_truss(sklearn_rfc_model, truss_dir)
        tr = TrussHandle(truss_dir)
        predictions = tr.docker_predict(
            {"inputs": [[0, 0, 0, 0]]}, local_port=8090, binary=True
        )
        assert len(predictions["probabilities"]) == 1
        assert np.shape(predictions["probabilities"]) == (1, 3)
