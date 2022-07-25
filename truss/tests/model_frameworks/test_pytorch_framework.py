import pytest
import torch
from torch import package
from truss.model_frameworks.pytorch import (
    TORCH_MODEL_PACKAGE_NAME,
    TORCH_MODEL_PICKLE_FILENAME,
    TORCH_PACKAGE_FILE,
    PyTorch,
)
from truss.tests.test_testing_utilities_for_other_tests import ensure_kill_all
from truss.truss_handle import TrussHandle


def test_serialize_model_to_directory(pytorch_model_with_numpy_import, tmp_path):
    pytorch = PyTorch()
    model = pytorch_model_with_numpy_import[0]
    pytorch.serialize_model_to_directory(model, tmp_path)
    pkg_file_path = tmp_path / TORCH_PACKAGE_FILE
    assert pkg_file_path.exists()
    imp = package.PackageImporter(pkg_file_path)
    model = imp.load_pickle(TORCH_MODEL_PACKAGE_NAME, TORCH_MODEL_PICKLE_FILENAME)
    result = model(torch.tensor([[0, 0, 0]], dtype=torch.float32)).tolist()
    assert len(result) == 1


def test_supports_model_class(pytorch_model_with_numpy_import):
    pytorch = PyTorch()
    assert pytorch.supports_model_class(pytorch_model_with_numpy_import[0].__class__)


@pytest.mark.integration
def test_run_image(pytorch_model_with_numpy_import, tmp_path):
    truss_dir = tmp_path / "truss"
    pytorch = PyTorch()
    model = pytorch_model_with_numpy_import[0]
    pytorch.to_truss(model, truss_dir)
    truss = TrussHandle(truss_dir)
    with ensure_kill_all():
        result = truss.docker_predict(
            {"inputs": [[0, 0, 0]]},
            local_port=8090,
        )
        assert len(result["predictions"]) == 1
