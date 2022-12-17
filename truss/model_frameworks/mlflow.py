from pathlib import Path
from typing import Dict, Set

from truss.constants import MLFLOW_REQ_MODULE_NAMES
from truss.model_framework import ModelFramework
from truss.templates.server.common.util import model_supports_predict_proba
from truss.truss_handle import TrussHandle
from truss.types import ModelFrameworkType

MODEL_FILENAME = "model.joblib"


class Mlflow(ModelFramework):
    def typ(self) -> ModelFrameworkType:
        return ModelFrameworkType.MLFLOW

    def required_python_depedencies(self) -> Set[str]:
        return MLFLOW_REQ_MODULE_NAMES

    def serialize_model_to_directory(self, model, target_directory: Path):
        import mlflow

        if isinstance(model, str):
            self._download_model_from_uri(uri=model, target_directory=target_directory)
        elif isinstance(model, mlflow.pyfunc.PyFuncModel):
            self._download_model_from_pyfunc(
                model=model, target_directory=target_directory
            )

    def model_metadata(self, model) -> Dict[str, str]:
        supports_predict_proba = model_supports_predict_proba(model)
        return {
            "model_binary_dir": "model",
            "supports_predict_proba": supports_predict_proba,
        }

    def supports_model_class(self, model_class) -> bool:
        model_framework, _, _ = model_class.__module__.partition(".")
        return model_framework == ModelFrameworkType.MLFLOW.value

    def to_truss(self, model, target_directory: Path) -> str:
        super().to_truss(model, target_directory)
        self._add_mlflow_requirements(target_directory)

    def _download_model_from_uri(self, uri: str, target_directory: Path):
        from mlflow.artifacts import download_artifacts

        download_artifacts(artifact_uri=uri, dst_path=target_directory)

    def _download_model_from_pyfunc(self, model, target_directory: Path):
        from mlflow.artifacts import download_artifacts

        run_id = model._model_meta.run_id
        download_artifacts(run_id=run_id, dst_path=target_directory)

    def _add_mlflow_requirements(self, target_directory: str):
        truss = TrussHandle(truss_dir=target_directory)
        requirements_file = (
            truss._spec.data_dir / "model" / "model" / "requirements.txt"
        )
        if not requirements_file.exists():
            return
        truss.update_requirements_from_file(requirements_file)
