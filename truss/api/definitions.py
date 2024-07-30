import time

import pydantic

from truss.remote.baseten import service
from truss.remote.baseten.core import (
    ACTIVE_STATUS,
    DEPLOYING_STATUSES,
)


class ModelDeployment(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(protected_namespaces=())

    model_id: str
    model_deployment_id: str
    _baseten_service: service.BasetenService

    @classmethod
    def from_service(cls, service: service.BasetenService) -> "ModelDeployment":
        model_deployment = cls(
            model_id=service._model_id,
            model_deployment_id=service._model_version_id,
        )

        model_deployment._baseten_service = service
        return model_deployment

    def wait_for_active(self, timeout_seconds: int = 60) -> str:
        """
        Waits for the deployment to be active.

        Args:
            timeout_seconds: The maximum time to wait for the deployment to be active.

        Returns:
            The status of the deployment.
        """
        start_time = time.time()
        for deployment_status in self._baseten_service.poll_deployment_status():
            if (
                timeout_seconds is not None
                and time.time() - start_time > timeout_seconds
            ):
                raise TimeoutError("Deployment timed out.")

            if deployment_status not in DEPLOYING_STATUSES:
                raise Exception(f"Deployment failed with status: {deployment_status}")

            if deployment_status == ACTIVE_STATUS:
                return deployment_status

        raise RuntimeError("Error polling deployment status.")
