"""Build DockerAuth from runtime_config fields."""

from typing import Optional

from truss.base.truss_config import DockerAuthType
from truss_train.definitions import (
    AWSIAMDockerAuth,
    DockerAuth,
    GCPServiceAccountJSONDockerAuth,
    RegistrySecretDockerAuth,
    SecretReference,
)

# Maps CLI flag values to (DockerAuthType, builder function)
_AUTH_METHODS = {
    "registry_secret": DockerAuthType.REGISTRY_SECRET,
    "gcp_service_account": DockerAuthType.GCP_SERVICE_ACCOUNT_JSON,
    "aws_iam": DockerAuthType.AWS_IAM,
}


def build_docker_auth(
    base_image: str, auth_method: Optional[str], auth_secret: Optional[str]
) -> Optional[DockerAuth]:
    """Build a DockerAuth object from runtime_config values, or None."""
    if not auth_method or not auth_secret:
        return None

    auth_type = _AUTH_METHODS.get(auth_method)
    if not auth_type:
        return None

    # Extract registry from image (everything before the first /)
    registry = base_image.split("/")[0] if "/" in base_image else ""

    kwargs: dict = {"auth_method": auth_type, "registry": registry}

    if auth_type == DockerAuthType.REGISTRY_SECRET:
        kwargs["registry_secret_docker_auth"] = RegistrySecretDockerAuth(
            secret_ref=SecretReference(name=auth_secret)
        )
    elif auth_type == DockerAuthType.GCP_SERVICE_ACCOUNT_JSON:
        kwargs["gcp_service_account_json_docker_auth"] = (
            GCPServiceAccountJSONDockerAuth(
                service_account_json_secret_ref=SecretReference(name=auth_secret)
            )
        )
    elif auth_type == DockerAuthType.AWS_IAM:
        # For AWS, auth_secret is comma-separated: "access_key_secret,secret_key_secret"
        parts = auth_secret.split(",")
        access_key = parts[0].strip()
        secret_key = parts[1].strip() if len(parts) > 1 else parts[0].strip()
        kwargs["aws_iam_docker_auth"] = AWSIAMDockerAuth(
            access_key_secret_ref=SecretReference(name=access_key),
            secret_access_key_secret_ref=SecretReference(name=secret_key),
        )

    return DockerAuth(**kwargs)
