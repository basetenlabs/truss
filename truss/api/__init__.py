from typing import Optional, cast

import truss
from truss.api import definitions
from truss.remote.baseten.service import BasetenService
from truss.remote.remote_factory import RemoteFactory
from truss.remote.truss_remote import RemoteConfig


def login(api_key: str):
    """
    Logs user into Baseten account. Persists information to ~/.trussrc file,
    so only needs to be invoked once.
    Args:
        api_key: Baseten API Key
    """
    remote_url = "https://app.baseten.co"
    remote_config = RemoteConfig(
        name="baseten",
        configs={
            "remote_provider": "baseten",
            "api_key": api_key,
            "remote_url": remote_url,
        },
    )
    RemoteFactory.update_remote_config(remote_config)


def push(
    target_directory: str,
    remote: Optional[str] = None,
    model_name: Optional[str] = None,
    publish: bool = False,
    promote: bool = False,
    preserve_previous_production_deployment: bool = False,
    trusted: bool = False,
    deployment_name: Optional[str] = None,
) -> definitions.ModelDeployment:
    """
    Pushes a Truss to Baseten.

    Args:
        target_directory: Directory of Truss to push.
        remote: Name of the remote in .trussrc to patch changes to.
        model_name: The name of the model, if different from the one in the config.yaml.
        publish: Push the truss as a published deployment. If no production deployment exists,
            promote the truss to production after deploy completes.
        promote: Push the truss as a published deployment. Even if a production deployment exists,
            promote the truss to production after deploy completes.
        preserve_previous_production_deployment: Preserve the previous production deployment’s autoscaling
            setting. When not specified, the previous production deployment will be updated to allow it to
            scale to zero. Can only be use in combination with `promote` option.
        trusted: Give Truss access to secrets on remote host.
        deployment_name: Name of the deployment created by the push. Can only be
            used in combination with `publish` or `promote`. Deployment name must
            only contain alphanumeric, ’.’, ’-’ or ’_’ characters.

    Returns:
        The newly created ModelDeployment.
    """

    if not remote:
        available_remotes = RemoteFactory.get_available_config_names()
        if len(available_remotes) == 1:
            remote = available_remotes[0]
        elif len(available_remotes) == 0:
            raise ValueError(
                "Please authenticate via truss.login and pass it as an argument."
            )
        else:
            raise ValueError(
                "Multiple remotes found. Please pass the remote as an argument."
            )

    remote_provider = RemoteFactory.create(remote=remote)
    tr = truss.load(target_directory)
    model_name = model_name or tr.spec.config.model_name
    if not model_name:
        raise ValueError(
            "No model name provided. Please specify a model name in config.yaml."
        )

    service = remote_provider.push(
        tr,
        model_name=model_name,
        publish=publish,
        trusted=trusted,
        promote=promote,
        preserve_previous_prod_deployment=preserve_previous_production_deployment,
        deployment_name=deployment_name,
    )  # type: ignore

    return definitions.ModelDeployment(cast(BasetenService, service))
