import pathlib
import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, cast

if TYPE_CHECKING:
    from rich import progress

from truss.api import definitions
from truss.cli.resolvers.model_team_resolver import resolve_model_team_name
from truss.remote.baseten.remote import BasetenRemote
from truss.remote.baseten.service import BasetenService
from truss.remote.remote_factory import RemoteFactory
from truss.remote.truss_remote import RemoteConfig
from truss.truss_handle.build import load


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


def whoami(remote: Optional[str] = None):
    """
    Returns account information for the current user.
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
    return remote_provider.whoami()


def _resolve_team_id(
    remote_provider: BasetenRemote,
    provided_team_name: Optional[str],
    remote_name: Optional[str],
    model_name: str,
) -> Optional[str]:
    """Resolve team_id using the same logic as the CLI's ``--team`` flag.

    Uses :func:`resolve_model_team_name` with ``allow_interactive=False`` so that
    ambiguous cases raise an error instead of prompting.
    """
    _, team_id = resolve_model_team_name(
        remote_provider=remote_provider,
        provided_team_name=provided_team_name,
        existing_model_name=model_name,
        allow_interactive=False,
        remote_name=remote_name,
    )
    return team_id


def push(
    target_directory: str,
    remote: Optional[str] = None,
    model_name: Optional[str] = None,
    publish: bool = False,
    promote: bool = False,
    preserve_previous_production_deployment: bool = False,
    trusted: Optional[bool] = None,
    disable_truss_download: bool = False,
    deployment_name: Optional[str] = None,
    environment: Optional[str] = None,
    progress_bar: Optional[Type["progress.Progress"]] = None,
    include_git_info: bool = False,
    preserve_env_instance_type: bool = True,
    deploy_timeout_minutes: Optional[int] = None,
    labels: Optional[Dict[str, Any]] = None,
    team: Optional[str] = None,
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
        preserve_previous_production_deployment: Preserve the previous production deployment's autoscaling
            setting. When not specified, the previous production deployment will be updated to allow it to
            scale to zero. Can only be use in combination with `promote` option.
        trusted: [DEPRECATED]
        deployment_name: Name of the deployment created by the push. Can only be
            used in combination with `publish` or `promote`. Deployment name must
            only contain alphanumeric, '.', '-' or '_' characters.
        environment: Name of stable environment on baseten.
        progress_bar: Optional `rich.progress.Progress` if output is desired.
        disable_truss_download: Disable downloading of the truss directory from the UI.
        include_git_info: Whether to attach git versioning info (sha, branch, tag) to
          deployments made from within a git repo. If set to True in `.trussrc`, it
          will always be attached.
        preserve_env_instance_type: When pushing a truss to an environment, whether to use the resources
          specified in the truss config to resolve the instance type or preserve the instance type
          configured in the specified environment.
        deploy_timeout_minutes: Optional timeout in minutes for the deployment operation.
        labels: Optional JSON-serializable dictionary of label key-value pairs.
        team: Name of the team to push the model to.

    Returns:
        The newly created ModelDeployment.
    """
    if trusted is not None:
        warnings.warn(
            "`trusted` is deprecated and will be ignored, all models are "
            "trusted by default now.",
            DeprecationWarning,
        )
    if labels is not None and not isinstance(labels, dict):
        raise ValueError("labels must be a JSON-serializable dictionary.")

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

    tr = load(target_directory)
    model_name = model_name or tr.spec.config.model_name
    if not model_name:
        raise ValueError(
            "No model name provided. Please specify a model name in config.yaml."
        )

    team_id = None
    if isinstance(remote_provider, BasetenRemote):
        team_id = _resolve_team_id(remote_provider, team, remote, model_name)

    service = remote_provider.push(
        tr,
        model_name=model_name,
        working_dir=pathlib.Path(target_directory),
        publish=publish,
        promote=promote,
        preserve_previous_prod_deployment=preserve_previous_production_deployment,
        deployment_name=deployment_name,
        environment=environment,
        disable_truss_download=disable_truss_download,
        progress_bar=progress_bar,
        include_git_info=include_git_info,
        preserve_env_instance_type=preserve_env_instance_type,
        deploy_timeout_minutes=deploy_timeout_minutes,
        team_id=team_id,
        labels=labels,
    )  # type: ignore

    return definitions.ModelDeployment(cast(BasetenService, service))
