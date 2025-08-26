import datetime
import json
import logging
import pathlib
import textwrap
from typing import IO, TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Tuple, Type

import requests

from truss.base.errors import ValidationError

if TYPE_CHECKING:
    from rich import progress

from truss.base.constants import PRODUCTION_ENVIRONMENT_NAME
from truss.remote.baseten import custom_types as b10_types
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.error import ApiError
from truss.remote.baseten.utils.tar import create_tar_with_progress_bar
from truss.remote.baseten.utils.time import iso_to_millis
from truss.remote.baseten.utils.transfer import multipart_upload_boto3
from truss.util.path import load_trussignore_patterns_from_truss_dir

logger = logging.getLogger(__name__)


DEPLOYING_STATUSES = ["BUILDING", "DEPLOYING", "LOADING_MODEL", "UPDATING"]
ACTIVE_STATUS = "ACTIVE"
NO_ENVIRONMENTS_EXIST_ERROR_MESSAGING = (
    "Model hasn't been deployed yet. No environments exist."
)

# Maximum number of iterations to prevent infinite loops when paginating logs
MAX_ITERATIONS = 10_000
MIN_BATCH_SIZE = 100

# LIMIT for the number of logs to fetch per request defined by the server
MAX_BATCH_SIZE = 1000

NANOSECONDS_PER_MILLISECOND = 1_000_000
MILLISECONDS_PER_HOUR = 60 * 60 * 1000


class ModelIdentifier:
    value: str


class ModelName(ModelIdentifier):
    def __init__(self, name: str):
        self.value = name


class ModelId(ModelIdentifier):
    def __init__(self, model_id: str):
        self.value = model_id


class ModelVersionId(ModelIdentifier):
    def __init__(self, model_version_id: str):
        self.value = model_version_id


class PatchState(NamedTuple):
    current_hash: str
    current_signature: str


class TrussPatches(NamedTuple):
    django_patch_state: PatchState
    container_patch_state: PatchState


class TrussWatchState(NamedTuple):
    is_container_built_from_push: bool
    patches: Optional[TrussPatches]


class ChainDeploymentHandleAtomic(NamedTuple):
    hostname: str
    chain_id: str
    chain_deployment_id: str
    is_draft: bool


class ModelVersionHandle(NamedTuple):
    version_id: str
    model_id: str
    hostname: str
    instance_type_name: Optional[str] = None


def get_chain_id_by_name(api: BasetenApi, chain_name: str) -> Optional[str]:
    """
    Check if a chain with the given name exists in the Baseten remote.

    Args:
        api: BasetenApi instance
        chain_name: Name of the chain to check for existence

    Returns:
        chain_id if present, otherwise None
    """
    chains = api.get_chains()

    chain_name_id_mapping = {chain["name"]: chain["id"] for chain in chains}
    return chain_name_id_mapping.get(chain_name)


def get_dev_chain_deployment(api: BasetenApi, chain_id: str):
    chain_deployments = api.get_chain_deployments(chain_id)
    dev_deployments = [
        deployment for deployment in chain_deployments if deployment["is_draft"]
    ]
    if not dev_deployments:
        return None
    newest_draft_deployment = max(
        dev_deployments, key=lambda d: datetime.datetime.fromisoformat(d["created"])
    )
    return newest_draft_deployment


def create_chain_atomic(
    api: BasetenApi,
    chain_name: str,
    entrypoint: b10_types.ChainletDataAtomic,
    dependencies: List[b10_types.ChainletDataAtomic],
    is_draft: bool,
    truss_user_env: b10_types.TrussUserEnv,
    environment: Optional[str],
) -> ChainDeploymentHandleAtomic:
    if environment and is_draft:
        logging.info(
            f"Automatically publishing Chain `{chain_name}` based on "
            "environment setting."
        )
        is_draft = False

    chain_id = get_chain_id_by_name(api, chain_name)

    # TODO(Tyron): Refactor for better readability:
    # 1. Prepare all arguments for `deploy_chain_atomic`.
    # 2. Validate argument combinations.
    # 3. Make a single invocation to `deploy_chain_atomic`.

    if is_draft:
        res = api.deploy_chain_atomic(
            entrypoint=entrypoint,
            dependencies=dependencies,
            chain_name=chain_name,
            is_draft=True,
            truss_user_env=truss_user_env,
        )
    elif chain_id:
        # This is the only case where promote has relevance, since
        # if there is no chain already, the first deployment will
        # already be production, and only published deployments can
        # be promoted.
        try:
            res = api.deploy_chain_atomic(
                entrypoint=entrypoint,
                dependencies=dependencies,
                chain_id=chain_id,
                environment=environment,
                truss_user_env=truss_user_env,
            )
        except ApiError as e:
            if (
                e.graphql_error_code
                == BasetenApi.GraphQLErrorCodes.RESOURCE_NOT_FOUND.value
            ):
                raise ValueError(
                    f"Environment `{environment}` does not exist. You can "
                    f"create environments in the Chains UI."
                ) from e

            raise e
    elif environment and environment != PRODUCTION_ENVIRONMENT_NAME:
        raise ValueError(NO_ENVIRONMENTS_EXIST_ERROR_MESSAGING)
    else:
        res = api.deploy_chain_atomic(
            entrypoint=entrypoint,
            dependencies=dependencies,
            chain_name=chain_name,
            truss_user_env=truss_user_env,
        )

    return ChainDeploymentHandleAtomic(
        chain_deployment_id=res["chain_deployment"]["id"],
        chain_id=res["chain_deployment"]["chain"]["id"],
        hostname=res["chain_deployment"]["chain"]["hostname"],
        is_draft=is_draft,
    )


def exists_model(api: BasetenApi, model_name: str) -> Optional[str]:
    """
    Check if a model with the given name exists in the Baseten remote.

    Args:
        api: BasetenApi instance
        model_name: Name of the model to check for existence

    Returns:
        model_id if present, otherwise None
    """
    try:
        model = api.get_model(model_name)
    except ApiError as e:
        if (
            e.graphql_error_code
            == BasetenApi.GraphQLErrorCodes.RESOURCE_NOT_FOUND.value
        ):
            return None

        raise e

    return model["model"]["id"]


def get_model_and_versions(api: BasetenApi, model_name: ModelName) -> Tuple[dict, List]:
    query_result = api.get_model(model_name.value)["model"]
    return query_result, query_result["versions"]


def get_dev_version_from_versions(versions: List[dict]) -> Optional[dict]:
    """Given a list of model version dicts, returns the development version.

    Args:
        versions: List of model version dicts from the Baseten API

    Returns:
        The version in versions corresponding to the development model, or None
        if no development model exists
    """
    for version in versions:
        if version["is_draft"] is True:
            return version
    return None


def get_dev_version(api: BasetenApi, model_name: str) -> Optional[dict]:
    """Queries the Baseten API and returns a dict representing the
    development version for the given model_name.

    Args:
        api: BasetenApi instance
        model_name: Name of the model to get the development version for

    Returns:
        The development version of the model
    """
    model = api.get_model(model_name)
    versions = model["model"]["versions"]
    return get_dev_version_from_versions(versions)


def get_truss_watch_state(api: BasetenApi, model_name: str) -> TrussWatchState:
    response = api.get_truss_watch_state(model_name)["truss_watch_state"]
    django_patch_state = (
        None
        if response["django_patch_state"] is None
        else PatchState(
            current_hash=response["django_patch_state"]["current_hash"],
            current_signature=response["django_patch_state"]["current_signature"],
        )
    )
    container_patch_state = (
        None
        if response["container_patch_state"] is None
        else PatchState(
            current_hash=response["container_patch_state"]["current_hash"],
            current_signature=response["container_patch_state"]["current_signature"],
        )
    )
    patches = None
    if django_patch_state and container_patch_state:
        patches = TrussPatches(
            django_patch_state=django_patch_state,
            container_patch_state=container_patch_state,
        )
    return TrussWatchState(
        is_container_built_from_push=response["is_container_built_from_push"],
        patches=patches,
    )


def get_prod_version_from_versions(versions: List[dict]) -> Optional[dict]:
    # Loop over versions instead of using the primary_version field because
    # primary_version is set to the development version ID if no published
    # models exist.
    for version in versions:
        if version["is_primary"]:
            return version
    return None


def archive_dir(
    dir: pathlib.Path, progress_bar: Optional[Type["progress.Progress"]] = None
) -> IO:
    """Archive a TrussHandle into a tar file.

    Returns:
        A file-like object containing the tar file
    """
    # check for a truss_ignore file and read the ignore patterns if it exists
    ignore_patterns = load_trussignore_patterns_from_truss_dir(dir)

    try:
        temp_file = create_tar_with_progress_bar(
            dir, ignore_patterns, progress_bar=progress_bar
        )
    except PermissionError:
        # workaround for Windows bug with Tempfile that causes PermissionErrors
        temp_file = create_tar_with_progress_bar(
            dir, ignore_patterns, delete=False, progress_bar=progress_bar
        )
    temp_file.file.seek(0)
    return temp_file


def upload_truss(
    api: BasetenApi,
    serialize_file: IO,
    progress_bar: Optional[Type["progress.Progress"]],
) -> str:
    """
    Upload a TrussHandle to the Baseten remote.

    Args:
        api: BasetenApi instance
        serialize_file: File-like object containing the serialized TrussHandle

    Returns:
        The S3 key of the uploaded file
    """
    temp_credentials_s3_upload = api.model_s3_upload_credentials()
    s3_key = temp_credentials_s3_upload.pop("s3_key")
    s3_bucket = temp_credentials_s3_upload.pop("s3_bucket")
    multipart_upload_boto3(
        serialize_file.name, s3_bucket, s3_key, temp_credentials_s3_upload, progress_bar
    )
    return s3_key


def create_truss_service(
    api: BasetenApi,
    model_name: str,
    s3_key: str,
    config: str,
    truss_user_env: b10_types.TrussUserEnv,
    semver_bump: str = "MINOR",
    preserve_previous_prod_deployment: bool = False,
    allow_truss_download: bool = False,
    is_draft: Optional[bool] = False,
    model_id: Optional[str] = None,
    deployment_name: Optional[str] = None,
    origin: Optional[b10_types.ModelOrigin] = None,
    environment: Optional[str] = None,
    preserve_env_instance_type: bool = True,
) -> ModelVersionHandle:
    """
    Create a model in the Baseten remote.

    Args:
        api: BasetenApi instance.
        model_name: Name of the model to create.
        s3_key: S3 key of the uploaded TrussHandle.
        config: Base64 encoded JSON string of the Truss config.
        semver_bump: Semver bump type, defaults to "MINOR".
        promote: Whether to promote the model after deploy, defaults to False.
        preserve_previous_prod_deployment: Whether to scale old production deployment
            to zero.
        deployment_name: Name to apply to the created deployment. Not applied to
            development model.

    Returns:
        A Model Version handle.
    """
    if is_draft:
        model_version_json = api.create_development_model_from_truss(
            model_name,
            s3_key,
            config,
            truss_user_env,
            allow_truss_download=allow_truss_download,
            origin=origin,
        )

        return ModelVersionHandle(
            version_id=model_version_json["id"],
            model_id=model_version_json["oracle"]["id"],
            hostname=model_version_json["oracle"]["hostname"],
            instance_type_name=(
                model_version_json["instance_type"]["name"]
                if "instance_type" in model_version_json
                else None
            ),
        )

    if model_id is None:
        if environment and environment != PRODUCTION_ENVIRONMENT_NAME:
            raise ValueError(NO_ENVIRONMENTS_EXIST_ERROR_MESSAGING)

        model_version_json = api.create_model_from_truss(
            model_name,
            s3_key,
            config,
            semver_bump,
            truss_user_env,
            allow_truss_download=allow_truss_download,
            deployment_name=deployment_name,
            origin=origin,
        )

        return ModelVersionHandle(
            version_id=model_version_json["id"],
            model_id=model_version_json["oracle"]["id"],
            hostname=model_version_json["oracle"]["hostname"],
            instance_type_name=(
                model_version_json["instance_type"]["name"]
                if "instance_type" in model_version_json
                else None
            ),
        )

    try:
        model_version_json = api.create_model_version_from_truss(
            model_id,
            s3_key,
            config,
            semver_bump,
            truss_user_env,
            preserve_previous_prod_deployment=preserve_previous_prod_deployment,
            deployment_name=deployment_name,
            environment=environment,
            preserve_env_instance_type=preserve_env_instance_type,
        )
    except ApiError as e:
        if (
            e.graphql_error_code
            == BasetenApi.GraphQLErrorCodes.RESOURCE_NOT_FOUND.value
        ):
            raise ValueError(
                f"Environment `{environment}` does not exist. You can create "
                "environments in the Baseten UI."
            ) from e
        raise e

    return ModelVersionHandle(
        version_id=model_version_json["id"],
        model_id=model_id,
        hostname=model_version_json["oracle"]["hostname"],
        instance_type_name=(
            model_version_json["instance_type"]["name"]
            if "instance_type" in model_version_json
            else None
        ),
    )


def validate_truss_config_against_backend(api: BasetenApi, config: str):
    """
    Validate a truss config as well as the truss version.

    Args:
        api: BasetenApi instance
        config: Base64 encoded JSON string of the Truss config

    Returns:
        None if the config is valid, otherwise raises an error message
    """
    valid_config = api.validate_truss(config)
    if not valid_config.get("success"):
        details = json.loads(valid_config.get("details"))
        errors = details.get("errors", [])
        if errors:
            error_messages = "\n".join(textwrap.indent(error, "  ") for error in errors)
            raise ValidationError(
                f"Validation failed with the following errors:\n{error_messages}"
            )


def _build_log_query_params(
    start_time: Optional[int], end_time: Optional[int], batch_size: int
) -> Dict[str, Any]:
    """
    Build query parameters for log fetching request.

    Args:
        start_time: Start time in milliseconds since epoch
        end_time: End time in milliseconds since epoch
        batch_size: Number of logs to fetch per request

    Returns:
        Dictionary of query parameters with None values removed
    """
    query_body = {
        "start_epoch_millis": start_time,
        "end_epoch_millis": end_time,
        "limit": batch_size,
        "direction": "asc",
    }

    return {k: v for k, v in query_body.items() if v is not None}


def _handle_server_error_backoff(
    error: requests.HTTPError, job_id: str, iteration: int, batch_size: int
) -> int:
    """
    Slash the batch size in half and return the new batch size
    """
    old_batch_size = batch_size
    new_batch_size = max(batch_size // 2, MIN_BATCH_SIZE)

    logging.warning(
        f"Server error (HTTP {error.response.status_code}) for job {job_id} at iteration {iteration}. "
        f"Reducing batch size from {old_batch_size} to {new_batch_size}. Retrying..."
    )

    return new_batch_size


def _process_batch_logs(
    batch_logs: List[Any], job_id: str, iteration: int, batch_size: int
) -> Tuple[bool, Optional[int], Optional[int]]:
    """
    Process a batch of logs and determine if pagination should continue.

    Args:
        batch_logs: List of logs from the current batch
        job_id: The job ID for logging
        iteration: Current iteration number for logging
        batch_size: Expected batch size

    Returns:
        Tuple of (should_continue, next_start_time, next_end_time)
    """

    # If no logs returned, we're done
    if not batch_logs:
        logging.info(f"No logs returned for job {job_id} at iteration {iteration}")
        return False, None, None

    # If we got fewer logs than the batch size, we've reached the end
    if len(batch_logs) == 0:
        logging.info(f"Reached end of logs for job {job_id} at iteration {iteration}")
        return False, None, None

    # Timestamp returned in nanoseconds for the last log in this batch converted
    # to milliseconds to use as start for next iteration
    last_log_timestamp = int(batch_logs[-1]["timestamp"]) // NANOSECONDS_PER_MILLISECOND

    # Update start time for next iteration (add 1ms to avoid overlap)
    next_start_time_ms = last_log_timestamp + 1

    # Set end time to 2 hours from next start time, maximum time delta allowed by the API
    next_end_time_ms = next_start_time_ms + 2 * MILLISECONDS_PER_HOUR

    return True, next_start_time_ms, next_end_time_ms


class BatchedTrainingLogsFetcher:
    """
    Iterator for fetching training job logs in batches using time-based pagination.

    This iterator handles the complexity of paginating through training job logs,
    including error handling, batch size adjustment, and time window management.
    """

    def __init__(
        self,
        api: BasetenApi,
        project_id: str,
        job_id: str,
        batch_size: int = MAX_BATCH_SIZE,
    ):
        self.api = api
        self.project_id = project_id
        self.job_id = job_id
        self.batch_size = batch_size
        self.iteration = 0
        self.current_start_time = None
        self.current_end_time = None
        self._initialize_time_window()

    def _initialize_time_window(self):
        training_job = self.api.get_training_job(self.project_id, self.job_id)
        self.current_start_time = iso_to_millis(
            training_job["training_job"]["created_at"]
        )
        self.current_end_time = self.current_start_time + 2 * MILLISECONDS_PER_HOUR

    def __iter__(self):
        return self

    def __next__(self) -> List[Any]:
        if self.iteration >= MAX_ITERATIONS:
            logging.warning(
                f"Reached maximum iteration limit ({MAX_ITERATIONS}) while paginating "
                f"training job logs for project_id={self.project_id}, job_id={self.job_id}."
            )
            raise StopIteration

        query_params = _build_log_query_params(
            self.current_start_time, self.current_end_time, self.batch_size
        )

        try:
            batch_logs = self.api._fetch_log_batch(
                self.project_id, self.job_id, query_params
            )

            should_continue, next_start_time, next_end_time = _process_batch_logs(
                batch_logs, self.job_id, self.iteration, self.batch_size
            )

            if not should_continue:
                logging.info(
                    f"Completed pagination for job {self.job_id}. Total iterations: {self.iteration + 1}"
                )
                raise StopIteration

            self.current_start_time = next_start_time  # type: ignore[assignment]
            self.current_end_time = next_end_time  # type: ignore[assignment]
            self.iteration += 1

            return batch_logs

        except requests.HTTPError as e:
            if 500 <= e.response.status_code < 600:
                if self.batch_size == MIN_BATCH_SIZE:
                    logging.error(
                        "Failed to fetch all training job logs due to persistent server errors. "
                        "Please try again later or contact support if the issue persists."
                    )
                    raise StopIteration
                self.batch_size = _handle_server_error_backoff(
                    e, self.job_id, self.iteration, self.batch_size
                )
                # Retry the same iteration with reduced batch size
                return self.__next__()
            else:
                logging.error(
                    f"HTTP error fetching logs for job {self.job_id} at iteration {self.iteration}: {e}"
                )
                raise StopIteration
        except Exception as e:
            logging.error(
                f"Error fetching logs for job {self.job_id} at iteration {self.iteration}: {e}"
            )
            raise StopIteration


def get_training_job_logs_with_pagination(
    api: BasetenApi, project_id: str, job_id: str, batch_size: int = MAX_BATCH_SIZE
) -> List[Any]:
    """
    This method implements forward time-based pagination by starting from the earliest
    available log and working forward in time. It uses the timestamp of the newest log in
    each batch as the start time for the next request.

    Returns:
        List of all logs in chronological order (oldest first)
    """
    all_logs = []

    logs_iterator = BatchedTrainingLogsFetcher(api, project_id, job_id, batch_size)

    for batch_logs in logs_iterator:
        all_logs.extend(batch_logs)

    logging.info(f"Completed pagination for job {job_id}. Total logs: {len(all_logs)}")

    return all_logs
