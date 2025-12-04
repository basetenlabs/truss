import logging
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional

import requests
from pydantic import BaseModel, Field

from truss.base.custom_types import SafeModel
from truss.remote.baseten import custom_types as b10_types
from truss.remote.baseten.auth import ApiKey, AuthService
from truss.remote.baseten.custom_types import APIKeyCategory
from truss.remote.baseten.error import ApiError
from truss.remote.baseten.rest_client import RestAPIClient
from truss.remote.baseten.utils.transfer import base64_encoded_json_str

logger = logging.getLogger(__name__)
PARAMS_INDENT = "\n                    "


class ChainAWSCredential(SafeModel):
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: str


class ChainUploadCredentials(SafeModel):
    s3_bucket: str
    s3_key: str
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: str

    @property
    def aws_credentials(self) -> ChainAWSCredential:
        return ChainAWSCredential(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )


class InstanceTypeV1(BaseModel):
    """An instance type."""

    id: str = Field(description="Identifier string for the instance type")
    name: str = Field(description="Name of the instance type")
    display_name: str = Field(
        alias="displayName", description="Display name of the instance type"
    )
    gpu_count: int = Field(
        alias="gpuCount", description="Number of GPUs on the instance type"
    )
    default: bool = Field(description="Whether this is the default instance type")
    gpu_memory: Optional[int] = Field(alias="gpuMemory", description="GPU memory in MB")
    node_count: int = Field(alias="nodeCount", description="Number of nodes")
    gpu_type: Optional[str] = Field(
        alias="gpuType", description="Type of GPU on the instance type"
    )
    millicpu_limit: int = Field(
        alias="millicpuLimit", description="CPU limit of the instance type in millicpu"
    )
    memory_limit: int = Field(
        alias="memoryLimit", description="Memory limit of the instance type in MB"
    )
    price: Optional[float] = Field(description="Price of the instance type")
    limited_capacity: Optional[bool] = Field(
        alias="limitedCapacity",
        description="Whether this instance type has limited capacity",
    )

    class Config:
        populate_by_name = True


API_URL_MAPPING = {
    "https://app.baseten.co": "https://api.baseten.co",
    "https://app.staging.baseten.co": "https://api.staging.baseten.co",
    "https://app.dev.baseten.co": "https://api.mc-dev.baseten.co",
    # For local development, this is how we map URLs
    "http://localhost:8000": "http://api.localhost:8000",
}

# If a non-standard domain is used with the baseten remote, default to
# using the production api routes
DEFAULT_API_DOMAIN = "https://api.baseten.co"


def _oracle_data_to_graphql_mutation(oracle: b10_types.OracleData) -> str:
    args = [
        f'model_name: "{oracle.model_name}"',
        f's3_key: "{oracle.s3_key}"',
        f'encoded_config_str: "{oracle.encoded_config_str}"',
    ]

    if oracle.semver_bump:
        args.append(f'semver_bump: "{oracle.semver_bump}"')

    if oracle.version_name:
        args.append(f'version_name: "{oracle.version_name}"')

    args_str = ",\n".join(args)

    return f"""{{
        {args_str}
    }}"""


def _chainlet_data_atomic_to_graphql_mutation(
    chainlet: b10_types.ChainletDataAtomic,
) -> str:
    oracle_data_string = _oracle_data_to_graphql_mutation(chainlet.oracle)
    args = [f'name: "{chainlet.name}"', f"oracle: {oracle_data_string}"]
    args_str = ",\n".join(args)
    return f"""{{
        {args_str}
    }}"""


class BasetenApi:
    _rest_api_client: RestAPIClient

    class GraphQLErrorCodes(Enum):
        RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"

    def __init__(self, remote_url: str, auth_service: AuthService) -> None:
        graphql_api_url = f"{remote_url}/graphql/"
        # Ensure we strip off trailing '/' to denormalize URLs.
        rest_api_url = API_URL_MAPPING.get(remote_url.strip("/"), DEFAULT_API_DOMAIN)

        self._remote_url = remote_url
        self._graphql_api_url = graphql_api_url
        self._rest_api_url = rest_api_url
        self._auth_service = auth_service
        self._auth_token = self._auth_service.authenticate()
        self._rest_api_client = RestAPIClient(
            base_url=self._rest_api_url, headers=self._auth_token.header()
        )

    @property
    def app_url(self) -> str:
        return self._remote_url

    @property
    def rest_api_url(self) -> str:
        return self._rest_api_url

    @property
    def auth_token(self) -> ApiKey:
        return self._auth_token

    def _post_graphql_query(self, query: str, variables: Optional[dict] = None) -> dict:
        headers = self._auth_token.header()
        payload: Dict[str, Any] = {"query": query}
        if variables is not None:
            payload["variables"] = variables

        resp = requests.post(
            self._graphql_api_url, json=payload, headers=headers, timeout=120
        )

        if not resp.ok:
            logger.error(f"GraphQL endpoint failed with error: {str(resp.content)}")
            resp.raise_for_status()

        resp_dict = resp.json()
        errors = resp_dict.get("errors")

        if errors:
            message = errors[0]["message"]
            error_code = errors[0].get("extensions", {}).get("code")
            if errors[0].get("extensions", {}).get("description") is not None:
                message = errors[0].get("extensions", {}).get("description")
            raise ApiError(message, error_code)

        return resp_dict

    def model_s3_upload_credentials(self):
        query_string = """
        {
            model_s3_upload_credentials {
                s3_bucket
                s3_key
                aws_access_key_id
                aws_secret_access_key
                aws_session_token
            }
        }
        """
        resp = self._post_graphql_query(query_string)
        return resp["data"]["model_s3_upload_credentials"]

    def create_model_from_truss(
        self,
        model_name: str,
        s3_key: str,
        config: str,
        semver_bump: str,
        truss_user_env: b10_types.TrussUserEnv,
        allow_truss_download: bool = True,
        deployment_name: Optional[str] = None,
        origin: Optional[b10_types.ModelOrigin] = None,
        environment: Optional[str] = None,
        deploy_timeout_minutes: Optional[int] = None,
        team_id: Optional[str] = None,
    ):
        query_string = f"""
            mutation ($trussUserEnv: String) {{
                create_model_from_truss(
                    name: "{model_name}"
                    s3_key: "{s3_key}"
                    config: "{config}"
                    semver_bump: "{semver_bump}"
                    truss_user_env: $trussUserEnv
                    allow_truss_download: {"true" if allow_truss_download else "false"}
                    {f'version_name: "{deployment_name}"' if deployment_name else ""}
                    {f"model_origin: {origin.value}" if origin else ""}
                    {f'environment_name: "{environment}"' if environment else ""}
                    {f"deploy_timeout_minutes: {deploy_timeout_minutes}" if deploy_timeout_minutes is not None else ""}
                    {f'team_id: "{team_id}"' if team_id else ""}
                ) {{
                    model_version {{
                        id
                        oracle {{
                            id
                            name
                            hostname
                        }}
                        instance_type {{
                            name
                        }}
                    }}
                }}
            }}
        """
        resp = self._post_graphql_query(
            query_string, variables={"trussUserEnv": truss_user_env.json()}
        )
        return resp["data"]["create_model_from_truss"]["model_version"]

    def create_model_version_from_truss(
        self,
        model_id: str,
        s3_key: str,
        config: str,
        semver_bump: str,
        truss_user_env: b10_types.TrussUserEnv,
        preserve_previous_prod_deployment: bool = False,
        deployment_name: Optional[str] = None,
        environment: Optional[str] = None,
        preserve_env_instance_type: bool = True,
        deploy_timeout_minutes: Optional[int] = None,
    ):
        query_string = f"""
            mutation ($trussUserEnv: String) {{
                create_model_version_from_truss(
                    model_id: "{model_id}"
                    s3_key: "{s3_key}"
                    config: "{config}"
                    semver_bump: "{semver_bump}"
                    truss_user_env: $trussUserEnv
                    scale_down_old_production: {"false" if preserve_previous_prod_deployment else "true"}
                    preserve_env_instance_type: {"true" if preserve_env_instance_type else "false"}
                    {f'name: "{deployment_name}"' if deployment_name else ""}
                    {f'environment_name: "{environment}"' if environment else ""}
                    {f"deploy_timeout_minutes: {deploy_timeout_minutes}" if deploy_timeout_minutes is not None else ""}
                ) {{
                    model_version {{
                        id
                        oracle {{
                            id
                            name
                            hostname
                        }}
                        instance_type {{
                            name
                        }}
                    }}
                }}
            }}
        """

        resp = self._post_graphql_query(
            query_string, variables={"trussUserEnv": truss_user_env.json()}
        )
        return resp["data"]["create_model_version_from_truss"]["model_version"]

    def create_development_model_from_truss(
        self,
        model_name,
        s3_key,
        config,
        truss_user_env: b10_types.TrussUserEnv,
        allow_truss_download=True,
        origin: Optional[b10_types.ModelOrigin] = None,
        deploy_timeout_minutes: Optional[int] = None,
        team_id: Optional[str] = None,
    ):
        query_string = f"""
            mutation ($trussUserEnv: String) {{
                deploy_draft_truss(name: "{model_name}"
                    s3_key: "{s3_key}"
                    config: "{config}"
                    truss_user_env: $trussUserEnv
                    allow_truss_download: {"true" if allow_truss_download else "false"}
                    {f"model_origin: {origin.value}" if origin else ""}
                    {f"deploy_timeout_minutes: {deploy_timeout_minutes}" if deploy_timeout_minutes is not None else ""}
                    {f'team_id: "{team_id}"' if team_id else ""}
                ) {{
                    model_version {{
                        id
                        oracle {{
                            id
                            name
                            hostname
                        }}
                        instance_type {{
                            name
                        }}
                    }}
                }}
            }}
        """

        resp = self._post_graphql_query(
            query_string, variables={"trussUserEnv": truss_user_env.json()}
        )
        return resp["data"]["deploy_draft_truss"]["model_version"]

    def deploy_chain_atomic(
        self,
        entrypoint: b10_types.ChainletDataAtomic,
        dependencies: List[b10_types.ChainletDataAtomic],
        truss_user_env: b10_types.TrussUserEnv,
        chain_id: Optional[str] = None,
        chain_name: Optional[str] = None,
        environment: Optional[str] = None,
        is_draft: bool = False,
        original_source_artifact_s3_key: Optional[str] = None,
        allow_truss_download: Optional[bool] = True,
        deployment_name: Optional[str] = None,
        team_id: Optional[str] = None,
    ):
        if allow_truss_download is None:
            allow_truss_download = True
        entrypoint_str = _chainlet_data_atomic_to_graphql_mutation(entrypoint)

        dependencies_str = ", ".join(
            [
                _chainlet_data_atomic_to_graphql_mutation(dependency)
                for dependency in dependencies
            ]
        )

        params = []
        if chain_id:
            params.append(f'chain_id: "{chain_id}"')
        if chain_name:
            params.append(f'chain_name: "{chain_name}"')
        if environment:
            params.append(f'environment: "{environment}"')
        if original_source_artifact_s3_key:
            params.append(
                f'original_source_artifact_s3_key: "{original_source_artifact_s3_key}"'
            )
        if team_id:
            params.append(f'team_id: "{team_id}"')

        params.append(f"is_draft: {str(is_draft).lower()}")
        if allow_truss_download is False:
            params.append("allow_truss_download: false")
        if deployment_name:
            params.append(f'deployment_name: "{deployment_name}"')

        params_str = PARAMS_INDENT.join(params)

        query_string = f"""
            mutation ($trussUserEnv: String) {{
                deploy_chain_atomic(
                    {params_str}
                    entrypoint: {entrypoint_str}
                    dependencies: [{dependencies_str}]
                    truss_user_env: $trussUserEnv
                ) {{
                    chain_deployment {{
                        id
                        chain {{
                            id
                            hostname
                        }}
                    }}
                }}
            }}
        """

        resp = self._post_graphql_query(
            query_string, variables={"trussUserEnv": truss_user_env.json()}
        )

        return resp["data"]["deploy_chain_atomic"]

    def get_chains(self, team_id: Optional[str] = None):
        if team_id:
            query_string = f"""
            {{
                chains(team_id: "{team_id}") {{
                    id
                    name
                    team {{
                        id
                        name
                    }}
                }}
            }}
            """
        else:
            query_string = """
            {
                chains(all: true) {
                    id
                    name
                    team {
                        id
                        name
                    }
                }
            }
            """

        resp = self._post_graphql_query(query_string)
        return resp["data"]["chains"]

    def get_chain_deployments(self, chain_id: str):
        query_string = f"""
        {{
          chain(id: "{chain_id}") {{
            deployments {{
              id
              created
              is_draft
            }}
          }}
        }}
        """
        resp = self._post_graphql_query(query_string)
        return resp["data"]["chain"]["deployments"]

    def get_chainlets_by_deployment_id(self, chain_deployment_id: str):
        query_string = f"""
        {{
            chain_deployment(id:"{chain_deployment_id}") {{
                chainlets {{
                    name
                    id
                    is_entrypoint
                    oracle {{
                        id
                        name
                    }}
                    oracle_version {{
                        id
                        is_draft
                        current_model_deployment_status {{
                            status
                        }}
                    }}
                }}
                chain {{
                  id
                }}
            }}
        }}
        """
        resp = self._post_graphql_query(query_string)
        chainlets = resp["data"]["chain_deployment"]["chainlets"]
        for chainlet in chainlets:
            chainlet["chain"] = {"id": resp["data"]["chain_deployment"]["chain"]["id"]}
        return chainlets

    def delete_chain(self, chain_id: str) -> Any:
        return self._rest_api_client.delete(f"v1/chains/{chain_id}")

    def delete_chain_deployment(self, chain_id: str, chain_deployment_id: str) -> Any:
        return self._rest_api_client.delete(
            f"v1/chains/{chain_id}/deployments/{chain_deployment_id}"
        )

    def models(self):
        query_string = """
        {
            models(all: true) {
                id,
                name
                team {
                    id
                    name
                }
                versions{
                    id,
                    semver,
                    current_deployment_status,
                    is_primary,
                }
            }
        }
        """

        resp = self._post_graphql_query(query_string)
        return resp["data"]

    def get_truss_watch_state(self, model_name: str):
        query_string = f"""
        {{
            truss_watch_state(name: "{model_name}") {{
                is_container_built_from_push
                django_patch_state {{
                    current_hash
                    current_signature
                }}
                container_patch_state {{
                    current_hash
                    current_signature
                }}
            }}
        }}
        """
        resp = self._post_graphql_query(query_string)
        return resp["data"]

    def get_model(self, model_name):
        query_string = f"""
        {{
            model(name: "{model_name}") {{
                id
                name
                hostname
                team {{
                    id
                    name
                }}
                versions {{
                    id
                    semver
                    truss_hash
                    truss_signature
                    is_draft
                    is_primary
                    current_model_deployment_status {{
                        status
                    }}
                }}
            }}
        }}
        """
        resp = self._post_graphql_query(query_string)
        return resp["data"]

    def get_model_by_id(self, model_id: str):
        query_string = f"""
        {{
            model(id: "{model_id}") {{
                id
                name
                hostname
                primary_version{{
                    id
                    semver
                    truss_hash
                    truss_signature
                    is_draft
                    current_model_deployment_status {{
                        status
                    }}
                }}
            }}
          }}
        """
        resp = self._post_graphql_query(query_string)
        return resp["data"]

    def get_model_version_by_id(self, model_version_id: str):
        query_string = f"""
        {{
            model_version(id: "{model_version_id}") {{
                id
                is_draft
                truss_hash
                truss_signature
                oracle{{
                    id
                    name
                    hostname
                }}
            }}
          }}
        """
        resp = self._post_graphql_query(query_string)
        return resp["data"]

    def patch_draft_truss_two_step(self, model_name, patch_request):
        patch = base64_encoded_json_str(patch_request.to_dict())
        query_string = f"""
        mutation ($trussUserEnv: String) {{
            stage_patch_for_draft_truss(
                name: "{model_name}"
                truss_user_env: $trussUserEnv
                patch: "{patch}"
            ) {{
                id
                name
                version_id
                succeeded
                needs_full_deploy
                error
            }}
        }}
        """
        truss_user_env = b10_types.TrussUserEnv.collect().json()
        resp = self._post_graphql_query(
            query_string, variables={"trussUserEnv": truss_user_env}
        )
        result = resp["data"]["stage_patch_for_draft_truss"]
        if not result["succeeded"]:
            logging.debug(f"Failed to stage patch: {result}")
            return result
        logging.debug("Succesfully staged patch. Syncing patch to truss...")

        return self.sync_draft_truss(model_name)

    def sync_draft_truss(self, model_name):
        query_string = f"""
        mutation ($trussUserEnv: String) {{
            sync_draft_truss(
                name: "{model_name}"
                truss_user_env: $trussUserEnv
            ) {{
                id
                name
                version_id
                succeeded
                needs_full_deploy
                error
            }}
        }}
        """
        truss_user_env = b10_types.TrussUserEnv.collect().json()
        resp = self._post_graphql_query(
            query_string, variables={"trussUserEnv": truss_user_env}
        )
        result = resp["data"]["sync_draft_truss"]
        if not result["succeeded"]:
            logging.debug(f"Failed to sync patch: {result}")
        return result

    def validate_truss(self, config: str):
        query_string = f"""
        query ($trussUserEnv: String) {{
            truss_validation(
                truss_user_env: $trussUserEnv
                config: "{config}"
            ) {{
                success
                details
            }}
        }}
        """
        truss_user_env = b10_types.TrussUserEnv.collect().json()
        resp = self._post_graphql_query(
            query_string, variables={"trussUserEnv": truss_user_env}
        )
        return resp["data"]["truss_validation"]

    def get_deployment(self, model_id: str, deployment_id: str) -> Any:
        return self._rest_api_client.get(
            f"v1/models/{model_id}/deployments/{deployment_id}"
        )

    def upsert_secret(self, name: str, value: str) -> Any:
        return self._rest_api_client.post(
            "v1/secrets", body={"name": name, "value": value}
        )

    def get_all_secrets(self) -> Any:
        return self._rest_api_client.get("v1/secrets")

    # NOTE(Tyron): `name` is required because all official
    # Baseten API keys should have a descriptive name.
    def create_api_key(self, api_key_type: APIKeyCategory, name: str) -> Any:
        return self._rest_api_client.post(
            "v1/api_keys", body={"type": api_key_type.value, "name": name}
        )

    def upsert_training_project(self, training_project, team_id: Optional[str] = None):
        if team_id:
            endpoint = f"v1/teams/{team_id}/training_projects"
        else:
            endpoint = "v1/training_projects"
        resp_json = self._rest_api_client.post(
            endpoint,
            body={"training_project": training_project.model_dump(exclude_none=True)},
        )
        return resp_json["training_project"]

    def create_training_job(self, project_id: str, job):
        resp_json = self._rest_api_client.post(
            f"v1/training_projects/{project_id}/jobs",
            body={"training_job": job.model_dump()},
        )
        return resp_json["training_job"]

    def stop_training_job(self, project_id: str, job_id: str):
        resp_json = self._rest_api_client.post(
            f"v1/training_projects/{project_id}/jobs/{job_id}/stop", body={}
        )
        return resp_json["training_job"]

    def list_training_jobs(self, project_id: str):
        # training_jobs, training_project
        resp_json = self._rest_api_client.get(f"v1/training_projects/{project_id}/jobs")
        return resp_json

    def search_training_jobs(
        self,
        statuses: Optional[List[str]] = None,
        project_id: Optional[str] = None,
        job_id: Optional[str] = None,
        order_by: List[dict[str, str]] = [{"field": "created_at", "order": "desc"}],
    ):
        resp_json = self._rest_api_client.post(
            "v1/training_jobs/search",
            body={
                "statuses": statuses,
                "project_id": project_id,
                "job_id": job_id,
                "order_by": order_by,
            },
        )
        return resp_json["training_jobs"]

    def get_training_job(self, project_id: str, job_id: str):
        # training_job, training_project
        resp_json = self._rest_api_client.get(
            f"v1/training_projects/{project_id}/jobs/{job_id}"
        )
        return resp_json

    def list_training_projects(self):
        resp_json = self._rest_api_client.get("v1/training_projects")
        return resp_json["training_projects"]

    def get_blob_credentials(self, blob_type: b10_types.BlobType):
        if blob_type == b10_types.BlobType.CHAIN:
            return self.get_chain_s3_upload_credentials()
        return self._rest_api_client.get(f"v1/blobs/credentials/{blob_type.value}")

    def get_chain_s3_upload_credentials(self) -> ChainUploadCredentials:
        """Get chain artifact credentials using GraphQL query."""
        query = """
        query {
            chain_s3_upload_credentials {
                s3_bucket
                s3_key
                aws_access_key_id
                aws_secret_access_key
                aws_session_token
            }
        }
        """
        response = self._post_graphql_query(query)

        return ChainUploadCredentials.model_validate(
            response["data"]["chain_s3_upload_credentials"]
        )

    def get_training_job_metrics(
        self,
        project_id: str,
        job_id: str,
        start_epoch_millis: Optional[int] = None,
        end_epoch_millis: Optional[int] = None,
    ):
        resp_json = self._rest_api_client.post(
            f"v1/training_projects/{project_id}/jobs/{job_id}/metrics",
            body=self._prepare_time_range_query(start_epoch_millis, end_epoch_millis),
        )
        return resp_json

    def recreate_training_job(self, project_id: str, job_id: str):
        resp_json = self._rest_api_client.post(
            f"v1/training_projects/{project_id}/jobs/{job_id}/recreate", body={}
        )
        return resp_json["training_job"]

    def list_training_job_checkpoints(self, project_id: str, job_id: str):
        resp_json = self._rest_api_client.get(
            f"v1/training_projects/{project_id}/jobs/{job_id}/checkpoints"
        )
        return resp_json

    def _prepare_time_range_query(
        self,
        start_epoch_millis: Optional[int] = None,
        end_epoch_millis: Optional[int] = None,
    ) -> Mapping[str, int]:
        payload = {}
        if start_epoch_millis:
            payload["start_epoch_millis"] = start_epoch_millis
        if end_epoch_millis:
            payload["end_epoch_millis"] = end_epoch_millis
        return payload

    def get_training_job_logs(
        self,
        project_id: str,
        job_id: str,
        start_epoch_millis: Optional[int] = None,
        end_epoch_millis: Optional[int] = None,
    ):
        resp_json = self._rest_api_client.post(
            f"v1/training_projects/{project_id}/jobs/{job_id}/logs",
            body=self._prepare_time_range_query(start_epoch_millis, end_epoch_millis),
        )

        # NB(nikhil): reverse order so latest logs are at the end
        return resp_json["logs"][::-1]

    def get_cache_summary(self, project_id: str):
        """Get cache summary for a training project."""
        resp_json = self._rest_api_client.get(
            f"v1/training_projects/{project_id}/cache/summary"
        )
        return resp_json

    def _fetch_log_batch(
        self, project_id: str, job_id: str, query_params: Dict[str, Any]
    ) -> List[Any]:
        """
        Fetch a single batch of logs from the API.
        """
        resp_json = self._rest_api_client.post(
            f"v1/training_projects/{project_id}/jobs/{job_id}/logs", body=query_params
        )
        return resp_json["logs"]

    def get_training_job_checkpoint_presigned_url(
        self, project_id: str, job_id: str, page_size: int = 100
    ) -> List[Dict[str, str]]:
        all_presigned_urls = []
        page_token: Optional[str] = None
        max_iterations = 1000

        for iteration in range(max_iterations):
            url = f"v1/training_projects/{project_id}/jobs/{job_id}/checkpoint_files"
            params = {"page_size": str(page_size)}
            if page_token:
                params["page_token"] = str(page_token)

            response = self._rest_api_client.get(url, url_params=params)
            all_presigned_urls.extend(response.get("presigned_urls", []))

            page_token = response.get("next_page_token")
            if not page_token:
                break
        else:
            # If we reach here, it means we hit the max iterations without finding a next page token
            logging.error(
                f"Reached maximum iteration limit ({max_iterations}) while paginating "
                f"checkpoint files for project_id={project_id}, job_id={job_id}"
            )

        return all_presigned_urls

    def get_training_job_presigned_url(self, project_id: str, job_id: str) -> str:
        response = self._rest_api_client.get(
            f"v1/training_projects/{project_id}/jobs/{job_id}/download"
        )

        presigned_url = response["artifact_presigned_urls"][0]

        return presigned_url

    def get_from_presigned_url(self, presigned_url: str) -> bytes:
        response = requests.get(presigned_url)
        return response.content

    def get_model_deployment_logs(
        self,
        model_id: str,
        deployment_id: str,
        start_epoch_millis: Optional[int] = None,
        end_epoch_millis: Optional[int] = None,
    ):
        resp_json = self._rest_api_client.post(
            f"v1/models/{model_id}/deployments/{deployment_id}/logs",
            body=self._prepare_time_range_query(start_epoch_millis, end_epoch_millis),
        )

        # NB(nikhil): reverse order so latest logs are at the end
        return resp_json["logs"][::-1]

    def create_model_version_from_inference_template(self, request_data: dict):
        """
        Create a model version from an inference template using GraphQL mutation.

        Args:
            request_data: Dictionary containing the request structure with metadata,
                         weights_sources, inference_stack, and instance_type_id
        """
        query_string = """
        mutation ($request: CreateModelVersionFromInferenceTemplateRequest!) {
            create_model_version_from_inference_template(request: $request) {
                model_version {
                    id
                    name
                }
                truss_config
            }
        }
        """
        resp = self._post_graphql_query(
            query_string, variables={"request": request_data}
        )
        return resp["data"]["create_model_version_from_inference_template"]

    def get_instance_types(self) -> List[InstanceTypeV1]:
        """
        Get all available instance types via GraphQL API.
        """
        query_string = """
        query Instances {
            listedInstances: listed_instances {
                id
                name
                millicpuLimit: millicpu_limit
                memoryLimit: memory_limit
                gpuCount: gpu_count
                gpuType: gpu_type
                gpuMemory: gpu_memory
                default
                displayName: display_name
                nodeCount: node_count
                price
                limitedCapacity: limited_capacity
            }
        }
        """

        resp = self._post_graphql_query(query_string)
        instance_types_data = resp["data"]["listedInstances"]
        return [
            InstanceTypeV1(**instance_type) for instance_type in instance_types_data
        ]

    def get_teams(self) -> Dict[str, Dict[str, str]]:
        """
        Get all available teams via GraphQL API.
        Returns a dictionary mapping team name to team data (with 'id' and 'name' keys).
        """
        query_string = """
        query Teams {
            teams {
                id
                name
            }
        }
        """

        resp = self._post_graphql_query(query_string)
        teams_data = resp["data"]["teams"]
        # Convert list to dict mapping team_name -> team
        return {team["name"]: team for team in teams_data}
