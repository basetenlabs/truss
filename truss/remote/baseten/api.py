import logging
from enum import Enum
from typing import Any, Dict, List, Optional

import requests

from truss.remote.baseten import custom_types as b10_types
from truss.remote.baseten.auth import ApiKey, AuthService
from truss.remote.baseten.error import ApiError
from truss.remote.baseten.utils.transfer import base64_encoded_json_str

logger = logging.getLogger(__name__)


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

TRUSS_USER_ENV = b10_types.TrussUserEnv.collect().json()


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
        allow_truss_download: bool = True,
        deployment_name: Optional[str] = None,
        origin: Optional[b10_types.ModelOrigin] = None,
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
                ) {{
                    model_version {{
                        id
                        oracle {{
                            id
                            name
                            hostname
                        }}
                    }}
                }}
            }}
        """
        resp = self._post_graphql_query(
            query_string, variables={"trussUserEnv": TRUSS_USER_ENV}
        )
        return resp["data"]["create_model_from_truss"]["model_version"]

    def create_model_version_from_truss(
        self,
        model_id: str,
        s3_key: str,
        config: str,
        semver_bump: str,
        preserve_previous_prod_deployment: bool = False,
        deployment_name: Optional[str] = None,
        environment: Optional[str] = None,
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
                    {f'name: "{deployment_name}"' if deployment_name else ""}
                    {f'environment_name: "{environment}"' if environment else ""}
                ) {{
                    model_version {{
                        id
                        oracle {{
                            hostname
                        }}
                    }}
                }}
            }}
        """

        resp = self._post_graphql_query(
            query_string, variables={"trussUserEnv": TRUSS_USER_ENV}
        )
        return resp["data"]["create_model_version_from_truss"]["model_version"]

    def create_development_model_from_truss(
        self,
        model_name,
        s3_key,
        config,
        allow_truss_download=True,
        origin: Optional[b10_types.ModelOrigin] = None,
    ):
        query_string = f"""
            mutation ($trussUserEnv: String) {{
                deploy_draft_truss(name: "{model_name}"
                    s3_key: "{s3_key}"
                    config: "{config}"
                    truss_user_env: $trussUserEnv
                    allow_truss_download: {"true" if allow_truss_download else "false"}
                    {f"model_origin: {origin.value}" if origin else ""}
                ) {{
                    model_version {{
                        id
                        oracle {{
                            id
                            name
                            hostname
                        }}
                    }}
                }}
            }}
        """

        resp = self._post_graphql_query(
            query_string, variables={"trussUserEnv": TRUSS_USER_ENV}
        )
        return resp["data"]["deploy_draft_truss"]["model_version"]

    def deploy_chain_atomic(
        self,
        entrypoint: b10_types.ChainletDataAtomic,
        dependencies: List[b10_types.ChainletDataAtomic],
        chain_id: Optional[str] = None,
        chain_name: Optional[str] = None,
        environment: Optional[str] = None,
        is_draft: bool = False,
    ):
        entrypoint_str = _chainlet_data_atomic_to_graphql_mutation(entrypoint)

        dependencies_str = ", ".join(
            [
                _chainlet_data_atomic_to_graphql_mutation(dependency)
                for dependency in dependencies
            ]
        )

        query_string = f"""
            mutation ($trussUserEnv: String) {{
                deploy_chain_atomic(
                    {f'chain_id: "{chain_id}"' if chain_id else ""}
                    {f'chain_name: "{chain_name}"' if chain_name else ""}
                    {f'environment: "{environment}"' if environment else ""}
                    is_draft: {str(is_draft).lower()}
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
            query_string, variables={"trussUserEnv": TRUSS_USER_ENV}
        )

        return resp["data"]["deploy_chain_atomic"]

    def get_chains(self):
        query_string = """
        {
            chains {
                id
                name
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
        url = f"{self._rest_api_url}/v1/chains/{chain_id}"
        headers = self._auth_token.header()
        resp = requests.delete(url, headers=headers)
        if not resp.ok:
            resp.raise_for_status()

        deployment = resp.json()
        return deployment

    def delete_chain_deployment(self, chain_id: str, chain_deployment_id: str) -> Any:
        url = f"{self._rest_api_url}/v1/chains/{chain_id}/deployments/{chain_deployment_id}"
        headers = self._auth_token.header()
        resp = requests.delete(url, headers=headers)
        if not resp.ok:
            resp.raise_for_status()

        deployment = resp.json()
        return deployment

    def models(self):
        query_string = """
        {
            models {
                id,
                name
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
        resp = self._post_graphql_query(
            query_string, variables={"trussUserEnv": TRUSS_USER_ENV}
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
        resp = self._post_graphql_query(
            query_string, variables={"trussUserEnv": TRUSS_USER_ENV}
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
        resp = self._post_graphql_query(
            query_string, variables={"trussUserEnv": TRUSS_USER_ENV}
        )
        return resp["data"]["truss_validation"]

    def get_deployment(self, model_id: str, deployment_id: str) -> Any:
        headers = self._auth_token.header()
        resp = requests.get(
            f"{self._rest_api_url}/v1/models/{model_id}/deployments/{deployment_id}",
            headers=headers,
        )
        if not resp.ok:
            resp.raise_for_status()

        deployment = resp.json()
        return deployment

    def upsert_secret(self, name: str, value: str) -> Any:
        headers = self._auth_token.header()
        data = {"name": name, "value": value}
        resp = requests.post(
            f"{self._rest_api_url}/v1/secrets", headers=headers, json=data
        )
        if not resp.ok:
            resp.raise_for_status()

        secret_info = resp.json()
        return secret_info

    def get_all_secrets(self) -> Any:
        headers = self._auth_token.header()
        resp = requests.get(f"{self._rest_api_url}/v1/secrets", headers=headers)
        if not resp.ok:
            resp.raise_for_status()

        secrets_info = resp.json()
        return secrets_info

    def upsert_training_project(self, training_project):
        headers = self._auth_token.header()
        resp = requests.post(
            f"{self._rest_api_url}/v1/training-projects",
            headers=headers,
            json={"training_project": training_project.model_dump()},
        )
        if not resp.ok:
            resp.raise_for_status()

        return resp.json()

    def create_training_job(self, project_id: str, job):
        headers = self._auth_token.header()
        resp = requests.post(
            f"{self._rest_api_url}/v1/training-projects/{project_id}/jobs",
            headers=headers,
            json={"training_job": job.model_dump()},
        )
        if not resp.ok:
            resp.raise_for_status()

        return resp.json()
