import logging
from enum import Enum
from typing import Any, List, Optional

import requests
from truss.remote.baseten import types as b10_types
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


def _chainlet_data_to_graphql_mutation(chainlet: b10_types.ChainletData):
    return f"""
        {{
            name: "{chainlet.name}",
            oracle_version_id: "{chainlet.oracle_version_id}",
            is_entrypoint: {'true' if chainlet.is_entrypoint else 'false'}
        }}
        """


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
    def remote_url(self) -> str:
        return self._remote_url

    @property
    def rest_api_url(self) -> str:
        return self._rest_api_url

    @property
    def auth_token(self) -> ApiKey:
        return self._auth_token

    def _post_graphql_query(self, query_string: str) -> dict:
        headers = self._auth_token.header()

        resp = requests.post(
            self._graphql_api_url,
            data={"query": query_string},
            headers=headers,
            timeout=120,
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
        client_version: str,
        is_trusted: bool,
        deployment_name: Optional[str] = None,
        origin: Optional[b10_types.ModelOrigin] = None,
    ):
        query_string = f"""
        mutation {{
            create_model_from_truss(
                name: "{model_name}",
                s3_key: "{s3_key}",
                config: "{config}",
                semver_bump: "{semver_bump}",
                client_version: "{client_version}",
                is_trusted: {'true' if is_trusted else 'false'},
                {f'version_name: "{deployment_name}"' if deployment_name else ""}
                {f'model_origin: {origin.value}' if origin else ""}
            ) {{
                id,
                name,
                version_id
            }}
        }}
        """

        resp = self._post_graphql_query(query_string)
        return resp["data"]["create_model_from_truss"]

    def create_model_version_from_truss(
        self,
        model_id: str,
        s3_key: str,
        config: str,
        semver_bump: str,
        client_version: str,
        is_trusted: bool,
        promote: bool = False,
        preserve_previous_prod_deployment: bool = False,
        deployment_name: Optional[str] = None,
    ):
        query_string = f"""
        mutation {{
            create_model_version_from_truss(
                model_id: "{model_id}"
                s3_key: "{s3_key}",
                config: "{config}",
                semver_bump: "{semver_bump}",
                client_version: "{client_version}",
                is_trusted: {'true' if is_trusted else 'false'},
                promote_after_deploy: {'true' if promote else 'false'},
                scale_down_old_production: {'false' if preserve_previous_prod_deployment else 'true'},
                {f'name: "{deployment_name}"' if deployment_name else ""}
            ) {{
                id
            }}
        }}
        """

        resp = self._post_graphql_query(query_string)
        return resp["data"]["create_model_version_from_truss"]

    def create_development_model_from_truss(
        self,
        model_name,
        s3_key,
        config,
        client_version,
        is_trusted=False,
        origin: Optional[b10_types.ModelOrigin] = None,
    ):
        query_string = f"""
        mutation {{
        deploy_draft_truss(name: "{model_name}",
                    s3_key: "{s3_key}",
                    config: "{config}",
                    client_version: "{client_version}",
                    is_trusted: {'true' if is_trusted else 'false'},
                    {f'model_origin: {origin.value}' if origin else ""}
    ) {{
            id,
            name,
            version_id
        }}
        }}
        """
        resp = self._post_graphql_query(query_string)
        return resp["data"]["deploy_draft_truss"]

    def deploy_chain(self, name: str, chainlet_data: List[b10_types.ChainletData]):
        chainlet_data_strings = [
            _chainlet_data_to_graphql_mutation(chainlet) for chainlet in chainlet_data
        ]

        chainlets_string = ", ".join(chainlet_data_strings)
        query_string = f"""
        mutation {{
        deploy_chain(
            name: "{name}",
            chainlets: [{chainlets_string}]
        ) {{
            id
            chain_id
            chain_deployment_id
        }}
        }}
        """
        resp = self._post_graphql_query(query_string)
        return resp["data"]["deploy_chain"]

    def deploy_draft_chain(
        self, name: str, chainlet_data: List[b10_types.ChainletData]
    ):
        chainlet_data_strings = [
            _chainlet_data_to_graphql_mutation(chainlet) for chainlet in chainlet_data
        ]
        chainlets_string = ", ".join(chainlet_data_strings)
        query_string = f"""
        mutation {{
        deploy_draft_chain(
            name: "{name}",
            chainlets: [{chainlets_string}]
        ) {{
            chain_id
            chain_deployment_id
        }}
        }}
        """
        resp = self._post_graphql_query(query_string)
        return resp["data"]["deploy_draft_chain"]

    def deploy_chain_deployment(
        self, chain_id: str, chainlet_data: List[b10_types.ChainletData]
    ):
        chainlet_data_strings = [
            _chainlet_data_to_graphql_mutation(chainlet) for chainlet in chainlet_data
        ]
        chainlets_string = ", ".join(chainlet_data_strings)
        query_string = f"""
        mutation {{
        deploy_chain_deployment(
            chain_id: "{chain_id}",
            chainlets: [{chainlets_string}]
        ) {{
            chain_id
            chain_deployment_id
        }}
        }}
        """
        resp = self._post_graphql_query(query_string)
        return resp["data"]["deploy_chain_deployment"]

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

    def get_chainlets_by_deployment_id(self, chain_deployment_id: str):
        query_string = f"""
        {{
            chain_deployment(id:"{chain_deployment_id}") {{
                chainlets{{
                    name
                    id
                    is_entrypoint
                    oracle_version {{
                        current_model_deployment_status {{
                            status
                        }}
                    }}
                 }}
            }}
        }}
        """
        resp = self._post_graphql_query(query_string)
        return resp["data"]["chain_deployment"]["chainlets"]

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

    def get_model(self, model_name):
        query_string = f"""
        {{
            model(name: "{model_name}") {{
                name
                id
                versions{{
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
                name
                id
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
                oracle{{
                    id
                }}
            }}
          }}
        """
        resp = self._post_graphql_query(query_string)
        return resp["data"]

    def patch_draft_truss(self, model_name, patch_request):
        patch = base64_encoded_json_str(patch_request.to_dict())
        query_string = f"""
        mutation {{
        patch_draft_truss(name: "{model_name}",
                    client_version: "TRUSS",
                    patch: "{patch}",
    ) {{
            id,
            name,
            version_id
            succeeded
            needs_full_deploy
            error
        }}
        }}
        """
        resp = self._post_graphql_query(query_string)
        return resp["data"]["patch_draft_truss"]

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
        resp = requests.get(
            f"{self._rest_api_url}/v1/secrets",
            headers=headers,
        )
        if not resp.ok:
            resp.raise_for_status()

        secrets_info = resp.json()
        return secrets_info
