import logging
from enum import Enum

import requests
from truss.remote.baseten.auth import AuthService
from truss.remote.baseten.error import ApiError
from truss.remote.baseten.utils.transfer import base64_encoded_json_str

logger = logging.getLogger(__name__)


class BasetenApi:
    """
    A client for the Baseten API.

    Args:
        api_url: The URL of the Baseten API.
        auth_service: An AuthService instance.
    """

    class GraphQLErrorCodes(Enum):
        RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"

    def __init__(self, api_url: str, auth_service: AuthService):
        self._api_url = api_url
        self._auth_service = auth_service
        self._auth_token = self._auth_service.authenticate()

    def _post_graphql_query(self, query_string: str) -> dict:
        headers = self._auth_token.header()
        resp = requests.post(
            self._api_url,
            data={"query": query_string},
            headers=headers,
            timeout=120,
        )

        if not resp.ok:
            logger.error(f"GraphQL endpoint failed with error: {resp.content}")  # type: ignore
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
    ):
        query_string = f"""
        mutation {{
            create_model_from_truss(
                name: "{model_name}",
                s3_key: "{s3_key}",
                config: "{config}",
                semver_bump: "{semver_bump}",
                client_version: "{client_version}",
                is_trusted: {'true' if is_trusted else 'false'}
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
    ):
        query_string = f"""
        mutation {{
            create_model_version_from_truss(
                model_id: "{model_id}"
                s3_key: "{s3_key}",
                config: "{config}",
                semver_bump: "{semver_bump}",
                client_version: "{client_version}",
                is_trusted: {'true' if is_trusted else 'false'}
                promote_after_deploy: {'true' if promote else 'false'}
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
    ):
        query_string = f"""
        mutation {{
        deploy_draft_truss(name: "{model_name}",
                    s3_key: "{s3_key}",
                    config: "{config}",
                    client_version: "{client_version}",
                    is_trusted: {'true' if is_trusted else 'false'}
    ) {{
            id,
            name,
            version_id
        }}
        }}
        """
        resp = self._post_graphql_query(query_string)
        return resp["data"]["deploy_draft_truss"]

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
