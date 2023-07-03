import logging

import requests
from truss.remote.baseten.auth import AuthService
from truss.remote.baseten.error import ApiError

logger = logging.getLogger(__name__)


class BasetenApi:
    """
    A client for the Baseten API.

    Args:
        api_url: The URL of the Baseten API.
        auth_service: An AuthService instance.
    """

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
            raise ApiError(errors[0]["message"], resp)
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
        model_name,
        s3_key,
        config,
        semver_bump,
        client_version,
        is_trusted=False,
    ):
        query_string = f"""
        mutation {{
        create_model_from_truss(name: "{model_name}",
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
