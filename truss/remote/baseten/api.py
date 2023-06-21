import logging
from typing import Optional

import requests
from truss.remote.baseten.auth import AuthService
from truss.remote.baseten.error import ApiError

logger = logging.getLogger(__name__)


class BasetenApi:
    def __init__(self, api_url: str, auth_service: Optional[AuthService] = None):
        self._api_url = api_url
        self._auth_service = auth_service or AuthService()
        self._auth_token = self._auth_service.authenticate()

    def _post_graphql_query(self, query_string: str) -> dict:
        headers = self._auth_token.headers()
        resp = requests.post(
            self._api_url,
            data={"query": query_string},
            headers=headers,
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
        external_model_version_id=None,
    ):
        query_string = f"""
        mutation {{
        create_model_from_truss(name: "{model_name}",
                    s3_key: "{s3_key}",
                    config: "{config}",
                    semver_bump: "{semver_bump}",
                    client_version: "{client_version}",
                    is_trusted: {'true' if is_trusted else 'false'}
                    external_model_version_id: "{external_model_version_id if external_model_version_id else ''}"
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
