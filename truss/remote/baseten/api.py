import logging

import requests
from truss.remote.baseten.auth import AuthToken, with_api_key
from truss.remote.baseten.error import ApiError

logger = logging.getLogger(__name__)


def _post_graphql_query(auth_token: AuthToken, query_string: str) -> dict:
    headers = auth_token.headers()

    resp = requests.post(
        # TODO(Abu): Make the URL configurable
        "https://app.baseten.co/graphql/",
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


@with_api_key
def model_s3_upload_credentials(api_key):
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
    resp = _post_graphql_query(api_key, query_string)
    return resp["data"]["model_s3_upload_credentials"]


@with_api_key
def create_model_from_truss(
    api_key,
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
    resp = _post_graphql_query(api_key, query_string)
    return resp["data"]["create_model_from_truss"]


@with_api_key
def models(auth_token: AuthToken):
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

    resp = _post_graphql_query(auth_token, query_string)
    return resp["data"]
