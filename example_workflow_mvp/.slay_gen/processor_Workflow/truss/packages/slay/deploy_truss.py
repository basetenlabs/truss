import enum
import logging
from pathlib import Path
from typing import Optional, cast

import requests
import truss
from slay import definitions, utils
from truss.remote import remote_factory, truss_remote
from truss.remote.baseten import service as b10_service


def get_api_key_from_truss_config() -> str:
    return remote_factory.load_config().get("baseten", "api_key")


class _BasetenEnv(enum.Enum):
    LOCAL = "local"
    STAGING = "staging"
    PROD = "prod"
    DEV = "dev"


def _infer_env(baseten_url: str) -> _BasetenEnv:
    if baseten_url in {"localhost", "127.0.0.1", "0.0.0.0"}:
        return _BasetenEnv.LOCAL

    if "staging" in baseten_url:
        return _BasetenEnv.STAGING

    if "dev" in baseten_url:
        return _BasetenEnv.DEV

    return _BasetenEnv.PROD


def _model_url(baseten_env: _BasetenEnv, model_id: str) -> str:
    if baseten_env == _BasetenEnv.LOCAL:
        return f"http://localhost:8000/models/{model_id}"

    if baseten_env == _BasetenEnv.STAGING:
        return f"https://app.staging.baseten.co/models/{model_id}"

    if baseten_env == _BasetenEnv.DEV:
        return f"https://app.dev.baseten.co/models/{model_id}"

    return f"https://model-{model_id}.api.baseten.co/production"


class _ConditionStatus(enum.Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    NOT_DONE = "NOT_DONE"


class BasetenClient:
    def __init__(self, baseten_url: str, baseten_api_key: str) -> None:
        self._baseten_url = baseten_url
        self._baseten_env = _infer_env(baseten_url)
        self._baseten_api_key = baseten_api_key
        self._remote_provider: truss_remote.TrussRemote = self._create_remote_provider()

    def deploy_truss(
        self, truss_root: Path, model_name: str
    ) -> definitions.BasetenRemoteDescriptor:
        tr = truss.load(str(truss_root))

        logging.info(f"Deploying model `{model_name}`.")
        service = self._remote_provider.push(
            tr, model_name=model_name, trusted=True, publish=False
        )
        if service is None:
            raise ValueError()
        service = cast(b10_service.BasetenService, service)

        model_service = definitions.BasetenRemoteDescriptor(
            b10_model_id=service.model_id,
            b10_model_version_id=service.model_version_id,
            b10_model_name=model_name,
            b10_model_url=_model_url(self._baseten_env, service.model_id),
        )
        return model_service

    def get_model(
        self, model_name: str
    ) -> Optional[definitions.BasetenRemoteDescriptor]:
        query_string = f"""
        {{
        model_version(name: "{model_name}") {{
            oracle{{
                id
                name
                versions{{
                    id
                    semver
                    current_deployment_status
                    truss_hash
                    truss_signature
                }}
            }}
        }}
        }}
        """
        try:
            resp = self._post_graphql_query(query_string, retries=True)["data"][
                "model_version"
            ]["oracle"]
        except Exception as e:
            return None

        model_id = resp["id"]
        model_version_id = resp["versions"][0]["id"]
        return definitions.BasetenRemoteDescriptor(
            b10_model_id=model_id,
            b10_model_version_id=model_version_id,
            b10_model_url=_model_url(self._baseten_env, model_id),
            b10_model_name=model_name,
        )

    def _create_remote_provider(self):
        remote_config = truss_remote.RemoteConfig(
            name="baseten",
            configs={
                "remote_provider": "baseten",
                "api_key": self._baseten_api_key,
                "remote_url": self._baseten_url,
            },
        )
        remote_factory.RemoteFactory.update_remote_config(remote_config)
        return remote_factory.RemoteFactory.create(remote="baseten")

    def _wait_for_model_to_be_ready(self, model_version_id: str):
        logging.info(f"Waiting for model {model_version_id} to be ready")

        def is_model_ready() -> _ConditionStatus:
            query_string = f"""
            {{
                model_version(id: "{model_version_id}") {{
                    current_model_deployment_status {{
                        status
                        reason
                    }}
                }}
            }}
            """
            resp = self._post_graphql_query(query_string, retries=True)
            status = resp["data"]["model_version"]["current_model_deployment_status"][
                "status"
            ]
            logging.info(f"Model status: {status}")
            if status == "MODEL_READY":
                return _ConditionStatus.SUCCESS
            if "FAILED" in status:
                return _ConditionStatus.FAILURE
            return _ConditionStatus.NOT_DONE

        is_ready = utils.wait_for_condition(is_model_ready, 1800)
        if not is_ready:
            raise RuntimeError("Model failed to be ready in 30 minutes")

    def _post_graphql_query(self, query_string: str, retries: bool = False) -> dict:
        headers = {"Authorization": f"Api-Key {self._baseten_api_key}"}
        while True:
            resp = requests.post(
                f"{self._baseten_url}/graphql/",
                data={"query": query_string},
                headers=headers,
                timeout=120,
            )
            if not resp.ok:
                if not retries:
                    logging.error(f"GraphQL endpoint failed with error: {resp.content}")
                    resp.raise_for_status()
                else:
                    logging.info(
                        f"GraphQL endpoint failed with error: {resp.content}, "
                        "retries are on, ignore"
                    )
            else:
                resp_dict = resp.json()
                errors = resp_dict.get("errors")
                if errors:
                    raise RuntimeError(errors[0]["message"], resp)
                return resp_dict
