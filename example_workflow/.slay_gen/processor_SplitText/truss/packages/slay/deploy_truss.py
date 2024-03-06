import enum
import logging
import time
from pathlib import Path
from typing import Callable

import truss
from slay import definitions
from truss.remote.remote_factory import RemoteFactory
from truss.remote.truss_remote import RemoteConfig, TrussRemote


class BasetenEnv(enum.Enum):
    LOCAL = "local"
    STAGING = "staging"
    PROD = "prod"
    DEV = "dev"


def infer_env(baseten_url: str) -> BasetenEnv:
    if _at_least_one_in({"localhost", "127.0.0.1", "0.0.0.0"}, baseten_url):
        return BasetenEnv.LOCAL

    if "staging" in baseten_url:
        return BasetenEnv.STAGING

    if "dev" in baseten_url:
        return BasetenEnv.DEV

    return BasetenEnv.PROD


def _at_least_one_in(needles: set[str], target: str) -> bool:
    for needle in needles:
        if needle in target:
            return True
    return False


def predict_url(baseten_env: BasetenEnv, model_id: str) -> str:
    if baseten_env == BasetenEnv.LOCAL:
        return f"http://localhost:8000/models/{model_id}/predict"

    if baseten_env == BasetenEnv.STAGING:
        return f"https://app.staging.baseten.co/models/{model_id}/predict"

    if baseten_env == BasetenEnv.DEV:
        return f"https://app.dev.baseten.co/models/{model_id}/predict"

    return f"https://model-{model_id}.api.baseten.co/production/predict"


class ConditionStatus(enum.Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    NOT_DONE = "NOT_DONE"


def wait_for_condition(
    condition: Callable[[], ConditionStatus],
    retries: int = 10,
    sleep_between_retries_secs: int = 1,
) -> bool:
    for _ in range(retries):
        cond_status = condition()
        if cond_status == ConditionStatus.SUCCESS:
            return True
        if cond_status == ConditionStatus.FAILURE:
            return False
        time.sleep(sleep_between_retries_secs)
    return False


class BasetenClient:
    def __init__(self, baseten_url: str, baseten_api_key: str) -> None:
        self._baseten_url = baseten_url
        self._baseten_env = infer_env(baseten_url)
        self._baseten_api_key = baseten_api_key
        self._remote_provider: TrussRemote = self._create_remote_provider()

    def ensure_deployed_truss(
        self, truss_root: Path, model_name: str
    ) -> definitions.BasetenRemoteDescriptor:
        tr = truss.load(str(truss_root))

        logging.info(f"Creating new deployment for model `{model_name}`.")
        service = self._remote_provider.push(
            tr, model_name=model_name, trusted=True, publish=True
        )
        model_service = definitions.BasetenRemoteDescriptor(
            b10_model_id=service.model_id,
            b10_model_version_id=service.model_version_id,
            b10_model_name=model_name,
            b10_predict_url=predict_url(self._baseten_env, service.model_id),
        )
        # self._wait_for_model_to_be_ready(model_service.model_version_id)
        return model_service

    # def get_model(self, model_name: str) -> Optional[definitions.BasetenRemoteDescriptor]:
    #     query_string = f"""
    #     {{
    #     model_version(name: "{model_name}") {{
    #         oracle{{
    #             id
    #             name
    #             versions{{
    #                 id
    #                 semver
    #                 current_deployment_status
    #                 truss_hash
    #                 truss_signature
    #             }}
    #         }}
    #     }}
    #     }}
    #     """
    #     try:
    #         resp = self._post_graphql_query(query_string, retries=True)["data"][
    #             "model_version"
    #         ]["oracle"]
    #     except Exception as e:
    #         return None

    #     model_id = resp["id"]
    #     model_version_id = resp["versions"][0]["id"]
    #     return definitions.BasetenRemoteDescriptor(
    #         model_id=model_id,
    #         model_version_id=model_version_id,
    #         predict_url=predict_url(self._baseten_env, model_id),
    #         model_name=model_name,
    #     )

    def _create_remote_provider(self):
        remote_config = RemoteConfig(
            name="baseten",
            configs={
                "remote_provider": "baseten",
                "api_key": self._baseten_api_key,
                "remote_url": self._baseten_url,
            },
        )
        RemoteFactory.update_remote_config(remote_config)
        return RemoteFactory.create(remote="baseten")

    # def _wait_for_model_to_be_ready(self, model_version_id: str):
    #     logging.info(f"Waiting for model {model_version_id} to be ready")

    #     def is_model_ready() -> ConditionStatus:
    #         query_string = f"""
    #         {{
    #             model_version(id: "{model_version_id}") {{
    #                 current_model_deployment_status {{
    #                     status
    #                     reason
    #                 }}
    #             }}
    #         }}
    #         """
    #         resp = self._post_graphql_query(query_string, retries=True)
    #         status = resp["data"]["model_version"]["current_model_deployment_status"][
    #             "status"
    #         ]
    #         logging.info(f"Model status: {status}")
    #         if status == "MODEL_READY":
    #             return ConditionStatus.SUCCESS
    #         if "FAILED" in status:
    #             return ConditionStatus.FAILURE
    #         return ConditionStatus.NOT_DONE

    #     is_ready = wait_for_condition(is_model_ready, 1800)
    #     if not is_ready:
    #         raise RuntimeError("Model failed to be ready in 30 minutes")

    # def _post_graphql_query(self, query_string: str, retries: bool = False) -> dict:
    #     headers = {"Authorization": f"Api-Key {self._baseten_api_key}"}
    #     while True:
    #         resp = requests.post(
    #             f"{self._baseten_url}/graphql/",
    #             data={"query": query_string},
    #             headers=headers,
    #             timeout=120,
    #         )
    #         if not resp.ok:
    #             if not retries:
    #                 logging.error(f"GraphQL endpoint failed with error: {resp.content}")  # type: ignore
    #                 resp.raise_for_status()
    #             else:
    #                 logging.info(f"GraphQL endpoint failed with error: {resp.content}, retries are on, ignore")  # type: ignore
    #         else:
    #             resp_dict = resp.json()
    #             errors = resp_dict.get("errors")
    #             if errors:
    #                 raise RuntimeError(errors[0]["message"], resp)
    #             return resp_dict
