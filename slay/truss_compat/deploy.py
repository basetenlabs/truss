import enum
import logging
import pathlib
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, cast

import httpx
import requests
import truss
from slay import code_gen, definitions, utils
from slay.utils import ConditionStatus
from truss import truss_config
from truss.remote import remote_factory, truss_remote
from truss.remote.baseten import service as b10_service


def _make_truss_config(
    slay_config: definitions.Config,
    stub_cls_to_url: Mapping[str, str],
    fallback_name: str,
) -> truss_config.TrussConfig:
    config = truss_config.TrussConfig()
    config.model_name = slay_config.name or fallback_name
    # Compute.
    compute = slay_config.get_compute_spec()
    config.resources.cpu = compute.cpu
    config.resources.accelerator = truss_config.AcceleratorSpec.from_str(compute.gpu)
    config.resources.use_gpu = bool(compute.gpu)

    # Image.
    image = slay_config.get_image_spec()
    config.base_image = truss_config.BaseImage(image=image.base_image)
    # config.python_version = image.python_version

    pip_requirements = []
    if image.pip_requirements_file:
        pip_requirements.extend(
            req
            for req in pathlib.Path(image.pip_requirements_file)
            .read_text()
            .splitlines()
            if not req.strip().startswith("#")
        )
    pip_requirements.extend(image.pip_requirements)
    config.requirements = pip_requirements

    # config.requirements = image.pip_requirements
    # config.requirements_file = image.pip_requirements_file

    config.system_packages = image.apt_requirements

    # Assets.
    assets = slay_config.get_asset_spec()
    config.secrets = assets.secrets
    if definitions.BASTEN_APY_SECRET_NAME not in config.secrets:
        config.secrets[definitions.BASTEN_APY_SECRET_NAME] = "***"
    else:
        logging.info(
            f"Workflows automatically add {definitions.BASTEN_APY_SECRET_NAME} "
            "to secrets - no need to manually add it."
        )
    config.model_cache.models = assets.cached

    # Metadata.
    slay_metadata = definitions.TrussMetadata(
        user_config=slay_config.user_config,
        stub_cls_to_url=stub_cls_to_url,
    )
    config.model_metadata[definitions.TRUSS_CONFIG_SLAY_KEY] = slay_metadata.dict()

    return config


def make_truss(
    processor_dir: pathlib.Path,
    processor_desrciptor: definitions.ProcessorAPIDescriptor,
    stub_cls_to_url: Mapping[str, str],
) -> pathlib.Path:
    # TODO: Handle if model uses truss config instead of `defautl_config`.
    # TODO: Handle file-based overrides when deploying.
    config = _make_truss_config(
        processor_desrciptor.processor_cls.default_config,
        stub_cls_to_url,
        fallback_name=processor_desrciptor.cls_name,
    )

    truss_dir = processor_dir / "truss"
    truss_dir.mkdir(exist_ok=True)
    config.write_to_yaml_file(truss_dir / "config.yaml", verbose=False)

    # Copy other sources.
    model_dir = truss_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        processor_dir / f"{code_gen.PROCESSOR_MODULE}.py", model_dir / "model.py"
    )

    pkg_dir = truss_dir / "packages"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    for module in processor_dir.glob("*.py"):
        if module.name != "model.py":
            shutil.copy(module, pkg_dir)

    # TODO This dependency should be handled automatically.
    # Also: apparently packages need an `__init__`, or crash.
    shutil.copytree(
        "/home/marius-baseten/workbench/truss/example_workflow/user_package",
        pkg_dir / "user_package",
        dirs_exist_ok=True,
    )

    # TODO This should be included in truss or a lib, not manually copied.
    shutil.copytree(
        "/home/marius-baseten/workbench/truss/slay",
        pkg_dir / "slay",
        dirs_exist_ok=True,
    )

    return truss_dir


def get_api_key_from_truss_config() -> str:
    return remote_factory.load_config().get("baseten", "api_key")


class _BasetenEnv(enum.Enum):
    LOCAL = enum.auto()
    STAGING = enum.auto()
    PROD = enum.auto()
    DEV = enum.auto()


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


class BasetenClient:
    def __init__(self, baseten_url: str, baseten_api_key: str) -> None:
        self._baseten_url = baseten_url
        self._baseten_env = _infer_env(baseten_url)
        self._baseten_api_key = baseten_api_key
        self._remote_provider: truss_remote.TrussRemote = self._create_remote_provider()

    def deploy_truss(self, truss_root: Path) -> definitions.BasetenRemoteDescriptor:
        tr = truss.load(str(truss_root))
        model_name = tr.spec.config.model_name
        assert model_name is not None

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

    def get_model(self, model_name: str) -> definitions.BasetenRemoteDescriptor:
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
            raise definitions.MissingDependencyError("Model cout not be found.") from e

        model_id = resp["id"]
        model_version_id = resp["versions"][0]["id"]
        return definitions.BasetenRemoteDescriptor(
            b10_model_id=model_id,
            b10_model_version_id=model_version_id,
            b10_model_url=_model_url(self._baseten_env, model_id),
            b10_model_name=model_name,
        )

    def _create_remote_provider(self) -> truss_remote.TrussRemote:
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

    def _wait_for_model_to_be_ready(self, model_version_id: str) -> None:
        logging.info(f"Waiting for model {model_version_id} to be ready")

        def is_model_ready() -> ConditionStatus:
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
                return ConditionStatus.SUCCESS
            if "FAILED" in status:
                return ConditionStatus.FAILURE
            return ConditionStatus.NOT_DONE

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
                    logging.error(
                        f"GraphQL endpoint failed with error: {resp.content.decode()}"
                    )
                    resp.raise_for_status()
                else:
                    logging.info(
                        f"GraphQL endpoint failed with error: {resp.content.decode()}, "
                        "retries are on, ignore"
                    )
            else:
                resp_dict = resp.json()
                errors = resp_dict.get("errors")
                if errors:
                    raise RuntimeError(errors[0]["message"], resp)
                return resp_dict


def call_workflow_dbg(
    remote: definitions.BasetenRemoteDescriptor,
    payload: Any,
    max_retries: int = 100,
    retry_wait_sec: int = 3,
) -> httpx.Response:
    api_key = get_api_key_from_truss_config()
    session = httpx.Client(
        base_url=remote.b10_model_url, headers={"Authorization": f"Api-Key {api_key}"}
    )
    for _ in range(max_retries):
        try:
            response = session.post(definitions.PREDICT_ENDPOINT, json=payload)
            return response
        except Exception:
            time.sleep(retry_wait_sec)
    raise
