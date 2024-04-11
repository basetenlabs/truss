import logging
import os
import pathlib
import shutil
from pathlib import Path
from typing import Mapping, Optional, cast

import slay
import truss
from slay import definitions
from slay.truss_adapter import model_skeleton
from truss import truss_config
from truss.contexts.image_builder import serving_image_builder
from truss.remote import remote_factory, truss_remote
from truss.remote.baseten import service as b10_service

_REQUIREMENTS_FILENAME = "pip_requirements.txt"
_MODEL_FILENAME = "model.py"
_MODEL_CLS_NAME = model_skeleton.TrussProcessorModel.__name__
_TRUSS_DIR = ".truss_gen"


def _copy_python_source_files(root_dir: pathlib.Path, dest_dir: pathlib.Path) -> None:
    """Copy all python files under root recursively, but skips generated code."""

    def python_files_only(path, names):
        return [
            name
            for name in names
            if os.path.isfile(os.path.join(path, name))
            and not name.endswith(".py")
            or definitions.GENERATED_CODE_DIR in name
        ]

    shutil.copytree(root_dir, dest_dir, ignore=python_files_only, dirs_exist_ok=True)


def _make_truss_config(
    truss_dir: pathlib.Path,
    slay_config: definitions.RemoteConfig,
    user_config: definitions.UserConfigT,
    processor_to_service: Mapping[str, definitions.ServiceDescriptor],
    model_name: str,
) -> truss_config.TrussConfig:
    """Generate a truss config for a processor."""
    config = truss_config.TrussConfig()
    config.model_name = model_name
    config.model_class_filename = _MODEL_FILENAME
    config.model_class_name = _MODEL_CLS_NAME
    # Compute.
    compute = slay_config.get_compute_spec()
    config.resources.cpu = str(compute.cpu)
    config.resources.accelerator = compute.gpu
    config.resources.use_gpu = bool(compute.gpu.count)
    # TODO: expose this setting directly.
    config.runtime.predict_concurrency = compute.cpu
    # Image.
    image = slay_config.get_docker_image_spec()
    config.base_image = truss_config.BaseImage(image=image.base_image)
    pip_requirements: list[str] = []
    if image.pip_requirements_file:
        image.pip_requirements_file.raise_if_not_exists()
        pip_requirements.extend(
            req
            for req in pathlib.Path(image.pip_requirements_file.abs_path)
            .read_text()
            .splitlines()
            if not req.strip().startswith("#")
        )
    pip_requirements.extend(image.pip_requirements)
    # `pip_requirements` will add server requirements which give version conflicts.
    # config.requirements = pip_requirements
    pip_requirements_file_path = truss_dir / _REQUIREMENTS_FILENAME
    pip_requirements_file_path.write_text("\n".join(pip_requirements))
    # TODO: apparently absolute paths don't work with remote build (but work in local).
    config.requirements_file = _REQUIREMENTS_FILENAME  # str(pip_requirements_file_path)
    config.system_packages = image.apt_requirements
    # Assets.
    assets = slay_config.get_asset_spec()
    config.secrets = assets.secrets
    if definitions.BASETEN_API_SECRET_NAME not in config.secrets:
        config.secrets[definitions.BASETEN_API_SECRET_NAME] = definitions.SECRET_DUMMY
    else:
        logging.info(
            f"Workflows automatically add {definitions.BASETEN_API_SECRET_NAME} "
            "to secrets - no need to manually add it."
        )
    config.model_cache.models = assets.cached
    # Metadata.
    slay_metadata: definitions.TrussMetadata = definitions.TrussMetadata(
        user_config=user_config, processor_to_service=processor_to_service
    )
    config.model_metadata[definitions.TRUSS_CONFIG_SLAY_KEY] = slay_metadata.dict()
    return config


def make_truss(
    processor_dir: pathlib.Path,
    workflow_root: pathlib.Path,
    slay_config: definitions.RemoteConfig,
    user_config: definitions.UserConfigT,
    model_name: str,
    processor_to_service: Mapping[str, definitions.ServiceDescriptor],
    maybe_stub_types_file: Optional[pathlib.Path],
) -> pathlib.Path:
    truss_dir = processor_dir / _TRUSS_DIR
    truss_dir.mkdir(exist_ok=True)
    config = _make_truss_config(
        truss_dir, slay_config, user_config, processor_to_service, model_name
    )
    config.write_to_yaml_file(
        truss_dir / serving_image_builder.CONFIG_FILE, verbose=True
    )
    # Copy other sources.
    model_dir = truss_dir / truss_config.DEFAULT_MODEL_MODULE_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        processor_dir / f"{definitions.PROCESSOR_MODULE}.py",
        model_dir / _MODEL_FILENAME,
    )
    pkg_dir = truss_dir / truss_config.DEFAULT_BUNDLED_PACKAGES_DIR
    pkg_dir.mkdir(parents=True, exist_ok=True)
    if maybe_stub_types_file is not None:
        shutil.copy(maybe_stub_types_file, pkg_dir)
    # TODO This assume all imports are absolute w.r.t workflow root (or site-packages).
    #   Also: apparently packages need an `__init__`, or crash.
    _copy_python_source_files(workflow_root, pkg_dir / pkg_dir)

    # TODO Truss package contains this from `{ include = "slay", from = "." }`
    #   pyproject.toml. But for quick dev loop just copy from local.
    shutil.copytree(
        os.path.dirname(slay.__file__),
        pkg_dir / "slay",
        dirs_exist_ok=True,
    )
    return truss_dir


class BasetenClient:
    """Helper to deploy models on baseten and inquire their status."""

    # TODO: use rest APIs where possible in stead of graphql_query.
    def __init__(self, baseten_url: str, baseten_api_key: str) -> None:
        remote_config = truss_remote.RemoteConfig(
            name="baseten",
            configs={
                "remote_provider": "baseten",
                "api_key": baseten_api_key,
                "remote_url": baseten_url,
            },
        )
        remote_factory.RemoteFactory.update_remote_config(remote_config)
        remote_factory.RemoteFactory.create(remote="baseten")
        self._remote_provider: truss_remote.TrussRemote = (
            remote_factory.RemoteFactory.create(remote="baseten")
        )

    def deploy_truss(
        self, truss_root: Path, publish: bool, promote: bool
    ) -> b10_service.BasetenService:
        truss_handle = truss.load(str(truss_root))
        model_name = truss_handle.spec.config.model_name
        assert model_name is not None
        logging.info(
            f"Deploying truss model to Baseten`{model_name}` "
            f"(publish={publish}, promote={promote})."
        )
        # Models must be trusted to use the API KEY secret.
        service = self._remote_provider.push(
            truss_handle,
            model_name=model_name,
            trusted=True,
            publish=publish,
            promote=promote,
        )
        if service is None:
            raise ValueError()
        return cast(b10_service.BasetenService, service)

    def get_logs_url(self, service: b10_service.BasetenService) -> str:
        return self._remote_provider.get_remote_logs_url(service)
