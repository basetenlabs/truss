import collections
import inspect
import logging
import os
import pathlib
import shutil
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Optional,
    Type,
    cast,
)

import truss
from truss import truss_config
from truss.contexts.image_builder import serving_image_builder
from truss.remote import remote_cli, remote_factory
from truss.remote.baseten import service as b10_service
from truss_chains import code_gen, definitions, framework, model_skeleton, utils

_REQUIREMENTS_FILENAME = "pip_requirements.txt"
_MODEL_FILENAME = "model.py"
_MODEL_CLS_NAME = model_skeleton.TrussChainletModel.__name__
_TRUSS_DIR = ".truss_gen"


# Truss Gen ############################################################################


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
    chains_config: definitions.RemoteConfig,
    user_config: definitions.UserConfigT,
    chainlet_to_service: Mapping[str, definitions.ServiceDescriptor],
    model_name: str,
) -> truss_config.TrussConfig:
    """Generate a truss config for a Chainlet."""
    config = truss_config.TrussConfig()
    config.model_name = model_name
    config.model_class_filename = _MODEL_FILENAME
    config.model_class_name = _MODEL_CLS_NAME
    # Compute.
    compute = chains_config.get_compute_spec()
    config.resources.cpu = str(compute.cpu_count)
    config.resources.accelerator = compute.accelerator
    config.resources.use_gpu = bool(compute.accelerator.count)
    # TODO: expose this setting directly.
    config.runtime.predict_concurrency = compute.cpu_count
    # Image.
    image = chains_config.docker_image
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
    assets = chains_config.get_asset_spec()
    config.secrets = assets.secrets
    if definitions.BASETEN_API_SECRET_NAME not in config.secrets:
        config.secrets[definitions.BASETEN_API_SECRET_NAME] = definitions.SECRET_DUMMY
    else:
        logging.info(
            f"Chains automatically add {definitions.BASETEN_API_SECRET_NAME} "
            "to secrets - no need to manually add it."
        )
    config.model_cache.models = assets.cached
    # Metadata.
    chains_metadata: definitions.TrussMetadata = definitions.TrussMetadata(
        user_config=user_config, chainlet_to_service=chainlet_to_service
    )
    config.model_metadata[definitions.TRUSS_CONFIG_CHAINS_KEY] = chains_metadata.dict()
    return config


def make_truss(
    chainlet_dir: pathlib.Path,
    chain_root: pathlib.Path,
    chains_config: definitions.RemoteConfig,
    user_config: definitions.UserConfigT,
    model_name: str,
    chainlet_to_service: Mapping[str, definitions.ServiceDescriptor],
    maybe_stub_types_file: Optional[pathlib.Path],
) -> pathlib.Path:
    truss_dir = chainlet_dir / _TRUSS_DIR
    truss_dir.mkdir(exist_ok=True)
    config = _make_truss_config(
        truss_dir, chains_config, user_config, chainlet_to_service, model_name
    )
    config.write_to_yaml_file(
        truss_dir / serving_image_builder.CONFIG_FILE, verbose=True
    )
    # Copy other sources.
    model_dir = truss_dir / truss_config.DEFAULT_MODEL_MODULE_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        chainlet_dir / f"{definitions.chainlet_MODULE}.py",
        model_dir / _MODEL_FILENAME,
    )
    pkg_dir = truss_dir / truss_config.DEFAULT_BUNDLED_PACKAGES_DIR
    pkg_dir.mkdir(parents=True, exist_ok=True)
    if maybe_stub_types_file is not None:
        shutil.copy(maybe_stub_types_file, pkg_dir)
    # TODO This assume all imports are absolute w.r.t chain root (or site-packages).
    #   Also: apparently packages need an `__init__`, or crash.
    _copy_python_source_files(chain_root, pkg_dir / pkg_dir)

    # TODO Truss package contains this from `{ include = "truss_chains", from = "." }`
    #   pyproject.toml. But for quick dev loop just copy from local.
    # shutil.copytree(
    #     os.path.dirname(chains.__file__),
    #     pkg_dir / "truss_chains",
    #     dirs_exist_ok=True,
    # )
    # Data resources.
    if chains_config.docker_image.data_dir:
        data_dir = truss_dir / truss_config.DEFAULT_DATA_DIRECTORY
        data_dir.mkdir(parents=True, exist_ok=True)
        user_data_dir = chains_config.docker_image.data_dir.abs_path
        shutil.copytree(user_data_dir, data_dir, dirs_exist_ok=True)

    return truss_dir


########################################################################################


class DeploymentOptions(definitions.SafeModelNonSerializable):
    chain_name: str
    only_generate_trusses: bool = False


class DeploymentOptionsBaseten(DeploymentOptions):
    remote_provider: remote_factory.TrussRemote
    publish: bool
    promote: bool

    @classmethod
    def create(
        cls,
        chain_name: str,
        publish: bool,
        promote: bool,
        only_generate_trusses: bool,
        remote: Optional[str] = None,
    ) -> "DeploymentOptionsBaseten":
        if not remote:
            remote = remote_cli.inquire_remote_name(
                remote_factory.RemoteFactory.get_available_config_names()
            )
        return DeploymentOptionsBaseten(
            chain_name=chain_name,
            remote_provider=remote_factory.RemoteFactory.create(remote=remote),
            publish=publish,
            promote=promote,
            only_generate_trusses=only_generate_trusses,
        )


class DeploymentOptionsLocalDocker(DeploymentOptions):
    # Local docker-to-docker requests don't need auth, but we need to set a
    # value different from `SECRET_DUMMY` to not trigger the check that the secret
    # is unset. Additionally, if local docker containers make calls to models deployed
    # on baseten, a real API key must be provided (i.e. the default must be overridden).
    baseten_chain_api_key: str = "docker_dummy_key"


def _deploy_to_baseten(
    truss_dir: pathlib.Path, options: DeploymentOptionsBaseten
) -> b10_service.BasetenService:
    truss_handle = truss.load(str(truss_dir))
    model_name = truss_handle.spec.config.model_name
    assert model_name is not None
    logging.info(
        f"Deploying truss model to Baseten`{model_name}` "
        f"(publish={options.publish}, promote={options.promote})."
    )
    # Models must be trusted to use the API KEY secret.
    service = options.remote_provider.push(
        truss_handle,
        model_name=model_name,
        trusted=True,
        publish=options.publish,
        promote=options.promote,
    )
    if not isinstance(service, b10_service.BasetenService):
        raise TypeError("Remote provider did not return baseten service.")
    return cast(b10_service.BasetenService, service)


class DockerService(b10_service.TrussService):
    def __init__(self, remote_url: str, is_draft: bool, **kwargs):
        super().__init__(remote_url, is_draft, **kwargs)

    def authenticate(self) -> Dict[str, str]:
        return {}

    def is_live(self) -> bool:
        response = self._send_request(self._service_url, "GET")
        if response.status_code == 200:
            return True
        return False

    def is_ready(self) -> bool:
        response = self._send_request(self._service_url, "GET")
        if response.status_code == 200:
            return True
        return False

    @property
    def logs_url(self) -> str:
        raise NotImplementedError()

    @property
    def predict_url(self) -> str:
        return f"{self._service_url}/v1/models/model:predict"

    def poll_deployment_status(self, sleep_secs: int = 1) -> Iterator[str]:
        raise NotImplementedError()


def _deploy_service(
    chainlet_dir: pathlib.Path,
    chain_root: pathlib.Path,
    chainlet_descriptor: definitions.ChainletAPIDescriptor,
    chainlet_name_to_url: Mapping[str, str],
    maybe_stub_types_file: Optional[pathlib.Path],
    options: DeploymentOptions,
) -> Optional[b10_service.TrussService]:
    # Filter needed services and customize options.
    dep_services = {}
    for dep in chainlet_descriptor.dependencies.values():
        dep_services[dep.name] = definitions.ServiceDescriptor(
            name=dep.name,
            predict_url=chainlet_name_to_url[dep.name],
            options=dep.options,
        )
    # Convert to truss and deploy.
    # TODO: support file-based config (and/or merge file and python-src config values).
    remote_config = chainlet_descriptor.chainlet_cls.remote_config
    chainlet_name = remote_config.name or chainlet_descriptor.name
    model_name = f"{options.chain_name}.{chainlet_name}"
    truss_dir = make_truss(
        chainlet_dir,
        chain_root,
        remote_config,
        chainlet_descriptor.chainlet_cls.default_user_config,
        model_name,
        dep_services,
        maybe_stub_types_file,
    )
    service: Optional[b10_service.TrussService]
    if options.only_generate_trusses:
        service = None
    elif isinstance(options, DeploymentOptionsLocalDocker):
        port = utils.get_free_port()
        truss_handle = truss.load(str(truss_dir))
        truss_handle.add_secret(
            definitions.BASETEN_API_SECRET_NAME, options.baseten_chain_api_key
        )
        truss_handle.docker_run(
            local_port=port,
            detach=True,
            wait_for_server_ready=True,
            network="host",
            container_name_prefix=model_name,
        )
        # http://localhost:{port} seems to only work *sometimes* with docker.
        service = DockerService(f"http://host.docker.internal:{port}", is_draft=True)
    elif isinstance(options, DeploymentOptionsBaseten):
        with utils.log_level(logging.INFO):
            service = _deploy_to_baseten(truss_dir, options)
    else:
        raise NotImplementedError(options)

    if service:
        logging.info(f"Deployed service `{chainlet_name}` @ {service.predict_url}.")
    return service


def _get_ordered_dependencies(
    chainlets: Iterable[Type[definitions.ABCChainlet]],
) -> Iterable[definitions.ChainletAPIDescriptor]:
    """Gather all Chainlets needed and returns a topologically ordered list."""
    needed_chainlets: set[definitions.ChainletAPIDescriptor] = set()

    def add_needed_chainlets(chainlet: definitions.ChainletAPIDescriptor):
        needed_chainlets.add(chainlet)
        for chainlet_descriptor in framework.global_chainlet_registry.get_dependencies(
            chainlet
        ):
            needed_chainlets.add(chainlet_descriptor)
            add_needed_chainlets(chainlet_descriptor)

    for chainlet_cls in chainlets:
        add_needed_chainlets(
            framework.global_chainlet_registry.get_descriptor(chainlet_cls)
        )
    # Iterating over the registry ensures topological ordering.
    return [
        descr
        for descr in framework.global_chainlet_registry.chainlet_descriptors
        if descr in needed_chainlets
    ]


class ChainService:
    _entrypoint: str
    _services: MutableMapping[str, b10_service.TrussService]

    def __init__(self, entrypoint: str, name: str) -> None:
        self.name = name
        self._entrypoint = entrypoint
        self._services = collections.OrderedDict()  # Preserve order.

    def add_service(self, name: str, service: b10_service.TrussService) -> None:
        self._services[name] = service

    @property
    def get_entrypoint(self) -> b10_service.TrussService:
        service = self._services.get(self._entrypoint)
        if not service:
            raise definitions.MissingDependencyError(
                f"Service for entrypoint `{self._entrypoint}` was not added."
            )
        return service

    @property
    def run_url(self) -> str:
        return self.get_entrypoint.predict_url

    def run(self, json: Dict) -> Any:
        return self.get_entrypoint.predict(json)

    def get_info(self) -> list[tuple[str, str, str]]:
        """Return list with elements (name, status, logs_url) for each chainlet."""
        return list(
            (name, next(service.poll_deployment_status(sleep_secs=0)), service.logs_url)
            for name, service in self._services.items()
        )


def deploy_remotely(
    entrypoint: Type[definitions.ABCChainlet],
    options: DeploymentOptions,
    non_entrypoint_root_dir: Optional[str] = None,
) -> ChainService:
    """
    * Gathers dependencies of `entrypoint`.
    * Generates stubs.
    * Generates truss model code, including stub initialization.
    * Deploys truss models to baseten.
    """
    # TODO: revisit how chain root is inferred/specified, current might be brittle.
    if non_entrypoint_root_dir:
        chain_root = pathlib.Path(non_entrypoint_root_dir).absolute()
    else:
        chain_root = pathlib.Path(inspect.getfile(entrypoint)).absolute().parent
    logging.info(f"Using project root for chain: `{chain_root}`.")

    chainlet_name_to_url: dict[str, str] = {}
    chain_service = ChainService(
        framework.global_chainlet_registry.get_descriptor(entrypoint).name,
        name=options.chain_name,
    )
    for chainlet_descriptor in _get_ordered_dependencies([entrypoint]):
        logging.info(f"Deploying `{chainlet_descriptor.name}`.")
        deps = framework.global_chainlet_registry.get_dependencies(chainlet_descriptor)
        chainlet_dir = code_gen.make_chainlet_dir(
            options.chain_name, chainlet_descriptor
        )
        chainlet_filepath = pathlib.Path(
            shutil.copy(
                chainlet_descriptor.src_path,
                chainlet_dir / f"{definitions.chainlet_MODULE}.py",
            )
        )
        maybe_stub_types_file = code_gen.gen_pydantic_models(
            chainlet_dir / f"{definitions.STUB_TYPE_MODULE}.py", deps
        )
        code_gen.generate_chainlet_source(
            chainlet_filepath, chainlet_descriptor, deps, maybe_stub_types_file
        )
        service = _deploy_service(
            chainlet_dir,
            chain_root,
            chainlet_descriptor,
            chainlet_name_to_url,
            maybe_stub_types_file,
            options,
        )
        if service:
            chain_service.add_service(chainlet_descriptor.name, service)
            chainlet_name_to_url[chainlet_descriptor.name] = service.predict_url
        else:
            chainlet_name_to_url[chainlet_descriptor.name] = "http://dummy"

    return chain_service
