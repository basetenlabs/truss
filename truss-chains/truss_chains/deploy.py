import collections
import inspect
import logging
import pathlib
from typing import Any, Dict, Iterable, Iterator, MutableMapping, Optional, Type, cast

import truss
from truss.remote.baseten import service as b10_service
from truss_chains import code_gen, definitions, framework, utils


def _deploy_to_baseten(
    truss_dir: pathlib.Path, options: definitions.DeploymentOptionsBaseten
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
    truss_dir: pathlib.Path,
    chainlet_descriptor: definitions.ChainletAPIDescriptor,
    options: definitions.DeploymentOptions,
) -> Optional[b10_service.TrussService]:
    service: Optional[b10_service.TrussService]
    if options.only_generate_trusses:
        service = None
    elif isinstance(options, definitions.DeploymentOptionsLocalDocker):
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
            container_name_prefix=chainlet_descriptor.name,
        )
        # http://localhost:{port} seems to only work *sometimes* with docker.
        service = DockerService(f"http://host.docker.internal:{port}", is_draft=True)
    elif isinstance(options, definitions.DeploymentOptionsBaseten):
        with utils.log_level(logging.INFO):
            service = _deploy_to_baseten(truss_dir, options)
    else:
        raise NotImplementedError(options)

    if service:
        logging.info(
            f"Deployed service `{chainlet_descriptor.name}` @ {service.predict_url}."
        )
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
    options: definitions.DeploymentOptions,
    non_entrypoint_root_dir: Optional[str] = None,
    gen_root: pathlib.Path = pathlib.Path("/tmp"),
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
    if isinstance(options, definitions.DeploymentOptionsBaseten):
        secrets_info = options.remote_provider.api.get_all_secrets()
        secret_names = {sec["name"] for sec in secrets_info["secrets"]}
        if definitions.BASETEN_API_SECRET_NAME not in secret_names:
            logging.info(
                "It seems you are using chains for the first time, since there "
                f"is no `{definitions.BASETEN_API_SECRET_NAME}` secret on baseten. "
                "Creating secret automatically."
            )
            options.remote_provider.api.upsert_secret(
                definitions.BASETEN_API_SECRET_NAME,
                options.remote_provider.api.auth_token.value,
            )

    for chainlet_descriptor in _get_ordered_dependencies([entrypoint]):
        logging.info(f"Deploying `{chainlet_descriptor.name}`.")
        deps = framework.global_chainlet_registry.get_dependencies(chainlet_descriptor)
        chainlet_dir = code_gen.gen_truss_chainlet(
            options,
            chainlet_descriptor,
            deps,
            chainlet_name_to_url,
            chain_root,
            gen_root,
        )
        service = _deploy_service(chainlet_dir, chainlet_descriptor, options)
        if service:
            chain_service.add_service(chainlet_descriptor.name, service)
            chainlet_name_to_url[chainlet_descriptor.name] = service.predict_url
        else:
            chainlet_name_to_url[chainlet_descriptor.name] = "http://dummy"

    return chain_service
