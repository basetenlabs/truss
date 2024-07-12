import collections
import inspect
import logging
import pathlib
import tempfile
import uuid
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    MutableMapping,
    Optional,
    Type,
    cast,
)

import truss
from tenacity import retry, stop_after_delay, wait_fixed
from truss.remote.baseten import remote as b10_remote
from truss.remote.baseten import service as b10_service
from truss.remote.baseten import types as b10_types

from truss_chains import code_gen, definitions, framework, utils


def _deploy_to_baseten(
    truss_dir: pathlib.Path, options: definitions.DeploymentOptionsBaseten
) -> b10_service.BasetenService:
    truss_handle = truss.load(str(truss_dir))
    model_name = truss_handle.spec.config.model_name
    assert model_name is not None
    if options.promote and not options.publish:
        logging.info("`promote=True` overrides `publish` to `True`.")
    logging.info(
        f"Deploying chainlet `{model_name}` as a truss model on Baseten "
        f"(publish={options.publish}, promote={options.promote})."
    )

    # Since we are deploying a model independently of the chain, we add a random suffix to
    # prevent us from running into issues with existing models with the same name.
    #
    # This is a bit of a hack for now. Once we support model_origin for Chains models, we
    # can drop the requirement for names on models.
    model_suffix = str(uuid.uuid4()).split("-")[0]

    # Models must be trusted to use the API KEY secret.
    service = options.remote_provider.push(
        truss_handle,
        model_name=f"{model_name}-{model_suffix}",
        trusted=True,
        publish=options.publish,
        promote=options.promote,
        origin=b10_types.ModelOrigin.CHAINS,
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
        logging.info(f"Running in docker container `{chainlet_descriptor.name}` ")
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
            f"Service created for `{chainlet_descriptor.name}` @ {service.predict_url}."
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


def _chainlet_logs_url(
    chain_id: str,
    chain_deployment_id: str,
    chainlet_id: str,
    remote: b10_remote.BasetenRemote,
) -> str:
    return f"{remote._remote_url}/chains/{chain_id}/logs/{chain_deployment_id}/{chainlet_id}"


def _chain_status_page_url(
    chain_id: str,
    remote: b10_remote.BasetenRemote,
) -> str:
    return f"{remote._remote_url}/chains/{chain_id}/overview"


class RemoteChainService:
    _remote: (
        b10_remote.BasetenRemote
    )  # TODO, make this a generic TypeVar for this calss
    _chain_id: str
    _chain_deployment_id: str

    def __init__(
        self, chain_id: str, chain_deployment_id: str, remote: b10_remote.BasetenRemote
    ) -> None:
        self._remote = remote
        self._chain_id = chain_id
        self._chain_deployment_id = chain_deployment_id

    @retry(stop=stop_after_delay(300), wait=wait_fixed(1), reraise=True)
    def get_info(self) -> list[b10_types.DeployedChainlet]:
        """Return list with elements (name, status, logs_url) for each chainlet."""
        chainlets = self._remote.get_chainlets(
            chain_deployment_id=self._chain_deployment_id
        )

        return [
            b10_types.DeployedChainlet(
                name=chainlet["name"],
                is_entrypoint=chainlet["is_entrypoint"],
                status=chainlet["oracle_version"]["current_model_deployment_status"][
                    "status"
                ],
                logs_url=_chainlet_logs_url(
                    self._chain_id,
                    self._chain_deployment_id,
                    chainlet["id"],
                    self._remote,
                ),
            )
            for chainlet in chainlets
        ]

    @property
    def status_page_url(self) -> str:
        return _chain_status_page_url(self._chain_id, self._remote)


class ChainService:
    # TODO: this exposes methods to users that should be internal (e.g. `add_service`).
    """Handle for a deployed chain.

    A ``ChainService`` is created and returned when using ``deploy_remotely``. It
    bundles the individual services for each chainlet in the chain, and provides
    utilities to query their status, invoke the entrypoint etc.
    """

    name: str
    _entrypoint: str
    _services: MutableMapping[str, b10_service.TrussService]
    _entrypoint_fake_json_data = Any
    _remote_chain_service: Optional[RemoteChainService]

    def __init__(self, entrypoint: str, name: str) -> None:
        """
        Args:
            entrypoint: Name of the entrypoint chainlet.
            name: Name of the chain.
        """
        self.name = name
        self._entrypoint = entrypoint
        self._services = collections.OrderedDict()  # Preserve order.
        self.entrypoint_fake_json_data = None

    def add_service(self, name: str, service: b10_service.TrussService) -> None:
        """
        Used to add a chainlet service during the deployment sequence of the chain.


        Args:
            name: Chainlet name.
            service: Service object for the chainlet.
        """
        self._services[name] = service

    def set_remote_chain_service(
        self, chain_id: str, chain_deployment_id: str, remote: b10_remote.BasetenRemote
    ) -> None:
        self._remote_chain_service = RemoteChainService(
            chain_id, chain_deployment_id, remote
        )

    @property
    def entrypoint_fake_json_data(self) -> Any:
        """Fake JSON example data that matches the entrypoint's input schema.
        This property must be externally populated.

        Raises:
            ValueError: If fake data was not set.
        """
        if self._entrypoint_fake_json_data is None:
            raise ValueError("Fake data was not set.")
        return self._entrypoint_fake_json_data

    @entrypoint_fake_json_data.setter
    def entrypoint_fake_json_data(self, fake_data: Any) -> None:
        self._entrypoint_fake_json_data = fake_data

    @property
    def get_entrypoint(self) -> b10_service.TrussService:
        """Returns the entrypoint's service handle.

        Raises:
            MissingDependencyError: If the entrypoint service was not added.
        """
        service = self._services.get(self._entrypoint)
        if not service:
            raise definitions.MissingDependencyError(
                f"Service for entrypoint `{self._entrypoint}` was not added."
            )
        return service

    @property
    def services(self) -> MutableMapping[str, b10_service.TrussService]:
        return self._services

    @property
    def entrypoint_name(self) -> str:
        return self._entrypoint

    @property
    def run_url(self) -> str:
        """URL to invoke the entrypoint."""
        return self.get_entrypoint.predict_url

    def run_remote(self, json: Dict) -> Any:
        """Invokes the entrypoint with JSON data.

        Returns:
            The JSON response."""
        return self.get_entrypoint.predict(json)

    def get_info(self) -> list[b10_types.DeployedChainlet]:
        """Queries the statuses of all chainlets in the chain.

        Returns:
            List with elements ``(name, status, logs_url)`` for each chainlet."""
        if not self._remote_chain_service:
            raise ValueError("Chain was not deployed remotely.")

        return self._remote_chain_service.get_info()

    @property
    def status_page_url(self) -> str:
        """Link to status page on Baseten."""
        if not self._remote_chain_service:
            raise ValueError("Chain was not deployed remotely.")

        return self._remote_chain_service.status_page_url


def deploy_remotely(
    entrypoint: Type[definitions.ABCChainlet],
    options: definitions.DeploymentOptions,
    non_entrypoint_root_dir: Optional[str] = None,
    gen_root: pathlib.Path = pathlib.Path(tempfile.gettempdir()),
) -> ChainService:
    # TODO: revisit how chain root is inferred/specified, current might be brittle.
    if non_entrypoint_root_dir:
        chain_root = pathlib.Path(non_entrypoint_root_dir).absolute()
    else:
        chain_root = pathlib.Path(inspect.getfile(entrypoint)).absolute().parent
    logging.info(
        f"Using chain workspace dir: `{chain_root}` (files under this dir will "
        "be included as dependencies in the remote deployments and are importable)."
    )

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
        else:  # only_generate_trusses case.
            chainlet_name_to_url[chainlet_descriptor.name] = "http://dummy"

    if (
        isinstance(options, definitions.DeploymentOptionsBaseten)
        and not options.only_generate_trusses
    ):
        chainlets: List[b10_types.ChainletData] = []
        entrypoint_name = chain_service.entrypoint_name

        for chainlet_name, truss_service in chain_service.services.items():
            baseten_service = cast(b10_service.BasetenService, truss_service)
            chainlets.append(
                b10_types.ChainletData(
                    name=chainlet_name,
                    oracle_version_id=baseten_service.model_version_id,
                    is_entrypoint=chainlet_name == entrypoint_name,
                )
            )

        chain_deployment_handle = options.remote_provider.create_chain(
            chain_name=chain_service.name,
            chainlets=chainlets,
            publish=options.publish,
            promote=options.promote,
        )

        chain_service.set_remote_chain_service(
            chain_deployment_handle.chain_id,
            chain_deployment_handle.chain_deployment_id,
            options.remote_provider,
        )

    return chain_service
