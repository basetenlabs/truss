import abc
import inspect
import logging
import pathlib
import re
import tempfile
import uuid
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    MutableMapping,
    NamedTuple,
    Optional,
    Type,
    cast,
)

import tenacity
import truss
from truss.remote.baseten import core as b10_core
from truss.remote.baseten import custom_types as b10_types
from truss.remote.baseten import remote as b10_remote
from truss.remote.baseten import service as b10_service

from truss_chains import code_gen, definitions, framework, utils

_MODEL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+-[0-9a-f]{8}$")


def _deploy_to_baseten(
    truss_dir: pathlib.Path, options: definitions.DeploymentOptionsBaseten
) -> b10_service.BasetenService:
    truss_handle = truss.load(str(truss_dir))
    model_name = truss_handle.spec.config.model_name
    assert model_name is not None
    assert bool(_MODEL_NAME_RE.match(model_name))
    if options.promote and not options.publish:
        logging.info("`promote=True` overrides `publish` to `True`.")
    logging.info(
        f"Deploying chainlet `{model_name}` as a truss model on Baseten "
        f"(publish={options.publish}, promote={options.promote})."
    )
    # Models must be trusted to use the API KEY secret.
    service = options.remote_provider.push(
        truss_handle,
        model_name=model_name,
        trusted=True,
        publish=options.publish,
        promote=options.promote,
        origin=b10_types.ModelOrigin.CHAINS,
    )
    return cast(b10_service.BasetenService, service)


class DockerTrussService(b10_service.TrussService):
    """This service is for Chainlets (not for Chains)."""

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
) -> b10_service.TrussService:
    service: b10_service.TrussService
    if isinstance(options, definitions.DeploymentOptionsLocalDocker):
        logging.info(
            f"Running in docker container `{chainlet_descriptor.display_name}` "
        )
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
            container_name_prefix=chainlet_descriptor.display_name,
        )
        # http://localhost:{port} seems to only work *sometimes* with docker.
        service = DockerTrussService(
            f"http://host.docker.internal:{port}", is_draft=True
        )
    elif isinstance(options, definitions.DeploymentOptionsBaseten):
        with utils.log_level(logging.INFO):
            service = _deploy_to_baseten(truss_dir, options)
    else:
        raise NotImplementedError(options)

    logging.info(
        f"Deployed `{chainlet_descriptor.display_name}` @ {service.predict_url}."
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


class ChainService(abc.ABC):
    """Handle for a deployed chain.

    A ``ChainService`` is created and returned when using ``deploy_remotely``. It
    bundles the individual services for each chainlet in the chain, and provides
    utilities to query their status, invoke the entrypoint etc.
    """

    _name: str
    _entrypoint_service: b10_service.TrussService
    _entrypoint_fake_json_data: Any

    def __init__(self, name: str, entrypoint_service: b10_service.TrussService):
        self._name = name
        self._entrypoint_service = entrypoint_service
        self._entrypoint_fake_json_data = None

    @property
    def name(self) -> str:
        return self._name

    @property
    @abc.abstractmethod
    def status_page_url(self) -> str:
        """Link to status page on Baseten."""

    @property
    @abc.abstractmethod
    def run_remote_url(self) -> str:
        """URL to invoke the entrypoint."""

    def run_remote(self, json: Dict) -> Any:
        """Invokes the entrypoint with JSON data.

        Returns:
            The JSON response."""
        return self._entrypoint_service.predict(json)

    @abc.abstractmethod
    def get_info(self) -> list[b10_types.DeployedChainlet]:
        """Queries the statuses of all chainlets in the chain.

        Returns:
            List of ``DeployedChainlet``, ``(name, is_entrypoint, status, logs_url)``
            for each chainlet."""

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


class BasetenChainService(ChainService):
    _chain_deployment_handle: b10_core.ChainDeploymentHandle
    _remote: b10_remote.BasetenRemote

    def __init__(
        self,
        name: str,
        entrypoint_service: b10_service.BasetenService,
        chain_deployment_handle: b10_core.ChainDeploymentHandle,
        remote: b10_remote.BasetenRemote,
    ) -> None:
        super().__init__(name, entrypoint_service)
        self._chain_deployment_handle = chain_deployment_handle
        self._remote = remote

    @property
    def run_remote_url(self) -> str:
        """URL to invoke the entrypoint."""
        return b10_service.make_invocation_url(
            self._remote.api.rest_api_url,
            b10_service.URLConfig.CHAIN,
            self._chain_deployment_handle.chain_id,
            self._chain_deployment_handle.chain_deployment_id,
            self._chain_deployment_handle.is_draft,
        )

    @property
    def status_page_url(self) -> str:
        """Link to status page on Baseten."""
        return (
            f"{self._remote.remote_url}/chains/"
            f"{self._chain_deployment_handle.chain_id}/overview"
        )

    @tenacity.retry(
        stop=tenacity.stop_after_delay(300), wait=tenacity.wait_fixed(1), reraise=True
    )
    def get_info(self) -> list[b10_types.DeployedChainlet]:
        """Queries the statuses of all chainlets in the chain.

        Returns:
            List of ``DeployedChainlet``, ``(name, is_entrypoint, status, logs_url)``,
                for each chainlet."""
        chainlets = self._remote.get_chainlets(
            chain_deployment_id=self._chain_deployment_handle.chain_deployment_id
        )
        return [
            b10_types.DeployedChainlet(
                name=chainlet["name"],
                is_entrypoint=chainlet["is_entrypoint"],
                status=chainlet["oracle_version"]["current_model_deployment_status"][
                    "status"
                ],
                logs_url=self._chainlet_logs_url(
                    chainlet["id"],
                ),
            )
            for chainlet in chainlets
        ]

    def _chainlet_logs_url(self, chainlet_id: str) -> str:
        return (
            f"{self._remote.remote_url}/chains/{self._chain_deployment_handle.chain_id}"
            f"/logs/{self._chain_deployment_handle.chain_deployment_id}/{chainlet_id}"
        )


class DockerChainService(ChainService):
    def __init__(self, name: str, entrypoint_service: DockerTrussService) -> None:
        super().__init__(name, entrypoint_service)

    @property
    def run_remote_url(self) -> str:
        """URL to invoke the entrypoint."""
        return self._entrypoint_service.predict_url

    @property
    def status_page_url(self) -> str:
        """Not Implemented.."""
        raise NotImplementedError()

    def get_info(self) -> list[b10_types.DeployedChainlet]:
        """Not Implemented.."""
        raise NotImplementedError()


def _get_chain_root(
    entrypoint: Type[definitions.ABCChainlet],
    non_entrypoint_root_dir: Optional[str] = None,
) -> pathlib.Path:
    # TODO: revisit how chain root is inferred/specified, current might be brittle.
    if non_entrypoint_root_dir:
        chain_root = pathlib.Path(non_entrypoint_root_dir).absolute()
    else:
        chain_root = pathlib.Path(inspect.getfile(entrypoint)).absolute().parent
    logging.info(
        f"Using chain workspace dir: `{chain_root}` (files under this dir will "
        "be included as dependencies in the remote deployments and are importable)."
    )
    return chain_root


def _create_baseten_chain(
    baseten_options: definitions.DeploymentOptionsBaseten,
    chainlet_services: list["_Deployer.ChainEntry"],
    entrypoint_service: b10_service.BasetenService,
):
    chainlet_data = []
    for chain_entry in chainlet_services:
        assert isinstance(chain_entry.service, b10_service.BasetenService)
        chainlet_data.append(
            b10_types.ChainletData(
                name=chain_entry.chainlet_display_name,
                oracle_version_id=chain_entry.service.model_version_id,
                is_entrypoint=chain_entry.is_entrypoint,
            )
        )
    chain_deployment_handle = baseten_options.remote_provider.create_chain(
        chain_name=baseten_options.chain_name,
        chainlets=chainlet_data,
        publish=baseten_options.publish,
        promote=baseten_options.promote,
    )
    return BasetenChainService(
        baseten_options.chain_name,
        entrypoint_service,
        chain_deployment_handle,
        baseten_options.remote_provider,
    )


def _create_chains_secret_if_missing(remote_provider: b10_remote.BasetenRemote) -> None:
    secrets_info = remote_provider.api.get_all_secrets()
    secret_names = {sec["name"] for sec in secrets_info["secrets"]}
    if definitions.BASETEN_API_SECRET_NAME not in secret_names:
        logging.info(
            "It seems you are using chains for the first time, since there "
            f"is no `{definitions.BASETEN_API_SECRET_NAME}` secret on baseten. "
            "Creating secret automatically."
        )
        remote_provider.api.upsert_secret(
            definitions.BASETEN_API_SECRET_NAME,
            remote_provider.api.auth_token.value,
        )


class _Deployer:
    class ChainEntry(NamedTuple):
        service: b10_service.TrussService
        chainlet_display_name: str
        is_entrypoint: bool

    def __init__(
        self,
        options: definitions.DeploymentOptions,
        gen_root: Optional[pathlib.Path] = None,
    ) -> None:
        self._options = options
        self._gen_root = gen_root or pathlib.Path(tempfile.gettempdir())
        if isinstance(self._options, definitions.DeploymentOptionsBaseten):
            _create_chains_secret_if_missing(self._options.remote_provider)

    def deploy(
        self,
        entrypoint: Type[definitions.ABCChainlet],
        non_entrypoint_root_dir: Optional[str] = None,
    ) -> Optional[ChainService]:
        chain_root = _get_chain_root(entrypoint, non_entrypoint_root_dir)
        chainlet_display_name_to_url: MutableMapping[str, str] = {}
        chainlet_services: list[_Deployer.ChainEntry] = []
        entrypoint_service = None
        for chainlet_descriptor in _get_ordered_dependencies([entrypoint]):
            model_base_name = chainlet_descriptor.display_name
            # Since we are deploying a distinct model for each deployment of the chain,
            # we add a random suffix.
            model_suffix = str(uuid.uuid4()).split("-")[0]
            model_name = f"{model_base_name}-{model_suffix}"
            logging.info(
                f"Generating truss chainlet model for `{chainlet_descriptor.name}`."
            )
            chainlet_dir = code_gen.gen_truss_chainlet(
                chain_root,
                self._gen_root,
                self._options.chain_name,
                chainlet_descriptor,
                model_name,
                chainlet_display_name_to_url,
                self._options.user_env,
            )
            if self._options.only_generate_trusses:
                chainlet_display_name_to_url[chainlet_descriptor.display_name] = (
                    "http://dummy"
                )
                continue

            is_entrypoint = chainlet_descriptor.chainlet_cls == entrypoint
            service = _deploy_service(chainlet_dir, chainlet_descriptor, self._options)
            chainlet_display_name_to_url[chainlet_descriptor.display_name] = (
                service.predict_url
            )
            chainlet_services.append(
                _Deployer.ChainEntry(
                    service, chainlet_descriptor.display_name, is_entrypoint
                )
            )
            if is_entrypoint:
                assert entrypoint_service is None
                entrypoint_service = service

        if self._options.only_generate_trusses:
            return None
        assert entrypoint_service is not None

        if isinstance(self._options, definitions.DeploymentOptionsBaseten):
            assert isinstance(entrypoint_service, b10_service.BasetenService)
            return _create_baseten_chain(
                self._options, chainlet_services, entrypoint_service
            )
        elif isinstance(self._options, definitions.DeploymentOptionsLocalDocker):
            assert isinstance(entrypoint_service, DockerTrussService)
            return DockerChainService(self._options.chain_name, entrypoint_service)
        else:
            raise NotImplementedError(self._options)


def deploy_remotely(
    entrypoint: Type[definitions.ABCChainlet],
    options: definitions.DeploymentOptions,
    non_entrypoint_root_dir: Optional[str] = None,
    gen_root: pathlib.Path = pathlib.Path(tempfile.gettempdir()),
) -> Optional[ChainService]:
    return _Deployer(options, gen_root).deploy(entrypoint, non_entrypoint_root_dir)
