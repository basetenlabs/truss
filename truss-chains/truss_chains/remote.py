import abc
import concurrent.futures
import inspect
import logging
import pathlib
import re
import tempfile
import textwrap
import traceback
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    NamedTuple,
    Optional,
    Type,
    cast,
)

import tenacity
import truss
import watchfiles

if TYPE_CHECKING:
    from rich import console as rich_console
from truss.remote import remote_cli, remote_factory
from truss.remote.baseten import core as b10_core
from truss.remote.baseten import custom_types as b10_types
from truss.remote.baseten import remote as b10_remote
from truss.remote.baseten import service as b10_service
from truss.util import log_utils
from truss.util import path as truss_path

from truss_chains import code_gen, definitions, framework, utils

_MODEL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+-[0-9a-f]{8}$")


def _push_to_baseten(
    truss_dir: pathlib.Path, options: definitions.PushOptionsBaseten
) -> b10_service.BasetenService:
    truss_handle = truss.load(str(truss_dir))
    model_name = truss_handle.spec.config.model_name
    assert model_name is not None
    assert bool(_MODEL_NAME_RE.match(model_name))
    if options.promote and not options.publish:
        logging.info("`promote=True` overrides `publish` to `True`.")
    logging.info(
        f"Pushing chainlet `{model_name}` as a truss model on Baseten "
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


def _push_service(
    truss_dir: pathlib.Path,
    chainlet_descriptor: definitions.ChainletAPIDescriptor,
    options: definitions.PushOptions,
) -> b10_service.TrussService:
    service: b10_service.TrussService
    if isinstance(options, definitions.PushOptionsLocalDocker):
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
    elif isinstance(options, definitions.PushOptionsBaseten):
        with utils.log_level(logging.INFO):
            service = _push_to_baseten(truss_dir, options)
    else:
        raise NotImplementedError(options)

    logging.info(f"Pushed `{chainlet_descriptor.display_name}`")
    logging.debug(f"Internal model endpoint: `{service.predict_url}`")
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

    A ``ChainService`` is created and returned when using ``push``. It
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
            List of ``DeployedChainlet`` for each chainlet."""

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
        return b10_service.URLConfig.invocation_url(
            self._remote.api.rest_api_url,
            b10_service.URLConfig.CHAIN,
            self._chain_deployment_handle.chain_id,
            self._chain_deployment_handle.chain_deployment_id,
            self._chain_deployment_handle.is_draft,
        )

    @property
    def status_page_url(self) -> str:
        """Link to status page on Baseten."""
        return b10_service.URLConfig.status_page_url(
            self._remote.remote_url,
            b10_service.URLConfig.CHAIN,
            self._chain_deployment_handle.chain_id,
        )

    @tenacity.retry(
        stop=tenacity.stop_after_delay(300), wait=tenacity.wait_fixed(1), reraise=True
    )
    def get_info(self) -> list[b10_types.DeployedChainlet]:
        """Queries the statuses of all chainlets in the chain.

        Returns:
            List of ``DeployedChainlet`` for each chainlet."""
        return self._remote.get_chainlets(
            self._chain_deployment_handle.chain_deployment_id
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
        """Not Implemented."""
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
    baseten_options: definitions.PushOptionsBaseten,
    chainlet_services: list["_Pusher.ChainEntry"],
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


class _Pusher:
    class ChainEntry(NamedTuple):
        service: b10_service.TrussService
        chainlet_display_name: str
        is_entrypoint: bool

    def __init__(
        self,
        options: definitions.PushOptions,
        gen_root: Optional[pathlib.Path] = None,
    ) -> None:
        self._options = options
        self._gen_root = gen_root or pathlib.Path(tempfile.gettempdir())
        if isinstance(self._options, definitions.PushOptionsBaseten):
            _create_chains_secret_if_missing(self._options.remote_provider)

    def push(
        self,
        entrypoint: Type[definitions.ABCChainlet],
        non_entrypoint_root_dir: Optional[str] = None,
    ) -> Optional[ChainService]:
        chain_root = _get_chain_root(entrypoint, non_entrypoint_root_dir)
        chainlet_display_name_to_url: MutableMapping[str, str] = {}
        chainlet_services: list[_Pusher.ChainEntry] = []
        entrypoint_service = None
        for chainlet_descriptor in _get_ordered_dependencies([entrypoint]):
            model_base_name = chainlet_descriptor.display_name
            # Since we are creating a distinct model for each deployment of the chain,
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
            service = _push_service(chainlet_dir, chainlet_descriptor, self._options)
            chainlet_display_name_to_url[chainlet_descriptor.display_name] = (
                service.predict_url
            )
            chainlet_services.append(
                _Pusher.ChainEntry(
                    service, chainlet_descriptor.display_name, is_entrypoint
                )
            )
            if is_entrypoint:
                assert entrypoint_service is None
                entrypoint_service = service

        if self._options.only_generate_trusses:
            return None
        assert entrypoint_service is not None

        if isinstance(self._options, definitions.PushOptionsBaseten):
            assert isinstance(entrypoint_service, b10_service.BasetenService)
            return _create_baseten_chain(
                self._options, chainlet_services, entrypoint_service
            )
        elif isinstance(self._options, definitions.PushOptionsLocalDocker):
            assert isinstance(entrypoint_service, DockerTrussService)
            return DockerChainService(self._options.chain_name, entrypoint_service)
        else:
            raise NotImplementedError(self._options)


def push(
    entrypoint: Type[definitions.ABCChainlet],
    options: definitions.PushOptions,
    non_entrypoint_root_dir: Optional[str] = None,
    gen_root: pathlib.Path = pathlib.Path(tempfile.gettempdir()),
) -> Optional[ChainService]:
    return _Pusher(options, gen_root).push(entrypoint, non_entrypoint_root_dir)


# Watch / Live Patching ################################################################


class _Watcher:
    _source: pathlib.Path
    _entrypoint: Optional[str]
    _deployed_chain_name: str
    _remote_provider: b10_remote.BasetenRemote
    _chainlet_data: Mapping[str, b10_types.DeployedChainlet]
    _watch_filter: Callable[[watchfiles.Change, str], bool]
    _console: "rich_console.Console"
    _error_console: "rich_console.Console"
    _show_stack_trace: bool

    def __init__(
        self,
        source: pathlib.Path,
        entrypoint: Optional[str],
        name: Optional[str],
        remote: Optional[str],
        console: "rich_console.Console",
        error_console: "rich_console.Console",
        show_stack_trace: bool,
    ) -> None:
        self._source = source
        self._entrypoint = entrypoint
        self._console = console
        self._error_console = error_console
        self._show_stack_trace = show_stack_trace
        if not remote:
            remote = remote_cli.inquire_remote_name(
                remote_factory.RemoteFactory.get_available_config_names()
            )
        self._remote_provider = cast(
            b10_remote.BasetenRemote,
            remote_factory.RemoteFactory.create(remote=remote),
        )
        with framework.import_target(source, entrypoint) as entrypoint_cls:
            self._deployed_chain_name = name or entrypoint_cls.__name__
            self._chain_root = _get_chain_root(entrypoint_cls)
            chainlet_names = set(
                desc.display_name
                for desc in _get_ordered_dependencies([entrypoint_cls])
            )

        chain_id = b10_core.get_chain_id_by_name(
            self._remote_provider.api, self._deployed_chain_name
        )
        if not chain_id:
            raise definitions.ChainsDeploymentError(
                f"Chain `{chain_id}` was not found."
            )
        self._status_page_url = b10_service.URLConfig.status_page_url(
            self._remote_provider.remote_url, b10_service.URLConfig.CHAIN, chain_id
        )
        chain_deployment = b10_core.get_dev_chain_deployment(
            self._remote_provider.api, chain_id
        )
        if chain_deployment is None:
            raise definitions.ChainsDeploymentError(
                f"No development deployment was found for Chain `{chain_id}`. "
                "You cannot live-patch production deployments. Check the Chain's "
                f"status page for available deployments: {self._status_page_url}."
            )
        deployed_chainlets = self._remote_provider.get_chainlets(chain_deployment["id"])
        non_draft_chainlets = [
            chainlet.name for chainlet in deployed_chainlets if not chainlet.is_draft
        ]
        assert not (
            non_draft_chainlets
        ), "If the chain is draft, the oracles must be draft."

        self._chainlet_data = {c.name: c for c in deployed_chainlets}
        self._assert_chainlet_names_same(chainlet_names)
        self._ignore_patterns = truss_path.load_trussignore_patterns()

        def watch_filter(_: watchfiles.Change, path: str) -> bool:
            return not truss_path.is_ignored(pathlib.Path(path), self._ignore_patterns)

        logging.getLogger("watchfiles.main").disabled = True
        self._watch_filter = watch_filter

    @property
    def _original_chainlet_names(self) -> set[str]:
        return set(self._chainlet_data.keys())

    @property
    def _chainlet_display_name_to_url(self) -> Mapping[str, str]:
        return {k: v.oracle_predict_url for k, v in self._chainlet_data.items()}

    def _assert_chainlet_names_same(self, new_names: set[str]) -> None:
        missing = self._original_chainlet_names - new_names
        added = new_names - self._original_chainlet_names
        if not (missing or added):
            return
        msg_parts = [
            "The deployed Chainlets and the Chainlets in the current workspace differ. "
            "Live patching is not possible if the set of Chainlet names differ."
        ]
        if missing:
            msg_parts.append(f"Chainlets missing in current workspace: {list(missing)}")
        if added:
            msg_parts.append(f"Chainlets added in current workspace: {list(added)}")

        raise definitions.ChainsDeploymentError("\n".join(msg_parts))

    def _code_gen_and_patch_thread(
        self, descr: definitions.ChainletAPIDescriptor, user_env: Mapping[str, str]
    ) -> tuple[b10_remote.PatchResult, list[str]]:
        with log_utils.LogInterceptor() as log_interceptor:
            # TODO: Maybe try-except code_gen errors explicitly.
            chainlet_dir = code_gen.gen_truss_chainlet(
                self._chain_root,
                pathlib.Path(tempfile.gettempdir()),
                self._deployed_chain_name,
                descr,
                self._chainlet_data[descr.display_name].oracle_name,
                self._chainlet_display_name_to_url,
                user_env,
            )
            patch_result = self._remote_provider.patch_for_chainlet(
                chainlet_dir, self._ignore_patterns
            )
            logs = log_interceptor.get_logs()
        return patch_result, logs

    def _patch(
        self,
        executor: concurrent.futures.Executor,
        user_env: Optional[Mapping[str, str]],
    ) -> None:
        exception_raised = None
        stack_trace = ""
        with log_utils.LogInterceptor() as log_interceptor, self._console.status(
            " Live Patching Chain.\n", spinner="arrow3"
        ):
            # Handle import errors gracefully (e.g. if user saved file, but there
            # are syntax errors, undefined symbols etc.).
            try:
                with framework.import_target(
                    self._source, self._entrypoint
                ) as entrypoint_cls:
                    chainlet_descriptors = _get_ordered_dependencies([entrypoint_cls])
                    chain_root_new = _get_chain_root(entrypoint_cls)
                    assert chain_root_new == self._chain_root
                    self._assert_chainlet_names_same(
                        set(desc.display_name for desc in chainlet_descriptors)
                    )
                    future_to_display_name = {}
                    for chainlet_descr in chainlet_descriptors:
                        future = executor.submit(
                            self._code_gen_and_patch_thread,
                            chainlet_descr,
                            user_env or {},
                        )
                        future_to_display_name[future] = chainlet_descr.display_name
                    # Threads need to finish while inside the `import_target`-context.
                    done_futures = {
                        future_to_display_name[future]: future
                        for future in concurrent.futures.as_completed(
                            future_to_display_name
                        )
                    }
            except Exception as e:
                exception_raised = e
                stack_trace = traceback.format_exc()
            finally:
                logs = log_interceptor.get_logs()

        if logs:
            formatted_logs = textwrap.indent("\n".join(logs), " " * 4)
            self._console.print(
                f"Intercepted logs from importing chain source code:\n{formatted_logs}"
            )

        if exception_raised:
            self._error_console.print(
                "Source files were changed, but pre-conditions for "
                "live patching are not given. Most likely there is a "
                "syntax in the source files or chainlet names changed. "
                "Try to fix the issue and save the file. Error:\n"
                f"{textwrap.indent(str(exception_raised), ' ' * 4)}"
            )
            if self._show_stack_trace:
                self._error_console.print(stack_trace)

            self._console.print(
                "The watcher will continue and if you can resolve the "
                "issue, subsequent patches might succeed.",
                style="blue",
            )
            return

        self._check_patch_results(done_futures)

    def _check_patch_results(
        self,
        display_name_to_done_future: Mapping[
            str, concurrent.futures.Future[tuple[b10_remote.PatchResult, list[str]]]
        ],
    ) -> None:
        has_errors = False
        for display_name, future in display_name_to_done_future.items():
            # It is not expected that code_gen_and_patch raises an exception, errors
            # should be handled by setting `b10_remote.PatchStatus`.
            # If an exception is raised anyway, it should bubble up the default way.
            patch_result, logs = future.result()
            if logs:
                formatted_logs = textwrap.indent("\n".join(logs), " " * 4)
                logs_output = f" [grey70]Intercepted logs:\n{formatted_logs}[grey70]"
            else:
                logs_output = ""

            if patch_result.status == b10_remote.PatchStatus.SUCCESS:
                self._console.print(
                    f"Patched Chainlet `{display_name}`.{logs_output}", style="green"
                )
            elif patch_result.status == b10_remote.PatchStatus.SKIPPED:
                self._console.print(
                    f"Nothing to do for Chainlet `{display_name}`.{logs_output}",
                    style="grey50",
                )
            else:
                has_errors = True
                self._error_console.print(
                    f"Failed to patch Chainlet `{display_name}`. "
                    f"{patch_result.message}{logs_output}"
                )

        if has_errors:
            msg = (
                "Some Chainlets could not be live patched. See above error messages. "
                "The watcher will continue, and try patching new changes. However, the "
                "safest way to proceed and ensure a consistent state is to re-deploy "
                "the the entire development Chain."
            )
            self._error_console.print(msg)

    def watch(self, user_env: Optional[Mapping[str, str]]) -> None:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Perform one initial patch at startup.
            self._patch(executor, user_env)
            self._console.print("👀 Watching for new changes.", style="blue")
            for _ in watchfiles.watch(
                self._chain_root, watch_filter=self._watch_filter, raise_interrupt=False
            ):
                self._patch(executor, user_env)
                self._console.print("👀 Watching for new changes.", style="blue")


def watch(
    source: pathlib.Path,
    entrypoint: Optional[str],
    name: Optional[str],
    remote: Optional[str],
    user_env: Optional[Mapping[str, str]],
    console: "rich_console.Console",
    error_console: "rich_console.Console",
    show_stack_trace: bool,
) -> None:
    console.print(
        (
            "👀 Starting to watch for Chain source code and applying live patches "
            "when changes are detected."
        ),
        style="blue",
    )
    patcher = _Watcher(
        source, entrypoint, name, remote, console, error_console, show_stack_trace
    )
    patcher.watch(user_env)
