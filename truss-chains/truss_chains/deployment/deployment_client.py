import abc
import concurrent.futures
import inspect
import json
import logging
import pathlib
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
    Optional,
    Type,
    cast,
)

import requests
import tenacity
import watchfiles

from truss.local import local_config_handler
from truss.remote import remote_factory
from truss.remote.baseten import core as b10_core
from truss.remote.baseten import custom_types as b10_types
from truss.remote.baseten import error as b10_errors
from truss.remote.baseten import remote as b10_remote
from truss.remote.baseten import service as b10_service
from truss.truss_handle import truss_handle
from truss.util import log_utils
from truss.util import path as truss_path
from truss_chains import framework, private_types, public_types, utils
from truss_chains.deployment import code_gen

if TYPE_CHECKING:
    from rich import console as rich_console
    from rich import progress


def _get_ordered_dependencies(
    chainlets: Iterable[Type[private_types.ABCChainlet]],
) -> Iterable[private_types.ChainletAPIDescriptor]:
    """Gather all Chainlets needed and returns a topologically ordered list."""
    needed_chainlets: set[private_types.ChainletAPIDescriptor] = set()

    def add_needed_chainlets(chainlet: private_types.ChainletAPIDescriptor):
        needed_chainlets.add(chainlet)
        for chainlet_descriptor in framework.get_dependencies(chainlet):
            needed_chainlets.add(chainlet_descriptor)
            add_needed_chainlets(chainlet_descriptor)

    for chainlet_cls in chainlets:
        add_needed_chainlets(framework.get_descriptor(chainlet_cls))
    # Get dependencies in topological order.
    return [
        descr
        for descr in framework.get_ordered_descriptors()
        if descr in needed_chainlets
    ]


def _get_chain_root(entrypoint: Type[private_types.ABCChainlet]) -> pathlib.Path:
    # TODO: revisit how chain root is inferred/specified, current might be brittle.
    chain_root = pathlib.Path(inspect.getfile(entrypoint)).absolute().parent
    logging.info(
        f"Using chain workspace dir: `{chain_root}` (files under this dir will "
        "be included as dependencies in the remote deployments and are importable)."
    )
    return chain_root


class ChainService(abc.ABC):
    """Handle for a deployed chain.

    A ``ChainService`` is created and returned when using ``push``. It
    bundles the individual services for each chainlet in the chain, and provides
    utilities to query their status, invoke the entrypoint etc.
    """

    _name: str
    _entrypoint_fake_json_data: Any

    def __init__(self, name: str):
        self._name = name
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

    @abc.abstractmethod
    def run_remote(self, json: Dict) -> Any:
        """Invokes the entrypoint with JSON data.

        Returns:
            The JSON response."""

    @abc.abstractmethod
    def get_info(self) -> list[b10_types.DeployedChainlet]:
        """Queries the statuses of all chainlets in the chain.

        Returns:
            List of ``DeployedChainlet``, ``(name, is_entrypoint, status, logs_url)``
            for each chainlet.
        """

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


def _generate_chainlet_artifacts(
    options: private_types.PushOptions, entrypoint: Type[private_types.ABCChainlet]
) -> tuple[b10_types.ChainletArtifact, list[b10_types.ChainletArtifact], bool]:
    chain_root = _get_chain_root(entrypoint)
    entrypoint_artifact: Optional[b10_types.ChainletArtifact] = None
    dependency_artifacts: list[b10_types.ChainletArtifact] = []
    chainlet_display_names: set[str] = set()

    use_local_src = False
    if isinstance(options, private_types.PushOptionsLocalDocker):
        use_local_src = options.use_local_src

    has_engine_builder_chainlets = False

    for chainlet_descriptor in _get_ordered_dependencies([entrypoint]):
        if framework.is_engine_builder_chainlet(chainlet_descriptor.chainlet_cls):
            has_engine_builder_chainlets = True

        chainlet_display_name = chainlet_descriptor.display_name

        if chainlet_display_name in chainlet_display_names:
            raise public_types.ChainsUsageError(
                f"Chainlet names must be unique. Found multiple Chainlets with the name: '{chainlet_display_name}'."
            )

        chainlet_display_names.add(chainlet_display_name)

        # Since we are creating a distinct model for each deployment of the chain,
        # we add a random suffix.
        model_suffix = str(uuid.uuid4()).split("-")[0]
        model_name = f"{chainlet_display_name}-{model_suffix}"

        chainlet_dir = code_gen.gen_truss_chainlet(
            chain_root,
            options.chain_name,
            chainlet_descriptor,
            model_name,
            use_local_src,
        )
        artifact = b10_types.ChainletArtifact(
            truss_dir=chainlet_dir,
            name=chainlet_descriptor.name,
            display_name=chainlet_display_name,
        )

        is_entrypoint = chainlet_descriptor.chainlet_cls == entrypoint

        if is_entrypoint:
            assert entrypoint_artifact is None

            entrypoint_artifact = artifact
        else:
            dependency_artifacts.append(artifact)

    assert entrypoint_artifact is not None

    return entrypoint_artifact, dependency_artifacts, has_engine_builder_chainlets


@framework.raise_validation_errors_before
def push(
    entrypoint: Type[private_types.ABCChainlet],
    options: private_types.PushOptions,
    progress_bar: Optional[Type["progress.Progress"]] = None,
) -> Optional[ChainService]:
    entrypoint_artifact, dependency_artifacts, has_engine_builder_chainlets = (
        _generate_chainlet_artifacts(options, entrypoint)
    )
    if options.only_generate_trusses:
        return None
    if isinstance(options, private_types.PushOptionsBaseten):
        if has_engine_builder_chainlets and not options.publish:
            raise public_types.ChainsDeploymentError(
                "This chain contains engine builder chainlets. Development models are "
                "not supportd, push with `--publish`."
            )
        return _create_baseten_chain(
            options, entrypoint_artifact, dependency_artifacts, progress_bar
        )
    elif isinstance(options, private_types.PushOptionsLocalDocker):
        if has_engine_builder_chainlets:
            raise public_types.ChainsDeploymentError(
                "This chain contains engine builder chainlets. Running in local docker "
                "is not supported."
            )
        return _create_docker_chain(options, entrypoint_artifact, dependency_artifacts)
    else:
        raise NotImplementedError(options)


def push_debug_docker(
    entrypoint: Type[private_types.ABCChainlet], chain_name: Optional[str] = None
) -> ChainService:
    if not chain_name:
        chain_name = entrypoint.name.lower()
    options = private_types.PushOptionsLocalDocker(
        chain_name=chain_name, only_generate_trusses=False, use_local_src=True
    )
    return cast(ChainService, push(entrypoint, options))


# Docker ###############################################################################


class DockerChainletService(b10_service.TrussService):
    """This service is for Chainlets (not for Chains)."""

    def __init__(self, port: int, **kwargs):
        remote_url = f"http://localhost:{port}"

        super().__init__(remote_url, is_draft=False, **kwargs)

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


def _push_service_docker(
    truss_dir: pathlib.Path,
    chainlet_display_name: str,
    options: private_types.PushOptionsLocalDocker,
    port: int,
) -> None:
    th = truss_handle.TrussHandle(truss_dir)
    th.add_secret(public_types._BASETEN_API_SECRET_NAME, options.baseten_chain_api_key)
    th.docker_run(
        local_port=port,
        detach=True,
        wait_for_server_ready=True,
        network="host",
        container_name_prefix=chainlet_display_name,
        disable_json_logging=True,
    )


class DockerChainService(ChainService):
    _entrypoint_service: DockerChainletService

    def __init__(self, name: str, entrypoint_service: DockerChainletService) -> None:
        super().__init__(name)
        self._entrypoint_service = entrypoint_service

    @property
    def run_remote_url(self) -> str:
        """URL to invoke the entrypoint."""
        return self._entrypoint_service.predict_url

    def run_remote(self, json: Dict) -> Any:
        """Invokes the entrypoint with JSON data.

        Returns:
            The JSON response."""
        return self._entrypoint_service.predict(json)

    @property
    def status_page_url(self) -> str:
        """Not Implemented."""
        raise NotImplementedError()

    def get_info(self) -> list[b10_types.DeployedChainlet]:
        """Not Implemented."""
        raise NotImplementedError()


def _create_docker_chain(
    docker_options: private_types.PushOptionsLocalDocker,
    entrypoint_artifact: b10_types.ChainletArtifact,
    dependency_artifacts: list[b10_types.ChainletArtifact],
) -> DockerChainService:
    chainlet_artifacts = [*dependency_artifacts, entrypoint_artifact]
    chainlet_to_predict_url: Dict[str, Dict[str, str]] = {}
    chainlet_to_service: Dict[str, DockerChainletService] = {}
    for chainlet_artifact in chainlet_artifacts:
        port = utils.get_free_port()
        service = DockerChainletService(port)

        docker_internal_url = service.predict_url.replace(
            "localhost", "host.docker.internal"
        )
        chainlet_to_predict_url[chainlet_artifact.display_name] = {
            "predict_url": docker_internal_url
        }
        chainlet_to_service[chainlet_artifact.name] = service

        local_config_handler.LocalConfigHandler.set_dynamic_config(
            private_types.DYNAMIC_CHAINLET_CONFIG_KEY,
            json.dumps(chainlet_to_predict_url),
        )

        truss_dir = chainlet_artifact.truss_dir
        logging.info(
            f"Building Chainlet `{chainlet_artifact.display_name}` docker image."
        )
        _push_service_docker(
            truss_dir, chainlet_artifact.display_name, docker_options, port
        )
        logging.info(
            f"Pushed Chainlet `{chainlet_artifact.display_name}` as docker container."
        )
        logging.debug(
            "Internal model endpoint: "
            f"`{chainlet_to_predict_url[chainlet_artifact.display_name]}`"
        )

    return DockerChainService(
        docker_options.chain_name, chainlet_to_service[entrypoint_artifact.name]
    )


# Baseten ##############################################################################


class BasetenChainService(ChainService):
    _chain_deployment_handle: b10_core.ChainDeploymentHandleAtomic
    _remote: b10_remote.BasetenRemote

    def __init__(
        self,
        name: str,
        chain_deployment_handle: b10_core.ChainDeploymentHandleAtomic,
        remote: b10_remote.BasetenRemote,
    ) -> None:
        super().__init__(name)
        self._chain_deployment_handle = chain_deployment_handle
        self._remote = remote

    @property
    def run_remote_url(self) -> str:
        """URL to invoke the entrypoint."""

        handle = self._chain_deployment_handle

        return b10_service.URLConfig.invoke_url(
            hostname=handle.hostname,
            config=b10_service.URLConfig.CHAIN,
            entity_version_id=handle.chain_deployment_id,
            is_draft=handle.is_draft,
        )

    def run_remote(self, json_data: Dict) -> Any:
        """Invokes the entrypoint with JSON data.

        Returns:
            The JSON response."""
        headers = self._remote._auth_service.authenticate().header()
        response = requests.post(
            self.run_remote_url, json=json_data, headers=headers, stream=True
        )
        if response.status_code == 401:
            raise ValueError(
                f"Authentication failed with status code {response.status_code}"
            )

        if response.headers.get("transfer-encoding") == "chunked":
            # Case of streaming response, the backend does not set an encoding, so
            # manually decode to the contents to utf-8 here.
            def decode_content():
                for chunk in response.iter_content(
                    chunk_size=8192, decode_unicode=True
                ):
                    # Depending on the content-type of the response,
                    # iter_content will either emit a byte stream, or a stream
                    # of strings. Only decode in the bytes case.
                    if isinstance(chunk, bytes):
                        yield chunk.decode(
                            response.encoding or b10_service.DEFAULT_STREAM_ENCODING
                        )
                    else:
                        yield chunk

            return decode_content()

        parsed_response = response.json()

        if "error" in parsed_response:
            # In the case that the model is in a non-ready state, the response
            # will be a json with an `error` key.
            return parsed_response

        return response.json()

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


def _create_baseten_chain(
    baseten_options: private_types.PushOptionsBaseten,
    entrypoint_artifact: b10_types.ChainletArtifact,
    dependency_artifacts: list[b10_types.ChainletArtifact],
    progress_bar: Optional[Type["progress.Progress"]],
):
    logging.info(
        f"Pushing Chain '{baseten_options.chain_name}' to Baseten "
        f"(publish={baseten_options.publish}, environment={baseten_options.environment})."
    )
    remote_provider = cast(
        b10_remote.BasetenRemote,
        remote_factory.RemoteFactory.create(remote=baseten_options.remote),
    )

    if remote_provider.include_git_info or baseten_options.include_git_info:
        truss_user_env = b10_types.TrussUserEnv.collect_with_git_info(
            baseten_options.working_dir
        )
    else:
        truss_user_env = b10_types.TrussUserEnv.collect()

    _create_chains_secret_if_missing(remote_provider)

    chain_deployment_handle = remote_provider.push_chain_atomic(
        baseten_options.chain_name,
        entrypoint_artifact,
        dependency_artifacts,
        truss_user_env,
        publish=baseten_options.publish,
        environment=baseten_options.environment,
        progress_bar=progress_bar,
    )
    return BasetenChainService(
        baseten_options.chain_name, chain_deployment_handle, remote_provider
    )


def _create_chains_secret_if_missing(remote_provider: b10_remote.BasetenRemote) -> None:
    secrets_info = remote_provider.api.get_all_secrets()
    secret_names = {sec["name"] for sec in secrets_info["secrets"]}
    if public_types._BASETEN_API_SECRET_NAME not in secret_names:
        logging.info(
            "It seems you are using chains for the first time, since there "
            f"is no `{public_types._BASETEN_API_SECRET_NAME}` secret on baseten. "
            "Creating secret automatically."
        )
        remote_provider.api.upsert_secret(
            public_types._BASETEN_API_SECRET_NAME, remote_provider.api.auth_token.value
        )


# Watch / Live Patching ################################################################


def _create_watch_filter(root_dir: pathlib.Path):
    ignore_patterns = truss_path.load_trussignore_patterns_from_truss_dir(root_dir)

    def watch_filter(_: watchfiles.Change, path: str) -> bool:
        return not truss_path.is_ignored(pathlib.Path(path), ignore_patterns)

    logging.getLogger("watchfiles.main").disabled = True
    return ignore_patterns, watch_filter


def _handle_intercepted_logs(logs: list[str], console: "rich_console.Console"):
    if logs:
        formatted_logs = textwrap.indent("\n".join(logs), " " * 4)
        console.print(f"Intercepted logs from importing source code:\n{formatted_logs}")


def _handle_import_error(
    exception: Exception,
    console: "rich_console.Console",
    error_console: "rich_console.Console",
    stack_trace: Optional[str] = None,
):
    error_console.print(
        "Source files were changed, but pre-conditions for "
        "live patching are not given. Most likely there is a "
        "syntax error in the source files or names changed. "
        "Try to fix the issue and save the file. Error:\n"
        f"{textwrap.indent(str(exception), ' ' * 4)}"
    )
    if stack_trace:
        error_console.print(stack_trace)

    console.print(
        "The watcher will continue and if you can resolve the "
        "issue, subsequent patches might succeed.",
        style="blue",
    )


class _ModelWatcher:
    _source: pathlib.Path
    _model_name: str
    _remote_provider: b10_remote.BasetenRemote
    _ignore_patterns: list[str]
    _watch_filter: Callable[[watchfiles.Change, str], bool]
    _console: "rich_console.Console"
    _error_console: "rich_console.Console"

    def __init__(
        self,
        source: pathlib.Path,
        model_name: str,
        remote_provider: b10_remote.BasetenRemote,
        console: "rich_console.Console",
        error_console: "rich_console.Console",
    ) -> None:
        self._source = source
        self._model_name = model_name
        self._remote_provider = remote_provider
        self._console = console
        self._error_console = error_console
        self._ignore_patterns, self._watch_filter = _create_watch_filter(
            source.absolute().parent
        )

        dev_version = b10_core.get_dev_version(self._remote_provider.api, model_name)
        if not dev_version:
            raise b10_errors.RemoteError(
                "No development model found. Run `truss push` then try again."
            )

    def _patch(self) -> None:
        exception_raised = None
        with (
            log_utils.LogInterceptor() as log_interceptor,
            self._console.status(" Live Patching Model.\n", spinner="arrow3"),
        ):
            try:
                gen_truss_path = code_gen.gen_truss_model_from_source(self._source)
                return self._remote_provider.patch(
                    gen_truss_path,
                    self._ignore_patterns,
                    self._console,
                    self._error_console,
                )
            except Exception as e:
                exception_raised = e
            finally:
                logs = log_interceptor.get_logs()

        _handle_intercepted_logs(logs, self._console)
        if exception_raised:
            _handle_import_error(exception_raised, self._console, self._error_console)

    def watch(self) -> None:
        # Perform one initial patch at startup.
        self._patch()
        self._console.print("👀 Watching for new changes.", style="blue")

        # TODO(nikhil): Improve detection of directory structure, since right now
        # we assume a flat structure
        root_dir = self._source.absolute().parent
        for _ in watchfiles.watch(
            root_dir, watch_filter=self._watch_filter, raise_interrupt=False
        ):
            self._patch()
            self._console.print("👀 Watching for new changes.", style="blue")


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
    _included_chainlets: set[str]

    def __init__(
        self,
        source: pathlib.Path,
        entrypoint: Optional[str],
        name: Optional[str],
        remote: str,
        console: "rich_console.Console",
        error_console: "rich_console.Console",
        show_stack_trace: bool,
        included_chainlets: Optional[list[str]],
    ) -> None:
        self._source = source
        self._entrypoint = entrypoint
        self._console = console
        self._error_console = error_console
        self._show_stack_trace = show_stack_trace
        self._remote_provider = cast(
            b10_remote.BasetenRemote, remote_factory.RemoteFactory.create(remote=remote)
        )
        with framework.ChainletImporter.import_target(
            source, entrypoint
        ) as entrypoint_cls:
            self._deployed_chain_name = name or entrypoint_cls.__name__
            self._chain_root = _get_chain_root(entrypoint_cls)
            chainlet_names = set(
                desc.display_name
                for desc in _get_ordered_dependencies([entrypoint_cls])
            )

        if included_chainlets:
            if not_matched := (set(included_chainlets) - chainlet_names):
                raise public_types.ChainsDeploymentError(
                    "Requested to watch specific chainlets, but did not find "
                    f"{not_matched} among available chainlets {chainlet_names}."
                )
            self._included_chainlets = set(included_chainlets)
        else:
            self._included_chainlets = chainlet_names

        chain_id = b10_core.get_chain_id_by_name(
            self._remote_provider.api, self._deployed_chain_name
        )
        if not chain_id:
            raise public_types.ChainsDeploymentError(
                f"Chain `{chain_id}` was not found."
            )
        self._status_page_url = b10_service.URLConfig.status_page_url(
            self._remote_provider.remote_url, b10_service.URLConfig.CHAIN, chain_id
        )
        chain_deployment = b10_core.get_dev_chain_deployment(
            self._remote_provider.api, chain_id
        )
        if chain_deployment is None:
            raise public_types.ChainsDeploymentError(
                f"No development deployment was found for Chain `{chain_id}`. "
                "You cannot live-patch production deployments. Check the Chain's "
                f"status page for available deployments: {self._status_page_url}."
            )
        deployed_chainlets = self._remote_provider.get_chainlets(chain_deployment["id"])
        non_draft_chainlets = [
            chainlet.name for chainlet in deployed_chainlets if not chainlet.is_draft
        ]
        assert not (non_draft_chainlets), (
            "If the chain is draft, the oracles must be draft."
        )

        self._chainlet_data = {c.name: c for c in deployed_chainlets}
        self._assert_chainlet_names_same(chainlet_names)
        self._ignore_patterns, self._watch_filter = _create_watch_filter(
            self._chain_root
        )

    @property
    def _original_chainlet_names(self) -> set[str]:
        return set(self._chainlet_data.keys())

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

        raise public_types.ChainsDeploymentError("\n".join(msg_parts))

    def _code_gen_and_patch_thread(
        self, descr: private_types.ChainletAPIDescriptor
    ) -> tuple[b10_remote.PatchResult, list[str]]:
        with log_utils.LogInterceptor() as log_interceptor:
            # TODO: Maybe try-except code_gen errors explicitly.
            chainlet_dir = code_gen.gen_truss_chainlet(
                self._chain_root,
                self._deployed_chain_name,
                descr,
                self._chainlet_data[descr.display_name].oracle_name,
                use_local_src=False,
            )
            patch_result = self._remote_provider.patch_for_chainlet(
                chainlet_dir, self._ignore_patterns
            )
            logs = log_interceptor.get_logs()
        return patch_result, logs

    def _patch(self, executor: concurrent.futures.Executor) -> None:
        exception_raised = None
        stack_trace = ""
        with (
            log_utils.LogInterceptor() as log_interceptor,
            self._console.status(" Live Patching Chain.\n", spinner="arrow3"),
        ):
            # Handle import errors gracefully (e.g. if user saved file, but there
            # are syntax errors, undefined symbols etc.).
            try:
                with framework.ChainletImporter.import_target(
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
                        if chainlet_descr.display_name not in self._included_chainlets:
                            self._console.print(
                                f"⏩ Skipping patching `{chainlet_descr.display_name}`.",
                                style="grey50",
                            )
                            continue

                        future = executor.submit(
                            self._code_gen_and_patch_thread, chainlet_descr
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

        _handle_intercepted_logs(logs, self._console)
        if exception_raised:
            _handle_import_error(
                exception_raised,
                self._console,
                self._error_console,
                stack_trace=stack_trace if self._show_stack_trace else None,
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
                    f"✅ Patched Chainlet `{display_name}`.{logs_output}", style="green"
                )
            elif patch_result.status == b10_remote.PatchStatus.SKIPPED:
                self._console.print(
                    f"💤 Nothing to do for Chainlet `{display_name}`.{logs_output}",
                    style="grey50",
                )
            else:
                has_errors = True
                self._error_console.print(
                    f"❌ Failed to patch Chainlet `{display_name}`. "
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

    def watch(self) -> None:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Perform one initial patch at startup.
            self._patch(executor)
            self._console.print("👀 Watching for new changes.", style="blue")
            for _ in watchfiles.watch(
                self._chain_root, watch_filter=self._watch_filter, raise_interrupt=False
            ):
                self._patch(executor)
                self._console.print("👀 Watching for new changes.", style="blue")


@framework.raise_validation_errors_before
def watch(
    source: pathlib.Path,
    entrypoint: Optional[str],
    name: Optional[str],
    remote: str,
    console: "rich_console.Console",
    error_console: "rich_console.Console",
    show_stack_trace: bool,
    included_chainlets: Optional[list[str]],
) -> None:
    console.print(
        (
            "👀 Starting to watch for Chain source code and applying live patches "
            "when changes are detected."
        ),
        style="blue",
    )
    patcher = _Watcher(
        source,
        entrypoint,
        name,
        remote,
        console,
        error_console,
        show_stack_trace,
        included_chainlets,
    )
    patcher.watch()


def watch_model(
    source: pathlib.Path,
    model_name: str,
    remote_provider: b10_remote.TrussRemote,
    console: "rich_console.Console",
    error_console: "rich_console.Console",
):
    patcher = _ModelWatcher(
        source=source,
        model_name=model_name,
        remote_provider=cast(b10_remote.BasetenRemote, remote_provider),
        console=console,
        error_console=error_console,
    )
    patcher.watch()
