import copy
import glob
import json
import logging
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.error import HTTPError

import requests
import yaml
from requests import exceptions
from requests.exceptions import ConnectionError
from requests.models import Response
from tenacity import (
    RetryError,
    Retrying,
    retry,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    stop_after_delay,
    wait_fixed,
)

from truss.constants import (
    INFERENCE_SERVER_PORT,
    TRUSS,
    TRUSS_DIR,
    TRUSS_HASH,
    TRUSS_MODIFIED_TIME,
)
from truss.contexts.image_builder.serving_image_builder import (
    ServingImageBuilderContext,
)
from truss.contexts.local_loader.load_model_local import LoadModelLocal
from truss.decorators import proxy_to_shadow_if_scattered
from truss.docker import (
    Docker,
    DockerStates,
    get_container_logs,
    get_container_state,
    get_containers,
    get_images,
    get_urls_from_container,
    kill_containers,
)
from truss.errors import ContainerIsDownError, ContainerNotFoundError
from truss.local.local_config_handler import LocalConfigHandler
from truss.notebook import is_notebook_or_ipython
from truss.patch.calc_patch import calc_truss_patch
from truss.patch.hash import directory_content_hash
from truss.patch.signature import calc_truss_signature
from truss.patch.types import TrussSignature
from truss.readme_generator import generate_readme
from truss.templates.shared.serialization import (
    truss_msgpack_deserialize,
    truss_msgpack_serialize,
)
from truss.trt_llm.validation import validate
from truss.truss_config import BaseImage, ExternalData, ExternalDataItem, TrussConfig
from truss.truss_spec import TrussSpec
from truss.types import Example, PatchDetails, PatchRequest
from truss.util.path import (
    copy_file_path,
    copy_tree_path,
    get_max_modified_time_of_dir,
    load_trussignore_patterns,
)
from truss.validation import validate_secret_name

logger: logging.Logger = logging.getLogger(__name__)

if is_notebook_or_ipython():
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))


class TrussHandle:
    def __init__(self, truss_dir: Path, validate: bool = True) -> None:
        self._truss_dir = truss_dir
        self._spec = TrussSpec(truss_dir)
        self._hash_for_mod_time: Optional[Tuple[float, str]] = None
        if validate:
            self.validate()

    def validate(self):
        self._validate_external_packages()
        self._validate_extensions()

    @property
    def spec(self) -> TrussSpec:
        return self._spec

    @staticmethod
    @retry(
        stop=stop_after_delay(20),
        wait=wait_fixed(1),
        retry=(
            retry_if_result(lambda response: response.status_code == 503)
            | retry_if_exception_type(exceptions.ConnectionError)
        ),
    )
    def _wait_for_predict(
        model_base_url: str, request: Dict, binary: bool = False
    ) -> Response:
        url = f"{model_base_url}/v1/models/model:predict"

        if binary:
            binary_data = truss_msgpack_serialize(request)

            return requests.post(
                url,
                data=binary_data,
                headers={"Content-Type": "application/octet-stream"},
            )

        return requests.post(url, json=request)

    @proxy_to_shadow_if_scattered
    def build_docker_build_context(self, build_dir: Optional[Path] = None):
        build_dir_path = Path(build_dir) if build_dir is not None else None
        image_builder = ServingImageBuilderContext.run(self._truss_dir)
        image_builder.prepare_image_build_dir(build_dir_path)

    def build_docker_image(self, *args, **kwargs):
        """[Deprected] Please use build_serving_docker_image."""
        return self.build_serving_docker_image(*args, **kwargs)

    @proxy_to_shadow_if_scattered
    def build_serving_docker_image(
        self,
        build_dir: Optional[Path] = None,
        tag: Optional[str] = None,
        cache: bool = True,
        network: Optional[str] = None,
    ):
        image = self._build_image(
            builder_context=ServingImageBuilderContext,
            labels=self._get_serving_lookup_labels(),
            build_dir=build_dir,
            tag=tag,
            cache=cache,
            network=network,
        )
        self._store_signature()
        return image

    def get_docker_image(self, labels: Dict):
        """[Deprecated] Do not use."""
        return _docker_image_from_labels(labels)

    @proxy_to_shadow_if_scattered
    def docker_run(
        self,
        build_dir: Optional[Path] = None,
        tag: Optional[str] = None,
        local_port: int = INFERENCE_SERVER_PORT,
        detach=True,
        patch_ping_url: Optional[str] = None,
        wait_for_server_ready: bool = True,
        network: Optional[str] = None,
        container_name_prefix: Optional[str] = None,
    ):
        """
        Builds a docker image and runs it as a container. For control trusses,
        tries to patch.

        Args:
            build_dir: Directory to use for creating docker build context.
            tag: Tags to apply to docker image.
            local_port: Local port to forward inference server to.
            detach: Run docker container in detached mode.
            patch_ping_url:  Mostly for testing, if supplied then a live
                             reload capable truss queries for truss changes
                             by hitting this url.
            wait_for_server_ready: If true, wait for server to pass readiness
              probe before returning.
            network: docker network name.
            container_name_prefix: optional docker container name prefix.

        Returns:
            Container, which can be used to get information about the running,
            including its id. The id can be used to kill the container.
        """
        from python_on_whales.exceptions import DockerException

        container_if_patched = self._try_patch()
        if container_if_patched is not None:
            container = container_if_patched
        else:
            image = self.build_serving_docker_image(
                build_dir=build_dir, tag=tag, network=network
            )
            secrets_mount_dir_path = _prepare_secrets_mount_dir()
            publish_ports = [[local_port, INFERENCE_SERVER_PORT]]

            # We are going to try running a new container, make sure previous one is gone
            self.kill_container()
            labels = self._get_serving_labels()

            envs = {}
            if patch_ping_url is not None:
                envs["PATCH_PING_URL_TRUSS"] = patch_ping_url

            if container_name_prefix:
                suffix = str(uuid.uuid4()).split("-")[0]
                name = f"{container_name_prefix}-{suffix}"
            else:
                name = None

            def _run_docker(gpus: Optional[str] = None):
                return Docker.client().run(
                    image.id,
                    publish=publish_ports,
                    detach=detach,
                    labels=labels,
                    mounts=[
                        [
                            "type=bind",
                            f"src={str(secrets_mount_dir_path)}",
                            "target=/secrets",
                        ],
                        [
                            "type=bind",
                            f"src={str(LocalConfigHandler.bptr_data_resolution_dir_path())}",
                            "target=/bptr",
                        ],
                    ],
                    gpus=gpus,
                    envs=envs,
                    add_hosts=[("host.docker.internal", "host-gateway")],
                    name=name,
                )

            try:
                container = _run_docker(
                    "all" if self._spec.config.resources.use_gpu else None
                )
            except DockerException:
                # This is in the case of testing where the Codespace doesn't have a GPU
                # and we need to run it anyways
                logger.warn("No GPU is available to docker. Running without a GPU.")
                container = _run_docker(None)

            logger.info(
                f"Model server started on port {local_port}, docker container id {container.id}"
            )
        model_base_url = f"http://localhost:{local_port}/v1/models/model"
        try:
            wait_for_truss(model_base_url, container, wait_for_server_ready)
        except ContainerNotFoundError as err:
            raise err
        except (ContainerIsDownError, HTTPError, ConnectionError) as err:
            logger.error(self.serving_container_logs(follow=False, stream=False))
            raise err

        return container

    def predict(
        self,
        request: Dict,
        use_docker: bool = False,
        build_dir: Optional[Path] = None,
        tag: Optional[str] = None,
        local_port: int = INFERENCE_SERVER_PORT,
        detach: bool = True,
        patch_ping_url: Optional[str] = None,
    ):
        if use_docker:
            return self.docker_predict(
                request,
                build_dir=build_dir,
                tag=tag,
                local_port=local_port,
                detach=detach,
                patch_ping_url=patch_ping_url,
            )
        else:
            return self.server_predict(request)

    def server_predict(self, request: Dict):
        """Run the prediction flow locally."""
        model = LoadModelLocal.run(self._truss_dir)
        return _prediction_flow(model, request)

    @proxy_to_shadow_if_scattered
    def docker_predict(
        self,
        request: Dict,
        build_dir: Optional[Path] = None,
        tag: Optional[str] = None,
        local_port: int = INFERENCE_SERVER_PORT,
        detach: bool = True,
        patch_ping_url: Optional[str] = None,
        binary: bool = False,
        stream: bool = False,
        network: Optional[str] = None,
    ):
        """
        Builds docker image, runs that as a docker container
        and makes a prediction request to the server running on the container.
        Kills the container afterwards. Mostly useful for testing.

        Args:
            request: Input to the predict function of model truss.
            build_dir: Directory to use for creating docker build context.
            tag: Tags to apply to docker image.
            local_port: Local port to forward inference server to.
            detach: Run docker container in detached mode.
            patch_ping_url:  Mostly for testing, if supplied then a live
                             reload capable truss queries for truss changes
                             by hitting this url.
            binary: Use msgpack to serialize the response
        """
        containers = self.get_serving_docker_containers_from_labels()
        if containers:
            container = containers[0]
        else:
            container = self.docker_run(
                build_dir,
                tag,
                local_port=local_port,
                detach=detach,
                patch_ping_url=patch_ping_url,
                network=network,
            )
        model_base_url = _get_url_from_container(container)

        resp = TrussHandle._wait_for_predict(model_base_url, request, binary)

        if resp.status_code == 500:
            raise requests.exceptions.HTTPError("500 error", response=resp)

        if resp.headers.get("transfer-encoding") == "chunked":
            # Streaming responses come back just as bytes, so we don't make assumptions
            # about the format being JSON or msgpack.
            return resp.content

        if binary:
            return truss_msgpack_deserialize(resp.content)

        return resp.json()

    @proxy_to_shadow_if_scattered
    def docker_build_setup(
        self, build_dir: Optional[Path] = None, use_hf_secret: bool = False
    ):
        """
        Set up a directory to build docker image from.

        Returns:
            docker build command.
        """
        image_builder = ServingImageBuilderContext.run(self._truss_dir)
        image_builder.prepare_image_build_dir(build_dir, use_hf_secret=use_hf_secret)
        return image_builder.docker_build_command(build_dir)

    def add_python_requirement(self, python_requirement: str):
        """Add a python requirement to truss model's config."""

        self._update_config(
            lambda conf: replace(
                conf, requirements=[*conf.requirements, python_requirement]
            )
        )

    def remove_python_requirement(self, python_requirement: str):
        """Remove a python requirement to truss model's config.

        Note that the requirement has to match exactly, with version included.
        TODO(pankaj) Make this work with just the name of python_requirement.
        """
        self._update_config(
            lambda conf: replace(
                conf,
                requirements=[
                    req for req in conf.requirements if req != python_requirement
                ],
            )
        )

    def add_environment_variable(self, env_var_name: str, env_var_value: str):
        """Add an environment variable to truss model's config."""
        if not env_var_value:
            logger.info("Enviroment value should not empty or none!")
            return

        self._update_config(
            lambda conf: replace(
                conf,
                environment_variables={
                    **conf.environment_variables,
                    env_var_name: env_var_value,
                },
            )
        )

    def add_secret(self, secret_name: str, default_secret_value: str = ""):
        validate_secret_name(secret_name)
        self._update_config(
            lambda conf: replace(
                conf,
                secrets={
                    **conf.secrets,
                    secret_name: default_secret_value,
                },
            )
        )

    def add_external_data_item(
        self,
        url: str,
        local_data_path: str,
        backend: Optional[str] = None,
        name: Optional[str] = None,
    ):
        # todo: write tests for this
        item = ExternalDataItem(url=url, local_data_path=local_data_path)
        if backend is not None:
            item = replace(item, backend=backend)
        if name is not None:
            item = replace(item, name=name)

        current_external_data: ExternalData = (
            self._spec.config.external_data or ExternalData([])
        )
        new_external_data = replace(
            current_external_data,
            items=current_external_data.items + [item],
        )
        self._update_config(
            lambda conf: replace(
                conf,
                external_data=new_external_data,
            )
        )

    def remove_all_external_data(self):
        self._update_config(
            lambda conf: replace(
                conf,
                external_data=None,
            )
        )

    def update_requirements(self, requirements: List[str]):
        """Update requirements in truss model's config.

        Replaces requirements in truss model's config with the provided list.
        """
        self._update_config(lambda conf: replace(conf, requirements=requirements))

    def update_requirements_from_file(self, requirements_filepath: str):
        """Update requirements in truss model's config.

        Replaces requirements in truss model's config with those from the file
        at the given path.
        """
        with Path(requirements_filepath).open() as req_file:
            self.update_requirements(
                [
                    line.strip()
                    for line in req_file.readlines()
                    if not line.strip().startswith("#")
                ]
            )

    def add_system_package(self, system_package: str):
        """Add a system package requirement to truss model's config."""
        self._update_config(
            lambda conf: replace(
                conf, system_packages=[*conf.system_packages, system_package]
            )
        )

    def remove_system_package(self, system_package: str):
        """Remove a system package requirement from truss model's config."""
        self._update_config(
            lambda conf: replace(
                conf,
                system_packages=[
                    pkg for pkg in conf.system_packages if pkg != system_package
                ],
            )
        )

    def add_data(self, file_dir_or_glob: str):
        """Add data to a truss model.

        Accepts a file path, a directory path or a glob. Everything is copied
        under the truss model's data directory.
        """
        self._copy_files(file_dir_or_glob, self._spec.data_dir)

    def add_bundled_package(self, file_dir_or_glob: str):
        """Add a bundled package to a truss model.

        Accepts a file path, a directory path or a glob. Everything is copied
        under the truss model's packages directory.
        """
        self._copy_files(file_dir_or_glob, self._spec.bundled_packages_dir)

    def add_external_package(self, external_dir_path: str):
        self._update_config(
            lambda conf: replace(
                conf,
                external_package_dirs=[*conf.external_package_dirs, external_dir_path],
            )
        )

    def clear_external_packages(self):
        self._update_config(lambda conf: replace(conf, external_package_dirs=[]))

    def examples(self) -> List[Example]:
        """List truss model's examples.

        Examples are a simple `name to input` dictionary.
        """
        return self._spec.examples

    def update_examples(self, examples: List[Example]):
        """Update truss model's examples.

        Existing examples are replaced whole with the given ones.
        """
        with self._spec.examples_path.open("w") as examples_file:
            examples_to_write = [example.to_dict() for example in examples]
            examples_file.write(yaml.dump(examples_to_write))

    def example(self, name_or_index: Union[str, int]) -> Example:
        """Return lookup an example by name or index.

        Index is 0 based. e.g. example(0) returns the first example.
        """
        examples = self.examples()
        if isinstance(name_or_index, str):
            example_name = name_or_index
            index = _find_example_by_name(examples, example_name)
            if index is None:
                raise ValueError(f"No example named {example_name} was found.")
            return examples[index]
        return self.examples()[name_or_index]

    def add_example(self, example_name: str, example_input: Dict):
        """Add example for truss model.

        If the example with the given name already exists then it is overwritten.
        """
        examples = copy.deepcopy(self.examples())
        index = _find_example_by_name(self.examples(), example_name)
        if index is None:
            examples.append(Example(example_name, example_input))
        else:
            examples[index] = Example(example_name, example_input)
        self.update_examples(examples)

    @proxy_to_shadow_if_scattered
    def get_all_docker_images(self):
        """Returns all docker images for this truss.

        Includes images created for previous state of the truss.
        """
        return get_images({TRUSS_DIR: str(self._truss_dir)})

    @proxy_to_shadow_if_scattered
    def get_docker_containers_from_labels(self, *args, **kwargs):
        """[Deprecated] Please use get_serving_docker_containers_from_labels."""
        return self.get_serving_docker_containers_from_labels(*args, **kwargs)

    @proxy_to_shadow_if_scattered
    def get_serving_docker_containers_from_labels(
        self,
        all: bool = False,
        labels: Optional[dict] = None,
    ) -> list:
        """Get serving docker containers, with given labels.

        Args:
            labels: Labels to match on. If none then use labels for this specific truss.
            all: If true return both running and not running containers.
        """
        if labels is None:
            labels = self._get_serving_lookup_labels()
        else:
            # Make sure we're looking for serving container for this truss.
            labels = {
                TRUSS: True,
                **labels,
            }

        return sorted(get_containers(labels, all=all), key=lambda c: c.created)

    def get_running_serving_container_ignore_hash(self):
        containers = self.get_serving_docker_containers_from_labels(
            labels={TRUSS_DIR: str(self._truss_dir)}
        )
        if containers is not None and len(containers) > 0:
            return containers[0]

    @proxy_to_shadow_if_scattered
    def kill_container(self):
        """Kill container

        Killing is done based on directory of the truss.
        """
        kill_containers({TRUSS_DIR: self._truss_dir})

    @proxy_to_shadow_if_scattered
    def container_logs(self, *args, **kwargs):
        """[Deprecate] Use serving_container_logs."""
        return self.serving_container_logs(*args, **kwargs)

    @proxy_to_shadow_if_scattered
    def serving_container_logs(self, follow=True, stream=True):
        """Get container logs for truss."""
        containers = self.get_serving_docker_containers_from_labels(all=True)
        if not containers:
            raise ValueError("No Container is running for truss!")
        return get_container_logs(containers[-1], follow, stream)

    def enable_gpu(self):
        """Enable gpu use for given model.

        This is suggestive, model serving environment may still use cpu, e.g. if
        the setup doesn't have access to a GPU.

        Note that truss would typically use a larger docker base image when this
        is enabled, for example to include the cuda libraries.
        """

        def enable_gpu_fn(conf: TrussConfig):
            new_resources = replace(conf.resources, use_gpu=True)
            return replace(conf, resources=new_resources)

        self._update_config(enable_gpu_fn)

    def set_base_image(self, image: str, python_executable_path: str):
        """Set the base image for a given truss"""

        def define_base_image_fn(conf: TrussConfig):
            if conf.base_image:
                new_base_image = replace(
                    conf.base_image,
                    image=image,
                    python_executable_path=python_executable_path,
                )
                return replace(conf, base_image=new_base_image)
            new_base_image = BaseImage(
                image,
                python_executable_path,
            )
            return replace(conf, base_image=new_base_image)

        self._update_config(define_base_image_fn)

    @proxy_to_shadow_if_scattered
    def patch_container(self, patch_request: PatchRequest):
        """Patch changes onto the container running this Truss.

        Useful for local incremental development.
        """
        if not self.spec.live_reload:
            raise ValueError("Not a control truss: applying patch is not supported.")

        # Note that we match on only the truss directory, not hash.
        container = self.get_running_serving_container_ignore_hash()
        if not container:
            raise ValueError(
                "Only running trusses can be patched: no running containers found for this truss."
            )

        model_base_url = _get_url_from_container(container)
        resp = requests.post(
            f"{model_base_url}/control/patch", json=patch_request.to_dict()
        )
        resp.raise_for_status()
        return resp.json()

    def truss_hash_on_container(self) -> Optional[str]:
        """[Deprecated] Use truss_hash_on_serving_container."""
        return self.truss_hash_on_serving_container()

    @proxy_to_shadow_if_scattered
    def truss_hash_on_serving_container(self) -> Optional[str]:
        """Get content hash of truss running on container."""
        if not self.spec.live_reload:
            raise ValueError(
                "Not a control truss fetching truss hash is not supported."
            )

        container = self.get_running_serving_container_ignore_hash()
        model_base_url = _get_url_from_container(container)
        resp = requests.get(f"{model_base_url}/control/truss_hash")
        resp.raise_for_status()
        respj = resp.json()
        if "error" in respj:
            logger.error("Unable to get hash of running container.")
            return None
        return respj["result"]

    def update_python_version(self, python_version: str):
        inferred_python_version = python_version
        if not python_version.startswith("py"):
            # support 3.9 style versions
            version_parts = python_version.split(".")
            inferred_python_version = f"py{version_parts[0]}{version_parts[1]}"

        self._update_config(
            lambda conf: replace(
                conf,
                python_version=inferred_python_version,
            )
        )

    def _control_serving_container_has_partially_applied_patch(self) -> Optional[bool]:
        """Check if there is a partially applied patch on the running live_reload capable container."""
        if not self.spec.live_reload:
            raise ValueError("Not a control truss, operation not supported.")

        container = self.get_running_serving_container_ignore_hash()
        model_base_url = _get_url_from_container(container)
        resp = requests.get(f"{model_base_url}/control/has_partially_applied_patch")
        resp.raise_for_status()
        respj = resp.json()
        if "error" in respj:
            logger.error(
                "Unable to check if control truss container has partially applied patch."
            )
            return None
        return respj["result"]

    @property
    def is_control_truss(self):
        return self._spec.live_reload

    @proxy_to_shadow_if_scattered
    def get_urls_from_truss(self):
        urls = []
        containers = self.get_serving_docker_containers_from_labels()
        for container in containers:
            urls.extend(get_urls_from_container(container)[INFERENCE_SERVER_PORT])
        return urls

    def generate_readme(self):
        return generate_readme(self._spec)

    def update_description(self, description: str):
        self._update_config(lambda conf: replace(conf, description=description))

    def live_reload(self, enable: bool = True):
        """Enable control plane.

        Control plane allows loading truss changes into the running model
        container. This is useful during development to iterate on model changes
        quickly.
        """

        def enable_live_reload_fn(conf: TrussConfig):
            return replace(conf, live_reload=enable)

        self._update_config(enable_live_reload_fn)

    @proxy_to_shadow_if_scattered
    def calc_patch(
        self, prev_truss_hash: str, truss_ignore_patterns: List[str]
    ) -> Optional[PatchDetails]:
        """Calculates patch of current truss from previous.

        Returns None if signature cannot be found locally for previous truss hash
        or if the change cannot be expressed with currently supported patches.
        """
        prev_sign_str = LocalConfigHandler.get_signature(prev_truss_hash)
        if prev_sign_str is None:
            logger.info(f"Signature not found for truss for hash {prev_truss_hash}")
            return None
        prev_sign = TrussSignature.from_dict(json.loads(prev_sign_str))
        ignore_patterns = truss_ignore_patterns + self._spec.hash_ignore_patterns
        patch_ops = calc_truss_patch(self._truss_dir, prev_sign, ignore_patterns)
        if patch_ops is None:
            return None

        return PatchDetails(
            prev_signature=prev_sign,
            prev_hash=prev_truss_hash,
            next_hash=directory_content_hash(self._truss_dir, ignore_patterns),
            next_signature=calc_truss_signature(self._truss_dir, ignore_patterns),
            patch_ops=patch_ops,
        )

    def gather(self) -> Path:
        """Convert a Truss with external dependencies into one without.

        Any external packages are copied under packages folder to form a Truss,
        where no parts of the Truss are outside the Truss folder. If the Truss
        doesn't have any external dependencies then this returns the handle to
        itself. Otherwise, a new truss is created with external dependencies
        gatherer and a handle to that truss is returned. These gathered trusses
        are caches and resused.
        """
        from truss.truss_gatherer import gather

        if not self.is_scattered():
            return self._truss_dir

        return gather(self._truss_dir)

    @property
    def max_modified_time(self) -> float:
        """Max modified time of all the files and directories that this Truss spans."""
        max_mod_time = get_max_modified_time_of_dir(self._truss_dir)
        if self.no_external_packages:
            return max_mod_time

        for path in self.spec.external_package_dirs_paths:
            max_mod_time_for_path = get_max_modified_time_of_dir(path)
            if max_mod_time_for_path > max_mod_time:
                max_mod_time = max_mod_time_for_path
        return max_mod_time

    @property
    def no_external_packages(self) -> bool:
        return len(self.spec.config.external_package_dirs) == 0

    def is_scattered(self) -> bool:
        """A scattered truss is one where parts of it are outside the truss directory.

        Many operations require a scattered truss to be gathered first.
        """
        return not self.no_external_packages

    def _store_signature(self):
        """Store truss signature"""
        sign = calc_truss_signature(self._truss_dir)
        truss_hash = self._serving_hash()
        LocalConfigHandler.add_signature(truss_hash, json.dumps(sign.to_dict()))

    def _copy_files(self, file_dir_or_glob: str, destination_dir: Path):
        item = file_dir_or_glob
        item_path = Path(item)
        if item_path.is_dir():
            copy_tree_path(item_path, destination_dir / item_path.name)
        else:
            filenames = glob.glob(item)
            for filename in filenames:
                filepath = Path(filename)
                copy_file_path(filepath, destination_dir / filepath.name)

    def _get_serving_labels(self) -> Dict[str, Any]:
        truss_mod_time = get_max_modified_time_of_dir(self._truss_dir)
        return {
            **self._get_serving_lookup_labels(),
            TRUSS_MODIFIED_TIME: truss_mod_time,
        }

    def _get_serving_lookup_labels(self) -> Dict[str, Any]:
        return {
            TRUSS_DIR: self._truss_dir,
            TRUSS_HASH: self._serving_hash(),
            TRUSS: True,
        }

    def _build_image(
        self,
        builder_context,
        labels: Dict[str, str],
        build_dir: Optional[Path] = None,
        tag: Optional[str] = None,
        cache: bool = True,
        network: Optional[str] = None,
    ):
        image = _docker_image_from_labels(labels=labels)
        if image is not None:
            return image

        build_dir_path = Path(build_dir) if build_dir is not None else None
        image_builder = builder_context.run(self._truss_dir)
        build_image_result = image_builder.build_image(
            build_dir_path, tag, labels=labels, cache=cache, network=network
        )
        return build_image_result

    def _update_config(self, update_config_fn: Callable[[TrussConfig], TrussConfig]):
        config = update_config_fn(self._spec.config)
        config.write_to_yaml_file(self._spec.config_path)
        # reload spec
        self._spec = TrussSpec(self._truss_dir)

    def _try_patch(self):
        if not self.is_control_truss:
            return None

        container = self.get_running_serving_container_ignore_hash()
        if container is None:
            return None

        running_truss_hash = self.truss_hash_on_serving_container()
        if running_truss_hash is None:
            return None

        current_hash = self._serving_hash()
        if running_truss_hash == current_hash:
            has_partially_applied_patch = (
                self._control_serving_container_has_partially_applied_patch()
            )
            if has_partially_applied_patch is True:
                return None
            else:
                return container

        logger.info(
            "Truss supports patching and a running "
            "container found: attempting to patch the container"
        )
        truss_ignore_patterns = load_trussignore_patterns()
        patch_details = self.calc_patch(running_truss_hash, truss_ignore_patterns)
        if patch_details is None:
            logger.info("Unable to calculate patch.")
            return None

        if patch_details.is_empty():
            logger.info(
                "While truss has changed, no serving related "
                "changes were found, skipping patching."
            )
            return container

        patch_request = PatchRequest(
            hash=current_hash,
            prev_hash=running_truss_hash,
            patches=patch_details.patch_ops,
        )
        resp = self.patch_container(patch_request)
        if "error" in resp:
            raise RuntimeError(f'Failed to patch control truss {resp["error"]}')
        self._store_signature()
        return container

    def _serving_hash(self) -> str:
        """Hash to be used for the serving image.

        Caches based on max mod time of files in truss. If truss is not touched
        then this avoids calculating the hash, which could be expensive for large
        model binaries.
        """
        truss_mod_time = get_max_modified_time_of_dir(self._truss_dir)
        # If mod time hasn't changed then hash must be the same
        if (
            self._hash_for_mod_time is not None
            and self._hash_for_mod_time[0] == truss_mod_time
        ):
            truss_hash = self._hash_for_mod_time[1]
        else:
            truss_hash = directory_content_hash(
                self._truss_dir, self._spec.hash_ignore_patterns
            )
            self._hash_for_mod_time = (truss_mod_time, truss_hash)
        return truss_hash

    def _validate_external_packages(self):
        if not self.no_external_packages:
            for path in self._spec.external_package_dirs_paths:
                if not path.exists():
                    raise RuntimeError(
                        f"Truss referes to external package at "
                        f"{path.resolve()} but that path does not exist."
                    )

    def _validate_extensions(self):
        # Only one extenstion right now.
        if self._spec.config.trt_llm is not None:
            validate(self._spec)


def _prediction_flow(model, request: Dict):
    """This flow attempts to mimic the request life-cycle of a server"""
    # TODO: can we just call ModelWrapper directly here?
    if hasattr(model, "preprocess"):
        request = model.preprocess(request)
    response = model.predict(request)
    if hasattr(model, "postprocess"):
        response = model.postprocess(response)
    return response


def _wait_for_docker_build(container) -> None:
    for attempt in Retrying(stop=stop_after_attempt(5), wait=wait_fixed(2)):
        state = get_container_state(container)
        logger.info(f"Container state: {state}")
        if state == DockerStates.OOMKILLED or state == DockerStates.DEAD:
            raise ContainerIsDownError(f"Container errored out in state: {state}.")
        with attempt:
            if state != DockerStates.RUNNING:
                raise ContainerIsDownError(f"Container stuck in state: {state.value}.")


@retry(
    stop=stop_after_delay(120),
    wait=wait_fixed(2),
    retry=(
        retry_if_result(lambda response: response.status_code == 503)
        | retry_if_exception_type(exceptions.ConnectionError)
    ),
)
def _wait_for_model_server(url: str) -> Response:
    return requests.get(url)


def wait_for_truss(
    url: str, container: str, wait_for_server_ready: bool = True
) -> None:
    from python_on_whales.exceptions import NoSuchContainer

    try:
        _wait_for_docker_build(container)
    except NoSuchContainer:
        raise ContainerNotFoundError(message=f"Container {container} was not found")
    except RetryError as retry_err:
        retry_err.reraise()
    if wait_for_server_ready:
        _wait_for_model_server(url)


def _prepare_secrets_mount_dir() -> Path:
    LocalConfigHandler.sync_secrets_mount_dir()
    return LocalConfigHandler.secrets_dir_path()


def _find_example_by_name(examples: List[Example], example_name: str) -> Optional[int]:
    for index, example in enumerate(examples):
        if example.name == example_name:
            return index
    return None


def _get_url_from_container(container) -> str:
    return get_urls_from_container(container)[INFERENCE_SERVER_PORT][0]


def _create_rand_dir_in_dot_truss(subdir: str) -> Path:
    rnd = str(uuid.uuid4())
    target_directory_path = Path(Path.home(), ".truss", subdir, rnd)
    target_directory_path.mkdir(parents=True)
    return target_directory_path


def _docker_image_from_labels(labels: Dict):
    """Get docker image from given labels.

    Assumes there's only one. Returns the first one it finds if there are many,
    no guarantees which one.
    """
    images = get_images(labels)
    if images and isinstance(images, list):
        return images[0]
