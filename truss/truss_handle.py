import copy
import glob
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import requests
import yaml
from tenacity import Retrying, stop_after_attempt, wait_fixed
from truss.constants import (
    INFERENCE_SERVER_PORT,
    TRUSS,
    TRUSS_DIR,
    TRUSS_HASH,
    TRUSS_MODIFIED_TIME,
)
from truss.contexts.image_builder.image_builder import ImageBuilderContext
from truss.contexts.local_loader.load_local import LoadLocal
from truss.docker import (
    Docker,
    get_container_logs,
    get_containers,
    get_images,
    get_urls_from_container,
    kill_containers,
)
from truss.local.local_config_handler import LocalConfigHandler
from truss.notebook import is_notebook_or_ipython
from truss.patch.calc_patch import calc_truss_patch
from truss.patch.hash import directory_content_hash
from truss.patch.signature import calc_truss_signature
from truss.patch.types import TrussSignature
from truss.readme_generator import generate_readme
from truss.truss_config import TrussConfig
from truss.truss_spec import TrussSpec
from truss.types import Example, PatchDetails
from truss.utils import copy_file_path, copy_tree_path, get_max_modified_time_of_dir
from truss.validation import validate_secret_name

logger = logging.getLogger(__name__)

if is_notebook_or_ipython():
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))


class TrussHandle:
    def __init__(self, truss_dir: Path) -> None:
        self._truss_dir = truss_dir
        self._spec = TrussSpec(truss_dir)
        self._hash_for_mod_time: Optional[Tuple[float, str]] = None

    @property
    def spec(self) -> TrussSpec:
        return self._spec

    def server_predict(self, request: dict):
        """Run the prediction flow locally."""
        model = LoadLocal.run(self._truss_dir)
        return _prediction_flow(model, request)

    def build_docker_build_context(self, build_dir: Path = None):
        build_dir_path = Path(build_dir) if build_dir is not None else None
        image_builder = ImageBuilderContext.run(self._truss_dir)
        image_builder.prepare_image_build_dir(build_dir_path)

    def build_docker_image(self, build_dir: Path = None, tag: str = None):
        """Builds docker image"""
        image = self.get_docker_image()
        if image is not None:
            return image
        build_dir_path = Path(build_dir) if build_dir is not None else None
        image_builder = ImageBuilderContext.run(self._truss_dir)
        build_image_result = image_builder.build_image(
            build_dir_path, tag, labels=self._get_labels()
        )
        self._store_signature()
        return build_image_result

    def get_docker_image(self):
        """Get docker image for truss if one exists."""
        images = self.get_docker_images_from_label()
        if images and isinstance(images, list):
            return images[0]

    def docker_run(
        self,
        build_dir: Path = None,
        tag: str = None,
        local_port: int = INFERENCE_SERVER_PORT,
        detach=True,
    ):
        """
        Builds a docker image and runs it as a container. For control trusses,
        tries to patch.

        Args:
            build_dir: Directory to use for creating docker build context. tag:
            Tags to apply to docker image. local_port: Local port to forward
            inference server to. detach: Run docker container in detached mode.

        Returns:
            Container, which can be used to get information about the running,
            including its id. The id can be used to kill the container.
        """
        container_if_patched = self._try_patch()
        if container_if_patched is not None:
            container = container_if_patched
        else:
            image = self.build_docker_image(build_dir=build_dir, tag=tag)
            built_tag = image.repo_tags[0]
            secrets_mount_dir_path = _prepare_secrets_mount_dir()
            publish_ports = [[local_port, INFERENCE_SERVER_PORT]]

            # We are going to try running a new container, make sure previous one is gone
            self.kill_container()
            labels = {
                **self._get_labels(),
                TRUSS: True,
            }
            container = Docker.client().run(
                built_tag,
                publish=publish_ports,
                detach=detach,
                labels=labels,
                mounts=[
                    [
                        "type=bind",
                        f"src={str(secrets_mount_dir_path)}",
                        "target=/secrets",
                    ]
                ],
                gpus="all" if self._spec.config.resources.use_gpu else None,
            )
            logger.info(
                f"Model server started on port {local_port}, docker container id {container.id}"
            )
        model_base_url = f"http://localhost:{local_port}/"
        try:
            _wait_for_model_server(model_base_url)
        except Exception as exc:
            for log in self.container_logs():
                logger.info(log)
            raise exc
        return container

    def docker_predict(
        self,
        request: dict,
        build_dir: Path = None,
        tag: str = None,
        local_port: int = INFERENCE_SERVER_PORT,
        detach: bool = True,
    ):
        """
        Builds docker image, runs that as a docker container
        and makes a prediction request to the server running on the container.
        Kills the container afterwards. Mostly useful for testing.
        """
        containers = self.get_docker_containers_from_labels()
        if containers:
            container = containers[0]
        else:
            container = self.docker_run(
                build_dir,
                tag,
                local_port=local_port,
                detach=detach,
            )
        model_base_url = _get_url_from_container(container)
        resp = requests.post(f"{model_base_url}/v1/models/model:predict", json=request)
        resp.raise_for_status()
        return resp.json()

    def docker_build_setup(self, build_dir: Path = None):
        """
        Set up a directory to build docker image from.

        Returns:
            docker build command.
        """
        image_builder = ImageBuilderContext.run(self._truss_dir)
        image_builder.prepare_image_build_dir(build_dir)
        return image_builder.docker_build_command(build_dir)

    def add_python_requirement(self, python_requirement: str):
        """Add a python requirement to truss model's config."""
        self._update_config(
            lambda conf: replace(
                conf, requirements=[*conf.requirements, python_requirement]
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
            self.update_requirements([line.strip() for line in req_file.readlines()])

    def add_system_package(self, system_package: str):
        """Add a system package requirement to truss model's config."""
        self._update_config(
            lambda conf: replace(
                conf, system_packages=[*conf.system_packages, system_package]
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
        under the truss model's data directory.
        """
        self._copy_files(file_dir_or_glob, self._spec.bundled_packages_dir)

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

    def add_example(self, example_name: str, example_input: dict):
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

    def get_docker_images_from_label(self):
        return get_images(self._get_labels())

    def get_all_docker_images(self):
        """Returns all docker images for this truss.

        Includes images created for previous state of the truss.
        """
        return get_images({TRUSS_DIR: str(self._truss_dir)})

    def get_docker_containers_from_labels(self, all=False, labels=None):
        if labels is None:
            labels = self._get_labels()
        return sorted(get_containers(labels, all=all), key=lambda c: c.created)

    def kill_container(self):
        """Kill container

        Killing is done based on directory of the truss.
        """
        kill_containers({TRUSS_DIR: self._truss_dir})

    def container_logs(self):
        containers = self.get_docker_containers_from_labels(all=True)
        if not containers:
            raise ValueError("No Container is running for truss!")
        return get_container_logs(containers[-1])

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

    def patch_container(self, patch_request: dict):
        """Patch changes onto the container running this Truss.

        Useful for local incremental development.
        TODO(pankaj): Turn patch_request into a dataclass
        """
        if not self.spec.use_control_plane:
            raise ValueError("Not a control truss: applying patch is not supported.")

        containers = self.get_docker_containers_from_labels(
            labels={TRUSS_DIR: str(self._truss_dir)}
        )
        if not containers:
            raise ValueError(
                "Only running trusses can be patched: no running containers found for this truss."
            )

        container = containers[0]
        model_base_url = _get_url_from_container(container)
        resp = requests.post(f"{model_base_url}/control/patch", json=patch_request)
        resp.raise_for_status()
        return resp.json()

    def truss_hash_on_container(self) -> Optional[str]:
        """Get content hash of truss running on container."""
        if not self.spec.use_control_plane:
            raise ValueError(
                "Not a control truss fetching truss hash is not supported."
            )

        containers = self.get_docker_containers_from_labels(
            labels={TRUSS_DIR: str(self._truss_dir)}
        )
        if containers is None or len(containers) == 0:
            return None

        container = containers[0]
        model_base_url = _get_url_from_container(container)
        resp = requests.get(f"{model_base_url}/control/truss_hash")
        resp.raise_for_status()
        respj = resp.json()
        if "error" in respj:
            logger.error("Unable to get hash of running container.")
            return None
        return respj["result"]

    @property
    def is_control_truss(self):
        return self._spec.use_control_plane

    def get_urls_from_truss(self):
        urls = []
        containers = self.get_docker_containers_from_labels()
        for container in containers:
            urls.extend(get_urls_from_container(container)[INFERENCE_SERVER_PORT])
        return urls

    def generate_readme(self):
        return generate_readme(self._spec)

    def update_description(self, description: str):
        self._update_config(lambda conf: replace(conf, description=description))

    def use_control_plane(self, enable: bool = True):
        """Enable control plane.

        Control plane allows loading truss changes into the running model
        container. This is useful during development to iterate on model changes
        quickly.
        """

        def enable_control_plane_fn(conf: TrussConfig):
            return replace(conf, use_control_plane=enable)

        self._update_config(enable_control_plane_fn)

    def calc_patch(self, prev_truss_hash: str) -> Optional[PatchDetails]:
        """Calculates patch of current truss from previous.

        Returns None if signature cannot be found locally for previous truss hash
        or if the change cannot be expressed with currently supported patches.
        """
        prev_sign_str = LocalConfigHandler.get_signature(prev_truss_hash)
        if prev_sign_str is None:
            logger.info(f"Signature not found for truss for hash {prev_truss_hash}")
            return None

        prev_sign = TrussSignature.from_dict(json.loads(prev_sign_str))
        patch_ops = calc_truss_patch(self._truss_dir, prev_sign)
        if patch_ops is None:
            return None

        next_sign = calc_truss_signature(self._truss_dir)
        return PatchDetails(
            prev_signature=prev_sign,
            prev_hash=prev_truss_hash,
            next_hash=directory_content_hash(self._truss_dir),
            next_signature=next_sign,
            patch_ops=patch_ops,
        )

    def _store_signature(self):
        """Store truss signature"""
        sign = calc_truss_signature(self._truss_dir)
        truss_hash = self._truss_hash()
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

    def _get_labels(self):
        truss_mod_time = get_max_modified_time_of_dir(self._truss_dir)
        truss_hash = self._truss_hash()
        return {
            TRUSS_MODIFIED_TIME: truss_mod_time,
            TRUSS_DIR: self._truss_dir,
            TRUSS_HASH: truss_hash,
        }

    def _update_config(self, update_config_fn: Callable[[TrussConfig], TrussConfig]):
        config = update_config_fn(self._spec.config)
        config.write_to_yaml_file(self._spec.config_path)
        # reload spec
        self._spec = TrussSpec(self._truss_dir)

    def _try_patch(self):
        if not self.is_control_truss:
            return None

        containers = self.get_docker_containers_from_labels(
            labels={TRUSS_DIR: str(self._truss_dir)}
        )

        if len(containers) == 0:
            return None

        container = containers[0]
        running_truss_hash = self.truss_hash_on_container()
        if running_truss_hash is None:
            return None

        current_hash = self._truss_hash()
        if running_truss_hash == current_hash:
            return container

        logger.info(
            "Truss supports patching and a running "
            "container found: attempting to patch the container"
        )
        patch_details = self.calc_patch(running_truss_hash)
        if patch_details is None:
            logger.info("Unable to calculate patch.")
            return None

        patch_request = {
            "hash": current_hash,
            "prev_hash": running_truss_hash,
            "patches": [patch.to_dict() for patch in patch_details.patch_ops],
        }
        resp = self.patch_container(patch_request)
        if "error" in resp:
            raise RuntimeError(f'Failed to patch control truss {resp["error"]}')
        self._store_signature()
        return container

    def _truss_hash(self) -> str:
        truss_mod_time = get_max_modified_time_of_dir(self._truss_dir)
        # If mod time hasn't changed then hash must be the same
        if (
            self._hash_for_mod_time is not None
            and self._hash_for_mod_time[0] == truss_mod_time
        ):
            truss_hash = self._hash_for_mod_time[1]
        else:
            truss_hash = directory_content_hash(self._truss_dir)
            self._hash_for_mod_time = (truss_mod_time, truss_hash)
        return truss_hash


def _prediction_flow(model, request: dict):
    """This flow attempts to mimic the request life-cycle of a kfserving server"""
    _validate_request_input(request)
    _map_instances_inputs(request)
    if hasattr(model, "preprocess"):
        request = model.preprocess(request)
    response = model.predict(request)
    if hasattr(model, "postprocess"):
        response = model.postprocess(response)
    return response


def _map_instances_inputs(request: dict):
    # TODO(pankaj) Share this code with baseten deployed code
    if "instances" in request and "inputs" not in request:
        request["inputs"] = request["instances"]
    elif "inputs" in request and "instances" not in request:
        request["instances"] = request["inputs"]
    return request


def _validate_request_input(request: dict):
    # TODO(pankaj) Should these checks be there?
    if _is_invalid_list_input_prop(request, "instances") or _is_invalid_list_input_prop(
        request, "inputs"
    ):
        raise Exception(reason='Expected "instances" or "inputs" to be a list')


def _is_invalid_list_input_prop(request: dict, prop: str):
    return prop in request and not _is_valid_list_type(request[prop])


def _is_valid_list_type(obj) -> bool:
    return isinstance(obj, (list, np.ndarray))


def _wait_for_model_server(url: str):
    for attempt in Retrying(stop=stop_after_attempt(10), wait=wait_fixed(2)):
        with attempt:
            resp = requests.get(url)
            resp.raise_for_status()


def _prepare_secrets_mount_dir() -> Path:
    LocalConfigHandler.sync_secrets_mount_dir()
    return LocalConfigHandler.secrets_dir_path()


def _find_example_by_name(examples: List[Example], example_name: str) -> Optional[int]:
    for index, example in enumerate(examples):
        if example.name == example_name:
            return index


def _get_url_from_container(container) -> str:
    return get_urls_from_container(container)[INFERENCE_SERVER_PORT][0]
