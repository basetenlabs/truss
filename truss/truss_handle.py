import copy
import glob
import logging
from dataclasses import replace
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import requests
import yaml
from tenacity import Retrying, stop_after_attempt, wait_fixed

from truss.constants import TRUSS, TRUSS_DIR, TRUSS_MODIFIED_TIME
from truss.contexts.image_builder.image_builder import ImageBuilderContext
from truss.contexts.local_loader.load_local import LoadLocal
from truss.docker import (Docker, get_container_logs, get_containers,
                          get_images, get_urls_from_container, kill_containers)
from truss.local.local_config_handler import LocalConfigHandler
from truss.readme_generator import generate_readme
from truss.truss_config import TrussConfig
from truss.truss_spec import TrussSpec
from truss.types import Example
from truss.utils import (copy_file_path, copy_tree_path,
                         get_max_modified_time_of_dir)
from truss.validation import validate_secret_name

logger = logging.getLogger(__name__)


class TrussHandle:
    def __init__(self, truss_dir: Path) -> None:
        self._truss_dir = truss_dir
        self._spec = TrussSpec(truss_dir)

    @property
    def spec(self) -> TrussSpec:
        return self._spec

    def server_predict(self, request: dict):
        """Run the prediction flow locally."""
        model = LoadLocal.run(self._truss_dir)
        return _prediction_flow(model, request)

    def build_docker_image(self, build_dir: Path = None, tag: str = None):
        """Builds docker image"""
        images = self.get_docker_images_from_label()
        if images and isinstance(images, list):
            return images[0]
        build_dir_path = Path(build_dir) if build_dir is not None else None
        image_builder = ImageBuilderContext.run(self._truss_dir)
        return image_builder.build_image(build_dir_path, tag, labels=self._get_labels())

    def docker_run(
        self,
        build_dir: Path = None,
        tag: str = None,
        local_port: int = 8080,
        detach=True,
    ):
        """
        Builds a docker image and runs it as a container.

        Returns:
            Container, which can be used to get information about the running,
            including its id. The id can be used to kill the container.
        """
        image = self.build_docker_image(build_dir=build_dir, tag=tag)
        built_tag = image.repo_tags[0]
        labels = self._get_labels()
        labels.update({TRUSS : True})
        secrets_mount_dir_path = _prepare_secrets_mount_dir()
        container = Docker.client().run(
            built_tag,
            publish=[[local_port, 8080]],
            detach=detach,
            labels=labels,
            mounts=[[
                'type=bind',
                f'src={str(secrets_mount_dir_path)}',
                'target=/secrets',
            ]],
            gpus='all' if self._spec.config.resources.use_gpu else None,
        )
        model_base_url = f'http://localhost:{local_port}/'
        try:
            _wait_for_model_server(model_base_url)
        except Exception as e:
            for log in self.container_logs():
                logging.info(log)
            raise e
        logger.info(f'Model server started on port {local_port}, docker container id {container.id}')
        return container

    def docker_predict(
        self,
        request: dict,
        build_dir: Path = None,
        tag: str = None,
        local_port: int = 8080,
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
            container = self.docker_run(build_dir, tag, local_port=local_port, detach=detach)
        model_base_url = get_urls_from_container(container)[0]
        resp = requests.post(f'{model_base_url}/v1/models/model:predict', json=request)
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
        self._update_config(lambda conf: replace(
            conf,
            requirements=[*conf.requirements, python_requirement]))

    def add_environment_variable(self, env_var_name: str, env_var_value: str):
        """Add an environment variable to truss model's config."""
        if not env_var_value:
            logger.info("Enviroment value should not empty or none!")
            return

        self._update_config(lambda conf: replace(
            conf,
            environment_variables={
                **conf.environment_variables,
                env_var_name: env_var_value,
            },
        ))

    def add_secret(self, secret_name: str, default_secret_value: str = ''):
        validate_secret_name(secret_name)
        self._update_config(lambda conf: replace(
            conf,
            secrets={
                **conf.secrets,
                secret_name: default_secret_value,
            },
        ))

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
        self._update_config(lambda conf: replace(
            conf,
            system_packages=[*conf.system_packages, system_package]))

    def add_data(self, file_dir_or_glob: str):
        """Add data to a truss model.

        Accepts a file path, a directory path or a glob. Everything is copied
        under the truss model's data directory.
        """
        item = file_dir_or_glob
        item_path = Path(item)
        if item_path.is_dir():
            copy_tree_path(item_path, self._spec.data_dir / item_path.name)
        else:
            filenames = glob.glob(item)
            for filename in filenames:
                filepath = Path(filename)
                copy_file_path(filepath, self._spec.data_dir / filepath.name)

    def examples(self) -> List[Example]:
        """List truss model's examples.

        Examples are a simple `name to input` dictionary.
        """
        return self._spec.examples

    def update_examples(self, examples: List[Example]):
        """Update truss model's examples.

        Existing examples are replaced whole with the given ones.
        """
        with self._spec.examples_path.open('w') as examples_file:
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
                raise ValueError(f'No example named {example_name} was found.')
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

    def get_docker_containers_from_labels(self, all=False):
        return sorted(get_containers(self._get_labels(), all=all), key=lambda c: c.created)

    def kill_container(self):
        kill_containers(self._get_labels())

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

    def _get_labels(self):
        return {
            TRUSS_MODIFIED_TIME : get_max_modified_time_of_dir(
                self._truss_dir
            ),
            TRUSS_DIR: self._truss_dir,
        }

    def _update_config(self, update_config_fn: Callable[[TrussConfig], TrussConfig]):
        config = update_config_fn(self._spec.config)
        config.write_to_yaml_file(self._spec.config_path)
        # reload spec
        self._spec = TrussSpec(self._truss_dir)

    def get_urls_from_truss(self):
        urls = []
        containers = self.get_docker_containers_from_labels()
        for container in containers:
            urls.extend(get_urls_from_container(container))
        return urls

    def generate_readme(self):
        return generate_readme(self._spec)

    def update_description(self, description: str):
        self._update_config(lambda conf: replace(
            conf,
            description=description))


def _prediction_flow(model, request: dict):
    """This flow attempts to mimic the request life-cycle of a kfserving server"""
    _validate_request_input(request)
    _map_instances_inputs(request)
    if hasattr(model, 'preprocess'):
        request = model.preprocess(request)
    response = model.predict(request)
    if hasattr(model, 'postprocess'):
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
    if _is_invalid_list_input_prop(request, 'instances') \
            or _is_invalid_list_input_prop(request, 'inputs'):
        raise Exception(
            reason="Expected \"instances\" or \"inputs\" to be a list"
        )


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
