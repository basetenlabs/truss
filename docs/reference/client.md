# Truss Python client reference

A list of the functions available with `import truss` and their arguments and properties.

### Truss creation

#### cleanup()

Cleans up .truss directory.

#### from_directory(truss_directory: str) -> truss.truss_handle.TrussHandle

Get a handle to a Truss. A Truss is a build context designed to be built as a container locally or uploaded into a model serving environment.

Args:
* truss_directory (str): The local directory of an existing Truss

Returns:
* TrussHandle

#### init(target_directory: str, data_files: List[str] = None, requirements_file: str = None) -> truss.truss_handle.TrussHandle

Initialize an empty placeholder Truss. A Truss is a build context designed
to be built as a container locally or uploaded into a model serving
environment. This placeholder structure can be filled to represent ML
models.

Args:

* target_directory: Absolute or relative path of the directory to create Truss in. The directory is created if it doesn't exist.

#### kill_all()

#### mk_truss(model: Any, target_directory: str = None, data_files: List[str] = None, requirements_file: str = None) -> truss.truss_handle.TrussHandle

Create a Truss with the given model. A Truss is a build context designed to
be built as a container locally or uploaded into a model serving environment.

Args:

* model (an in-memory model object): A model object to be deployed (e.g. a keras sklearn, or pytorch model object)
* target_directory (str, optional): The local directory target for the Truss. Otherwise a temporary directory will be generated
* data_files (List[str], optional): Additional files required for model operation. Can be a glob that resolves to files for the root directory or a directory path.
* requirements_file (str, optional): A file of packages in a PIP requirements format to be installed in the container environment.

Returns:

* TrussHandle: A handle to the generated Truss that provides easy access to content inside.

### Truss Use

#### bundled_package(self, file_dir_or_glob: str)
      Add a bundled package to a truss model.

      Accepts a file path, a directory path or a glob. Everything is copied
      under the truss model's packages directory.

#### add_data(self, file_dir_or_glob: str)
      Add data to a truss model.

      Accepts a file path, a directory path or a glob. Everything is copied
      under the truss model's data directory.

#### add_environment_variable(self, env_var_name: str, env_var_value: str)
      Add an environment variable to truss model's config.

#### add_example(self, example_name: str, example: dict)
      Add example for truss model.

      If the example with the given name already exists then it is overwritten.

#### add_python_requirement(self, python_requirement: str)
      Add a python requirement to truss model's config.

#### add_secret(self, secret_name: str, default_secret_value: str = '')

#### add_system_package(self, system_package: str)
      Add a system package requirement to truss model's config.

#### build_docker_image(self, build_dir: pathlib.Path = None, tag: str = None)
      Builds docker image

#### container_logs(self)

#### docker_build_setup(self, build_dir: pathlib.Path = None)
      Set up a directory to build docker image from.

      Returns:
          docker build command.

#### docker_predict(self, request: dict, build_dir: pathlib.Path = None, tag: str = None, local_port: int = 8080, detach: bool = True)
      Builds docker image, runs that as a docker container
      and makes a prediction request to the server running on the container.
      Kills the container afterwards. Mostly useful for testing.

#### docker_run(self, build_dir: pathlib.Path = None, tag: str = None, local_port: int = 8080, detach=True)
      Builds a docker image and runs it as a container.

      Returns:
          Container, which can be used to get information about the running,
          including its id. The id can be used to kill the container.

#### enable_gpu(self)
      Enable gpu use for given model.

      This is suggestive, model serving environment may still use cpu, e.g. if
      the setup doesn't have access to a GPU.

      Note that truss would typically use a larger docker base image when this
      is enabled, for example to include the cuda libraries.

#### example(self, name_or_index: Union[str, int]) -> Dict
      Return lookup an example by name or index.

      Index is 0 based. e.g. example(0) returns the first example.

#### examples(self) -> Dict[str, Dict]
      List truss model's examples.

      Examples are a simple `name to input` dictionary.

#### generate_readme(self)

#### get_docker_containers_from_labels(self, all=False)

#### get_docker_images_from_label(self)

#### get_urls_from_truss(self)

#### kill_container(self)

#### server_predict(self, request: dict)
      Run the prediction flow locally.

#### update_examples(self, examples: Dict[str, Dict])
      Update truss model's examples.

      Existing examples are replaced whole with the given ones.

#### update_requirements(self, requirements: List[str])
      Update requirements in truss model's config.

      Replaces requirements in truss model's config with the provided list.

#### update_requirements_from_file(self, requirements_filepath: str)
      Update requirements in truss model's config.

      Replaces requirements in truss model's config with those from the file
      at the given path.
