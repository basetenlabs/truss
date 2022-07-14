# Truss Python client reference

A list of the functions available with `import truss` and their arguments and properties

Help on module truss.build in truss:

NAME
    truss.build

FUNCTIONS
    cleanup()
        Cleans up .truss directory.
    
    from_directory(truss_directory: str) -> truss.truss_handle.TrussHandle
        Get a handle to a Truss. A Truss is a build context designed to be built
        as a container locally or uploaded into a baseten serving environment.
        
        Args:
            truss_directory (str): The local directory of an existing Truss
        Returns:
            TrussHandle
    
    init(target_directory: str, data_files: List[str] = None, requirements_file: str = None) -> truss.truss_handle.TrussHandle
        Initialize an empty placeholder Truss. A Truss is a build context designed
        to be built as a container locally or uploaded into a baseten serving
        environment. This placeholder structure can be filled to represent ML
        models.
        
        Args:
            target_directory: Absolute or relative path of the directory to create
                              Truss in. The directory is created if it doesn't exist.
    
    kill_all()
    
    mk_truss(model: Any, target_directory: str = None, data_files: List[str] = None, requirements_file: str = None) -> truss.truss_handle.TrussHandle
        Create a Truss with the given model. A Truss is a build context designed to
        be built as a container locally or uploaded into a baseten serving environment.
        
        Args:
            model (an in-memory model object): A model object to be deployed (e.g. a keras, sklearn, or pytorch model
                object)
            target_directory (str, optional): The local directory target for the Truss. Otherwise a temporary directory
                will be generated
            data_files (List[str], optional): Additional files required for model operation. Can be a glob that resolves to
                files for the root directory or a directory path.
            requirements_file (str, optional): A file of packages in a PIP requirements format to be installed in the
                container environment.
        Returns:
            TrussHandle: A handle to the generated Truss that provides easy access to content inside.

DATA
    Any = typing.Any
        Special type indicating an unconstrained type.
        
        - Any is compatible with every type.
        - Any assumed to have all methods.
        - All values assumed to be instances of Any.
        
        Note that all the above statements are true from the point of view of
        static type checkers. At runtime, Any should not be used with instance
        or class checks.
    
    CONFIG_FILE = 'config.yaml'
    DEFAULT_EXAMPLES_FILENAME = 'examples.yaml'
    List = typing.List
        A generic version of list.
    
    TEMPLATES_DIR = PosixPath('/workspaces/truss/truss/templates')
    TRUSS = 'truss'

FILE
    /workspaces/truss/truss/build.py


Help on module truss.truss_handle in truss:

NAME
    truss.truss_handle

CLASSES
    builtins.object
        TrussHandle
    
    class TrussHandle(builtins.object)
     |  TrussHandle(truss_dir: pathlib.Path) -> None
     |  
     |  Methods defined here:
     |  
     |  __init__(self, truss_dir: pathlib.Path) -> None
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  add_data(self, file_dir_or_glob: str)
     |      Add data to a truss model.
     |      
     |      Accepts a file path, a directory path or a glob. Everything is copied
     |      under the truss model's data directory.
     |  
     |  add_environment_variable(self, env_var_name: str, env_var_value: str)
     |      Add an environment variable to truss model's config.
     |  
     |  add_example(self, example_name: str, example: dict)
     |      Add example for truss model.
     |      
     |      If the example with the given name already exists then it is overwritten.
     |  
     |  add_python_requirement(self, python_requirement: str)
     |      Add a python requirement to truss model's config.
     |  
     |  add_secret(self, secret_name: str, default_secret_value: str = '')
     |  
     |  add_system_package(self, system_package: str)
     |      Add a system package requirement to truss model's config.
     |  
     |  build_docker_image(self, build_dir: pathlib.Path = None, tag: str = None)
     |      Builds docker image
     |  
     |  container_logs(self)
     |  
     |  docker_build_setup(self, build_dir: pathlib.Path = None)
     |      Set up a directory to build docker image from.
     |      
     |      Returns:
     |          docker build command.
     |  
     |  docker_predict(self, request: dict, build_dir: pathlib.Path = None, tag: str = None, local_port: int = 8080, detach: bool = True)
     |      Builds docker image, runs that as a docker container
     |      and makes a prediction request to the server running on the container.
     |      Kills the container afterwards. Mostly useful for testing.
     |  
     |  docker_run(self, build_dir: pathlib.Path = None, tag: str = None, local_port: int = 8080, detach=True)
     |      Builds a docker image and runs it as a container.
     |      
     |      Returns:
     |          Container, which can be used to get information about the running,
     |          including its id. The id can be used to kill the container.
     |  
     |  enable_gpu(self)
     |      Enable gpu use for given model.
     |      
     |      This is suggestive, model serving environment may still use cpu, e.g. if
     |      the setup doesn't have access to a GPU.
     |      
     |      Note that truss would typically use a larger docker base image when this
     |      is enabled, for example to include the cuda libraries.
     |  
     |  example(self, name_or_index: Union[str, int]) -> Dict
     |      Return lookup an example by name or index.
     |      
     |      Index is 0 based. e.g. example(0) returns the first example.
     |  
     |  examples(self) -> Dict[str, Dict]
     |      List truss model's examples.
     |      
     |      Examples are a simple `name to input` dictionary.
     |  
     |  generate_readme(self)
     |  
     |  get_docker_containers_from_labels(self, all=False)
     |  
     |  get_docker_images_from_label(self)
     |  
     |  get_urls_from_truss(self)
     |  
     |  kill_container(self)
     |  
     |  server_predict(self, request: dict)
     |      Run the prediction flow locally.
     |  
     |  update_examples(self, examples: Dict[str, Dict])
     |      Update truss model's examples.
     |      
     |      Existing examples are replaced whole with the given ones.
     |  
     |  update_requirements(self, requirements: List[str])
     |      Update requirements in truss model's config.
     |      
     |      Replaces requirements in truss model's config with the provided list.
     |  
     |  update_requirements_from_file(self, requirements_filepath: str)
     |      Update requirements in truss model's config.
     |      
     |      Replaces requirements in truss model's config with those from the file
     |      at the given path.
     |  
     |  ----------------------------------------------------------------------
     |  Readonly properties defined here:
     |  
     |  spec
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)

DATA
    Callable = typing.Callable
        Callable type; Callable[[int], str] is a function of (int) -> str.
        
        The subscription syntax must always be used with exactly two
        values: the argument list and the return type.  The argument list
        must be a list of types or ellipsis; the return type must be a single type.
        
        There is no syntax to indicate optional or keyword arguments,
        such function types are rarely used as callback types.
    
    Dict = typing.Dict
        A generic version of dict.
    
    List = typing.List
        A generic version of list.
    
    TRUSS = 'truss'
    TRUSS_DIR = 'truss_dir'
    TRUSS_MODIFIED_TIME = 'truss_modified_time'
    Union = typing.Union
        Union type; Union[X, Y] means either X or Y.
        
        To define a union, use e.g. Union[int, str].  Details:
        - The arguments must be types and there must be at least one.
        - None as an argument is a special case and is replaced by
          type(None).
        - Unions of unions are flattened, e.g.::
        
            Union[Union[int, str], float] == Union[int, str, float]
        
        - Unions of a single argument vanish, e.g.::
        
            Union[int] == int  # The constructor actually returns int
        
        - Redundant arguments are skipped, e.g.::
        
            Union[int, str, int] == Union[int, str]
        
        - When comparing unions, the argument order is ignored, e.g.::
        
            Union[int, str] == Union[str, int]
        
        - You cannot subclass or instantiate a union.
        - You can use Optional[X] as a shorthand for Union[X, None].
    
    logger = <Logger truss.truss_handle (WARNING)>

FILE
    /workspaces/truss/truss/truss_handle.py


