"""
Open questions - safe translation from processors.py to this generated file.

* How to deal with locally defined symbols from `processors.py`?
    - Force all processors in the same file to use the same image/deps and
      add any non `processor`-definition source and imports as common "header" here.
    - Specifically backtrack which symbols are used (recursively) and only add those.
* Locally defined symbols must not reference processors!
* Can processors reference processors?
    - What about class-processors that are not instantiated in `processors.py`?

"""
import collections
import contextlib
import inspect
import logging
import os
import shutil
import sys
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    Type,
    get_origin,
    get_type_hints,
)

import pydantic

from . import code_gen, types

CONFIG_ARG_NAME = "config"


# Checking of processor class definition ###############################################


def is_function_or_method(obj):
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        return True
    elif hasattr(obj, "__func__") and inspect.isfunction(obj.__func__):
        return True
    return False


def validate_method_names_and_get_endpoint_names(cls) -> set[str]:
    # TODO: differentiate between static, class and instance methods better.
    base_callables: set[str] = set()
    for base in cls.__bases__ + (object,):
        base_callables.update(
            name for name, _ in inspect.getmembers(base, is_function_or_method)
        )

    cls_callables = {name for name, _ in inspect.getmembers(cls, is_function_or_method)}

    added_public_callables = {
        name for name in cls_callables - base_callables if not name.startswith("_")
    }
    if not added_public_callables:
        raise types.APIDefinitonError(
            f"`{cls.__name__}` must implement at least one additional public method."
        )
    return added_public_callables


def validate_base_classes(
    cls: Type[types.ABCProcessor], endpoint_names: Iterable[str]
) -> None:
    # What if multiple public methods are added?
    ...


def validate_and_describe_endpoint(
    cls: Type[types.ABCProcessor], endpoint_name: str
) -> types.EndpointAPIDescriptor:
    endpoint_method = getattr(cls, endpoint_name)  # This is the unbound method.
    signature = inspect.signature(endpoint_method)
    params = list(signature.parameters.values())
    assert params[0].name == "self"
    # if len(params) > 2:
    #     raise APIDefinitonError()

    type_hints = get_type_hints(endpoint_method)

    # input_type = params[1].annotation
    # if not issubclass(input_type, pydantic.BaseModel):
    #     raise APIDefinitonError(
    #         f"Endpoint method `{endpoint_name}` must have a pydantic model as input data type."
    #     )

    if "return" not in type_hints or params[1].annotation not in type_hints.values():
        raise types.APIDefinitonError(
            f"Endpoint method `{endpoint_name}` must have a return type annotation."
        )
    output_type = type_hints["return"]

    # if not issubclass(output_type, pydantic.BaseModel):
    #     raise APIDefinitonError(
    #         f"Endpoint method `{endpoint_name}` must have a pydantic model as output data type."
    #     )
    # TODO: make sure output can be serialized.

    if inspect.isasyncgenfunction(endpoint_method):
        is_async = True
        is_generator = True
    elif inspect.iscoroutinefunction(endpoint_method):
        is_async = True
        is_generator = False
    else:
        is_async = False
        is_generator = inspect.isgeneratorfunction(endpoint_method)

    return types.EndpointAPIDescriptor(
        name=endpoint_name,
        input_types=[],
        output_type=output_type,
        is_async=is_async,
        is_generator=is_generator,
    )


def get_class_type(var):
    origin = get_origin(var)
    return origin if origin is not None else var


def validate_init_signature_and_get_dependencies(
    cls: Type[types.ABCProcessor],
) -> Mapping[str, Type[types.ABCProcessor]]:
    signature = inspect.signature(cls.__init__)
    params = list(signature.parameters.values())
    assert params[0].name == "self"
    if len(params) <= 1:
        raise types.APIDefinitonError()
    if params[1].name != CONFIG_ARG_NAME:
        raise types.APIDefinitonError(params)
    param_1_type = get_class_type(params[1].annotation)

    if not issubclass(param_1_type, types.Config):
        raise types.APIDefinitonError(params)

    depdendencies = {}
    for param in params[2:]:
        # TODO: deal with subclasses, unions, optionals, check default value etc.
        if not issubclass(param.annotation, types.ABCProcessor):
            raise types.APIDefinitonError(param)
        if param.annotation in depdendencies:
            raise types.APIDefinitonError(param)
        depdendencies[param.name] = param.annotation
    return depdendencies


def validate_variable_access(cls: Type[types.ABCProcessor]) -> None:
    # Access processors only via `provided` in `__init__`. No globals.
    ...


def check_and_register_class(cls) -> None:
    endpoint_names = validate_method_names_and_get_endpoint_names(cls)
    validate_base_classes(cls, endpoint_names)
    endpoint_descriptors = []
    for name in endpoint_names:
        endpoint_descriptors.append(validate_and_describe_endpoint(cls, name))
    dependencies = validate_init_signature_and_get_dependencies(cls)
    processor_descriptor = types.ProcessorAPIDescriptor(
        processor_cls=cls,
        depdendencies=dependencies,
        endpoints=endpoint_descriptors,
    )
    _global_processor_registry.register_processor(processor_descriptor)


########################################################################################


class _BaseProvisionPlaceholder:
    ...


class ProcessorProvisionPlaceholder(_BaseProvisionPlaceholder):
    def __init__(self, processor_cls: Type[types.ABCProcessor]) -> None:
        self._processor_cls = processor_cls


class ConfigProvisionPlaceholder(_BaseProvisionPlaceholder):
    ...


class ProcessorRegistry:
    # Because dependencies are reuired to be present when registering a processor,
    # this dict contains natively a topological sorting of the dependency graph.
    _processors: collections.OrderedDict[
        Type[types.ABCProcessor], types.ProcessorAPIDescriptor
    ]

    def __init__(self) -> None:
        self._processors = collections.OrderedDict()

    def register_processor(self, processor_descriptor: types.ProcessorAPIDescriptor):
        for dep in processor_descriptor.depdendencies.values():
            if dep not in self._processors:
                raise types.MissingDependencyError(dep)

        self._processors[processor_descriptor.processor_cls] = processor_descriptor

    @property
    def processor_descriptors(self) -> list[types.ProcessorAPIDescriptor]:
        return list(self._processors.values())

    @property
    def processor_classes(self) -> list[Type[types.ABCProcessor]]:
        return list(self._processors.keys())

    def get_dependencies(
        self, processor: Type[types.ABCProcessor]
    ) -> Iterable[Type[types.ABCProcessor]]:
        return [desc for desc in self._processors[processor].depdendencies.values()]


_global_processor_registry = ProcessorRegistry()


# Processor class runtime utils ########################################################


def determine_arguments(func, **kwargs):
    sig = inspect.signature(func)
    bound_args = sig.bind_partial(**kwargs)
    bound_args.apply_defaults()
    return bound_args.arguments


def check_init_args(cls, kwargs) -> None:
    final_args = determine_arguments(cls.__init__, **kwargs)
    for name, value in final_args.items():
        if isinstance(value, _BaseProvisionPlaceholder):
            raise types.UsageError(
                f"When initializing class `{cls.__name__}`, for "
                f"default argument `{name}` a symbolic placeholder value "
                "was passed. It is required to instantiate and run the classes "
                f"in a `{run_local.__name__}` context or deploy them remotely."
            )


# Local Deployment #####################################################################


def _create_modified_init(
    processor_descriptor: types.ProcessorAPIDescriptor,
    cls_to_instance: MutableMapping[Type[types.ABCProcessor], types.ABCProcessor],
):
    original_init = processor_descriptor.processor_cls.__init__

    def modified_init(self: types.ABCProcessor, **kwargs) -> None:
        print(f"Patched `__init__` of `{processor_descriptor.processor_cls}`.")
        if hasattr(processor_descriptor.processor_cls, "default_config"):
            config = processor_descriptor.processor_cls.default_config
        else:
            config = types.Config()

        for arg_name, dep_cls in processor_descriptor.depdendencies.items():
            if arg_name in kwargs:
                logging.debug(
                    f"Use explicitly given instance for `{arg_name}` of "
                    f"type `{dep_cls.__name__}`."
                )
                continue

            if dep_cls in cls_to_instance:
                logging.debug(
                    f"Use mapped instace for `{arg_name}` of type `{dep_cls.__name__}`."
                )
                instance = cls_to_instance[dep_cls]
            else:
                logging.debug(
                    f"Create new instace for `{arg_name}` of type `{dep_cls.__name__}`."
                )
                assert dep_cls._init_is_patched
                instance = dep_cls()
                cls_to_instance[dep_cls] = instance

            kwargs[arg_name] = instance

        original_init(self, config=config, **kwargs)

    return modified_init


@contextlib.contextmanager
def run_local() -> Any:
    type_to_instance: MutableMapping[Type[types.ABCProcessor], types.ABCProcessor] = {}
    original_inits: MutableMapping[Type[types.ABCProcessor], Callable] = {}

    for processor_descriptor in _global_processor_registry.processor_descriptors:
        original_inits[
            processor_descriptor.processor_cls
        ] = processor_descriptor.processor_cls.__init__
        patched_init = _create_modified_init(processor_descriptor, type_to_instance)
        processor_descriptor.processor_cls.__init__ = patched_init  # type: ignore[method-assign]
        processor_descriptor.processor_cls._init_is_patched = True
    try:
        yield
    finally:
        # Restore original classes to unpatched state.
        for processor_cls, original_init in original_inits.items():
            processor_cls.__init__ = original_init  # type: ignore[method-assign]
            processor_cls._init_is_patched = False


# Remote Deployment ####################################################################
"""
Generate a remote processor
* Create `./generated/processor_{name}/` directory.
* Copy all from CWD.
* Convert the processor class into a truss model:
    * Inheri



"""


class RemoteServiceDescriptor(pydantic.BaseModel):
    url: str


class StubDescriptor(pydantic.BaseModel):
    url: str


def create_remote_service(
    processor: Type[types.ABCProcessor], stubs: list[StubDescriptor]
):
    # TODO: better way of dealing with (abs) paths.
    # TODO: tracking relative imports / deps of workflow file.
    workflow_filepath = os.path.abspath(sys.argv[0])
    workflow_dir = os.path.dirname(workflow_filepath)
    processor_dir = os.path.join(
        workflow_dir, "generated", f"processor_{processor.__name__}"
    )
    os.makedirs(processor_dir, exist_ok=True)
    model_filepath = shutil.copy(
        workflow_filepath, os.path.join(processor_dir, "model.py")
    )

    code_gen.edit_model_file(model_filepath)


def create_stub(remote: RemoteServiceDescriptor) -> StubDescriptor:
    ...


def deploy_remotely(processors: Iterable[Type[types.ABCProcessor]]) -> None:
    needed_processors = set()

    def add_needed_procssors(proc: Type[types.ABCProcessor]):
        for dep in _global_processor_registry.get_dependencies(proc):
            needed_processors.add(dep)
            add_needed_procssors(dep)

    for proc in processors:
        add_needed_procssors(proc)

    ordered_processors = [
        cls
        for cls in _global_processor_registry.processor_classes
        if cls in needed_processors
    ]
    stubs = []
    for processor in ordered_processors:
        remote_service = create_remote_service(processor, stubs)
        stub = create_stub(remote_service)
        stubs.append(stub)
