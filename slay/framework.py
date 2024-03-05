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
import functools
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
    Protocol,
    Type,
    get_args,
    get_origin,
    get_type_hints,
)

import httpx
import pydantic
from slay import code_gen, definitions

CONTEXT_ARG_NAME = "context"


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
        raise definitions.APIDefinitonError(
            f"`{cls.__name__}` must implement at least one additional public method."
        )
    return added_public_callables


def validate_base_classes(
    cls: Type[definitions.ABCProcessor], endpoint_names: Iterable[str]
) -> None:
    # What if multiple public methods are added?
    ...


def validate_and_describe_endpoint(
    cls: Type[definitions.ABCProcessor], endpoint_name: str
) -> definitions.EndpointAPIDescriptor:
    endpoint_method = getattr(cls, endpoint_name)  # This is the unbound method.
    signature = inspect.signature(endpoint_method)
    params = list(signature.parameters.values())
    assert params[0].name == "self"

    input_name_and_types = []
    for param in params[1:]:  # Skip self argument.
        assert param.annotation is not None
        type_descriptor = definitions.TypeDescriptor(raw=param.annotation)
        input_name_and_types.append((param.name, type_descriptor))

    if get_origin(signature.return_annotation) is tuple:
        output_types = list(
            definitions.TypeDescriptor(raw=arg)
            for arg in get_args(signature.return_annotation)
        )
    else:
        output_types = [definitions.TypeDescriptor(raw=signature.return_annotation)]

    if inspect.isasyncgenfunction(endpoint_method):
        is_async = True
        is_generator = True
    elif inspect.iscoroutinefunction(endpoint_method):
        is_async = True
        is_generator = False
    else:
        is_async = False
        is_generator = inspect.isgeneratorfunction(endpoint_method)

    return definitions.EndpointAPIDescriptor(
        name=endpoint_name,
        input_name_and_tyes=input_name_and_types,
        output_types=output_types,
        is_async=is_async,
        is_generator=is_generator,
    )


def get_class_type(var):
    origin = get_origin(var)
    return origin if origin is not None else var


def validate_init_signature_and_get_dependencies(
    cls: Type[definitions.ABCProcessor],
) -> Mapping[str, Type[definitions.ABCProcessor]]:
    signature = inspect.signature(cls.__init__)
    params = list(signature.parameters.values())
    assert params[0].name == "self"
    if len(params) <= 1:
        raise definitions.APIDefinitonError()
    if params[1].name != CONTEXT_ARG_NAME:
        raise definitions.APIDefinitonError(params)
    param_1_type = get_class_type(params[1].annotation)

    if not issubclass(param_1_type, definitions.Context):
        raise definitions.APIDefinitonError(params)

    depdendencies = {}
    for param in params[2:]:
        # TODO: deal with subclasses, unions, optionals, check default value etc.
        default = param.default
        if not isinstance(default, ProcessorProvisionPlaceholder):
            raise definitions.APIDefinitonError(param)

        processor_cls = default._processor_cls
        if not issubclass(param.annotation, Protocol) and not issubclass(
            processor_cls, param.annotation
        ):
            definitions.APIDefinitonError(processor_cls)
        if not issubclass(processor_cls, definitions.ABCProcessor):
            raise definitions.APIDefinitonError(param)
        if processor_cls in depdendencies:
            raise definitions.APIDefinitonError(param)
        depdendencies[param.name] = processor_cls
    return depdendencies


def validate_variable_access(cls: Type[definitions.ABCProcessor]) -> None:
    # Access processors only via `provided` in `__init__`. No globals.
    ...


def check_and_register_class(cls) -> None:
    endpoint_names = validate_method_names_and_get_endpoint_names(cls)
    validate_base_classes(cls, endpoint_names)
    endpoint_descriptors = []
    for name in endpoint_names:
        endpoint_descriptors.append(validate_and_describe_endpoint(cls, name))
    dependencies = validate_init_signature_and_get_dependencies(cls)
    processor_descriptor = definitions.ProcessorAPIDescriptor(
        processor_cls=cls,
        depdendencies=dependencies,
        endpoints=endpoint_descriptors,
        src_file=os.path.abspath(inspect.getfile(cls)),
    )
    logging.debug(f"{processor_descriptor}\n")
    _global_processor_registry.register_processor(processor_descriptor)


########################################################################################


class _BaseProvisionPlaceholder:
    ...


class ProcessorProvisionPlaceholder(_BaseProvisionPlaceholder):
    def __init__(self, processor_cls: Type[definitions.ABCProcessor]) -> None:
        self._processor_cls = processor_cls

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._processor_cls.__name__})"


class ContextProvisionPlaceholder(_BaseProvisionPlaceholder):
    def __str__(self) -> str:
        return f"{self.__class__.__name__}"


class ProcessorRegistry:
    # Because dependencies are reuired to be present when registering a processor,
    # this dict contains natively a topological sorting of the dependency graph.
    _processors: collections.OrderedDict[
        Type[definitions.ABCProcessor], definitions.ProcessorAPIDescriptor
    ]

    def __init__(self) -> None:
        self._processors = collections.OrderedDict()

    def register_processor(
        self, processor_descriptor: definitions.ProcessorAPIDescriptor
    ):
        for dep in processor_descriptor.depdendencies.values():
            if dep not in self._processors:
                raise definitions.MissingDependencyError(dep)

        self._processors[processor_descriptor.processor_cls] = processor_descriptor

    @property
    def processor_descriptors(self) -> list[definitions.ProcessorAPIDescriptor]:
        return list(self._processors.values())

    # @property
    # def processor_descriptors(self) -> list[definitions.ProcessorAPIDescriptor]:
    #     return list(self._processors.values())

    def get_descriptor(
        self, processor_cls: Type[definitions.ABCProcessor]
    ) -> definitions.ProcessorAPIDescriptor:
        return self._processors[processor_cls]

    def get_dependencies(
        self, processor: definitions.ProcessorAPIDescriptor
    ) -> Iterable[definitions.ProcessorAPIDescriptor]:
        return [
            self._processors[desc]
            for desc in self._processors[processor.processor_cls].depdendencies.values()
        ]


_global_processor_registry = ProcessorRegistry()


# Processor class runtime utils ########################################################


def determine_arguments(func, **kwargs):
    sig = inspect.signature(func)
    bound_args = sig.bind_partial(**kwargs)
    bound_args.apply_defaults()
    return bound_args.arguments


def check_init_args(cls, original_init, kwargs) -> None:
    final_args = determine_arguments(original_init, **kwargs)
    for name, value in final_args.items():
        if isinstance(value, _BaseProvisionPlaceholder):
            raise definitions.UsageError(
                f"When initializing class `{cls.__name__}`, for "
                f"default argument `{name}` a symbolic placeholder value "
                f"was passed (`{value}`). Processors must be either a) localy "
                f" instantiated in `{run_local.__name__}` context or b) deployed "
                "remotely."
            )


# Local Deployment #####################################################################


def _create_modified_init_for_local(
    processor_descriptor: definitions.ProcessorAPIDescriptor,
    cls_to_instance: MutableMapping[
        Type[definitions.ABCProcessor], definitions.ABCProcessor
    ],
):
    original_init = processor_descriptor.processor_cls.__init__

    def modified_init(self: definitions.ABCProcessor, **kwargs) -> None:
        logging.debug(f"Patched `__init__` of `{processor_descriptor.processor_cls}`.")
        if hasattr(processor_descriptor.processor_cls, "default_config"):
            defaults = processor_descriptor.processor_cls.default_config
            context = definitions.Context(name=defaults.name or self.__class__.__name__)
        else:
            context = definitions.Context(name=self.__class__.__name__)

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

        original_init(self, context=context, **kwargs)

    return modified_init


@contextlib.contextmanager
def run_local() -> Any:
    type_to_instance: MutableMapping[
        Type[definitions.ABCProcessor], definitions.ABCProcessor
    ] = {}
    original_inits: MutableMapping[Type[definitions.ABCProcessor], Callable] = {}

    for processor_descriptor in _global_processor_registry.processor_descriptors:
        original_inits[
            processor_descriptor.processor_cls
        ] = processor_descriptor.processor_cls.__init__
        patched_init = _create_modified_init_for_local(
            processor_descriptor, type_to_instance
        )
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


class RemoteServiceDescriptor(pydantic.BaseModel):
    url: str


class StubDescriptor(pydantic.BaseModel):
    url: str


def _create_processor_dir(workflow_root, processor_descriptor):
    processor_name = processor_descriptor.processor_cls.__name__
    processor_dir = os.path.join(
        workflow_root, ".slay_gen", f"processor_{processor_name}"
    )
    os.makedirs(processor_dir, exist_ok=True)
    return processor_dir


def create_remote_service(
    workflow_root,
    processor_descriptor: definitions.ProcessorAPIDescriptor,
):
    processor_dir = _create_processor_dir(workflow_root, processor_descriptor)
    # TODO: copy other local deps.
    processor_filepath = shutil.copy(
        processor_descriptor.src_file,
        os.path.join(processor_dir, "user_dependencies.py"),
    )
    code_gen.modify_source_file(processor_filepath, processor_descriptor)


def deploy_remotely(processors: Iterable[Type[definitions.ABCProcessor]]) -> None:
    workflow_filepath = os.path.abspath(sys.argv[0])
    workflow_root = os.path.dirname(workflow_filepath)

    needed_processors: set[definitions.ProcessorAPIDescriptor] = set()

    def add_needed_procssors(proc: definitions.ProcessorAPIDescriptor):
        needed_processors.add(proc)
        for dep in _global_processor_registry.get_dependencies(proc):
            needed_processors.add(dep)
            add_needed_procssors(dep)

    for proc_cls in processors:
        proc = _global_processor_registry.get_descriptor(proc_cls)
        add_needed_procssors(proc)

    ordered_processors = [
        cls
        for cls in _global_processor_registry.processor_descriptors
        if cls in needed_processors
    ]

    for processor_descriptor in ordered_processors:
        processor_dir = _create_processor_dir(workflow_root, processor_descriptor)
        code_gen.generate_stubs_for_deps(
            os.path.join(processor_dir, "dependencies.py"),
            _global_processor_registry.get_dependencies(processor_descriptor),
        )
        create_remote_service(workflow_root, processor_descriptor)
