import asyncio
import contextlib
import logging
import pathlib
import re
from typing import AsyncIterator, Iterator, List

import pydantic
import pytest

import truss_chains as chains
from truss_chains import definitions, framework, public_api, utils

utils.setup_dev_logging(logging.DEBUG)

TEST_ROOT = pathlib.Path(__file__).parent.resolve()

# Assert that naive chainlet initialization is detected and prevented. #################


class Chainlet1(chains.ChainletBase):
    def run_remote(self) -> str:
        return self.__class__.name


class Chainlet2(chains.ChainletBase):
    def run_remote(self) -> str:
        return self.__class__.name


class InitInInit(chains.ChainletBase):
    def __init__(self, chainlet2=chains.depends(Chainlet2)):
        self.chainlet1 = Chainlet1()
        self.chainlet2 = chainlet2

    def run_remote(self) -> str:
        return self.chainlet1.run_remote()


class InitInRun(chains.ChainletBase):
    def run_remote(self) -> str:
        Chainlet1()
        return "abc"


def foo():
    return Chainlet1()


class InitWithFn(chains.ChainletBase):
    def __init__(self):
        foo()

    def run_remote(self) -> str:
        return self.__class__.name


def test_raises_init_in_init():
    match = "Chainlets cannot be naively instantiated"
    with pytest.raises(definitions.ChainsRuntimeError, match=match):
        with chains.run_local():
            InitInInit()


def test_raises_init_in_run():
    match = "Chainlets cannot be naively instantiated"
    with pytest.raises(definitions.ChainsRuntimeError, match=match):
        with chains.run_local():
            chain = InitInRun()
            chain.run_remote()


def test_raises_init_in_function():
    match = "Chainlets cannot be naively instantiated"
    with pytest.raises(definitions.ChainsRuntimeError, match=match):
        with chains.run_local():
            InitWithFn()


def test_raises_depends_usage():
    class InlinedDepends(chains.ChainletBase):
        def __init__(self):
            self.chainlet1 = chains.depends(Chainlet1)

        def run_remote(self) -> str:
            return self.chainlet1.run_remote()

    match = (
        "`chains.depends(Chainlet1)` was used, but not as an argument to the `__init__`"
    )
    with pytest.raises(definitions.ChainsRuntimeError, match=re.escape(match)):
        with chains.run_local():
            chain = InlinedDepends()
            chain.run_remote()


def test_raises_model_requires_predict_method():
    class ModelWithRunRemote(chains.ModelBase):
        def run_remote(self) -> str:
            return self.__class__.name

    match = "Models must have a `predict` method."
    with pytest.raises(definitions.ChainsUsageError, match=re.escape(match)):
        with chains.run_local():
            ModelWithRunRemote()


def test_raises_model_dependencies_not_allowed():
    class ModelWithDependencies(chains.ModelBase):
        def __init__(self, c1=chains.depends(Chainlet1)):
            self.c1 = c1

        def run_remote(self) -> str:
            return self.__class__.name

    match = "The only supported argument to `__init__` for Models"
    with pytest.raises(definitions.ChainsUsageError, match=re.escape(match)):
        with chains.run_local():
            ModelWithDependencies()


# The problem with supporting helper functions in `run_local` is that the stack trace
# looks similar to the forbidden one in `InitInRun`.
@pytest.mark.skip(reason="Helper functions not supported yet.")
def test_ok_with_subclass_and_helper_fn():
    def build():
        return Chainlet1()

    with chains.run_local():
        chain = build()
        print(chain.run_remote())


# Test sub-classing (incl. detection of naive chainlet instantiation). #################


class BaseChainlet(chains.ChainletBase):
    def __init__(self):
        self.base_value = "base_value"
        logging.info("########## Init Base")

    async def run_remote(self) -> str:
        return self.__class__.name


class IntermediateChainlet(BaseChainlet):
    def __init__(self):
        logging.info("########## Start init Intermediate")
        super().__init__()
        self.added_value = "added_value"
        logging.info("########## Finish init Intermediate")

    async def run_remote(self) -> str:
        return self.__class__.name


class DerivedChainlet(IntermediateChainlet):
    def __init__(self):
        logging.info("########## Start init Derived")
        super().__init__()
        self.base_value = "overridden_base_value"
        logging.info("########## Finish init Derived")

    async def run_remote(self) -> str:
        return self.__class__.name


class InitInInitSub(chains.ChainletBase):
    def __init__(self, a=chains.depends(BaseChainlet)):
        self.b = DerivedChainlet()
        self.a = a

    async def run_remote(self) -> str:
        return await self.b.run_remote()


class CorrectChain(chains.ChainletBase):
    def __init__(
        self, a=chains.depends(BaseChainlet), b=chains.depends(DerivedChainlet)
    ):
        self.a = a
        self.b = b

    async def run_remote(self) -> str:
        return await self.a.run_remote() + " " + await self.b.run_remote()


# Make sure there are no other validations errors from above definitions..
framework.raise_validation_errors()


def test_raises_init_in_init_subclass():
    match = "Chainlets cannot be naively instantiated"
    with pytest.raises(definitions.ChainsRuntimeError, match=match):
        with chains.run_local():
            InitInInitSub()


def test_ok_with_subclass():
    with chains.run_local():
        chain = CorrectChain()
        assert chain.a.base_value == "base_value"
        assert chain.b.base_value == "overridden_base_value"
        assert chain.b.added_value == "added_value"
        result = asyncio.run(chain.run_remote())
        assert result == "BaseChainlet DerivedChainlet"


# Assert that Chain(let) definitions are validated #####################################


@contextlib.contextmanager
def _raise_errors():
    framework._global_chainlet_registry.clear()
    framework.raise_validation_errors()
    yield
    framework._global_chainlet_registry.clear()
    framework.raise_validation_errors()


TEST_FILE = __file__


def test_raises_without_depends():
    match = (
        rf"{TEST_FILE}:\d+ \(WithoutDepends\.__init__\) \[kind: TYPE_ERROR\].*must "
        r"have dependency Chainlets with default values from `chains.depends`"
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class WithoutDepends(chains.ChainletBase):
            def __init__(self, chainlet1):
                self.chainlet1 = chainlet1

            def run_remote(self) -> str:
                return self.chainlet1.run_remote()


class SomeModel(pydantic.BaseModel):
    foo: int


def test_raises_unsupported_return_type_list_object():
    match = (
        rf"{TEST_FILE}:\d+ \(UnsupportedArgType\.run_remote\) \[kind: IO_TYPE_ERROR\].*"
        r"Unsupported I/O type for `return_type`"
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class UnsupportedArgType(chains.ChainletBase):
            def run_remote(self) -> list[object]:
                return [SomeModel(foo=0)]


def test_raises_unsupported_return_type_list_object_legacy():
    match = (
        rf"{TEST_FILE}:\d+ \(UnsupportedArgType\.run_remote\) \[kind: IO_TYPE_ERROR\].*"
        r"Unsupported I/O type for `return_type`"
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class UnsupportedArgType(chains.ChainletBase):
            def run_remote(self) -> List[object]:
                return [SomeModel(foo=0)]


def test_raises_unsupported_arg_type_list_object():
    match = (
        rf"{TEST_FILE}:\d+ \(UnsupportedArgType\.run_remote\) \[kind: IO_TYPE_ERROR\].*"
        r"Unsupported I/O type for `arg`"
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class UnsupportedArgType(chains.ChainletBase):
            def run_remote(self, arg: list[object]) -> None:
                return


def test_raises_unsupported_arg_type_object():
    match = (
        rf"{TEST_FILE}:\d+ \(UnsupportedArgType\.run_remote\) \[kind: IO_TYPE_ERROR\].*"
        r"Unsupported I/O type for `arg` of type `<class 'object'>`"
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class UnsupportedArgType(chains.ChainletBase):
            def run_remote(self, arg: object) -> None:
                return


def test_raises_unsupported_arg_type_str_annot():
    match = (
        rf"{TEST_FILE}:\d+ \(UnsupportedArgType\.run_remote\) \[kind: IO_TYPE_ERROR\].*"
        r"A string-valued type annotation was found for `arg` of type `SomeModel`"
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class UnsupportedArgType(chains.ChainletBase):
            def run_remote(self, arg: "SomeModel") -> None:
                return


def test_raises_endpoint_no_method():
    match = (
        rf"{TEST_FILE}:\d+ \(StaticMethod\.run_remote\) \[kind: TYPE_ERROR\].*"
        r"`run_remote` must be a method"
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class StaticMethod(chains.ChainletBase):
            @staticmethod
            def run_remote() -> None:
                return


def test_raises_endpoint_no_method_arg():
    match = (
        rf"{TEST_FILE}:\d+ \(StaticMethod\.run_remote\) \[kind: TYPE_ERROR\].*"
        r"`run_remote` must be a method"
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class StaticMethod(chains.ChainletBase):
            @staticmethod
            def run_remote(arg: "SomeModel") -> None:
                return


def test_raises_endpoint_not_annotated():
    match = (
        rf"{TEST_FILE}:\d+ \(NoArgAnnot\.run_remote\) \[kind: IO_TYPE_ERROR\].*"
        r"Arguments of endpoints must have type annotations."
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class NoArgAnnot(chains.ChainletBase):
            def run_remote(self, arg) -> None:
                return


def test_raises_endpoint_return_not_annotated():
    match = (
        rf"{TEST_FILE}:\d+ \(NoReturnAnnot\.run_remote\) \[kind: IO_TYPE_ERROR\].*"
        r"Return values of endpoints must be type annotated."
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class NoReturnAnnot(chains.ChainletBase):
            def run_remote(self):
                return


def test_raises_endpoint_return_not_supported():
    match = (
        rf"{TEST_FILE}:\d+ \(ReturnNotSupported\.run_remote\) \[kind: IO_TYPE_ERROR\].*"
        r"Unsupported I/O type for `return_type` of type `<class 'object'>`"
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class ReturnNotSupported(chains.ChainletBase):
            def run_remote(self) -> object:
                return object()


def test_raises_no_endpoint():
    match = (
        rf"{TEST_FILE}:\d+ \(NoEndpoint\) \[kind: MISSING_API_ERROR\].*"
        r"Chainlets must have a `run_remote` method."
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class NoEndpoint(chains.ChainletBase):
            def rum_remote(self) -> object:
                return object()


def test_raises_context_not_trailing():
    match = (
        rf"{TEST_FILE}:\d+ \(ContextNotTrailing\.__init__\) \[kind: TYPE_ERROR\].*"
        r"The init argument name `context` is reserved for the optional context "
        f"argument, which must be trailing"
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class Chainlet1(chains.ChainletBase):
            def run_remote(self) -> str:
                return self.__class__.name

        class ContextNotTrailing(chains.ChainletBase):
            def __init__(self, context, chainlet1=chains.depends(Chainlet1)): ...


def test_raises_not_dep_marker():
    match = (
        rf"{TEST_FILE}:\d+ \(NoDepMarker\.__init__\) \[kind: TYPE_ERROR\].*"
        r"Any arguments of a Chainlet\'s __init__ \(besides `context`\) must have "
        f"dependency Chainlets with default values from `chains.depends`-directive"
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class Chainlet1(chains.ChainletBase):
            def run_remote(self) -> str:
                return self.__class__.name

        class NoDepMarker(chains.ChainletBase):
            def __init__(self, chainlet1=Chainlet1): ...


def test_raises_dep_not_chainlet():
    match = (
        rf"{TEST_FILE}:\d+ \(DepNotChainlet\.__init__\) \[kind: TYPE_ERROR\].*"
        r"`chains.depends` must be used with a Chainlet class as argument, got <class "
        f"'truss_chains.definitions.RPCOptions'>"
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class Chainlet1(chains.ChainletBase):
            def run_remote(self) -> str:
                return self.__class__.name

        class DepNotChainlet(chains.ChainletBase):
            def __init__(self, chainlet1=chains.depends(definitions.RPCOptions)): ...


def test_raises_dep_not_chainlet_annot():
    match = (
        rf"{TEST_FILE}:\d+ \(DepNotChainletAnnot\.__init__\) \[kind: TYPE_ERROR\].*"
        r"The type annotation for `chainlet1` must be a class/subclass of the "
        "Chainlet type"
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class Chainlet1(chains.ChainletBase):
            def run_remote(self) -> str:
                return self.__class__.name

        class DepNotChainletAnnot(chains.ChainletBase):
            def __init__(
                self,
                chainlet1: definitions.RPCOptions = chains.depends(Chainlet1),  # type: ignore
            ): ...


def test_raises_context_missing_default():
    match = (
        rf"{TEST_FILE}:\d+ \(ContextMissingDefault\.__init__\) \[kind: TYPE_ERROR\].*"
        r"If `Chainlet` uses context for initialization, it must have "
        r"`context` argument of type `<class 'truss_chains.definitions.DeploymentContext'>`"
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class ContextMissingDefault(chains.ChainletBase):
            def __init__(self, context=None): ...


def test_raises_context_wrong_annot():
    match = (
        rf"{TEST_FILE}:\d+ \(ConextWrongAnnot\.__init__\) \[kind: TYPE_ERROR\].*"
        r"If `Chainlet` uses context for initialization, it must have "
        r"`context` argument of type `<class 'truss_chains.definitions.DeploymentContext'>`"
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class ConextWrongAnnot(chains.ChainletBase):
            def __init__(self, context: object = chains.depends_context()): ...


def test_raises_chainlet_reuse():
    match = (
        rf"{TEST_FILE}:\d+ \(ChainletReuse\.__init__\) \[kind: TYPE_ERROR\].*"
        r"The same Chainlet class cannot be used multiple times for different arguments"
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class Chainlet1(chains.ChainletBase):
            def run_remote(self) -> str:
                return self.__class__.name

        class ChainletReuse(chains.ChainletBase):
            def __init__(
                self, dep1=chains.depends(Chainlet1), dep2=chains.depends(Chainlet1)
            ): ...

            def run_remote(self) -> None:
                return


def test_collects_multiple_errors():
    match = r"The user defined code does not comply with the required spec"

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class MultiIssue(chains.ChainletBase):
            def __init__(self, context, chainlet1):
                self.chainlet1 = chainlet1

            def run_remote(argument: object): ...

        assert len(framework._global_error_collector._errors) == 5


def test_collects_multiple_errors_run_local():
    class MultiIssue(chains.ChainletBase):
        def __init__(self, context, chainlet1):
            self.chainlet1 = chainlet1

        def run_remote(argument: object): ...

    match = r"The user defined code does not comply with the required spec"
    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():
        with public_api.run_local():
            MultiIssue()


def test_raises_iterator_no_yield():
    match = (
        rf"{TEST_FILE}:\d+ \(IteratorNoYield\.run_remote\) \[kind: IO_TYPE_ERROR\].*"
        r"If the endpoint returns an iterator \(streaming\), it must have `yield` statements"
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class IteratorNoYield(chains.ChainletBase):
            async def run_remote(self) -> AsyncIterator[str]:
                return "123"  # type: ignore[return-value]


def test_raises_yield_no_iterator():
    match = (
        rf"{TEST_FILE}:\d+ \(YieldNoIterator\.run_remote\) \[kind: IO_TYPE_ERROR\].*"
        r"If the endpoint is streaming \(has `yield` statements\), the return type must be an iterator"
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class YieldNoIterator(chains.ChainletBase):
            async def run_remote(self) -> str:  # type: ignore[misc]
                yield "123"


def test_raises_iterator_sync():
    match = (
        rf"{TEST_FILE}:\d+ \(IteratorSync\.run_remote\) \[kind: IO_TYPE_ERROR\].*"
        r"Streaming endpoints \(containing `yield` statements\) are only supported for async endpoints"
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class IteratorSync(chains.ChainletBase):
            def run_remote(self) -> Iterator[str]:
                yield "123"


def test_raises_iterator_no_arg():
    match = (
        rf"{TEST_FILE}:\d+ \(IteratorNoArg\.run_remote\) \[kind: IO_TYPE_ERROR\].*"
        r"Iterators must be annotated with type \(one of \['bytes', 'str'\]\)"
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class IteratorNoArg(chains.ChainletBase):
            async def run_remote(self) -> AsyncIterator:
                yield "123"


def test_raises_is_healthy_not_a_method() -> None:
    match = rf"{TEST_FILE}:\d+ \(IsHealthyNotMethod\) \[kind: TYPE_ERROR\].* `is_healthy` must be a method."

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class IsHealthyNotMethod(chains.ChainletBase):
            is_healthy: int = 3

            async def run_remote(self) -> str:
                return ""


def test_raises_is_healthy_no_arg():
    match = (
        rf"{TEST_FILE}:\d+ \(IsHealthyNoArg\.is_healthy\) \[kind: TYPE_ERROR\].*"
        r"`is_healthy` must be a method, i.e. with `self` as first argument. Got function with no arguments."
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class IsHealthyNoArg(chains.ChainletBase):
            async def is_healthy() -> bool:
                return True

            async def run_remote(self) -> str:
                return ""


def test_raises_is_healthy_first_arg_not_self():
    match = (
        rf"{TEST_FILE}:\d+ \(IsHealthyNoSelfArg\.is_healthy\) \[kind: TYPE_ERROR\].*"
        r"`is_healthy` must be a method, i.e. with `self` as first argument. Got `hi` as first argument."
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class IsHealthyNoSelfArg(chains.ChainletBase):
            def is_healthy(hi) -> bool:
                return True

            async def run_remote(self) -> str:
                return ""


def test_raises_is_healthy_multiple_args():
    match = rf"{TEST_FILE}:\d+ \(IsHealthyManyArgs\.is_healthy\) \[kind: TYPE_ERROR\].* `is_healthy` must have only one argument: `self`."

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class IsHealthyManyArgs(chains.ChainletBase):
            def is_healthy(self, hi) -> bool:
                return True

            async def run_remote(self) -> str:
                return ""


def test_raises_is_healthy_not_type_annotated():
    match = (
        rf"{TEST_FILE}:\d+ \(IsHealthyNotTyped\.is_healthy\) \[kind: IO_TYPE_ERROR\].*"
        r"Return value of health check must be type annotated. Got:\n\tis_healthy\(self\) -> !MISSING!"
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class IsHealthyNotTyped(chains.ChainletBase):
            def is_healthy(self):
                return True

            async def run_remote(self) -> str:
                return ""


def test_raises_is_healthy_not_boolean_typed():
    match = (
        rf"{TEST_FILE}:\d+ \(IsHealthyNotBoolTyped\.is_healthy\) \[kind: IO_TYPE_ERROR\].*"
        r"Return value of health check must be a boolean. Got:\n\tis_healthy\(self\) -> str -> <class 'str'>"
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class IsHealthyNotBoolTyped(chains.ChainletBase):
            def is_healthy(self) -> str:  # type: ignore[misc]
                return "not ready"

            async def run_remote(self) -> str:
                return ""


def test_import_model_requires_entrypoint():
    model_src = TEST_ROOT / "import" / "model_without_inheritance.py"
    match = r"No Model class in `.+` inherits from"
    with pytest.raises(ValueError, match=match), _raise_errors():
        with framework.ModelImporter.import_target(model_src):
            pass


def test_import_model_requires_single_entrypoint():
    model_src = TEST_ROOT / "import" / "standalone_with_multiple_entrypoints.py"
    match = r"Multiple Model classes in `.+` inherit from"
    with pytest.raises(ValueError, match=match), _raise_errors():
        with framework.ModelImporter.import_target(model_src):
            pass


def test_raises_websocket_with_other_args():
    match = (
        rf"{TEST_FILE}:\d+ \(WebsocketWithOtherArgs\.run_remote\) \[kind: IO_TYPE_ERROR\].*"
        r"When using a websocket as input, no other arguments are allowed"
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class WebsocketWithOtherArgs(chains.ChainletBase):
            def run_remote(
                self, websocket: chains.WebSocketProtocol, name: str
            ) -> None:
                pass


def test_raises_websocket_as_output():
    match = (
        rf"{TEST_FILE}:\d+ \(WebsocketOutput\.run_remote\) \[kind: IO_TYPE_ERROR\].*"
        r"Websockets cannot be used as output type"
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class WebsocketOutput(chains.ChainletBase):
            def run_remote(self) -> chains.WebSocketProtocol: ...  # type: ignore[empty-body]


def test_raises_websocket_as_dependency():
    match = (
        rf"{TEST_FILE}:\d+ \(WebsocketAsDependency\.__init__\) \[kind: TYPE_ERROR\].*"
        r"websockets can only be used in the entrypoint.*"
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class Dependency(chains.ChainletBase):
            def run_remote(self, websocket: chains.WebSocketProtocol) -> None:
                pass

        class WebsocketAsDependency(chains.ChainletBase):
            def __init__(self, dependency=chains.depends(Dependency)):
                self._dependency = dependency

            def run_remote(self) -> None:
                pass


def test_raises_websocket_with_return():
    match = (
        rf"{TEST_FILE}:\d+ \(WebsocketOutput\.run_remote\) \[kind: IO_TYPE_ERROR\].*"
        r"Websocket endpoints must have `None` as return type."
    )

    with pytest.raises(definitions.ChainsUsageError, match=match), _raise_errors():

        class WebsocketOutput(chains.ChainletBase):
            async def run_remote(self, websocket: chains.WebSocketProtocol) -> int:
                return 1
