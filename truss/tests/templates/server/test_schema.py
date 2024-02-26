import inspect
from typing import AsyncGenerator, Awaitable, Generator, Union

from pydantic import BaseModel
from truss.templates.server.common.schema import TrussSchema


class ModelInput(BaseModel):
    input: str
    stream: bool


class ModelOutput(BaseModel):
    output: str


def test_truss_schema_pydantic_empty_annotations():
    class Model:
        def predict(self, request):
            return "hello"

    model = Model()

    input_signature = inspect.signature(model.predict).parameters
    output_signature = inspect.signature(model.predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)

    assert schema is None


def test_truss_schema_pydantic_input_and_output():
    class Model:
        def predict(self, request: ModelInput) -> ModelOutput:
            return ModelOutput(output=request.input)

    model = Model()

    input_signature = inspect.signature(model.predict).parameters
    output_signature = inspect.signature(model.predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)

    assert schema.input_type == ModelInput
    assert schema.output_type == ModelOutput
    assert not schema.supports_streaming


def test_truss_schema_non_pydantic_input():
    class Model:
        def predict(self, request: str) -> ModelOutput:
            return ModelOutput(output=request)

    model = Model()

    input_signature = inspect.signature(model.predict).parameters
    output_signature = inspect.signature(model.predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)

    assert schema is None


def test_truss_schema_non_pydantic_output():
    class Model:
        def predict(self, request: ModelInput) -> str:
            return request.input

    model = Model()

    input_signature = inspect.signature(model.predict).parameters
    output_signature = inspect.signature(model.predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)

    assert schema is None


def test_truss_schema_list_types():
    class Model:
        def predict(self, request: list[str]) -> list[str]:
            return ["foo", "bar"]

    model = Model()

    input_signature = inspect.signature(model.predict).parameters
    output_signature = inspect.signature(model.predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)

    assert schema is None


def test_truss_schema_dict_types():
    class Model:
        def predict(self, request: dict[str, str]) -> dict[str, str]:
            return {"foo": "bar"}

    model = Model()

    input_signature = inspect.signature(model.predict).parameters
    output_signature = inspect.signature(model.predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)

    assert schema is None


def test_truss_schema_async():
    class Model:
        async def predict(self, request: ModelInput) -> Awaitable[ModelOutput]:
            return ModelOutput(output=request.input)

    model = Model()

    input_signature = inspect.signature(model.predict).parameters
    output_signature = inspect.signature(model.predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)

    assert schema.input_type == ModelInput
    assert schema.output_type == ModelOutput
    assert not schema.supports_streaming


def test_truss_schema_streaming():
    class Model:
        def predict(self, request: ModelInput) -> Generator[str, None, None]:
            yield "hello"

    model = Model()

    input_signature = inspect.signature(model.predict).parameters
    output_signature = inspect.signature(model.predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)

    assert schema.input_type == ModelInput
    assert schema.output_type is None
    assert schema.supports_streaming


def test_truss_schema_streaming_async():
    class Model:
        async def predict(self, request: ModelInput) -> AsyncGenerator[str, None]:
            yield "hello"

    model = Model()

    input_signature = inspect.signature(model.predict).parameters
    output_signature = inspect.signature(model.predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)

    assert schema.input_type == ModelInput
    assert schema.output_type is None
    assert schema.supports_streaming


def test_truss_schema_union_sync():
    class Model:
        def predict(
            self, request: ModelInput
        ) -> Union[ModelOutput, Generator[str, None, None]]:
            if request.stream:
                return (yield "hello")
            else:
                return ModelOutput(output=request.input)

    model = Model()

    input_signature = inspect.signature(model.predict).parameters
    output_signature = inspect.signature(model.predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)
    assert schema.input_type == ModelInput
    assert schema.output_type == ModelOutput
    assert schema.supports_streaming


def test_truss_schema_union_async():
    class Model:
        async def predict(
            self,
            request: ModelInput,
        ) -> Union[Awaitable[ModelOutput], AsyncGenerator[str, None]]:
            if request.stream:

                def inner():
                    for i in range(2):
                        yield str(i)

                return inner()

            return ModelOutput(output=request.input)

    model = Model()

    input_signature = inspect.signature(model.predict).parameters
    output_signature = inspect.signature(model.predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)
    assert schema.input_type == ModelInput
    assert schema.output_type is ModelOutput
    assert schema.supports_streaming


def test_truss_schema_union_async_non_pydantic():
    class Model:
        async def predict(
            self,
            request: ModelInput,
        ) -> Union[Awaitable[str], AsyncGenerator[str, None]]:
            return "hello"

    model = Model()

    input_signature = inspect.signature(model.predict).parameters
    output_signature = inspect.signature(model.predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)
    assert schema is None


def test_truss_schema_union_non_pydantic():
    class Model:
        def predict(self, request: ModelInput) -> Union[str, int]:
            return "hello"

    model = Model()

    input_signature = inspect.signature(model.predict).parameters
    output_signature = inspect.signature(model.predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)

    assert schema is None


def test_truss_schema_async_non_pydantic():
    class Model:
        async def predict(self, request: str) -> Awaitable[str]:
            return "hello"

    model = Model()

    input_signature = inspect.signature(model.predict).parameters
    output_signature = inspect.signature(model.predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)
    assert schema is None


def test_truss_schema_union_three_arms():
    class ModelOutput2(BaseModel):
        output2: str

    class ModelOutput3(BaseModel):
        output3: str

    class Model:
        def predict(
            self, request: ModelInput
        ) -> Union[ModelOutput, ModelOutput2, ModelOutput3]:
            return ModelOutput(output=request.input)

    model = Model()
    input_signature = inspect.signature(model.predict).parameters
    output_signature = inspect.signature(model.predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)

    assert schema is None
