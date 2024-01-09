import inspect
from typing import AsyncGenerator, Awaitable, Generator, Union

from pydantic import BaseModel
from truss.templates.server.common.schema import TrussSchema


class ModelInput(BaseModel):
    input: str
    stream: bool


class ModelOutput(BaseModel):
    output: str


def test_truss_schema_pydantic_input_and_output():
    def predict(request: ModelInput) -> ModelOutput:
        return ModelOutput(output=request.input)

    input_signature = inspect.signature(predict).parameters
    output_signature = inspect.signature(predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)

    assert schema.input_type == ModelInput
    assert schema.output_type == ModelOutput
    assert not schema.supports_streaming


def test_truss_schema_non_pydantic_input():
    def predict(request: str) -> ModelOutput:
        return ModelOutput(output=request)

    input_signature = inspect.signature(predict).parameters
    output_signature = inspect.signature(predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)

    assert schema is None


def test_truss_schema_non_pydantic_output():
    def predict(request: ModelInput) -> str:
        return request.input

    input_signature = inspect.signature(predict).parameters
    output_signature = inspect.signature(predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)

    assert schema is None


def test_truss_schema_async():
    async def predict(request: ModelInput) -> Awaitable[ModelOutput]:
        return ModelOutput(output=request.input)

    input_signature = inspect.signature(predict).parameters
    output_signature = inspect.signature(predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)

    assert schema.input_type == ModelInput
    assert schema.output_type == ModelOutput
    assert not schema.supports_streaming


def test_truss_schema_streaming():
    def predict(request: ModelInput) -> Generator[str, None, None]:
        yield "hello"

    input_signature = inspect.signature(predict).parameters
    output_signature = inspect.signature(predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)

    assert schema.input_type == ModelInput
    assert schema.output_type is None
    assert schema.supports_streaming


def test_truss_schema_streaming_async():
    async def predict(request: ModelInput) -> AsyncGenerator[str, None]:
        yield "hello"

    input_signature = inspect.signature(predict).parameters
    output_signature = inspect.signature(predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)

    assert schema.input_type == ModelInput
    assert schema.output_type is None
    assert schema.supports_streaming


def test_truss_schema_union_sync():
    def predict(request: ModelInput) -> Union[ModelOutput, Generator[str, None, None]]:
        if request.stream:
            return (yield "hello")
        else:
            return ModelOutput(output=request.input)

    input_signature = inspect.signature(predict).parameters
    output_signature = inspect.signature(predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)
    assert schema.input_type == ModelInput
    assert schema.output_type is ModelOutput
    assert schema.supports_streaming


def test_truss_schema_union_async():
    async def predict(
        request: ModelInput,
    ) -> Union[Awaitable[ModelOutput], AsyncGenerator[str, None]]:
        if request.stream:

            def inner():
                for i in range(2):
                    yield str(i)

            return inner()

        return ModelOutput(output=request.input)

    input_signature = inspect.signature(predict).parameters
    output_signature = inspect.signature(predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)
    assert schema.input_type == ModelInput
    assert schema.output_type is ModelOutput
    assert schema.supports_streaming


def test_truss_schema_union_async_non_pydantic():
    async def predict(
        request: ModelInput,
    ) -> Union[Awaitable[str], AsyncGenerator[str, None]]:
        return "hello"

    input_signature = inspect.signature(predict).parameters
    output_signature = inspect.signature(predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)
    assert schema is None


def test_truss_schema_union_non_pydantic():
    def predict(request: ModelInput) -> Union[str, int]:
        return "hello"

    input_signature = inspect.signature(predict).parameters
    output_signature = inspect.signature(predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)

    assert schema is None


def test_truss_schema_async_non_pydantic():
    async def predict(request: str) -> Awaitable[str]:
        return "hello"

    input_signature = inspect.signature(predict).parameters
    output_signature = inspect.signature(predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)
    assert schema is None


def test_truss_schema_union_three_arms():
    class ModelOutput2(BaseModel):
        output2: str

    class ModelOutput3(BaseModel):
        output3: str

    def predict(request: ModelInput) -> Union[ModelOutput, ModelOutput2, ModelOutput3]:
        return ModelOutput(output=request.input)

    input_signature = inspect.signature(predict).parameters
    output_signature = inspect.signature(predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)

    assert schema is None
