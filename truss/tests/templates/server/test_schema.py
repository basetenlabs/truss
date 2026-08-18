import inspect
from typing import AsyncGenerator, Awaitable, Generator, Union

from pydantic import BaseModel

from truss.templates.server._truss_common.schema import TrussSchema


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


def test_truss_schema_pydantic_empty_input():
    class Model:
        def predict(self) -> ModelOutput:
            return ModelOutput(output="hello")

    model = Model()

    input_signature = inspect.signature(model.predict).parameters
    output_signature = inspect.signature(model.predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)

    assert schema.input_type is None
    assert schema.output_type == ModelOutput


def test_truss_schema_pydantic_empty_output():
    class Model:
        def predict(self, _: ModelInput) -> None:
            return None

    model = Model()

    input_signature = inspect.signature(model.predict).parameters
    output_signature = inspect.signature(model.predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)

    assert schema.input_type == ModelInput
    assert schema.output_type is None


def test_truss_schema_pydantic_empty_input_and_output():
    class Model:
        def predict(self) -> None:
            return None

    model = Model()

    input_signature = inspect.signature(model.predict).parameters
    output_signature = inspect.signature(model.predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)

    assert schema is None


def test_truss_schema_non_pydantic_input():
    class Model:
        def predict(self, request: str) -> ModelOutput:
            return ModelOutput(output=request)

    model = Model()

    input_signature = inspect.signature(model.predict).parameters
    output_signature = inspect.signature(model.predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)

    assert schema.input_type is None
    assert schema.output_type == ModelOutput


def test_truss_schema_non_pydantic_output():
    class Model:
        def predict(self, request: ModelInput) -> str:
            return request.input

    model = Model()

    input_signature = inspect.signature(model.predict).parameters
    output_signature = inspect.signature(model.predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)

    assert schema.input_type == ModelInput
    assert schema.output_type is None


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
            self, request: ModelInput
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
            self, request: ModelInput
        ) -> Union[Awaitable[str], AsyncGenerator[str, None]]:
            return "hello"

    model = Model()

    input_signature = inspect.signature(model.predict).parameters
    output_signature = inspect.signature(model.predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)
    assert schema.input_type == ModelInput
    assert schema.output_type is None


def test_truss_schema_union_non_pydantic():
    class Model:
        def predict(self, request: ModelInput) -> Union[str, int]:
            return "hello"

    model = Model()

    input_signature = inspect.signature(model.predict).parameters
    output_signature = inspect.signature(model.predict).return_annotation

    schema = TrussSchema.from_signature(input_signature, output_signature)

    assert schema.input_type == ModelInput
    assert schema.output_type is None


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

    assert schema.input_type == ModelInput
    assert schema.output_type is None


# -- postprocess schema tests --------------------------------------------------
# These mirror what ModelDescriptor._gen_truss_schema does when postprocess is
# defined: it combines the *input* parameters from predict (or preprocess) with
# the *return annotation* from postprocess to build the TrussSchema.


def test_truss_schema_postprocess_pydantic_output():
    """predict has a pydantic input type; postprocess supplies the pydantic output type."""

    class Model:
        def predict(self, request: ModelInput) -> dict:
            return {"output": request.input}

        def postprocess(self, result: dict) -> ModelOutput:
            return ModelOutput(output=result["output"])

    model = Model()

    # _gen_truss_schema reads predict.parameters for input, postprocess.return for output.
    input_params = inspect.signature(model.predict).parameters
    output_annotation = inspect.signature(model.postprocess).return_annotation

    schema = TrussSchema.from_signature(input_params, output_annotation)

    assert schema is not None
    assert schema.input_type == ModelInput
    assert schema.output_type == ModelOutput
    assert not schema.supports_streaming


def test_truss_schema_preprocess_postprocess_pydantic():
    """preprocess supplies the pydantic input type; postprocess supplies the pydantic output."""

    class Model:
        def preprocess(self, request: ModelInput) -> str:
            return request.input

        def predict(self, request: str) -> str:
            return request

        def postprocess(self, result: str) -> ModelOutput:
            return ModelOutput(output=result)

    model = Model()

    # _gen_truss_schema reads preprocess.parameters for input, postprocess.return for output.
    input_params = inspect.signature(model.preprocess).parameters
    output_annotation = inspect.signature(model.postprocess).return_annotation

    schema = TrussSchema.from_signature(input_params, output_annotation)

    assert schema is not None
    assert schema.input_type == ModelInput
    assert schema.output_type == ModelOutput
    assert not schema.supports_streaming


def test_truss_schema_postprocess_non_pydantic_output():
    """postprocess with a non-pydantic return annotation yields no output_type."""

    class Model:
        def predict(self, request: ModelInput) -> dict:
            return {"output": request.input}

        def postprocess(self, result: dict) -> str:
            return result["output"]

    model = Model()

    input_params = inspect.signature(model.predict).parameters
    output_annotation = inspect.signature(model.postprocess).return_annotation

    schema = TrussSchema.from_signature(input_params, output_annotation)

    assert schema is not None
    assert schema.input_type == ModelInput
    assert schema.output_type is None


def test_truss_schema_postprocess_no_return_annotation():
    """postprocess with no return annotation yields no output_type."""

    class Model:
        def predict(self, request: ModelInput):
            return {"output": request.input}

        def postprocess(self, result):
            return result

    model = Model()

    input_params = inspect.signature(model.predict).parameters
    output_annotation = inspect.signature(model.postprocess).return_annotation

    schema = TrussSchema.from_signature(input_params, output_annotation)

    # input_type is present but output_type is absent.
    assert schema is not None
    assert schema.input_type == ModelInput
    assert schema.output_type is None
