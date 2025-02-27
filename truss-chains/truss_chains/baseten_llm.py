from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class ModelInput(BaseModel):
    """This class mirrors the `CompletionCreateParamsBase` in the `openai-python` repository.

    However, that class is a TypedDict rather than a pydantic model, so we redefine it here
    to take advantage of pydantic's validation features. In addition, we define helper methods
    to get the formatted prompt, tools to use, and response format to adhere to.

    Unsupported parameters:
    - https://platform.openai.com/docs/api-reference/chat/create#chat-create-store
      - OpenAI platform specific
    - https://platform.openai.com/docs/api-reference/chat/create#chat-create-metadata
      - OpenAI platform specific
    - https://platform.openai.com/docs/api-reference/chat/create#chat-create-frequency_penalty
      - Frequency penalty is not currently passed through to briton
    - https://platform.openai.com/docs/api-reference/chat/create#chat-create-logit_bias
      - User provided logit biasing is not implemented
    - https://platform.openai.com/docs/api-reference/chat/create#chat-create-logprobs
      - Returning log probabilities is not implemented
    - https://platform.openai.com/docs/api-reference/chat/create#chat-create-top_logprobs
      - Returning log probabilities is not implemented
    - https://platform.openai.com/docs/api-reference/chat/create#chat-create-service_tier
      - OpenAI platform specific
    - https://platform.openai.com/docs/api-reference/chat/create#chat-create-user
      - OpenAI platform specific
    - https://platform.openai.com/docs/api-reference/chat/create#chat-create-function_call
      - Deprecated
    - https://platform.openai.com/docs/api-reference/chat/create#chat-create-functions
      - Deprecated
    """

    class Tool(BaseModel):
        """An element in the top level `tools` field."""

        class Function(BaseModel):
            name: str
            description: Optional[str] = Field(None)
            parameters_: Optional[Dict[str, Any]] = Field(None, alias="parameters")
            return_: Optional[Dict[str, Any]] = Field(None, alias="return")

            @model_validator(mode="after")
            def definitions_valid(cls, values):
                if "definitions" in values.parameters and "$defs" in values.parameters:
                    raise ValueError(
                        "Both pydantic v1 and v2 definitions found; please check schema."
                    )
                return values

            @property
            def parameters(self) -> Dict[str, Any]:
                if self.parameters_ is None:
                    return {"properties": {}}
                elif "properties" not in self.parameters_:
                    return {"properties": {}, **self.parameters_}
                else:
                    return self.parameters_

            @property
            def parameters_without_definitions(self) -> Dict[str, Any]:
                parameters = self.parameters.copy()
                for keyword in ["definitions", "$defs"]:
                    parameters.pop(keyword, None)
                return parameters

            @property
            def definitions(self) -> Optional[tuple[Dict[str, Any], str]]:
                for keyword in ["definitions", "$defs"]:
                    if keyword in self.parameters:
                        return self.parameters[keyword], keyword
                return None

            @property
            def json_schema(self) -> Dict[str, Any]:
                return {
                    "type": "object",
                    "properties": {
                        "name": {"const": self.name},
                        "parameters": self.parameters_without_definitions,
                    },
                    "required": ["name", "parameters"],
                }

        type: Literal["function"]
        function: Function

    class ToolChoice(BaseModel):
        """The top level `tool_choice` field."""

        class FunctionChoice(BaseModel):
            name: str

        type: Literal["function"]
        function: FunctionChoice

    class SchemaResponseFormat(BaseModel):
        """The top level `response_format` field."""

        class JsonSchema(BaseModel):
            """`schema_` holds the actual json schema"""

            schema_: Dict[str, Any] = Field(..., alias="schema")

        type: Literal["json_schema"]
        json_schema: JsonSchema

    class JsonResponseFormat(BaseModel):
        type: Literal["json_object"]

    class TextResponseFormat(BaseModel):
        type: Literal["text"]

    class StreamOptions(BaseModel):
        """The top level `stream_options` field."""

        include_usage: bool

    class LookaheadDecodingConfig(BaseModel):
        window_size: int
        ngram_size: int
        verification_set_size: int

    model: Optional[str] = Field("")

    messages: Optional[List[Dict[str, Any]]] = Field(None)
    prompt: Optional[str] = Field(None, min_length=1)

    max_tokens: Optional[int] = Field(None)
    max_completion_tokens: Optional[int] = Field(None)

    stream: Optional[bool] = Field(None)
    stream_options: Optional[StreamOptions] = Field(None)

    seed: Optional[int] = Field(None)
    random_seed: Optional[int] = Field(None)

    frequency_penalty: Optional[float] = Field(0)
    presence_penalty: Optional[float] = Field(0)
    length_penalty: Optional[float] = Field(None)

    # Not part of openai spec but supported by briton
    repetition_penalty: Optional[float] = Field(None)
    temperature: Optional[float] = Field(1.0)
    top_p: Optional[float] = Field(1.0)
    runtime_top_p: Optional[float] = Field(None)
    top_k: Optional[int] = Field(50)
    runtime_top_k: Optional[int] = Field(None)
    stop: Optional[Union[str, List[str]]] = Field(None)
    bad_words_: Optional[Union[str, List[str]]] = Field(None)
    skip_special_tokens: Optional[List[str]] = Field(None)
    response_format: Optional[
        Union[SchemaResponseFormat, JsonResponseFormat, TextResponseFormat]
    ] = Field(None)
    tools: Optional[List[Tool]] = Field(None)
    tool_choice: Optional[Union[Literal["none", "required", "auto"], ToolChoice]] = (
        Field(None)
    )
    parallel_tool_calls: Optional[bool] = Field(True)
    beam_width: Optional[Literal[1]] = Field(None)
    n: Optional[int] = Field(1)
    end_id: Optional[int] = Field(None)
    pad_id: Optional[int] = Field(None)
    # WiM fields
    margins_prompt: Optional[str] = Field(None)
    margins_stop_sequences: Optional[List[str]] = Field(["NO#"])
    max_chunk_size: Optional[int] = Field(4096)
    # Lookahead Decoding
    lookahead_decoding_config: Optional[LookaheadDecodingConfig] = Field(None)
