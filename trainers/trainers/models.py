"""Operation types exchanged between the queue, controller, and worker."""

from __future__ import annotations

import base64
from datetime import datetime
from typing import Annotated, Literal, Self, Union

from pydantic import BaseModel, Discriminator, Field, Tag


# ── Tensor / model-input primitives ────────────────────────────────


class TensorData(BaseModel):
    data: list
    dtype: str
    shape: list[int]

    @classmethod
    def from_list(cls, data: list, dtype: str = "float32") -> Self:
        flat = data
        shape = []
        level = data
        while isinstance(level, list):
            shape.append(len(level))
            level = level[0] if level else []
        return cls(data=flat, dtype=dtype, shape=shape)


class EncodedTextChunk(BaseModel):
    type: Literal["encoded_text"] = "encoded_text"
    tokens: list[int]

    @property
    def length(self) -> int:
        return len(self.tokens)


class ImageChunk(BaseModel):
    type: Literal["image"] = "image"
    data: str  # base64-encoded
    format: str
    expected_tokens: int = 0

    @classmethod
    def from_bytes(cls, raw: bytes, fmt: str, expected_tokens: int = 0) -> Self:
        return cls(
            data=base64.b64encode(raw).decode(),
            format=fmt,
            expected_tokens=expected_tokens,
        )

    def to_bytes(self) -> bytes:
        return base64.b64decode(self.data)

    @property
    def length(self) -> int:
        return self.expected_tokens


ModelInputChunk = Annotated[
    Union[
        Annotated[EncodedTextChunk, Tag("encoded_text")],
        Annotated[ImageChunk, Tag("image")],
    ],
    Discriminator("type"),
]


class ModelInput(BaseModel):
    chunks: list[ModelInputChunk] = Field(default_factory=list)

    @classmethod
    def from_ints(cls, tokens: list[int]) -> Self:
        return cls(chunks=[EncodedTextChunk(tokens=tokens)])

    @classmethod
    def empty(cls) -> Self:
        return cls(chunks=[])

    @property
    def length(self) -> int:
        return sum(c.length for c in self.chunks)

    def to_ints(self) -> list[int]:
        out: list[int] = []
        for chunk in self.chunks:
            if not isinstance(chunk, EncodedTextChunk):
                raise TypeError(f"Cannot convert {type(chunk).__name__} to ints")
            out.extend(chunk.tokens)
        return out

    def append(self, chunk: ModelInputChunk) -> Self:
        return type(self)(chunks=[*self.chunks, chunk])


class Datum(BaseModel):
    model_input: ModelInput
    loss_fn_inputs: dict[str, TensorData] = Field(default_factory=dict)


# ── Result types ───────────────────────────────────────────────────

LossFnOutput = dict[str, TensorData]


class ForwardBackwardResult(BaseModel):
    loss_fn_output_type: str = ""
    loss_fn_outputs: list[LossFnOutput] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


class OptimStepResult(BaseModel):
    metrics: dict[str, float] | None = None


class ToInferenceResult(BaseModel):
    mode: str = ""


class SaveStateResult(BaseModel):
    mode: str = ""


# ── Chat / training primitives ───────────────────────────────────


class Message(BaseModel):
    role: str
    content: str


# ── Sample types ──────────────────────────────────────────────────


class SampleInput(BaseModel):
    messages: list[Message]
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0


class SampleOutput(BaseModel):
    generated_text: str
    messages: list[Message]


class SampleResult(BaseModel):
    outputs: list[SampleOutput] = Field(default_factory=list)


# ── Detail payloads per operation type ───────────────────────────


class ForwardBackwardDetails(BaseModel):
    batch: list[Datum] = Field(min_length=1)
    loss_fn: str = "cross_entropy"
    loss_fn_config: dict | None = None


class OptimStepDetails(BaseModel):
    pass


class SampleDetails(BaseModel):
    inputs: list[SampleInput] = Field(min_length=1)


class SaveWeightsAndGetSamplingClientDetails(BaseModel):
    pass


class SaveStateDetails(BaseModel):
    checkpoint_dir: str


# ── Operation variants ───────────────────────────────────────────


class _OperationBase(BaseModel):
    operation_id: str
    created_at: datetime


class ForwardBackwardOp(_OperationBase):
    type: Literal["forward_backward"] = "forward_backward"
    forward_backward_details: ForwardBackwardDetails


class OptimStepOp(_OperationBase):
    type: Literal["optim_step"] = "optim_step"
    optim_step_details: OptimStepDetails = Field(default_factory=OptimStepDetails)


class SampleOp(_OperationBase):
    type: Literal["sample"] = "sample"
    sample_details: SampleDetails


class SaveWeightsAndGetSamplingClientOp(_OperationBase):
    type: Literal["save_weights_and_get_sampling_client"] = "save_weights_and_get_sampling_client"
    save_weights_and_get_sampling_client_details: SaveWeightsAndGetSamplingClientDetails = Field(
        default_factory=SaveWeightsAndGetSamplingClientDetails
    )


class SaveStateOp(_OperationBase):
    type: Literal["save_state"] = "save_state"
    save_state_details: SaveStateDetails


Operation = Annotated[
    Union[
        Annotated[ForwardBackwardOp, Tag("forward_backward")],
        Annotated[OptimStepOp, Tag("optim_step")],
        Annotated[SampleOp, Tag("sample")],
        Annotated[SaveWeightsAndGetSamplingClientOp, Tag("save_weights_and_get_sampling_client")],
        Annotated[SaveStateOp, Tag("save_state")],
    ],
    Discriminator("type"),
]


# ── Status updates sent back to the queue ────────────────────────


class OperationStatus(BaseModel):
    operation_id: str
    status: Literal["pending", "in_progress", "completed", "failed"]
    result: dict | None = None


# ── Queue server request/response models ─────────────────────────


class EnqueueRequest(BaseModel):
    ops: list[Operation]


class EnqueueResponse(BaseModel):
    op_ids: list[str]


class GetOpStatusesRequest(BaseModel):
    op_ids: list[str]


class GetOpStatusesResponse(BaseModel):
    statuses: list[OperationStatus]
