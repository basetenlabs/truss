"""Types for the trainers SDK.

Core data types (Datum, ModelInput, TensorData, etc.) are re-exported from the
tinker SDK to ensure wire-format compatibility. Trainers-specific types are
defined here.
"""

from __future__ import annotations

# Re-export tinker types as the canonical data models.
# Pinned to tinker >=0.13.1,<0.14 in pyproject.toml.
from tinker.types import (  # noqa: F401
    AdamParams,
    Datum,
    EncodedTextChunk,
    ForwardBackwardOutput,
    ImageChunk,
    LoadWeightsResponse,
    ModelInput,
    ModelInputChunk,
    OptimStepResponse,
    SampledSequence,
    SampleResponse,
    SamplingParams,
    SaveWeightsForSamplerResponse,
    SaveWeightsResponse,
    TensorData,
)

from pydantic import BaseModel, Field


# ── Trainers-specific types (not in tinker) ──────────────────────


class Message(BaseModel):
    role: str
    content: str


class SampleInput(BaseModel):
    messages: list[Message]
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0


class SampleDetails(BaseModel):
    inputs: list[SampleInput] = Field(min_length=1)


class ForwardBackwardDetails(BaseModel):
    batch: list[Datum] = Field(min_length=1)
    loss_fn: str = "cross_entropy"
    loss_fn_config: dict | None = None
