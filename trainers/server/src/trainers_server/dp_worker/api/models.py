from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

class StatusResult(BaseModel):
    mode: Literal["training", "inference"]
    step: int
    model_id: str
    device: str
    last_loss: Optional[float] = None
    grad_norm: Optional[float] = None
    gpu_memory: Dict[str, int] = Field(default_factory=dict)


class InferenceServerConfig(BaseModel):
    tensor_parallel_size: int = 1
    gpus: List[int] = [1]
    gpu_memory_utilization: float = 0.9


class TrainingServerConfig(BaseModel):
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_length: int = 2048
    gpus: List[int] = [0]


class RLControllerConfig(BaseModel):
    model_id: str = "Qwen/Qwen3-0.6B"
    inference: InferenceServerConfig = Field(default_factory=InferenceServerConfig)
    training: TrainingServerConfig = Field(default_factory=TrainingServerConfig)