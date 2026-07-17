from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, field_validator, model_validator

from truss.base.llm_config import TrussLLMSharedConfig

logger = logging.getLogger(__name__)


def _format_cli_arg(key: str, value: Any) -> str:
    flag = key.replace("_", "-")
    if isinstance(value, bool):
        return f"--{flag}" if value else ""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        import json

        return f"--{flag} {json.dumps(value)}"
    return f"--{flag} {value}"


def _format_patch_kwargs(patch_kwargs: Dict[str, Any]) -> List[str]:
    parts: List[str] = []
    for k, v in patch_kwargs.items():
        cli = _format_cli_arg(k, v)
        if cli:
            parts.append(cli)
    return parts


class VLLMConfiguration(TrussLLMSharedConfig):
    model: str = Field(..., description="Model ID or local path to serve. e.g. meta-llama/Llama-2-7b-hf")
    port: int = Field(default=8000, description="Port for the vLLM OpenAI-compatible server.")
    host: str = Field(default="0.0.0.0", description="Host to bind the vLLM server.")
    gpu_memory_utilization: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Fraction of GPU memory to use (0.0 - 1.0)."
    )
    version_overrides: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="Version overrides, e.g. {vllm_version: '0.19.1'} -> resolved via backend constance. "
        "Mirrors trt_llm.version_overrides pattern but kept generic for vLLM.",
    )

    @field_validator("gpu_memory_utilization")
    @classmethod
    def _validate_gpu_memory_utilization(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v <= 0.0 or v > 1.0):
            raise ValueError("gpu_memory_utilization must be in (0.0, 1.0]")
        return v

    @field_validator("tensor_parallel_size")
    @classmethod
    def _validate_tp(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        return v

    @model_validator(mode="after")
    def _validate_model(self) -> "VLLMConfiguration":
        if not self.model:
            raise ValueError("model must be specified for vLLM")
        return self

    def build_start_command(self, accelerator_count: Optional[int] = None) -> str:
        cmd_parts = ["vllm serve", self.model]

        tp = self.tensor_parallel_size
        if tp is None and accelerator_count is not None and accelerator_count > 0:
            tp = accelerator_count

        simple_flags = {
            "port": self.port,
            "host": self.host,
            "tensor_parallel_size": tp,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "served_model_name": self.served_model_name,
        }
        for key, value in simple_flags.items():
            if value is not None:
                arg = _format_cli_arg(key, value)
                if arg:
                    cmd_parts.append(arg)

        if self.trust_remote_code:
            cmd_parts.append("--trust-remote-code")

        for arg in _format_patch_kwargs(self.patch_kwargs):
            cmd_parts.append(arg)

        for arg in self.extra_args:
            cmd_parts.append(arg)

        return " ".join(cmd_parts)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        return super().model_dump(**kwargs)
