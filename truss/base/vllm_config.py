from __future__ import annotations

import json
import shlex
from typing import Any, Dict, List, Optional

from pydantic import Field, model_validator

from truss.base.llm_config import TrussLLMSharedConfig


def _format_cli_arg(key: str, value: Any) -> str:
    """Formats one `vllm serve` flag, shell-quoted for the docker_server start_command."""
    flag = key.replace("_", "-")
    if isinstance(value, bool):
        return f"--{flag}" if value else f"--no-{flag}"
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return f"--{flag} {shlex.quote(json.dumps(value))}"
    return f"--{flag} {shlex.quote(str(value))}"


def _format_patch_kwargs(patch_kwargs: Dict[str, Any]) -> List[str]:
    parts: List[str] = []
    for k, v in patch_kwargs.items():
        cli = _format_cli_arg(k, v)
        if cli:
            parts.append(cli)
    return parts


class VLLMConfiguration(TrussLLMSharedConfig):
    model: str = Field(
        ...,
        description="Model ID or local path to serve. e.g. meta-llama/Llama-2-7b-hf",
    )
    port: int = Field(
        default=8000, description="Port for the vLLM OpenAI-compatible server."
    )
    host: str = Field(default="0.0.0.0", description="Host to bind the vLLM server.")
    gpu_memory_utilization: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Fraction of GPU memory to use (0.0 - 1.0].",
    )

    @model_validator(mode="after")
    def _validate_model(self) -> "VLLMConfiguration":
        if not self.model:
            raise ValueError("model must be specified for vLLM")
        return self

    def build_start_command(self, accelerator_count: Optional[int] = None) -> str:
        cmd_parts = ["vllm serve", shlex.quote(self.model)]

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
