import shlex
from typing import Literal, Optional

import pydantic

from truss.base.custom_types import ConfigModel

NATIVE_COMMAND_PREFIX = "# Truss-managed BEIv3\n"


class BEIModelConfiguration(ConfigModel):
    checkpoint: str = pydantic.Field(
        min_length=1,
        description="Hugging Face repository to download through weights, or an absolute path to manually mounted weights.",
    )
    revision: Optional[str] = None
    served_model_name: Optional[str] = None
    quantization: Literal["none", "fp8", "mxfp8", "nvfp4"] = "none"

    @pydantic.field_validator("checkpoint", "revision", "served_model_name")
    @classmethod
    def _validate_model_value(cls, value, info):
        if value is not None and (
            not value.strip()
            or any(character in value for character in (";", "\x00", "\n", "\r"))
            or (info.field_name != "checkpoint" and value.lower() == "none")
        ):
            raise ValueError(f"Invalid BEI {info.field_name}: {value!r}")
        return value


class BEIConfiguration(ConfigModel):
    backend: Literal["nvidia", "tpu"]
    version_override: Optional[str] = None
    configuration: list[BEIModelConfiguration] = pydantic.Field(min_length=1)

    @pydantic.model_validator(mode="after")
    def _validate_models(self):
        names = [
            model.served_model_name or model.checkpoint for model in self.configuration
        ]
        if len(set(names)) != len(names):
            raise ValueError("BEI served model names must be unique.")
        return self

    def start_command(self, model_paths: list[str]) -> str:
        argv = [
            "/usr/local/bin/bei",
            "--hostname=0.0.0.0",
            "--port=8080",
            "--json-output",
        ]
        if self.backend == "nvidia":
            argv.extend(["--backend=libtorch", "--torch-device=cuda"])
        else:
            argv.append("--backend=python-backend")
        # NVIDIA MXFP8/NVFP4 need BF16; mixed lists need aligned dtype flags.
        needs_dtype = self.backend == "nvidia" and any(
            model.quantization in ("mxfp8", "nvfp4") for model in self.configuration
        )
        for model, model_path in zip(self.configuration, model_paths):
            argv.extend(
                [
                    f"--model-id={model_path}",
                    f"--revision={model.revision or 'main'}",
                    f"--served-model-name={model.served_model_name or model.checkpoint}",
                    f"--dynamic-quantization={model.quantization}",
                ]
            )
            if needs_dtype:
                argv.append(
                    "--dtype="
                    + (
                        "bfloat16"
                        if model.quantization in ("mxfp8", "nvfp4")
                        else "auto"
                    )
                )
            if self.backend == "nvidia" and len(self.configuration) > 1:
                argv.append("--backend-device-ids=auto")
        return NATIVE_COMMAND_PREFIX + "exec " + shlex.join(argv)
