import shlex
from typing import Literal, Optional

import pydantic

from truss.base.custom_types import ConfigModel

NATIVE_COMMAND_PREFIX = "# Truss-managed BEIv3\n"


class BEIModelConfiguration(ConfigModel):
    checkpoint: str = pydantic.Field(min_length=1)
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
        if self.backend == "tpu" and (
            len(self.configuration) != 1 or self.configuration[0].quantization != "none"
        ):
            raise ValueError("BEI TPU currently supports one unquantized model.")
        return self

    def start_command(self, use_hf_secret: bool) -> str:
        argv = [
            "/usr/local/bin/bei",
            "--hostname=0.0.0.0",
            "--port=7997",
            "--json-output",
        ]
        if self.backend == "nvidia":
            argv.extend(["--backend=libtorch", "--torch-device=cuda"])
        else:
            argv.extend(["--backend=python-backend", "--dtype=bfloat16"])
        for model in self.configuration:
            argv.extend(
                [
                    f"--model-id={model.checkpoint}",
                    f"--revision={model.revision or 'main'}",
                    f"--served-model-name={model.served_model_name or model.checkpoint}",
                    f"--dynamic-quantization={model.quantization}",
                ]
            )
            if self.backend == "nvidia":
                argv.append(
                    "--dtype="
                    + (
                        "bfloat16"
                        if model.quantization in ("mxfp8", "nvfp4")
                        else "auto"
                    )
                )
                if len(self.configuration) > 1:
                    argv.append("--backend-device-ids=auto")
        script = NATIVE_COMMAND_PREFIX
        if use_hf_secret:
            script += """set -eu
if [ -s /secrets/hf_access_token ]; then
    export BEI_HF_TOKEN="$(cat /secrets/hf_access_token)"
elif [ -n "${TRUSS_SECRET_hf_access_token:-}" ]; then
    export BEI_HF_TOKEN="$TRUSS_SECRET_hf_access_token"
fi
if [ -z "${BEI_HF_TOKEN:-}${HF_TOKEN:-}" ]; then
    echo 'Missing hf_access_token secret.' >&2
    exit 1
fi
"""
        return script + "exec " + shlex.join(argv)
