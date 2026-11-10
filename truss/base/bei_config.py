from __future__ import annotations

import re
import shlex
from enum import Enum
from typing import Optional

import pydantic
from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id

from truss.base import custom_types

NATIVE_COMMAND_PREFIX = "# Truss-managed BEIv3\n"
HF_SECRET_DEFAULT_ENV = "TRUSS_BEI_HF_TOKEN_DEFAULT"
_IMAGE_TAG_MAX_LENGTH = 128
_DOCKER_TAG_COMPONENT_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
_UNAVAILABLE_QUANTIZATION_MODES = {"mxfp8", "nvfp4"}
_FORBIDDEN_VALUE_CHARACTERS = {
    ";": "';' is reserved as the runtime multi-value delimiter",
    "\x00": "NUL characters are not allowed",
    "\n": "newlines are not allowed",
    "\r": "newlines are not allowed",
}


def _unavailable_quantization_error(value: str) -> ValueError:
    return ValueError(
        f"Quantization mode '{value}' is not available in BEIv3 yet. "
        "The MXFP8 and NVFP4 modes are not supported; currently supported modes "
        "are 'none' and 'fp8' (NVIDIA only)."
    )


class BEIBackend(str, Enum):
    NVIDIA = "nvidia"
    TPU = "tpu"


class BEIQuantization(str, Enum):
    NONE = "none"
    FP8 = "fp8"


def _validate_safe_value(value: str, field_name: str) -> str:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    for character, reason in _FORBIDDEN_VALUE_CHARACTERS.items():
        if character in value:
            raise ValueError(f"{field_name} contains an invalid character: {reason}")
    return value


class BEIModelConfiguration(custom_types.ConfigModel):
    checkpoint: str = pydantic.Field(
        ...,
        description="HuggingFace repository ID or absolute local directory containing the model.",
    )
    revision: Optional[str] = pydantic.Field(
        default=None,
        description="HuggingFace revision. Defaults to the backend's default (main).",
    )
    served_model_name: Optional[str] = pydantic.Field(
        default=None,
        description="Name exposed by the model server. Defaults to the checkpoint.",
    )
    quantization: BEIQuantization = BEIQuantization.NONE

    @pydantic.field_validator("checkpoint")
    @classmethod
    def _validate_checkpoint(cls, value: str) -> str:
        value = _validate_safe_value(value, "checkpoint")
        if not value.startswith("/"):
            try:
                validate_repo_id(value)
            except HFValidationError as e:
                raise ValueError(
                    f"HuggingFace repository validation failed: {e}"
                ) from e
        return value

    @pydantic.field_validator("revision")
    @classmethod
    def _validate_revision(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if value.lower() == "none":
            raise ValueError(
                "revision 'none' is reserved by BEIv3; use null for the default revision"
            )
        return _validate_safe_value(value, "revision")

    @pydantic.field_validator("served_model_name")
    @classmethod
    def _validate_served_model_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if value.lower() == "none":
            raise ValueError("served_model_name 'none' is reserved by BEIv3")
        return _validate_safe_value(value, "served_model_name")

    @pydantic.field_validator("quantization", mode="before")
    @classmethod
    def _validate_quantization(cls, value):
        if isinstance(value, str) and value.lower() in _UNAVAILABLE_QUANTIZATION_MODES:
            raise _unavailable_quantization_error(value)
        return value

    @property
    def effective_served_model_name(self) -> str:
        return self.served_model_name or self.checkpoint


class BEIConfiguration(custom_types.ConfigModel):
    backend: BEIBackend
    version_override: Optional[str] = pydantic.Field(
        default=None, description="Optional version/tag prefix for the backend image."
    )
    configuration: list[BEIModelConfiguration] = pydantic.Field(
        ..., min_length=1, description="Models served by this BEIv3 deployment."
    )

    @pydantic.field_validator("version_override")
    @classmethod
    def _validate_version_override(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if not value:
            raise ValueError("version_override must not be empty")
        if ":" in value:
            raise ValueError(
                "version_override must be a tag prefix only and must not contain ':'"
            )
        if "/" in value:
            raise ValueError(
                "version_override must be a tag prefix only and must not contain a registry"
            )
        if _DOCKER_TAG_COMPONENT_PATTERN.fullmatch(value) is None:
            raise ValueError(
                "version_override must be a valid Docker tag component "
                "(alphanumeric, '_', '.', or '-', not starting with '.' or '-')"
            )
        return value

    @pydantic.model_validator(mode="after")
    def _validate_backend_compatibility(self) -> "BEIConfiguration":
        seen_names: set[str] = set()
        for model in self.configuration:
            name = model.effective_served_model_name
            if name in seen_names:
                raise ValueError(
                    f"Duplicate effective served_model_name '{name}'. "
                    "Each model must have a unique served name, including defaults "
                    "derived from the checkpoint."
                )
            seen_names.add(name)

        if self.backend == BEIBackend.TPU:
            if any(
                model.quantization == BEIQuantization.FP8
                for model in self.configuration
            ):
                raise ValueError(
                    "Quantization mode 'fp8' is not supported with the TPU backend. "
                    "TPU currently supports only 'none' quantization."
                )
            if len(self.configuration) > 1:
                raise ValueError(
                    "Multiple models are not supported with the TPU backend until "
                    "hardware sharing support is confirmed."
                )

        if self.version_override is not None:
            final_tag = f"{self.version_override}-{self.backend.value}"
            if len(final_tag) > _IMAGE_TAG_MAX_LENGTH:
                raise ValueError(
                    f"version_override produces a Docker tag of length {len(final_tag)}; "
                    f"the maximum is {_IMAGE_TAG_MAX_LENGTH} and the "
                    f"'-{self.backend.value}' suffix counts toward this limit."
                )

        return self

    def start_command(self, use_hf_secret: bool) -> str:
        """Launch the native executable using only the prebuilt image's shell."""
        argv = [
            "/usr/local/bin/bei",
            "--hostname=0.0.0.0",
            "--port=7997",
            "--json-output",
        ]
        if self.backend == BEIBackend.NVIDIA:
            argv.extend(["--backend=libtorch", "--torch-device=cuda"])
        else:
            argv.extend(["--backend=python-backend", "--dtype=bfloat16"])
        for model in self.configuration:
            argv.extend(
                [
                    f"--model-id={model.checkpoint}",
                    f"--revision={model.revision or 'main'}",
                    f"--served-model-name={model.effective_served_model_name}",
                    f"--enable-fp8-dynamic={'true' if model.quantization == BEIQuantization.FP8 else 'false'}",
                ]
            )
            if self.backend == BEIBackend.NVIDIA and len(self.configuration) > 1:
                argv.append("--backend-device-ids=auto")
        script = [
            NATIVE_COMMAND_PREFIX.rstrip(),
            "set -eu",
            """if [ -z "${BEI_HF_TOKEN:-}" ] && [ -n "${HF_TOKEN:-}" ]; then
    export BEI_HF_TOKEN="$HF_TOKEN"
fi""",
        ]
        if use_hf_secret:
            script.append("""truss_hf_token=''
if [ -f /secrets/hf_access_token ]; then
    truss_hf_token="$(cat /secrets/hf_access_token)"
elif [ "${TRUSS_SECRET_hf_access_token+x}" = x ]; then
    truss_hf_token="$TRUSS_SECRET_hf_access_token"
elif [ "${TRUSS_BEI_HF_TOKEN_DEFAULT+x}" = x ]; then
    truss_hf_token="$TRUSS_BEI_HF_TOKEN_DEFAULT"
fi
if [ -n "$truss_hf_token" ]; then
    BEI_HF_TOKEN="$truss_hf_token"
fi
if [ -z "${BEI_HF_TOKEN:-}" ]; then
    echo 'The declared hf_access_token secret is missing at runtime.' >&2
    exit 1
fi
export BEI_HF_TOKEN""")
        script.append("exec " + shlex.join(argv))
        return "\n".join(script)
