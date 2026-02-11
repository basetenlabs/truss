"""SFT training frameworks - transformers, trl, megatron (ms-swift)."""

from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Union

from truss_train.definitions import AutoSFT, Runtime, SecretReference, TrainingProject


def get_environment_variables(sft_config: AutoSFT) -> Dict[str, Union[str, SecretReference]]:
    """Build env vars with HF_TOKEN default, merged with user's environment_variables."""
    env: Dict[str, Union[str, SecretReference]] = {
        "HF_TOKEN": SecretReference(name="hf_access_token"),
    }
    if sft_config.environment_variables:
        for k, v in sft_config.environment_variables.items():
            if isinstance(v, dict) and ("secret" in v or "name" in v):
                secret_name = v.get("secret") or v.get("name", "")
                env[k] = SecretReference(name=secret_name)
            elif isinstance(v, str):
                env[k] = v
    return env


@dataclass
class FrameworkOutput:
    """Output from a framework's generation."""

    runtime: Runtime
    train_script: str
    base_image: Optional[str] = None


class SFTFramework(Protocol):
    """Protocol for SFT framework implementations."""

    name: str

    def generate(self, sft_config: AutoSFT) -> FrameworkOutput:
        """Generate runtime config and training script from AutoSFT config."""
        ...


def get_framework(sft_config: AutoSFT) -> SFTFramework:
    """Select and return the appropriate framework for the given config."""
    # Auto-select megatron (ms-swift) for multinode training when framework not specified
    node_count = sft_config.node_count or 1
    if sft_config.framework is None and node_count > 1:
        framework_name = "megatron"
    else:
        framework_name = (sft_config.framework or "transformers").lower()

    if framework_name == "transformers":
        from truss_train.sft.frameworks.transformers_framework import (
            TransformersFramework,
        )

        return TransformersFramework()
    if framework_name == "trl":
        from truss_train.sft.frameworks.trl_framework import TRLFramework

        return TRLFramework()
    if framework_name == "megatron":
        from truss_train.sft.frameworks.megatron_framework import MegatronFramework

        return MegatronFramework()

    raise ValueError(
        f"Unknown framework: {framework_name}. "
        f"Supported: transformers, trl, megatron"
    )
