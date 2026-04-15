"""ServiceClient — factory for creating training and sampling clients."""

from __future__ import annotations

import os

from trainers.sampling_client import SamplingClient
from trainers.training_client import TrainingClient


class ServiceClient:
    """Factory for creating TrainingClient and SamplingClient instances.

    Usage:
        service_client = trainers.ServiceClient()
        training_client = service_client.create_lora_training_client(
            base_model="Qwen/Qwen3-8B", rank=32,
        )
        sampling_client = service_client.create_sampling_client()

    Configuration is read from environment variables:
        TRAINERS_BASE_URL — worker base URL (required)
        TRAINERS_API_KEY  — API key for authentication (optional)

    Or pass base_url/api_key directly to __init__.
    """

    def __init__(
        self, base_url: str | None = None, *, api_key: str | None = None
    ) -> None:
        self._base_url = base_url or os.environ.get("TRAINERS_BASE_URL", "")
        self._api_key = api_key or os.environ.get("TRAINERS_API_KEY")
        if not self._base_url:
            raise ValueError(
                "base_url is required. Pass it directly or set TRAINERS_BASE_URL."
            )

    def create_lora_training_client(
        self,
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        timeout: float = 600.0,
    ) -> TrainingClient:
        """Create a LoRA training client.

        Args:
            base_model: HuggingFace model ID (e.g. "Qwen/Qwen3-8B").
            rank: LoRA rank.
            seed: Random seed for reproducibility.
            timeout: HTTP timeout for training operations.
        """
        # TODO: send base_model, rank, seed to the worker to initialize LoRA
        return TrainingClient(
            self._base_url,
            api_key=self._api_key,
            base_model=base_model,
            timeout=timeout,
        )

    def create_training_client_from_state(
        self, path: str, *, base_url: str | None = None, timeout: float = 600.0
    ) -> TrainingClient:
        """Return a TrainingClient for a trainer pod started from a checkpoint.

        The checkpoint is loaded at pod startup via BT_LOAD_CHECKPOINT_DIR (set
        by the plumbing layer). The path argument is accepted for API compatibility
        but is not used by the SDK — the pod has already loaded the checkpoint
        before this method is called.

        Args:
            path: Checkpoint path (used by the plumbing layer to configure the pod).
            base_url: URL of the running trainer pod. Defaults to self._base_url.
            timeout: HTTP timeout for training operations.
        """
        url = base_url or self._base_url
        return TrainingClient(url, api_key=self._api_key, timeout=timeout)

    def create_training_client_from_state_with_optimizer(
        self, path: str, *, base_url: str | None = None, timeout: float = 600.0
    ) -> TrainingClient:
        """Return a TrainingClient for a trainer pod started from a checkpoint.

        Like create_training_client_from_state but the pod also restores optimizer
        state (trainer_state.pt), so training resumes from the exact same step.

        Args:
            path: Checkpoint path (used by the plumbing layer to configure the pod).
            base_url: URL of the running trainer pod. Defaults to self._base_url.
            timeout: HTTP timeout for training operations.
        """
        url = base_url or self._base_url
        return TrainingClient(url, api_key=self._api_key, timeout=timeout)

    def create_sampling_client(
        self, base_model: str | None = None, model_path: str | None = None
    ) -> SamplingClient:
        """Create a sampling client for text generation.

        Args:
            base_model: HuggingFace model ID for base model sampling.
            model_path: Path to saved weights for fine-tuned sampling.
        """
        return SamplingClient(self._base_url, api_key=self._api_key)

    def get_server_capabilities(self) -> dict:
        """Query the backend for supported features."""
        raise NotImplementedError("get_server_capabilities() is not yet implemented")
