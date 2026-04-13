"""ServiceClient — factory for creating training and sampling clients."""

from __future__ import annotations

from trainers.sampling_client import SamplingClient
from trainers.training_client import TrainingClient


class ServiceClient:
    """Factory for creating TrainingClient and SamplingClient instances.

    Analogous to tinker.ServiceClient — creates clients connected to a
    training backend.
    """

    def __init__(self, base_url: str, *, api_key: str | None = None) -> None:
        self._base_url = base_url
        self._api_key = api_key

    def create_training_client(
        self,
        base_model: str,
        *,
        rank: int = 16,
        seed: int | None = None,
        poll_interval: float = 0.5,
        timeout: float | None = None,
    ) -> TrainingClient:
        """Create a LoRA training client."""
        # TODO: pass base_model, rank, seed to the backend to initialize
        return TrainingClient(
            self._base_url,
            api_key=self._api_key,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    def create_training_client_from_state(
        self,
        path: str,
    ) -> TrainingClient:
        """Resume training from a saved checkpoint (weights only)."""
        raise NotImplementedError(
            "create_training_client_from_state() is not yet implemented"
        )

    def create_training_client_from_state_with_optimizer(
        self,
        path: str,
    ) -> TrainingClient:
        """Resume training from a saved checkpoint (weights + optimizer state)."""
        raise NotImplementedError(
            "create_training_client_from_state_with_optimizer() is not yet implemented"
        )

    def create_sampling_client(
        self,
        base_model: str | None = None,
        model_path: str | None = None,
    ) -> SamplingClient:
        """Create a sampling client for text generation."""
        return SamplingClient(self._base_url, api_key=self._api_key)

    def get_server_capabilities(self) -> dict:
        """Query the backend for supported features."""
        raise NotImplementedError(
            "get_server_capabilities() is not yet implemented"
        )
