"""SamplingClient for text generation from trained or base models."""

from __future__ import annotations

import httpx

from trainers.models import ModelInput, SamplingParams
from trainers.training_client import SampleResult, _load_tokenizer


class SamplingClient:
    """Client for generating text from a model.

    Created via ServiceClient.create_sampling_client() or
    TrainingClient.save_weights_and_get_sampling_client().

    Supports pickling for use with multiprocessing.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        base_model: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._base_model = base_model
        self._timeout = timeout

    def sample(
        self,
        prompt: ModelInput,
        num_samples: int = 1,
        sampling_params: SamplingParams | None = None,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ) -> SampleResult:
        """Generate text completions from a prompt."""
        body = {
            "prompt": prompt.model_dump(mode="json"),
            "num_samples": num_samples,
            "sampling_params": (sampling_params or SamplingParams()).model_dump(mode="json"),
        }
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(f"{self._base_url}/sample", json=body, headers=headers)
            resp.raise_for_status()
            return SampleResult.model_validate(resp.json())

    def compute_logprobs(self, prompt: ModelInput) -> list[float | None]:
        """Compute log-probabilities for each token in the prompt."""
        raise NotImplementedError("SamplingClient.compute_logprobs() is not yet implemented")

    def get_tokenizer(self):
        """Return the tokenizer used by the model.

        Requires base_model to be set via ServiceClient.create_sampling_client(base_model=...).
        """
        if self._base_model is None:
            raise ValueError(
                "base_model was not set. Pass base_model= to create_sampling_client()."
            )
        return _load_tokenizer(self._base_model)

    def close(self) -> None:
        pass

    def __getstate__(self):
        return {
            "base_url": self._base_url,
            "api_key": self._api_key,
            "base_model": self._base_model,
            "timeout": self._timeout,
        }

    def __setstate__(self, state):
        self._base_url = state["base_url"]
        self._api_key = state["api_key"]
        self._base_model = state.get("base_model")
        self._timeout = state.get("timeout", 300.0)
