"""SamplingClient for text generation from trained or base models."""

from __future__ import annotations

from trainers.models import ModelInput, SampleResponse, SamplingParams


class SamplingClient:
    """Client for generating text from a model.

    Created via ServiceClient.create_sampling_client() or
    TrainingClient.save_weights_and_get_sampling_client().

    Supports pickling for use with multiprocessing.
    """

    def __init__(self, base_url: str, *, api_key: str | None = None) -> None:
        self._base_url = base_url
        self._api_key = api_key

    def sample(
        self,
        prompt: ModelInput,
        num_samples: int = 1,
        sampling_params: SamplingParams | None = None,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ) -> SampleResponse:
        """Generate text completions from a prompt."""
        raise NotImplementedError("SamplingClient.sample() is not yet implemented")

    def compute_logprobs(self, prompt: ModelInput) -> list[float | None]:
        """Compute log-probabilities for each token in the prompt."""
        raise NotImplementedError("SamplingClient.compute_logprobs() is not yet implemented")

    def get_tokenizer(self):
        """Return the tokenizer used by the model."""
        raise NotImplementedError("SamplingClient.get_tokenizer() is not yet implemented")

    def close(self) -> None:
        pass

    def __getstate__(self):
        return {"base_url": self._base_url, "api_key": self._api_key}

    def __setstate__(self, state):
        self._base_url = state["base_url"]
        self._api_key = state["api_key"]
