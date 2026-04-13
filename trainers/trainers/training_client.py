"""TrainingClient — direct HTTP client for a dp_worker backend with futures for pipelining."""

from __future__ import annotations

import concurrent.futures
from types import TracebackType
from typing import Callable, Generic, TypeVar

import httpx

from trainers.models import (
    AdamParams,
    Datum,
    ForwardBackwardOutput,
    LoadWeightsResponse,
    ModelInput,
    OptimStepResponse,
    SampledSequence,
    SampleResponse,
    SamplingParams,
    SaveWeightsResponse,
)

T = TypeVar("T")


class OperationFuture(Generic[T]):
    """Wraps a concurrent.futures.Future with typed result access."""

    def __init__(self, future: concurrent.futures.Future[T]) -> None:
        self._future = future

    def result(self, timeout: float | None = None) -> T:
        return self._future.result(timeout=timeout)

    def done(self) -> bool:
        return self._future.done()

    def __await__(self):
        import asyncio
        return asyncio.to_thread(self.result).__await__()


class TrainingClient:
    """Client that talks directly to a dp_worker instance.

    All training operations return OperationFuture[T] for pipelining —
    multiple operations can be dispatched before waiting for results.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 600.0,
        max_workers: int = 4,
    ) -> None:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Api-Key {api_key}"
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(headers=headers, timeout=timeout)
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def __enter__(self) -> TrainingClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._pool.shutdown(wait=False)
        self._client.close()

    def _post(self, path: str, json: dict | None = None) -> httpx.Response:
        resp = self._client.post(f"{self._base_url}{path}", json=json)
        resp.raise_for_status()
        return resp

    def _submit(self, fn: Callable[[], T]) -> OperationFuture[T]:
        return OperationFuture(self._pool.submit(fn))

    # -- Training operations (implemented) --

    def forward_backward(
        self,
        batch: list[Datum],
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict | None = None,
    ) -> OperationFuture[ForwardBackwardOutput]:
        body = {
            "batch": [d.model_dump(mode="json") for d in batch],
            "loss_fn": loss_fn,
            "loss_fn_config": loss_fn_config,
        }

        def _call() -> ForwardBackwardOutput:
            resp = self._post("/forward_backward", json=body)
            return ForwardBackwardOutput.model_validate(resp.json())

        return self._submit(_call)

    def optim_step(self, adam_params: AdamParams | None = None) -> OperationFuture[OptimStepResponse]:
        body = adam_params.model_dump(mode="json") if adam_params else {}

        def _call() -> OptimStepResponse:
            resp = self._post("/optim_step", json=body)
            return OptimStepResponse.model_validate(resp.json())

        return self._submit(_call)

    def to_inference(self) -> OperationFuture[SaveWeightsResponse]:
        def _call() -> SaveWeightsResponse:
            resp = self._post("/to_inference")
            return SaveWeightsResponse.model_validate(resp.json())

        return self._submit(_call)

    def sample(
        self,
        prompt: ModelInput,
        num_samples: int = 1,
        sampling_params: SamplingParams | None = None,
    ) -> OperationFuture[SampleResponse]:
        body = {
            "prompt": prompt.model_dump(mode="json"),
            "num_samples": num_samples,
            "sampling_params": (sampling_params or SamplingParams()).model_dump(mode="json"),
        }

        def _call() -> SampleResponse:
            resp = self._post("/sample", json=body)
            return SampleResponse.model_validate(resp.json())

        return self._submit(_call)

    def save_state(self, checkpoint_dir: str) -> OperationFuture[SaveWeightsResponse]:
        body = {"checkpoint_dir": checkpoint_dir}

        def _call() -> SaveWeightsResponse:
            resp = self._post("/save_state", json=body)
            return SaveWeightsResponse.model_validate(resp.json())

        return self._submit(_call)

    def health(self) -> None:
        resp = self._client.get(f"{self._base_url}/health")
        resp.raise_for_status()

    # -- Stubbed: not yet implemented --

    def forward(
        self,
        batch: list[Datum],
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict | None = None,
    ) -> OperationFuture[ForwardBackwardOutput]:
        """Forward pass without gradient computation."""
        raise NotImplementedError("forward() is not yet implemented")

    def forward_backward_custom(
        self,
        batch: list[Datum],
        loss_fn: str,
    ) -> OperationFuture[ForwardBackwardOutput]:
        """Forward-backward with a custom PyTorch loss function."""
        raise NotImplementedError("forward_backward_custom() is not yet implemented")

    def save_weights_for_sampler(
        self,
        name: str,
        ttl_seconds: int | None = None,
    ) -> OperationFuture[SaveWeightsResponse]:
        """Save current weights for use by a SamplingClient."""
        raise NotImplementedError("save_weights_for_sampler() is not yet implemented")

    def load_state(self, path: str) -> OperationFuture[LoadWeightsResponse]:
        """Load model weights from a checkpoint path."""
        raise NotImplementedError("load_state() is not yet implemented")

    def load_state_with_optimizer(self, path: str) -> OperationFuture[LoadWeightsResponse]:
        """Load model weights and optimizer state from a checkpoint path."""
        raise NotImplementedError("load_state_with_optimizer() is not yet implemented")
