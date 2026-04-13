"""High-level producer clients for the training queue system."""

from __future__ import annotations

import asyncio
import threading
import uuid
from datetime import datetime, timezone
from types import TracebackType

from trainers.queue_client import AsyncQueueClient, QueueClient
from trainers.models import (
    Datum,
    ForwardBackwardDetails,
    ForwardBackwardOp,
    Operation,
    OperationStatus,
    OptimStepDetails,
    OptimStepOp,
    SampleDetails,
    SampleInput,
    SampleOp,
    SaveStateDetails,
    SaveStateOp,
    SaveWeightsAndGetSamplingClientOp,
)


class OperationFailedError(Exception):
    """Raised when a tracked operation reaches the ``"failed"`` state."""

    def __init__(self, operation_id: str, result: dict | None) -> None:
        self.operation_id = operation_id
        self.result = result
        super().__init__(f"Operation {operation_id} failed: {result}")


class OperationFuture:
    """Handle returned by producer clients to track an enqueued operation."""

    def __init__(self, operation_id: str, poller: _Poller, default_timeout: float | None = None) -> None:
        self._operation_id = operation_id
        self._poller = poller
        self._default_timeout = default_timeout
        self._status: OperationStatus | None = None
        self._polled_event = threading.Event()
        self._done_event = threading.Event()
        self._registered = False

    @property
    def operation_id(self) -> str:
        return self._operation_id

    def _ensure_registered(self) -> None:
        if not self._registered:
            self._registered = True
            self._poller.register(self)

    def is_done(self) -> bool:
        self._ensure_registered()
        self._polled_event.wait()
        return self._done_event.is_set()

    def status(self) -> OperationStatus | None:
        return self._status

    def result(self, timeout: float | None = None) -> dict | None:
        self._ensure_registered()
        effective_timeout = timeout if timeout is not None else self._default_timeout
        if not self._done_event.wait(timeout=effective_timeout):
            raise TimeoutError(f"Operation {self._operation_id} did not complete within {effective_timeout}s")
        if self._status is not None and self._status.status == "failed":
            raise OperationFailedError(self._operation_id, self._status.result)
        return self._status.result if self._status is not None else None

    def __await__(self):
        return asyncio.to_thread(self.result).__await__()


class _Poller:
    """Background daemon thread that batches ``get_op_statuses`` calls."""

    def __init__(self, queue_client: QueueClient, poll_interval: float = 0.5) -> None:
        self._queue_client = queue_client
        self._poll_interval = poll_interval
        self._futures: dict[str, OperationFuture] = {}
        self._lock = threading.Lock()
        self._has_work = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def register(self, future: OperationFuture) -> None:
        with self._lock:
            self._futures[future.operation_id] = future
            self._has_work.set()
        self._ensure_started()

    def _ensure_started(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._has_work.wait(timeout=self._poll_interval)
            if self._stop_event.is_set():
                break

            with self._lock:
                tracked = dict(self._futures)

            if not tracked:
                self._has_work.clear()
                continue

            op_ids = list(tracked.keys())
            try:
                statuses = self._queue_client.get_op_statuses(op_ids)
            except Exception:
                self._stop_event.wait(timeout=self._poll_interval)
                continue

            status_map = {s.operation_id: s for s in statuses}
            done_ids: list[str] = []
            for op_id, future in tracked.items():
                st = status_map.get(op_id)
                if st is None:
                    continue
                future._status = st
                future._polled_event.set()
                if st.status in ("completed", "failed"):
                    future._done_event.set()
                    done_ids.append(op_id)

            if done_ids:
                with self._lock:
                    for op_id in done_ids:
                        self._futures.pop(op_id, None)
                    if not self._futures:
                        self._has_work.clear()

            self._stop_event.wait(timeout=self._poll_interval)

    def stop(self) -> None:
        self._stop_event.set()
        self._has_work.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)


def _make_forward_backward_op(
    batch: list[Datum],
    loss_fn: str,
    loss_fn_config: dict | None,
) -> ForwardBackwardOp:
    return ForwardBackwardOp(
        operation_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        forward_backward_details=ForwardBackwardDetails(
            batch=batch,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
        ),
    )


def _make_optim_step_op() -> OptimStepOp:
    return OptimStepOp(
        operation_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        optim_step_details=OptimStepDetails(),
    )


def _make_to_inference_op() -> SaveWeightsAndGetSamplingClientOp:
    return SaveWeightsAndGetSamplingClientOp(
        operation_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
    )


def _make_sample_op(inputs: list[SampleInput]) -> SampleOp:
    return SampleOp(
        operation_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        sample_details=SampleDetails(inputs=inputs),
    )


def _make_save_state_op(checkpoint_dir: str) -> SaveStateOp:
    return SaveStateOp(
        operation_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        save_state_details=SaveStateDetails(checkpoint_dir=checkpoint_dir),
    )


class TrainingClient:
    """Synchronous high-level client for enqueuing training operations."""

    def __init__(
        self,
        base_url: str,
        *,
        client: QueueClient | None = None,
        poll_interval: float = 0.5,
        timeout: float | None = None,
    ) -> None:
        self._base_url = base_url
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = QueueClient(base_url)
            self._owns_client = True
        self._timeout = timeout
        self._poller = _Poller(self._client, poll_interval=poll_interval)

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
        self._poller.stop()
        if self._owns_client:
            self._client.close()

    def forward_backward(
        self,
        batch: list[Datum],
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict | None = None,
    ) -> OperationFuture:
        op = _make_forward_backward_op(batch, loss_fn, loss_fn_config)
        self._client.enqueue_ops([op])
        return OperationFuture(op.operation_id, self._poller, default_timeout=self._timeout)

    def optim_step(self) -> OperationFuture:
        op = _make_optim_step_op()
        self._client.enqueue_ops([op])
        return OperationFuture(op.operation_id, self._poller, default_timeout=self._timeout)

    def to_inference(self) -> OperationFuture:
        op = _make_to_inference_op()
        self._client.enqueue_ops([op])
        return OperationFuture(op.operation_id, self._poller, default_timeout=self._timeout)

    def sample(self, inputs: list[SampleInput]) -> OperationFuture:
        op = _make_sample_op(inputs)
        self._client.enqueue_ops([op])
        return OperationFuture(op.operation_id, self._poller, default_timeout=self._timeout)

    def save_state(self, checkpoint_dir: str) -> OperationFuture:
        op = _make_save_state_op(checkpoint_dir)
        self._client.enqueue_ops([op])
        return OperationFuture(op.operation_id, self._poller, default_timeout=self._timeout)


class AsyncTrainingClient:
    """Asynchronous high-level client for enqueuing training operations."""

    def __init__(
        self,
        base_url: str,
        *,
        client: AsyncQueueClient | None = None,
        poll_interval: float = 0.5,
        timeout: float | None = None,
    ) -> None:
        self._base_url = base_url
        if client is not None:
            self._async_client = client
            self._owns_async_client = False
        else:
            self._async_client = AsyncQueueClient(base_url)
            self._owns_async_client = True
        self._timeout = timeout
        self._sync_client = QueueClient(base_url)
        self._poller = _Poller(self._sync_client, poll_interval=poll_interval)

    async def __aenter__(self) -> AsyncTrainingClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        self._poller.stop()
        self._sync_client.close()
        if self._owns_async_client:
            await self._async_client.close()

    async def forward_backward(
        self,
        batch: list[Datum],
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict | None = None,
    ) -> OperationFuture:
        op = _make_forward_backward_op(batch, loss_fn, loss_fn_config)
        await self._async_client.enqueue_ops([op])
        return OperationFuture(op.operation_id, self._poller, default_timeout=self._timeout)

    async def optim_step(self) -> OperationFuture:
        op = _make_optim_step_op()
        await self._async_client.enqueue_ops([op])
        return OperationFuture(op.operation_id, self._poller, default_timeout=self._timeout)

    async def to_inference(self) -> OperationFuture:
        op = _make_to_inference_op()
        await self._async_client.enqueue_ops([op])
        return OperationFuture(op.operation_id, self._poller, default_timeout=self._timeout)

    async def sample(self, inputs: list[SampleInput]) -> OperationFuture:
        op = _make_sample_op(inputs)
        await self._async_client.enqueue_ops([op])
        return OperationFuture(op.operation_id, self._poller, default_timeout=self._timeout)

    async def save_state(self, checkpoint_dir: str) -> OperationFuture:
        op = _make_save_state_op(checkpoint_dir)
        await self._async_client.enqueue_ops([op])
        return OperationFuture(op.operation_id, self._poller, default_timeout=self._timeout)
