"""Integration tests for the trainers SDK.

Exercises the full training loop lifecycle:
  1. Forward-backward (gradient accumulation)
  2. Optimizer step
  3. Switch to inference
  4. Sample generations

A background thread simulates the worker side by consuming ops and
returning mock results.
"""

import os
import threading
import time
import uuid

import pytest

from trainers import TrainingClient
from trainers.models import (
    Datum,
    Message,
    ModelInput,
    OperationStatus,
    SampleInput,
    TensorData,
)


SERVER_URL = os.environ.get("TRAINERS_URL", "https://trm.demo.api.baseten.co")
API_KEY = os.environ.get("TRAINERS_API_KEY", "")


def _make_client(**kwargs) -> TrainingClient:
    # Use a unique client_id per test run to avoid racing with the live TRE.
    if "client_id" not in kwargs:
        kwargs["client_id"] = f"test-{uuid.uuid4().hex[:8]}"
    return TrainingClient(SERVER_URL, api_key=API_KEY, **kwargs)


def _mock_worker(client: TrainingClient, op_count: int, results: dict):
    """Simulate a worker: pop ops and mark them completed with mock results."""
    completed = 0
    while completed < op_count:
        op = client._client.pop_op()
        if op is None:
            time.sleep(0.2)
            continue
        result = results.get(op.type, {"status": "ok"})
        client._client.update_op_status(
            OperationStatus(
                operation_id=op.operation_id,
                status="completed",
                result=result,
            )
        )
        completed += 1


# --- Tests ---


@pytest.mark.integration
def test_health():
    client = _make_client()
    client._client.health()
    client.close()


@pytest.mark.integration
def test_enqueue_and_status_lifecycle():
    """Enqueue an op, verify pending → in_progress → completed."""
    from trainers.models import ForwardBackwardOp, ForwardBackwardDetails
    from datetime import datetime, timezone

    client = _make_client()

    op = ForwardBackwardOp(
        operation_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        forward_backward_details=ForwardBackwardDetails(
            batch=[Datum(model_input=ModelInput.from_ints([1, 2, 3]))],
            loss_fn="cross_entropy",
        ),
    )

    # Enqueue → pending.
    op_ids = client._client.enqueue_ops([op])
    assert op_ids == [op.operation_id]
    statuses = client._client.get_op_statuses([op.operation_id])
    assert statuses[0].status == "pending"

    # Pop → in_progress.
    popped = client._client.pop_op()
    assert popped is not None
    statuses = client._client.get_op_statuses([op.operation_id])
    assert statuses[0].status == "in_progress"

    # Complete.
    client._client.update_op_status(
        OperationStatus(
            operation_id=op.operation_id,
            status="completed",
            result={"metrics": {"loss": 0.42}},
        )
    )
    statuses = client._client.get_op_statuses([op.operation_id])
    assert statuses[0].status == "completed"

    client.close()


@pytest.mark.integration
def test_training_loop():
    """Full training loop: forward_backward x2 → optim_step → to_inference → sample."""
    client = _make_client(timeout=30.0, poll_interval=0.5)

    mock_results = {
        "forward_backward": {"metrics": {"loss": 0.5}},
        "optim_step": {"metrics": {"step": 1.0, "lr": 5e-6}},
        "save_weights_and_get_sampling_client": {"mode": "inference"},
        "sample": {
            "outputs": [
                {"generated_text": "four", "messages": [{"role": "assistant", "content": "four"}]},
            ]
        },
    }

    # 2 forward_backward + 1 optim_step + 1 to_inference + 1 sample = 5 ops
    worker = threading.Thread(
        target=_mock_worker,
        args=(client, 5, mock_results),
        daemon=True,
    )
    worker.start()

    # -- Gradient accumulation --
    batch = [
        Datum(
            model_input=ModelInput.from_ints(list(range(10))),
            loss_fn_inputs={"reward": TensorData.from_list([1.0])},
        ),
        Datum(
            model_input=ModelInput.from_ints(list(range(10, 20))),
            loss_fn_inputs={"reward": TensorData.from_list([0.5])},
        ),
    ]

    for step in range(2):
        future = client.forward_backward(batch=batch, loss_fn="cross_entropy")
        result = future.result(timeout=15.0)
        assert result is not None
        assert "loss" in result["metrics"]

    # -- Optimizer step --
    future = client.optim_step()
    result = future.result(timeout=15.0)
    assert result is not None
    assert "metrics" in result

    # -- Switch to inference --
    future = client.to_inference()
    result = future.result(timeout=15.0)
    assert result is not None
    assert result["mode"] == "inference"

    # -- Sample --
    future = client.sample([
        SampleInput(
            messages=[Message(role="user", content="What is 2+2?")],
            max_tokens=32,
            temperature=0.0,
        ),
    ])
    result = future.result(timeout=15.0)
    assert result is not None
    assert len(result["outputs"]) == 1

    worker.join(timeout=5.0)
    client.close()
