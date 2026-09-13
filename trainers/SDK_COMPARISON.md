# Trainers SDK vs Tinker SDK Comparison

Comparison of the **trainers SDK** (`trainers/`) and the **tinker SDK** (`tinker` package v0.13.1, used by tinker-cookbook).

---

## 1. Architecture — Now Aligned

Both SDKs now use the same **3-client pattern**:

| Role | Tinker | Trainers |
|------|--------|----------|
| Factory/entry point | `tinker.ServiceClient` | `trainers.ServiceClient` |
| Training operations | `tinker.TrainingClient` | `trainers.TrainingClient` |
| Inference/generation | `tinker.SamplingClient` | `trainers.SamplingClient` |

The trainers SDK re-exports all core data types directly from `tinker.types` (`models.py`), so `Datum`, `ModelInput`, `AdamParams`, `ForwardBackwardOutput`, `OptimStepResponse`, `SamplingParams`, `TensorData`, etc. are the **same Pydantic models** in both SDKs.

**Key structural difference**: the trainers `TrainingClient` talks to the dp_worker via direct HTTP (`httpx.Client` + `ThreadPoolExecutor`), while tinker's `TrainingClient` uses an `InternalClientHolder` with an async event loop, connection pool, telemetry, and ordered request dispatch.

```python
# trainers: direct HTTP to dp_worker
client = TrainingClient("http://worker:8001", timeout=600.0)
future = client.forward_backward(data=batch)    # httpx POST, result via thread pool
result = future.result()                         # ForwardBackwardOutput
```

```python
# tinker: managed async client with connection pooling
service_client = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(base_model="Qwen/Qwen3-8B")
future = training_client.forward_backward(data, "cross_entropy")  # async dispatch
result = future.result()                                           # ForwardBackwardOutput
```

## 2. API Surface — Implemented vs Stubbed

The trainers `TrainingClient` defines the full tinker-compatible method set, but only a subset is implemented. The rest raise `NotImplementedError`:

| Method | Tinker | Trainers |
|--------|--------|----------|
| `forward_backward(data, loss_fn, loss_fn_config)` | Implemented | **Implemented** |
| `optim_step(adam_params)` | Required `AdamParams` | **Implemented** (optional, defaults to `AdamParams()`) |
| `save_weights_and_get_sampling_client(name, ttl)` | Implemented | **Implemented** (POST `/to_inference`) |
| `sample(prompt, num_samples, sampling_params)` | On `SamplingClient` | **Implemented** (on `TrainingClient` directly) |
| `save_state(name, ttl_seconds)` | Implemented | **Implemented** |
| `forward(data, loss_fn)` | Implemented | Stubbed |
| `forward_backward_custom(data, loss_fn)` | Implemented | Stubbed |
| `save_weights_for_sampler(name, ttl)` | Implemented | Stubbed |
| `load_state(path)` | Implemented | Stubbed |
| `load_state_with_optimizer(path)` | Implemented | Stubbed |

Similarly, `ServiceClient` and `SamplingClient` are partially stubbed:

| Method | Tinker | Trainers |
|--------|--------|----------|
| `ServiceClient.create_training_client()` | Implemented (`create_lora_training_client`) | **Implemented** (TODO: pass model/rank to backend) |
| `ServiceClient.create_sampling_client()` | Implemented | **Implemented** (returns stub client) |
| `ServiceClient.create_training_client_from_state()` | Implemented | Stubbed |
| `ServiceClient.create_training_client_from_state_with_optimizer()` | Implemented | Stubbed |
| `ServiceClient.get_server_capabilities()` | Implemented | Stubbed |
| `SamplingClient.sample()` | Implemented | Stubbed |
| `SamplingClient.compute_logprobs()` | Implemented | Stubbed |
| `SamplingClient.get_tokenizer()` | Implemented | Stubbed |

## 3. Sampling — Same Interface, Different Location

Both SDKs now use token-level sampling with `ModelInput` + `SamplingParams`:

```python
# trainers: sample() lives on TrainingClient directly
result = client.sample(
    prompt=ModelInput.from_ints([1, 2, 3]),
    num_samples=1,
    sampling_params=SamplingParams(max_tokens=32, temperature=0.0),
).result()
# result.sequences[0].tokens, result.sequences[0].logprobs
```

```python
# tinker: sample() lives on a separate SamplingClient
sampling_path = training_client.save_weights_for_sampler("001", ttl_seconds=604800).result().path
sampling_client = service_client.create_sampling_client(model_path=sampling_path)
result = sampling_client.sample(
    prompt=model_input,
    num_samples=16,
    sampling_params=tinker.SamplingParams(max_tokens=256, stop=renderer.get_stop_sequences()),
).result()
# result.sequences[0].tokens, result.sequences[0].logprobs
```

The interface is nearly identical. The difference is that tinker requires creating a `SamplingClient` from saved weights (decoupled from training), while trainers exposes `sample()` directly on the `TrainingClient`.

Note: trainers defines its own `SampleResult`/`SampledSequence` in `training_client.py` rather than using tinker's `SampleResponse` — the fields are the same but they're separate types.

## 4. Return Types — Both Typed

Both SDKs now return typed Pydantic models via generic futures:

```python
# trainers: OperationFuture[T]
fwd_bwd: OperationFuture[ForwardBackwardOutput] = client.forward_backward(data=batch)
optim:   OperationFuture[OptimStepResponse]      = client.optim_step(AdamParams(learning_rate=4e-5))
sample:  OperationFuture[SampleResult]            = client.sample(prompt=..., sampling_params=...)

result = fwd_bwd.result()
result.metrics["loss"]                   # typed dict access
result.loss_fn_outputs                   # list[LossFnOutput]
```

```python
# tinker: APIFuture[T] (with async support, retry, telemetry)
fwd_bwd: APIFuture[ForwardBackwardOutput] = training_client.forward_backward(data, "cross_entropy")
optim:   APIFuture[OptimStepResponse]      = training_client.optim_step(adam_params)
```

Trainers uses `concurrent.futures.Future` under the hood via a `ThreadPoolExecutor`. Tinker uses a custom `_APIFuture` backed by an asyncio event loop with ordered dispatch, combined futures for chunked requests, telemetry hooks, and queue state logging.

## 5. Pipelining

Both SDKs support dispatching multiple operations before collecting results. Trainers achieves this via thread pool submission:

```python
# trainers: dispatch 3 forward_backward ops then collect
futures = [client.forward_backward(data=batch) for _ in range(3)]
results = [f.result(timeout=5.0) for f in futures]
```

Tinker achieves this via its async request queue with ordered `_take_turn` guarantees — requests are dispatched in order even when pipelined.

## 6. Remaining Gaps

Features fully implemented in tinker but not yet in trainers:

| Feature | Status in Trainers |
|---------|--------------------|
| Auto-chunking large batches (1024 datums / 5MB) | Not implemented |
| Retry/backpressure handling | Not implemented |
| Request ordering guarantees (`_take_turn`) | Not implemented |
| Telemetry / observability | Not implemented |
| Async variants (`forward_backward_async`, etc.) | Not implemented |
| `forward()` (forward-only, no gradients) | Stubbed |
| `forward_backward_custom()` (PyTorch custom loss) | Stubbed |
| `save_weights_for_sampler()` / separate `SamplingClient` flow | Stubbed |
| `load_state()` / `load_state_with_optimizer()` | Stubbed |
| Checkpoint resume via `ServiceClient` | Stubbed |
| `get_tokenizer()` / `get_info()` on TrainingClient | Not present |
| `compute_logprobs()` on SamplingClient | Stubbed |
| Picklable SamplingClient (multi-process) | Stubbed (`__getstate__` exists but methods don't work) |

## Summary

The trainers SDK has converged significantly toward the tinker SDK's API design. It now shares the same 3-client pattern (`ServiceClient` / `TrainingClient` / `SamplingClient`), the same data types (re-exported from `tinker.types`), the same method signatures (`forward_backward`, `optim_step` with `AdamParams`, token-level `sample`), and typed return values via generic futures.

The core training loop (`forward_backward` -> `optim_step` -> `save_weights_and_get_sampling_client` -> `sample` -> `save_state`) is fully functional with direct HTTP calls to the dp_worker. The remaining work is implementing the stubbed methods (checkpoint load/resume, forward-only, custom loss, separate `SamplingClient` flow) and adding the infrastructure features (auto-chunking, retries, request ordering, telemetry, async) that tinker's production client provides.
