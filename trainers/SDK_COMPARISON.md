# Trainers SDK vs Tinker SDK Comparison

Comparison of the **trainers SDK** (`trainers/`) and the **tinker SDK** (`tinker` package used by tinker-cookbook).

---

## 1. Architecture: Queue vs Direct

**Trainers SDK** uses a **queue-based architecture** — a single `TrainingClient` enqueues operations to a Training Request Manager (TRM), then polls for completion:

```python
# trainers: everything goes through one client + queue
client = TrainingClient(SERVER_URL, api_key=API_KEY, timeout=30.0, poll_interval=0.5)
future = client.forward_backward(batch, loss_fn="cross_entropy")
result = future.result()  # polls TRM until completed/failed
```

**Tinker SDK** uses a **3-client direct architecture** — `ServiceClient` creates separate `TrainingClient` and `SamplingClient` that talk directly to the backend:

```python
# tinker: separate clients for different concerns
service_client = tinker.ServiceClient(base_url=...)
training_client = service_client.create_lora_training_client(base_model="...", rank=32)
sampling_client = service_client.create_sampling_client(model_path=path)
```

## 2. `optim_step` — No params vs Required `AdamParams`

This is one of the biggest API differences:

```python
# trainers: no optimizer configuration at all
future = client.optim_step()
```

```python
# tinker: explicit Adam optimizer params every call (allows LR scheduling)
adam_params = tinker.AdamParams(learning_rate=4e-5, beta1=0.9, beta2=0.95, eps=1e-8)
optim_step_future = training_client.optim_step(adam_params)
```

The trainers SDK has no way to configure learning rate, betas, or do LR scheduling. The tinker SDK passes `AdamParams` on every call, enabling per-step LR schedules (e.g. the linear decay in `sl_loop.py`).

## 3. Sampling — Chat messages vs Tokenized ModelInput

**Trainers** uses a **chat-message interface** for sampling:

```python
# trainers: chat-style SampleInput with messages
client.to_inference()  # mode switch first
client.sample([
    SampleInput(
        messages=[Message(role="user", content="What is 2+2?")],
        max_tokens=32, temperature=0.0,
    )
])
```

**Tinker** uses a **token-level interface** with a separate `SamplingClient`:

```python
# tinker: tokenized prompt + SamplingParams + separate client
sampling_path = training_client.save_weights_for_sampler(name="001", ttl_seconds=...).result().path
sampling_client = service_client.create_sampling_client(model_path=sampling_path)
future = sampling_client.sample(
    prompt=model_input,              # ModelInput (tokenized)
    num_samples=config.group_size,   # generate N completions at once
    sampling_params=tinker.SamplingParams(max_tokens=256, stop=renderer.get_stop_sequences()),
)
result = future.result()
for seq in result.sequences:         # typed: .tokens, .logprobs per sequence
    ...
```

Key differences here:
- **Trainers** requires a `to_inference()` mode switch on the same client; **Tinker** creates a separate `SamplingClient` from saved weights (no mode switch)
- **Trainers** takes `messages` (string-level); **Tinker** takes `ModelInput` (token-level), giving control over tokenization/rendering
- **Tinker** supports `num_samples` for group sampling (essential for GRPO), returning per-sequence `tokens` and `logprobs`; **Trainers** has no equivalent

## 4. Return Types — Raw dicts vs Typed models

**Trainers** `OperationFuture.result()` returns `dict | None`:
```python
result = future.result(timeout=15.0)
assert result["metrics"]["loss"]       # raw dict access
assert result["outputs"][0]["generated_text"]
```

**Tinker** `APIFuture[T].result()` returns typed objects:
```python
fwd_bwd_result = fwd_bwd_future.result()      # ForwardBackwardOutput
optim_result = optim_step_future.result()      # OptimStepResponse
sample_result = future.result()                 # SampleResponse

# typed access:
fwd_bwd_result.loss_fn_outputs[i]["logprobs"]  # TensorData
optim_result.metrics                            # dict[str, float]
sample_result.sequences[i].tokens               # list[int]
sample_result.sequences[i].logprobs             # list[float]
```

## 5. Features present in Tinker but missing in Trainers

| Feature | Tinker | Trainers |
|---------|--------|----------|
| `forward()` (no gradients) | Yes | No |
| `forward_backward_custom()` (custom PyTorch loss) | Yes | No |
| `save_weights_for_sampler()` | Yes (tinker:// paths) | No (uses `to_inference()`) |
| `load_state()` / `load_state_with_optimizer()` | Yes | No |
| Resume from checkpoint | `create_training_client_from_state_with_optimizer()` | No |
| Auto-chunking large batches | Yes (MAX_CHUNK_LEN=1024) | No |
| Retry/backpressure handling | Built-in `RetryHandler` | No |
| `compute_logprobs()` | Yes (on SamplingClient) | No |
| `get_tokenizer()` / `get_info()` | Yes | No |
| Multimodal (ImageChunk) | Supported in types | Supported in types |
| Picklable SamplingClient (multi-process) | Yes | No |
| `TensorData.from_torch()` | Yes | No (only `from_list()`) |

## Summary

The **trainers SDK** is a thin queue-based wrapper where a single client enqueues operations and polls for results. It's simpler but lower-featured — no optimizer config, chat-level sampling only, untyped results.

The **tinker SDK** is a full-featured client with separated concerns (training vs sampling), token-level control, typed responses, optimizer params per step, checkpoint resume, auto-chunking, retries, and multi-process support. The `test_training_loop` in trainers covers the same *lifecycle* (fwd_bwd -> optim -> inference -> sample) but with a much thinner API surface.
