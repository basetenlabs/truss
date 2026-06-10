# Streaming ASR Truss - Whisper Large V3

This is the truss implementation of the streaming ASR chain, providing real-time speech-to-text transcription with optional speaker diarization.

## Code Sharing Architecture

This truss **reuses all shared code from the chain implementation** via `external_package_dirs` in `config.yaml`. Only minimal, truss-specific overrides are defined locally.

### What's Shared

| Package | Location | Description |
|---------|----------|-------------|
| `streaming_utils/` | `streaming-utils/streaming_utils/` | All shared streaming code: services, stream processing, utils, adapters |
| `whisper-utils/` | `whisper-utils/` | Common data types (`WhisperParams`, `WhisperResult`, etc.) |
| `whisper-runtime/` | `whisper-runtime/` | WhisperRT TensorRT-LLM runtime |
| `silero-vad/` | `silero-vad/src/` | Voice Activity Detection model |

### What's Overridden (in `model/truss_overrides/`)

Only **~105 lines** of truss-specific code:

- **`services/whisper_service_truss.py`** - Inherits from `WhisperServiceBase`, overrides `_do_inference()` to use local `call_whisper_model()` instead of `run_remote()`

### How It Works

The truss uses a **WebSocket adapter pattern** instead of monkey-patching:

1. `FastAPIWebSocketAdapter` (in `streaming-utils`) wraps FastAPI's WebSocket to match the `WebSocketProtocol` interface expected by `StreamProcessor`
2. `WhisperServiceTruss` registers itself on `WhisperServiceBase` via the singleton pattern, so shared workers retrieve it via `WhisperServiceBase.get_instance()`
3. `StreamProcessor` and all workers are used directly from `streaming_utils/` — no subclassing needed

```python
# In model.py websocket handler:
from streaming_utils.websocket_adapter import FastAPIWebSocketAdapter
from streaming_utils.stream_processing.stream_processor import StreamProcessor

ws_adapter = FastAPIWebSocketAdapter(ws)  # Bridge FastAPI WebSocket → WebSocketProtocol
stream_processor = StreamProcessor(ws_adapter, stream_id, metadata, ...)
await stream_processor.start()
```

## Key Differences from Chain Implementation

| Aspect | Chain Implementation | Truss Implementation |
|--------|---------------------|---------------------|
| Architecture | WebSocket entry (CPU) + Whisper chainlet (GPU) | Single GPU pod handles everything |
| Whisper calls | `run_remote()` to separate chainlet | Local `call_whisper_model()` |
| Network latency | Yes (between pods) | No (all local) |
| Deployment | Two services to manage | Single service |
| WebSocket handling | `truss_chains.WebSocketProtocol` | FastAPI WebSocket via `FastAPIWebSocketAdapter` |
| Code location | `asr-chains/streaming-asr-chain/` | Reuses chain code + minimal overrides |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│             Streaming ASR Truss [Single GPU Pod]            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  FastAPI WebSocket                                          │
│       │                                                     │
│       ▼                                                     │
│  FastAPIWebSocketAdapter (streaming-utils)                  │
│       │                                                     │
│       ▼                                                     │
│  StreamProcessor (streaming_utils)                          │
│       │                                                     │
│       ├── WhisperServiceTruss (truss_overrides)             │
│       │      └── Local WhisperRT model                      │
│       │      └── call_whisper_model()                        │
│       │                                                     │
│       ├── VADService (streaming_utils)                      │
│       └── AudioBuffer (streaming_utils)                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
asr-truss/streaming-asr-truss/
├── config.yaml                 # Uses external_package_dirs to share chain code
├── benchmark_asr.py            # Benchmark script
├── model/
│   ├── model.py               # Main entry point, loads models, handles WebSocket
│   └── truss_overrides/       # Only truss-specific overrides
│       └── services/
│           └── whisper_service_truss.py  # Local model calls (inherits WhisperServiceBase)
```

## The Key Override: WhisperServiceTruss

**Chain version** (`packages/services/whisper_service_chain.py`):
```python
class WhisperServiceChain(WhisperServiceBase):
    async def _do_inference(self, ...):
        # Remote call to separate GPU pod
        whisper_result = await self.whisper_chainlet.run_remote(whisper_chainlet_input)
        return whisper_result
```

**Truss version** (`truss_overrides/services/whisper_service_truss.py`):
```python
class WhisperServiceTruss(WhisperServiceBase):
    async def _do_inference(self, ...):
        # Local call on same GPU pod
        whisper_result = await call_whisper_model(
            whisper_model=self.whisper_model,
            thread_pool_executor=self.executor,
            ...
        )
        return whisper_result
```

All other logic (audio preprocessing, timestamp adjustment, error handling) is inherited from `WhisperServiceBase` in `streaming-utils/`.

## Dependencies (via external_package_dirs)

From `config.yaml`:
- `whisper-utils/` - Common data types and utilities
- `whisper-runtime/` - WhisperRT TensorRT-LLM runtime
- `silero-vad/` - Voice Activity Detection
- `streaming-utils/` - All shared streaming code (services, stream processing, utils, adapters)
- `asr-chains/` - For `whisper_chain_utils` and `diart`

## Deployment

```bash
# From the baseten-whisper root
truss push asr-truss/streaming-asr-truss/
```

## Usage

Connect via WebSocket and send:
1. Metadata JSON (first message)
2. Audio chunks (binary data)
3. `{"type": "end_audio"}` to finish

Receive:
- Partial transcripts (real-time)
- Final transcripts (after VAD detects speech end)

## Notes

- Diarization is currently **not supported** in this truss version (requires separate chainlet)
- To add diarization support, you would need to either:
  1. Load the diarization model locally (increases GPU memory)
  2. Keep using a separate diarization chainlet (hybrid approach)
