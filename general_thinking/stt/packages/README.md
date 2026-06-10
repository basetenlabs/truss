# ASR Chains

Central location for all ASR chains and shared components. This structure eliminates code duplication by colocating chains with shared chainlets and utilities.

## Structure

```
asr-chains/
├── chainlets/                     # Shared worker chainlets for ASR processing
│   ├── whisper_chainlet.py            # Transcriber chainlet using Whisper model
│   └── diarizer_chainlet.py           # Speaker diarization chainlet
├── whisper_chain_utils/           # Shared data types and utilities for chains
│   ├── whisper_chain_data_types.py    # Input/output data types for chainlets
│   └── diarizer_async_batcher.py      # Async batch processor for diarization
├── diart/                         # Speaker diarization library
├── streaming-asr-chain/           # Real-time streaming ASR chain
│   ├── transcribe_entry.py            # WebSocket entry for transcription only
│   ├── transcribe_diarize_entry.py    # WebSocket entry with diarization
│   └── packages/                      # Chain-specific code only
│       └── services/
│           └── whisper_service_chain.py   # Chain WhisperService (run_remote)
└── whisper-chain/                 # Batch ASR chain
    └── chunk_and_transcribe.py        # VAD chunking + transcription entry
```

## Code Sharing

All shared streaming utilities now live in `streaming-utils/streaming_utils/`:
- Services (VADService), stream processing (StreamProcessor, AudioBuffer, workers), and utilities (error handling, audio utils, etc.)
- **`WhisperServiceBase`** - Abstract base class for WhisperService with shared transcription logic
- **`FastAPIWebSocketAdapter`** - Bridges FastAPI WebSocket to `WebSocketProtocol` interface for truss

Only chain-specific code remains in `packages/`:
- `whisper_service_chain.py` - Chain WhisperService implementation that calls `run_remote()` to a separate GPU chainlet

### WhisperService Implementations

| Implementation | File | Class | Inference |
|---------------|------|-------|-----------|
| Chain | `packages/services/whisper_service_chain.py` | `WhisperServiceChain` | `run_remote()` to GPU chainlet |
| Truss | `asr-truss/.../whisper_service_truss.py` | `WhisperServiceTruss` | Local `call_whisper_model()` |

Both inherit from `WhisperServiceBase` (in `streaming-utils/`) and share all audio preprocessing, parameter handling, timestamp adjustment, and error handling logic.

## How It Works

All chains are colocated in the `asr-chains/` directory alongside the shared components. This eliminates the need for symlinks - each chain can directly import shared modules since they're in the same package.

## Usage

Entry point files import chainlets using the standard module path:

```python
import chainlets.whisper_chainlet

# Use in chains.depends()
whisper: chainlets.whisper_chainlet.Transcriber = chains.depends(
    chainlets.whisper_chainlet.Transcriber,
    ...
)
```

The path setup in each entry file ensures that:
1. The `asr-chains/` directory is added to `sys.path`
2. The `streaming-utils/` directory is added to `sys.path`
3. Python can find `chainlets`, `whisper_chain_utils`, `diart`, and `streaming_utils` modules
4. Chain-specific `packages` are importable relative to the entry file

## Chains

- **streaming-asr-chain**: Real-time streaming ASR with optional speaker diarization via WebSocket
- **whisper-chain**: Batch ASR processing with VAD chunking for long audio files

## Adding New Chainlets

1. Create the new chainlet in `asr-chains/chainlets/`
2. Export it from `chainlets/__init__.py` if needed
3. Import and use in any chain's entry point
