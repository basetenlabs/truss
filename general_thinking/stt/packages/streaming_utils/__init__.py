# streaming_utils package
# Shared utilities for both chain and truss streaming ASR implementations
#
# Structure:
#   streaming_utils/
#   ├── whisper_service_base.py     - Abstract base class for WhisperService
#   ├── websocket_adapter.py        - FastAPIWebSocketAdapter (bridges FastAPI ↔ WebSocketProtocol)
#   ├── utilities.py                - Streaming metrics (Prometheus)
#   ├── services/
#   │   └── vad_service.py          - Voice Activity Detection service
#   ├── stream_processing/
#   │   ├── stream_processor.py     - Main WebSocket stream orchestrator
#   │   ├── audio_buffer.py         - Audio buffer management
#   │   ├── stream_state.py         - Stream state tracking
#   │   └── workers/                - Async worker classes
#   │       ├── base_worker.py
#   │       ├── stream_transcription_worker.py
#   │       ├── stream_diarization_worker.py
#   │       └── stream_assignment_worker.py
#   └── utils/
#       ├── audio_utils.py          - Audio conversion utilities
#       ├── error_utils.py          - Error types and logging
#       ├── message_types.py        - WebSocket message types
#       ├── websocket_utils.py      - WebSocket manager
#       ├── constants.py            - Shared constants
#       └── compile_cache.py        - Torch compile cache utilities
