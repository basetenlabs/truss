import os
import sys
from pathlib import Path

from truss import truss_config

from .data_types import WHISPER_MODEL, PostProcessingFlags

DEFAULT_CHUNKER_GPU_FOR_CHAIN = truss_config.Accelerator.L4
DEFAULT_TRANSCRIBER_GPU_FOR_CHAIN = truss_config.Accelerator.H100
DEFAULT_WHISPER_MODEL = WHISPER_MODEL.LARGE_V3_TURBO


_LOCAL_WHISPER_LIB = Path(__file__).resolve().parent.parent.parent / "whisper-runtime"
sys.path.append(str(_LOCAL_WHISPER_LIB))

WHISPER_SAMPLE_RATE = 16000
WHISPER_ENCODER_DOWNSAMPLING_FACTOR = 2

MAX_BATCH_SIZE = {
    truss_config.Accelerator.H100_40GB: 8 if os.getenv("ENABLE_DIARIZATION") == "true" else 18,
    truss_config.Accelerator.H100: 13 if os.getenv("ENABLE_DIARIZATION") == "true" else 32,
    truss_config.Accelerator.L4: 12,
    truss_config.Accelerator.A100: 10,
}

FREE_GPU_MEMORY_FRACTION = {
    truss_config.Accelerator.H100_40GB: 0.5,
    truss_config.Accelerator.H100: 0.2,
    truss_config.Accelerator.L4: 0.5,
    truss_config.Accelerator.A100: 0.5,
}

CROSS_KV_CACHE_FRACTION = {
    truss_config.Accelerator.H100_40GB: 0.2,
    truss_config.Accelerator.H100: 0.5,
    truss_config.Accelerator.L4: 0.2,
    truss_config.Accelerator.A100: 0.2,
}

AUDIO_DOWNLOAD_TIMEOUT = 300

# Whisper runtime constants
CHUNK_PADDING_SECONDS = 0.25

# Missing chunk retry constants
COMPRESSION_THRESHOLD = 2.4
RETRY_CHUNK_PADDING_SECONDS = 0.25

# Silero VAD config overrides
VAD_CONFIG_OVERRIDES = {
    "max_speech_duration_s": 29,
    "min_silence_duration_ms": 3000,
    "retry_max_speech_duration_s": 10,  # for advanced users
    "retry_min_silence_duration_ms": 3000,  # for advanced users
    "speech_pad_ms": 0,
}
VAD_LOOKBACK_WINDOW_SECS = 4

# Maximum allowed concurrent outgoing requests from chainlet
CONCURRENCY_LIMIT_FROM_CHUNKER_CHAINLET = 1000

AUDIO_LENGTH_BUCKETS_SECONDS = [10, 30, 60, 300, 1800, 3600, 7200, 14400]

CHUNK_LENGTH_BUCKETS_SECONDS = [0.5, 1, 5, 10, 20, 30]

PARTIAL_FINAL_LATENCY_BUCKETS_SECONDS = [
    0.5,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
    40.0,
    50.0,
    60.0,
    90.0,
    120.0,
    180.0,
]

HEALTH_CHECK_LATENCY_BUCKETS_SECONDS = [
    0.001,
    0.01,
    0.1,
    1.0,
    2.0,
    5.0,
    7.5,
    10.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
    90.0,
    120.0,
    180.0,
]

# for diarization
MINIMAL_TIME_SEGMENT_DURATION = 0.01

DEFAULT_STREAMING_POST_PROCESSING_FLAGS = PostProcessingFlags()

DEFAULT_HTTP_POST_PROCESSING_FLAGS = PostProcessingFlags(
    remove_spaces_in_ja_zh=True,
)
