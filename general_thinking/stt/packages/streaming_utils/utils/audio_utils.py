import logging
import time
from typing import Optional

import numpy as np
import torch
import torchaudio
from streaming_utils.utils.constants import WHISPER_SAMPLE_RATE

logger = logging.getLogger(__name__)


def bytes_to_float_tensor(chunk: bytes, encoding: str, stream_id: str = "unknown") -> torch.Tensor:
    """
    Converts audio bytes to a 16kHz mono float32 torch.Tensor with enhanced logging.

    Args:
        chunk: Audio bytes
        encoding: Audio encoding ('pcm_s16le' or 'pcm_mulaw')
        stream_id: Stream identifier for logging

    Returns:
        torch.Tensor: 16kHz mono float32 audio tensor

    Raises:
        ValueError: If encoding is not supported
    """
    start_time = time.time()

    try:
        if encoding not in ("pcm_s16le", "pcm_mulaw"):
            raise ValueError(
                f"Unsupported encoding: {encoding}. Expected 'pcm_s16le' or 'pcm_mulaw'"
            )

        if encoding == "pcm_s16le":
            # Default assumption: 16000 Hz, mono
            input_sample_rate = 16000
            int16_array = np.frombuffer(chunk, dtype=np.int16)
            float32_audio = int16_array.astype(np.float32) / 32768.0
            waveform = torch.from_numpy(float32_audio).unsqueeze(0)  # shape: [1, N]

        elif encoding == "pcm_mulaw":
            # Default assumption: 8000 Hz, mono
            input_sample_rate = 8000
            ulaw_array = np.frombuffer(chunk, dtype=np.uint8)
            ulaw_tensor = torch.from_numpy(ulaw_array).to(torch.int64)
            decoder = torchaudio.transforms.MuLawDecoding(quantization_channels=256)
            waveform = decoder(ulaw_tensor).unsqueeze(0)  # shape: [1, N]

        # Resample if needed
        if input_sample_rate != WHISPER_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=input_sample_rate, new_freq=WHISPER_SAMPLE_RATE
            )
            waveform = resampler(waveform)

        # Final output
        audio_tensor = waveform.squeeze(0)

        duration = time.time() - start_time
        logger.debug(
            f"🎵 Stream {stream_id}: Audio conversion completed in {duration:.3f}s | "
            f"encoding={encoding}, input_size_bytes={len(chunk)}, output_shape={audio_tensor.shape}, "
            f"duration_seconds={audio_tensor.shape[0] / WHISPER_SAMPLE_RATE:.3f}, sample_rate={WHISPER_SAMPLE_RATE}"
        )

        return audio_tensor

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Stream {stream_id}: Audio conversion failed after {duration:.3f}s: {e}")
        raise


def logprob_to_confidence(log_prob: float, stream_id: str = "unknown") -> Optional[float]:
    """
    Converts Whisper log probability to a confidence score in [0.0, 1.0]
    using a logistic sigmoid transformation with logging.
    """
    if log_prob is None:
        return None

    try:
        confidence = round(1 / (1 + np.exp(log_prob)), 3)
        logger.debug(
            f"📊 Stream {stream_id}: Confidence calculation",
            {"log_prob": log_prob, "confidence": confidence},
        )
        return confidence
    except Exception as e:
        logger.error(f"❌ Stream {stream_id}: Confidence calculation failed: {e}")
        return None


def ensure_json_serializable(obj, stream_id: str = "unknown"):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    """
    try:
        if isinstance(obj, dict):
            return {key: ensure_json_serializable(value, stream_id) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [ensure_json_serializable(item, stream_id) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    except Exception as e:
        logger.error(f"❌ Stream {stream_id}: JSON serialization conversion failed: {e}")
        return str(obj)  # Fallback to string representation


def calculate_audio_duration(
    audio_tensor: torch.Tensor, sample_rate: int = WHISPER_SAMPLE_RATE
) -> float:
    """Calculate audio duration in seconds."""
    return audio_tensor.shape[0] / sample_rate


def log_audio_processing_stats(
    stream_id: str, operation: str, audio_tensor: torch.Tensor, additional_stats: dict = None
) -> None:
    """Log audio processing statistics for monitoring."""
    import time

    stats = {
        "operation": operation,
        "tensor_shape": audio_tensor.shape,
        "duration_seconds": calculate_audio_duration(audio_tensor),
        "dtype": str(audio_tensor.dtype),
        "device": str(audio_tensor.device),
    }

    if additional_stats:
        stats.update(additional_stats)

    logger.debug(f"🎵 Stream {stream_id}: Audio processing stats", stats)


def validate_audio_tensor(audio_tensor: torch.Tensor, stream_id: str = "unknown") -> bool:
    """Validate audio tensor for processing."""
    try:
        if audio_tensor is None:
            logger.warning(f"⚠️ Stream {stream_id}: Audio tensor is None")
            return False

        if audio_tensor.numel() == 0:
            logger.warning(f"⚠️ Stream {stream_id}: Audio tensor is empty")
            return False

        if torch.isnan(audio_tensor).any():
            logger.warning(f"⚠️ Stream {stream_id}: Audio tensor contains NaN values")
            return False

        if torch.isinf(audio_tensor).any():
            logger.warning(f"⚠️ Stream {stream_id}: Audio tensor contains infinite values")
            return False

        return True

    except Exception as e:
        logger.error(f"❌ Stream {stream_id}: Audio tensor validation failed: {e}")
        return False
