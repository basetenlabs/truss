"""
Core runtime components for Whisper model inference.
"""

from .tokenizer import Tokenizer
from .trt_model import WhisperTRTModel
from .utils import get_compression_ratio
from .whisperrt import WhisperRT

__all__ = [
    "WhisperRT",
    "Tokenizer",
    "WhisperTRTModel",
    "get_compression_ratio",
    "download_model",
]

# Lazy loading for heavy components
def __getattr__(name):
    if name == "WhisperRT":
        from .whisperrt import WhisperRT

        return WhisperRT
    if name == "WhisperTRTModel":
        from .trt_model import WhisperTRTModel

        return WhisperTRTModel
    raise AttributeError(f"module {__name__} has no attribute {name}")
