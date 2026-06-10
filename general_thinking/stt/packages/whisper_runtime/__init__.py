"""
Core package initialization for Whisper Runtime.
Provides access to main components and utilities.
"""

# Type hints for better IDE support
from typing import TYPE_CHECKING

from .configs import SAMPLE_RATE
from .runtime.utils import get_compression_ratio
from .runtime.whisperrt import WhisperRT

if TYPE_CHECKING:
    from .chunker import Chunker
    from .runtime.tokenizer import Tokenizer

import logging

# Set up logging with null handler
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "SAMPLE_RATE",
    "WhisperRT",
    "get_compression_ratio",
]
