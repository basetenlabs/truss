from typing import List, NamedTuple

from pydantic import BaseModel
from torch import Tensor

SUPPORTED_SAMPLE_RATE = 16_000
DEFAULT_NUM_BEAMS = 1
DEFAULT_MAX_NEW_TOKENS = 128


class BatchWhisperItem(NamedTuple):
    mel: Tensor
    prompt_ids: Tensor
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    num_beams: int = DEFAULT_NUM_BEAMS


class Segment(BaseModel):
    start: float
    end: float
    text: str


class WhisperResult(BaseModel):
    segments: List[Segment]
    language_code: str
