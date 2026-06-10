from typing import Dict

import pydantic
from truss_chains import pydantic_numpy
from whisper_utils.data_types import FallbackChunk, MicroChunkInfo, WhisperParams


class WhisperChainletInput(pydantic.BaseModel):
    whisper_params: WhisperParams = WhisperParams()
    seg_info: MicroChunkInfo = MicroChunkInfo()
    audio_wav: pydantic_numpy.NumpyArrayField
    asr_options: Dict = {}
    fallback_chunks: list[FallbackChunk] = []


class DiarizationInput(pydantic.BaseModel):
    audio_wav: pydantic_numpy.NumpyArrayField
    time_offset: float


class DiarizationOutput(pydantic.BaseModel):
    segmentation: pydantic_numpy.NumpyArrayField
    embedding: pydantic_numpy.NumpyArrayField
