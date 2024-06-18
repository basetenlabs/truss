from typing import Literal, Optional

import pydantic

# External Models ######################################################################


class TranscribeParams(pydantic.BaseModel):
    wav_sampling_rate_hz: Literal[16000] = pydantic.Field(
        default=16000,
        description=" This is a constant of Whisper and should not be changed.",
    )
    macro_chunk_size_sec: int = pydantic.Field(
        default=300,
        description="This is the top-level splitting into larger 'macro' chunks (each "
        "will be split again into smaller 'micro chunks' to send to whisper).",
    )
    macro_chunk_overlap_sec: int = pydantic.Field(
        default=0,
        description="Overlap to avoid cutting off words at the end of a macro-chunk.",
    )
    micro_chunk_size_sec: int = pydantic.Field(
        default=5,
        description="Each macro-chunk is split into micro-chunks. When using silence "
        "detection, this is the *maximal* size (i.e. an actual micro-chunk could be "
        "smaller): A point of minimal silence searched in the second half of the "
        "maximal micro chunk duration using the smoothed absolute waveform.",
    )
    silence_detection_smoothing_num_samples: int = pydantic.Field(
        default=1600,
        description="Number of samples to determine width of box smoothing "
        "filter. With sampling of 16 kHz, 1600 samples is 1/10 second.",
    )


class _BaseSegment(pydantic.BaseModel):
    # Common to internal whisper and external segment.
    start_time_sec: float
    end_time_sec: float
    text: str


class Segment(_BaseSegment):
    language: str
    language_code: str = pydantic.Field(
        ...,
        description="IETF language tag, e.g. 'en', see. "
        "https://en.wikipedia.org/wiki/IETF_language_tag.",
    )


class TranscribeOutput(pydantic.BaseModel):
    segments: list[Segment]
    input_duration_sec: float
    processing_duration_sec: float
    speedup: float


# Internal Models ######################################################################


class WhisperInput(pydantic.BaseModel):
    audio_b64: str


class WhisperSegment(_BaseSegment):
    pass


class WhisperResult(pydantic.BaseModel):
    segments: list[WhisperSegment]
    # Note: `language` is only `None` if `segments` are empty.
    language: Optional[str]
    language_code: Optional[str] = pydantic.Field(
        ...,
        description="IETF language tag, e.g. 'en', see. "
        "https://en.wikipedia.org/wiki/IETF_language_tag.",
    )


class ChunkInfo(pydantic.BaseModel):
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    start_time_str: str
    is_last: bool
    macro_chunk: int
    micro_chunk: Optional[int] = None
    processing_duration: Optional[float] = None


class SegmentList(pydantic.BaseModel):
    segments: list[Segment]
    chunk_info: ChunkInfo


class WavInfo(pydantic.BaseModel):
    num_channels: int
    sampling_rate_hz: int
    bytes_per_sample: int

    def get_chunk_size_bytes(self, chunk_duration_sec: int) -> int:
        return (
            chunk_duration_sec
            * self.sampling_rate_hz
            * self.bytes_per_sample
            * self.num_channels
        )

    def get_chunk_duration_sec(self, num_bytes: int) -> float:
        return (
            num_bytes
            / self.bytes_per_sample
            / self.sampling_rate_hz
            / self.num_channels
        )
