import enum
from typing import Optional

import pydantic
from truss_chains import utils


class TranscribeParams(pydantic.BaseModel):
    # This is a constant of Whisper and should not be changed.
    wav_sampling_rate_hz: int = pydantic.Field(16000, const=True)  # type: ignore[call-arg]
    # This is the top-level splitting into larger "macro" chunks (each will be split
    # again into smaller "micro chunks" to send to whisper).
    macro_chunk_size_sec: int = 300
    # Each macro-chunk is split into micro-chunks.
    # When using silence detection, this is the *maximal* size (i.e. an actual
    # micro-chunk could be smaller):
    # A point of minimal silence searched in the second half
    # of the maximal micro chunk duration using the smoothed absolute waveform.
    micro_chunk_size_sec: int = 5
    # With sampling of 16 kHz, 1600 is 1/10 second.
    silence_detection_smoothing_num_samples: int = 1600
    result_webhook_url: str = "123"


class WhisperOutput(pydantic.BaseModel):
    text: str
    language: str
    bcp47_key: str


class SegmentInfo(pydantic.BaseModel):
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    macro_chunk: Optional[int] = None
    micro_chunk: Optional[int] = None


class Segment(pydantic.BaseModel):
    transcription: WhisperOutput
    segment_info: SegmentInfo


class TranscribeOutput(pydantic.BaseModel):
    segments: list[Segment]
    input_duration_sec: float
    processing_duration_sec: float
    speedup: float


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


########################################################################################


class JobStatus(utils.StrEnum):
    SUCCEEDED = enum.auto()
    PERMAFAILED = enum.auto()


class JobDescriptor(pydantic.BaseModel):
    media_url: str
    media_id: int
    job_uuid: str


class WorkletInput(pydantic.BaseModel):
    class Config:
        allow_population_by_field_name = True

    media_for_transcription: list[JobDescriptor] = pydantic.Field(
        ..., alias="media_for_transciption"
    )  # This typo is for backwards compatibility.


class TranscriptionSegmentExternal(pydantic.BaseModel):
    start: float  # Seconds.
    end: float  # Seconds.
    text: str
    language: str  # E.g. "English".
    bcp47_key: str  # E.g. "en".


class TranscriptionExternal(pydantic.BaseModel):
    media_url: str
    media_id: int  # Seems to be just 0 or 1.
    job_uuid: str
    status: JobStatus
    # TODO: this is not a great name.
    text: Optional[list[TranscriptionSegmentExternal]] = None
    failure_reason: Optional[str] = None
