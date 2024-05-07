import enum
from typing import Literal, Optional

import pydantic
from truss_chains import utils


class TranscribeParams(pydantic.BaseModel):
    wav_sampling_rate_hz: Literal[16000] = pydantic.Field(
        16000,
        description=" This is a constant of Whisper and should not be changed.",
    )
    macro_chunk_size_sec: int = pydantic.Field(
        300,
        description="This is the top-level splitting into larger 'macro' chunks (each "
        "will be split again into smaller 'micro chunks' to send to whisper).",
    )
    micro_chunk_size_sec: int = pydantic.Field(
        5,
        description="Each macro-chunk is split into micro-chunks. When using silence "
        "detection, this is the *maximal* size (i.e. an actual micro-chunk could be "
        "smaller): A point of minimal silence searched in the second half of the "
        "maximal micro chunk duration using the smoothed absolute waveform.",
    )
    silence_detection_smoothing_num_samples: int = pydantic.Field(
        1600,
        description="Number of samples to determine width of box smoothing "
        "filter. With sampling of 16 kHz, 1600 samples is 1/10 second.",
    )


class WhisperOutput(pydantic.BaseModel):
    text: str
    language: str
    bcp47_key: str = pydantic.Field(
        ...,
        description="IETF language tag, e.g. 'en', see. "
        "https://en.wikipedia.org/wiki/IETF_language_tag.",
    )


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
    # Do not change, taken from existing App.
    QUEUED = enum.auto()
    SUCCEEDED = enum.auto()
    PERMAFAILED = enum.auto()
    DEBUG_RESULT = enum.auto()


class JobDescriptor(pydantic.BaseModel):
    # Do not change, taken from existing App.
    media_url: str
    media_id: int
    job_uuid: str
    status: Optional[JobStatus] = None


class WorkletInput(pydantic.BaseModel):
    # This typo `media_for_transciption` is for backwards compatibility.
    # Also this would ideally be `list[JobDescriptor]` instead of string...
    media_for_transciption: str


class BatchOutput(pydantic.BaseModel):
    # Do not change, taken from existing App.
    success: bool
    jobs: list[JobDescriptor]
    error_message: Optional[str] = None


class TranscriptionSegmentExternal(pydantic.BaseModel):
    # Do not change, taken from existing App.
    start: float = pydantic.Field(..., description="In seconds.")
    end: float = pydantic.Field(..., description="In seconds.")
    text: str
    language: str = pydantic.Field(..., description="E.g. 'English'")
    bcp47_key: str = pydantic.Field(
        ...,
        description="IETF language tag, e.g. 'en', see. "
        "https://en.wikipedia.org/wiki/IETF_language_tag.",
    )


class TranscriptionExternal(pydantic.BaseModel):
    # Do not change, taken from existing App.
    media_url: str
    media_id: int  # Seems to be just 0 or 1.
    job_uuid: str
    status: JobStatus
    # TODO: this is not a great name.
    text: Optional[list[TranscriptionSegmentExternal]] = None
    failure_reason: Optional[str] = None

    @classmethod
    def from_job_descr(cls, job_descr: JobDescriptor, status: JobStatus):
        return TranscriptionExternal(
            media_url=job_descr.media_url,
            media_id=job_descr.media_id,
            job_uuid=job_descr.job_uuid,
            status=status,
        )


class TranscribeInput(pydantic.BaseModel):
    # Do not change, taken from existing App.
    job_descr: JobDescriptor
    params: TranscribeParams


class RetryConfig(pydantic.BaseModel):
    # Do not change, matches `async_predict` data-models..
    max_attempts: int = 1
    initial_delay_ms: int = 0
    max_delay_ms: int = 5000


class AsyncTranscriptionRequest(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(protected_namespaces=())
    # Do not change, matches `async_predict` data-models.
    model_input: TranscribeInput
    webhook_endpoint: str
    inference_retry_config: RetryConfig
