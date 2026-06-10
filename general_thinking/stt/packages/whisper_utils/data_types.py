from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class WHISPER_MODEL(Enum):
    LARGE_V3 = "large-v3-251107"
    LARGE_V2 = "large-v2-251107"
    LARGE_V3_TURBO = "large-v3-turbo-251107"
    MEDIUM = "medium-251107"


# External Models ######################################################################


class MicroChunkInfo(BaseModel):
    start_time_sec: float = 0.0
    end_time_sec: float = 0.0
    duration_sec: float = 0.0
    chunk_index: int = -1


class WhisperSamplingParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    beam_width: int = 1
    length_penalty: float = 2.0
    repetition_penalty: float = 1.01
    beam_search_diversity_rate: Optional[float] = None
    no_repeat_ngram_size: Optional[int] = None
    sampling_temperatures: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    max_new_tokens: int = 400


class DiarizationGranularity(str, Enum):  # inherit from str if you want JSON-friendly values
    WORD = "word"
    SEGMENT = "segment"
    SENTENCE = "sentence"
    SPLIT_SEGMENT = "split_segment"


class WhisperParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: Optional[str] = ""
    output_language: Optional[str] = "auto"
    audio_language: Optional[str] = "en"
    language_detection_only: bool = False
    language_options: List[str] = []
    use_dynamic_preprocessing: bool = False
    use_missing_chunk_retry: bool = False
    missing_chunk_wpm_threshold: int = 100  # if words per minute transcribed is less than missing_chunk_wpm_threshold, retry the chunk
    show_word_timestamps: bool = False
    enable_vad: bool = True
    whisper_sampling_params: WhisperSamplingParams = WhisperSamplingParams()
    enable_diarization: Optional[bool] = None
    diarization_granularity: DiarizationGranularity = DiarizationGranularity.SENTENCE
    enable_word_level_diarization: bool = False  # keep for backward compatibility
    enable_sentence_level_diarization: bool = False  # keep for backward compatibility
    enable_final_diarization: bool = False
    prefix: Optional[str] = ""
    show_beam_results: bool = False
    enable_chunk_level_language_detection: Optional[bool] = False


# Word Level Timestamps
class Word(BaseModel):
    start_time: float
    end_time: float
    word: str
    prob: float
    speaker: Optional[str] = None


## whisper_rt types ##################################################################
class Segment(BaseModel):
    start_time: float
    end_time: float
    text: str
    log_prob: Optional[float] = None
    word_timestamps: list[Word] = []
    speaker: Optional[str] = None
    speaker_confidence: Optional[float] = None
    possible_hallucination: bool = False
    beam_results: Optional[List[str]] = None
    language_code: Optional[str] = None
    language_prob: Optional[float] = None


class FallbackChunk(BaseModel):
    start: int
    end: int


class ASRTimingInfoDurations(BaseModel):
    read_audio_duration_s: Optional[float] = None
    chunking_duration_s: Optional[float] = None
    language_detection_duration_s: Optional[float] = None
    transcription_duration_s: Optional[float] = None
    diarization_duration_s: Optional[float] = None
    speaker_assignment_duration_s: Optional[float] = None
    total_duration_s: Optional[float] = None


class ASRTimingInfo(BaseModel):
    """Stores timing information for various stages of audio processing."""

    overall_start_time: Optional[float] = None
    overall_end_time: Optional[float] = None
    start_read_audio_time: Optional[float] = None
    end_read_audio_time: Optional[float] = None
    start_chunking_time: Optional[float] = None
    end_chunking_time: Optional[float] = None
    start_language_detection_time: Optional[float] = None
    end_language_detection_time: Optional[float] = None
    start_transcription_time: Optional[float] = None
    end_transcription_time: Optional[float] = None
    start_diarization_time: Optional[float] = None
    end_diarization_time: Optional[float] = None
    start_speaker_assignment_time: Optional[float] = None
    end_speaker_assignment_time: Optional[float] = None
    durations: ASRTimingInfoDurations = ASRTimingInfoDurations()


class WhisperResult(BaseModel):
    segments: list[Segment] = []
    language_code: Optional[str] = None
    language_prob: Optional[float] = None
    diarization: List[Dict] = []
    timing_info: Optional[ASRTimingInfoDurations] = None
    enable_chunk_level_language_detection: Optional[bool] = False
    audio_length_sec: Optional[float] = None


class StreamingWhisperResult(WhisperResult):
    is_final: bool = False
    transcription_num: int = 0
    next_partial: Optional[Segment] = None
    pipeline_latency: Optional[float] = None
    is_end_of_audio_flush: Optional[bool] = None


## API Types ##################################################################
class AudioSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: Optional[str] = None
    audio_b64: Optional[str] = None
    audio_bytes: Optional[bytes] = None


class WhisperInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio: AudioSource
    asr_options: Dict = {}  # to be deprecated, use whisper_params.whisper_sampling_params instead
    whisper_params: WhisperParams = WhisperParams()
    include_timing_info: bool = False
    vad_config: Dict = {}
    diarization_config: Dict = {}


class StreamingParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    encoding: str = "pcm_s16le"
    sample_rate: int = 16000
    enable_partial_transcripts: bool = False
    partial_transcript_interval_s: float = 0.5
    final_transcript_max_duration_s: int = 30
    prioritize_latency: bool = True
    skip_partial_if_hallucination: bool = True
    include_pipeline_latency_metric: bool = False


class StreamingWhisperInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    whisper_params: WhisperParams = WhisperParams()
    streaming_params: StreamingParams = StreamingParams()
    streaming_vad_config: Dict = {}
    streaming_diarization_config: Dict = {}
    diarization_config: Dict = {}
    stream_id: Optional[str] = None
    include_timing_info: bool = False


class WhisperTrussInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    whisper_input: WhisperInput


class PostProcessingFlags(BaseModel):
    remove_hyphens_followed_by_space: bool = Field(
        default=False, description="Remove hyphens followed by a space"
    )
    remove_spaces_in_ja_zh: bool = Field(
        default=False, description="Remove spaces in Japanese and Chinese"
    )
    remove_non_speech_events_asterisk: bool = Field(
        default=False, description="Remove non-speech events between asterisks"
    )
    remove_non_speech_events_bracket: bool = Field(
        default=False, description="Remove non-speech events between brackets"
    )
    remove_non_speech_events_parenthesis: bool = Field(
        default=False, description="Remove non-speech events between parentheses"
    )
