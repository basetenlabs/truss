import asyncio
import base64
import copy
import io
import logging
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional

import ffmpeg
import httpx
import numpy as np
import regex
import torch
from fastapi import HTTPException
from truss import truss_config

from .constants import (
    CHUNK_PADDING_SECONDS,
    COMPRESSION_THRESHOLD,
    MAX_BATCH_SIZE,
    MINIMAL_TIME_SEGMENT_DURATION,
    RETRY_CHUNK_PADDING_SECONDS,
    WHISPER_SAMPLE_RATE,
)
from .data_types import (
    ASRTimingInfo,
    ASRTimingInfoDurations,
    FallbackChunk,
    MicroChunkInfo,
    PostProcessingFlags,
    Segment,
    WhisperInput,
    WhisperParams,
    WhisperResult,
)

logger = logging.getLogger(__name__)


def get_gpu_accelerator():
    """
    Detects the GPU accelerator type based on the actual GPU device.

    Returns:
        truss_config.Accelerator: The accelerator type for the current GPU.
    """
    gpu_name = torch.cuda.get_device_name(0).upper()

    if "MIG" in gpu_name:
        return truss_config.Accelerator.H100_40GB

    if "H100" in gpu_name:
        return truss_config.Accelerator.H100

    if "A100" in gpu_name:
        return truss_config.Accelerator.A100

    if "L4" in gpu_name:
        return truss_config.Accelerator.L4

    if "B200" in gpu_name:
        return truss_config.Accelerator.B200

    if "H200" in gpu_name:
        return truss_config.Accelerator.H200

    raise ValueError(f"Incompatible GPU for Whisper: {gpu_name}")


def get_max_batch_size():
    """
    Gets the maximum batch size for the current GPU.

    Returns:
        int: The maximum batch size for the current GPU accelerator.
    """
    gpu_accelerator = get_gpu_accelerator()
    return MAX_BATCH_SIZE[gpu_accelerator]


async def download_audio(client: httpx.AsyncClient, audio_url: str) -> bytes:
    """Download audio file from URL asynchronously."""
    logger.debug("Downloading audio file")
    try:
        req = await client.get(audio_url)
        req.raise_for_status()
        return req.content
    except Exception as e:
        raise ValueError(
            f"Failed to download audio: {str(e)}",
            req.status_code if "req" in locals() else None,
        )


def load_audio_torchcodec(audio_content: bytes) -> torch.Tensor:
    """Load and decode audio from bytes, resample to 16 kHz mono."""
    from torchcodec.decoders import AudioDecoder

    decoder = AudioDecoder(io.BytesIO(audio_content), sample_rate=WHISPER_SAMPLE_RATE)
    result = decoder.get_all_samples()

    # Validate decoder output before proceeding
    if result is None or not hasattr(result, "data"):
        raise ValueError("Failed to decode audio: decoder returned no data.")

    wav = result.data  # expected shape: [channels, samples]

    if not isinstance(wav, torch.Tensor) or wav.numel() == 0:
        raise ValueError("Failed to decode audio: decoded waveform is empty or invalid.")

    if wav.ndim != 2 or wav.shape[0] == 0 or wav.shape[1] == 0:
        raise ValueError("Failed to decode audio: decoded waveform has an unexpected shape.")
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0)
    else:
        wav = wav.squeeze(0)

    wav = wav.to(torch.float16)
    logger.debug(f"Loaded audio using torchcodec ({wav.shape[0] / WHISPER_SAMPLE_RATE:.1f}s)")
    return wav


async def load_audio(whisper_input: WhisperInput, client: httpx.AsyncClient) -> torch.Tensor:
    if whisper_input.audio.url:
        try:
            audio_content = await download_audio(client, whisper_input.audio.url)
        except Exception as e:
            logger.error(
                f"Failed to download audio for link: {whisper_input.audio.url}. Error: {e}"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download audio from the provided url: {whisper_input.audio.url}. Check the logs for more details.",
            )
    elif whisper_input.audio.audio_b64:
        # Decode base64 audio otherwise
        try:
            audio_content = base64.b64decode(whisper_input.audio.audio_b64)
        except Exception as e:
            logger.error(f"Failed to decode base64 audio: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail="Failed to decode base64 audio. Make sure the audio is in base64 format.",
            )
    elif whisper_input.audio.audio_bytes:
        audio_content = whisper_input.audio.audio_bytes
    else:
        raise HTTPException(status_code=400, detail="No audio source provided")

    """Load audio using torchcodec with ffmpeg fallback."""
    if whisper_input.whisper_params.use_dynamic_preprocessing:
        return load_audio_ffmpeg(audio_content, True)

    try:
        return load_audio_torchcodec(audio_content)
    except Exception as e:
        logger.warning(f"Torchcodec failed to load audio, falling back to ffmpeg: {str(e)}")
        try:
            return load_audio_ffmpeg(audio_content, False)
        except ffmpeg.Error:
            raise ValueError(
                "Failed to load audio with both torchcodec and ffmpeg. ffmpeg was not able to load the audio."
            )
        except Exception as other_error:
            raise ValueError(
                f"Failed to load audio with both torchcodec and ffmpeg: {str(other_error)}"
            )


def load_audio_ffmpeg(audio_content: bytes, dynamic_preprocessing: bool) -> torch.Tensor:
    """Load audio using ffmpeg fallback, attempting to recover partial content from corrupted files."""

    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
        temp_file.write(audio_content)
        temp_file.flush()

        stream = ffmpeg.input(temp_file.name)
        if dynamic_preprocessing:
            stream = ffmpeg.output(
                stream,
                "pipe:",
                format="f32le",
                acodec="pcm_f32le",
                ac=1,
                ar=16000,
                af="dynaudnorm",
            )
        else:
            stream = ffmpeg.output(
                stream, "pipe:", format="f32le", acodec="pcm_f32le", ac=1, ar=16000
            )
        try:
            out, stderr = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)

            # Process whatever data we got
            if len(out) > 0:
                audio_np = np.frombuffer(out, np.float32).copy()
                wav = torch.from_numpy(audio_np).to(torch.float16)
                logger.info(f"Loaded audio using ffmpeg fallback - got {len(audio_np)} samples")
                return wav
            else:
                raise ValueError("ffmpeg could not load audio at all")

        except ffmpeg.Error as e:
            # If we got any output data despite the error, try to use it
            if e.stdout is not None and len(e.stdout) > 0:
                audio_np = np.frombuffer(e.stdout, np.float32)
                wav = torch.from_numpy(audio_np).to(torch.float16)
                logger.warning(
                    f"Partially loaded corrupted audio - recovered {len(audio_np)} samples"
                )
                return wav
            else:
                raise e


def update_segments_timestamps(retry_results: list[WhisperResult], seg_infos: list[MicroChunkInfo]):
    all_segments = []

    for i, result in enumerate(retry_results):
        if not isinstance(result, WhisperResult):
            # logger.warn("Result is not a WhisperResult, converting to WhisperResult")
            whisper_transcribe_result = WhisperResult(**result)
        else:
            whisper_transcribe_result = result

        language_code = whisper_transcribe_result.language_code
        language_prob = whisper_transcribe_result.language_prob

        # logger.debug(f"Updating timestamps for {whisper_transcribe_result=}")

        if len(whisper_transcribe_result.segments) > 0:
            for segment in whisper_transcribe_result.segments:
                segment.start_time += seg_infos[i].start_time_sec or 0
                segment.end_time += seg_infos[i].start_time_sec or 0
                segment.language_code = language_code
                segment.language_prob = language_prob

                for alignment in segment.word_timestamps:
                    alignment.start_time += segment.start_time
                    alignment.end_time += segment.start_time
        all_segments.extend(whisper_transcribe_result.segments)
    return all_segments


async def detect_audio_language(
    audio_wav: torch.Tensor,
    whisper_params: WhisperParams,
    asr_options: dict,
    speech_timestamps: list[dict],
    collect_chunks_fn: Callable,
    call_whisper_detect_language: Callable,
) -> WhisperResult:
    sts = speech_timestamps[0]
    audio_chunk = collect_chunks_fn([sts], audio_wav)
    language_detection_result: WhisperResult = await call_whisper_detect_language(
        audio_chunk, whisper_params, asr_options
    )
    language_detection_result.segments = []
    return language_detection_result


def log_time(timing_info: ASRTimingInfo, duration_secs: float):
    """
    Log processing times for different stages of audio processing.

    Args:
        timing_info: ASRTimingInfo object containing timing data for various processing stages
    """
    # Calculate individual component times
    audio_processing_time = (
        f"{timing_info.durations.read_audio_duration_s:.2f}s"
        if timing_info.durations.read_audio_duration_s is not None
        else "N/A"
    )
    chunking_time = (
        f"{timing_info.durations.chunking_duration_s:.2f}s"
        if timing_info.durations.chunking_duration_s is not None
        else "N/A"
    )
    language_detection_time = (
        f"{timing_info.durations.language_detection_duration_s:.2f}s"
        if timing_info.durations.language_detection_duration_s is not None
        else "N/A"
    )
    transcription_time = (
        f"{timing_info.durations.transcription_duration_s:.2f}s"
        if timing_info.durations.transcription_duration_s is not None
        else "N/A"
    )
    diarization_time = (
        f"{timing_info.durations.diarization_duration_s:.2f}s"
        if timing_info.durations.diarization_duration_s is not None
        else "N/A"
    )
    speaker_assignment_time = (
        f"{timing_info.durations.speaker_assignment_duration_s:.2f}s"
        if timing_info.durations.speaker_assignment_duration_s is not None
        else "N/A"
    )

    # Calculate overall processing time
    overall_time = (
        f"{(timing_info.overall_end_time - timing_info.overall_start_time):.2f}s"
        if (timing_info.overall_end_time and timing_info.overall_start_time)
        else "N/A"
    )

    # Log all timing information
    logger.debug(
        f"""Timing Info:
            Audio Duration: {duration_secs:.2f}s,
            Audio Processing Time: {audio_processing_time},
            Chunking Time: {chunking_time},
            Language Detection Time: {language_detection_time},
            Transcription Time: {transcription_time},
            Diarization Time: {diarization_time},
            Speaker Assignment Time: {speaker_assignment_time},
            Total Processing Time: {overall_time}"""
    )

    # Additional detailed logging at debug level
    logger.debug(f"Detailed timing metrics: {timing_info}")


def perform_vad_chunking(
    wav, vad_model, vad_config_overrides, whisper_input_vad_config, use_missing_chunk_retry
) -> tuple[list[dict], list[list[FallbackChunk]]]:
    """
    Perform Voice Activity Detection (VAD) chunking on audio data.

    Args:
        wav: Audio waveform tensor
        vad_model: The VAD model to use for chunking
        vad_config_overrides: Configuration overrides for the VAD model
        whisper_input_vad_config: VAD config overrides from whisper_input
        use_missing_chunk_retry: Whether to use missing chunk retry

    Returns:
        Tuple of (speech_timestamps, fallback_chunks)

    Raises:
        HTTPException: If VAD chunking fails
    """
    from silero_vad.utils_vad import BatchSettings, get_speech_timestamps

    try:
        vad_config_overrides = vad_config_overrides
        if whisper_input_vad_config:
            vad_config_overrides.update(whisper_input_vad_config)
        logger.debug(f"VAD config overrides: {vad_config_overrides}")

        # Ensure the wav tensor is on the same device as the VAD model
        model_device = vad_model.device
        wav_device = wav.device

        if wav_device != model_device:
            wav = wav.to(model_device)

        speech_timestamps, fallback_chunks = get_speech_timestamps(
            wav, vad_model, batch_settings=BatchSettings(), **vad_config_overrides
        )

        logger.debug(f"Detected {len(speech_timestamps)} speech segments from VAD")
        if use_missing_chunk_retry:
            segment_fallback_chunks = []
            if len(speech_timestamps) != len(fallback_chunks):
                logger.warning(
                    f"Speech timestamps and fallback chunks have different lengths: {len(speech_timestamps)} != {len(fallback_chunks)}, skipping fallback chunks"
                )
            else:
                for i, sts in enumerate(speech_timestamps):
                    if len(fallback_chunks[i]) > 0:
                        segment_fallback_chunks.append(
                            [
                                FallbackChunk(
                                    start=max(0, x["start"] - sts["start"]),
                                    end=x["end"] - sts["start"],
                                )
                                for x in fallback_chunks[i]
                            ]
                        )
                    else:
                        segment_fallback_chunks.append([])
        else:
            segment_fallback_chunks = [[] for _ in speech_timestamps]

        return speech_timestamps, segment_fallback_chunks
    except Exception as e:
        logger.error(f"Error during VAD chunking: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during VAD chunking: {str(e)}")


async def call_whisper_model(
    whisper_model,
    thread_pool_executor: ThreadPoolExecutor,
    seg_info: MicroChunkInfo,
    audio_chunk: np.ndarray,
    fallback_chunks: list[FallbackChunk],
    whisper_params: WhisperParams,
    asr_options: dict,
) -> WhisperResult:
    """
    Transcribe an audio chunk.

    Args:
        whisper_model: The Whisper model to use for transcription
        thread_pool_executor: The thread pool executor to use for transcription
        seg_info: The segment information for the chunk to transcribe
        audio_chunk: The audio chunk to transcribe
        fallback_chunks: The fallback chunks to use for the chunk
        whisper_params: The whisper parameters to use for the chunk
        asr_options: The ASR options to use for the chunk
    """

    logger.debug(
        f"Transcribing chunk {seg_info.chunk_index} with {seg_info=}, {whisper_params=}, {asr_options=}"
    )

    durations = ASRTimingInfoDurations()
    start_transcription_time = time.time()

    from whisper_runtime import get_compression_ratio

    try:
        result = await asyncio.wrap_future(
            thread_pool_executor.submit(
                whisper_model.transcribe,
                audio_wav=audio_chunk,
                asr_options=asr_options,
                whisper_sampling_params=whisper_params.whisper_sampling_params,
                output_language=whisper_params.output_language,
                audio_language=whisper_params.audio_language,
                language_detection_only=whisper_params.language_detection_only,
                language_options=whisper_params.language_options,
                prompt=whisper_params.prompt,
                show_word_timestamps=whisper_params.show_word_timestamps,
                begin_padding_seconds=CHUNK_PADDING_SECONDS,
                prefix=whisper_params.prefix,
                show_beam_results=whisper_params.show_beam_results,
            )
        )

        if result["segments"] and whisper_params.use_missing_chunk_retry:
            num_words = sum(len(segment["text"].split()) for segment in result["segments"])
            audio_length_sec = result["segments"][-1]["end_time"]
            wpm = num_words / audio_length_sec * 60
            compression_ratio = get_compression_ratio(result["segments"][0]["text"])

            # if the WPM is below a certain threshold (indicated missing block) or the compression ratio is too high, we will try to retry with the fallback chunks
            if (
                wpm < whisper_params.missing_chunk_wpm_threshold
                or compression_ratio > COMPRESSION_THRESHOLD
            ):
                logger.info(
                    f"Retrying chunk {seg_info.chunk_index} with {wpm=} < {whisper_params.missing_chunk_wpm_threshold} or {compression_ratio=} > {COMPRESSION_THRESHOLD}"
                )
                retry_results = []
                # transcribe each chunk
                for fallback_chunk in fallback_chunks:
                    # Check if fallback_chunk is a list (old format) or FallbackChunk object (new format)
                    start, end = fallback_chunk.start, fallback_chunk.end
                    duration = (end - start) / WHISPER_SAMPLE_RATE
                    logger.debug(
                        f"Transcribing fallback chunk: [{start / WHISPER_SAMPLE_RATE}: {end / WHISPER_SAMPLE_RATE}]"
                    )
                    chunk = audio_chunk[start:end]
                    try:
                        retry_result = await asyncio.wrap_future(
                            thread_pool_executor.submit(
                                whisper_model.transcribe,
                                audio_wav=chunk,
                                asr_options=asr_options,
                                whisper_sampling_params=whisper_params.whisper_sampling_params,
                                output_language=whisper_params.output_language,
                                audio_language=whisper_params.audio_language,
                                language_detection_only=False,
                                language_options=[],
                                prompt=whisper_params.prompt,
                                show_word_timestamps=whisper_params.show_word_timestamps,
                                begin_padding_seconds=RETRY_CHUNK_PADDING_SECONDS,
                                prefix=whisper_params.prefix,
                            )
                        )

                        # adjust the timestamps to match the original audio
                        retry_result["segments"][0]["start_time"] += start / WHISPER_SAMPLE_RATE
                        retry_result["segments"][0]["end_time"] += start / WHISPER_SAMPLE_RATE

                        # if the compression ratio is still too high, we will not include the result
                        if (
                            get_compression_ratio(retry_result["segments"][0]["text"])
                            < COMPRESSION_THRESHOLD
                        ):
                            retry_results.append(retry_result)
                    except Exception as e:
                        print(f"Error during transcription: {str(e)}")
                        raise
                result["segments"] = [r["segments"][0] for r in retry_results]

        logger.debug(f"WhisperResult: {result}")
        durations.transcription_duration_s = time.time() - start_transcription_time
        result["timing_info"] = durations.model_dump()
        result["audio_length_sec"] = audio_chunk.shape[0] / WHISPER_SAMPLE_RATE
        return WhisperResult(**result)
    except Exception as e:
        import traceback

        stack_trace = traceback.format_exc()
        raise RuntimeError(f"Failed to transcribe chunk {seg_info.chunk_index}: {stack_trace}")


async def diarize(
    diarization_pipeline,
    wav: torch.Tensor,
    device: torch.device,
    config: Optional[dict] = None,
    thread_pool_executor: Optional[ThreadPoolExecutor] = None,
) -> tuple:
    """
    Perform speaker diarization on the audio.

    Args:
        diarization_pipeline: The diarization pipeline to use
        wav: Audio waveform tensor
        device: The device to run diarization on
        config: Diarization configuration
        thread_pool_executor: Thread pool executor for running blocking operations

    Returns:
        Tuple of (diarization segments, timing_info)
    """
    from pyannote.audio.pipelines.utils.hook import Hooks, ProgressHook, TimingHook

    from .constants import WHISPER_SAMPLE_RATE
    from .data_types import ASRTimingInfo

    diarization_timing = ASRTimingInfo()

    logger.info("Starting diarization")
    diarization_timing.start_diarization_time = time.time()

    # Set device early to avoid unnecessary transfers
    torch.cuda.set_device(device)

    # Prepare waveform efficiently
    # Clone is necessary to avoid modifying the input tensor, but we can optimize the operations
    waveform = wav.clone()

    # Reshape operations
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 2 and waveform.shape[0] > waveform.shape[1]:
        waveform = waveform.t()

    # Convert to float32 and move to device in one operation (more efficient than separate ops)
    # For long audio files, moving to GPU early reduces CPU-GPU transfer overhead
    waveform = waveform.to(device=device, dtype=torch.float32, non_blocking=True)

    diarization = None

    def _run_diarization():
        """Blocking function to run diarization in thread pool"""
        with torch.no_grad():
            logger.info("Starting speaker diarization with verbose metrics...")
            result = diarization_pipeline(
                {"waveform": waveform, "sample_rate": WHISPER_SAMPLE_RATE}
            )
            logger.info("Diarization completed!")
        return result

    try:
        if thread_pool_executor:
            logger.info("Running diarization in thread pool")
            diarization = await asyncio.wrap_future(thread_pool_executor.submit(_run_diarization))
        else:
            logger.info("Running diarization directly (blocking)")
            diarization = _run_diarization()
    finally:
        # Ensure waveform is deleted and CUDA cache is cleared
        del waveform

    diarization_timing.end_diarization_time = time.time()
    diarization_timing.durations.diarization_duration_s = (
        diarization_timing.end_diarization_time - diarization_timing.start_diarization_time
    )

    return diarization, diarization_timing


def assign_speakers_to_segments(
    transcription_segments: list,
    diarization_stack,
    fill_nearest=False,
    diarization_granularity=None,
) -> tuple:
    """
    Assign speakers to segments based on the diarization result.

    Args:
        transcription_segments: List of transcription segments
        diarization_result: Diarization result from pyannote
        fill_nearest: Whether to fill gaps with nearest speaker
        diarization_granularity: DiarizationGranularity enum or string value

    Returns:
        Tuple of (transcription_segments, diarization_data)
    """
    from whisper_utils.data_types import DiarizationGranularity

    # Default to SEGMENT mode if not specified
    if diarization_granularity is None:
        diarization_granularity = DiarizationGranularity.SEGMENT

    diarization_data, diarize_speakers, diarize_starts, diarize_ends = diarization_stack

    if not diarization_data:
        logger.debug(f"No diarization data found for {transcription_segments=}")
        return transcription_segments, diarization_data

    # Convert to numpy arrays for vectorized operations
    diarize_starts = np.array(diarize_starts)
    diarize_ends = np.array(diarize_ends)
    diarize_speakers = np.array(diarize_speakers)

    transcription_segments_before = copy.deepcopy(transcription_segments)

    # Handle both string and enum values
    if isinstance(diarization_granularity, DiarizationGranularity):
        diarization_granularity = diarization_granularity.value

    if diarization_granularity == "word":
        # Word-level assignment (existing implementation)
        # Check if any segment has non-empty word_timestamps
        has_word_timestamps = any(len(seg.word_timestamps) > 0 for seg in transcription_segments)

        if has_word_timestamps:
            # Collect all words with their segment info
            all_words = []
            word_starts = []
            word_ends = []

            for seg_idx, seg in enumerate(transcription_segments):
                if seg.word_timestamps:
                    for word in seg.word_timestamps:
                        if word.start_time is not None:
                            all_words.append(word)
                            word_starts.append(word.start_time)
                            if word.end_time == word.start_time:
                                word_ends.append(word.end_time + MINIMAL_TIME_SEGMENT_DURATION)
                            else:
                                word_ends.append(word.end_time)

            if all_words:
                word_starts = np.array(word_starts)
                word_ends = np.array(word_ends)

                # Vectorized calculation for all words at once
                # Shape: (num_words, num_diarization_segments)
                word_intersections = np.maximum(
                    0,
                    np.minimum(word_ends[:, np.newaxis], diarize_ends[np.newaxis, :])
                    - np.maximum(word_starts[:, np.newaxis], diarize_starts[np.newaxis, :]),
                )

                # For each word, find the speaker with maximum overlap or nearest speaker
                for word_idx, word in enumerate(all_words):
                    intersections = word_intersections[word_idx]
                    word_start = word_starts[word_idx]
                    word_end = word_ends[word_idx]

                    # Check if there's any actual overlap
                    if np.any(intersections > 0):
                        # Use existing overlap-based logic
                        valid_mask = intersections > 0
                        valid_intersections = intersections[valid_mask]
                        valid_speakers = diarize_speakers[valid_mask]
                        valid_indices = np.where(valid_mask)[
                            0
                        ]  # Get indices of overlapping segments

                        # Log detailed overlap information
                        logger.debug(
                            f"🔍 Word '{word.word}' at [{word_start:.3f}-{word_end:.3f}] overlaps with:"
                        )
                        for i, (intersection, speaker, idx) in enumerate(
                            zip(valid_intersections, valid_speakers, valid_indices)
                        ):
                            diarize_start = diarize_starts[idx]
                            diarize_end = diarize_ends[idx]
                            logger.debug(
                                f"  🔍 Speaker '{speaker}': [{diarize_start:.3f}-{diarize_end:.3f}] (overlap: {intersection:.3f}s)"
                            )

                        unique_speakers, inverse_indices = np.unique(
                            valid_speakers, return_inverse=True
                        )
                        speaker_overlaps = np.bincount(inverse_indices, weights=valid_intersections)
                        max_idx = np.argmax(speaker_overlaps)
                        selected_speaker = unique_speakers[max_idx]

                        # Log speaker selection details
                        logger.debug(
                            f"  🎯 Selected speaker '{selected_speaker}' with total overlap: {speaker_overlaps[max_idx]:.3f}s"
                        )
                        if len(speaker_overlaps) > 1:
                            logger.debug(
                                f"  📊 All speaker overlaps: {dict(zip(unique_speakers, speaker_overlaps))}"
                            )

                        word.speaker = selected_speaker

                    elif fill_nearest:
                        # No overlap found, find nearest speaker by temporal distance
                        word_start = word_starts[word_idx]
                        word_end = word_ends[word_idx]

                        # Calculate temporal distance to each diarization segment
                        distances = np.zeros(len(diarize_starts))

                        for i in range(len(diarize_starts)):
                            seg_start = diarize_starts[i]
                            seg_end = diarize_ends[i]

                            if word_end < seg_start:
                                # Word is before this diarization segment
                                distances[i] = seg_start - word_end
                            elif word_start > seg_end:
                                # Word is after this diarization segment
                                distances[i] = word_start - seg_end
                            else:
                                # This should not happen since we already checked for overlap
                                distances[i] = 0

                        # Find the speaker with minimum distance
                        min_distance = np.min(distances)
                        min_indices = np.where(distances == min_distance)[0]

                        if len(min_indices) > 1:
                            # Tie-breaking: prefer speakers that come after the word (temporal direction preference)
                            future_speakers = []
                            past_speakers = []

                            for idx in min_indices:
                                if diarize_starts[idx] > word_end:
                                    future_speakers.append(idx)
                                else:
                                    past_speakers.append(idx)

                            # Prefer future speakers, fallback to past speakers
                            if future_speakers:
                                nearest_idx = future_speakers[0]  # First future speaker
                            else:
                                nearest_idx = past_speakers[-1]  # Last past speaker (most recent)
                        else:
                            nearest_idx = min_indices[0]

                        word.speaker = diarize_speakers[nearest_idx]

                        logger.debug(
                            f"Word '{word.word}' at [{word_start:.2f}-{word_end:.2f}] "
                            f"assigned to nearest speaker '{word.speaker}' "
                            f"(distance: {distances[nearest_idx]:.2f}s, "
                            f"tie-broken: {'future' if diarize_starts[nearest_idx] > word_end else 'past'})"
                        )

        # Rearrange segments based on their speaker
        transcription_segments = rearrange_segments_post_diarization(transcription_segments)

        # If no word timestamps were available, fall back to segment-level assignment
        if not has_word_timestamps or not all_words:
            logger.debug("No word timestamps available, falling back to segment-level assignment")
            for segment in transcription_segments:
                if segment.start_time is not None and segment.end_time is not None:
                    # Calculate intersection with each diarization segment
                    segment_start = segment.start_time
                    segment_end = segment.end_time

                    # Vectorized calculation for this segment
                    intersections = np.maximum(
                        0,
                        np.minimum(segment_end, diarize_ends)
                        - np.maximum(segment_start, diarize_starts),
                    )

                    # Check if there's any actual overlap
                    if np.any(intersections > 0):
                        # Use overlap-based logic
                        valid_mask = intersections > 0
                        valid_intersections = intersections[valid_mask]
                        valid_speakers = diarize_speakers[valid_mask]
                        valid_indices = np.where(valid_mask)[
                            0
                        ]  # Get indices of overlapping segments

                        unique_speakers, inverse_indices = np.unique(
                            valid_speakers, return_inverse=True
                        )
                        speaker_overlaps = np.bincount(inverse_indices, weights=valid_intersections)
                        max_idx = np.argmax(speaker_overlaps)
                        segment.speaker = unique_speakers[max_idx]

                        # Log detailed overlap information
                        logger.debug(
                            f"🔍 Segment '{segment.text[:50]}...' at [{segment_start:.3f}-{segment_end:.3f}] overlaps with:"
                        )
                        for i, (intersection, speaker, idx) in enumerate(
                            zip(valid_intersections, valid_speakers, valid_indices)
                        ):
                            diarize_start = diarize_starts[idx]
                            diarize_end = diarize_ends[idx]
                            logger.debug(
                                f"  🔍 Speaker '{speaker}': [{diarize_start:.3f}-{diarize_end:.3f}] (overlap: {intersection:.3f}s)"
                            )

                        # Log speaker selection details
                        logger.debug(
                            f"  🎯 Selected speaker '{segment.speaker}' with total overlap: {speaker_overlaps[max_idx]:.3f}s"
                        )
                        if len(speaker_overlaps) > 1:
                            logger.debug(
                                f"  📊 All speaker overlaps: {dict(zip(unique_speakers, speaker_overlaps))}"
                            )

                    elif fill_nearest:
                        # No overlap found, find nearest speaker by temporal distance
                        distances = np.zeros(len(diarize_starts))

                        for i in range(len(diarize_starts)):
                            seg_start = diarize_starts[i]
                            seg_end = diarize_ends[i]

                            if segment_end < seg_start:
                                # Segment is before this diarization segment
                                distances[i] = seg_start - segment_end
                            elif segment_start > seg_end:
                                # Segment is after this diarization segment
                                distances[i] = segment_start - seg_end
                            else:
                                # This should not happen since we already checked for overlap
                                distances[i] = 0

                        # Find the speaker with minimum distance
                        min_distance = np.min(distances)
                        min_indices = np.where(distances == min_distance)[0]

                        if len(min_indices) > 1:
                            # Tie-breaking: prefer speakers that come after the segment (temporal direction preference)
                            future_speakers = []
                            past_speakers = []

                            for idx in min_indices:
                                if diarize_starts[idx] > segment_end:
                                    future_speakers.append(idx)
                                else:
                                    past_speakers.append(idx)

                            # Prefer future speakers, fallback to past speakers
                            if future_speakers:
                                nearest_idx = future_speakers[0]  # First future speaker
                            else:
                                nearest_idx = past_speakers[-1]  # Last past speaker (most recent)
                        else:
                            nearest_idx = min_indices[0]

                        segment.speaker = diarize_speakers[nearest_idx]

                        logger.debug(
                            f"Segment '{segment.text[:50]}...' at [{segment_start:.2f}-{segment_end:.2f}] "
                            f"assigned to nearest speaker '{segment.speaker}' "
                            f"(distance: {distances[nearest_idx]:.2f}s, "
                            f"tie-broken: {'future' if diarize_starts[nearest_idx] > segment_end else 'past'})"
                        )
                    else:
                        # No overlap and fill_nearest is False, assign UNKNOWN
                        segment.speaker = "UNKNOWN"
                        logger.debug(
                            f"Segment '{segment.text[:50]}...' at [{segment_start:.2f}-{segment_end:.2f}] "
                            f"assigned to UNKNOWN (no overlap found)"
                        )

    else:
        # Segment-level assignment
        for segment in transcription_segments:
            if segment.start_time is not None and segment.end_time is not None:
                # Calculate intersection with each diarization segment
                segment_start = segment.start_time
                segment_end = segment.end_time

                # Vectorized calculation for this segment
                intersections = np.maximum(
                    0,
                    np.minimum(segment_end, diarize_ends)
                    - np.maximum(segment_start, diarize_starts),
                )

                # Check if there's any actual overlap
                if np.any(intersections > 0):
                    # Use overlap-based logic
                    valid_mask = intersections > 0
                    valid_intersections = intersections[valid_mask]
                    valid_speakers = diarize_speakers[valid_mask]
                    valid_indices = np.where(valid_mask)[0]  # Get indices of overlapping segments

                    unique_speakers, inverse_indices = np.unique(
                        valid_speakers, return_inverse=True
                    )
                    speaker_overlaps = np.bincount(inverse_indices, weights=valid_intersections)
                    max_idx = np.argmax(speaker_overlaps)
                    segment.speaker = unique_speakers[max_idx]

                    # Log detailed overlap information
                    logger.debug(
                        f"🔍 Segment '{segment.text[:50]}...' at [{segment_start:.3f}-{segment_end:.3f}] overlaps with:"
                    )
                    for i, (intersection, speaker, idx) in enumerate(
                        zip(valid_intersections, valid_speakers, valid_indices)
                    ):
                        diarize_start = diarize_starts[idx]
                        diarize_end = diarize_ends[idx]
                        logger.debug(
                            f"  🔍 Speaker '{speaker}': [{diarize_start:.3f}-{diarize_end:.3f}] (overlap: {intersection:.3f}s)"
                        )

                    # Log speaker selection details
                    logger.debug(
                        f"  🎯 Selected speaker '{segment.speaker}' with total overlap: {speaker_overlaps[max_idx]:.3f}s"
                    )
                    if len(speaker_overlaps) > 1:
                        logger.debug(
                            f"  📊 All speaker overlaps: {dict(zip(unique_speakers, speaker_overlaps))}"
                        )

                elif fill_nearest:
                    # No overlap found, find nearest speaker by temporal distance
                    distances = np.zeros(len(diarize_starts))

                    for i in range(len(diarize_starts)):
                        seg_start = diarize_starts[i]
                        seg_end = diarize_ends[i]

                        if segment_end < seg_start:
                            # Segment is before this diarization segment
                            distances[i] = seg_start - segment_end
                        elif segment_start > seg_end:
                            # Segment is after this diarization segment
                            distances[i] = segment_start - seg_end
                        else:
                            # This should not happen since we already checked for overlap
                            distances[i] = 0

                    # Find the speaker with minimum distance
                    min_distance = np.min(distances)
                    min_indices = np.where(distances == min_distance)[0]

                    if len(min_indices) > 1:
                        # Tie-breaking: prefer speakers that come after the segment (temporal direction preference)
                        future_speakers = []
                        past_speakers = []

                        for idx in min_indices:
                            if diarize_starts[idx] > segment_end:
                                future_speakers.append(idx)
                            else:
                                past_speakers.append(idx)

                        # Prefer future speakers, fallback to past speakers
                        if future_speakers:
                            nearest_idx = future_speakers[0]  # First future speaker
                        else:
                            nearest_idx = past_speakers[-1]  # Last past speaker (most recent)
                    else:
                        nearest_idx = min_indices[0]

                    segment.speaker = diarize_speakers[nearest_idx]

                    logger.debug(
                        f"Segment '{segment.text[:50]}...' at [{segment_start:.2f}-{segment_end:.2f}] "
                        f"assigned to nearest speaker '{segment.speaker}' "
                        f"(distance: {distances[nearest_idx]:.2f}s, "
                        f"tie-broken: {'future' if diarize_starts[nearest_idx] > segment_end else 'past'})"
                    )
                else:
                    # No overlap and fill_nearest is False, assign UNKNOWN
                    segment.speaker = "UNKNOWN"
                    logger.debug(
                        f"Segment '{segment.text[:50]}...' at [{segment_start:.2f}-{segment_end:.2f}] "
                        f"assigned to UNKNOWN (no overlap found)"
                    )

    logger.debug(
        f"🟢 {diarization_data=}\n🟢 {transcription_segments_before=}\n🟢 {transcription_segments=}"
    )

    return transcription_segments, diarization_data


def rearrange_segments_post_diarization(segments: list) -> list:
    """
    Rearrange words into segments based on their speaker.
    Groups consecutive words from the same speaker into new segments.
    """
    from .data_types import Segment

    # If there are no segments, return empty list
    if not segments:
        return []

    # Extract all words with their speaker information
    all_words = []
    for segment in segments:
        if segment.word_timestamps:
            for word in segment.word_timestamps:
                # Use "UNKNOWN" if no speaker information is available
                speaker = getattr(word, "speaker", "UNKNOWN") or "UNKNOWN"
                # Set the speaker attribute in case it didn't exist
                word.speaker = speaker
                all_words.append(word)

    # No word timestamps available, return original segments
    if not all_words:
        return segments

    # Group consecutive words by the same speaker into new segments
    new_segments = []
    current_speaker = None
    current_words = []

    for word in all_words:
        # Check if speaker is changing
        if current_speaker is None or word.speaker != current_speaker:
            # If we have accumulated words, create a new segment
            if current_words:
                # Use the start_time of the first word in the group
                new_segment = Segment(
                    text=" ".join(w.word for w in current_words),
                    start_time=current_words[0].start_time,
                    end_time=current_words[-1].end_time,
                    word_timestamps=current_words,  # No need for copy
                    speaker=current_speaker,
                    # speaker_confidence=1.0,
                )
                new_segments.append(new_segment)

            # Start a new segment
            current_speaker = word.speaker
            current_words = [word]
        else:
            # Continue current segment
            current_words.append(word)

    # Add the last segment if there are remaining words
    if current_words:
        new_segment = Segment(
            text=" ".join(w.word for w in current_words),
            start_time=current_words[0].start_time,
            end_time=current_words[-1].end_time,
            word_timestamps=current_words,  # No need for copy
            speaker=current_speaker,
            # speaker_confidence=1.0,
        )
        new_segments.append(new_segment)

    # If we couldn't create any new segments, return original ones
    if not new_segments:
        return segments

    return new_segments


def get_terminal_punctuation() -> set[str]:
    """
    Get the terminal punctuation for all languages.
    """
    sterm_pattern = regex.compile(r"\p{STerm}")
    return {chr(i) for i in range(0x110000) if sterm_pattern.match(chr(i))}


def split_segments_into_sentences(segments: list[Segment]) -> list[Segment]:
    """
    Split segments into sentence-level segments based on terminal punctuation.

    Args:
        segments: List of segments to split into sentences

    Returns:
        List of sentence-level segments
    """
    if not segments:
        return []

    terminal_punctuation = get_terminal_punctuation()
    sentence_segments = []

    for segment in segments:
        # If segment has no word timestamps, keep it as is
        if not segment.word_timestamps:
            sentence_segments.append(segment)
            continue

        # Check if segment contains terminal punctuation
        segment_text = segment.text
        if not any(c in terminal_punctuation for c in segment_text):
            # No terminal punctuation, keep segment as is
            sentence_segments.append(segment)
            continue

        # Split segment into sentences based on word timestamps
        curr_word_buff = []

        for word in segment.word_timestamps:
            curr_word_buff.append(word)
            # Check if the last character of the word is terminal punctuation
            if (
                word.word
                and len(word.word.rstrip()) > 0
                and word.word.rstrip()[-1] in terminal_punctuation
            ):
                # Create sentence segment
                sentence_segment = Segment(
                    start_time=curr_word_buff[0].start_time,
                    end_time=curr_word_buff[-1].end_time,
                    text=" ".join([w.word for w in curr_word_buff]),
                    word_timestamps=curr_word_buff.copy(),
                    log_prob=segment.log_prob,
                    speaker=segment.speaker,
                    speaker_confidence=segment.speaker_confidence,
                    possible_hallucination=segment.possible_hallucination,
                    beam_results=segment.beam_results,
                    language_code=segment.language_code,
                    language_prob=segment.language_prob,
                )
                sentence_segments.append(sentence_segment)
                curr_word_buff = []

        # Handle remaining words (no terminal punctuation at the end)
        if curr_word_buff:
            remaining_segment = Segment(
                start_time=curr_word_buff[0].start_time,
                end_time=curr_word_buff[-1].end_time,
                text=" ".join([w.word for w in curr_word_buff]),
                word_timestamps=curr_word_buff.copy(),
                log_prob=segment.log_prob,
                speaker=segment.speaker,
                speaker_confidence=segment.speaker_confidence,
                possible_hallucination=segment.possible_hallucination,
                beam_results=segment.beam_results,
                language_code=segment.language_code,
                language_prob=segment.language_prob,
            )
            sentence_segments.append(remaining_segment)

    return sentence_segments


def merge_segments(*segments) -> Segment:
    """Merge multiple segments into one."""
    if not segments or not any(segments):
        return Segment(start_time=0, end_time=0, text="")

    valid_segments = [seg for seg in segments if seg is not None]
    if not valid_segments:
        return None

    def get_majority_language_code(segments: List[Segment]) -> Optional[str]:
        language_codes = [segment.language_code for segment in segments if segment.language_code]
        if not language_codes:
            return None
        return Counter(language_codes).most_common(1)[0][0]

    return Segment(
        start_time=min([segment.start_time for segment in valid_segments]),
        end_time=max([segment.end_time for segment in valid_segments]),
        text=" ".join([segment.text for segment in valid_segments]),
        word_timestamps=[word for segment in valid_segments for word in segment.word_timestamps],
        language_code=get_majority_language_code(valid_segments),
    )


def post_process_whisper_result(
    whisper_result: WhisperResult, post_processing_flags: PostProcessingFlags
) -> WhisperResult:
    """Post-process whisper result."""
    for segment in whisper_result.segments:
        if post_processing_flags.remove_hyphens_followed_by_space:
            segment.text = segment.text.replace("- ", "")
        if post_processing_flags.remove_spaces_in_ja_zh:
            if segment.language_code and segment.language_code in ["ja", "zh"]:
                segment.text = segment.text.replace(" ", "")
        if post_processing_flags.remove_non_speech_events_asterisk:
            segment.text = regex.sub(r"\*[^*]+\*", "", segment.text)
        if post_processing_flags.remove_non_speech_events_bracket:
            segment.text = regex.sub(r"\[[^\]]+\]", "", segment.text)
        if post_processing_flags.remove_non_speech_events_parenthesis:
            segment.text = regex.sub(r"\([^\)]+\)", "", segment.text)
    return whisper_result


def copy_timing_info(timing_info: ASRTimingInfo, source_timing: ASRTimingInfo) -> None:
    """Copy timing information from source ASRTimingInfo to target ASRTimingInfo."""
    for attr_name in source_timing.__dict__:
        if hasattr(timing_info, attr_name) and getattr(source_timing, attr_name) is not None:
            # Special handling for durations: merge fields instead of replacing the entire object
            if attr_name == "durations":
                source_durations = getattr(source_timing, attr_name)
                target_durations = getattr(timing_info, attr_name)
                # Copy only non-None duration fields from source to target
                for duration_field in source_durations.__dict__:
                    if hasattr(source_durations, duration_field):
                        source_value = getattr(source_durations, duration_field)
                        if source_value is not None:
                            setattr(target_durations, duration_field, source_value)
            else:
                setattr(timing_info, attr_name, getattr(source_timing, attr_name))


def apply_diarization_config_overrides(whisper_input: WhisperInput) -> WhisperInput:
    """
    Applies configuration overrides to the WhisperInput object based on the diarization assignment mode.

    If the assignment mode is set to WORD or SENTENCE, ensures that `show_word_timestamps` is enabled.

    Parameters:
        whisper_input (WhisperInput): The input object containing whisper parameters to be modified.

    Returns:
        WhisperInput: The modified WhisperInput object with updated configuration.
    """
    from whisper_utils.data_types import DiarizationGranularity

    mode = whisper_input.whisper_params.diarization_granularity

    if mode in (DiarizationGranularity.WORD, DiarizationGranularity.SENTENCE):
        logger.info(
            f"Using {mode.value}-level diarization. Setting `show_word_timestamps` to `true`"
        )
        if not whisper_input.whisper_params.show_word_timestamps:
            whisper_input.whisper_params.show_word_timestamps = True

    return whisper_input


def build_diarization_stack(diarization_result_data):
    """
    Converts a diarization result into a stack of data structures for speaker assignment.

    Args:
        diarization_result_data: An iterable with diarization tracks, expected to support
            itertracks(yield_label=True), yielding (turn, _, speaker) tuples where
            turn has 'start' and 'end' attributes and speaker is a label.

    Returns:
        tuple: A tuple containing:
            - diarization_data_list (list of dict): Each dict has keys 'speaker', 'start', 'end'.
            - diarize_speakers (list): List of speaker labels.
            - diarize_starts (list): List of start times for each segment.
            - diarize_ends (list): List of end times for each segment.
    """
    diarization_data_list = []
    diarize_speakers = []
    diarize_starts = []
    diarize_ends = []

    for turn, _, speaker in diarization_result_data.itertracks(yield_label=True):
        diarization_data_list.append({"speaker": speaker, "start": turn.start, "end": turn.end})
        diarize_speakers.append(speaker)
        diarize_starts.append(turn.start)
        diarize_ends.append(turn.end)

    return (diarization_data_list, diarize_speakers, diarize_starts, diarize_ends)
