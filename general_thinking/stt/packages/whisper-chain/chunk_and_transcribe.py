import asyncio
import logging
import sys
import time
from pathlib import Path

# Path setup - MUST be before chainlet imports
_BASETEN_WHISPER_ROOT = Path(__file__).resolve().parent.parent.parent
_WHISPER_UTILS_LIB = _BASETEN_WHISPER_ROOT / "whisper-utils"
_LOCAL_SILERO_VAD_LIB = _BASETEN_WHISPER_ROOT / "silero-vad" / "src"
_ASR_CHAINS_LIB = _BASETEN_WHISPER_ROOT / "asr-chains"
sys.path.append(str(_WHISPER_UTILS_LIB))
sys.path.append(str(_ASR_CHAINS_LIB))

import chainlets.whisper_chainlet
import truss_chains as chains
from whisper_chain_utils.whisper_chain_data_types import WhisperChainletInput
from whisper_utils.constants import (
    CONCURRENCY_LIMIT_FROM_CHUNKER_CHAINLET,
    DEFAULT_CHUNKER_GPU_FOR_CHAIN,
)
from whisper_utils.data_types import PostProcessingFlags, WhisperInput, WhisperResult

# Configure the logging
logger = logging.getLogger(__name__)


@chains.mark_entrypoint("ASR Chain - Whisper Large V3")
class Chunker(chains.ChainletBase):
    remote_config = chains.RemoteConfig(
        docker_image=chains.DockerImage(
            apt_requirements=["ffmpeg", "libsox-dev", "sox", "wget"],
            pip_requirements=[
                "pandas==2.2.3",
                "torch==2.4.1",
                "torchaudio==2.4.1",
                "ffmpeg-python==0.2.0",
                "numpy==1.26.4",
                "httpx==0.27.2",
                "regex==2025.9.18",
            ],
            external_package_dirs=[
                chains.make_abs_path_here(str(_LOCAL_SILERO_VAD_LIB)),
                chains.make_abs_path_here(str(_WHISPER_UTILS_LIB)),
                chains.make_abs_path_here(str(_ASR_CHAINS_LIB)),
            ],
        ),
        compute=chains.Compute(
            gpu=DEFAULT_CHUNKER_GPU_FOR_CHAIN,
            predict_concurrency=128,
        ),
        options=chains.ChainletOptions(
            enable_debug_logs=False,
            enable_b10_tracing=False,
        ),
    )
    _context: chains.DeploymentContext

    _whisper: chainlets.whisper_chainlet.Transcriber

    def __init__(
        self,
        whisper: chainlets.whisper_chainlet.Transcriber = chains.depends(
            chainlets.whisper_chainlet.Transcriber,
            retries=3,
            use_binary=True,
            concurrency_limit=CONCURRENCY_LIMIT_FROM_CHUNKER_CHAINLET,
        ),
        context: chains.DeploymentContext = chains.depends_context(),
    ) -> None:
        # Store parameters without importing dependencies
        self._context = context
        self._whisper = whisper

        # Initialize attributes that will be set when methods are called
        self.vad_model = None
        self.client = None
        self.audio_length_histogram_seconds = None
        self.chunk_length_histogram_seconds = None
        self._initialized = False

    def _ensure_initialized(self):
        """Initialize dependencies only when needed"""
        if self._initialized:
            return

        # Import dependencies inside the method
        import httpx
        import torch
        from prometheus_client import Histogram
        from silero_vad import load_cpu_model, load_gpu_model
        from whisper_utils.constants import (
            AUDIO_DOWNLOAD_TIMEOUT,
            AUDIO_LENGTH_BUCKETS_SECONDS,
            CHUNK_LENGTH_BUCKETS_SECONDS,
        )

        # Add silero-vad path to sys.path for local development
        sys.path.append(str(_LOCAL_SILERO_VAD_LIB))

        self.vad_model = load_gpu_model() if torch.cuda.is_available() else load_cpu_model()
        self.client = httpx.AsyncClient(timeout=AUDIO_DOWNLOAD_TIMEOUT)

        self.audio_length_histogram_seconds = Histogram(
            "audio_length_histogram_seconds",
            "Histogram of audio lengths (seconds)",
            buckets=AUDIO_LENGTH_BUCKETS_SECONDS,
        )
        self.chunk_length_histogram_seconds = Histogram(
            "chunk_length_histogram_seconds",
            "Histogram of chunk lengths (seconds)",
            buckets=CHUNK_LENGTH_BUCKETS_SECONDS,
        )

        self._initialized = True

    async def call_whisper_detect_language(
        self, audio_wav, whisper_params, asr_options: dict
    ) -> WhisperResult:
        # Ensure dependencies are initialized
        self._ensure_initialized()

        whisper_params = whisper_params.model_copy(deep=True)
        whisper_params.language_detection_only = True
        whisper_chainlet_input = WhisperChainletInput(
            audio_wav=audio_wav.numpy(),
            whisper_params=whisper_params,
            asr_options=asr_options,
        )
        whisper_transcribe_result = await self._whisper.run_remote(whisper_chainlet_input)
        return whisper_transcribe_result

    async def call_whisper_transcribe(
        self,
        audio_wav,
        whisper_input,
        fallback_chunks,
        seg_info,
        audio_language: str,
    ) -> WhisperResult:
        # Ensure dependencies are initialized
        self._ensure_initialized()

        whisper_chainlet_input = WhisperChainletInput(
            seg_info=seg_info,
            audio_wav=audio_wav.numpy(),
            asr_options=whisper_input.asr_options,
            fallback_chunks=fallback_chunks,
            whisper_params=whisper_input.whisper_params,
        )
        whisper_chainlet_input.whisper_params.audio_language = audio_language
        whisper_transcribe_result = await self._whisper.run_remote(whisper_chainlet_input)

        return whisper_transcribe_result

    async def run_remote(
        self,
        whisper_input: WhisperInput,
    ) -> WhisperResult:
        # Ensure dependencies are initialized
        self._ensure_initialized()
        from silero_vad.utils_vad import collect_chunks
        from whisper_utils.constants import (
            DEFAULT_HTTP_POST_PROCESSING_FLAGS,
            VAD_CONFIG_OVERRIDES,
            WHISPER_SAMPLE_RATE,
        )
        from whisper_utils.data_types import ASRTimingInfo, MicroChunkInfo, WhisperResult
        from whisper_utils.utilities import (
            detect_audio_language,
            load_audio,
            log_time,
            perform_vad_chunking,
            post_process_whisper_result,
            update_segments_timestamps,
        )

        whisper_input = WhisperInput.model_validate(whisper_input, strict=True)

        start_time = time.time()

        logger.info(f"Received whisper_input: {whisper_input}")

        if not whisper_input.whisper_params.enable_vad:
            logger.warning(
                "Setting whisper_input.whisper_params.enable_vad to True to use ASR Chain"
            )
            whisper_input.whisper_params.enable_vad = True

        tasks = []
        seg_infos = []
        timing_info = ASRTimingInfo()

        # audio preprocessing
        timing_info.start_read_audio_time = time.time()
        wav = await load_audio(whisper_input, self.client)
        duration_secs = len(wav) / WHISPER_SAMPLE_RATE
        logger.info(f"Loaded audio of `{duration_secs:.1f}` seconds.")
        self.audio_length_histogram_seconds.observe(duration_secs)
        timing_info.end_read_audio_time = time.time()
        timing_info.durations.read_audio_duration_s = (
            timing_info.end_read_audio_time - timing_info.start_read_audio_time
        )

        # VAD chunking
        timing_info.start_chunking_time = time.time()
        speech_timestamps, fallback_chunks = perform_vad_chunking(
            wav,
            self.vad_model,
            VAD_CONFIG_OVERRIDES,
            whisper_input.vad_config,
            whisper_input.whisper_params.use_missing_chunk_retry,
        )
        if not speech_timestamps:
            logger.warning(
                f"VAD did not detect any speech segments returning empty result. The audio might be silent or contain only background noise. Audio url: {whisper_input.audio.url}"
            )
            return WhisperResult()
        timing_info.end_chunking_time = time.time()
        timing_info.durations.chunking_duration_s = (
            timing_info.end_chunking_time - timing_info.start_chunking_time
        )

        # language detection
        audio_language = whisper_input.whisper_params.audio_language
        audio_language_prob = None
        if audio_language == "auto":
            if whisper_input.whisper_params.enable_chunk_level_language_detection:
                logger.info("Chunk level language detection enabled")
            else:
                timing_info.start_language_detection_time = time.time()
                language_result = await detect_audio_language(
                    audio_wav=wav,
                    whisper_params=whisper_input.whisper_params,
                    asr_options=whisper_input.asr_options,
                    speech_timestamps=speech_timestamps,
                    collect_chunks_fn=collect_chunks,
                    call_whisper_detect_language=self.call_whisper_detect_language,
                )
                audio_language = language_result.language_code
                audio_language_prob = language_result.language_prob
                timing_info.end_language_detection_time = time.time()
                timing_info.durations.language_detection_duration_s = (
                    timing_info.end_language_detection_time
                    - timing_info.start_language_detection_time
                )
                logger.info(
                    f"Detected language: {audio_language} with prob {audio_language_prob} in {timing_info.end_language_detection_time - timing_info.start_language_detection_time:.2f} seconds"
                )
                if whisper_input.whisper_params.language_detection_only:
                    language_result.timing_info = (
                        timing_info.durations if whisper_input.include_timing_info else None
                    )
                    return language_result
        elif (
            whisper_input.whisper_params.language_detection_only
            or whisper_input.whisper_params.enable_chunk_level_language_detection
        ):
            logger.warning(
                "Please set 'audio_language' to 'auto' to enable 'language_detection_only' or 'enable_chunk_level_language_detection'"
            )

        # transcribe audio chunks in parallel
        timing_info.start_transcription_time = time.time()
        for i, sts in enumerate(speech_timestamps):
            audio_chunk = collect_chunks([sts], wav)
            seg_info = MicroChunkInfo(
                start_time_sec=sts["start"] / WHISPER_SAMPLE_RATE,
                end_time_sec=sts["end"] / WHISPER_SAMPLE_RATE,
                duration_sec=sts["end"] / WHISPER_SAMPLE_RATE - sts["start"] / WHISPER_SAMPLE_RATE,
                chunk_index=i,
            )
            self.chunk_length_histogram_seconds.observe(seg_info.duration_sec)

            tasks.append(
                asyncio.ensure_future(
                    self.call_whisper_transcribe(
                        audio_wav=audio_chunk,
                        whisper_input=whisper_input,
                        fallback_chunks=fallback_chunks[i],
                        seg_info=seg_info,
                        audio_language=audio_language,
                    )
                )
            )
            seg_infos.append(seg_info)

            await asyncio.sleep(0)

        logger.debug("Waiting for all transcription tasks to complete")
        results = await asyncio.gather(*tasks)
        logger.debug("All transcription tasks completed")
        timing_info.end_transcription_time = time.time()
        timing_info.durations.transcription_duration_s = (
            timing_info.end_transcription_time - timing_info.start_transcription_time
        )

        segments = update_segments_timestamps(results, seg_infos)

        log_time(timing_info, duration_secs)

        timing_info.durations.total_duration_s = time.time() - start_time

        return post_process_whisper_result(
            WhisperResult(
                segments=segments,
                language_code=audio_language,
                language_prob=audio_language_prob,
                timing_info=timing_info.durations if whisper_input.include_timing_info else None,
                enable_chunk_level_language_detection=whisper_input.whisper_params.enable_chunk_level_language_detection,
                audio_length_sec=duration_secs,
            ),
            DEFAULT_HTTP_POST_PROCESSING_FLAGS,
        )
