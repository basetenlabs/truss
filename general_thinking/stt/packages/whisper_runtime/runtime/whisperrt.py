import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import ctranslate2
import numpy as np
import tokenizers
import torch
from huggingface_hub import snapshot_download
from opencc import OpenCC
from whisper_utils.data_types import WhisperSamplingParams

from ..audio import LogMelSpectrogram
from ..chunker.chunker import Chunker
from ..configs import SAMPLE_RATE, TIME_PRECISION
from ..data import BasicSegmenter, WhisperAudioProcessor
from .tokenizer import Tokenizer
from .trt_model import WhisperTRTModel
from .utils import array_to_mono_tensor, get_whisper_engine_repo_id, get_word_aligner_model_repo_id

# Configure the logging
logger = logging.getLogger(__name__)

WORD_ALIGNER_MODEL_N_MELS_DICT = {
    "tiny": 80,
}


class WhisperRTConfig:
    """
    Configuration class for the WhisperRT model.

    Attributes:
        end_id (int): End-of-sequence token ID.
        pad_id (int): Padding token ID.
        max_new_tokens (int): Maximum number of new tokens to generate.
        beam_width (int): Beam width for beam search.
        length_penalty (float): Length penalty for beam search.
        repetition_penalty (float): Penalty for token repetition.
        beam_search_diversity_rate (float): Beam search diversity rate.
        no_repeat_ngram_size (int): No repeat ngram size.
        compression_ratio_threshold (float): Threshold for compression ratio.
        log_prob_threshold (float): Threshold for log probabilities.
        suppress_tokens (List[List[int]]): Tokens to suppress during generation.
        show_word_timestamps (bool): Whether to include word-level timestamps.
        sampling_temperatures (Tuple[float, ...]): Sampling temperatures for retries.
        word_aligner_model (str): Model to use for word alignment.
        audio_language (str): Language of the input audio if known. Set to `auto` for variable audio languages.
    """

    def __init__(
        self,
        end_id: int,
        pad_id: int,
        max_new_tokens: int,
        beam_width: int = 1,
        length_penalty: float = 2.0,
        repetition_penalty: float = 1.01,
        beam_search_diversity_rate: float = None,
        no_repeat_ngram_size: int = None,
        compression_ratio_threshold: float = 2.4,
        log_prob_threshold: float = -1.0,
        suppress_tokens: List[List[int]] = [[-1]],
        sampling_temperatures: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        word_aligner_model: str = "tiny",
    ):
        self.end_id = end_id
        self.pad_id = pad_id
        self.max_new_tokens = max_new_tokens
        self.beam_width = beam_width
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty
        self.beam_search_diversity_rate = beam_search_diversity_rate
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.compression_ratio_threshold = compression_ratio_threshold
        self.log_prob_threshold = log_prob_threshold
        self.suppress_tokens = suppress_tokens
        self.sampling_temperatures = sampling_temperatures
        self.word_aligner_model = word_aligner_model
        self.word_aligner_model_n_mels = WORD_ALIGNER_MODEL_N_MELS_DICT[word_aligner_model]

    def __str__(self):
        return "\n".join([f"{key}: {value}" for key, value in self.__dict__.items()])


class WhisperRT:
    # Chinese language variants that need to be normalized to "zh" for Whisper
    _CHINESE_VARIANTS = {"zh-hant", "zh-hans"}

    def __init__(
        self,
        model_name: str,
        vad_model=None,
        device: str = "cuda",
        device_index: int = 0,
        compute_type: str = "float16",
        merge_chunks: bool = True,
        dta_padding: float = 3.0,
        use_dynamic_time_axis: bool = False,
        max_speech_len: float = 29.0,
        max_text_token_len: int = 448,
        speech_segmenter_options: Optional[Dict] = None,
        cpu_threads: int = 4,
        num_workers: int = 1,
        use_vad: bool = False,
        max_batch_size: int = 6,
        max_beam_width: int = 5,
        enable_word_timestamp: bool = True,
        hf_token: Optional[str] = None,
        model_repo_id_override: Optional[str] = None,
    ):
        """
        Initializes the WhisperRT model.

        Args:
            model_name (str): Whisper Model version.
            vad_model: Voice activity detection model.
            device (str): Device to run the model ('cpu' or 'cuda').
            device_index (int): Device index for GPU.
            compute_type (str): Compute precision type ('float16', 'float32', etc.).
            merge_chunks (bool): Whether to merge chunks during processing.
            dta_padding (float): Padding for dynamic time axis in seconds.
            use_dynamic_time_axis (bool): Use dynamic time axis for processing.
            max_speech_len (float): Maximum speech segment length in seconds.
            max_text_token_len (int): Maximum number of text tokens per segment.
            speech_segmenter_options (Dict): Options for the speech segmenter.
            cpu_threads (int): Number of CPU threads to use.
            num_workers (int): Number of worker processes.
            use_vad (bool): Whether to use voice activity detection for segmentation.
            max_batch_size (int): Maximum batch size for processing.
            max_beam_width (int): Maximum beam size for beam search.
            hf_token (Optional[str]): Hugging Face token for downloading models.
        """
        logger.info(f"Initializing WhisperRT with model: {model_name}")

        if speech_segmenter_options is None:
            speech_segmenter_options = {}

        self.max_batch_size = max_batch_size

        # download whisper model engine
        try:
            whisper_engine_repo_id = model_repo_id_override or get_whisper_engine_repo_id(
                model_name=model_name
            )
            self.model_path = snapshot_download(whisper_engine_repo_id, token=hf_token)
            logger.info(f"Whisper engine: {whisper_engine_repo_id}")
            logger.debug(f"Whisper engine downloaded to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to download Whisper engine: {e}")
            raise e

        # initialize whisper model
        tokenizer_file = os.path.join(self.model_path, "tokenizer.json")
        logger.debug(f"Loading tokenizer from: {tokenizer_file}")
        self.tokenizer = Tokenizer(tokenizers.Tokenizer.from_file(tokenizer_file))
        self.device = device
        logger.debug(f"Using device: {self.device}")
        # Get the n_mels and dtype from encoder config.json
        try:
            config_path = Path(self.model_path) / "encoder" / "config.json"
            with open(config_path, "r") as f:
                self.encoder_config = json.load(f)
        except FileNotFoundError as e:
            logger.error(
                f"Failed to initialize WhisperRT. Configuration file not found at {config_path}"
            )
            raise e
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to initialize WhisperRT. Error decoding JSON from the configuration file at {config_path}"
            )
            raise e
        self.n_mels = self.encoder_config["pretrained_config"]["n_mels"]
        self.dtype = self.encoder_config["pretrained_config"]["dtype"]
        self.model = WhisperTRTModel(
            self.model_path,
            self.n_mels,
            self.dtype,
            self.tokenizer,
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
        )
        self.merge_chunks = merge_chunks
        self.max_speech_len = max_speech_len
        self.dta_padding = dta_padding
        self.use_dynamic_time_axis = use_dynamic_time_axis
        self.max_text_token_len = max_text_token_len
        self.enable_word_timestamp = enable_word_timestamp
        self.config = WhisperRTConfig(
            end_id=self.tokenizer.eot,
            pad_id=self.tokenizer.eot,
            max_new_tokens=self.max_text_token_len,
        )

        if self.enable_word_timestamp:
            # download word aligner model
            try:
                word_aligner_model_repo_id = get_word_aligner_model_repo_id(
                    model_name=self.config.word_aligner_model
                )
                self.aligner_model_path = snapshot_download(
                    word_aligner_model_repo_id, token=hf_token
                )
                logger.info(
                    f"Word aligner model: {word_aligner_model_repo_id} downloaded to {self.aligner_model_path}"
                )
            except Exception as e:
                logger.error(f"Failed to download Word aligner model: {e}")
                raise e
            # initialize word aligner model
            try:
                logger.info("Initializing word aligner model")
                self.aligner_model = ctranslate2.models.Whisper(
                    self.aligner_model_path,
                    device=device,
                    device_index=device_index,
                    compute_type=compute_type,
                    intra_threads=cpu_threads,
                    inter_threads=num_workers,
                )
            except Exception as e:
                logger.error(f"Failed to initialize word aligner model: {e}")
                raise e

        # Rescaled Params
        self.dta_padding_samples = int(self.dta_padding * SAMPLE_RATE)
        self.max_initial_prompt_len = self.max_text_token_len // 2 - 1

        self.preprocessor = LogMelSpectrogram(n_mels=self.n_mels).to(self.device)
        self.aligner_preprocessor = LogMelSpectrogram(
            n_mels=self.config.word_aligner_model_n_mels
        ).to(
            self.device
        )  # TODO: automatically get n_mels from model config

        # use_vad flag for VAD-based segmentation or no segmentation (if audio is already segmented)
        if use_vad:
            self.vad_model = vad_model
            self.speech_segmenter_options = speech_segmenter_options
            self.speech_segmenter_options["max_seg_len"] = self.max_speech_len
            self.speech_segmenter = Chunker(
                self.vad_model, device=self.device, **self.speech_segmenter_options
            )
        else:
            self.speech_segmenter = BasicSegmenter()

        self.audio_processor = WhisperAudioProcessor(
            self.device,
            self.tokenizer,
            speech_segmenter=self.speech_segmenter,
            dta_padding=self.dta_padding,
            max_speech_len=self.max_speech_len,
            max_initial_prompt_len=self.max_initial_prompt_len,
            use_dynamic_time_axis=self.use_dynamic_time_axis,
            merge_chunks=self.merge_chunks,
            detect_language_fn=self._get_language,
            enable_word_timestamp=self.enable_word_timestamp,
        )
        self.chinese_t2s_converter = OpenCC("t2s")
        self.chinese_s2t_converter = OpenCC("s2t")

    @torch.no_grad()
    def transcribe(
        self,
        audio_wav: Union[torch.tensor, np.ndarray],
        asr_options: Dict = {},
        whisper_sampling_params: Optional[WhisperSamplingParams] = None,
        output_language: Optional[str] = "en",
        audio_language: Optional[str] = "en",
        language_detection_only: bool = False,
        language_options: Optional[List[str]] = [],
        prompt: Optional[str] = None,
        show_word_timestamps: bool = False,
        begin_padding_seconds: float = 0.0,
        prefix: Optional[str] = None,
        show_beam_results: bool = False,
        _bypass_vad: bool = True,
    ) -> List[Dict]:
        """
        Transcribes the given audio files.

        Args:
            audio_wav (List[Union[torch.tensor, np.ndarray]]): List of 1D audio tensors
            output_language (Optional[str]): Desired language of transcript (defaults to English).
            audio_language (Optional[str]): Language of the input audio if known. Set to "auto" to detect automatically (defaults to English).
            language_detection_only (bool): If True, only performs language detection.
            language_options (Optional[List[str]]): List of languages to consider for language detection.
            prompt (Optional[str]): Initial prompt for Whisper.
            show_word_timestamps (bool): Whether to show word level timestamps.
            begin_padding_seconds (float): Padding duration in seconds to add at the beginning of each segment.
            prefix (Optional[str]): Prefix for whisper (for more info see https://github.com/openai/whisper/discussions/117#discussioncomment-3727051)
            show_beam_results (bool): Whether to show the top N candidates.
            _bypass_vad (bool): Whether to bypass VAD. Only set this to True when you know the audio is <30 seconds (mainly used for missing chunk retry)
        Returns:
            List[Dict]: List of dictionaries containing transcription results.
        """
        logger.debug(
            f"Transcribing with config: {asr_options=}, {whisper_sampling_params=}, {output_language=}, {audio_language=}, {language_detection_only=}, {language_options=}, {prompt=}, {show_word_timestamps=}, {begin_padding_seconds=}, {_bypass_vad=}, {prefix=}, {show_beam_results=}"
        )

        if asr_options:
            logger.warning(
                "whisper_input.asr_options is being deprecated, use whisper_input.whisper_params.whisper_sampling_params instead"
            )
            if "beam_size" in asr_options:
                asr_options["beam_width"] = asr_options["beam_size"]
            self._update_whisper_config(asr_options)
            logger.debug(f"Updated WhisperRTConfig with asr_options: {asr_options}")
        elif whisper_sampling_params:
            self._update_whisper_config(whisper_sampling_params)
            logger.debug(
                f"Updated WhisperRTConfig with whisper_sampling_params: {whisper_sampling_params}"
            )

        logger.debug(f"Using WhisperRTConfig:\n{self.config}")

        # deal with multichannel audio
        audio_wav = array_to_mono_tensor(audio_wav)

        target_language = audio_language
        if audio_language == "auto":
            logger.info("Detecting language of the audio files")
            # Normalize language_options: convert zh-hant/zh-hans to zh for language detection
            # (Whisper only recognizes "zh" as a language code, not zh-hant/zh-hans)
            normalized_language_options = []
            if language_options:
                language_options_set = set()
                for lang in language_options:
                    normalized_lang = "zh" if lang in self._CHINESE_VARIANTS else lang
                    language_options_set.add(normalized_lang)
                normalized_language_options = list(language_options_set)
            lang_code, lang_probs = self.audio_processor.detect_language(
                audio_wav, normalized_language_options
            )
            logger.debug(f"language detection result: {lang_code=} {lang_probs=}")
            lang_prob = lang_probs[lang_code]
            # Determine tasks, lang_codes, and language_probs based on detected language and user input
            task = "translate" if output_language == "en" and lang_code != "en" else "transcribe"
            if language_detection_only:
                return {
                    "language_code": lang_code,
                    "language_prob": lang_prob,
                    "segments": [
                        {
                            "language_code": lang_code,
                            "language_prob": lang_prob,
                            "start_time": 0.0,
                            "end_time": len(audio_wav) / SAMPLE_RATE,
                            "text": "",
                        }
                    ],
                }
        else:
            if audio_language in ["zh-hant", "zh-hans"]:
                audio_language = "zh"
            task = (
                "translate" if output_language == "en" and audio_language != "en" else "transcribe"
            )
            lang_code = audio_language
            lang_prob = None

        self.tokenizer.task = task
        self.tokenizer.language = lang_code

        transcribed_segments = []

        all_segments = self.audio_processor.process_audio_file(
            audio_wav=audio_wav,
            lang_code=lang_code,
            task=task,
            prompt=prompt,
            show_word_timestamps=show_word_timestamps,
            prefix=prefix,
            _bypass_vad=_bypass_vad,
        )

        # Prepare the data by batching all segments together after processing
        signals, prompts, seq_lens, seg_metadatas = self.audio_processor.data_collate_fn(
            all_segments
        )
        total_segments = len(signals)

        # TODO: Implement batching for larger number of segments (truss)
        batch_size = 1

        for idx in range(0, total_segments, batch_size):
            batch_signals = signals[idx : idx + batch_size]
            batch_prompts = prompts[idx : idx + batch_size]
            batch_seq_lens = seq_lens[idx : idx + batch_size]
            batch_seg_metadatas = seg_metadatas[idx : idx + batch_size]

            mels, batch_seq_lens = self.preprocessor(
                batch_signals, batch_seq_lens, begin_padding_seconds
            )
            aligner_mels, _ = self.aligner_preprocessor(batch_signals, batch_seq_lens)
            res = self.generate_segment_batched(
                mels.to(self.device),
                aligner_mels.to(self.device),
                batch_prompts,
                batch_seq_lens,
                batch_seg_metadatas,
                show_word_timestamps,
                show_beam_results,
            )

            for res_idx, _seg_metadata in enumerate(batch_seg_metadatas):
                segment_response = {
                    **res[res_idx],
                    "start_time": round(_seg_metadata["start_time"], 3),
                    "end_time": round(_seg_metadata["end_time"], 3),
                    "language_code": lang_code,
                    "language_prob": lang_prob,
                }
                if not isinstance(segment_response["log_prob"], float):
                    logger.error(f"Log probability is not a float: {segment_response['log_prob']}")
                transcribed_segments.append(segment_response)

        if target_language == "zh-hant":
            for segment in transcribed_segments:
                segment["text"] = self.chinese_s2t_converter.convert(segment["text"])
        elif target_language == "zh-hans":
            for segment in transcribed_segments:
                segment["text"] = self.chinese_t2s_converter.convert(segment["text"])
        elif (target_language == "auto" or target_language == "zh") and lang_code == "zh":
            # Apply Chinese script conversion from language_options
            # HTTP truss & chain runs detection first, causing target_language == "zh"
            is_simplified = language_options is not None and "zh-hans" in language_options
            is_traditional = language_options is not None and "zh-hant" in language_options

            if is_simplified and is_traditional:
                logger.warning(
                    "Both zh-hans and zh-hant were set. Please choose only one. Skipping script conversion..."
                )
            elif is_simplified:
                for segment in transcribed_segments:
                    segment["text"] = self.chinese_t2s_converter.convert(segment["text"])
            elif is_traditional:
                for segment in transcribed_segments:
                    segment["text"] = self.chinese_s2t_converter.convert(segment["text"])
        response = {
            "language_code": lang_code,
            "language_prob": lang_prob,
            "segments": transcribed_segments,
        }

        return response

    def warmup(self):
        self.transcribe(
            audio_wav=torch.randn(1, 16000),
            asr_options={},
            whisper_sampling_params=None,
            output_language="en",
            audio_language="en",
            language_detection_only=False,
            language_options=[],
            prompt="Hello, how are you?",
            show_word_timestamps=True if self.enable_word_timestamp else False,
            begin_padding_seconds=0.0,
            prefix=None,
            _bypass_vad=True,
        )

    def generate_segment_batched(
        self,
        features: torch.Tensor,
        aligner_features: torch.Tensor,
        prompts: List[List[int]],
        seq_lens: torch.Tensor,
        seg_metadata: List[Dict],
        show_word_timestamps: bool,
        show_beam_results: bool,
    ) -> List[Dict]:
        """
        Generates transcriptions for a batch of audio segments.

        Args:
            features (torch.Tensor): Mel spectrogram features.
            aligner_features (torch.Tensor): Mel spectrogram features for aligner.
            prompts (List[List[int]]): List of token prompts for each segment.
            seq_lens (torch.Tensor): Sequence lengths for each segment.
            seg_metadata (List[Dict]): Metadata for each segment.
            show_word_timestamps (bool): Whether to show word-level timestamps.
            show_beam_results (bool): Whether to show the top N candidates.
        Returns:
            List[Dict]: List of transcription results for each segment.
        """
        # logger.debug(f"Generating transcription for batch of {len(features)} segments")
        (
            batched_results,
            texts,
            possible_hallucination,
            beam_results,
        ) = self.model.generate_with_temperature_fallback(
            features=features,
            prompts=prompts,
            beam_width=self.config.beam_width,
            repetition_penalty=self.config.repetition_penalty,
            length_penalty=self.config.length_penalty,
            beam_search_diversity_rate=self.config.beam_search_diversity_rate,
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,
            compression_ratio_threshold=self.config.compression_ratio_threshold,
            log_prob_threshold=self.config.log_prob_threshold,
            sampling_temperatures=self.config.sampling_temperatures,
            pad_id=self.config.pad_id,
            end_id=self.config.end_id,
            suppress_tokens=self.config.suppress_tokens,
            max_new_tokens=self.config.max_new_tokens,
            show_beam_results=show_beam_results,
        )

        response = []
        for idx in range(batched_results.number_of_segments):
            text = texts[idx].strip()
            beam_results_stripped = [beam_result.strip() for beam_result in beam_results]
            response.append(
                {
                    "text": text,
                    "log_prob": batched_results.cum_log_probs_aligned[idx][0],
                    "possible_hallucination": possible_hallucination,
                    "beam_results": beam_results_stripped,
                }
            )

        # logger.debug(f"Before word alignment: {response=}")

        if show_word_timestamps:
            if not self.enable_word_timestamp:
                raise ValueError(
                    "Please set enable_word_timestamp to True to enable `show_word_timestamps`"
                )
            text_tokens = [
                [
                    _t
                    for _t in output_tokens[0][len(prompt_tokens) :]
                    if _t < self.tokenizer.eot and _t >= 0
                ]
                + [self.tokenizer.eot]
                for prompt_tokens, output_tokens in zip(
                    prompts, batched_results.output_tokens_aligned
                )
            ]
            sot_seqs = [tuple(_[-4:]) for _ in prompts]
            try:
                word_timings = self._align_words(
                    aligner_features, texts, text_tokens, sot_seqs, seq_lens, seg_metadata
                )
                for _response, _word_timings in zip(response, word_timings):
                    _response["word_timestamps"] = _word_timings
            except Exception as e:
                logger.error(f"Error during word alignment: {e}")
                raise RuntimeError(f"Error during word alignment: {e}")

        # logger.debug(f"After word alignment: {response=}")
        return response

    # Helpers
    def _update_whisper_config(self, config_overrides: Dict | WhisperSamplingParams):
        if isinstance(config_overrides, Dict):
            for key, value in config_overrides.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        elif isinstance(config_overrides, WhisperSamplingParams):
            for key, value in config_overrides.model_dump().items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

    def _get_language(
        self,
        features: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> Tuple[List[str], List[Dict[str, float]]]:
        """
        Detect the language of the audio features.
        """
        features, _ = self.preprocessor(features.to(self.device), seq_len)
        languages, language_probs = self.model.detect_language(
            features=features,
            beam_width=self.config.beam_width,
            repetition_penalty=self.config.repetition_penalty,
            length_penalty=self.config.length_penalty,
            beam_search_diversity_rate=self.config.beam_search_diversity_rate,
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,
            compression_ratio_threshold=self.config.compression_ratio_threshold,
            log_prob_threshold=self.config.log_prob_threshold,
            sampling_temperatures=self.config.sampling_temperatures,
            pad_id=self.config.pad_id,
            end_id=self.config.end_id,
            suppress_tokens=self.config.suppress_tokens,
        )
        return languages, language_probs

    def _assign_word_timings(
        self,
        alignments: List[Tuple[int, int]],
        text_token_probs: np.ndarray,
        words: List[str],
        word_tokens: List[List[int]],
    ) -> List[Dict]:
        """
        Assigns timing information to each word based on alignments.

        Args:
            alignments (List[Tuple[int, int]]): List of aligned token and time indices.
            text_token_probs (np.ndarray): Probabilities of each text token.
            words (List[str]): List of words in the transcript.
            word_tokens (List[List[int]]): Tokens corresponding to each word.

        Returns:
            List[Dict]: Timing information for each word.
        """

        text_indices = np.array([pair[0] for pair in alignments])
        time_indices = np.array([pair[1] for pair in alignments])

        if len(word_tokens) <= 1:
            return []

        word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
        if len(word_boundaries) <= 1:
            return []

        jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
        jump_times = time_indices[jumps] * TIME_PRECISION
        start_times = jump_times[word_boundaries[:-1]]
        end_times = jump_times[word_boundaries[1:]]
        word_probs = [
            np.mean(text_token_probs[i:j])
            for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
        ]

        return [
            dict(
                word=word,
                start_time=round(start, 2),
                end_time=round(end, 2),
                prob=round(prob, 2),
            )
            for word, start, end, prob in zip(words, start_times, end_times, word_probs)
        ]

    def _align_words(
        self,
        aligner_features: torch.Tensor,
        texts: List[str],
        text_tokens: List[List[int]],
        sot_seqs: List[Tuple[int]],
        seq_lens: torch.Tensor,
        seg_metadata: List[Dict],
    ) -> List[List[Dict]]:
        """
        Aligns words with audio features to obtain word-level timing information.

        Args:
            aligner_features (torch.Tensor): Mel spectrogram features.
            texts (List[str]): Transcribed texts for each segment.
            text_tokens (List[List[int]]): Token sequences for each text.
            sot_seqs (List[Tuple[int]]): Start-of-transcript token sequences.
            seq_lens (torch.Tensor): Sequence lengths for each segment.
            seg_metadata (List[Dict]): Metadata for each segment.

        Returns:
            List[List[Dict]]: Word-level timing information for each segment.
        """
        # logger.info("Aligning words with audio features")
        lang_codes = [_["lang_code"] for _ in seg_metadata]
        word_tokens = self.tokenizer.split_to_word_tokens_batch(texts, text_tokens, lang_codes)

        start_seq_wise_req = {}
        for _idx, _sot_seq in enumerate(sot_seqs):
            try:
                start_seq_wise_req[_sot_seq].append(_idx)
            except:
                start_seq_wise_req[_sot_seq] = [_idx]

        token_alignments = [[] for _ in seg_metadata]
        for start_seq, req_idx in start_seq_wise_req.items():
            logger.debug(f"Aligning tokens for sequence starting with {start_seq}")
            res = self.aligner_model.align(
                ctranslate2.StorageView.from_array(aligner_features[req_idx]),
                start_sequence=list(start_seq),
                text_tokens=[text_tokens[_] for _ in req_idx],
                num_frames=list(seq_lens[req_idx].detach().cpu().numpy()),
                median_filter_width=7,
            )

            for _res, _req_idx in zip(res, req_idx):
                token_alignments[_req_idx] = _res

        word_timings = []
        for _idx, _seg_metadata in enumerate(seg_metadata):
            _word_timings = self._assign_word_timings(
                token_alignments[_idx].alignments,
                token_alignments[_idx].text_token_probs,
                word_tokens[_idx][0],
                word_tokens[_idx][1],
            )

            stitched_seg = _seg_metadata["stitched_seg"]

            current_seg_idx = 0
            current_offset = _seg_metadata["start_time"]

            for w in _word_timings:
                while (w["start_time"] + current_offset) >= stitched_seg[current_seg_idx][1]:
                    current_seg_idx += 1
                    current_offset += (
                        stitched_seg[current_seg_idx][0] - stitched_seg[current_seg_idx - 1][1]
                    )

                w["start_time"] += current_offset
                w["end_time"] += current_offset

            word_timings.append(_word_timings)
        logger.debug(f"Word alignment completed for {len(word_timings)} segments")
        return word_timings
