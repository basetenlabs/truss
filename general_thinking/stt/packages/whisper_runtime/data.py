import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from .audio import pad_or_trim
from .chunker.chunker import Chunker
from .configs import *

# Configure the logging
logger = logging.getLogger(__name__)


def merge_speech_segments(start_ends: List[Tuple[int, int]], max_len=27.0):
    """
    Stitches speech segments together based on the maximum segment length.

    Args:
        start_ends (list of tuples): List of (start, end) times for each speech segment.
        max_len (float): Maximum length of a stitched segment in seconds.

    Returns:
        list of list of tuples: List of stitched speech segments, where each stitched segment is a list of (start, end) times.
    """

    # Calculate the duration of each speech segment
    speech_durations = [end - start for start, end in start_ends]

    stitched_speech_segments = []

    # Initialize the first segment
    curr_seg = [0]
    curr_dur = speech_durations[0]
    idx = 1

    # Iterate through the segments and stitch them based on max_len
    while idx < len(start_ends):
        if curr_dur + speech_durations[idx] > max_len:
            # If adding the next segment exceeds max_len, finalize the current segment
            stitched_speech_segments.append([start_ends[_] for _ in curr_seg])
            curr_seg = [idx]
            curr_dur = speech_durations[idx]
        else:
            # Otherwise, add the segment to the current segment
            curr_dur += speech_durations[idx]
            curr_seg.append(idx)

        idx += 1

    # Append the last segment
    stitched_speech_segments.append([start_ends[_] for _ in curr_seg])

    return stitched_speech_segments


class BasicSegmenter:
    """
    BasicSegmenter is a simple audio segmenter that segments the entire audio signal as a single segment.
    It is used when voice activity detection (VAD) is not required.

    Attributes:
        sampling_rate (int): The sampling rate of the audio signal.
    """

    def __init__(self, sampling_rate=16000):
        """
        Initializes the BasicSegmenter with a given sampling rate.

        Args:
            sampling_rate (int, optional): The sampling rate of the audio signal. Defaults to 16000.
        """
        self.sampling_rate = sampling_rate

    def __call__(self, audio_signal=np.ndarray) -> Tuple[List[Tuple[float, float]], np.ndarray]:
        """
        Segments the audio signal into a single segment.

        Args:
            audio_signal (np.ndarray): The audio signal to be segmented.

        Returns:
            Tuple[List[Tuple[float, float]], np.ndarray]: A tuple containing a list with a single (start, end) tuple
                                                          and the original audio signal.
        """
        audio_duration = len(audio_signal) / self.sampling_rate

        if audio_duration > 30:
            raise RuntimeError("Audio duration exceeds 30 seconds.")

        return [[0, audio_duration]], audio_signal


@dataclass
class AudioSegment:
    """
    Data class representing an audio segment with associated metadata.

    Attributes:
        audio (np.ndarray): The audio data of the segment.
        prompt (List[int]): The tokenized prompt sequence.
        initial_prompt_tokens (List[int]): Tokenized initial prompt (e.g., user-provided text).
        seq_len (int): The length of the audio sequence in samples.
        seg_metadata (Dict): Metadata about the segment (e.g., file ID, start/end times).
    """

    audio: np.ndarray
    prompt: list
    initial_prompt_tokens: list
    seq_len: int
    seg_metadata: dict
    prefix: list


class WhisperAudioProcessor:
    def __init__(
        self,
        device: torch.device,
        tokenizer,
        speech_segmenter: BasicSegmenter | Chunker,
        dta_padding: float = 3.0,
        enable_word_timestamp: bool = True,
        max_speech_len: float = 29.0,
        max_initial_prompt_len: int = 223,
        merge_chunks: bool = True,
        use_dynamic_time_axis: bool = False,
        detect_language_fn=None,
    ):
        """
        Initializes the WhisperAudioProcessor.

        Args:
            device (torch.device): The device to use for tensor operations.
            tokenizer: The tokenizer to use for encoding prompts.
            speech_segmenter (callable): The speech segmenter to use for segmenting audio.
            dta_padding (float, optional): Padding (in seconds) to add when using dynamic time axis. Defaults to 3.0.
            enable_word_timestamp (bool, optional): Whether to exclude timestamps from the prompt. Defaults to True.
            max_speech_len (float, optional): Maximum length of a speech segment in seconds. Defaults to 29.0.
            max_initial_prompt_len (int, optional): Maximum length of the initial prompt in tokens. Defaults to 223.
            merge_chunks (bool, optional): Whether to merge adjacent speech segments into larger chunks. Defaults to True.
            use_dynamic_time_axis (bool, optional): Whether to use a dynamic time axis based on segment length. Defaults to False.
            detect_language_fn (callable, optional): Function to detect language from audio segments.
        """
        self.device = device
        self.tokenizer = tokenizer
        self.speech_segmenter = speech_segmenter
        self.dta_padding = int(dta_padding * SAMPLE_RATE)  # Convert padding from seconds to samples
        self.enable_word_timestamp = enable_word_timestamp
        self.max_speech_len = max_speech_len
        self.max_initial_prompt_len = max_initial_prompt_len
        self.use_dynamic_time_axis = use_dynamic_time_axis
        self.merge_chunks = merge_chunks
        self.detect_language_fn = detect_language_fn

    def detect_language(
        self, audio_wav: torch.tensor, language_options: List[str] = []
    ) -> Tuple[List[str], List[Dict]]:
        """
        Detects the language of the audio file

        Args:
            audio_wav (torch.tensor): 1D audio tensor
            language_options (List[str], optional): List of languages to consider for language detection. If empty, all languages will be considered.

        Returns:
            Tuple[str, str]: Detected language and the probability.
        """

        # segment the audio, and detect the language on the first segment
        start_ends, _ = self.speech_segmenter(audio_signal=audio_wav)

        if not start_ends:
            raise ValueError(f"No speech segments detected for language detection.")

        # Extract the audio data for the first segment
        first_st, first_et = start_ends[0]
        first_segment_audio = audio_wav[int(first_st * SAMPLE_RATE) : int(first_et * SAMPLE_RATE)]
        seq_lengths = torch.tensor([first_segment_audio.shape[-1]]).to(self.device)
        audio_segment = pad_or_trim(first_segment_audio)

        # Detect the language of the first segment
        detected_lang, lang_probs = self.detect_language_fn(
            audio_segment.unsqueeze(0).to(self.device),
            seq_lengths,
        )
        try:
            if language_options:
                for lang in language_options:
                    if lang not in lang_probs[0]:
                        raise ValueError(
                            f'Language "{lang}" provided in lang_options is not supported by model'
                        )
                filtered_language_probs = {
                    k: v for k, v in lang_probs[0].items() if k in language_options
                }
                sorted_language_probs = dict(
                    sorted(filtered_language_probs.items(), key=lambda item: item[1], reverse=True)
                )
                top_filtered_language = next(iter(sorted_language_probs))
                return top_filtered_language, sorted_language_probs
        except Exception as e:
            logger.error(
                f"Failed to use user-provided language options:{language_options}, falling back to default language detection. {e}"
            )
        return detected_lang[0], lang_probs[0]

    def get_segmented_audio_signal(
        self,
        start_ends: List[Tuple[float, float]],
        audio_signal: np.ndarray,
        lang: str,
        task: str,
        initial_prompt: str,
        show_word_timestamps: bool,
        prefix: str,
        sr: int = SAMPLE_RATE,
    ) -> List[AudioSegment]:
        """
        Segments the audio signal based on start and end times, and prepares AudioSegment instances.

        Args:
            start_ends (List[Tuple[float, float]]): List of (start_time, end_time) tuples for each segment.
            audio_signal (np.ndarray): The full audio signal array.
            lang (str): The language code for transcription.
            task (str): The task type (e.g., 'transcribe' or 'translate').
            initial_prompt (str): The initial prompt text provided by the user.
            show_word_timestamps (bool): Whether to show word timestamps.
            prefix (str): Prefix for whisper (for more info see https://github.com/openai/whisper/discussions/117#discussioncomment-3727051)
            sr (int, optional): The sampling rate of the audio signal. Defaults to SAMPLE_RATE.

        Returns:
            List[AudioSegment]: List of AudioSegment instances prepared for model input.

        Reference: how to format whisper prompt https://github.com/openai/whisper/blob/main/whisper/audio.py#L100
        """
        # Tokenize and truncate the initial prompt if provided
        initial_prompt_tokens = (
            self.tokenizer.encode(" " + initial_prompt.strip())[-self.max_initial_prompt_len :]
            if initial_prompt
            else []
        )

        # Tokenize the prefix if provided
        prefix_tokens = self.tokenizer.encode(" " + prefix.strip()) if prefix else []

        # Create the prompt sequence with task and language tokens
        prompt = self.tokenizer.sot_sequence(task=task, lang=lang)

        prompt.append(
            self.tokenizer.no_timestamps
        )  # we use separate model for word alignment so we do not need timestamps from Whisper output

        segments = []

        if self.merge_chunks:
            # Merge adjacent speech segments into larger chunks
            stitched_segments = merge_speech_segments(start_ends, max_len=self.max_speech_len)
            for stitched_seg in stitched_segments:
                # Concatenate audio data from the stitched segments
                audio = np.concatenate(
                    [audio_signal[int(st * sr) : int(et * sr)] for st, et in stitched_seg]
                )

                seq_len = audio.shape[-1]  # Length of the concatenated audio sequence
                seg_metadata = {
                    "start_time": stitched_seg[0][0],  # Start time of the first segment
                    "end_time": stitched_seg[-1][1],  # End time of the last segment
                    "stitched_seg": stitched_seg,  # List of stitched segments
                    "lang_code": lang,
                    "time_length": stitched_seg[-1][1]
                    - stitched_seg[0][0],  # Total time length of the stitched segments
                }

                # Create an AudioSegment instance and add it to the list
                segments.append(
                    AudioSegment(
                        audio=audio,
                        prompt=prompt,
                        initial_prompt_tokens=initial_prompt_tokens,
                        seq_len=seq_len,
                        seg_metadata=seg_metadata,
                        prefix=prefix_tokens,
                    )
                )
        else:
            # Process each segment individually without merging
            for st, et in start_ends:
                # Extract the audio data for the segment
                audio = audio_signal[int(st * sr) : int(et * sr)]

                seq_len = audio.shape[-1]  # Length of the audio sequence
                seg_metadata = {
                    "start_time": st,
                    "end_time": et,
                    "lang_code": lang,
                    "time_length": et - st,
                }

                # Create an AudioSegment instance and add it to the list
                segments.append(
                    AudioSegment(
                        audio=audio,
                        prompt=prompt,
                        initial_prompt_tokens=initial_prompt_tokens,
                        seq_len=seq_len,
                        seg_metadata=seg_metadata,
                        prefix=prefix_tokens,
                    )
                )

        return segments

    def data_collate_fn(
        self, batch: List[AudioSegment]
    ) -> Tuple[torch.Tensor, List[List[int]], torch.Tensor, List[Dict]]:
        """
        Collates a batch of data for processing.

        Args:
            batch (List[AudioSegment]): List of AudioSegment instances.

        Returns:
            Tuple[torch.Tensor, List[List[int]], torch.Tensor, List[Dict]]:
                - signal_batch: Tensor of shape (batch_size, max_len) containing audio signals.
                - prompt_batch: List of tokenized prompts for each segment.
                - seq_len: Tensor containing the sequence lengths of each audio segment.
                - seg_metadata: List of metadata dictionaries for each segment.
        """
        # Determine the maximum sequence length in the batch
        if self.use_dynamic_time_axis:
            # Use the maximum sequence length plus padding, capped at N_SAMPLES
            max_len = min(
                max([segment.seq_len for segment in batch]) + self.dta_padding,
                N_SAMPLES,
            )
        else:
            # Use a fixed maximum length
            max_len = N_SAMPLES

        # Prepare the signal batch by padding or trimming each audio segment to max_len
        signal_batch = torch.stack(
            [
                torch.from_numpy(pad_or_trim(segment.audio, length=max_len)).to(self.device).float()
                for segment in batch
            ]
        )

        # Sequence lengths of each segment (before padding/trimming)
        seq_len = torch.tensor([segment.seq_len for segment in batch]).to(self.device)

        # Determine the maximum initial prompt length in the batch
        initial_prompt_max_len = max([len(segment.initial_prompt_tokens) for segment in batch])

        # Prepare the prompt batch
        prompt_batch = []
        for segment in batch:
            # Create padding for the initial prompt to align lengths
            initial_prompt_padding = (
                (
                    [self.tokenizer.sot_prev]
                    + [self.tokenizer.silent_token]
                    * (initial_prompt_max_len - len(segment.initial_prompt_tokens))
                    + segment.initial_prompt_tokens
                )
                if len(segment.initial_prompt_tokens) > 0
                else []
            )
            # Combine the initial prompt padding with the main prompt tokens
            prompt_batch.append(initial_prompt_padding + segment.prompt)

            # add prefix to the prompt
            prompt_batch.append(segment.prefix)

        # Collect segment metadata
        seg_metadata = [segment.seg_metadata for segment in batch]

        return signal_batch, prompt_batch, seq_len, seg_metadata

    def process_audio_file(
        self,
        audio_wav: torch.tensor,
        lang_code: str,
        task: str,
        prompt: str,
        show_word_timestamps: bool,
        prefix: str,
        _bypass_vad: bool,
    ) -> List[AudioSegment]:
        """
        Processes an audio file and returns a list of AudioSegment instances.

        Args:
            audio_wav (torch.tensor): 1D torch tensor.
            lang_code (str): Language of the audio.
            task (str): Task (e.g., 'transcribe' or 'translate').
            prompt (str): Prompt for Whisper.
            show_word_timestamps (bool): Whether to show word timestamps.
            prefix (str): Prefix for whisper (for more info see https://github.com/openai/whisper/discussions/117#discussioncomment-3727051)
            _bypass_vad (bool, optional): Whether to bypass VAD. Only set this to True if you know the audio is <30 seconds (mainly used for missing chunk fallback)

        Returns:
            List[AudioSegment]: List of AudioSegment instances prepared for batching.
        """

        if not _bypass_vad:
            start_ends, audio_signal = self.speech_segmenter(audio_signal=audio_wav)
        else:
            duration = len(audio_wav) / SAMPLE_RATE
            if duration > 30:
                raise ValueError("Audio duration exceeds 30 seconds. Please set _bypass_vad=False.")
            start_ends = [[0, duration]]
            audio_signal = audio_wav
        segments = self.get_segmented_audio_signal(
            start_ends, audio_signal, lang_code, task, prompt, show_word_timestamps, prefix
        )
        return segments
