import io
import re
from pathlib import Path
from typing import Optional

import tensorrt_llm
import torch
import torchaudio
from torch import Tensor

from whisper_trt.assets import download_assets
from whisper_trt.batching import WhisperBatchProcessor
from whisper_trt.custom_types import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NUM_BEAMS,
    SUPPORTED_SAMPLE_RATE,
    BatchWhisperItem,
    Segment,
    WhisperResult,
)
from whisper_trt.modeling import WhisperDecoding, WhisperEncoding
from whisper_trt.tokenizer import REVERSED_LANGUAGES, get_tokenizer
from whisper_trt.utils import log_mel_spectrogram

SEGMENTS_PATTERN = re.compile(r"<\|([\d.]+)\|>([^<]+)<\|([\d.]+)\|>")
LANG_CODE_PATTERN = re.compile(r"<\|([a-z]{2})\|>")


class WhisperModel(object):
    def __init__(
        self,
        engine_dir,
        tokenizer_name="multilingual",
        debug_mode=False,
        assets_dir=None,
        max_queue_time=0.01,  # 10 ms by default
    ):
        world_size = 1
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

        engine_dir = Path(engine_dir)

        self.assets_dir = assets_dir
        if self.assets_dir is None:
            self.assets_dir = download_assets()

        self.encoder = WhisperEncoding(engine_dir)
        self.decoder = WhisperDecoding(
            engine_dir, runtime_mapping, debug_mode=debug_mode
        )
        self.batch_size = self.decoder.decoder_config["max_batch_size"]
        self.n_mels = self.encoder.n_mels
        self.tokenizer = get_tokenizer(
            name=tokenizer_name,
            num_languages=self.encoder.num_languages,
            tokenizer_dir=self.assets_dir,
        )
        self.eot_id = self.tokenizer.encode(
            "<|endoftext|>", allowed_special=self.tokenizer.special_tokens_set
        )[0]

        self.batch_processor = WhisperBatchProcessor(
            self, max_batch_size=self.batch_size, max_queue_time=max_queue_time
        )

    def preprocess_audio(self, binary_data) -> dict:
        audio_stream = io.BytesIO(binary_data)
        waveform, sample_rate = torchaudio.load(audio_stream)

        # Resample audio to rate compatible with what the model was trained at
        if sample_rate != SUPPORTED_SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=SUPPORTED_SAMPLE_RATE
            )(waveform)
            sample_rate = SUPPORTED_SAMPLE_RATE

        return waveform

    def _get_text_prefix(
        self,
        language: str = "english",
        prompt: Optional[str] = None,
        timestamps: bool = False,
        task: str = "transcribe",
        prefix: Optional[str] = None,
    ):
        try:
            language_code = REVERSED_LANGUAGES[language]
        except KeyError:
            language_code = language
        text_prefix = f"<|startoftranscript|><|{language_code}|><|{task}|>"
        if prompt is not None:
            text_prefix = f"<|startofprev|> {prompt}" + text_prefix
        if timestamps:
            text_prefix += "<|0.00|>"
        else:
            text_prefix += "<|notimestamps|>"
        if prefix is not None:
            text_prefix += prefix
        return text_prefix

    def process_batch(
        self,
        mel_batch,
        decoder_input_ids,
        num_beams=DEFAULT_NUM_BEAMS,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    ) -> Tensor:
        encoder_output = self.encoder.get_audio_features(mel_batch)
        output_ids = self.decoder.generate(
            decoder_input_ids,
            encoder_output,
            self.eot_id,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        return output_ids

    def decode_output_ids(self, output_ids, text_prefix):
        text = self.tokenizer.decode(output_ids[0]).strip()
        text.replace(text_prefix, "")
        return text

    async def detect_audio_and_language(self, mel) -> Optional[str]:
        """
        Detects the audio and language from the given mel spectrogram.

        Args:
            mel: The mel spectrogram of the audio.

        Returns:
            The detected language code, or None if no speech is detected.
        """
        text_prefix = "<|startoftranscript|>"

        prompt_ids = self.tokenizer.encode(
            text_prefix, allowed_special=self.tokenizer.special_tokens_set
        )

        output_ids = await self.batch_processor.process(
            item=BatchWhisperItem(mel=mel, prompt_ids=prompt_ids, max_new_tokens=1)
        )
        text = self.decode_output_ids(output_ids, text_prefix)
        if text == "<|nospeech|>":
            return None
        return text.replace(text_prefix, "").replace("<|", "").replace("|>", "")

    async def transcribe(
        self,
        waveform,
        prompt: Optional[str] = None,
        language: Optional[str] = None,
        timestamps: bool = False,
        num_beams: int = DEFAULT_NUM_BEAMS,
        prefix: Optional[str] = None,
        task: str = "transcribe",
        max_new_tokens=128,
    ):
        mel = await log_mel_spectrogram(
            waveform.numpy(),
            self.n_mels,
            device="cuda",
            mel_filters_dir=self.assets_dir,
        )
        mel = mel.type(torch.float16)
        if language is None:
            language = await self.detect_audio_and_language(mel)
            if language is None:
                # No speech was detected. Can result empty segments
                return WhisperResult(segments=[], language_code=None)
        text_prefix = self._get_text_prefix(
            language=language,
            prompt=prompt,
            timestamps=timestamps,
            prefix=prefix,
            task=task,
        )

        prompt_ids = self.tokenizer.encode(
            text_prefix, allowed_special=self.tokenizer.special_tokens_set
        )

        output_ids: Tensor = await self.batch_processor.process(
            item=BatchWhisperItem(
                mel=mel, prompt_ids=prompt_ids, max_new_tokens=max_new_tokens
            )
        )

        return self._postprocess_transcript(
            self.decode_output_ids(output_ids, text_prefix)
        )

    def _postprocess_transcript(self, transcribed_text: str) -> WhisperResult:
        """
        Post-process the output of the transcription model.
        """
        language_code = LANG_CODE_PATTERN.findall(transcribed_text)[0]

        # Find all matches in the input string
        matches = SEGMENTS_PATTERN.findall(transcribed_text)

        # Process matches to create the desired output format
        segments = []
        for match in matches:
            start, text, end = match

            segments.append(
                Segment(
                    **{"start": float(start), "end": float(end), "text": text.strip()}
                )
            )

        return WhisperResult(segments=segments, language_code=language_code)
