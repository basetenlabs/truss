import logging
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from whisper_trt import WhisperModel

import torch
from async_batcher.batcher import AsyncBatcher
from torch import Tensor

from whisper_trt.custom_types import DEFAULT_NUM_BEAMS, BatchWhisperItem

FIXED_TEXT_PRFIX = "<|startoftranscript|><|en|><|transcribe|><|0.00|>"


class WhisperBatchProcessor(AsyncBatcher[List[BatchWhisperItem], List[str]]):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model: "WhisperModel" = model

    def concat_and_pad_mels(self, tensors: List[Tensor]):
        """Concatenates mel spectrograms to the maximum batch size using the last mel spectrogram as padding."""
        while len(tensors) < self.max_batch_size:
            tensors.append(tensors[-1])
        res = torch.cat(tensors, dim=0).type(torch.float16)
        return res

    def concat_and_pad_prompts(self, prompts: List[List]) -> Tensor:
        """Concatenates prompts to the maximum batch size using the last prompt as padding."""
        while len(prompts) < self.max_batch_size:
            prompts.append(prompts[-1])
        return Tensor(prompts)

    def process_batch(self, batch: List[BatchWhisperItem]) -> List[float]:
        logging.warn(f"Processing batch of size {len(batch)}")

        # Need to pad the batch up to the maximum batch size
        decoder_input_ids = self.concat_and_pad_prompts(
            [item.prompt_ids for item in batch]
        )
        mel_batch = self.concat_and_pad_mels([item.mel for item in batch])

        max_new_tokens = max(item.max_new_tokens for item in batch)
        batch_result = self.model.process_batch(
            mel_batch,
            decoder_input_ids,
            max_new_tokens=max_new_tokens,
            num_beams=DEFAULT_NUM_BEAMS,
        )
        # Splicing to len(batch) is needed to remove the padding we add
        # during `concat_and_pad_mels` and `concat_and_pad_prompts`
        return batch_result[: len(batch)]
