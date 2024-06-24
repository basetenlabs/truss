from typing import TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from whisper_trt import WhisperModel

from whisper_trt.types import BatchWhisperItem, DEFAULT_NUM_BEAMS

from async_batcher.batcher import AsyncBatcher
from torch import Tensor
import torch

FIXED_TEXT_PRFIX = "<|startoftranscript|><|en|><|transcribe|><|0.00|>"


class WhisperBatchProcessor(AsyncBatcher[list[BatchWhisperItem], list[str]]):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model: "WhisperModel" = model

    def concat_and_pad_mels(self, tensors: list[Tensor]):
        while len(tensors) < self.max_batch_size:
            tensors.append(tensors[-1])
        res = torch.cat(tensors, dim=0).type(torch.float16)
        return res

    def concat_and_pad_prompts(self, prompts: list[list]) -> Tensor:
        while len(prompts) < self.max_batch_size:
            prompts.append(prompts[-1])
        return Tensor(prompts)

    def process_batch(self, batch: list[BatchWhisperItem]) -> list[float]:
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
        return batch_result[: len(batch)]
