# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import tensorrt_llm
import tensorrt_llm.logger as logger
import torch
from async_batcher.batcher import AsyncBatcher
from datasets import load_dataset
from tensorrt_llm._utils import str_dtype_to_torch, str_dtype_to_trt, trt_dtype_to_torch
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from tensorrt_llm.runtime.session import Session, TensorInfo
from torch.utils.data import DataLoader
from whisper_lib.tokenizer import get_tokenizer
from whisper_lib.whisper_utils import N_SAMPLES, log_mel_spectrogram, pad_or_trim

TEXT_PREFIX = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"

# Num beams is the number of paths the model traverses before transcribing the text
NUM_BEAMS = 3


class WhisperEncoding:
    def __init__(self, engine_dir):
        self.session = self.get_session(engine_dir)

    def get_session(self, engine_dir):
        config_path = engine_dir / "encoder_config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        dtype = config["builder_config"]["precision"]
        n_mels = config["builder_config"]["n_mels"]
        num_languages = config["builder_config"]["num_languages"]

        self.dtype = dtype
        self.n_mels = n_mels
        self.num_languages = num_languages

        serialize_path = engine_dir / f"whisper_encoder_{self.dtype}_tp1_rank0.engine"

        with open(serialize_path, "rb") as f:
            session = Session.from_serialized_engine(f.read())

        return session

    def get_audio_features(self, mel):

        input_lengths = torch.tensor(
            [mel.shape[2] // 2 for _ in range(mel.shape[0])],
            dtype=torch.int32,
            device=mel.device,
        )

        inputs = OrderedDict()
        inputs["x"] = mel
        inputs["input_lengths"] = input_lengths

        output_list = [
            TensorInfo("x", str_dtype_to_trt(self.dtype), mel.shape),
            TensorInfo("input_lengths", str_dtype_to_trt("int32"), input_lengths.shape),
        ]

        output_info = (self.session).infer_shapes(output_list)

        logger.debug(f"output info {output_info}")
        outputs = {
            t.name: torch.empty(
                tuple(t.shape), dtype=trt_dtype_to_torch(t.dtype), device="cuda"
            )
            for t in output_info
        }
        stream = torch.cuda.current_stream()
        ok = self.session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
        assert ok, "Engine execution failed"
        stream.synchronize()
        audio_features = outputs["output"]
        return audio_features


class WhisperDecoding:
    def __init__(self, engine_dir, runtime_mapping, debug_mode=False):

        self.decoder_config = self.get_config(engine_dir)
        self.decoder_generation_session = self.get_session(
            engine_dir, runtime_mapping, debug_mode
        )

    def get_config(self, engine_dir):
        config_path = engine_dir / "decoder_config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        decoder_config = OrderedDict()
        decoder_config.update(config["plugin_config"])
        decoder_config.update(config["builder_config"])
        return decoder_config

    def get_session(self, engine_dir, runtime_mapping, debug_mode=False):
        dtype = self.decoder_config["precision"]
        serialize_path = engine_dir / f"whisper_decoder_{dtype}_tp1_rank0.engine"
        with open(serialize_path, "rb") as f:
            decoder_engine_buffer = f.read()

        decoder_model_config = ModelConfig(
            max_batch_size=self.decoder_config["max_batch_size"],
            max_beam_width=self.decoder_config["max_beam_width"],
            num_heads=self.decoder_config["num_heads"],
            num_kv_heads=self.decoder_config["num_heads"],
            hidden_size=self.decoder_config["hidden_size"],
            vocab_size=self.decoder_config["vocab_size"],
            num_layers=self.decoder_config["num_layers"],
            gpt_attention_plugin=self.decoder_config["gpt_attention_plugin"],
            remove_input_padding=self.decoder_config["remove_input_padding"],
            cross_attention=self.decoder_config["cross_attention"],
            has_position_embedding=self.decoder_config["has_position_embedding"],
            has_token_type_embedding=self.decoder_config["has_token_type_embedding"],
        )
        decoder_generation_session = tensorrt_llm.runtime.GenerationSession(
            decoder_model_config,
            decoder_engine_buffer,
            runtime_mapping,
            debug_mode=debug_mode,
        )

        return decoder_generation_session

    def generate(
        self, decoder_input_ids, encoder_outputs, eot_id, max_new_tokens=40, num_beams=1
    ):
        encoder_input_lengths = torch.tensor(
            [encoder_outputs.shape[1] for x in range(encoder_outputs.shape[0])],
            dtype=torch.int32,
            device="cuda",
        )

        decoder_input_lengths = torch.tensor(
            [decoder_input_ids.shape[-1] for _ in range(decoder_input_ids.shape[0])],
            dtype=torch.int32,
            device="cuda",
        )
        decoder_max_input_length = torch.max(decoder_input_lengths).item()

        cross_attention_mask = (
            torch.ones([encoder_outputs.shape[0], 1, encoder_outputs.shape[1]])
            .int()
            .cuda()
        )

        # generation config
        sampling_config = SamplingConfig(
            end_id=eot_id, pad_id=eot_id, num_beams=num_beams
        )
        self.decoder_generation_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            max_new_tokens,
            beam_width=num_beams,
            encoder_max_input_length=encoder_outputs.shape[1],
        )

        torch.cuda.synchronize()

        decoder_input_ids = decoder_input_ids.type(torch.int32).cuda()
        output_ids = self.decoder_generation_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_outputs,
            encoder_input_lengths=encoder_input_lengths,
            cross_attention_mask=cross_attention_mask,
        )
        torch.cuda.synchronize()

        # get the list of int from output_ids tensor
        output_ids = output_ids.cpu().numpy().tolist()
        return output_ids


class WhisperTRTLLM(object):
    def __init__(
        self,
        engine_dir,
        tokenizer_name="multilingual",
        assets_dir=None,
    ):
        world_size = 1
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)
        engine_dir = Path(engine_dir)

        self.encoder = WhisperEncoding(engine_dir)
        self.decoder = WhisperDecoding(engine_dir, runtime_mapping, debug_mode=False)
        self.n_mels = self.encoder.n_mels
        self.tokenizer = get_tokenizer(
            name=tokenizer_name,
            num_languages=self.encoder.num_languages,
            tokenizer_dir=assets_dir,
        )
        self.eot_id = self.tokenizer.encode(
            "<|endoftext|>", allowed_special=self.tokenizer.special_tokens_set
        )[0]

    def process_batch(
        self,
        mel,
        text_prefix="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        num_beams=1,
    ):
        prompt_id = self.tokenizer.encode(
            text_prefix, allowed_special=self.tokenizer.special_tokens_set
        )
        prompt_id = torch.tensor(prompt_id)
        batch_size = mel.shape[0]
        decoder_input_ids = prompt_id.repeat(batch_size, 1)

        encoder_output = self.encoder.get_audio_features(mel)
        output_ids = self.decoder.generate(
            decoder_input_ids,
            encoder_output,
            self.eot_id,
            max_new_tokens=96,
            num_beams=num_beams,
        )
        texts = []
        for i in range(len(output_ids)):
            text = self.tokenizer.decode(output_ids[i][0]).strip()
            texts.append(text)
        return texts


def decode_wav_file(
    input_file_path,
    model,
    text_prefix="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
    dtype="float16",
    batch_size=1,
    num_beams=1,
    normalizer=None,
    mel_filters_dir=None,
):
    mel, total_duration = log_mel_spectrogram(
        input_file_path,
        model.n_mels,
        device="cuda",
        return_duration=True,
        mel_filters_dir=mel_filters_dir,
    )
    mel = mel.type(str_dtype_to_torch(dtype))
    mel = mel.unsqueeze(0)
    # repeat the mel spectrogram to match the batch size
    mel = mel.repeat(batch_size, 1, 1)
    predictions = model.process_batch(mel, text_prefix, num_beams)
    prediction = predictions[0]

    # remove all special tokens in the prediction
    prediction = re.sub(r"<\|.*?\|>", "", prediction)
    if normalizer:
        prediction = normalizer(prediction)
    print(f"prediction: {prediction}")
    results = [(0, [""], prediction.split())]
    return results, total_duration


def collate_wrapper(batch):
    speeches, durations, labels, ids = [], [], [], []
    for item in batch:
        speech = item["audio"]["array"]
        duration = speech.shape[-1]
        speech = pad_or_trim(speech, N_SAMPLES)
        speech = speech.astype(np.float32)
        speech = torch.from_numpy(speech)
        speeches.append(speech)
        durations.append(duration)
        labels.append(item["text"])
        ids.append(item["id"])
    return speeches, durations, labels, ids


def decode_dataset(
    model,
    dataset,
    text_prefix="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
    dtype="float16",
    batch_size=1,
    num_beams=1,
    normalizer=None,
    sample_rate=16000,
    mel_filters_dir=None,
):
    librispeech_dummy = load_dataset(dataset, "clean", split="validation")

    data_loader = DataLoader(
        librispeech_dummy,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_wrapper,
    )
    results = []
    total_duration = 0
    for batch in data_loader:
        waveforms, durations, texts, ids = batch
        total_duration += sum(durations) / sample_rate

        for wave in waveforms:
            assert wave.is_pinned()

        features = [
            log_mel_spectrogram(
                wave, model.n_mels, device="cuda", mel_filters_dir=mel_filters_dir
            ).unsqueeze(0)
            for wave in waveforms
        ]
        features = torch.cat(features, dim=0).type(str_dtype_to_torch(dtype))
        predictions = model.process_batch(features, text_prefix, num_beams)
        for wav_id, label, prediction in zip(ids, texts, predictions):
            # remove all special tokens in the prediction
            prediction = re.sub(r"<\|.*?\|>", "", prediction)
            if normalizer:
                prediction, label = normalizer(prediction), normalizer(label)
            print(f"wav_id: {wav_id}, label: {label}, prediction: {prediction}")
            results.append((wav_id, label.split(), prediction.split()))
    return results, total_duration


class MlBatcher(AsyncBatcher[list[torch.Tensor], list[str]]):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model: WhisperTRTLLM = model

    def process_batch(self, batch: list[torch.Tensor]) -> list[float]:
        # Need to pad the batch up to the maximum batch size
        features = torch.cat(batch, dim=0).type(torch.float16)
        return self.model.process_batch(features, TEXT_PREFIX, NUM_BEAMS)
