import json
from collections import OrderedDict

import tensorrt_llm
import tensorrt_llm.logger as logger
import torch
from tensorrt_llm._utils import str_dtype_to_trt, trt_dtype_to_torch
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from tensorrt_llm.runtime.session import Session, TensorInfo

from whisper_trt.types import DEFAULT_NUM_BEAMS


class WhisperEncoding:
    def __init__(self, engine_dir):
        self.session = self.get_session(engine_dir)

    def get_session(self, engine_dir):
        config_path = engine_dir / "encoder" / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        self.dtype = config["pretrained_config"]["dtype"]
        self.n_mels = config["pretrained_config"]["n_mels"]
        self.num_languages = config["pretrained_config"]["num_languages"]

        serialize_path = engine_dir / "encoder" / "rank0.engine"

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
        config_path = engine_dir / "decoder" / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        decoder_config = OrderedDict()
        decoder_config.update(config["pretrained_config"])
        decoder_config.update(config["build_config"])
        return decoder_config

    def get_session(self, engine_dir, runtime_mapping, debug_mode=False):
        serialize_path = engine_dir / "decoder" / "rank0.engine"
        with open(serialize_path, "rb") as f:
            decoder_engine_buffer = f.read()

        decoder_model_config = ModelConfig(
            max_batch_size=self.decoder_config["max_batch_size"],
            max_beam_width=self.decoder_config["max_beam_width"],
            num_heads=self.decoder_config["num_attention_heads"],
            num_kv_heads=self.decoder_config["num_attention_heads"],
            hidden_size=self.decoder_config["hidden_size"],
            vocab_size=self.decoder_config["vocab_size"],
            num_layers=self.decoder_config["num_hidden_layers"],
            gpt_attention_plugin=self.decoder_config["plugin_config"][
                "gpt_attention_plugin"
            ],
            remove_input_padding=self.decoder_config["plugin_config"][
                "remove_input_padding"
            ],
            cross_attention=True,
            has_position_embedding=self.decoder_config["has_position_embedding"],
            has_token_type_embedding=False,
        )
        decoder_generation_session = tensorrt_llm.runtime.GenerationSession(
            decoder_model_config,
            decoder_engine_buffer,
            runtime_mapping,
            debug_mode=debug_mode,
        )

        return decoder_generation_session

    def generate(
        self,
        decoder_input_ids,
        encoder_outputs,
        eot_id,
        max_new_tokens=40,
        num_beams=DEFAULT_NUM_BEAMS,
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
