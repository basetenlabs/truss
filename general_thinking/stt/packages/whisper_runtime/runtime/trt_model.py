import json
import logging
import math
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorrt_llm.bindings.executor as trtllm
import torch
from prometheus_client import Gauge
from tensorrt_llm._utils import str_dtype_to_torch, str_dtype_to_trt, trt_dtype_to_torch
from tensorrt_llm.runtime.session import Session, TensorInfo
from whisper_utils.constants import (
    CROSS_KV_CACHE_FRACTION,
    FREE_GPU_MEMORY_FRACTION,
    WHISPER_ENCODER_DOWNSAMPLING_FACTOR,
)
from whisper_utils.utilities import get_gpu_accelerator

from .tokenizer import Tokenizer
from .utils import get_compression_ratio, get_object_fields

# LOGGING
logger = logging.getLogger(__name__)

IN_FLIGHT_BATCH_SIZE_GAUGE = Gauge("in_flight_batch_size_gauge", "Gauge of in flight batch sizes")


class ExecutorResponse:
    """
    Stores the outputs from the executor after model inference.

    Attributes:
        output_tokens_aligned (List[List[int]]): Output tokens aligned by request ID.
        cum_log_probs_aligned (List[np.ndarray]): Cumulative log probabilities aligned by request ID.
        generation_logits_aligned (List[np.ndarray]): Generation logits aligned by request ID.
        highest_probable_beam (List[int]): Indices of the highest probable beam for each request.
        number_of_segments (int): Number of segments processed.
    """

    def __init__(
        self,
        output_tokens: Dict[int, Dict[int, List[int]]],
        cum_log_probs: Dict[int, np.ndarray],
        generation_logits: Dict[int, np.ndarray],
        request_ids: List[int],
    ):
        # Have a list of keys to align the dictionaries
        keys = sorted(list(output_tokens.keys()))
        if keys != sorted(list(cum_log_probs.keys())) or keys != sorted(
            list(generation_logits.keys())
        ):
            raise ValueError(
                "Mismatch in keys among output_tokens, cum_log_probs, and generation_logits."
            )

        # Convert the 3 dictionaries to 3 aligned lists, by corresponding to the dictionary keys
        self.output_tokens_aligned = self._dict_to_list(output_tokens, keys)
        self.cum_log_probs_aligned = self._dict_to_list(cum_log_probs, keys)
        self.generation_logits_aligned = self._dict_to_list(generation_logits, keys)

        # Extra params

        self.highest_probable_beam = [
            0
        ]  # trtllm will always return the first beam as highest_probable_beam
        self.number_of_segments = len(self.output_tokens_aligned)

    def _dict_to_list(self, d, keys):
        return [d[key] for key in keys]

    def __str__(self):
        def summarize_list(lst, max_elements=5):
            if len(lst) > max_elements:
                return lst[:max_elements] + ["..."]
            return lst

        return (
            f"ExecutorResponse(\n"
            f"  output_tokens={summarize_list(self.output_tokens_aligned)},\n"
            f"  cum_log_probs={summarize_list(self.cum_log_probs_aligned)},\n"
            f"  generation_logits={summarize_list(self.generation_logits_aligned)},\n"
            f"  highest_probable_beam={self.highest_probable_beam},\n"
            f"  number_of_segments={self.number_of_segments}\n"
            f")"
        )


class WhisperTRTModel:
    def __init__(
        self,
        engine_dir: str,
        n_mels: int,
        dtype: str,
        tokenizer: Tokenizer,
        compute_type: str = "float16",
        max_batch_size: int = 6,
        max_beam_width: int = 5,
    ):
        """
        Initializes the WhisperTRTModel with encoder and decoder.

        Args:
            engine_dir (str or Path): Directory containing the engine files.
            tokenizer (Tokenizer): Tokenizer for encoding and decoding text.
            compute_type (str, optional): Compute precision type, defaults to "float16".
            max_batch_size (int, optional): Maximum batch size for the model, defaults to 6.
            max_beam_width (int, optional): Maximum beam size for sampling, defaults to 5.
        """
        engine_dir = Path(engine_dir)
        self.dtype = dtype

        self.encoder = WhisperEncoding(engine_dir, dtype=self.dtype)
        self.decoder = WhisperDecoding(
            engine_dir,
            tokenizer,
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
        )
        self.n_mels = n_mels
        self.compute_type = compute_type
        self.tokenizer = tokenizer

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Encodes the mel spectrogram into audio features.

        Args:
            mel (torch.Tensor): Input mel spectrogram.

        Returns:
            torch.Tensor: Encoded audio features.
        """
        logger.debug(f"encoder input: {mel.shape=} {mel=}")
        return self.encoder.get_audio_features(mel)

    def detect_language(
        self,
        features: torch.Tensor,
        beam_width: int,
        repetition_penalty: float,
        length_penalty: float,
        beam_search_diversity_rate: float,
        no_repeat_ngram_size: int,
        compression_ratio_threshold: float,
        log_prob_threshold: float,
        sampling_temperatures: List[float],
        pad_id: int,
        end_id: int,
        suppress_tokens: List[int],
    ) -> Tuple[List[str], List[Dict[str, float]]]:
        """
        Detects the language of the input audio features.

        Args:
            features (torch.Tensor): Encoded audio features.
            beam_width (int): Beam width for sampling.
            temperature (float): Temperature for sampling.
            repetition_penalty (float): Repetition penalty.
            length_penalty (float): Length penalty.
            beam_search_diversity_rate (float): Beam search diversity rate.
            no_repeat_ngram_size (int): No repeat ngram size.
            compression_ratio_threshold (float): Threshold for compression ratio.
            log_prob_threshold (float): Threshold for log probabilities.
            sampling_temperatures (List[float]): List of temperatures for sampling.
            pad_id (int): Padding token ID.
            end_id (int): End-of-sequence token ID.
            suppress_tokens (List[int]): Tokens to suppress during generation.

        Returns:
            Tuple[List[str], List[Dict[str, float]]]: Detected languages and their probabilities.
        """

        if features.shape[1] == self.n_mels:
            features = torch.stack(self.encode(features).chunk(len(features)))

        features = features.to(dtype=str_dtype_to_torch(self.compute_type))

        sampling_config = trtllm.SamplingConfig(
            beam_width=beam_width,
            temperature=0.0,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            beam_search_diversity_rate=beam_search_diversity_rate,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        decoder_input_ids = torch.tensor([[self.tokenizer.sot]] * len(features)).to(features.device)

        try:
            outputs = self.decoder.generate(
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=features,
                sampling_config=sampling_config,
                end_id=end_id,
                pad_id=pad_id,
                suppress_tokens=suppress_tokens,
                max_new_tokens=1,
            )
        except Exception as e:
            logger.error(f"Error in decoder generate: {e}")
            raise e

        logits = outputs.generation_logits_aligned

        # Process logits for all chunks
        languages = []
        language_probs = []

        for req_idx, chunk_logits in enumerate(logits):
            highest_probable_beam = outputs.highest_probable_beam[req_idx]
            chunk_logits = chunk_logits[highest_probable_beam]
            mask = torch.ones(chunk_logits.shape[-1], dtype=torch.bool, device=chunk_logits.device)
            mask[list(self.tokenizer.all_language_tokens)] = False
            chunk_logits[:, mask] = -np.inf
            language_token_probs = chunk_logits.softmax(dim=-1).cpu()

            chunk_language_probs = {
                c: language_token_probs[0, j].item()
                for j, c in zip(
                    self.tokenizer.all_language_tokens,
                    self.tokenizer.all_language_codes,
                )
            }

            languages.append(max(chunk_language_probs, key=chunk_language_probs.get))
            language_probs.append(chunk_language_probs)

        return languages, language_probs

    def generate_with_temperature_fallback(
        self,
        features: torch.Tensor,
        prompts: List[List[int]],
        beam_width: int,
        repetition_penalty: float,
        length_penalty: float,
        beam_search_diversity_rate: float,
        no_repeat_ngram_size: int,
        compression_ratio_threshold: float,
        log_prob_threshold: float,
        sampling_temperatures: List[float],
        pad_id: int,
        end_id: int,
        suppress_tokens: List[int],
        max_new_tokens: int,
        show_beam_results: bool,
    ) -> Tuple[ExecutorResponse, List[str], bool, List[str]]:
        """
        Generate text with temperature fallback.

        Args:
            features (torch.Tensor): Encoded audio features.
            prompts (List[List[int]]): Prompts for decoding.
            beam_width (int): Beam width for sampling.
            repetition_penalty (float): Repetition penalty.
            length_penalty (float): Length penalty.
            beam_search_diversity_rate (float): Beam search diversity rate.
            no_repeat_ngram_size (int): No repeat ngram size.
            compression_ratio_threshold (float): Threshold for compression ratio.
            log_prob_threshold (float): Threshold for log probabilities.
            sampling_temperatures (List[float]): List of temperatures for sampling.
            pad_id (int): Padding token ID.
            end_id (int): End-of-sequence token ID.
            suppress_tokens (List[int]): Tokens to suppress during generation.
            max_new_tokens (int): Maximum number of new tokens to generate.
            show_beam_results (bool): Whether to show the top N candidates.
        Returns:
            Tuple[ExecutorResponse, List[str], bool]: Aggregated response containing output tokens, log probabilities, and logits. The boolean indicates if the transcription is possible to be hallucinated.

        """
        if features.shape[1] == self.n_mels:
            features = torch.stack(self.encode(features).chunk(len(prompts)))

        features = features.to(dtype=str_dtype_to_torch(self.compute_type))

        decoder_input_ids = torch.tensor(prompts)

        sampling_config = trtllm.SamplingConfig(
            beam_width=beam_width,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            beam_search_diversity_rate=beam_search_diversity_rate,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        fallback_candidates = []

        beam_results = []

        if beam_width > 1:
            sampling_temperatures = [0.0]
            logging.debug(
                f"Beam search cannot work with temperature > 0.0. Setting {sampling_temperatures=}."
            )  # TODO: remove this once trtllm can support running different beam widths at the same time

        for temperature in sampling_temperatures:

            sampling_config.temperature = temperature

            if temperature > 0.0:
                sampling_config.beam_width = 1  # disable beam search when temperature > 0

            batched_results = self.decoder.generate(
                decoder_input_ids=decoder_input_ids,
                sampling_config=sampling_config,
                encoder_outputs=features,
                end_id=end_id,
                pad_id=pad_id,
                suppress_tokens=suppress_tokens,
                max_new_tokens=max_new_tokens,
            )

            # show all decoded batched_results.output_tokens_aligned[0] as debugging info
            outputs_from_all_beams = {}
            for beam_idx, output_tokens in batched_results.output_tokens_aligned[0].items():
                outputs_from_all_beams[beam_idx] = self.tokenizer.decode(
                    output_tokens, decoder_input_ids[0]
                )

            # Enhanced debug logging to compare beam outputs
            beam_comparison = "\n".join(
                [f"Beam {beam_idx}: {text}" for beam_idx, text in outputs_from_all_beams.items()]
            )
            logging.debug(f"=== Beam Results ===\n{beam_comparison}\n======================")

            if show_beam_results:
                beam_results = [
                    item[1] for item in sorted(outputs_from_all_beams.items(), key=lambda x: x[0])
                ]
            else:
                beam_results = []

            verified_text = []

            highest_probable_beam_idx = batched_results.highest_probable_beam[0]
            output_tokens = batched_results.output_tokens_aligned[0][highest_probable_beam_idx]
            avg_log_prob = batched_results.cum_log_probs_aligned[0][highest_probable_beam_idx] / (
                len(output_tokens) + 1
            )

            text = self.tokenizer.decode(output_tokens, decoder_input_ids[0])
            compression_ratio = get_compression_ratio(text)

            if temperature > 0.0:
                logging.debug(
                    f"Retrying temperature {temperature}, compression_ratio {compression_ratio} vs compression_ratio_threshold {compression_ratio_threshold}, avg_log_prob {avg_log_prob} vs log_prob_threshold {log_prob_threshold}"
                )
            else:
                logging.debug(
                    f"Trying temperature {temperature}, compression_ratio {compression_ratio} vs compression_ratio_threshold {compression_ratio_threshold}, avg_log_prob {avg_log_prob} vs log_prob_threshold {log_prob_threshold}"
                )

            possible_hallucination: bool = (
                compression_ratio >= compression_ratio_threshold
                or avg_log_prob <= log_prob_threshold
            )

            if not possible_hallucination:
                logging.debug(
                    f"Verified with temperature: {temperature}, compression_ratio {compression_ratio} vs compression_ratio_threshold {compression_ratio_threshold}, avg_log_prob {avg_log_prob} vs log_prob_threshold {log_prob_threshold}"
                )
                verified_text = [text]
                return batched_results, verified_text, possible_hallucination, beam_results

            fallback_candidates.append((batched_results, [text], avg_log_prob))

        logging.info(
            f"Looped through all temperatures, fall back to the one with the highest avg_log_prob"
        )
        best_candidate = max(fallback_candidates, key=lambda x: x[2])

        batched_results = best_candidate[0]
        verified_text = best_candidate[1]

        # If we get here, we have not verified the transcription, so it is possible to be hallucinated
        return batched_results, verified_text, True, beam_results


class WhisperEncoding:
    """
    Encodes audio input features using a TensorRT session.

    Args:
        engine_dir (Path): Directory containing the engine files.
    """

    def __init__(self, engine_dir: Path, dtype: str):
        self.session = self.get_session(engine_dir)
        self.dtype = dtype
        self.thread_storage = threading.local()

    def get_session(self, engine_dir: Path):
        """
        Loads the TensorRT session for the encoder.

        Args:
            engine_dir (Path): Directory containing the encoder engine files.

        Returns:
            Session: Loaded TensorRT session.
        """

        serialize_path = engine_dir / "encoder" / "rank0.engine"
        try:
            with open(serialize_path, "rb") as f:
                session = Session.from_serialized_engine(f.read())
        except FileNotFoundError:
            logger.error(f"Serialized engine file not found at {serialize_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading serialized engine: {e}")
            raise

        return session

    def _remove_tensor_padding(
        self,
        input_tensor: torch.Tensor,
        input_tensor_lengths: Optional[torch.Tensor] = None,
        pad_value=None,
    ):
        """
        Removes padding from a tensor based on the provided lengths.

        Args:
            input_tensor (torch.Tensor): The input tensor from which to remove padding.
            input_tensor_lengths (Optional[torch.Tensor]): Lengths of valid data in each sequence.
            pad_value (int, optional): The value used for padding. Defaults to 0.

        Returns:
            torch.Tensor: Tensor with padding removed.
        """
        if pad_value is not None:
            # Text tensor case: batch, seq_len
            assert torch.all(
                input_tensor[:, 0] != pad_value
            ), "First token in each sequence should not be pad_value"
            assert input_tensor_lengths is None

            mask = input_tensor != pad_value
            output_tensor = input_tensor[mask].view(1, -1)

        else:
            # Audio tensor case: batch, seq_len, feature_len
            assert (
                input_tensor_lengths is not None
            ), "input_tensor_lengths must be provided for 3D input_tensor"
            batch_size = input_tensor.shape[0]

            valid_sequences = []

            for i in range(batch_size):
                valid_length = input_tensor_lengths[i]
                valid_sequences.append(input_tensor[i, :valid_length])

            output_tensor = torch.cat(valid_sequences, dim=0)

        return output_tensor

    def get_audio_features(self, mel):
        """
        Encodes mel spectrogram into audio features using the TensorRT session.

        Args:
            mel (torch.Tensor): Input mel spectrogram.

        Returns:
            torch.Tensor: Encoded audio features.
        """
        if not self.thread_storage.__dict__.get("execution_context"):
            self.thread_storage.execution_context = self.session.engine.create_execution_context()
        mel = mel.type(str_dtype_to_torch(self.dtype))
        input_lengths = torch.full(
            (mel.shape[0],), mel.shape[2], dtype=torch.int32, device=mel.device
        )

        batch_size, seq_len = mel.shape[0], mel.shape[2]

        # compute position ids
        position_ids = (
            torch.arange(
                math.ceil(seq_len / WHISPER_ENCODER_DOWNSAMPLING_FACTOR),
                dtype=torch.int32,
                device=mel.device,
            )
            .expand(batch_size, -1)
            .contiguous()
        )

        # remove input padding
        mel = mel.transpose(1, 2)
        mel = self._remove_tensor_padding(mel, input_lengths)
        position_ids = self._remove_tensor_padding(position_ids, input_lengths, pad_value=None)

        inputs = OrderedDict()
        inputs["input_features"] = mel
        inputs["input_lengths"] = input_lengths
        inputs["position_ids"] = position_ids

        output_list = [
            TensorInfo("input_features", str_dtype_to_trt(self.dtype), mel.shape),
            TensorInfo("input_lengths", str_dtype_to_trt("int32"), input_lengths.shape),
            TensorInfo("position_ids", str_dtype_to_trt("int32"), position_ids.shape),
        ]

        output_info = (self.session).infer_shapes(
            output_list, context=self.thread_storage.execution_context
        )

        outputs = {
            t.name: torch.empty(tuple(t.shape), dtype=trt_dtype_to_torch(t.dtype), device="cuda")
            for t in output_info
        }

        stream = torch.cuda.current_stream()

        ok = self.session.run(
            inputs=inputs,
            outputs=outputs,
            stream=stream.cuda_stream,
            context=self.thread_storage.execution_context,
        )
        assert ok, "Engine execution failed"
        stream.synchronize()
        audio_features = outputs["encoder_output"].cpu()

        return audio_features


class WhisperDecoding:
    """
    Decodes the output tokens using a TensorRT decoder.

    Args:
        engine_dir (Path): Directory containing the decoder engine files.
        tokenizer (Tokenizer): Tokenizer for encoding and decoding text.
        max_batch_size (int): Maximum batch size for the model.
        max_beam_width (int): Maximum beam size for sampling
    """

    def __init__(self, engine_dir, tokenizer: Tokenizer, max_batch_size: int, max_beam_width: int):

        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_beam_width = max_beam_width

        self.decoder_config = self.get_config(engine_dir)
        self.executor = self.get_executor(engine_dir)
        self.in_flight_batch_size_gauge = IN_FLIGHT_BATCH_SIZE_GAUGE

    def get_config(self, engine_dir):
        config_path = engine_dir / "decoder" / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        decoder_config = OrderedDict()
        decoder_config.update(config["build_config"]["plugin_config"])
        decoder_config.update(config["pretrained_config"])
        decoder_config.update(config["build_config"])
        return decoder_config

    def get_executor(self, engine_dir):
        gpu_accelerator = get_gpu_accelerator()
        kv_cache_config = trtllm.KvCacheConfig(
            free_gpu_memory_fraction=FREE_GPU_MEMORY_FRACTION[gpu_accelerator],
            cross_kv_cache_fraction=CROSS_KV_CACHE_FRACTION[gpu_accelerator],
            enable_block_reuse=False,
        )

        config = trtllm.ExecutorConfig(
            max_beam_width=self.max_beam_width,
            max_batch_size=self.max_batch_size,
            kv_cache_config=kv_cache_config,
            gather_generation_logits=True,
        )

        return trtllm.Executor(
            engine_dir / "decoder",
            trtllm.ModelType.DECODER_ONLY,
            config,
        )

    def wait_for_responses(
        self,
        request_ids: list[int],
        executor: trtllm.Executor,
        sampling_config: trtllm.SamplingConfig,
    ) -> ExecutorResponse:
        """
        Waits for responses from the executor for given request IDs.

        Args:
            request_ids (List[int]): List of request IDs to wait for.
            executor (trtllm.Executor): Executor handling the requests.

        Returns:
            ExecutorResponse: Aggregated response containing output tokens, log probabilities, and logits.
        """

        output_tokens = {
            req_id: {beam: [] for beam in range(sampling_config.beam_width)}
            for req_id in request_ids
        }

        cum_log_probs = {req_id: {} for req_id in request_ids}
        logits = {req_id: {} for req_id in request_ids}
        num_finished = 0
        while num_finished < len(request_ids):
            stats = executor.get_latest_iteration_stats()

            if len(stats) > 0:
                logger.debug(f"Num requests curently in flight: {stats[-1].num_active_requests}")
                self.in_flight_batch_size_gauge.set(stats[-1].num_active_requests)

            responses = executor.await_responses(request_ids)
            for response in responses:
                response = response[0]
                req_id = response.request_id
                if not response.has_error():
                    result = response.result
                    logger.debug(f"decoder output: {get_object_fields(result)}")
                    cum_log_probs[req_id] = result.cum_log_probs
                    logits[req_id] = result.generation_logits
                    logger.debug(f"generation_logits shape: {result.generation_logits.shape}")
                    num_finished += 1 if result.is_final else 0
                    for beam, out_tokens in enumerate(result.output_token_ids):
                        output_tokens[req_id][beam].extend(out_tokens)
                else:
                    raise RuntimeError(str(req_id) + " encountered error: " + response.error_msg)
        return ExecutorResponse(output_tokens, cum_log_probs, logits, request_ids)

    def generate(
        self,
        decoder_input_ids,
        encoder_outputs,
        sampling_config,
        end_id,
        pad_id,
        suppress_tokens,
        max_new_tokens,
    ) -> ExecutorResponse:
        output_config = trtllm.OutputConfig()
        output_config.return_generation_logits = True
        output_config.return_log_probs = True
        output_config.exclude_input_from_output = False

        logger.debug(f"output_config: {get_object_fields(output_config)}")

        logger.debug(
            f"decoder input: {decoder_input_ids=}, {encoder_outputs=}, {encoder_outputs.shape=}, sampling_config={get_object_fields(sampling_config)}, end_id={end_id}, pad_id={pad_id}, suppress_tokens={suppress_tokens}, max_new_tokens={max_new_tokens}"
        )

        requests = [
            trtllm.Request(
                input_token_ids=input_token_ids.tolist(),
                end_id=end_id,
                pad_id=pad_id,
                max_tokens=max_new_tokens,  # TODO: remove this, can be automatically derived from model config
                bad_words=suppress_tokens,
                encoder_input_features=features,
                streaming=False,
                sampling_config=sampling_config,
                output_config=output_config,
            )
            for input_token_ids, features in zip(decoder_input_ids, encoder_outputs)
        ]

        request_ids = self.executor.enqueue_requests(requests)

        output = self.wait_for_responses(request_ids, self.executor, sampling_config)
        return output
