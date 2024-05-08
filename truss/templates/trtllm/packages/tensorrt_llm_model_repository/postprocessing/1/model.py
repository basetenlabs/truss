# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import os
from collections import OrderedDict

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer

# https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/decoders/strip.rs#L8
INVALID_UNICODE_CHAR = "�"


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # Parse model configs
        model_config = json.loads(args["model_config"])
        # NOTE: Keep this in sync with the truss model.py variable
        tokenizer_dir = os.environ["TRITON_TOKENIZER_REPOSITORY"]
        hf_auth_token = os.environ.get("HUGGING_FACE_HUB_TOKEN", None)

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            legacy=False,
            padding_side="left",
            trust_remote_code=True,
            token=hf_auth_token,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Parse model output configs
        output_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT")
        # Convert Triton types to numpy types
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

        self.state_dict = OrderedDict()
        # TODO(pankaj) This should come from the batch size
        self.cache_size = 2048

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for idx, request in enumerate(requests):
            # Get request ID
            request_id = request.request_id()

            # Get input tensors
            tokens_batch = (
                pb_utils.get_input_tensor_by_name(request, "TOKENS_BATCH")
                .as_numpy()
                .flatten()
            )

            if len(tokens_batch) == 0:
                continue

            # Postprocess output data
            prev_token = self._get_var(request_id, "prev_token")
            token_buffer = self._get_var(request_id, "token_buffer")
            token_buffer = token_buffer if token_buffer is not None else []
            current_tokens = np.concatenate(
                (np.array(token_buffer, dtype=int), tokens_batch), dtype=int
            )
            current_tokens_decoded = self.tokenizer.decode(current_tokens)

            if len(current_tokens_decoded) == 0:
                responses.append(pb_utils.InferenceResponse())
                continue

            if current_tokens_decoded[-1] == INVALID_UNICODE_CHAR:
                # If the last token is invalid, we need to keep it in the buffer
                # for the next request to see if this is a multi-token unicode
                # character.
                self._store_var(request_id, "token_buffer", current_tokens)
                responses.append(pb_utils.InferenceResponse())
                continue

            if prev_token is None:
                delta = current_tokens_decoded
            else:
                # TODO(pankaj) Figure out how to make tokenizer.decode not
                # ignore initial whitespace so we can avoid this hack.
                # Get string with and without previous token and diff. This hack
                # is needed because tokenizer.decode strips initial whitespace.
                old_string = self.tokenizer.decode(prev_token)
                with_prev_token = np.concatenate((prev_token, current_tokens))
                new_string = self.tokenizer.decode(with_prev_token)
                delta = self._compute_delta(old_string, new_string)

            # The previous token is the last character of the decoded sequence
            # which includes the multi-token unicode character.
            self._store_var(request_id, "prev_token", current_tokens)
            self._store_var(request_id, "token_buffer", None)

            # Create output tensor
            output_tensor = pb_utils.Tensor(
                "OUTPUT", np.array([delta]).astype(self.output_dtype)
            )

            # Get cum log probs
            cum_log_probs = pb_utils.get_input_tensor_by_name(request, "CUM_LOG_PROBS")

            # Get sequence length
            output_log_probs = pb_utils.get_input_tensor_by_name(
                request, "OUTPUT_LOG_PROBS"
            )

            # Get context logits
            context_logits = pb_utils.get_input_tensor_by_name(
                request, "CONTEXT_LOGITS"
            )

            # Get generation logits
            generation_logits = pb_utils.get_input_tensor_by_name(
                request, "GENERATION_LOGITS"
            )

            outputs = []
            outputs.append(output_tensor)

            if cum_log_probs:
                out_cum_log_probs = pb_utils.Tensor(
                    "OUT_CUM_LOG_PROBS", cum_log_probs.as_numpy()
                )
                outputs.append(out_cum_log_probs)
            else:
                out_cum_log_probs = pb_utils.Tensor(
                    "OUT_CUM_LOG_PROBS", np.array([[0.0]], dtype=np.float32)
                )
                outputs.append(out_cum_log_probs)

            if output_log_probs:
                out_output_log_probs = pb_utils.Tensor(
                    "OUT_OUTPUT_LOG_PROBS", output_log_probs.as_numpy()
                )
                outputs.append(out_output_log_probs)
            else:
                out_output_log_probs = pb_utils.Tensor(
                    "OUT_OUTPUT_LOG_PROBS", np.array([[[0.0]]], dtype=np.float32)
                )
                outputs.append(out_output_log_probs)

            if context_logits:
                out_context_logits = pb_utils.Tensor(
                    "OUT_CONTEXT_LOGITS", context_logits.as_numpy()
                )
                outputs.append(out_context_logits)
            else:
                out_context_logits = pb_utils.Tensor(
                    "OUT_CONTEXT_LOGITS", np.array([[[0.0]]], dtype=np.float32)
                )
                outputs.append(out_context_logits)

            if generation_logits:
                out_generation_logits = pb_utils.Tensor(
                    "OUT_GENERATION_LOGITS", generation_logits.as_numpy()
                )
                outputs.append(out_generation_logits)
            else:
                out_generation_logits = pb_utils.Tensor(
                    "OUT_GENERATION_LOGITS", np.array([[[[0.0]]]], dtype=np.float32)
                )
                outputs.append(out_generation_logits)

            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)

        return responses

    def finalize(self):
        print("Cleaning up...")

    def _store_var(self, request_id, var_name, var):
        if request_id in self.state_dict:
            self.state_dict[request_id][var_name] = var
            self.state_dict.move_to_end(request_id)
        else:
            if len(self.state_dict) > self.cache_size:
                self.state_dict.popitem(last=False)
            self.state_dict[request_id] = {"prev_token": None, "token_buffer": None}
            self.state_dict[request_id][var_name] = var

    def _get_var(self, request_id, var_name):
        if request_id in self.state_dict:
            return self.state_dict[request_id][var_name]
        return None

    def _compute_delta(self, prev_str, new_str):
        delta = "".join(
            [
                char
                for index, char in enumerate(new_str)
                if index >= len(prev_str) or char != prev_str[index]
            ]
        )
        return delta

    def _postprocessing(self, tokens):
        decoded_tokens = self.tokenizer.decode(tokens)
        return decoded_tokens
