import re
import zlib

import numpy as np
import tensorrt_llm
import torch

_ALIGNER_MODELS = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "tiny": "Systran/faster-whisper-tiny",
    "base.en": "Systran/faster-whisper-base.en",
    "base": "Systran/faster-whisper-base",
    "small.en": "Systran/faster-whisper-small.en",
    "small": "Systran/faster-whisper-small",
    "medium.en": "Systran/faster-whisper-medium.en",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
}


def get_word_aligner_model_repo_id(model_name: str) -> str:
    if model_name not in _ALIGNER_MODELS:
        raise ValueError(
            f"Model {model_name} not found in the Baseten model registry from Hugging Face"
        )
    return _ALIGNER_MODELS[model_name]


# Return the model name for hugging face
def get_whisper_engine_repo_id(model_name: str) -> str:
    if not torch.cuda.is_available():
        raise ValueError(
            f"Model {model_name} not found in the Baseten model registry from Hugging Face"
        )

    gpu_type: str = torch.cuda.get_device_name(0)
    tensorrt_llm_version = tensorrt_llm.__version__
    return "baseten-admin/" + re.sub(
        r"[^a-zA-Z0-9]",
        "_",
        f"whisper_trt_{model_name}_{gpu_type}_{tensorrt_llm_version}",
    )


def array_to_mono_tensor(waveform):
    # if numpy array, convert to torch tensor
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)

    assert isinstance(waveform, torch.Tensor), (
        f"Audio must be a torch.Tensor or numpy.ndarray, " f"but got {type(waveform)}"
    )

    # if multichannel, average the channels
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)

    assert waveform.dim() == 1, f"Audio must be a 1D or 2D tensor, but got {waveform.dim()}D"

    return waveform


def get_compression_ratio(text: str) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


def get_object_fields(obj):
    result = {}
    for attr in dir(obj):
        # Skip private and built-in attributes
        if attr.startswith("_"):
            continue

        value = getattr(obj, attr)
        result[attr] = value
    return ", ".join(f"{attr}: {value}" for attr, value in result.items())
