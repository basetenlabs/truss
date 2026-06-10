from importlib.metadata import version
try:
    __version__ = version(__name__)
except:
    pass

from silero_vad.utils_vad import OnnxWrapper, SileroVAD

def load_model(device_name: str, model_type: str = "jit"):
    import torch
    from silero_vad.utils_vad import init_jit_model
    from pathlib import Path

    model_name = f"silero_vad.{model_type}"
    model_path = Path(__file__).parent / 'data' / model_name

    if model_type == "jit":
        model = init_jit_model(str(model_path), torch.device(device_name))
        return SileroVAD(model, device_name)
    elif model_type == "onnx":
        model = OnnxWrapper(str(model_path), force_onnx_cpu=device_name == "cpu")
        return SileroVAD(model, device_name)
    
    raise ValueError(f"Invalid model type: {model_type}")

def load_cpu_model(model_type: str = "jit"):
    return load_model("cpu", model_type)
    
def load_gpu_model(model_type: str = "jit"):
    return load_model("cuda", model_type)