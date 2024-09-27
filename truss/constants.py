import os
import pathlib
from typing import Set

SKLEARN = "sklearn"
TENSORFLOW = "tensorflow"
KERAS = "keras"
XGBOOST = "xgboost"
PYTORCH = "pytorch"
CUSTOM = "custom"
HUGGINGFACE_TRANSFORMER = "huggingface_transformer"
LIGHTGBM = "lightgbm"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = pathlib.Path(BASE_DIR, "truss")

TEMPLATES_DIR = pathlib.Path(CODE_DIR, "templates")
DOCKER_SERVER_TEMPLATES_DIR = pathlib.Path(CODE_DIR, "templates", "docker_server")
SERVER_CODE_DIR: pathlib.Path = TEMPLATES_DIR / "server"
TRITON_SERVER_CODE_DIR: pathlib.Path = TEMPLATES_DIR / "triton"
AUDIO_MODEL_TRTLLM_TRUSS_DIR: pathlib.Path = TEMPLATES_DIR / "trtllm-audio"
TRTLLM_TRUSS_DIR: pathlib.Path = TEMPLATES_DIR / "trtllm-briton"
SHARED_SERVING_AND_TRAINING_CODE_DIR_NAME = "shared"
SHARED_SERVING_AND_TRAINING_CODE_DIR: pathlib.Path = (
    TEMPLATES_DIR / SHARED_SERVING_AND_TRAINING_CODE_DIR_NAME
)
CONTROL_SERVER_CODE_DIR: pathlib.Path = TEMPLATES_DIR / "control"

SUPPORTED_PYTHON_VERSIONS = {"3.8", "3.9", "3.10", "3.11"}
MAX_SUPPORTED_PYTHON_VERSION_IN_CUSTOM_BASE_IMAGE = "3.12"
MIN_SUPPORTED_PYTHON_VERSION_IN_CUSTOM_BASE_IMAGE = "3.8"

TRTLLM_PREDICT_CONCURRENCY = 512
TRTLLM_MIN_MEMORY_REQUEST_GI = 24
HF_MODELS_API_URL = "https://huggingface.co/api/models"
HF_ACCESS_TOKEN_KEY = "hf_access_token"

# Alias for TEMPLATES_DIR
SERVING_DIR: pathlib.Path = TEMPLATES_DIR

REQUIREMENTS_TXT_FILENAME = "requirements.txt"
USER_SUPPLIED_REQUIREMENTS_TXT_FILENAME = "user_requirements.txt"
BASE_SERVER_REQUIREMENTS_TXT_FILENAME = "base_server_requirements.txt"
SERVER_REQUIREMENTS_TXT_FILENAME = "server_requirements.txt"
SYSTEM_PACKAGES_TXT_FILENAME = "system_packages.txt"

FILENAME_CONSTANTS_MAP = {
    "config_requirements_filename": REQUIREMENTS_TXT_FILENAME,
    "user_supplied_requirements_filename": USER_SUPPLIED_REQUIREMENTS_TXT_FILENAME,
    "base_server_requirements_filename": BASE_SERVER_REQUIREMENTS_TXT_FILENAME,
    "server_requirements_filename": SERVER_REQUIREMENTS_TXT_FILENAME,
    "system_packages_filename": SYSTEM_PACKAGES_TXT_FILENAME,
}

SERVER_DOCKERFILE_TEMPLATE_NAME = "server.Dockerfile.jinja"
MODEL_DOCKERFILE_NAME = "Dockerfile"

README_TEMPLATE_NAME = "README.md.jinja"
MODEL_README_NAME = "README.md"

CONFIG_FILE = "config.yaml"
DOCKERFILE = "Dockerfile"
# Used to indicate whether to associate a container with Truss
TRUSS = "truss"
# Used to create unique identifier based on last time truss was updated
TRUSS_MODIFIED_TIME = "truss_modified_time"
# Path of the Truss used to identify which Truss is being referred
TRUSS_DIR = "truss_dir"
TRUSS_HASH = "truss_hash"

HUGGINGFACE_TRANSFORMER_MODULE_NAME: Set[str] = set({})

# list from https://scikit-learn.org/stable/developers/advanced_installation.html
SKLEARN_REQ_MODULE_NAMES: Set[str] = {
    "numpy",
    "scipy",
    "joblib",
    "scikit-learn",
    "threadpoolctl",
}

XGBOOST_REQ_MODULE_NAMES: Set[str] = {"xgboost"}

# list from https://www.tensorflow.org/install/pip
# if problematic, lets look to https://www.tensorflow.org/install/source
TENSORFLOW_REQ_MODULE_NAMES: Set[str] = {
    "tensorflow",
}

LIGHTGBM_REQ_MODULE_NAMES: Set[str] = {
    "lightgbm",
}

# list from https://pytorch.org/get-started/locally/
PYTORCH_REQ_MODULE_NAMES: Set[str] = {
    "torch",
    "torchvision",
    "torchaudio",
}

MLFLOW_REQ_MODULE_NAMES: Set[str] = {"mlflow"}

INFERENCE_SERVER_PORT = 8080

HTTP_PUBLIC_BLOB_BACKEND = "http_public"

REGISTRY_BUILD_SECRET_PREFIX = "DOCKER_REGISTRY_"

TRTLLM_BASE_IMAGE = "baseten/briton-server:5fa9436e_v0.0.11"
TRTLLM_PYTHON_EXECUTABLE = "/usr/bin/python3"
BASE_TRTLLM_REQUIREMENTS = [
    "grpcio==1.62.3",
    "grpcio-tools==1.62.3",
    "transformers==4.44.2",
    "truss==0.9.31",
    "outlines==0.0.46",
    "torch==2.4.0",
    "sentencepiece==0.2.0",
]
AUDIO_MODEL_TRTLLM_REQUIREMENTS = [
    "--extra-index-url https://pypi.nvidia.com",
    "tensorrt_cu12_bindings==10.2.0.post1",
    "tensorrt_cu12_libs==10.2.0.post1",
    "tensorrt-cu12==10.2.0.post1",
    "tensorrt_llm==0.12.0.dev2024072301",
    "hf_transfer",
    "janus",
    "kaldialign",
    "librosa",
    "mpi4py==3.1.4",
    "safetensors",
    "soundfile",
    "tiktoken",
    "torchaudio",
    "async-batcher>=0.2.0",
    "pydantic>=2.7.1",
]
AUDIO_MODEL_TRTLLM_SYSTEM_PACKAGES = [
    "python3.10-venv",
    "openmpi-bin",
    "libopenmpi-dev",
]
OPENAI_COMPATIBLE_TAG = "openai-compatible"
