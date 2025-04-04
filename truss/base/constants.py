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

ARM_PLATFORMS = ("aarch64", "arm64")


_TRUSS_ROOT = pathlib.Path(__file__).parent.parent.resolve()

TEMPLATES_DIR = _TRUSS_ROOT / "templates"
TRADITIONAL_CUSTOM_TEMPLATE_DIR = TEMPLATES_DIR / "custom"
PYTHON_DX_CUSTOM_TEMPLATE_DIR = TEMPLATES_DIR / "custom_python_dx"
DOCKER_SERVER_TEMPLATES_DIR = TEMPLATES_DIR / "docker_server"
SERVER_CODE_DIR: pathlib.Path = TEMPLATES_DIR / "server"
TRITON_SERVER_CODE_DIR: pathlib.Path = TEMPLATES_DIR / "triton"
TRTLLM_TRUSS_DIR: pathlib.Path = TEMPLATES_DIR / "trtllm-briton"
SHARED_SERVING_AND_TRAINING_CODE_DIR_NAME = "shared"
SHARED_SERVING_AND_TRAINING_CODE_DIR: pathlib.Path = (
    TEMPLATES_DIR / SHARED_SERVING_AND_TRAINING_CODE_DIR_NAME
)
CONTROL_SERVER_CODE_DIR: pathlib.Path = TEMPLATES_DIR / "control"
CHAINS_CODE_DIR: pathlib.Path = _TRUSS_ROOT.parent / "truss-chains" / "truss_chains"
TRUSS_CODE_DIR: pathlib.Path = _TRUSS_ROOT.parent / "truss"

SUPPORTED_PYTHON_VERSIONS = {"3.8", "3.9", "3.10", "3.11"}
MAX_SUPPORTED_PYTHON_VERSION_IN_CUSTOM_BASE_IMAGE = "3.12"
MIN_SUPPORTED_PYTHON_VERSION_IN_CUSTOM_BASE_IMAGE = "3.8"

TRTLLM_PREDICT_CONCURRENCY = 512
BEI_TRTLLM_CLIENT_BATCH_SIZE = 128
BEI_MAX_CONCURRENCY_TARGET_REQUESTS = 2048
BEI_REQUIRED_MAX_NUM_TOKENS = 16384

TRTLLM_MIN_MEMORY_REQUEST_GI = 10
HF_MODELS_API_URL = "https://huggingface.co/api/models"
HF_ACCESS_TOKEN_KEY = "hf_access_token"
TRUSSLESS_MAX_PAYLOAD_SIZE = "64M"
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
MODEL_CACHE_PATH = pathlib.Path("/app/model_cache")
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
TENSORFLOW_REQ_MODULE_NAMES: Set[str] = {"tensorflow"}

LIGHTGBM_REQ_MODULE_NAMES: Set[str] = {"lightgbm"}

# list from https://pytorch.org/get-started/locally/
PYTORCH_REQ_MODULE_NAMES: Set[str] = {"torch", "torchvision", "torchaudio"}

MLFLOW_REQ_MODULE_NAMES: Set[str] = {"mlflow"}

INFERENCE_SERVER_PORT = 8080

HTTP_PUBLIC_BLOB_BACKEND = "http_public"

REGISTRY_BUILD_SECRET_PREFIX = "DOCKER_REGISTRY_"

TRTLLM_SPEC_DEC_TARGET_MODEL_NAME = "target"
TRTLLM_SPEC_DEC_DRAFT_MODEL_NAME = "draft"
TRTLLM_BASE_IMAGE = "baseten/briton-server:v0.17.0-e882027"
TRTLLM_PYTHON_EXECUTABLE = "/usr/local/briton/venv/bin/python"
BASE_TRTLLM_REQUIREMENTS = ["briton==0.5.0"]
BEI_TRTLLM_BASE_IMAGE = "baseten/bei:0.0.20"

BEI_TRTLLM_PYTHON_EXECUTABLE = "/usr/bin/python3"

OPENAI_COMPATIBLE_TAG = "openai-compatible"
OPENAI_NON_COMPATIBLE_TAG = "force-legacy-api-non-openai-compatible"  # deprecated - openai-compatible is now the default


PRODUCTION_ENVIRONMENT_NAME = "production"
