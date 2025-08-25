import pathlib

CUSTOM = "custom"

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
# Must be sorted ascendingly.
SUPPORTED_PYTHON_VERSIONS = ["3.8", "3.9", "3.10", "3.11", "3.12", "3.13"]

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


INFERENCE_SERVER_PORT = 8080

HTTP_PUBLIC_BLOB_BACKEND = "http_public"

REGISTRY_BUILD_SECRET_PREFIX = "DOCKER_REGISTRY_"

TRTLLM_SPEC_DEC_TARGET_MODEL_NAME = "target"
TRTLLM_SPEC_DEC_DRAFT_MODEL_NAME = "draft"
TRTLLM_BASE_IMAGE = "baseten/briton-server:v0.18.1-cefe1b1"
# TODO: build the image so that the default path `python3` can be used - then remove here.
TRTLLM_PYTHON_EXECUTABLE = "/usr/local/briton/venv/bin/python"
BEI_TRTLLM_BASE_IMAGE = "baseten/bei:0.0.24"
# TODO: build the image so that the default path `python3` can be used - then remove here.
OPENAI_COMPATIBLE_TAG = "openai-compatible"
OPENAI_NON_COMPATIBLE_TAG = "force-legacy-api-non-openai-compatible"  # deprecated - openai-compatible is now the default


PRODUCTION_ENVIRONMENT_NAME = "production"

TRUSS_BASE_IMAGE_NAME = "baseten/truss-server-base"
