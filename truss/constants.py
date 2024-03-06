import os
import pathlib

TRUSS_PACKAGE_DIR = pathlib.Path(__file__).resolve().parent

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
TRTLLM_TRUSS_DIR: pathlib.Path = TEMPLATES_DIR / "trtllm"

SUPPORTED_PYTHON_VERSIONS = {"3.8", "3.9", "3.10", "3.11"}


REQUIREMENTS_TXT_FILENAME = "requirements.txt"
USER_SUPPLIED_REQUIREMENTS_TXT_FILENAME = "user_requirements.txt"
SYSTEM_PACKAGES_TXT_FILENAME = "system_packages.txt"

FILENAME_CONSTANTS_MAP = {
    "config_requirements_filename": REQUIREMENTS_TXT_FILENAME,
    "user_supplied_requirements_filename": USER_SUPPLIED_REQUIREMENTS_TXT_FILENAME,
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

INFERENCE_SERVER_PORT = 8080

HTTP_PUBLIC_BLOB_BACKEND = "http_public"

REGISTRY_BUILD_SECRET_PREFIX = "DOCKER_REGISTRY_"

TRTLLM_BASE_IMAGE = "baseten/trtllm-build-server:r23.12_baseten_v0.7.1_20240111"
BASE_TRTLLM_REQUIREMENTS = [
    "tritonclient[all]==2.42.0",
    "transformers==4.33.1",
    "jinja2==3.1.3",
]
