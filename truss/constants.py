import os
import pathlib

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
SERVER_CODE_DIR = TEMPLATES_DIR / "server"
TRAINING_JOB_WRAPPER_CODE_DIR_NAME = "training"
TRAINING_JOB_WRAPPER_CODE_DIR = TEMPLATES_DIR / TRAINING_JOB_WRAPPER_CODE_DIR_NAME
SHARED_SERVING_AND_TRAINING_CODE_DIR_NAME = "shared"
SHARED_SERVING_AND_TRAINING_CODE_DIR = (
    TEMPLATES_DIR / SHARED_SERVING_AND_TRAINING_CODE_DIR_NAME
)
CONTROL_SERVER_CODE_DIR = TEMPLATES_DIR / "control"


# Alias for TEMPLATES_DIR
SERVING_DIR = TEMPLATES_DIR

REQUIREMENTS_TXT_FILENAME = "requirements.txt"
SERVER_REQUIREMENTS_TXT_FILENAME = "server_requirements.txt"
TRAINING_REQUIREMENTS_TXT_FILENAME = "training_requirements.txt"
SYSTEM_PACKAGES_TXT_FILENAME = "system_packages.txt"

SERVER_DOCKERFILE_TEMPLATE_NAME = "server.Dockerfile.jinja"
TRAINING_DOCKERFILE_TEMPLATE_NAME = "training.Dockerfile.jinja"
MODEL_DOCKERFILE_NAME = "Dockerfile"
TRAINING_DOCKERFILE_NAME = "Dockerfile"

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
TRAINING_TRUSS_HASH = "training_truss_hash"
TRAINING_LABEL = "training"

HUGGINGFACE_TRANSFORMER_MODULE_NAME = {}

# list from https://scikit-learn.org/stable/developers/advanced_installation.html
SKLEARN_REQ_MODULE_NAMES = {
    "numpy",
    "scipy",
    "joblib",
    "scikit-learn",
    "threadpoolctl",
}

XGBOOST_REQ_MODULE_NAMES = {"xgboost"}

# list from https://www.tensorflow.org/install/pip
# if problematic, lets look to https://www.tensorflow.org/install/source
TENSORFLOW_REQ_MODULE_NAMES = {
    "tensorflow",
}

LIGHTGBM_REQ_MODULE_NAMES = {
    "lightgbm",
}

# list from https://pytorch.org/get-started/locally/
PYTORCH_REQ_MODULE_NAMES = {
    "torch",
    "torchvision",
    "torchaudio",
}

MLFLOW_REQ_MODULE_NAMES = {"mlflow"}

INFERENCE_SERVER_PORT = 8080

TRAINING_VARIABLES_FILENAME = "variables.yaml"
