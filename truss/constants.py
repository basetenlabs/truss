import os
import pathlib

SKLEARN = 'sklearn'
TENSORFLOW = 'tensorflow'
KERAS = 'keras'
XGBOOST = 'xgboost'
PYTORCH = 'pytorch'
CUSTOM = 'custom'
HUGGINGFACE_TRANSFORMER = 'huggingface_transformer'
LIGHTGBM = 'lightgbm'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = pathlib.Path(BASE_DIR, 'truss')

TEMPLATES_DIR = pathlib.Path(CODE_DIR, 'templates')
SERVER_CODE_DIR = TEMPLATES_DIR / 'server'

# Alias for TEMPLATES_DIR
SERVING_DIR = TEMPLATES_DIR

REQUIREMENTS_TXT_FILENAME = 'requirements.txt'
SERVER_REQUIREMENTS_TXT_FILENAME = 'server_requirements.txt'
SYSTEM_PACKAGES_TXT_FILENAME = 'system_packages.txt'

SERVER_DOCKERFILE_TEMPLATE_NAME = 'server.Dockerfile.jinja'
MODEL_DOCKERFILE_NAME = 'Dockerfile'

README_TEMPLATE_NAME = 'README.md.jinja'
MODEL_README_NAME = 'README.md'

CONFIG_FILE = 'config.yaml'
DOCKERFILE = "Dockerfile"
# Used to indicate whether to associate a container with Truss
TRUSS = "truss"
# Used to create unique identifier based on last time truss was updated
TRUSS_MODIFIED_TIME = "truss_modified_time"
# Path of the Truss used to identify which Truss is being referred
TRUSS_DIR = "truss_dir"
