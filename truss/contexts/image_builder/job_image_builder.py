from __future__ import annotations
import json

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import boto3
import yaml
from botocore import UNSIGNED
from botocore.client import Config
from google.cloud import storage
from huggingface_hub import get_hf_file_metadata, hf_hub_url, list_repo_files
from huggingface_hub.utils import filter_repo_objects
from truss.base import constants
from truss.base.constants import (
    BASE_SERVER_REQUIREMENTS_TXT_FILENAME,
    BASE_TRTLLM_REQUIREMENTS,
    BEI_MAX_CONCURRENCY_TARGET_REQUESTS,
    BEI_TRTLLM_BASE_IMAGE,
    BEI_TRTLLM_CLIENT_BATCH_SIZE,
    BEI_TRTLLM_PYTHON_EXECUTABLE,
    CHAINS_CODE_DIR,
    CONTROL_SERVER_CODE_DIR,
    DOCKER_SERVER_TEMPLATES_DIR,
    FILENAME_CONSTANTS_MAP,
    MAX_SUPPORTED_PYTHON_VERSION_IN_CUSTOM_BASE_IMAGE,
    MIN_SUPPORTED_PYTHON_VERSION_IN_CUSTOM_BASE_IMAGE,
    MODEL_DOCKERFILE_NAME,
    REQUIREMENTS_TXT_FILENAME,
    SERVER_CODE_DIR,
    SERVER_DOCKERFILE_TEMPLATE_NAME,
    SERVER_REQUIREMENTS_TXT_FILENAME,
    SHARED_SERVING_AND_TRAINING_CODE_DIR,
    SHARED_SERVING_AND_TRAINING_CODE_DIR_NAME,
    SYSTEM_PACKAGES_TXT_FILENAME,
    TEMPLATES_DIR,
    TRTLLM_BASE_IMAGE,
    TRTLLM_PREDICT_CONCURRENCY,
    TRTLLM_PYTHON_EXECUTABLE,
    TRTLLM_TRUSS_DIR,
    TRUSSLESS_MAX_PAYLOAD_SIZE,
    USER_SUPPLIED_REQUIREMENTS_TXT_FILENAME,
)
from truss.base.trt_llm_config import TRTLLMConfiguration, TrussTRTLLMModel
from truss.base.truss_config import (
    DEFAULT_BUNDLED_PACKAGES_DIR,
    BaseImage,
    DockerServer,
    TrussConfig,
    DockerAuthType,
    DockerAuthSettings,
)
from truss.base.truss_spec import TrussSpec
from truss.contexts.image_builder.cache_warmer import (
    AWSCredentials,
    parse_s3_credentials_file,
)
from truss.contexts.image_builder.image_builder import ImageBuilder
from truss.contexts.image_builder.util import (
    TRUSS_BASE_IMAGE_VERSION_TAG,
    file_is_not_empty,
    to_dotted_python_version,
    truss_base_image_name,
    truss_base_image_tag,
)
from truss.contexts.truss_context import TrussContext
from truss.truss_handle.patch.hash import directory_content_hash
from truss.util.jinja import read_template_from_fs
from truss.util.path import (
    build_truss_target_directory,
    copy_tree_or_file,
    copy_tree_path,
    load_trussignore_patterns,
)
from pydantic import BaseModel

BUILD_SERVER_DIR_NAME = "server"
BUILD_CONTROL_SERVER_DIR_NAME = "control"
BUILD_SERVER_EXTENSIONS_PATH = "extensions"
BUILD_CHAINS_DIR_NAME = "truss_chains"

BUILD_COMMANDS_TXT_FILENAME = "build_commands.txt"

CONFIG_FILE = "config.yaml"
USER_TRUSS_IGNORE_FILE = ".truss_ignore"
GCS_CREDENTIALS = "service_account.json"
S3_CREDENTIALS = "s3_credentials.json"

HF_ACCESS_TOKEN_FILE_NAME = "hf-access-token"

CLOUD_BUCKET_CACHE = Path("/app/model_cache/")
HF_SOURCE_DIR = Path("./root/.cache/huggingface/hub/")
HF_CACHE_DIR = Path("/root/.cache/huggingface/hub/")

JOB_IMAGE_SPEC_FILE = "job_image_spec.json"
JOB_DOCKERFILE_TEMPLATE_NAME = "job.Dockerfile.jinja"

class CustomImage(BaseModel):
    image: str
    docker_auth: Optional[DockerAuthSettings] = None

class DockerImage(BaseModel):
    base_image: CustomImage
    apt_requirements: List[str] = []
    pip_requirements: List[str] = []
    # do we need apt requirement and build commands?
    build_commands: List[str] = []

class SecretWithValue(BaseModel):
    name: str
    environment_variable: str
    value: str

class CodeBundle(BaseModel):
    local_dir_path: Path
    target_dir_path: Path

class FinetuneConfig(BaseModel):
    docker_image: DockerImage
    secrets: List[SecretWithValue] = []
    code_bundle: CodeBundle

class JobImageSpec:
    def __init__(self, truss_dir: Path):
        self._truss_dir = truss_dir
        with open(truss_dir / JOB_IMAGE_SPEC_FILE, "r") as f:
            json_spec = json.load(f)
        self._spec = FinetuneConfig.model_validate(json_spec)
    
    @property
    def spec(self) -> FinetuneConfig:
        return self._spec
        

class JobImageBuilder(ImageBuilder):
    def __init__(self, truss_dir: Path):
        self._truss_dir = truss_dir
        self._job_spec = JobImageSpec(truss_dir)
    
    def _copy_into_build_dir(
        self, from_path: Path, build_dir: Path, path_in_build_dir: str
    ):
        copy_tree_or_file(from_path, build_dir / path_in_build_dir)  # type: ignore[operator]

    def prepare_image_build_dir(self, build_dir: Optional[Path] = None):
        truss_dir = self._truss_dir
        spec = self._job_spec.spec
        build_dir = build_truss_target_directory("jobs")
        copy_tree_path(truss_dir, build_dir)

        # copy code bundle
        self._copy_into_build_dir(spec.code_bundle.local_dir_path, build_dir, spec.code_bundle.target_dir_path)

        # copy system requirements
        self._copy_into_build_dir(truss_dir / SYSTEM_PACKAGES_TXT_FILENAME, build_dir, SYSTEM_PACKAGES_TXT_FILENAME)

        # copy user pip requirements
        self._copy_into_build_dir(truss_dir / USER_SUPPLIED_REQUIREMENTS_TXT_FILENAME, build_dir, USER_SUPPLIED_REQUIREMENTS_TXT_FILENAME)

        # copy build commands
        self._copy_into_build_dir(truss_dir / BUILD_COMMANDS_TXT_FILENAME, build_dir, BUILD_COMMANDS_TXT_FILENAME)

        # copy secrets
        for secret in spec.secrets:
            self._copy_into_build_dir(truss_dir / secret.name, build_dir, secret.environment_variable)

        return self._render_dockerfile(build_dir)
    
    def _render_dockerfile(self, build_dir: Path):
        spec = self._job_spec.spec

        dockerfile_template = read_template_from_fs(
            TEMPLATES_DIR, JOB_DOCKERFILE_TEMPLATE_NAME
        )
        base_image_name_and_tag = spec.docker_image.base_image.image
        should_install_system_requirements = file_is_not_empty(
            build_dir / SYSTEM_PACKAGES_TXT_FILENAME
        )
        should_install_user_requirements_file = file_is_not_empty(
            build_dir / USER_SUPPLIED_REQUIREMENTS_TXT_FILENAME
        )
        should_run_build_commands = file_is_not_empty(
            build_dir / BUILD_COMMANDS_TXT_FILENAME
        )
        dockerfile_contents = dockerfile_template.render(
            should_install_system_requirements=should_install_system_requirements,
            base_image_name_and_tag=base_image_name_and_tag,
            should_install_user_requirements_file=should_install_user_requirements_file,
            should_run_build_commands=should_run_build_commands,
            build_commands=spec.docker_image.build_commands,
            bundled_code_dir=spec.code_bundle.target_dir_path,
            secrets=spec.secrets,
            **FILENAME_CONSTANTS_MAP,
        )
        docker_file_path = build_dir / MODEL_DOCKERFILE_NAME
        docker_file_path.write_text(dockerfile_contents)


