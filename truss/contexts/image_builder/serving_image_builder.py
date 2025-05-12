from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import boto3
import packaging.version
import yaml
from botocore import UNSIGNED
from botocore.client import Config
from google.cloud import storage
from huggingface_hub import get_hf_file_metadata, hf_hub_url, list_repo_files
from huggingface_hub.utils import filter_repo_objects

from truss.base import constants, truss_config
from truss.base.constants import (
    BASE_SERVER_REQUIREMENTS_TXT_FILENAME,
    BEI_MAX_CONCURRENCY_TARGET_REQUESTS,
    BEI_REQUIRED_MAX_NUM_TOKENS,
    BEI_TRTLLM_BASE_IMAGE,
    BEI_TRTLLM_CLIENT_BATCH_SIZE,
    BEI_TRTLLM_PYTHON_EXECUTABLE,
    CHAINS_CODE_DIR,
    CONTROL_SERVER_CODE_DIR,
    DOCKER_SERVER_TEMPLATES_DIR,
    FILENAME_CONSTANTS_MAP,
    MODEL_CACHE_PATH,
    MODEL_DOCKERFILE_NAME,
    REQUIREMENTS_TXT_FILENAME,
    SERVER_CODE_DIR,
    SERVER_DOCKERFILE_TEMPLATE_NAME,
    SERVER_REQUIREMENTS_TXT_FILENAME,
    SHARED_SERVING_AND_TRAINING_CODE_DIR,
    SHARED_SERVING_AND_TRAINING_CODE_DIR_NAME,
    SUPPORTED_PYTHON_VERSIONS,
    SYSTEM_PACKAGES_TXT_FILENAME,
    TEMPLATES_DIR,
    TRTLLM_BASE_IMAGE,
    TRTLLM_PREDICT_CONCURRENCY,
    TRTLLM_PYTHON_EXECUTABLE,
    TRTLLM_TRUSS_DIR,
    TRUSS_BASE_IMAGE_NAME,
    TRUSS_CODE_DIR,
    TRUSSLESS_MAX_PAYLOAD_SIZE,
    USER_SUPPLIED_REQUIREMENTS_TXT_FILENAME,
)
from truss.base.trt_llm_config import TRTLLMConfiguration, TrussTRTLLMModel
from truss.base.truss_config import (
    DEFAULT_BUNDLED_PACKAGES_DIR,
    BaseImage,
    DockerServer,
    TrussConfig,
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
    truss_base_image_tag,
)
from truss.contexts.truss_context import TrussContext
from truss.truss_handle.patch.hash import directory_content_hash
from truss.util.basetenpointer import model_cache_hf_to_b10ptr
from truss.util.jinja import read_template_from_fs
from truss.util.path import (
    build_truss_target_directory,
    copy_tree_or_file,
    copy_tree_path,
    load_trussignore_patterns,
)

BUILD_SERVER_DIR_NAME = "server"
BUILD_CONTROL_SERVER_DIR_NAME = "control"
BUILD_SERVER_EXTENSIONS_PATH = "extensions"
BUILD_CHAINS_DIR_NAME = "truss_chains"
BUILD_TRUSS_DIR_NAME = "truss"

CONFIG_FILE = "config.yaml"
USER_TRUSS_IGNORE_FILE = ".truss_ignore"
GCS_CREDENTIALS = "service_account.json"
S3_CREDENTIALS = "s3_credentials.json"

HF_ACCESS_TOKEN_FILE_NAME = "hf-access-token"

CLOUD_BUCKET_CACHE = MODEL_CACHE_PATH

HF_SOURCE_DIR = Path("./root/.cache/huggingface/hub/")
HF_CACHE_DIR = Path("/root/.cache/huggingface/hub/")


class RemoteCache(ABC):
    def __init__(self, repo_name, data_dir, revision=None):
        self.repo_name = repo_name
        self.data_dir = data_dir
        self.revision = revision

    @staticmethod
    def from_repo(repo_name: str, data_dir: Path) -> "RemoteCache":
        repository_class: Type["RemoteCache"]
        if repo_name.startswith("gs://"):
            repository_class = GCSCache
        elif repo_name.startswith("s3://"):
            repository_class = S3Cache
        else:
            repository_class = HuggingFaceCache
        return repository_class(repo_name, data_dir)

    def filter(self, allow_patterns, ignore_patterns):
        return list(
            filter_repo_objects(
                items=self.list_files(revision=self.revision),
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
        )

    @abstractmethod
    def list_files(self, revision=None):
        pass

    @abstractmethod
    def prepare_for_cache(self, filenames):
        pass


class GCSCache(RemoteCache):
    def list_files(self, revision=None):
        gcs_credentials_file = self.data_dir / GCS_CREDENTIALS

        if gcs_credentials_file.exists():
            storage_client = storage.Client.from_service_account_json(
                gcs_credentials_file
            )
        else:
            storage_client = storage.Client.create_anonymous_client()

        bucket_name, prefix = split_path(self.repo_name, prefix="gs://")
        blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

        all_objects = []
        for blob in blobs:
            # leave out folders
            if blob.name[-1] == "/":
                continue
            all_objects.append(blob.name)

        return all_objects

    def prepare_for_cache(self, filenames):
        bucket_name, _ = split_path(self.repo_name, prefix="gs://")

        files_to_cache = []
        for filename in filenames:
            file_location = str(CLOUD_BUCKET_CACHE / bucket_name / filename)
            files_to_cache.append(CachedFile(source=file_location, dst=file_location))

        return files_to_cache


class S3Cache(RemoteCache):
    def list_files(self, revision=None) -> List[str]:
        s3_credentials_file = self.data_dir / S3_CREDENTIALS

        if s3_credentials_file.exists():
            s3_credentials: AWSCredentials = parse_s3_credentials_file(
                self.data_dir / S3_CREDENTIALS
            )
            session = boto3.Session(
                aws_access_key_id=s3_credentials.access_key_id,
                aws_secret_access_key=s3_credentials.secret_access_key,
                aws_session_token=s3_credentials.session_token,
                region_name=s3_credentials.region,
            )
            s3 = session.resource("s3")
        else:
            s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))

        # path may be like "folderA/folderB"
        bucket_name, path = split_path(self.repo_name, prefix="s3://")
        bucket = s3.Bucket(bucket_name)

        all_objects = []
        for blob in bucket.objects.filter(Prefix=path):
            # leave out folders
            if blob.key[-1] == "/":
                continue
            all_objects.append(blob.key)

        return all_objects

    def prepare_for_cache(self, filenames):
        bucket_name, _ = split_path(self.repo_name, prefix="s3://")

        files_to_cache = []
        for filename in filenames:
            file_location = str(CLOUD_BUCKET_CACHE / bucket_name / filename)
            files_to_cache.append(CachedFile(source=file_location, dst=file_location))

        return files_to_cache


def hf_cache_file_from_location(path: str):
    src_file_location = str(HF_SOURCE_DIR / path)
    dst_file_location = str(HF_CACHE_DIR / path)
    cache_file = CachedFile(source=src_file_location, dst=dst_file_location)
    return cache_file


class HuggingFaceCache(RemoteCache):
    def list_files(self, revision=None):
        return list_repo_files(self.repo_name, revision=revision)

    def prepare_for_cache(self, filenames):
        files_to_cache = []
        repo_folder_name = f"models--{self.repo_name.replace('/', '--')}"
        for filename in filenames:
            hf_url = hf_hub_url(self.repo_name, filename)
            hf_file_metadata = get_hf_file_metadata(hf_url)

            files_to_cache.append(
                hf_cache_file_from_location(
                    f"{repo_folder_name}/blobs/{hf_file_metadata.etag}"
                )
            )

        # snapshots is just a set of folders with symlinks -- we can copy the entire thing separately
        files_to_cache.append(
            hf_cache_file_from_location(f"{repo_folder_name}/snapshots/")
        )

        # refs just has files with revision commit hashes
        files_to_cache.append(hf_cache_file_from_location(f"{repo_folder_name}/refs/"))

        files_to_cache.append(hf_cache_file_from_location("version.txt"))

        return files_to_cache


def get_credentials_to_cache(data_dir: Path) -> List[str]:
    gcs_credentials_file = data_dir / GCS_CREDENTIALS
    s3_credentials_file = data_dir / S3_CREDENTIALS
    credentials = [gcs_credentials_file, s3_credentials_file]

    credentials_to_cache = []
    for file in credentials:
        if file.exists():
            build_path = Path(*file.parts[-2:])
            credentials_to_cache.append(str(build_path))

    return credentials_to_cache


def split_path(path, prefix="gs://"):
    # Remove the 'gs://' prefix
    path = path.replace(prefix, "")

    # Split on the first slash
    parts = path.split("/", 1)

    bucket_name = parts[0]
    path = parts[1] if len(parts) > 1 else ""

    return bucket_name, path


@dataclass
class CachedFile:
    source: str
    dst: str


def get_files_to_model_cache_v1(config: TrussConfig, truss_dir: Path, build_dir: Path):
    assert config.model_cache.is_v1

    def copy_into_build_dir(from_path: Path, path_in_build_dir: str):
        copy_tree_or_file(from_path, build_dir / path_in_build_dir)  # type: ignore[operator]

    remote_model_files = {}
    local_files_to_cache: List[CachedFile] = []

    curr_dir = Path(__file__).parent.resolve()
    copy_into_build_dir(curr_dir / "cache_warmer.py", "cache_warmer.py")
    for model in config.model_cache.models:
        repo_id = model.repo_id
        revision = model.revision

        allow_patterns = model.allow_patterns
        ignore_patterns = model.ignore_patterns

        model_cache = RemoteCache.from_repo(repo_id, truss_dir / config.data_dir)
        remote_filtered_files = model_cache.filter(allow_patterns, ignore_patterns)
        local_files_to_cache += model_cache.prepare_for_cache(remote_filtered_files)

        remote_model_files[repo_id] = {
            "files": remote_filtered_files,
            "revision": revision,
        }

    copy_into_build_dir(
        TEMPLATES_DIR / "cache_requirements.txt", "cache_requirements.txt"
    )
    return remote_model_files, local_files_to_cache


def build_model_cache_v2_and_copy_bptr_manifest(config: TrussConfig, build_dir: Path):
    assert config.model_cache.is_v2
    # builds BasetenManifest for caching
    basetenpointers = model_cache_hf_to_b10ptr(config.model_cache)
    # write json of bastenpointers into build dir
    with open(build_dir / "bptr-manifest", "w") as f:
        f.write(basetenpointers.model_dump_json())


def generate_docker_server_nginx_config(build_dir, config):
    nginx_template = read_template_from_fs(
        DOCKER_SERVER_TEMPLATES_DIR, "proxy.conf.jinja"
    )

    assert config.docker_server.predict_endpoint is not None, (
        "docker_server.predict_endpoint is required to use custom server"
    )
    assert config.docker_server.server_port is not None, (
        "docker_server.server_port is required to use custom server"
    )
    assert config.docker_server.readiness_endpoint is not None, (
        "docker_server.readiness_endpoint is required to use custom server"
    )
    assert config.docker_server.liveness_endpoint is not None, (
        "docker_server.liveness_endpoint is required to use custom server"
    )

    nginx_content = nginx_template.render(
        server_endpoint=config.docker_server.predict_endpoint,
        readiness_endpoint=config.docker_server.readiness_endpoint,
        liveness_endpoint=config.docker_server.liveness_endpoint,
        server_port=config.docker_server.server_port,
        client_max_body_size=TRUSSLESS_MAX_PAYLOAD_SIZE,
    )
    nginx_filepath = build_dir / "proxy.conf"
    nginx_filepath.write_text(nginx_content)


def generate_docker_server_supervisord_config(build_dir, config):
    supervisord_template = read_template_from_fs(
        DOCKER_SERVER_TEMPLATES_DIR, "supervisord.conf.jinja"
    )
    assert config.docker_server.start_command is not None, (
        "docker_server.start_command is required to use custom server"
    )
    supervisord_contents = supervisord_template.render(
        start_command=config.docker_server.start_command
    )
    supervisord_filepath = build_dir / "supervisord.conf"
    supervisord_filepath.write_text(supervisord_contents)


class ServingImageBuilderContext(TrussContext):
    @staticmethod
    def run(truss_dir: Path):
        return ServingImageBuilder(truss_dir)


class ServingImageBuilder(ImageBuilder):
    def __init__(self, truss_dir: Path) -> None:
        self._truss_dir = truss_dir
        self._spec = TrussSpec(truss_dir)

    @property
    def default_tag(self):
        return f"{self._spec.model_framework_name}-model:latest"

    def _copy_into_build_dir(
        self, from_path: Path, build_dir: Path, path_in_build_dir: str
    ):
        copy_tree_or_file(from_path, build_dir / path_in_build_dir)  # type: ignore[operator]

    def prepare_trtllm_bei_encoder_build_dir(self, build_dir: Path):
        """prepares the build directory for a trtllm ENCODER model to launch a Baseten Embeddings Inference (BEI) server"""
        config = self._spec.config
        assert (
            config.trt_llm
            and config.trt_llm.build
            and config.trt_llm.build.base_model == TrussTRTLLMModel.ENCODER
        ), (
            "prepare_trtllm_bei_encoder_build_dir should only be called for ENCODER tensorrt-llm model"
        )
        # TRTLLM has performance degradation with batch size >> 32, so we limit the runtime settings
        # runtime batch size may not be higher than what the build settings of the model allow
        # to 32 even if the engine.rank0 allows for higher batch_size
        runtime_max_batch_size = min(config.trt_llm.build.max_batch_size, 32)
        # make sure the user gets good performance, enforcing max_num_tokens here and in engine-builder
        runtime_max_batch_tokens = max(
            config.trt_llm.build.max_num_tokens, BEI_REQUIRED_MAX_NUM_TOKENS
        )
        port = 7997
        start_command = " ".join(
            [
                "truss-transfer-cli && text-embeddings-router",
                f"--port {port}",
                # assert the max_batch_size is within trt-engine limits
                f"--max-batch-requests {runtime_max_batch_size}",
                # assert the max_num_tokens is within trt-engine limits
                f"--max-batch-tokens {runtime_max_batch_tokens}",
                # how many sentences can be in a single json payload.
                # limited default to improve request based autoscaling.
                f"--max-client-batch-size {BEI_TRTLLM_CLIENT_BATCH_SIZE}",
                # how many concurrent requests can be handled by the server until 429 is returned.
                # limited by https://docs.baseten.co/performance/concurrency#concurrency-target
                # 2048 is a safe max value for the server
                f"--max-concurrent-requests {BEI_MAX_CONCURRENCY_TARGET_REQUESTS}",
            ]
        )
        self._spec.config.docker_server = DockerServer(
            start_command=f"/bin/sh -c '{start_command}'",
            server_port=port,
            # mount the following predict endpoint location
            predict_endpoint=config.trt_llm.runtime.webserver_default_route
            or "/v1/embeddings",
            readiness_endpoint="/health",
            liveness_endpoint="/health",
        )
        copy_tree_path(DOCKER_SERVER_TEMPLATES_DIR, build_dir, ignore_patterns=[])

        # Flex builds fill in the latest image during `docker_build_setup` on the
        # baseten backend. So only the image is not set, we use the constant
        # `BEI_TRTLLM_BASE_IMAGE` bundled in this context builder. If everyone uses flex
        # builds, we can remove the constant and setting the image here.
        if not (
            config.base_image and config.base_image.image.startswith("baseten/bei")
        ):
            config.base_image = BaseImage(
                image=BEI_TRTLLM_BASE_IMAGE,
                python_executable_path=BEI_TRTLLM_PYTHON_EXECUTABLE,
            )

    def prepare_trtllm_decoder_build_dir(self, build_dir: Path):
        """prepares the build directory for a trtllm decoder-like models to launch BRITON server"""
        config = self._spec.config
        assert (
            config.trt_llm
            and config.trt_llm.build
            and config.trt_llm.build.base_model != TrussTRTLLMModel.ENCODER
        ), (
            "prepare_trtllm_decoder_build_dir should only be called for decoder tensorrt-llm model"
        )

        # trt_llm is treated as an extension at model run time.
        self._copy_into_build_dir(
            TRTLLM_TRUSS_DIR / "src",
            build_dir,
            f"{BUILD_SERVER_DIR_NAME}/{BUILD_SERVER_EXTENSIONS_PATH}/trt_llm",
        )
        # TODO(pankaj) Do this differently. This is not ideal, user
        # supplied code in bundled packages can conflict with those from
        # the trtllm extension. We don't want to put this in the build
        # directory directly either because of chances of conflict there
        # as well and the noise it can create there. We need to find a
        # new place that's made available in model's pythonpath. This is
        # a bigger lift and feels overkill right now. Worth revisiting
        # if we come across cases of actual conflicts.
        self._copy_into_build_dir(
            TRTLLM_TRUSS_DIR / DEFAULT_BUNDLED_PACKAGES_DIR,
            build_dir,
            DEFAULT_BUNDLED_PACKAGES_DIR,
        )

        config.runtime.predict_concurrency = TRTLLM_PREDICT_CONCURRENCY
        # Flex builds fill in the latest image during `docker_build_setup` on the
        # baseten backend. So only the image is not set, we use the constant
        # `TRTLLM_BASE_IMAGE` bundled in this context builder. If everyone uses flex
        # builds, we can remove the constant and setting the image here.
        if not (
            config.base_image
            and config.base_image.image.startswith("baseten/briton-server:")
        ):
            config.base_image = BaseImage(
                image=TRTLLM_BASE_IMAGE, python_executable_path=TRTLLM_PYTHON_EXECUTABLE
            )

    def prepare_image_build_dir(
        self, build_dir: Optional[Path] = None, use_hf_secret: bool = False
    ):
        """
        Prepare a directory for building the docker image from.
        """
        truss_dir = self._truss_dir
        spec = self._spec
        config = spec.config
        model_framework_name = spec.model_framework_name
        if build_dir is None:
            # TODO(pankaj) We probably don't need model framework specific directory.
            build_dir = build_truss_target_directory(model_framework_name)

        data_dir = build_dir / config.data_dir  # type: ignore[operator]

        truss_ignore_patterns = []
        if (truss_dir / USER_TRUSS_IGNORE_FILE).exists():
            truss_ignore_patterns = load_trussignore_patterns(
                truss_dir / USER_TRUSS_IGNORE_FILE
            )

        # Copy over truss
        copy_tree_path(truss_dir, build_dir, ignore_patterns=truss_ignore_patterns)
        if (
            isinstance(config.trt_llm, TRTLLMConfiguration)
            and config.trt_llm.build is not None
        ):
            if config.trt_llm.build.base_model == TrussTRTLLMModel.ENCODER:
                # Run the specific encoder build
                self.prepare_trtllm_bei_encoder_build_dir(build_dir=build_dir)
            else:
                self.prepare_trtllm_decoder_build_dir(build_dir=build_dir)

        if config.docker_server is not None:
            self._copy_into_build_dir(
                TEMPLATES_DIR / "docker_server_requirements.txt",
                build_dir,
                "docker_server_requirements.txt",
            )

            generate_docker_server_nginx_config(build_dir, config)

            generate_docker_server_supervisord_config(build_dir, config)

        # Override config.yml
        with (build_dir / CONFIG_FILE).open("w") as config_file:
            yaml.dump(config.to_dict(verbose=True), config_file)

        external_data_files: list = []
        data_dir = Path("/app/data/")
        if self._spec.external_data is not None:
            for ext_file in self._spec.external_data.items:
                external_data_files.append(
                    (ext_file.url, (data_dir / ext_file.local_data_path).resolve())
                )

        # No model cache provided, initialize empty
        model_files = {}
        cached_files = []
        if config.model_cache.is_v1:
            logging.warning(
                "`model_cache` with `use_volume=False` (legacy) is deprecated. This will bake the model weights into the image."
                "We recommend upgrading to the pattern of using `use_volume=True`, keeping the weights outside of the container."
                "read more on the migration guide here: https://docs.baseten.co/development/model/model-cache"
                f"Config: {config.model_cache}"
            )
            # bakes model weights into the image
            model_files, cached_files = get_files_to_model_cache_v1(
                config, truss_dir, build_dir
            )

        if config.model_cache.is_v2:
            if config.trt_llm:
                raise RuntimeError(
                    "TensorRTLLM models is already occupying and using `model_cache` by default. "
                    "Additional huggingface weights are not allowed. "
                    "Feel free to reach out to us if you need this feature."
                )
            logging.info(
                f"`model_cache` with `use_volume=True` is enabled. Creating {config.model_cache}"
            )
            # adds a lazy pointer, will be downloaded at runtimes
            build_model_cache_v2_and_copy_bptr_manifest(
                config=config, build_dir=build_dir
            )

        # Copy inference server code
        self._copy_into_build_dir(SERVER_CODE_DIR, build_dir, BUILD_SERVER_DIR_NAME)
        self._copy_into_build_dir(
            SHARED_SERVING_AND_TRAINING_CODE_DIR,
            build_dir,
            BUILD_SERVER_DIR_NAME + "/" + SHARED_SERVING_AND_TRAINING_CODE_DIR_NAME,
        )

        # Copy control server code
        if config.live_reload:
            self._copy_into_build_dir(
                CONTROL_SERVER_CODE_DIR, build_dir, BUILD_CONTROL_SERVER_DIR_NAME
            )
            self._copy_into_build_dir(
                SHARED_SERVING_AND_TRAINING_CODE_DIR,
                build_dir,
                BUILD_CONTROL_SERVER_DIR_NAME
                + "/control/"
                + SHARED_SERVING_AND_TRAINING_CODE_DIR_NAME,
            )

        if config.use_local_src:
            self._copy_into_build_dir(CHAINS_CODE_DIR, build_dir, BUILD_CHAINS_DIR_NAME)
            self._copy_into_build_dir(TRUSS_CODE_DIR, build_dir, BUILD_TRUSS_DIR_NAME)

        # Copy base TrussServer requirements if supplied custom base image
        base_truss_server_reqs_filepath = SERVER_CODE_DIR / REQUIREMENTS_TXT_FILENAME
        if config.base_image:
            self._copy_into_build_dir(
                base_truss_server_reqs_filepath,
                build_dir,
                BASE_SERVER_REQUIREMENTS_TXT_FILENAME,
            )

        # Copy model framework specific requirements file
        server_reqs_filepath = (
            TEMPLATES_DIR / model_framework_name / REQUIREMENTS_TXT_FILENAME
        )
        should_install_server_requirements = file_is_not_empty(server_reqs_filepath)
        if should_install_server_requirements:
            self._copy_into_build_dir(
                server_reqs_filepath, build_dir, SERVER_REQUIREMENTS_TXT_FILENAME
            )

        with open(base_truss_server_reqs_filepath, "r") as f:
            base_server_requirements = f.read()

        if config.docker_server:
            # when docker server is enabled, no need to install truss requirements
            # only install user-provided python requirements
            user_provided_python_requirements = spec.requirements_txt
        else:
            # If the user has provided python requirements,
            # append the truss server requirements, so that any conflicts
            # are detected and cause a build failure. If there are no
            # requirements provided, we just pass an empty string,
            # as there's no need to install anything.
            # TODO (BT-10217): above reasoning leads to inconsistencies. To get consistent
            #  images tentatively add server requirements always. This whole point needs
            #  more thought and potentially a re-design.
            user_provided_python_requirements = (
                base_server_requirements + spec.requirements_txt
                if spec.requirements
                else base_server_requirements
            )
        if spec.requirements_file is not None:
            self._copy_into_build_dir(
                truss_dir / spec.requirements_file,
                build_dir,
                USER_SUPPLIED_REQUIREMENTS_TXT_FILENAME,
            )
        (build_dir / REQUIREMENTS_TXT_FILENAME).write_text(
            user_provided_python_requirements
        )
        (build_dir / SYSTEM_PACKAGES_TXT_FILENAME).write_text(spec.system_packages_txt)

        self._render_dockerfile(
            build_dir,
            should_install_server_requirements,
            model_files,
            use_hf_secret,
            cached_files,
            external_data_files,
            self._spec.build_commands,
        )

    def _render_dockerfile(
        self,
        build_dir: Path,
        should_install_server_requirements: bool,
        model_files: Dict[str, Any],
        use_hf_secret: bool,
        cached_files: List[str],
        external_data_files: List[Tuple[str, str]],
        build_commands: List[str],
    ):
        config = self._spec.config
        data_dir = build_dir / config.data_dir
        model_dir = build_dir / config.model_module_dir
        bundled_packages_dir = build_dir / config.bundled_packages_dir
        dockerfile_template = read_template_from_fs(
            TEMPLATES_DIR, SERVER_DOCKERFILE_TEMPLATE_NAME
        )
        python_version = truss_config.to_dotted_python_version(config.python_version)
        if config.base_image:
            base_image_name_and_tag = config.base_image.image
        else:
            tag = truss_base_image_tag(
                python_version=python_version,
                use_gpu=config.resources.use_gpu,  # type: ignore  # computed field.
                version_tag=TRUSS_BASE_IMAGE_VERSION_TAG,
            )
            base_image_name_and_tag = f"{TRUSS_BASE_IMAGE_NAME}:{tag}"
        should_install_system_requirements = file_is_not_empty(
            build_dir / SYSTEM_PACKAGES_TXT_FILENAME
        )
        should_install_python_requirements = file_is_not_empty(
            build_dir / REQUIREMENTS_TXT_FILENAME
        )
        should_install_user_requirements_file = file_is_not_empty(
            build_dir / USER_SUPPLIED_REQUIREMENTS_TXT_FILENAME
        )

        min_py_version = packaging.version.parse(SUPPORTED_PYTHON_VERSIONS[0])
        max_py_version = packaging.version.parse(SUPPORTED_PYTHON_VERSIONS[-1])

        hf_access_token = config.secrets.get(constants.HF_ACCESS_TOKEN_KEY)
        dockerfile_contents = dockerfile_template.render(
            should_install_server_requirements=should_install_server_requirements,
            base_image_name_and_tag=base_image_name_and_tag,
            max_supported_python_version_in_custom_base_image=max_py_version,
            min_supported_python_version_in_custom_base_image=min_py_version,
            max_supported_python_minor_version_in_custom_base_image=max_py_version.minor,
            min_supported_python_minor_version_in_custom_base_image=min_py_version.minor,
            supported_python_major_version_in_custom_base_image=min_py_version.major,
            should_install_system_requirements=should_install_system_requirements,
            should_install_requirements=should_install_python_requirements,
            should_install_user_requirements_file=should_install_user_requirements_file,
            config=config,
            python_version=python_version,
            control_python_version=SUPPORTED_PYTHON_VERSIONS[-1],  # Use highest.
            live_reload=config.live_reload,
            data_dir_exists=data_dir.exists(),
            model_dir_exists=model_dir.exists(),
            bundled_packages_dir_exists=bundled_packages_dir.exists(),
            truss_hash=directory_content_hash(
                self._truss_dir, self._spec.hash_ignore_patterns
            ),
            models=model_files,
            use_hf_secret=use_hf_secret,
            cached_files=cached_files,
            credentials_to_cache=get_credentials_to_cache(data_dir),
            model_cache_v1=config.model_cache.is_v1,
            model_cache_v2=config.model_cache.is_v2,
            hf_access_token=hf_access_token,
            hf_access_token_file_name=HF_ACCESS_TOKEN_FILE_NAME,
            external_data_files=external_data_files,
            build_commands=build_commands,
            use_local_src=config.use_local_src,
            **FILENAME_CONSTANTS_MAP,
        )
        # Consolidate repeated empty lines to single empty lines.
        dockerfile_contents = re.sub(
            r"(\r?\n){3,}", r"\n\n", dockerfile_contents
        ).strip()
        docker_file_path = build_dir / MODEL_DOCKERFILE_NAME
        docker_file_path.write_text(dockerfile_contents)
