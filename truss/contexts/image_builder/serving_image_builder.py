from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from google.cloud import storage
from huggingface_hub import get_hf_file_metadata, hf_hub_url, list_repo_files
from huggingface_hub.utils import filter_repo_objects
from truss.constants import (
    BASE_SERVER_REQUIREMENTS_TXT_FILENAME,
    CONTROL_SERVER_CODE_DIR,
    MODEL_DOCKERFILE_NAME,
    REQUIREMENTS_TXT_FILENAME,
    SERVER_CODE_DIR,
    SERVER_DOCKERFILE_TEMPLATE_NAME,
    SERVER_REQUIREMENTS_TXT_FILENAME,
    SHARED_SERVING_AND_TRAINING_CODE_DIR,
    SHARED_SERVING_AND_TRAINING_CODE_DIR_NAME,
    SYSTEM_PACKAGES_TXT_FILENAME,
    TEMPLATES_DIR,
    TRITON_SERVER_CODE_DIR,
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
from truss.patch.hash import directory_content_hash
from truss.truss_config import Build, HuggingFaceModel, ModelServer, TrussConfig
from truss.truss_spec import TrussSpec
from truss.util.download import download_external_data
from truss.util.jinja import read_template_from_fs
from truss.util.path import (
    build_truss_target_directory,
    copy_tree_or_file,
    copy_tree_path,
)

BUILD_SERVER_DIR_NAME = "server"
BUILD_CONTROL_SERVER_DIR_NAME = "control"

CONFIG_FILE = "config.yaml"

HF_ACCESS_TOKEN_SECRET_NAME = "hf_access_token"
HF_ACCESS_TOKEN_FILE_NAME = "hf-access-token"


def create_triton_build_dir(config: TrussConfig, build_dir: Path, truss_dir: Path):
    _spec = TrussSpec(truss_dir)
    if not build_dir.exists():
        build_dir.mkdir(parents=True)

    # The triton server expects a specific directory structure. We create this directory structure
    # in the build directory. The structure is:
    #   build_dir
    #   ├── model # Name of "model", used during invocation
    #   │   └── 1 # Version of the model, used during invocation
    #   │       └── truss # User-defined code
    #   │           ├── config.yml
    #   │           ├── model.py
    #   │           ├── # other truss files
    #   │       └── model.py # Triton server code
    #   │       └── # other triton server files

    model_repository_path = build_dir / "model"
    user_truss_path = model_repository_path / "truss"  # type: ignore[operator]
    data_dir = model_repository_path / config.data_dir  # type: ignore[operator]

    copy_tree_path(TRITON_SERVER_CODE_DIR / "model", model_repository_path)
    copy_tree_path(TRITON_SERVER_CODE_DIR / "root", build_dir)
    copy_tree_path(truss_dir, user_truss_path)
    copy_tree_path(
        SHARED_SERVING_AND_TRAINING_CODE_DIR,
        model_repository_path / SHARED_SERVING_AND_TRAINING_CODE_DIR_NAME,
    )

    # Override config.yml
    with (model_repository_path / "truss" / CONFIG_FILE).open("w") as config_file:
        yaml.dump(config.to_dict(verbose=True), config_file)

    # Download external data
    download_external_data(_spec.external_data, data_dir)

    (build_dir / REQUIREMENTS_TXT_FILENAME).write_text(_spec.requirements_txt)
    (build_dir / SYSTEM_PACKAGES_TXT_FILENAME).write_text(_spec.system_packages_txt)


def split_gs_path(gs_path):
    # Remove the 'gs://' prefix
    path = gs_path.replace("gs://", "")

    # Split on the first slash
    parts = path.split("/", 1)

    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    return bucket_name, prefix


def list_bucket_files(bucket_name, data_dir, is_trusted=False):
    # TODO(varun): provide support for aws s3

    if is_trusted:
        storage_client = storage.Client.from_service_account_json(
            data_dir / "service_account.json"
        )
    else:
        storage_client = storage.Client()
    bucket_name, prefix = split_gs_path(bucket_name)
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    all_objects = []
    for blob in blobs:
        # leave out folders
        if blob.name[-1] == "/":
            continue
        all_objects.append(blob.name)

    return all_objects


def list_files(repo_id, data_dir, revision=None):
    if repo_id.startswith(("s3://", "gs://")):
        return list_bucket_files(repo_id, data_dir, is_trusted=True)
    else:
        # we assume it's a HF bucket
        return list_repo_files(repo_id, revision=revision)


def update_model_key(config: TrussConfig) -> str:
    server_name = config.build.model_server

    if server_name == ModelServer.TGI:
        return "model_id"
    elif server_name == ModelServer.VLLM:
        return "model"

    raise ValueError(
        f"Invalid server name (must be `TGI` or `VLLM`, not `{server_name}`)."
    )


def update_model_name(config: TrussConfig, model_key: str) -> str:
    if model_key not in config.build.arguments:
        # We should definitely just use the same key across both vLLM and TGI
        raise KeyError(
            "Key for model missing in config or incorrect key used. Use `model` for VLLM and `model_id` for TGI."
        )
    model_name = config.build.arguments[model_key]
    if "gs://" in model_name:
        # if we are pulling from a gs bucket, we want to alias it as a part of the cache
        model_to_cache = HuggingFaceModel(model_name)
        config.hf_cache.models.append(model_to_cache)

        config.build.arguments[
            model_key
        ] = f"/app/hf_cache/{model_name.replace('gs://', '')}"
    return model_name


def get_files_to_cache(config: TrussConfig, truss_dir: Path, build_dir: Path):
    def copy_into_build_dir(from_path: Path, path_in_build_dir: str):
        copy_tree_or_file(from_path, build_dir / path_in_build_dir)  # type: ignore[operator]

    model_files = {}
    cached_files: List[str] = []
    if config.hf_cache:
        curr_dir = Path(__file__).parent.resolve()
        copy_into_build_dir(curr_dir / "cache_warmer.py", "cache_warmer.py")
        for model in config.hf_cache.models:
            repo_id = model.repo_id
            revision = model.revision

            allow_patterns = model.allow_patterns
            ignore_patterns = model.ignore_patterns

            filtered_repo_files = list(
                filter_repo_objects(
                    items=list_files(
                        repo_id, truss_dir / config.data_dir, revision=revision
                    ),
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                )
            )

            cached_files = fetch_files_to_cache(
                cached_files, repo_id, filtered_repo_files
            )

            model_files[repo_id] = {"files": filtered_repo_files, "revision": revision}

    copy_into_build_dir(
        TEMPLATES_DIR / "cache_requirements.txt", "cache_requirements.txt"
    )
    return model_files, cached_files


def fetch_files_to_cache(cached_files: list, repo_id: str, filtered_repo_files: list):
    if "gs://" in repo_id:
        bucket_name, _ = split_gs_path(repo_id)
        repo_id = f"gs://{bucket_name}"

        for filename in filtered_repo_files:
            cached_files.append(f"/app/hf_cache/{bucket_name}/{filename}")
    else:
        repo_folder_name = f"models--{repo_id.replace('/', '--')}"
        for filename in filtered_repo_files:
            hf_url = hf_hub_url(repo_id, filename)
            hf_file_metadata = get_hf_file_metadata(hf_url)

            cached_files.append(f"{repo_folder_name}/blobs/{hf_file_metadata.etag}")

        # snapshots is just a set of folders with symlinks -- we can copy the entire thing separately
        cached_files.append(f"{repo_folder_name}/snapshots/")

        # refs just has files with revision commit hashes
        cached_files.append(f"{repo_folder_name}/refs/")

        cached_files.append("version.txt")

    return cached_files


def update_config_and_gather_files(
    config: TrussConfig, truss_dir: Path, build_dir: Path
):
    if config.build.model_server != ModelServer.TrussServer:
        model_key = update_model_key(config)
        update_model_name(config, model_key)
    return get_files_to_cache(config, truss_dir, build_dir)


def create_tgi_build_dir(
    config: TrussConfig, build_dir: Path, truss_dir: Path, use_hf_secret: bool
):
    copy_tree_path(truss_dir, build_dir)

    if not build_dir.exists():
        build_dir.mkdir(parents=True)

    model_files, cached_file_paths = update_config_and_gather_files(
        config, truss_dir, build_dir
    )

    hf_access_token = config.secrets.get(HF_ACCESS_TOKEN_SECRET_NAME)
    dockerfile_template = read_template_from_fs(
        TEMPLATES_DIR, "tgi/tgi.Dockerfile.jinja"
    )

    data_dir = build_dir / "data"
    credentials_file = data_dir / "service_account.json"
    dockerfile_content = dockerfile_template.render(
        hf_access_token=hf_access_token,
        models=model_files,
        hf_cache=config.hf_cache,
        data_dir_exists=data_dir.exists(),
        credentials_exists=credentials_file.exists(),
        cached_files=cached_file_paths,
        use_hf_secret=use_hf_secret,
        hf_access_token_file_name=HF_ACCESS_TOKEN_FILE_NAME,
    )
    dockerfile_filepath = build_dir / "Dockerfile"
    dockerfile_filepath.write_text(dockerfile_content)

    build_args = config.build.arguments.copy()
    endpoint = build_args.pop("endpoint", "generate_stream")

    nginx_template = read_template_from_fs(TEMPLATES_DIR, "tgi/proxy.conf.jinja")
    nginx_content = nginx_template.render(endpoint=endpoint)
    nginx_filepath = build_dir / "proxy.conf"
    nginx_filepath.write_text(nginx_content)

    args = " ".join([f"--{k.replace('_', '-')}={v}" for k, v in build_args.items()])
    supervisord_template = read_template_from_fs(
        TEMPLATES_DIR, "tgi/supervisord.conf.jinja"
    )
    supervisord_contents = supervisord_template.render(extra_args=args)
    supervisord_filepath = build_dir / "supervisord.conf"
    supervisord_filepath.write_text(supervisord_contents)


def create_vllm_build_dir(
    config: TrussConfig, build_dir: Path, truss_dir: Path, use_hf_secret
):
    copy_tree_path(truss_dir, build_dir)

    server_endpoint_config = {
        "Completions": "/v1/completions",
        "ChatCompletions": "/v1/chat/completions",
    }
    if not build_dir.exists():
        build_dir.mkdir(parents=True)

    build_config: Build = config.build
    server_endpoint = server_endpoint_config[build_config.arguments.pop("endpoint")]

    model_files, cached_file_paths = update_config_and_gather_files(
        config, truss_dir, build_dir
    )

    hf_access_token = config.secrets.get(HF_ACCESS_TOKEN_SECRET_NAME)
    dockerfile_template = read_template_from_fs(
        TEMPLATES_DIR, "vllm/vllm.Dockerfile.jinja"
    )
    nginx_template = read_template_from_fs(TEMPLATES_DIR, "vllm/proxy.conf.jinja")

    data_dir = build_dir / "data"
    credentials_file = data_dir / "service_account.json"
    dockerfile_content = dockerfile_template.render(
        hf_access_token=hf_access_token,
        models=model_files,
        should_install_server_requirements=True,
        hf_cache=config.hf_cache,
        data_dir_exists=data_dir.exists(),
        credentials_exists=credentials_file.exists(),
        cached_files=cached_file_paths,
        use_hf_secret=use_hf_secret,
        hf_access_token_file_name=HF_ACCESS_TOKEN_FILE_NAME,
    )
    dockerfile_filepath = build_dir / "Dockerfile"
    dockerfile_filepath.write_text(dockerfile_content)

    nginx_content = nginx_template.render(server_endpoint=server_endpoint)
    nginx_filepath = build_dir / "proxy.conf"
    nginx_filepath.write_text(nginx_content)

    args = " ".join(
        [f"--{k.replace('_', '-')}={v}" for k, v in build_config.arguments.items()]
    )
    supervisord_template = read_template_from_fs(
        TEMPLATES_DIR, "vllm/supervisord.conf.jinja"
    )
    supervisord_contents = supervisord_template.render(extra_args=args)
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

        if config.build.model_server is ModelServer.TGI:
            create_tgi_build_dir(config, build_dir, truss_dir, use_hf_secret)
            return
        elif config.build.model_server is ModelServer.VLLM:
            create_vllm_build_dir(config, build_dir, truss_dir, use_hf_secret)
            return
        elif config.build.model_server is ModelServer.TRITON:
            create_triton_build_dir(config, build_dir, truss_dir)
            return

        data_dir = build_dir / config.data_dir  # type: ignore[operator]

        def copy_into_build_dir(from_path: Path, path_in_build_dir: str):
            copy_tree_or_file(from_path, build_dir / path_in_build_dir)  # type: ignore[operator]

        # Copy over truss
        copy_tree_path(truss_dir, build_dir)

        # Override config.yml
        with (build_dir / CONFIG_FILE).open("w") as config_file:
            yaml.dump(config.to_dict(verbose=True), config_file)

        # Download external data
        download_external_data(self._spec.external_data, data_dir)

        # Download from HuggingFace
        model_files, cached_files = update_config_and_gather_files(
            config, truss_dir, build_dir
        )

        # Copy inference server code
        copy_into_build_dir(SERVER_CODE_DIR, BUILD_SERVER_DIR_NAME)
        copy_into_build_dir(
            SHARED_SERVING_AND_TRAINING_CODE_DIR,
            BUILD_SERVER_DIR_NAME + "/" + SHARED_SERVING_AND_TRAINING_CODE_DIR_NAME,
        )

        # Copy control server code
        if config.live_reload:
            copy_into_build_dir(CONTROL_SERVER_CODE_DIR, BUILD_CONTROL_SERVER_DIR_NAME)
            copy_into_build_dir(
                SHARED_SERVING_AND_TRAINING_CODE_DIR,
                BUILD_CONTROL_SERVER_DIR_NAME
                + "/control/"
                + SHARED_SERVING_AND_TRAINING_CODE_DIR_NAME,
            )

        # Copy base TrussServer requirements if supplied custom base image
        base_truss_server_reqs_filepath = SERVER_CODE_DIR / REQUIREMENTS_TXT_FILENAME
        if config.base_image:
            copy_into_build_dir(
                base_truss_server_reqs_filepath, BASE_SERVER_REQUIREMENTS_TXT_FILENAME
            )

        # Copy model framework specific requirements file
        server_reqs_filepath = (
            TEMPLATES_DIR / model_framework_name / REQUIREMENTS_TXT_FILENAME
        )
        should_install_server_requirements = file_is_not_empty(server_reqs_filepath)
        if should_install_server_requirements:
            copy_into_build_dir(server_reqs_filepath, SERVER_REQUIREMENTS_TXT_FILENAME)

        with open(base_truss_server_reqs_filepath, "r") as f:
            base_server_requirements = f.read()

        # If the user has provided python requirements,
        # append the truss server requirements, so that any conflicts
        # are detected and cause a build failure. If there are no
        # requirements provided, we just pass an empty string,
        # as there's no need to install anything.
        user_provided_python_requirements = (
            base_server_requirements + spec.requirements_txt
            if spec.requirements
            else ""
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
        )

    def _render_dockerfile(
        self,
        build_dir: Path,
        should_install_server_requirements: bool,
        model_files: Dict[str, Any],
        use_hf_secret: bool,
        cached_files: List[str],
    ):
        config = self._spec.config
        data_dir = build_dir / config.data_dir
        bundled_packages_dir = build_dir / config.bundled_packages_dir
        credentials_file = data_dir / "service_account.json"
        dockerfile_template = read_template_from_fs(
            TEMPLATES_DIR, SERVER_DOCKERFILE_TEMPLATE_NAME
        )
        python_version = to_dotted_python_version(config.python_version)
        if config.base_image:
            base_image_name_and_tag = config.base_image.image
        else:
            base_image_name = truss_base_image_name(job_type="server")
            tag = truss_base_image_tag(
                python_version=python_version,
                use_gpu=config.resources.use_gpu,
                version_tag=TRUSS_BASE_IMAGE_VERSION_TAG,
            )
            base_image_name_and_tag = f"{base_image_name}:{tag}"
        should_install_system_requirements = file_is_not_empty(
            build_dir / SYSTEM_PACKAGES_TXT_FILENAME
        )
        should_install_python_requirements = file_is_not_empty(
            build_dir / REQUIREMENTS_TXT_FILENAME
        )

        hf_access_token = config.secrets.get(HF_ACCESS_TOKEN_SECRET_NAME)
        dockerfile_contents = dockerfile_template.render(
            should_install_server_requirements=should_install_server_requirements,
            base_image_name_and_tag=base_image_name_and_tag,
            should_install_system_requirements=should_install_system_requirements,
            should_install_requirements=should_install_python_requirements,
            config=config,
            python_version=python_version,
            live_reload=config.live_reload,
            data_dir_exists=data_dir.exists(),
            bundled_packages_dir_exists=bundled_packages_dir.exists(),
            truss_hash=directory_content_hash(self._truss_dir),
            models=model_files,
            use_hf_secret=use_hf_secret,
            cached_files=cached_files,
            credentials_exists=credentials_file.exists(),
            hf_cache=len(config.hf_cache.models) > 0,
            hf_access_token=hf_access_token,
            hf_access_token_file_name=HF_ACCESS_TOKEN_FILE_NAME,
        )
        docker_file_path = build_dir / MODEL_DOCKERFILE_NAME
        docker_file_path.write_text(dockerfile_contents)
