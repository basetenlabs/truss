from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from huggingface_hub import list_repo_files
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

    def prepare_image_build_dir(self, build_dir: Optional[Path] = None):
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
        model_files = {}
        if config.hf_cache:
            curr_dir = Path(__file__).parent.resolve()
            copy_into_build_dir(curr_dir / "cache_warmer.py", "cache_warmer.py")
            for model in config.hf_cache.models:
                repo_id = model.repo_id
                revision = model.revision
                model_files[repo_id] = {
                    "files": list_repo_files(repo_id, revision=revision),
                    "revision": revision,
                }

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
        if config.base_image:
            base_truss_server_reqs_filepath = (
                SERVER_CODE_DIR / REQUIREMENTS_TXT_FILENAME
            )
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

        (build_dir / REQUIREMENTS_TXT_FILENAME).write_text(spec.requirements_txt)
        (build_dir / SYSTEM_PACKAGES_TXT_FILENAME).write_text(spec.system_packages_txt)

        self._render_dockerfile(
            build_dir, should_install_server_requirements, model_files
        )

    def _render_dockerfile(
        self,
        build_dir: Path,
        should_install_server_requirements: bool,
        model_files: Dict[str, Any],
    ):
        config = self._spec.config
        data_dir = build_dir / config.data_dir
        bundled_packages_dir = build_dir / config.bundled_packages_dir
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
        )
        docker_file_path = build_dir / MODEL_DOCKERFILE_NAME
        docker_file_path.write_text(dockerfile_contents)
