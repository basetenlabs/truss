from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from truss.constants import (
    REQUIREMENTS_TXT_FILENAME,
    SHARED_SERVING_AND_TRAINING_CODE_DIR,
    SHARED_SERVING_AND_TRAINING_CODE_DIR_NAME,
    SYSTEM_PACKAGES_TXT_FILENAME,
    TEMPLATES_DIR,
    TRAINING_DOCKERFILE_NAME,
    TRAINING_DOCKERFILE_TEMPLATE_NAME,
    TRAINING_JOB_WRAPPER_CODE_DIR,
    TRAINING_JOB_WRAPPER_CODE_DIR_NAME,
    TRAINING_REQUIREMENTS_TXT_FILENAME,
)
from truss.contexts.image_builder.image_builder import ImageBuilder
from truss.contexts.truss_context import TrussContext
from truss.truss_spec import TrussSpec
from truss.utils import build_truss_target_directory, copy_file_path, copy_tree_path

BUILD_TRAINING_DIR_NAME = "training"


class TrainingImageBuilderContext(TrussContext):
    @staticmethod
    def run(truss_dir: Path):
        return TrainingImageBuilder(truss_dir)


class TrainingImageBuilder(ImageBuilder):
    def __init__(self, truss_dir: Path) -> None:
        self._truss_dir = truss_dir
        self._spec = TrussSpec(truss_dir)

    @property
    def default_tag(self):
        return f"{self._spec.model_framework_name}-train:latest"

    def prepare_image_build_dir(self, build_dir: Path = None):
        """Prepare a directory for building the docker image from.

        Returns:
            docker command to build the docker image.
        """
        if build_dir is None:
            build_dir = build_truss_target_directory("train")

        copy_tree_path(self._spec.truss_dir, build_dir)

        if (self._spec.truss_dir / TRAINING_JOB_WRAPPER_CODE_DIR_NAME).exists():
            err_msg = (
                "Unable to build training image: training job "
                f"wrapper code directory name {TRAINING_JOB_WRAPPER_CODE_DIR_NAME}"
                " conflicts with an existing directory in the truss."
            )
            raise ValueError(err_msg)

        copy_tree_path(
            TRAINING_JOB_WRAPPER_CODE_DIR,
            build_dir / BUILD_TRAINING_DIR_NAME,
        )
        copy_tree_path(
            SHARED_SERVING_AND_TRAINING_CODE_DIR,
            build_dir
            / BUILD_TRAINING_DIR_NAME
            / SHARED_SERVING_AND_TRAINING_CODE_DIR_NAME,
        )

        copy_file_path(
            TEMPLATES_DIR
            / self._spec.model_framework_name
            / TRAINING_REQUIREMENTS_TXT_FILENAME,
            build_dir / TRAINING_REQUIREMENTS_TXT_FILENAME,
        )
        with (build_dir / REQUIREMENTS_TXT_FILENAME).open("w") as req_file:
            req_file.write(self._spec.requirements_txt)

        with (build_dir / SYSTEM_PACKAGES_TXT_FILENAME).open("w") as req_file:
            req_file.write(self._spec.system_packages_txt)

        bundled_packages_dir_exists = (
            build_dir / self._spec.config.bundled_packages_dir
        ).exists()
        template_loader = FileSystemLoader(str(TEMPLATES_DIR))
        template_env = Environment(loader=template_loader)
        dockerfile_template = template_env.get_template(
            TRAINING_DOCKERFILE_TEMPLATE_NAME
        )
        dockerfile_contents = dockerfile_template.render(
            config=self._spec.config,
            bundled_packages_dir_exists=bundled_packages_dir_exists,
        )
        docker_file_path = build_dir / TRAINING_DOCKERFILE_NAME
        with docker_file_path.open("w") as docker_file:
            docker_file.write(dockerfile_contents)
