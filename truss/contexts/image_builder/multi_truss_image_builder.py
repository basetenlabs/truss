from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from truss.constants import (
    MULTI_SERVER_DOCKERFILE_NAME,
    MULTI_SERVER_DOCKERFILE_TEMPLATE_NAME,
    REQUIREMENTS_TXT_FILENAME,
    SERVER_CODE_DIR,
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
from truss.multi_truss.spec import MultiTrussSpec
from truss.patch.hash import directory_content_hash
from truss.truss_spec import TrussSpec
from truss.utils import build_truss_target_directory, copy_tree_path

BUILD_SERVER_DIR_NAME = "server"
BUILD_CONTROL_SERVER_DIR_NAME = "control"


class MultiTrussImageBuilderContext(TrussContext):
    @staticmethod
    def run(dir: Path):
        return MultiTrussImageBuilder(dir)


class MultiTrussImageBuilder(ImageBuilder):
    def __init__(self, multi_truss_dir: Path) -> None:
        self._truss_dir = multi_truss_dir
        self._spec = MultiTrussSpec(multi_truss_dir)

    @property
    def default_tag(self):
        # TODO: generate meaningful tag
        return "mult-truss:latest"

    def prepare_image_build_dir(self, build_dir: Path = None):
        """Prepare a directory for building the docker image from.

        Returns:
            docker command to build the docker image.
        """
        if build_dir is None:
            # TODO: generate a real name here
            build_dir = build_truss_target_directory("multi-truss-build")
            # todo: Add a logging statement here, suggesting how to clean up the directory.

        # TODO: Split  out MultiTrussServer into a separate directory
        # For now, it's small enough to not matter
        copy_tree_path(
            SERVER_CODE_DIR,
            build_dir / BUILD_SERVER_DIR_NAME,
        )
        copy_tree_path(
            SHARED_SERVING_AND_TRAINING_CODE_DIR,
            build_dir
            / BUILD_SERVER_DIR_NAME
            / SHARED_SERVING_AND_TRAINING_CODE_DIR_NAME,
        )

        def handle_truss_context(spec: TrussSpec):
            model_build_dir: Path = build_dir / spec.name

            copy_tree_path(spec.truss_dir, model_build_dir)

            # TODO: handle server requirements
            server_reqs_filepath = (
                TEMPLATES_DIR / spec.model_framework_name / REQUIREMENTS_TXT_FILENAME
            )
            should_install_server_requirements = (
                server_reqs_filepath.exists()
                and file_is_not_empty(server_reqs_filepath)
            )
            if should_install_server_requirements:
                # Need to append files since this will be called multiple times
                server_requirements_text = ""
                with (
                    TEMPLATES_DIR
                    / spec.model_framework_name
                    / REQUIREMENTS_TXT_FILENAME
                ).open("r") as server_template_file:
                    server_requirements_text = "\n".join(
                        server_template_file.readlines()
                    )

                with (build_dir / SERVER_REQUIREMENTS_TXT_FILENAME).open(
                    "a"
                ) as req_file:
                    req_file.write(server_requirements_text)

            # Need to append files since this will be called multiple times
            with (build_dir / REQUIREMENTS_TXT_FILENAME).open("a") as req_file:
                req_file.write(spec.requirements_txt)

            with (build_dir / SYSTEM_PACKAGES_TXT_FILENAME).open("a") as req_file:
                req_file.write(spec.system_packages_txt)
            return spec.name

        model_build_dir_names = []
        for truss_path in self._spec.prepared_truss_dir_paths:
            model_build_dir_names.append(
                handle_truss_context(TrussSpec(truss_dir=truss_path))
            )
        print(build_dir)
        print(model_build_dir_names)

        template_loader = FileSystemLoader(str(TEMPLATES_DIR))
        template_env = Environment(loader=template_loader)
        dockerfile_template = template_env.get_template(
            MULTI_SERVER_DOCKERFILE_TEMPLATE_NAME
        )
        config = self._spec.config

        # TODO: update to different base image if needed
        base_image_name = truss_base_image_name(job_type="server")
        tag = truss_base_image_tag(
            python_version=to_dotted_python_version(config.python_version),
            use_gpu=config.resources.use_gpu,
            live_reload=False,
            version_tag=TRUSS_BASE_IMAGE_VERSION_TAG,
            # multi_server=True,
        )
        base_image_name_and_tag = f"{base_image_name}:{tag}"
        should_install_system_requirements = file_is_not_empty(
            build_dir / SYSTEM_PACKAGES_TXT_FILENAME
        )
        should_install_requirements = file_is_not_empty(
            build_dir / REQUIREMENTS_TXT_FILENAME
        )
        should_install_server_requirements = file_is_not_empty(
            build_dir / SERVER_REQUIREMENTS_TXT_FILENAME
        )
        dockerfile_contents = dockerfile_template.render(
            should_install_server_requirements=should_install_server_requirements,
            base_image_name_and_tag=base_image_name_and_tag,
            should_install_system_requirements=should_install_system_requirements,
            should_install_requirements=should_install_requirements,
            config=config,
            data_dir_exists=False,
            bundled_packages_dir_exists=False,
            truss_hash=directory_content_hash(self._truss_dir),
            # TODO(handle prepared dirs)
            model_dir_names=model_build_dir_names,
        )
        docker_file_path = build_dir / MULTI_SERVER_DOCKERFILE_NAME
        with docker_file_path.open("w") as docker_file:
            docker_file.write(dockerfile_contents)
