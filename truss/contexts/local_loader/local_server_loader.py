#!/usr/bin/env python3
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Optional

from truss.contexts.image_builder.serving_image_builder import ServingImageBuilder
from truss.contexts.local_loader.docker_build_emulator import DockerBuildEmulator
from truss.contexts.local_loader.truss_file_syncer import TrussFilesSyncer
from truss.patch.local_truss_patch_applier import LocalTrussPatchApplier
from truss.util.path import build_truss_shadow_target_directory


def create_venv_builder():
    import venv

    class VenvBuilder(venv.EnvBuilder):
        """Virtual Environment Builder

        This class handles setting up a virtual environment in `venv_dir`.
        """

        def __init__(self, venv_dir: Path, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.context: Any = None
            self.venv_dir = venv_dir

        def post_setup(self, context):
            """Implement base class `post_setup` hook to store venv context."""
            self.context = context

        def create_with_requirements(self, req_files: List[str]) -> None:
            """Create virtualenv with the requirements specified in the files provided.

            Args:
                `req_file` (List[str]): List of requirement files as subpaths to `self.venv_dir`.
            """
            from yaspin import yaspin

            with yaspin(text="Creating virtual environment") as spinner:
                env_dir = str(self.venv_dir / ".env")
                super().create(env_dir)
                spinner.ok("✅")
            with yaspin(text="Installing depedencies") as spinner:
                self._upgrade_pip()
                for req_file in req_files:
                    self._install_pip_requirements(self.venv_dir / req_file)
                spinner.ok("✅")

        def _upgrade_pip(self):
            """Helper function to upgrade pip."""
            subprocess.check_call(
                [
                    self.context.env_exe,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "pip",
                ],
                stdout=subprocess.PIPE,
            )

        def _install_pip_requirements(self, req_file: Path):
            """Helper function to install requirements from file."""
            if req_file.exists():
                pip_install_command = [
                    self.context.env_exe,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(req_file.absolute()),
                ]
                subprocess.check_call(
                    pip_install_command,
                    stdout=subprocess.PIPE,
                )

    return VenvBuilder


class LocalServerLoader:
    """Handle the setup and loading of truss server locally."""

    def __init__(
        self,
        truss_path: Path,
        context_builder: ServingImageBuilder,
        port: Optional[int] = None,
    ) -> None:
        self.truss_path = truss_path
        self.context_builder = context_builder
        self.port = port or 8080

    def watch(
        self,
        build_dir: Optional[Path] = None,
        venv_dir: Optional[Path] = None,
    ):
        """Run the server and watch for changes"""
        from yaspin import yaspin

        def _prep_or_create_dir(location: Optional[Path], stub: str) -> Path:
            if location is None:
                location = build_truss_shadow_target_directory(stub, self.truss_path)
            else:
                if not location.exists():
                    location.mkdir(parents=True)
            return location

        build_dir = _prep_or_create_dir(build_dir, "build")
        venv_dir = _prep_or_create_dir(venv_dir, "server_venv")

        with yaspin(text="Preparing truss context") as spinner:
            self.context_builder.prepare_image_build_dir(build_dir)
            dockerfile_path = build_dir / "Dockerfile"
            docker_build_emulator = DockerBuildEmulator(dockerfile_path, build_dir)
            build_result = docker_build_emulator.run(venv_dir)
            spinner.ok("✅")

        VenvBuilder = create_venv_builder()
        venv_builder = VenvBuilder(venv_dir, with_pip=True)
        requirements_files = [
            "app/requirements.txt",
            "requirements.txt",
        ]
        venv_builder.create_with_requirements(requirements_files)

        TrussFilesSyncer(
            self.truss_path,
            LocalTrussPatchApplier(
                venv_dir / "app/",
                venv_builder.context.env_exe,
                logging.Logger(__name__),
            ),
        ).start()

        subprocess.check_call(
            [
                venv_builder.context.env_exe,
                "-m",
                "uvicorn",
                "local_inference_server:app",
                "--reload",
                "--port",
                str(self.port),
            ],
            cwd=str(venv_dir / str(build_result.workdir).replace("/", "", 1)),
            env={
                **build_result.env,
                "SETUP_JSON_LOGGER": "False",
            },
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
