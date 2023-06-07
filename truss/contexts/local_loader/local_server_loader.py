#!/usr/bin/env python3
import subprocess
import sys
import venv
from pathlib import Path
from typing import Any, List, Optional

from truss.contexts.image_builder.serving_image_builder import ServingImageBuilder
from truss.contexts.local_loader.docker_build_emulator import DockerBuildEmulator
from truss.contexts.local_loader.truss_file_syncer import TrussFilesSyncer
from truss.util.path import build_truss_shadow_target_directory


class VenvBuilder(venv.EnvBuilder):
    def __init__(self, venv_dir: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context: Any = None
        self.venv_dir = venv_dir

    def post_setup(self, context):
        self.context = context

    def setup(self, req_files: List[str]) -> None:
        env_dir = str(self.venv_dir / ".env")
        super().create(env_dir)
        self._upgrade_pip()
        for req_file in req_files:
            self._install_pip_requirements(self.venv_dir / req_file)

    def _upgrade_pip(self):
        subprocess.check_call(
            [
                self.context.env_exe,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
            ]
        )

    def _install_pip_requirements(self, req_file: Path):
        if req_file.exists():
            pip_install_command = [
                self.context.env_exe,
                "-m",
                "pip",
                "install",
                "-r",
                str(req_file.absolute()),
            ]
            subprocess.check_call(pip_install_command)


class LocalServerLoader:
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
        if build_dir is None:
            build_dir = build_truss_shadow_target_directory(
                "build_dir", self.truss_path
            )
        else:
            if not build_dir.exists():
                build_dir.mkdir(parents=True)

        if venv_dir is None:
            venv_dir = build_truss_shadow_target_directory(
                "venv",
                self.truss_path,
            )
        else:
            if not venv_dir.exists():
                venv_dir.mkdir(parents=True)

        self.context_builder.prepare_image_build_dir(build_dir)
        dockerfile_path = build_dir / "Dockerfile"
        docker_build_emulator = DockerBuildEmulator(dockerfile_path, build_dir)
        build_result = docker_build_emulator.run(venv_dir)

        venv_builder = VenvBuilder(venv_dir, with_pip=True)
        requirements_files = [
            "app/requirements.txt",
            "requirements.txt",
        ]
        venv_builder.setup(requirements_files)

        TrussFilesSyncer(self.truss_path, venv_dir / "app/").start()

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
