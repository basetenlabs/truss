#!/usr/bin/env python3
import subprocess
import sys
import venv
from pathlib import Path
from typing import Optional

from truss.contexts.image_builder.serving_image_builder import ServingImageBuilder
from truss.contexts.local_loader.docker_build_emulator import DockerBuildEmulator
from truss.contexts.local_loader.truss_file_watcher import TrussFilesWatcher
from truss.util.path import build_truss_target_directory


class EnvBuilder(venv.EnvBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context = None

    def post_setup(self, context):
        self.context = context


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
            build_dir = build_truss_target_directory("build_dir")
        else:
            if not build_dir.exists():
                build_dir.mkdir(parents=True)

        if venv_dir is None:
            venv_dir = build_truss_target_directory("venv")
        else:
            if not venv_dir.exists():
                venv_dir.mkdir(parents=True)

        self.context_builder.prepare_image_build_dir(build_dir)
        dockerfile_path = build_dir / "Dockerfile"
        docker_build = DockerBuildEmulator(dockerfile_path)
        docker_build.run(build_dir, venv_dir)

        execution_env_vars = docker_build.env_vars

        # print(f" *** Created temporary directory '{target_dir_path}'.")
        venv_builder = EnvBuilder(with_pip=True)
        venv_builder.create(str(venv_dir / ".env"))
        venv_context = venv_builder.context
        subprocess.check_call(
            [
                venv_context.env_exe,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
            ]
        )
        requirements_files = [
            "app/requirements.txt",
            "requirements.txt",
        ]
        for req_file in requirements_files:
            req_file_path = venv_dir / req_file

            if req_file_path.exists():
                print(f"Installing requirements for {req_file_path}")
                pip_install_command = [
                    venv_context.env_exe,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(req_file_path.absolute()),
                ]
                subprocess.check_call(pip_install_command)

        # Modify path to run in proper root
        venv_entry_point = [
            venv_context.env_exe,
            "-m",
            "uvicorn",
            "local_inference_server:app",
            "--reload",
            "--port",
            str(self.port),
        ]
        TrussFilesWatcher(self.truss_path, venv_dir / "app/").start()
        execution_env_vars = {
            **execution_env_vars,
            "SETUP_JSON_LOGGER": "False",
        }

        subprocess.check_call(
            venv_entry_point,
            cwd=str(venv_dir / str(docker_build.work_dir).replace("/", "", 1)),
            env=execution_env_vars,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
