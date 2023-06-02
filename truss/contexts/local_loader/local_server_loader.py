#!/usr/bin/env python3
import subprocess
import sys
import venv
from pathlib import Path
from typing import Dict, List, Optional

import dockerfile
from truss.contexts.image_builder.serving_image_builder import ServingImageBuilder
from truss.util.path import build_truss_target_directory, copy_path


class EnvBuilder(venv.EnvBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context = None

    def post_setup(self, context):
        self.context = context


class DockerBuildEmulator:
    def __init__(self, dockerfile_path: Path) -> None:
        self.commands = dockerfile.parse_file(str(dockerfile_path))
        self.env_vars: Dict[str, str] = {}
        self.entry_point: List[str] = []

    def run(self, context_dir: Path, fs_root_dir: Path):
        for cmd in self.commands:
            if cmd.cmd == "ENV":
                values = cmd.value
                self.env_vars[values[0]] = values[1]
            if cmd.cmd == "ENTRYPOINT":
                self.entry_point = list(cmd.value)
            if cmd.cmd == "COPY":
                # symlink to path
                src, dst = cmd.value
                src = src.replace("./", "", 1)
                dst = dst.replace("/", "", 1)
                copy_path(context_dir / src, fs_root_dir / dst)
        pass


class LocalServerLoader:
    def __init__(self, context_builder: ServingImageBuilder) -> None:
        self.context_builder = context_builder

    def watch(self, build_dir: Optional[Path] = None, venv_dir: Optional[Path] = None):
        if build_dir is None:
            build_dir = build_truss_target_directory("build_dir")

        if venv_dir is None:
            venv_dir = build_truss_target_directory("venv")

        self.context_builder.prepare_image_build_dir(build_dir)
        dockerfile_path = build_dir / "Dockerfile"
        docker_build = DockerBuildEmulator(dockerfile_path)
        docker_build.run(build_dir, venv_dir)

        execution_env_vars = docker_build.env_vars

        # print(f" *** Created temporary directory '{target_dir_path}'.")
        venv_builder = EnvBuilder(with_pip=True)
        venv_builder.create(str(venv_dir / ".env"))
        venv_context = venv_builder.context

        requirements_files = [
            "server/requirements.txt",
            "control/requirements.txt",
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
        # Drop python cmd from entrypoint
        _ = docker_build.entry_point.pop(0)
        f_path = docker_build.entry_point.pop(0)
        f_path = f_path.replace("/", "", 1)
        venv_entry_point = [
            venv_context.env_exe,
            f_path,
            *docker_build.entry_point,
        ]
        print(venv_entry_point)
        subprocess.check_call(
            venv_entry_point,
            cwd=str(venv_dir),
            env=execution_env_vars,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
