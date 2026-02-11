"""Generate a TrainingProject from AutoSFT config for truss train sft."""

import json
import tempfile
from pathlib import Path
from typing import Optional

from truss.base import truss_config
from truss_train.definitions import (
    AutoSFT,
    CacheConfig,
    Compute,
    Image,
    SecretReference,
    TrainingJob,
    TrainingProject,
)
from truss_train.memory_scoping import scope_from_config
from truss_train.sft import get_framework


def generate_sft_project(sft_config: AutoSFT) -> Path:
    """
    Generate a training project directory from AutoSFT config.
    Framework (transformers, trl, megatron) is selected from config or auto-selected.
    When memory is set, accelerator and node_count are auto-scoped (H100/H200/multinode H100).
    Returns path to config.py in the generated directory.
    """
    # Apply memory scoping when memory or model_params_b is set and accelerator/node_count not specified
    if sft_config.memory is not None or sft_config.model_params_b is not None:
        scoped = scope_from_config(
            sft_config.model,
            sft_config.memory,
            model_params_b_override=sft_config.model_params_b,
        )
        if scoped is not None:
            acc_spec, node_count = scoped
            if sft_config.accelerator is None:
                sft_config.accelerator = (
                    f"{acc_spec.accelerator.value}:{acc_spec.count}"
                )
            if sft_config.node_count is None:
                sft_config.node_count = node_count

    framework = get_framework(sft_config)
    framework_output = framework.generate(sft_config)

    with tempfile.TemporaryDirectory(prefix="truss_sft_") as tmpdir:
        project_dir = Path(tmpdir)
        config_path = project_dir / "config.py"
        train_script_path = project_dir / "train.py"
        sft_config_path = project_dir / "sft_config.json"

        sft_config_path.write_text(
            json.dumps(sft_config.model_dump(mode="json"), indent=2)
        )
        # Write train script - use run.sh for shell scripts, train.py for Python
        if framework_output.train_script.startswith("#!/bin/bash"):
            run_script_path = project_dir / "run.sh"
            run_script_path.write_text(framework_output.train_script)
        else:
            train_script_path.write_text(framework_output.train_script)
        _write_data_loader(project_dir / "data_loader.py")

        training_project = _build_training_project(
            sft_config, framework_output
        )
        _write_config_py(config_path, training_project)

        persistent_dir = Path(tempfile.mkdtemp(prefix="truss_sft_"))
        for f in project_dir.iterdir():
            (persistent_dir / f.name).write_bytes(f.read_bytes())

        return persistent_dir / "config.py"


def _build_training_project(
    sft_config: AutoSFT,
    framework_output,
) -> TrainingProject:
    """Build TrainingProject from AutoSFT config and framework output."""
    project_name = sft_config.project_name or _default_project_name(sft_config.model)
    base_image = (
        sft_config.base_image
        or framework_output.base_image
        or "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"
    )

    if sft_config.accelerator:
        accelerator_spec = truss_config.AcceleratorSpec.model_validate(
            sft_config.accelerator
        )
    else:
        accelerator_spec = truss_config.AcceleratorSpec(
            accelerator=truss_config.Accelerator.H100, count=1
        )


    node_count = sft_config.node_count or 1

    runtime = framework_output.runtime
    # Multinode + cache requires require_cache_affinity=False (H200 2-node etc.)
    if node_count > 1 and runtime.cache_config and runtime.cache_config.enabled:
        cc = runtime.cache_config
        runtime = runtime.model_copy(
            update={
                "cache_config": CacheConfig(
                    enabled=cc.enabled,
                    enable_legacy_hf_mount=cc.enable_legacy_hf_mount,
                    require_cache_affinity=False,
                    mount_base_path=cc.mount_base_path,
                )
            }
        )

    training_job = TrainingJob(
        image=Image(base_image=base_image),
        compute=Compute(node_count=node_count, accelerator=accelerator_spec),
        runtime=runtime,
    )

    return TrainingProject(name=project_name, job=training_job)


def _default_project_name(model: str) -> str:
    """Derive project name from model (e.g. meta-llama/Llama-2-7b -> Llama-2-7b-SFT)."""
    name = model.split("/")[-1] if "/" in model else model
    return f"{name}-SFT"


def _serialize_env_vars(env_vars: dict) -> str:
    """Serialize environment_variables dict to valid Python code."""
    parts = []
    for k, v in env_vars.items():
        if isinstance(v, SecretReference):
            parts.append(f'    "{k}": definitions.SecretReference(name="{v.name}"),')
        else:
            escaped = str(v).replace("\\", "\\\\").replace('"', '\\"')
            parts.append(f'    "{k}": "{escaped}",')
    return "{\n" + "\n".join(parts) + "\n}"


def _write_config_py(config_path: Path, training_project: TrainingProject) -> None:
    """Write config.py that defines the TrainingProject."""
    acc = training_project.job.compute.accelerator
    acc_str = (
        f"truss_config.AcceleratorSpec(accelerator=truss_config.Accelerator.{acc.accelerator.name}, count={acc.count})"
        if acc and acc.accelerator
        else "None"
    )
    env_vars_str = _serialize_env_vars(
        training_project.job.runtime.environment_variables
    )
    cc = training_project.job.runtime.cache_config
    cache_config_str = (
        f"definitions.CacheConfig(enabled={cc.enabled}, require_cache_affinity={cc.require_cache_affinity})"
        if cc
        else "definitions.CacheConfig(enabled=True)"
    )
    content = f'''# Auto-generated from AutoSFT config
from truss_train import definitions
from truss.base import truss_config

training_project = definitions.TrainingProject(
    name="{training_project.name}",
    job=definitions.TrainingJob(
        image=definitions.Image(base_image="{training_project.job.image.base_image}"),
        compute=definitions.Compute(
            node_count={training_project.job.compute.node_count},
            accelerator={acc_str},
        ),
        runtime=definitions.Runtime(
            start_commands={training_project.job.runtime.start_commands!r},
            environment_variables={env_vars_str},
            cache_config={cache_config_str},
            checkpointing_config=definitions.CheckpointingConfig(enabled=True),
        ),
    ),
)
'''
    config_path.write_text(content)


def _write_data_loader(data_loader_path: Path) -> None:
    """Write data_loader.py (UniversalLLMLoader) into the project."""
    content = (Path(__file__).parent / "data_loader.py").read_text()
    data_loader_path.write_text(content)
