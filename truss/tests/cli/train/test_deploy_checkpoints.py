import os
from pathlib import Path

import truss_train.definitions as definitions
from truss.base import truss_config
from truss.cli.train.deploy_checkpoints import _render_vllm_lora_truss_config


def test_render_vllm_lora_truss_config():
    training_job_id = "kowpeqj"
    deploy_config = definitions.CheckpointDeployConfig(
        checkpoint_details=definitions.CheckpointDetails(
            checkpoints=[
                definitions.Checkpoint(
                    id="checkpoint-1", name="checkpoint-1", lora_rank=16
                )
            ],
            base_model_id="google/gemma-3-27b-it",
        ),
        model_name="gemma-3-27b-it-vLLM-LORA",
        compute=definitions.Compute(
            accelerator=truss_config.AcceleratorSpec(accelerator="H100", count=4)
        ),
        runtime=definitions.CheckpointDeployRuntime(
            environment_variables={"HF_TOKEN": "hf_access_token"}
        ),
    )
    rendered_truss = _render_vllm_lora_truss_config(training_job_id, deploy_config)
    test_truss = truss_config.TrussConfig.from_yaml(
        Path(
            os.path.dirname(__file__),
            "resources/test_deploy_from_checkpoint_config.yml",
        )
    )
    assert test_truss.model_name == rendered_truss.model_name
    assert (
        test_truss.training_checkpoints.checkpoints[0].id
        == rendered_truss.training_checkpoints.checkpoints[0].id
    )
    assert (
        test_truss.training_checkpoints.checkpoints[0].name
        == rendered_truss.training_checkpoints.checkpoints[0].name
    )
    assert (
        test_truss.docker_server.start_command
        == rendered_truss.docker_server.start_command
    )
    assert test_truss.resources.accelerator == rendered_truss.resources.accelerator
    assert test_truss.secrets == rendered_truss.secrets
    assert test_truss.training_checkpoints == rendered_truss.training_checkpoints
