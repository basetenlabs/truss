import pytest
from pydantic import ValidationError

from truss.base import truss_config
from truss_train.definitions import (
    AvailabilityModel,
    AWSAssumeRoleDockerAuth,
    AWSOIDCDockerAuth,
    BasetenCheckpoint,
    CheckpointList,
    Compute,
    DockerAuth,
    GCPOIDCDockerAuth,
    Image,
    LoadCheckpointConfig,
    LoopsCheckpoint,
    LoRACheckpoint,
    ModelWeightsFormat,
    Runtime,
    TrainingJob,
)


class TestLoopsCheckpoint:
    def test_from_checkpoint_defaults_to_trainer_target(self):
        ckpt = LoopsCheckpoint.from_checkpoint(
            run_id="run123", checkpoint_name="step-100"
        )
        assert ckpt.model_dump() == {
            "typ": "loops_checkpoint",
            "run_id": "run123",
            "checkpoint_name": "step-100",
            "target": "trainer",
        }

    def test_serializes_in_load_checkpoint_config(self):
        config = LoadCheckpointConfig(
            enabled=True,
            checkpoints=[
                LoopsCheckpoint.from_checkpoint(
                    run_id="run123", checkpoint_name="step-100", target="sampler"
                )
            ],
        )
        dumped = config.model_dump()
        assert dumped["checkpoints"][0]["typ"] == "loops_checkpoint"
        assert dumped["checkpoints"][0]["target"] == "sampler"

    def test_invalid_target_raises(self):
        with pytest.raises(ValidationError):
            LoopsCheckpoint.from_checkpoint(
                run_id="run123", checkpoint_name="step-100", target="bogus"
            )

    def test_mixed_checkpoint_types_round_trip(self):
        """A Baseten and a Loops checkpoint in the same list each keep their
        own discriminator through serialization (the SDK union is matched
        left-to-right, so a regression here would silently mis-tag entries)."""
        config = LoadCheckpointConfig(
            enabled=True,
            checkpoints=[
                BasetenCheckpoint.from_named_checkpoint(
                    checkpoint_name="step-1", job_id="lqz4pw4"
                ),
                LoopsCheckpoint.from_checkpoint(
                    run_id="run123", checkpoint_name="step-100", target="sampler"
                ),
            ],
        )
        dumped = config.model_dump()
        assert dumped["checkpoints"][0]["typ"] == "baseten_named_checkpoint"
        assert dumped["checkpoints"][1]["typ"] == "loops_checkpoint"
        assert dumped["checkpoints"][1]["run_id"] == "run123"


class TestComputeAvailabilityModel:
    def test_defaults_to_dedicated(self):
        assert Compute().availability_model == AvailabilityModel.DEDICATED
        assert Compute().model_dump()["availability_model"] == "dedicated"

    def test_spot_serializes_to_string_value(self):
        dumped = Compute(availability_model=AvailabilityModel.SPOT).model_dump()
        assert dumped["availability_model"] == "spot"

    def test_invalid_value_raises(self):
        with pytest.raises(ValidationError):
            Compute(availability_model="on_demand")


def test_aws_oidc_docker_auth_serializes_in_training_image():
    image = Image(
        base_image="123456789012.dkr.ecr.us-west-2.amazonaws.com/training:latest",
        docker_auth=DockerAuth(
            auth_method=truss_config.DockerAuthType.AWS_OIDC,
            registry="123456789012.dkr.ecr.us-west-2.amazonaws.com",
            aws_oidc_docker_auth=AWSOIDCDockerAuth(
                role_arn="arn:aws:iam::123456789012:role/training-image-pull",
                region="us-west-2",
            ),
        ),
    )

    dumped = image.model_dump()

    assert dumped["docker_auth"]["auth_method"] == "AWS_OIDC"
    assert dumped["docker_auth"]["aws_oidc_docker_auth"] == {
        "role_arn": "arn:aws:iam::123456789012:role/training-image-pull",
        "region": "us-west-2",
    }


def test_aws_assume_role_docker_auth_serializes_in_training_image():
    image = Image(
        base_image="123456789012.dkr.ecr.us-west-2.amazonaws.com/training:latest",
        docker_auth=DockerAuth(
            auth_method=truss_config.DockerAuthType.AWS_ASSUME_ROLE,
            registry="123456789012.dkr.ecr.us-west-2.amazonaws.com",
            aws_assume_role_docker_auth=AWSAssumeRoleDockerAuth(
                role_arn="arn:aws:iam::123456789012:role/training-image-pull",
                region="us-west-2",
            ),
        ),
    )

    dumped = image.model_dump()

    assert dumped["docker_auth"]["auth_method"] == "AWS_ASSUME_ROLE"
    assert dumped["docker_auth"]["aws_assume_role_docker_auth"] == {
        "role_arn": "arn:aws:iam::123456789012:role/training-image-pull",
        "region": "us-west-2",
    }


def test_gcp_oidc_docker_auth_serializes_in_training_image():
    image = Image(
        base_image="us-west1-docker.pkg.dev/project/repository/training:latest",
        docker_auth=DockerAuth(
            auth_method=truss_config.DockerAuthType.GCP_OIDC,
            registry="us-west1-docker.pkg.dev",
            gcp_oidc_docker_auth=GCPOIDCDockerAuth(
                service_account="image-pull@example.iam.gserviceaccount.com",
                workload_identity_provider=(
                    "projects/123/locations/global/workloadIdentityPools/pool/providers/provider"
                ),
            ),
        ),
    )

    dumped = image.model_dump()

    assert dumped["docker_auth"]["auth_method"] == "GCP_OIDC"
    assert dumped["docker_auth"]["gcp_oidc_docker_auth"]["service_account"] == (
        "image-pull@example.iam.gserviceaccount.com"
    )


@pytest.mark.parametrize(
    ("auth_method", "expected_field"),
    [
        (truss_config.DockerAuthType.AWS_OIDC, "aws_oidc_docker_auth"),
        (truss_config.DockerAuthType.AWS_ASSUME_ROLE, "aws_assume_role_docker_auth"),
        (truss_config.DockerAuthType.GCP_OIDC, "gcp_oidc_docker_auth"),
    ],
)
def test_oidc_and_assume_role_docker_auth_require_matching_config(
    auth_method, expected_field
):
    with pytest.raises(ValidationError, match=f"{expected_field} must be provided"):
        DockerAuth(auth_method=auth_method, registry="registry.example.com")


def test_oidc_docker_auth_rejects_conflicting_config():
    with pytest.raises(
        ValidationError, match="gcp_oidc_docker_auth cannot be specified"
    ):
        DockerAuth(
            auth_method=truss_config.DockerAuthType.AWS_OIDC,
            registry="registry.example.com",
            aws_oidc_docker_auth=AWSOIDCDockerAuth(
                role_arn="arn:aws:iam::123456789012:role/training-image-pull",
                region="us-west-2",
            ),
            gcp_oidc_docker_auth=GCPOIDCDockerAuth(
                service_account="image-pull@example.iam.gserviceaccount.com",
                workload_identity_provider=(
                    "projects/123/locations/global/workloadIdentityPools/pool/providers/provider"
                ),
            ),
        )


def _minimal_job(**kwargs):
    return TrainingJob(
        image=Image(base_image="hello-world"),
        compute=Compute(),
        runtime=Runtime(),
        **kwargs,
    )


class TestTrainingJobWeightsAuthValidation:
    """Training jobs allow the weight auth methods supported by the training API."""

    def test_weights_with_aws_oidc_are_accepted(self):
        job = _minimal_job(
            weights=[
                truss_config.WeightsSource(
                    source="s3://bucket/path",
                    mount_location="/weights",
                    auth=truss_config.WeightsAuth(
                        auth_method=truss_config.WeightsAuthMethod.AWS_OIDC,
                        aws_oidc_role_arn="arn:aws:iam::123:role/foo",
                        aws_oidc_region="us-west-2",
                    ),
                )
            ]
        )
        assert job.model_dump()["weights"][0]["auth"]["auth_method"] == "AWS_OIDC"

    def test_weights_with_gcp_oidc_are_accepted(self):
        job = _minimal_job(
            weights=[
                truss_config.WeightsSource(
                    source="gs://bucket/path",
                    mount_location="/weights",
                    auth=truss_config.WeightsAuth(
                        auth_method=truss_config.WeightsAuthMethod.GCP_OIDC,
                        gcp_oidc_service_account="my-sa@project.iam.gserviceaccount.com",
                        gcp_oidc_workload_id_provider="projects/123/locations/global/workloadIdentityPools/pool/providers/provider",
                    ),
                )
            ]
        )
        assert job.model_dump()["weights"][0]["auth"]["auth_method"] == "GCP_OIDC"

    def test_weights_with_aws_assume_role_are_accepted(self):
        job = _minimal_job(
            weights=[
                truss_config.WeightsSource(
                    source="s3://bucket/path",
                    mount_location="/weights",
                    auth=truss_config.WeightsAuth(
                        auth_method=truss_config.WeightsAuthMethod.AWS_ASSUME_ROLE,
                        aws_assume_role_arn="arn:aws:iam::123:role/foo",
                        aws_assume_role_region="us-west-2",
                    ),
                )
            ]
        )
        dumped_auth = job.model_dump()["weights"][0]["auth"]
        assert dumped_auth["auth_method"] == "AWS_ASSUME_ROLE"
        assert dumped_auth["aws_assume_role_arn"] == "arn:aws:iam::123:role/foo"
        assert dumped_auth["aws_assume_role_region"] == "us-west-2"

    def test_weights_with_custom_secret_auth_accepted(self):
        job = _minimal_job(
            weights=[
                truss_config.WeightsSource(
                    source="s3://bucket/path",
                    mount_location="/weights",
                    auth=truss_config.WeightsAuth(
                        auth_method=truss_config.WeightsAuthMethod.CUSTOM_SECRET,
                        auth_secret_name="my-secret",
                    ),
                )
            ]
        )
        assert len(job.weights) == 1
        assert (
            job.weights[0].auth.auth_method
            == truss_config.WeightsAuthMethod.CUSTOM_SECRET
        )
        assert job.weights[0].auth.auth_secret_name == "my-secret"

    def test_weights_with_top_level_auth_secret_name_accepted(self):
        job = _minimal_job(
            weights=[
                truss_config.WeightsSource(
                    source="s3://bucket/path",
                    mount_location="/weights",
                    auth_secret_name="my-secret",
                )
            ]
        )
        assert len(job.weights) == 1
        assert job.weights[0].auth_secret_name == "my-secret"

    def test_weights_empty_accepted(self):
        job = _minimal_job(weights=[])
        assert job.weights == []

    def test_weights_no_auth_accepted(self):
        job = _minimal_job(
            weights=[
                truss_config.WeightsSource(
                    source="hf://owner/repo", mount_location="/weights"
                )
            ]
        )
        assert len(job.weights) == 1
        assert job.weights[0].auth is None


class TestCheckpointListLoopsCheckpoints:
    """truss-train CheckpointList: loops_checkpoint_ids field, validator, to_truss_config."""

    def test_loops_checkpoint_ids_round_trip_to_truss_config(self):
        ckpt_list = CheckpointList(
            base_model_id="Qwen/Qwen3-8B", loops_checkpoint_ids=["tcp_a", "tcp_b"]
        )
        truss_ckpt_list = ckpt_list.to_truss_config()
        assert truss_ckpt_list.artifact_references == []
        assert truss_ckpt_list.loops_checkpoint_ids == ["tcp_a", "tcp_b"]
        assert truss_ckpt_list.download_folder == ckpt_list.download_folder

    def test_artifact_references_round_trip_to_truss_config(self):
        ckpt_list = CheckpointList(
            base_model_id="Qwen/Qwen3-8B",
            checkpoints=[
                LoRACheckpoint(
                    training_job_id="tj_abc",
                    checkpoint_name="step-1",
                    model_weight_format=ModelWeightsFormat.LORA,
                )
            ],
        )
        truss_ckpt_list = ckpt_list.to_truss_config()
        assert len(truss_ckpt_list.artifact_references) == 1
        assert truss_ckpt_list.artifact_references[0].training_job_id == "tj_abc"
        assert truss_ckpt_list.loops_checkpoint_ids == []

    def test_mixing_checkpoints_and_loops_ids_raises(self):
        with pytest.raises(ValidationError, match="Cannot mix"):
            CheckpointList(
                checkpoints=[
                    LoRACheckpoint(
                        training_job_id="tj_abc",
                        checkpoint_name="step-1",
                        model_weight_format=ModelWeightsFormat.LORA,
                    )
                ],
                loops_checkpoint_ids=["tcp_xyz"],
            )
