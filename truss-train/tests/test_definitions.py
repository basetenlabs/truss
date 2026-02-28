import pytest
from pydantic import ValidationError

from truss.base import truss_config
from truss_train.definitions import Compute, Image, Runtime, TrainingJob


def _minimal_job(**kwargs):
    return TrainingJob(
        image=Image(base_image="hello-world"),
        compute=Compute(),
        runtime=Runtime(),
        **kwargs,
    )


class TestTrainingJobWeightsAuthValidation:
    """Training jobs only allow CUSTOM_SECRET with auth_secret_name for weights; OIDC is not supported."""

    def test_weights_with_aws_oidc_raises(self):
        with pytest.raises(ValidationError, match="CUSTOM_SECRET with auth_secret_name"):
            _minimal_job(
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

    def test_weights_with_gcp_oidc_raises(self):
        with pytest.raises(ValidationError, match="CUSTOM_SECRET with auth_secret_name"):
            _minimal_job(
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
        assert job.weights[0].auth.auth_method == truss_config.WeightsAuthMethod.CUSTOM_SECRET
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
                    source="hf://owner/repo",
                    mount_location="/weights",
                )
            ]
        )
        assert len(job.weights) == 1
        assert job.weights[0].auth is None
