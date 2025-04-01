import pathlib
from unittest import mock

import pytest

from truss.base import truss_config
from truss_train import deployment
from truss_train.definitions import Compute, Image, Runtime, TrainingJob


@mock.patch("truss.remote.baseten.utils.transfer.multipart_upload_boto3")
@mock.patch("truss.remote.baseten.api.BasetenApi.get_blob_credentials")
@pytest.mark.parametrize("use_read_write_cache", [True, False])
@pytest.mark.parametrize(
    "compute",
    [
        Compute(),
        Compute(
            accelerator=truss_config.AcceleratorSpec(
                accelerator=truss_config.Accelerator.T4, count=1
            )
        ),
        Compute(
            node_count=2,
            accelerator=truss_config.AcceleratorSpec(
                accelerator=truss_config.Accelerator.H100, count=8
            ),
        ),
    ],
)
def test_prepare_push(
    get_blob_credentials_mock,
    multipart_upload_boto3_mock,
    compute,
    use_read_write_cache,
):
    mock_api = mock.Mock()
    mock_api.get_blob_credentials.return_value = {
        "s3_bucket": "test-s3-bucket",
        "s3_key": "test-s3-key",
        "creds": {},
    }

    prepared_job = deployment.prepare_push(
        mock_api,
        pathlib.Path(__file__),
        TrainingJob(
            image=Image(base_image="hello-world"),
            compute=compute,
            runtime=Runtime(use_read_write_cache=use_read_write_cache),
        ),
    )
    assert len(prepared_job.runtime_artifacts) == 1
    assert prepared_job.runtime_artifacts[0].s3_key == "test-s3-key"
    assert prepared_job.runtime_artifacts[0].s3_bucket == "test-s3-bucket"
    if compute.accelerator:
        assert (
            prepared_job.compute.accelerator.accelerator
            == compute.accelerator.accelerator
        )
    else:
        assert prepared_job.compute.accelerator is None
    assert prepared_job.runtime.use_read_write_cache == use_read_write_cache
    # ensure that serialization works
    prepared_job.model_dump()
