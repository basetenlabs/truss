import pathlib
from unittest import mock

from truss_train import deployment
from truss_train.definitions import Image, TrainingJob


@mock.patch("truss.remote.baseten.utils.transfer.multipart_upload_boto3")
@mock.patch("truss.remote.baseten.api.BasetenApi.get_blob_credentials")
def test_prepare_push(get_blob_credentials_mock, multipart_upload_boto3_mock):
    mock_api = mock.Mock()
    mock_api.get_blob_credentials.return_value = {
        "s3_bucket": "test-s3-bucket",
        "s3_key": "test-s3-key",
        "creds": {},
    }

    prepared_job = deployment.prepare_push(
        mock_api,
        pathlib.Path(__file__),
        TrainingJob(image=Image(base_image="hello-world")),
    )
    assert len(prepared_job.runtime_artifacts) == 1
    assert prepared_job.runtime_artifacts[0].s3_key == "test-s3-key"
    assert prepared_job.runtime_artifacts[0].s3_bucket == "test-s3-bucket"
