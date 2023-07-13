from tempfile import NamedTemporaryFile
from unittest.mock import MagicMock

from truss.remote.baseten import core


def test_exists_model():
    api = MagicMock()
    api.models.return_value = {
        "models": [
            {"id": "1", "name": "first model"},
            {"id": "2", "name": "second model"},
        ]
    }

    assert core.exists_model(api, "first model")
    assert core.exists_model(api, "second model")
    assert not core.exists_model(api, "third model")


def test_upload_truss():
    api = MagicMock()
    api.model_s3_upload_credentials.return_value = {
        "s3_key": "key",
        "s3_bucket": "bucket",
    }
    core.multipart_upload_boto3 = MagicMock()
    core.multipart_upload_boto3.return_value = None
    test_file = NamedTemporaryFile()
    assert core.upload_truss(api, test_file) == "key"
