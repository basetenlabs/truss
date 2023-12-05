from tempfile import NamedTemporaryFile
from unittest.mock import MagicMock

from truss.remote.baseten import core
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.error import ApiError


def test_exists_model():
    def mock_get_model(model_name):
        if model_name == "first model":
            return {"model": {"id": "1"}}
        elif model_name == "second model":
            return {"model": {"id": "2"}}
        else:
            raise ApiError(
                "Oracle not found",
                BasetenApi.GraphQLErrorCodes.RESOURCE_NOT_FOUND.value,
            )

    api = MagicMock()
    api.get_model.side_effect = mock_get_model

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


def test_get_dev_version_from_versions():
    versions = [
        {"id": "1", "is_draft": False},
        {"id": "2", "is_draft": True},
    ]
    dev_version = core.get_dev_version_from_versions(versions)
    assert dev_version["id"] == "2"


def test_get_dev_version_from_versions_error():
    versions = [
        {"id": "1", "is_draft": False},
    ]
    dev_version = core.get_dev_version_from_versions(versions)
    assert dev_version is None


def test_get_dev_version():
    versions = [
        {"id": "1", "is_draft": False},
        {"id": "2", "is_draft": True},
    ]
    api = MagicMock()
    api.get_model.return_value = {"model": {"versions": versions}}

    dev_version = core.get_dev_version(api, "my_model")
    assert dev_version["id"] == "2"


def test_get_prod_version_from_versions():
    versions = [
        {"id": "1", "is_draft": False, "is_primary": False},
        {"id": "2", "is_draft": True, "is_primary": False},
        {"id": "3", "is_draft": False, "is_primary": True},
    ]
    prod_version = core.get_prod_version_from_versions(versions)
    assert prod_version["id"] == "3"


def test_get_prod_version_from_versions_error():
    versions = [
        {"id": "1", "is_draft": True, "is_primary": False},
        {"id": "2", "is_draft": False, "is_primary": False},
    ]
    prod_version = core.get_prod_version_from_versions(versions)
    assert prod_version is None
