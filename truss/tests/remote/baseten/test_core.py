from tempfile import NamedTemporaryFile
from unittest.mock import MagicMock

import pytest
from truss.constants import PRODUCTION_ENVIRONMENT_NAME
from truss.remote.baseten import core
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.core import create_truss_service
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


@pytest.mark.parametrize(
    "environment",
    [
        None,
        PRODUCTION_ENVIRONMENT_NAME,
    ],
)
def test_create_truss_service_handles_eligible_environment_values(environment):
    api = MagicMock()
    return_value = {
        "id": "id",
        "version_id": "model_version_id",
    }
    api.create_model_from_truss.return_value = return_value
    model_id, model_version_id = create_truss_service(
        api,
        "model_name",
        "s3_key",
        "config",
        is_trusted=False,
        preserve_previous_prod_deployment=False,
        is_draft=False,
        model_id=None,
        deployment_name="deployment_name",
        environment=environment,
    )
    assert model_id == return_value["id"]
    assert model_version_id == return_value["version_id"]
    api.create_model_from_truss.assert_called_once()


@pytest.mark.parametrize(
    "model_id",
    [
        "some_model_id",
        None,
    ],
)
def test_create_truss_services_handles_is_draft(model_id):
    api = MagicMock()
    return_value = {
        "id": "id",
        "version_id": "model_version_id",
    }
    api.create_development_model_from_truss.return_value = return_value
    model_id, model_version_id = create_truss_service(
        api,
        "model_name",
        "s3_key",
        "config",
        is_trusted=False,
        preserve_previous_prod_deployment=False,
        is_draft=True,
        model_id=model_id,
        deployment_name="deployment_name",
    )
    assert model_id == return_value["id"]
    assert model_version_id == return_value["version_id"]
    api.create_development_model_from_truss.assert_called_once()


@pytest.mark.parametrize(
    "inputs",
    [
        {
            "environment": None,
            "deployment_name": "some deployment",
            "is_trusted": True,
            "preserve_previous_prod_deployment": False,
        },
        {
            "environment": PRODUCTION_ENVIRONMENT_NAME,
            "deployment_name": None,
            "is_trusted": True,
            "preserve_previous_prod_deployment": False,
        },
        {
            "environment": "staging",
            "deployment_name": "some_deployment_name",
            "is_trusted": False,
            "preserve_previous_prod_deployment": True,
        },
    ],
)
def test_create_truss_service_handles_existing_model(inputs):
    api = MagicMock()
    return_value = {
        "id": "model_version_id",
    }
    api.create_model_version_from_truss.return_value = return_value
    model_id, model_version_id = create_truss_service(
        api,
        "model_name",
        "s3_key",
        "config",
        is_draft=False,
        model_id="model_id",
        **inputs,
    )

    assert model_id == "model_id"
    assert model_version_id == return_value["id"]
    api.create_model_version_from_truss.assert_called_once()
    _, kwargs = api.create_model_version_from_truss.call_args
    for k, v in inputs.items():
        assert kwargs[k] == v
