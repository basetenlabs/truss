import click
import pytest
import requests_mock
from truss.remote.baseten.core import ModelId, ModelName, ModelVersionId
from truss.remote.baseten.remote import BasetenRemote
from truss.truss_handle import TrussHandle

_TEST_REMOTE_URL = "http://test_remote.com"
_TEST_REMOTE_GRAPHQL_PATH = "http://test_remote.com/graphql/"


def test_get_service_by_version_id():
    remote = BasetenRemote(_TEST_REMOTE_URL, "api_key")

    version = {
        "id": "version_id",
        "oracle": {
            "id": "model_id",
        },
    }
    model_version_response = {"data": {"model_version": version}}

    with requests_mock.Mocker() as m:
        m.post(
            _TEST_REMOTE_GRAPHQL_PATH,
            json=model_version_response,
        )
        service = remote.get_service(model_identifier=ModelVersionId("version_id"))

    assert service.model_id == "model_id"
    assert service.model_version_id == "version_id"


def test_get_service_by_version_id_no_version():
    remote = BasetenRemote(_TEST_REMOTE_URL, "api_key")
    model_version_response = {"errors": [{"message": "error"}]}
    with requests_mock.Mocker() as m:
        m.post(
            _TEST_REMOTE_GRAPHQL_PATH,
            json=model_version_response,
        )
        with pytest.raises(click.UsageError):
            remote.get_service(model_identifier=ModelVersionId("version_id"))


def test_get_service_by_model_name():
    remote = BasetenRemote(_TEST_REMOTE_URL, "api_key")

    versions = [
        {"id": "1", "is_draft": False, "is_primary": False},
        {"id": "2", "is_draft": False, "is_primary": True},
        {"id": "3", "is_draft": True, "is_primary": False},
    ]
    model_response = {
        "data": {
            "model": {
                "name": "model_name",
                "id": "model_id",
                "versions": versions,
            }
        }
    }

    with requests_mock.Mocker() as m:
        m.post(
            _TEST_REMOTE_GRAPHQL_PATH,
            json=model_response,
        )

        # Check that the production version is returned when published is True.
        service = remote.get_service(
            model_identifier=ModelName("model_name"), published=True
        )
        assert service.model_id == "model_id"
        assert service.model_version_id == "2"

        # Check that the development version is returned when published is False.
        service = remote.get_service(
            model_identifier=ModelName("model_name"), published=False
        )
        assert service.model_id == "model_id"
        assert service.model_version_id == "3"


def test_get_service_by_model_name_no_dev_version():
    remote = BasetenRemote(_TEST_REMOTE_URL, "api_key")

    versions = [
        {"id": "1", "is_draft": False, "is_primary": True},
    ]
    model_response = {
        "data": {
            "model": {
                "name": "model_name",
                "id": "model_id",
                "versions": versions,
            }
        }
    }

    with requests_mock.Mocker() as m:
        m.post(
            _TEST_REMOTE_GRAPHQL_PATH,
            json=model_response,
        )

        # Check that the production version is returned when published is True.
        service = remote.get_service(
            model_identifier=ModelName("model_name"), published=True
        )
        assert service.model_id == "model_id"
        assert service.model_version_id == "1"

        # Since no development version exists, calling get_service with
        # published=False should raise an error.
        with pytest.raises(click.UsageError):
            remote.get_service(
                model_identifier=ModelName("model_name"), published=False
            )


def test_get_service_by_model_name_no_prod_version():
    remote = BasetenRemote(_TEST_REMOTE_URL, "api_key")

    versions = [
        {"id": "1", "is_draft": True, "is_primary": False},
    ]
    model_response = {
        "data": {
            "model": {
                "name": "model_name",
                "id": "model_id",
                "versions": versions,
            }
        }
    }

    with requests_mock.Mocker() as m:
        m.post(
            _TEST_REMOTE_GRAPHQL_PATH,
            json=model_response,
        )

        # Since no production version exists, calling get_service with
        # published=True should raise an error.
        with pytest.raises(click.UsageError):
            remote.get_service(model_identifier=ModelName("model_name"), published=True)

        # Check that the development version is returned when published is False.
        service = remote.get_service(
            model_identifier=ModelName("model_name"), published=False
        )
        assert service.model_id == "model_id"
        assert service.model_version_id == "1"


def test_get_service_by_model_id():
    remote = BasetenRemote(_TEST_REMOTE_URL, "api_key")

    model_response = {
        "data": {
            "model": {
                "name": "model_name",
                "id": "model_id",
                "primary_version": {"id": "version_id"},
            }
        }
    }

    with requests_mock.Mocker() as m:
        m.post(
            _TEST_REMOTE_GRAPHQL_PATH,
            json=model_response,
        )

        service = remote.get_service(model_identifier=ModelId("model_id"))
        assert service.model_id == "model_id"
        assert service.model_version_id == "version_id"


def test_get_service_by_model_id_no_model():
    remote = BasetenRemote(_TEST_REMOTE_URL, "api_key")
    model_response = {"errors": [{"message": "error"}]}
    with requests_mock.Mocker() as m:
        m.post(
            _TEST_REMOTE_GRAPHQL_PATH,
            json=model_response,
        )
        with pytest.raises(click.UsageError):
            remote.get_service(model_identifier=ModelId("model_id"))


def test_push_raised_value_error_when_deployment_name_and_not_publish(
    custom_model_truss_dir_with_pre_and_post,
):
    remote = BasetenRemote(_TEST_REMOTE_URL, "api_key")
    model_response = {
        "data": {
            "model": {
                "name": "model_name",
                "id": "model_id",
                "primary_version": {"id": "version_id"},
            }
        }
    }
    with requests_mock.Mocker() as m:
        m.post(
            _TEST_REMOTE_GRAPHQL_PATH,
            json=model_response,
        )
        th = TrussHandle(custom_model_truss_dir_with_pre_and_post)

        with pytest.raises(
            ValueError,
            match="Deployment name cannot be used for development deployment",
        ):
            remote.push(th, "model_name", False, False, False, False, "dep_name")


def test_push_raised_value_error_when_deployment_name_is_not_valid(
    custom_model_truss_dir_with_pre_and_post,
):
    remote = BasetenRemote(_TEST_REMOTE_URL, "api_key")
    model_response = {
        "data": {
            "model": {
                "name": "model_name",
                "id": "model_id",
                "primary_version": {"id": "version_id"},
            }
        }
    }
    with requests_mock.Mocker() as m:
        m.post(
            _TEST_REMOTE_GRAPHQL_PATH,
            json=model_response,
        )
        th = TrussHandle(custom_model_truss_dir_with_pre_and_post)

        with pytest.raises(
            ValueError,
            match="Deployment name must only contain alphanumeric, -, _ and . characters",
        ):
            remote.push(th, "model_name", True, False, False, False, "dep//name")


def test_push_raised_value_error_when_keep_previous_prod_settings_and_not_promote(
    custom_model_truss_dir_with_pre_and_post,
):
    remote = BasetenRemote(_TEST_REMOTE_URL, "api_key")
    model_response = {
        "data": {
            "model": {
                "name": "model_name",
                "id": "model_id",
                "primary_version": {"id": "version_id"},
            }
        }
    }
    with requests_mock.Mocker() as m:
        m.post(
            _TEST_REMOTE_GRAPHQL_PATH,
            json=model_response,
        )
        th = TrussHandle(custom_model_truss_dir_with_pre_and_post)

        with pytest.raises(
            ValueError,
            match="preserve-previous-production-deployment can only be used with the '--promote' option",
        ):
            remote.push(th, "model_name", False, False, False, True)
