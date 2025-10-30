import pydantic
import pytest
import requests_mock

from truss.remote.baseten import custom_types as b10_types
from truss.remote.baseten.core import (
    ModelId,
    ModelName,
    ModelVersionId,
    create_chain_atomic,
)
from truss.remote.baseten.custom_types import ChainletDataAtomic, OracleData
from truss.remote.baseten.error import RemoteError
from truss.truss_handle.truss_handle import TrussHandle


def assert_request_matches_expected_query(request, expected_query) -> None:
    query = request.json()["query"]
    actual_lines = tuple(
        line.strip() for line in query.strip().split("\n") if line.strip()
    )
    expected_lines = tuple(
        line.strip() for line in expected_query.split("\n") if line.strip()
    )
    assert actual_lines == expected_lines


def test_get_service_by_version_id(remote, test_remote_graphql_path):
    version = {"id": "version_id", "oracle": {"id": "model_id", "hostname": "hostname"}}
    model_version_response = {"data": {"model_version": version}}

    with requests_mock.Mocker() as m:
        m.post(test_remote_graphql_path, json=model_version_response)
        service = remote.get_service(model_identifier=ModelVersionId("version_id"))

    assert service.model_id == "model_id"
    assert service.model_version_id == "version_id"


def test_get_service_by_version_id_no_version(remote, test_remote_graphql_path):
    model_version_response = {"errors": [{"message": "error"}]}
    with requests_mock.Mocker() as m:
        m.post(test_remote_graphql_path, json=model_version_response)
        with pytest.raises(RemoteError):
            remote.get_service(model_identifier=ModelVersionId("version_id"))


def test_get_service_by_model_name(remote, test_remote_graphql_path):
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
                "hostname": "hostname",
                "versions": versions,
            }
        }
    }

    with requests_mock.Mocker() as m:
        m.post(test_remote_graphql_path, json=model_response)
        service = remote.get_service(model_identifier=ModelName("model_name"))

    assert service.model_id == "model_id"
    assert service.model_version_id == "2"


def test_get_service_by_model_name_no_primary(remote, test_remote_graphql_path):
    versions = [
        {"id": "1", "is_draft": False, "is_primary": False},
        {"id": "3", "is_draft": True, "is_primary": False},
    ]
    model_response = {
        "data": {
            "model": {
                "name": "model_name",
                "id": "model_id",
                "hostname": "hostname",
                "versions": versions,
            }
        }
    }

    with requests_mock.Mocker() as m:
        m.post(test_remote_graphql_path, json=model_response)
        service = remote.get_service(model_identifier=ModelName("model_name"))

    assert service.model_id == "model_id"
    assert service.model_version_id == "1"


def test_get_service_by_model_name_no_deployed_versions(
    remote, test_remote_graphql_path
):
    versions = [{"id": "3", "is_draft": True, "is_primary": False}]
    model_response = {
        "data": {
            "model": {
                "name": "model_name",
                "id": "model_id",
                "hostname": "hostname",
                "versions": versions,
            }
        }
    }

    with requests_mock.Mocker() as m:
        m.post(test_remote_graphql_path, json=model_response)
        with pytest.raises(RemoteError):
            remote.get_service(model_identifier=ModelName("model_name"))


def test_get_service_by_model_name_no_model(remote, test_remote_graphql_path):
    model_response = {"data": {"model": None}}

    with requests_mock.Mocker() as m:
        m.post(test_remote_graphql_path, json=model_response)
        with pytest.raises(RemoteError):
            remote.get_service(model_identifier=ModelName("model_name"))


def test_get_service_by_model_name_and_is_draft(remote, test_remote_graphql_path):
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
                "hostname": "hostname",
                "versions": versions,
            }
        }
    }

    with requests_mock.Mocker() as m:
        m.post(test_remote_graphql_path, json=model_response)
        service = remote.get_service(
            model_identifier=ModelName("model_name"), published=False
        )

    assert service.model_id == "model_id"
    assert service.model_version_id == "3"


def test_get_service_by_model_id(remote, test_remote_graphql_path):
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
                "hostname": "hostname",
                "versions": versions,
            }
        }
    }

    with requests_mock.Mocker() as m:
        m.post(test_remote_graphql_path, json=model_response)
        service = remote.get_service(model_identifier=ModelId("model_id"))

    assert service.model_id == "model_id"
    assert service.model_version_id == "2"


def test_get_chain_atomic_by_name(remote, test_remote_graphql_path):
    chain_response = {
        "data": {
            "chain": {
                "name": "chain_name",
                "id": "chain_id",
                "oracle": {"hostname": "hostname"},
            }
        }
    }

    with requests_mock.Mocker() as m:
        m.post(test_remote_graphql_path, json=chain_response)
        chain = remote.get_chain(chain_name="chain_name")

    assert chain.name == "chain_name"
    assert chain.oracle_data == OracleData(
        chain_id="chain_id", chain_deployment_hostname="hostname"
    )


def test_create_chain_atomic(remote, test_remote_graphql_path):
    chain_response = {
        "data": {
            "chain": {
                "name": "chain_name",
                "id": "chain_id",
                "oracle": {"hostname": "hostname"},
            }
        }
    }

    with requests_mock.Mocker() as m:
        m.post(test_remote_graphql_path, json=chain_response)
        chain = create_chain_atomic(remote._graphql_client, "chain_name", None, {})

    assert chain.name == "chain_name"
    assert chain.oracle_data == OracleData(
        chain_id="chain_id", chain_deployment_hostname="hostname"
    )


def test_patch_chain_name_already_exists(remote, test_remote_graphql_path):
    with requests_mock.Mocker() as m:
        m.post(
            test_remote_graphql_path,
            json={"errors": [{"message": "Chain with name chain_name already exists"}]},
        )
        with pytest.raises(ValueError, match="Chain.*already exists"):
            create_chain_atomic(remote._graphql_client, "chain_name", None, {})


def test_patch_chain(remote, test_remote_graphql_path):
    chain_deployment_response = {"data": {"chain_deployment": {"id": "deployment_id"}}}

    with requests_mock.Mocker() as m:
        m.post(test_remote_graphql_path, json=chain_deployment_response)
        deployment_handle = remote.push_chain_atomic(
            chain_name="chain_name",
            chain_id="chain_id",
            entrypoint="entrypoint",
            chainlets_data=[],
        )

    assert deployment_handle.chain_deployment_id == "deployment_id"


def test_push_model(
    custom_model_truss_dir_with_pre_and_post,
    remote,
    mocked_push_requests,
    mock_upload_truss,
):
    remote.push(
        TrussHandle(custom_model_truss_dir_with_pre_and_post),
        "model_name",
        custom_model_truss_dir_with_pre_and_post,
        publish=True,
    )


def test_get_dev_version_from_versions(remote):
    versions = [
        {"id": "1", "is_draft": False, "is_primary": False},
        {"id": "2", "is_draft": False, "is_primary": True},
        {"id": "3", "is_draft": True, "is_primary": False},
    ]

    dev_version = remote._get_dev_version_from_versions(versions)

    assert dev_version["id"] == "3"


def test_get_dev_version_from_versions_no_dev(remote):
    versions = [
        {"id": "1", "is_draft": False, "is_primary": False},
        {"id": "2", "is_draft": False, "is_primary": True},
    ]

    dev_version = remote._get_dev_version_from_versions(versions)

    assert dev_version is None


def test_get_production_version_from_versions(remote):
    versions = [
        {"id": "1", "is_draft": False, "is_primary": False},
        {"id": "2", "is_draft": False, "is_primary": True},
        {"id": "3", "is_draft": True, "is_primary": False},
    ]

    prod_version = remote._get_production_version_from_versions(versions)

    assert prod_version["id"] == "2"


def test_get_production_version_from_versions_no_prod(remote):
    versions = [
        {"id": "1", "is_draft": False, "is_primary": False},
        {"id": "3", "is_draft": True, "is_primary": False},
    ]

    prod_version = remote._get_production_version_from_versions(versions)

    assert prod_version["id"] == "1"


def test_push_passes_deploy_timeout_minutes_to_create_truss_service(
    custom_model_truss_dir_with_pre_and_post,
    remote,
    mocked_push_requests,
    mock_upload_truss,
    mock_create_truss_service,
    mock_truss_handle,
):
    remote.push(
        mock_truss_handle,
        "model_name",
        mock_truss_handle.truss_dir,
        publish=True,
        deploy_timeout_minutes=450,
    )

    mock_create_truss_service.assert_called_once()
    _, kwargs = mock_create_truss_service.call_args
    assert kwargs["deploy_timeout_minutes"] == 450


def test_push_passes_none_deploy_timeout_minutes_when_not_specified(
    custom_model_truss_dir_with_pre_and_post,
    remote,
    mocked_push_requests,
    mock_upload_truss,
    mock_create_truss_service,
    mock_truss_handle,
):
    remote.push(
        mock_truss_handle, "model_name", mock_truss_handle.truss_dir, publish=True
    )

    mock_create_truss_service.assert_called_once()
    _, kwargs = mock_create_truss_service.call_args
    assert kwargs.get("deploy_timeout_minutes") is None


def test_push_integration_deploy_timeout_minutes_propagated(
    custom_model_truss_dir_with_pre_and_post,
    remote,
    mocked_push_requests,
    mock_upload_truss,
    mock_create_truss_service,
    mock_truss_handle,
):
    remote.push(
        mock_truss_handle,
        "model_name",
        mock_truss_handle.truss_dir,
        publish=True,
        environment="staging",
        deploy_timeout_minutes=750,
    )

    mock_create_truss_service.assert_called_once()
    _, kwargs = mock_create_truss_service.call_args
    assert kwargs["deploy_timeout_minutes"] == 750
    assert kwargs["environment"] == "staging"


def test_cli_push_passes_deploy_timeout_minutes_to_create_truss_service(
    custom_model_truss_dir_with_pre_and_post,
    remote,
    mocked_push_requests,
    mock_upload_truss,
    mock_create_truss_service,
):
    from unittest.mock import patch

    from click.testing import CliRunner

    from truss.cli.cli import truss_cli

    runner = CliRunner()
    with patch("truss.cli.cli.RemoteFactory.create", return_value=remote):
        result = runner.invoke(
            truss_cli,
            [
                "push",
                str(custom_model_truss_dir_with_pre_and_post),
                "--remote",
                "baseten",
                "--model-name",
                "model_name",
                "--publish",
                "--deploy-timeout-minutes",
                "450",
            ],
        )

    assert result.exit_code == 0
    mock_create_truss_service.assert_called_once()
    _, kwargs = mock_create_truss_service.call_args
    assert kwargs["deploy_timeout_minutes"] == 450


def test_cli_push_passes_none_deploy_timeout_minutes_when_not_specified(
    custom_model_truss_dir_with_pre_and_post,
    remote,
    mocked_push_requests,
    mock_upload_truss,
    mock_create_truss_service,
):
    from unittest.mock import patch

    from click.testing import CliRunner

    from truss.cli.cli import truss_cli

    runner = CliRunner()
    with patch("truss.cli.cli.RemoteFactory.create", return_value=remote):
        result = runner.invoke(
            truss_cli,
            [
                "push",
                str(custom_model_truss_dir_with_pre_and_post),
                "--remote",
                "baseten",
                "--model-name",
                "model_name",
                "--publish",
            ],
        )

    assert result.exit_code == 0
    mock_create_truss_service.assert_called_once()
    _, kwargs = mock_create_truss_service.call_args
    assert kwargs.get("deploy_timeout_minutes") is None


def test_cli_push_integration_deploy_timeout_minutes_propagated(
    custom_model_truss_dir_with_pre_and_post,
    remote,
    mocked_push_requests,
    mock_upload_truss,
    mock_create_truss_service,
):
    from unittest.mock import patch

    from click.testing import CliRunner

    from truss.cli.cli import truss_cli

    runner = CliRunner()
    with patch("truss.cli.cli.RemoteFactory.create", return_value=remote):
        result = runner.invoke(
            truss_cli,
            [
                "push",
                str(custom_model_truss_dir_with_pre_and_post),
                "--remote",
                "baseten",
                "--model-name",
                "model_name",
                "--publish",
                "--environment",
                "staging",
                "--deploy-timeout-minutes",
                "750",
            ],
        )

    assert result.exit_code == 0
    mock_create_truss_service.assert_called_once()
    _, kwargs = mock_create_truss_service.call_args
    assert kwargs["deploy_timeout_minutes"] == 750
    assert kwargs["environment"] == "staging"


def test_cli_push_api_integration_deploy_timeout_minutes_propagated(
    custom_model_truss_dir_with_pre_and_post,
    mock_remote_factory,
    temp_trussrc_dir,
    mock_available_config_names,
):
    from unittest.mock import MagicMock, patch

    from click.testing import CliRunner

    from truss.cli.cli import truss_cli

    mock_service = MagicMock()
    mock_service.model_id = "model_id"
    mock_service.model_version_id = "version_id"
    mock_remote_factory.push.return_value = mock_service

    runner = CliRunner()
    with patch(
        "truss.cli.cli.RemoteFactory.get_available_config_names",
        return_value=["baseten"],
    ):
        result = runner.invoke(
            truss_cli,
            [
                "push",
                str(custom_model_truss_dir_with_pre_and_post),
                "--remote",
                "baseten",
                "--model-name",
                "model_name",
                "--publish",
                "--deploy-timeout-minutes",
                "1200",
            ],
        )

    assert result.exit_code == 0
    mock_remote_factory.push.assert_called_once()
    _, push_kwargs = mock_remote_factory.push.call_args
    assert push_kwargs.get("deploy_timeout_minutes") == 1200


def test_push_chainlet_data_atomic(remote, test_remote_graphql_path):
    response = {
        "data": {"chainlet_data": {"id": "chainlet_data_id", "s3_key": "s3_key"}}
    }

    chainlet_data = b10_types.ChainletDataAtomic(
        chainlet_name="chainlet_name", oracle_version_id="oracle_version_id"
    )

    with requests_mock.Mocker() as m:
        m.post(test_remote_graphql_path, json=response)
        result = remote.push_chainlet_data_atomic(chainlet_data)

    assert result == ChainletDataAtomic(
        name="chainlet_name",
        oracle_predict_url=None,
        oracle_version_id="oracle_version_id",
        s3_key="s3_key",
    )


def test_validate_chainlets_data_atomic(remote):
    with pydantic.ValidationError as e:
        b10_types.ChainletDataAtomic(chainlet_name="chainlet_name")
    # Should require oracle_version_id
    assert "oracle_version_id" in str(e)
