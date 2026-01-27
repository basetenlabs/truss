from unittest import mock

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

_TEST_REMOTE_URL = "http://test_remote.com"
_TEST_REMOTE_GRAPHQL_PATH = "http://test_remote.com/graphql/"

TRUSS_RC_CONTENT = """
[baseten]
remote_provider = baseten
api_key = test_key
remote_url = http://test.com
""".strip()


def assert_request_matches_expected_query(request, expected_query) -> None:
    query = request.json()["query"]
    actual_lines = tuple(
        line.strip() for line in query.strip().split("\n") if line.strip()
    )
    expected_lines = tuple(
        line.strip() for line in expected_query.split("\n") if line.strip()
    )
    assert actual_lines == expected_lines


def test_get_service_by_version_id(remote):
    version = {"id": "version_id", "oracle": {"id": "model_id", "hostname": "hostname"}}
    model_version_response = {"data": {"model_version": version}}

    with requests_mock.Mocker() as m:
        m.post(_TEST_REMOTE_GRAPHQL_PATH, json=model_version_response)
        service = remote.get_service(model_identifier=ModelVersionId("version_id"))

    assert service.model_id == "model_id"
    assert service.model_version_id == "version_id"


def test_get_service_by_version_id_no_version(remote):
    model_version_response = {"errors": [{"message": "error"}]}
    with requests_mock.Mocker() as m:
        m.post(_TEST_REMOTE_GRAPHQL_PATH, json=model_version_response)
        with pytest.raises(RemoteError):
            remote.get_service(model_identifier=ModelVersionId("version_id"))


def test_get_service_by_model_name(remote):
    versions = [
        {"id": "1", "is_draft": False, "is_primary": False},
        {"id": "2", "is_draft": False, "is_primary": True},
        {"id": "3", "is_draft": True, "is_primary": False},
    ]

    # Mock responses for the new team disambiguation flow
    teams_response = {
        "data": {"teams": [{"id": "team1", "name": "Team Alpha", "default": False}]}
    }
    models_response = {
        "data": {
            "models": [
                {
                    "name": "model_name",
                    "id": "model_id",
                    "hostname": "hostname",
                    "team": {"id": "team1", "name": "Team Alpha"},
                    "versions": versions,
                }
            ]
        }
    }

    with requests_mock.Mocker() as m:
        m.post(
            _TEST_REMOTE_GRAPHQL_PATH,
            [
                {"json": teams_response},
                {"json": models_response},
                {"json": teams_response},
                {"json": models_response},
            ],
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


def test_get_service_by_model_name_no_dev_version(remote):
    versions = [{"id": "1", "is_draft": False, "is_primary": True}]

    # Mock responses for the new team disambiguation flow
    teams_response = {
        "data": {"teams": [{"id": "team1", "name": "Team Alpha", "default": False}]}
    }
    models_response = {
        "data": {
            "models": [
                {
                    "name": "model_name",
                    "id": "model_id",
                    "hostname": "hostname",
                    "team": {"id": "team1", "name": "Team Alpha"},
                    "versions": versions,
                }
            ]
        }
    }

    with requests_mock.Mocker() as m:
        m.post(
            _TEST_REMOTE_GRAPHQL_PATH,
            [
                {"json": teams_response},
                {"json": models_response},
                {"json": teams_response},
                {"json": models_response},
            ],
        )

        # Check that the production version is returned when published is True.
        service = remote.get_service(
            model_identifier=ModelName("model_name"), published=True
        )
        assert service.model_id == "model_id"
        assert service.model_version_id == "1"

        # Since no development version exists, calling get_service with
        # published=False should raise an error.
        with pytest.raises(RemoteError):
            remote.get_service(
                model_identifier=ModelName("model_name"), published=False
            )


def test_get_service_by_model_name_no_prod_version(remote):
    versions = [{"id": "1", "is_draft": True, "is_primary": False}]

    # Mock responses for the new team disambiguation flow
    teams_response = {
        "data": {"teams": [{"id": "team1", "name": "Team Alpha", "default": False}]}
    }
    models_response = {
        "data": {
            "models": [
                {
                    "name": "model_name",
                    "id": "model_id",
                    "hostname": "hostname",
                    "team": {"id": "team1", "name": "Team Alpha"},
                    "versions": versions,
                }
            ]
        }
    }

    with requests_mock.Mocker() as m:
        m.post(
            _TEST_REMOTE_GRAPHQL_PATH,
            [
                {"json": teams_response},
                {"json": models_response},
                {"json": teams_response},
                {"json": models_response},
            ],
        )

        # Since no production version exists, calling get_service with
        # published=True should raise an error.
        with pytest.raises(RemoteError):
            remote.get_service(model_identifier=ModelName("model_name"), published=True)

        # Check that the development version is returned when published is False.
        service = remote.get_service(
            model_identifier=ModelName("model_name"), published=False
        )
        assert service.model_id == "model_id"
        assert service.model_version_id == "1"


def test_get_service_by_model_id(remote):
    model_response = {
        "data": {
            "model": {
                "name": "model_name",
                "id": "model_id",
                "primary_version": {"id": "version_id"},
                "hostname": "hostname",
            }
        }
    }

    with requests_mock.Mocker() as m:
        m.post(_TEST_REMOTE_GRAPHQL_PATH, json=model_response)

        service = remote.get_service(model_identifier=ModelId("model_id"))
        assert service.model_id == "model_id"
        assert service.model_version_id == "version_id"


def test_get_service_by_model_id_no_model(remote):
    model_response = {"errors": [{"message": "error"}]}
    with requests_mock.Mocker() as m:
        m.post(_TEST_REMOTE_GRAPHQL_PATH, json=model_response)
        with pytest.raises(RemoteError):
            remote.get_service(model_identifier=ModelId("model_id"))


def test_push_raised_value_error_when_deployment_name_and_not_publish(
    custom_model_truss_dir_with_pre_and_post, remote
):
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
        m.post(_TEST_REMOTE_GRAPHQL_PATH, json=model_response)
        th = TrussHandle(custom_model_truss_dir_with_pre_and_post)

        with pytest.raises(
            ValueError,
            match="Deployment name cannot be used for development deployment",
        ):
            remote.push(
                th,
                "model_name",
                th.truss_dir,
                publish=False,
                promote=False,
                preserve_previous_prod_deployment=False,
                deployment_name="dep_name",
            )


def test_push_raised_value_error_when_deployment_name_is_not_valid(
    custom_model_truss_dir_with_pre_and_post, remote
):
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
        m.post(_TEST_REMOTE_GRAPHQL_PATH, json=model_response)
        th = TrussHandle(custom_model_truss_dir_with_pre_and_post)

        with pytest.raises(
            ValueError,
            match="Deployment name must only contain alphanumeric, -, _ and . characters",
        ):
            remote.push(
                th,
                "model_name",
                th.truss_dir,
                publish=True,
                promote=False,
                preserve_previous_prod_deployment=False,
                deployment_name="dep//name",
            )


def test_push_raised_value_error_when_keep_previous_prod_settings_and_not_promote(
    custom_model_truss_dir_with_pre_and_post, remote
):
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
        m.post(_TEST_REMOTE_GRAPHQL_PATH, json=model_response)
        th = TrussHandle(custom_model_truss_dir_with_pre_and_post)

        with pytest.raises(
            ValueError,
            match="preserve-previous-production-deployment can only be used with the '--promote' option",
        ):
            remote.push(
                th,
                "model_name",
                th.truss_dir,
                publish=False,
                promote=False,
                preserve_previous_prod_deployment=True,
            )


@pytest.mark.parametrize("deploy_timeout_minutes", [9, 1441])
def test_push_raised_value_error_when_deploy_timeout_minutes_is_invalid(
    deploy_timeout_minutes, custom_model_truss_dir_with_pre_and_post, remote
):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)

    with pytest.raises(
        ValueError,
        match="deploy-timeout-minutes must be between 10 minutes and 1440 minutes \(24 hours\)",
    ):
        remote.push(
            th,
            "model_name",
            th.truss_dir,
            publish=True,
            promote=False,
            preserve_previous_prod_deployment=False,
            deployment_name="dep_name",
            deploy_timeout_minutes=deploy_timeout_minutes,
        )


def test_create_chain_with_no_publish(remote):
    mock_deploy_response = {
        "chain_deployment": {
            "id": "new-chain-deployment-id",
            "chain": {"id": "new-chain-id", "hostname": "hostname"},
        }
    }

    with (
        mock.patch.object(remote.api, "get_chains", return_value=[]) as mock_get_chains,
        mock.patch.object(
            remote.api, "deploy_chain_atomic", return_value=mock_deploy_response
        ) as mock_deploy,
    ):
        deployment_handle = create_chain_atomic(
            api=remote.api,
            chain_name="draft_chain",
            entrypoint=ChainletDataAtomic(
                name="chainlet-1",
                oracle=OracleData(
                    model_name="model-1",
                    s3_key="s3-key-1",
                    encoded_config_str="encoded-config-str-1",
                ),
            ),
            dependencies=[],
            truss_user_env=b10_types.TrussUserEnv.collect(),
            is_draft=True,
            environment=None,
        )

        mock_get_chains.assert_called_once()
        mock_deploy.assert_called_once()

        call_kwargs = mock_deploy.call_args.kwargs
        assert call_kwargs["chain_name"] == "draft_chain"
        assert call_kwargs.get("is_draft") is True
        assert call_kwargs.get("deploy_timeout_minutes") is None

        assert deployment_handle.chain_id == "new-chain-id"
        assert deployment_handle.chain_deployment_id == "new-chain-deployment-id"


def test_create_chain_no_existing_chain(remote):
    mock_deploy_response = {
        "chain_deployment": {
            "id": "new-chain-deployment-id",
            "chain": {"id": "new-chain-id", "hostname": "hostname"},
        }
    }

    with (
        mock.patch.object(remote.api, "get_chains", return_value=[]) as mock_get_chains,
        mock.patch.object(
            remote.api, "deploy_chain_atomic", return_value=mock_deploy_response
        ) as mock_deploy,
    ):
        deployment_handle = create_chain_atomic(
            api=remote.api,
            chain_name="new_chain",
            entrypoint=ChainletDataAtomic(
                name="chainlet-1",
                oracle=OracleData(
                    model_name="model-1",
                    s3_key="s3-key-1",
                    encoded_config_str="encoded-config-str-1",
                ),
            ),
            dependencies=[],
            truss_user_env=b10_types.TrussUserEnv.collect(),
            is_draft=False,
            environment=None,
        )

        mock_get_chains.assert_called_once()
        mock_deploy.assert_called_once()

        call_kwargs = mock_deploy.call_args.kwargs
        assert call_kwargs["chain_name"] == "new_chain"
        assert call_kwargs.get("is_draft") is not True
        assert call_kwargs.get("deploy_timeout_minutes") is None

        assert deployment_handle.chain_id == "new-chain-id"
        assert deployment_handle.chain_deployment_id == "new-chain-deployment-id"


def test_create_chain_with_deployment_name(remote):
    with requests_mock.Mocker() as m:
        m.post(
            _TEST_REMOTE_GRAPHQL_PATH,
            [
                {"json": {"data": {"chains": []}}},
                {
                    "json": {
                        "data": {
                            "deploy_chain_atomic": {
                                "chain_deployment": {
                                    "id": "new-chain-deployment-id",
                                    "chain": {
                                        "id": "new-chain-id",
                                        "hostname": "hostname",
                                    },
                                }
                            }
                        }
                    }
                },
            ],
        )

        deployment_name = "chain-deployment"
        create_chain_atomic(
            api=remote.api,
            chain_name="new_chain",
            entrypoint=ChainletDataAtomic(
                name="chainlet-1",
                oracle=OracleData(
                    model_name="model-1",
                    s3_key="s3-key-1",
                    encoded_config_str="encoded-config-str-1",
                ),
            ),
            dependencies=[],
            truss_user_env=b10_types.TrussUserEnv.collect(),
            is_draft=False,
            environment=None,
            deployment_name=deployment_name,
        )

        create_chain_graphql_request = m.request_history[1]

        assert (
            'deployment_name: "chain-deployment"'
            in create_chain_graphql_request.json()["query"]
        )


def test_create_chain_with_existing_chain_promote_to_environment_publish_false(remote):
    mock_deploy_response = {
        "chain_deployment": {
            "id": "new-chain-deployment-id",
            "chain": {"id": "new-chain-id", "hostname": "hostname"},
        }
    }

    with (
        mock.patch.object(
            remote.api,
            "get_chains",
            return_value=[{"id": "old-chain-id", "name": "old_chain"}],
        ) as mock_get_chains,
        mock.patch.object(
            remote.api, "deploy_chain_atomic", return_value=mock_deploy_response
        ) as mock_deploy,
    ):
        deployment_handle = create_chain_atomic(
            api=remote.api,
            chain_name="old_chain",
            entrypoint=ChainletDataAtomic(
                name="chainlet-1",
                oracle=OracleData(
                    model_name="model-1",
                    s3_key="s3-key-1",
                    encoded_config_str="encoded-config-str-1",
                ),
            ),
            dependencies=[],
            truss_user_env=b10_types.TrussUserEnv.collect(),
            is_draft=True,
            environment="production",
        )

        mock_get_chains.assert_called_once()
        mock_deploy.assert_called_once()

        call_kwargs = mock_deploy.call_args.kwargs
        assert call_kwargs["chain_id"] == "old-chain-id"
        assert call_kwargs["environment"] == "production"
        assert call_kwargs.get("is_draft") is not True
        assert call_kwargs.get("deploy_timeout_minutes") is None

        assert deployment_handle.chain_id == "new-chain-id"
        assert deployment_handle.chain_deployment_id == "new-chain-deployment-id"


def test_create_chain_existing_chain_publish_true_no_promotion(remote):
    mock_deploy_response = {
        "chain_deployment": {
            "id": "new-chain-deployment-id",
            "chain": {"id": "new-chain-id", "hostname": "hostname"},
        }
    }

    with (
        mock.patch.object(
            remote.api,
            "get_chains",
            return_value=[{"id": "old-chain-id", "name": "old_chain"}],
        ) as mock_get_chains,
        mock.patch.object(
            remote.api, "deploy_chain_atomic", return_value=mock_deploy_response
        ) as mock_deploy,
    ):
        deployment_handle = create_chain_atomic(
            api=remote.api,
            chain_name="old_chain",
            entrypoint=ChainletDataAtomic(
                name="chainlet-1",
                oracle=OracleData(
                    model_name="model-1",
                    s3_key="s3-key-1",
                    encoded_config_str="encoded-config-str-1",
                ),
            ),
            dependencies=[],
            truss_user_env=b10_types.TrussUserEnv.collect(),
            is_draft=False,
            environment=None,
        )

        mock_get_chains.assert_called_once()
        mock_deploy.assert_called_once()

        call_kwargs = mock_deploy.call_args.kwargs
        assert call_kwargs["chain_id"] == "old-chain-id"
        assert call_kwargs.get("is_draft") is not True
        assert call_kwargs.get("deploy_timeout_minutes") is None

        assert deployment_handle.chain_id == "new-chain-id"
        assert deployment_handle.chain_deployment_id == "new-chain-deployment-id"


@pytest.mark.parametrize("publish", [True, False])
def test_push_raised_value_error_when_disable_truss_download_for_existing_model(
    publish, custom_model_truss_dir_with_pre_and_post, remote
):
    models_response = {
        "data": {
            "models": [
                {
                    "id": "model_id",
                    "name": "model_name",
                    "team": {"id": "team_id", "name": "Team Name"},
                    "versions": [],
                }
            ]
        }
    }
    validation_response = {
        "data": {"truss_validation": {"success": True, "details": "{}"}}
    }

    def response_callback(request, context):
        query = request.json().get("query", "")
        if "models(" in query:
            return models_response
        elif "truss_validation" in query:
            return validation_response
        return {"data": {}}

    with requests_mock.Mocker() as m:
        m.post(_TEST_REMOTE_GRAPHQL_PATH, json=response_callback)
        th = TrussHandle(custom_model_truss_dir_with_pre_and_post)

        with pytest.raises(
            ValueError, match="disable-truss-download can only be used for new models"
        ):
            remote.push(
                th,
                "model_name",
                th.truss_dir,
                publish=publish,
                disable_truss_download=True,
            )


def test_push_raised_validation_error_for_extra_fields(tmp_path, remote):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
    model_name: Hello
    extra_field: 123
    who_am_i: 0.2
    """)
    th = TrussHandle(tmp_path)
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
        m.post(_TEST_REMOTE_GRAPHQL_PATH, json=model_response)
        with pytest.raises(
            pydantic.ValidationError,
            match="Extra fields not allowed: \[extra_field, who_am_i\]",
        ):
            remote.push(th, "model_name", th.truss_dir)


def test_push_passes_deploy_timeout_minutes_to_create_truss_service(
    custom_model_truss_dir_with_pre_and_post,
    remote,
    mock_baseten_requests,
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
    mock_baseten_requests,
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
    mock_baseten_requests,
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


def test_api_push_integration_deploy_timeout_minutes_propagated(
    custom_model_truss_dir_with_pre_and_post,
    mock_remote_factory,
    temp_trussrc_dir,
    mock_available_config_names,
    mock_truss_handle,
):
    from truss.api import push

    push(
        str(mock_truss_handle.truss_dir),
        remote="baseten",
        model_name="test_model",
        deploy_timeout_minutes=1200,
    )

    # Verify the remote.push was called with deploy_timeout_minutes
    mock_remote_factory.push.assert_called_once()
    _, push_kwargs = mock_remote_factory.push.call_args
    assert push_kwargs.get("deploy_timeout_minutes") == 1200
