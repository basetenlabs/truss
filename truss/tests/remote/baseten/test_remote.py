import pathlib
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
from truss.remote.baseten.remote import BasetenRemote
from truss.truss_handle.truss_handle import TrussHandle

_TEST_REMOTE_URL = "http://test_remote.com"
_TEST_REMOTE_GRAPHQL_PATH = "http://test_remote.com/graphql/"

TRUSS_RC_CONTENT = """
[baseten]
remote_provider = baseten
api_key = test_key
remote_url = http://test.com
""".strip()


@pytest.fixture
def remote():
    return BasetenRemote(_TEST_REMOTE_URL, "api_key")


@pytest.fixture
def model_response():
    return {
        "data": {
            "model": {
                "name": "model_name",
                "id": "model_id",
                "primary_version": {"id": "version_id"},
            }
        }
    }


@pytest.fixture
def mock_model_version_handle():
    from truss.remote.baseten.core import ModelVersionHandle

    return ModelVersionHandle(
        version_id="version_id", model_id="model_id", hostname="hostname"
    )


@pytest.fixture
def setup_push_mocks(model_response):
    def _setup(m):
        # Mock for get_model query - matches queries containing "model(name"
        m.post(
            _TEST_REMOTE_GRAPHQL_PATH,
            json=model_response,
            additional_matcher=lambda req: "model(name" in req.json().get("query", ""),
        )
        # Mock for validate_truss query - matches queries containing "truss_validation"
        m.post(
            _TEST_REMOTE_GRAPHQL_PATH,
            json={"data": {"truss_validation": {"success": True, "details": "{}"}}},
            additional_matcher=lambda req: "truss_validation"
            in req.json().get("query", ""),
        )
        # Mock for model_s3_upload_credentials query
        m.post(
            _TEST_REMOTE_GRAPHQL_PATH,
            json={
                "data": {
                    "model_s3_upload_credentials": {
                        "s3_bucket": "bucket",
                        "s3_key": "key",
                        "aws_access_key_id": "key_id",
                        "aws_secret_access_key": "secret",
                        "aws_session_token": "token",
                    }
                }
            },
            additional_matcher=lambda req: "model_s3_upload_credentials"
            in req.json().get("query", ""),
        )
        m.post(
            "http://test_remote.com/v1/models/model_id/upload",
            json={"s3_bucket": "bucket", "s3_key": "key"},
        )
        m.post(
            "http://test_remote.com/v1/blobs/credentials/truss",
            json={
                "s3_bucket": "bucket",
                "s3_key": "key",
                "aws_access_key_id": "key_id",
                "aws_secret_access_key": "secret",
                "aws_session_token": "token",
            },
        )
        # Mock for create_model_version_from_truss mutation
        m.post(
            "http://test_remote.com/graphql/",
            json={
                "data": {
                    "create_model_version_from_truss": {
                        "model_version": {
                            "id": "version_id",
                            "oracle": {"id": "model_id", "hostname": "hostname"},
                        }
                    }
                }
            },
            additional_matcher=lambda req: "create_model_version_from_truss"
            in req.json().get("query", ""),
        )

    return _setup


@pytest.fixture
def mock_remote_factory():
    """Fixture that mocks RemoteFactory.create and returns a configured mock remote."""
    from unittest.mock import MagicMock, patch

    from truss.remote.remote_factory import RemoteFactory

    with patch.object(RemoteFactory, "create") as mock_factory:
        mock_remote = MagicMock()
        mock_service = MagicMock()
        mock_service.model_id = "model_id"
        mock_service.model_version_id = "version_id"
        mock_remote.push.return_value = mock_service
        mock_factory.return_value = mock_remote
        yield mock_remote


@pytest.fixture
def temp_trussrc_dir():
    """Fixture that creates a temporary directory with a .trussrc file."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        trussrc_path = pathlib.Path(tmpdir) / ".trussrc"
        trussrc_path.write_text(TRUSS_RC_CONTENT)
        yield tmpdir


@pytest.fixture
def mock_available_config_names():
    """Fixture that patches RemoteFactory.get_available_config_names."""
    from unittest.mock import patch

    with patch(
        "truss.api.RemoteFactory.get_available_config_names", return_value=["baseten"]
    ):
        yield


@pytest.fixture
def mocked_push_requests(setup_push_mocks):
    """Fixture that provides a configured requests_mock.Mocker with push mocks setup."""
    with requests_mock.Mocker() as m:
        setup_push_mocks(m)
        yield m


@pytest.fixture
def mock_upload_truss():
    """Fixture that patches upload_truss and returns a mock."""
    with mock.patch("truss.remote.baseten.remote.upload_truss") as mock_upload:
        mock_upload.return_value = "s3_key"
        yield mock_upload


@pytest.fixture
def mock_create_truss_service(mock_model_version_handle):
    """Fixture that patches create_truss_service and returns a mock."""
    with mock.patch("truss.remote.baseten.remote.create_truss_service") as mock_create:
        mock_create.return_value = mock_model_version_handle
        yield mock_create


@pytest.fixture
def mock_truss_handle(custom_model_truss_dir_with_pre_and_post):
    from truss.truss_handle.truss_handle import TrussHandle

    truss_handle = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    return truss_handle


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
        m.post(_TEST_REMOTE_GRAPHQL_PATH, json=model_response)

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
        m.post(_TEST_REMOTE_GRAPHQL_PATH, json=model_response)

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
        m.post(_TEST_REMOTE_GRAPHQL_PATH, json=model_response)

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


def test_create_chain_with_no_publish(remote):
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

        get_chains_graphql_request = m.request_history[0]
        create_chain_graphql_request = m.request_history[1]

        expected_get_chains_query = """
            {
                chains {
                    id
                    name
                }
            }
        """.strip()

        assert_request_matches_expected_query(
            get_chains_graphql_request, expected_get_chains_query
        )

        chainlets_string = """
            {
                name: "chainlet-1",
                oracle: {
                    model_name: "model-1",
                    s3_key: "s3-key-1",
                    encoded_config_str: "encoded-config-str-1",
                    semver_bump: "MINOR"
                }
            }
        """.strip()

        # Note that if publish=False and promote=True, we set publish to True and create
        # a non-draft deployment
        expected_create_chain_mutation = f"""
            mutation ($trussUserEnv: String) {{
                deploy_chain_atomic(
                    chain_name: "draft_chain"
                    is_draft: true
                    entrypoint: {chainlets_string}
                    dependencies: []
                    truss_user_env: $trussUserEnv
                ) {{
                    chain_deployment {{
                        id
                        chain {{
                            id
                            hostname
                        }}
                    }}
                }}
            }}
        """.strip()

        assert_request_matches_expected_query(
            create_chain_graphql_request, expected_create_chain_mutation
        )
        assert deployment_handle.chain_id == "new-chain-id"
        assert deployment_handle.chain_deployment_id == "new-chain-deployment-id"


def test_create_chain_no_existing_chain(remote):
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

        get_chains_graphql_request = m.request_history[0]
        create_chain_graphql_request = m.request_history[1]

        expected_get_chains_query = """
            {
                chains {
                    id
                    name
                }
            }
        """.strip()

        assert_request_matches_expected_query(
            get_chains_graphql_request, expected_get_chains_query
        )

        chainlets_string = """
            {
                name: "chainlet-1",
                oracle: {
                    model_name: "model-1",
                    s3_key: "s3-key-1",
                    encoded_config_str: "encoded-config-str-1",
                    semver_bump: "MINOR"
                }
            }
        """.strip()

        expected_create_chain_mutation = f"""
            mutation ($trussUserEnv: String) {{
                deploy_chain_atomic(
                    chain_name: "new_chain"
                    is_draft: false
                    entrypoint: {chainlets_string}
                    dependencies: []
                    truss_user_env: $trussUserEnv
                ) {{
                    chain_deployment {{
                        id
                        chain {{
                            id
                            hostname
                        }}
                    }}
                }}
            }}
        """.strip()

        assert_request_matches_expected_query(
            create_chain_graphql_request, expected_create_chain_mutation
        )

        assert deployment_handle.chain_id == "new-chain-id"
        assert deployment_handle.chain_deployment_id == "new-chain-deployment-id"


def test_create_chain_with_existing_chain_promote_to_environment_publish_false(remote):
    with requests_mock.Mocker() as m:
        m.post(
            _TEST_REMOTE_GRAPHQL_PATH,
            [
                {
                    "json": {
                        "data": {
                            "chains": [{"id": "old-chain-id", "name": "old_chain"}]
                        }
                    }
                },
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

        get_chains_graphql_request = m.request_history[0]
        create_chain_graphql_request = m.request_history[1]

        expected_get_chains_query = """
            {
                chains {
                    id
                    name
                }
            }
        """.strip()

        assert_request_matches_expected_query(
            get_chains_graphql_request, expected_get_chains_query
        )

        # Note that if publish=False and environment!=None, we set publish to True and create
        # a non-draft deployment
        chainlets_string = """
            {
                name: "chainlet-1",
                oracle: {
                    model_name: "model-1",
                    s3_key: "s3-key-1",
                    encoded_config_str: "encoded-config-str-1",
                    semver_bump: "MINOR"
                }
            }
        """.strip()

        expected_create_chain_mutation = f"""
            mutation ($trussUserEnv: String) {{
                deploy_chain_atomic(
                    chain_id: "old-chain-id"
                    environment: "production"
                    is_draft: false
                    entrypoint: {chainlets_string}
                    dependencies: []
                    truss_user_env: $trussUserEnv
                ) {{
                    chain_deployment {{
                        id
                        chain {{
                            id
                            hostname
                        }}
                    }}
                }}
            }}
        """.strip()

        assert_request_matches_expected_query(
            create_chain_graphql_request, expected_create_chain_mutation
        )

        assert deployment_handle.chain_id == "new-chain-id"
        assert deployment_handle.chain_deployment_id == "new-chain-deployment-id"


def test_create_chain_existing_chain_publish_true_no_promotion(remote):
    with requests_mock.Mocker() as m:
        m.post(
            _TEST_REMOTE_GRAPHQL_PATH,
            [
                {
                    "json": {
                        "data": {
                            "chains": [{"id": "old-chain-id", "name": "old_chain"}]
                        }
                    }
                },
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

        get_chains_graphql_request = m.request_history[0]
        create_chain_graphql_request = m.request_history[1]

        expected_get_chains_query = """
            {
                chains {
                    id
                    name
                }
            }
        """.strip()

        assert_request_matches_expected_query(
            get_chains_graphql_request, expected_get_chains_query
        )

        chainlets_string = """
            {
                name: "chainlet-1",
                oracle: {
                    model_name: "model-1",
                    s3_key: "s3-key-1",
                    encoded_config_str: "encoded-config-str-1",
                    semver_bump: "MINOR"
                }
            }
        """.strip()

        expected_create_chain_mutation = f"""
            mutation ($trussUserEnv: String) {{
                deploy_chain_atomic(
                    chain_id: "old-chain-id"
                    is_draft: false
                    entrypoint: {chainlets_string}
                    dependencies: []
                    truss_user_env: $trussUserEnv
                ) {{
                    chain_deployment {{
                        id
                        chain {{
                            id
                            hostname
                        }}
                    }}
                }}
            }}
        """.strip()

        assert_request_matches_expected_query(
            create_chain_graphql_request, expected_create_chain_mutation
        )

        assert deployment_handle.chain_id == "new-chain-id"
        assert deployment_handle.chain_deployment_id == "new-chain-deployment-id"


@pytest.mark.parametrize("publish", [True, False])
def test_push_raised_value_error_when_disable_truss_download_for_existing_model(
    publish, custom_model_truss_dir_with_pre_and_post, remote
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


def test_push_passes_deploy_timeout_to_create_truss_service(
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
        deploy_timeout=450,
    )

    mock_create_truss_service.assert_called_once()
    _, kwargs = mock_create_truss_service.call_args
    assert kwargs["deploy_timeout"] == 450


def test_push_passes_none_deploy_timeout_when_not_specified(
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
    assert kwargs.get("deploy_timeout") is None


def test_push_integration_deploy_timeout_propagated(
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
        deploy_timeout=750,
    )

    mock_create_truss_service.assert_called_once()
    _, kwargs = mock_create_truss_service.call_args
    assert kwargs["deploy_timeout"] == 750
    assert kwargs["environment"] == "staging"


def test_api_push_integration_deploy_timeout_propagated(
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
        deploy_timeout=1200,
    )

    # Verify the remote.push was called with deploy_timeout
    mock_remote_factory.push.assert_called_once()
    _, push_kwargs = mock_remote_factory.push.call_args
    assert push_kwargs.get("deploy_timeout") == 1200
