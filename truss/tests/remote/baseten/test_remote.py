from urllib import parse

import pytest
import requests_mock
import truss
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


def request_matches_expected_query(request, expected_query):
    unescaped_content = parse.unquote_plus(request.text)

    # Remove 'query=' prefix and any leading/trailing whitespace
    actual_query = unescaped_content.replace("query=", "").strip()

    return tuple(
        line.strip() for line in actual_query.split("\n") if line.strip()
    ) == tuple(line.strip() for line in expected_query.split("\n") if line.strip())


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
        with pytest.raises(RemoteError):
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
        with pytest.raises(RemoteError):
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
        with pytest.raises(RemoteError):
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
        with pytest.raises(RemoteError):
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
            remote.push(
                th,
                "model_name",
                publish=False,
                trusted=False,
                promote=False,
                preserve_previous_prod_deployment=False,
                deployment_name="dep_name",
            )


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
            remote.push(
                th,
                "model_name",
                publish=True,
                trusted=False,
                promote=False,
                preserve_previous_prod_deployment=False,
                deployment_name="dep//name",
            )


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
            remote.push(
                th,
                "model_name",
                publish=False,
                trusted=False,
                promote=False,
                preserve_previous_prod_deployment=True,
            )


def test_create_chain_with_no_publish():
    remote = BasetenRemote(_TEST_REMOTE_URL, "api_key")

    with requests_mock.Mocker() as m:
        m.post(
            _TEST_REMOTE_GRAPHQL_PATH,
            [
                {"json": {"data": {"chains": []}}},
                {
                    "json": {
                        "data": {
                            "deploy_chain_atomic": {
                                "chain_id": "new-chain-id",
                                "chain_deployment_id": "new-chain-deployment-id",
                                "entrypoint_model_id": "new-entrypoint-model-id",
                                "entrypoint_model_version_id": "new-entrypoint-model-version-id",
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
                    is_trusted=True,
                ),
            ),
            dependencies=[],
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

        assert request_matches_expected_query(
            get_chains_graphql_request, expected_get_chains_query
        )

        chainlets_string = """
            {
                name: "chainlet-1",
                oracle: {
                    model_name: "model-1",
                    s3_key: "s3-key-1",
                    encoded_config_str: "encoded-config-str-1",
                    is_trusted: true,
                    semver_bump: "MINOR"
                }
            }
        """.strip()

        # Note that if publish=False and promote=True, we set publish to True and create
        # a non-draft deployment
        expected_create_chain_mutation = f"""
            mutation {{
                deploy_chain_atomic(
                    chain_name: "draft_chain"
                    is_draft: true
                    entrypoint: {chainlets_string}
                    dependencies: []
                    client_version: "truss=={truss.version()}"
                ) {{
                    chain_id
                    chain_deployment_id
                    entrypoint_model_id
                    entrypoint_model_version_id
                }}
            }}
        """.strip()

        assert request_matches_expected_query(
            create_chain_graphql_request, expected_create_chain_mutation
        )

        assert deployment_handle.chain_id == "new-chain-id"
        assert deployment_handle.chain_deployment_id == "new-chain-deployment-id"


def test_create_chain_no_existing_chain():
    remote = BasetenRemote(_TEST_REMOTE_URL, "api_key")

    with requests_mock.Mocker() as m:
        m.post(
            _TEST_REMOTE_GRAPHQL_PATH,
            [
                {"json": {"data": {"chains": []}}},
                {
                    "json": {
                        "data": {
                            "deploy_chain_atomic": {
                                "chain_id": "new-chain-id",
                                "chain_deployment_id": "new-chain-deployment-id",
                                "entrypoint_model_id": "new-entrypoint-model-id",
                                "entrypoint_model_version_id": "new-entrypoint-model-version-id",
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
                    is_trusted=True,
                ),
            ),
            dependencies=[],
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

        assert request_matches_expected_query(
            get_chains_graphql_request, expected_get_chains_query
        )

        chainlets_string = """
            {
                name: "chainlet-1",
                oracle: {
                    model_name: "model-1",
                    s3_key: "s3-key-1",
                    encoded_config_str: "encoded-config-str-1",
                    is_trusted: true,
                    semver_bump: "MINOR"
                }
            }
        """.strip()

        expected_create_chain_mutation = f"""
            mutation {{
                deploy_chain_atomic(
                    chain_name: "new_chain"
                    is_draft: false
                    entrypoint: {chainlets_string}
                    dependencies: []
                    client_version: "truss=={truss.version()}"
                ) {{
                    chain_id
                    chain_deployment_id
                    entrypoint_model_id
                    entrypoint_model_version_id
                }}
            }}
        """.strip()

        assert request_matches_expected_query(
            create_chain_graphql_request, expected_create_chain_mutation
        )

        assert deployment_handle.chain_id == "new-chain-id"
        assert deployment_handle.chain_deployment_id == "new-chain-deployment-id"


def test_create_chain_with_existing_chain_promote_to_environment_publish_false():
    remote = BasetenRemote(_TEST_REMOTE_URL, "api_key")

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
                                "chain_id": "new-chain-id",
                                "chain_deployment_id": "new-chain-deployment-id",
                                "entrypoint_model_id": "new-entrypoint-model-id",
                                "entrypoint_model_version_id": "new-entrypoint-model-version-id",
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
                    is_trusted=True,
                ),
            ),
            dependencies=[],
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

        assert request_matches_expected_query(
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
                    is_trusted: true,
                    semver_bump: "MINOR"
                }
            }
        """.strip()

        expected_create_chain_mutation = f"""
            mutation {{
                deploy_chain_atomic(
                    chain_id: "old-chain-id"
                    environment: "production"
                    is_draft: false
                    entrypoint: {chainlets_string}
                    dependencies: []
                    client_version: "truss=={truss.version()}"
                ) {{
                    chain_id
                    chain_deployment_id
                    entrypoint_model_id
                    entrypoint_model_version_id
                }}
            }}
        """.strip()

        assert request_matches_expected_query(
            create_chain_graphql_request, expected_create_chain_mutation
        )

        assert deployment_handle.chain_id == "new-chain-id"
        assert deployment_handle.chain_deployment_id == "new-chain-deployment-id"


def test_create_chain_existing_chain_publish_true_no_promotion():
    remote = BasetenRemote(_TEST_REMOTE_URL, "api_key")

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
                                "chain_id": "new-chain-id",
                                "chain_deployment_id": "new-chain-deployment-id",
                                "entrypoint_model_id": "new-entrypoint-model-id",
                                "entrypoint_model_version_id": "new-entrypoint-model-version-id",
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
                    is_trusted=True,
                ),
            ),
            dependencies=[],
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

        assert request_matches_expected_query(
            get_chains_graphql_request, expected_get_chains_query
        )

        chainlets_string = """
            {
                name: "chainlet-1",
                oracle: {
                    model_name: "model-1",
                    s3_key: "s3-key-1",
                    encoded_config_str: "encoded-config-str-1",
                    is_trusted: true,
                    semver_bump: "MINOR"
                }
            }
        """.strip()

        expected_create_chain_mutation = f"""
            mutation {{
                deploy_chain_atomic(
                    chain_id: "old-chain-id"
                    is_draft: false
                    entrypoint: {chainlets_string}
                    dependencies: []
                    client_version: "truss=={truss.version()}"
                ) {{
                    chain_id
                    chain_deployment_id
                    entrypoint_model_id
                    entrypoint_model_version_id
                }}
            }}
        """.strip()

        assert request_matches_expected_query(
            create_chain_graphql_request, expected_create_chain_mutation
        )

        assert deployment_handle.chain_id == "new-chain-id"
        assert deployment_handle.chain_deployment_id == "new-chain-deployment-id"


@pytest.mark.parametrize(
    "publish",
    [True, False],
)
def test_push_raised_value_error_when_disable_truss_download_for_existing_model(
    publish,
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
            match="disable-truss-download can only be used for new models",
        ):
            remote.push(th, "model_name", publish=publish, disable_truss_download=True)
