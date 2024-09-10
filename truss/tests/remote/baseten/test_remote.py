from urllib import parse

import pytest
import requests_mock
from truss.remote.baseten.core import ModelId, ModelName, ModelVersionId
from truss.remote.baseten.custom_types import ChainletData
from truss.remote.baseten.error import RemoteError
from truss.remote.baseten.remote import BasetenRemote
from truss.truss_handle import TrussHandle

_TEST_REMOTE_URL = "http://test_remote.com"
_TEST_REMOTE_GRAPHQL_PATH = "http://test_remote.com/graphql/"


def match_graphql_request(request, expected_query):
    unescaped_content = parse.unquote_plus(request.text)

    # Remove 'query=' prefix and any leading/trailing whitespace
    graphql_query = unescaped_content.replace("query=", "").strip()

    assert graphql_query == expected_query


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
                            "deploy_draft_chain": {
                                "chain_id": "new-chain-id",
                                "chain_deployment_id": "new-chain-deployment-id",
                            }
                        }
                    }
                },
            ],
        )

        deployment_handle = remote.create_chain(
            "draft_chain",
            [
                ChainletData(
                    name="chainlet-1",
                    oracle_version_id="some-ov-id",
                    is_entrypoint=True,
                )
            ],
            publish=False,
            promote=False,
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

        match_graphql_request(get_chains_graphql_request, expected_get_chains_query)
        # Note that if publish=False and promote=True, we set publish to True and create
        # a non-draft deployment
        expected_create_chain_mutation = """
        mutation {
        deploy_draft_chain(
            name: "draft_chain",
            chainlets: [
        {
            name: "chainlet-1",
            oracle_version_id: "some-ov-id",
            is_entrypoint: true
        }
        ]
        ) {
            chain_id
            chain_deployment_id
        }
        }
        """.strip()

        match_graphql_request(
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
                            "deploy_chain": {
                                "id": "new-chain-id",
                                "chain_id": "new-chain-id",
                                "chain_deployment_id": "new-chain-deployment-id",
                            }
                        }
                    }
                },
            ],
        )

        deployment_handle = remote.create_chain(
            "new_chain",
            [
                ChainletData(
                    name="chainlet-1",
                    oracle_version_id="some-ov-id",
                    is_entrypoint=True,
                )
            ],
            publish=True,
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

        match_graphql_request(get_chains_graphql_request, expected_get_chains_query)

        expected_create_chain_mutation = """
        mutation {
        deploy_chain(
            name: "new_chain",
            chainlets: [
        {
            name: "chainlet-1",
            oracle_version_id: "some-ov-id",
            is_entrypoint: true
        }
        ]
        ) {
            id
            chain_id
            chain_deployment_id
        }
        }
        """.strip()

        match_graphql_request(
            create_chain_graphql_request, expected_create_chain_mutation
        )

        assert deployment_handle.chain_id == "new-chain-id"
        assert deployment_handle.chain_deployment_id == "new-chain-deployment-id"


def test_create_chain_with_existing_chain_promote_true_publish_false():
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
                            "deploy_chain_deployment": {
                                "id": "new-chain-id",
                                "chain_id": "new-chain-id",
                                "chain_deployment_id": "new-chain-deployment-id",
                            }
                        }
                    }
                },
            ],
        )

        deployment_handle = remote.create_chain(
            "old_chain",
            [
                ChainletData(
                    name="chainlet-1",
                    oracle_version_id="some-ov-id",
                    is_entrypoint=True,
                )
            ],
            publish=False,
            promote=True,
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

        match_graphql_request(get_chains_graphql_request, expected_get_chains_query)
        # Note that if publish=False and promote=True, we set publish to True and create
        # a non-draft deployment
        expected_create_chain_mutation = """
        mutation {
        deploy_chain_deployment(
            chain_id: "old-chain-id",
            chainlets: [
        {
            name: "chainlet-1",
            oracle_version_id: "some-ov-id",
            is_entrypoint: true
        }
        ],
            promote_after_deploy: true,
        ) {
            chain_id
            chain_deployment_id
        }
        }
        """.strip()

        match_graphql_request(
            create_chain_graphql_request, expected_create_chain_mutation
        )

        assert deployment_handle.chain_id == "new-chain-id"
        assert deployment_handle.chain_deployment_id == "new-chain-deployment-id"


def test_create_chain_existing_chain_publish_true_promote_false():
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
                            "deploy_chain_deployment": {
                                "id": "new-chain-id",
                                "chain_id": "new-chain-id",
                                "chain_deployment_id": "new-chain-deployment-id",
                            }
                        }
                    }
                },
            ],
        )

        deployment_handle = remote.create_chain(
            "old_chain",
            [
                ChainletData(
                    name="chainlet-1",
                    oracle_version_id="some-ov-id",
                    is_entrypoint=True,
                )
            ],
            publish=True,
            promote=False,
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

        match_graphql_request(get_chains_graphql_request, expected_get_chains_query)
        # Note promote_after_deploy is false
        expected_create_chain_mutation = """
        mutation {
        deploy_chain_deployment(
            chain_id: "old-chain-id",
            chainlets: [
        {
            name: "chainlet-1",
            oracle_version_id: "some-ov-id",
            is_entrypoint: true
        }
        ],
            promote_after_deploy: false,
        ) {
            chain_id
            chain_deployment_id
        }
        }
        """.strip()

        match_graphql_request(
            create_chain_graphql_request, expected_create_chain_mutation
        )

        assert deployment_handle.chain_id == "new-chain-id"
        assert deployment_handle.chain_deployment_id == "new-chain-deployment-id"
