from unittest import mock

import pytest

from truss.remote.baseten.remote import BasetenRemote
from truss.remote.remote_factory import API_KEY_ENV, ENV_REMOTE_NAME, REMOTE_URL_ENV
from truss_chains import public_api
from truss_chains.deployment import deployment_client


class _Entrypoint:
    pass


@pytest.fixture
def push_calls():
    service = mock.Mock(spec=deployment_client.BasetenChainService)
    with (
        mock.patch(
            "truss.remote.remote_factory.RemoteFactory.create",
            return_value=mock.Mock(spec=BasetenRemote),
        ) as create,
        mock.patch.object(deployment_client, "push", return_value=service),
    ):
        yield create


def test_push_without_a_remote_uses_the_env_remote(push_calls, monkeypatch):
    monkeypatch.setenv(REMOTE_URL_ENV, "https://app.test.com")
    monkeypatch.setenv(API_KEY_ENV, "env_key")

    public_api.push(_Entrypoint, chain_name="my-chain")

    push_calls.assert_called_once_with(remote=ENV_REMOTE_NAME)


def test_push_without_a_remote_defaults_to_baseten(push_calls, monkeypatch):
    monkeypatch.delenv(REMOTE_URL_ENV, raising=False)
    monkeypatch.delenv(API_KEY_ENV, raising=False)

    public_api.push(_Entrypoint, chain_name="my-chain")

    push_calls.assert_called_once_with(remote="baseten")
