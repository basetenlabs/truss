from unittest import mock

import pytest

from truss.remote.baseten.api import BasetenApi


@pytest.fixture
def mock_auth_service():
    auth_service = mock.Mock()
    auth_token = mock.Mock(headers=lambda: {"Authorization": "Api-Key token"})
    auth_service.authenticate.return_value = auth_token
    return auth_service


@pytest.fixture
def baseten_api(mock_auth_service):
    return BasetenApi("https://app.test.com", mock_auth_service)
