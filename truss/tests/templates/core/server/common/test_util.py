# This file doesn't test anything, but provides utilities for testing.
from unittest import mock


def model_supports_predict_proba():
    mock_not_predict_proba = mock.Mock(name="mock_not_predict_proba")
    mock_not_predict_proba.predict_proba.return_value = False

    mock_check_proba = mock.Mock(name="mock_check_proba")
    mock_check_proba.predict_proba.return_value = True
    mock_check_proba._check_proba.return_value = True

    mock_not_check_proba = mock.Mock(name="mock_not_check_proba")
    mock_not_check_proba.predict_proba.return_value = True
    mock_not_check_proba._check_proba.side_effect = AttributeError

    assert not model_supports_predict_proba(mock_not_predict_proba)
    assert model_supports_predict_proba(mock_check_proba)
    assert not model_supports_predict_proba(mock_not_check_proba)
