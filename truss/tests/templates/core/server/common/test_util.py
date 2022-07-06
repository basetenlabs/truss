from unittest import mock

from truss.templates.server.common.util import \
    assign_request_to_inputs_instances_after_validation


def test_assign_request_to_inputs_instances_after_validation():
    inputs_input = [1, 2, 3, 4]
    inputs_dict = {'inputs': inputs_input}
    instances_input = [5, 6, 7, 8]
    instances_dict = {'instances': instances_input}

    processed_inputs = assign_request_to_inputs_instances_after_validation(inputs_dict)
    processed_instances = assign_request_to_inputs_instances_after_validation(instances_dict)

    assert processed_inputs['instances'] == processed_inputs['inputs'] == inputs_input
    assert processed_instances['instances'] == processed_instances['inputs'] == instances_input


def model_supports_predict_proba():
    mock_not_predict_proba = mock.Mock(name='mock_not_predict_proba')
    mock_not_predict_proba.predict_proba.return_value = False

    mock_check_proba = mock.Mock(name='mock_check_proba')
    mock_check_proba.predict_proba.return_value = True
    mock_check_proba._check_proba.return_value = True

    mock_not_check_proba = mock.Mock(name='mock_not_check_proba')
    mock_not_check_proba.predict_proba.return_value = True
    mock_not_check_proba._check_proba.side_effect = AttributeError

    assert not model_supports_predict_proba(mock_not_predict_proba)
    assert model_supports_predict_proba(mock_check_proba)
    assert not model_supports_predict_proba(mock_not_check_proba)
