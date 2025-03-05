from unittest.mock import patch

import pytest

from truss.base.constants import TRTLLM_MIN_MEMORY_REQUEST_GI
from truss.trt_llm.config_checks import (
    is_missing_secrets_for_trt_llm_builder,
    memory_updated_for_trt_llm_builder,
)
from truss.truss_handle.truss_handle import TrussHandle


@patch("truss.trt_llm.config_checks._is_model_public")
@pytest.mark.parametrize(
    "has_secret, is_model_public, expected_result",
    [
        (False, False, True),
        (False, True, False),
        (True, False, False),
        (True, True, False),
    ],
)
def test_is_missing_secrets_for_trt_llm_builder(
    _is_model_public_mock,
    has_secret,
    is_model_public,
    expected_result,
    custom_model_trt_llm,
):
    _is_model_public_mock.return_value = is_model_public
    handle = TrussHandle(custom_model_trt_llm)
    if has_secret:
        handle.add_secret("hf_access_token")
    assert is_missing_secrets_for_trt_llm_builder(handle) == expected_result


def test_check_and_update_memory_for_trt_llm_builder(custom_model_trt_llm):
    handle = TrussHandle(custom_model_trt_llm)
    assert memory_updated_for_trt_llm_builder(handle)
    assert handle.spec.memory == f"{TRTLLM_MIN_MEMORY_REQUEST_GI}Gi"
    assert handle.spec.memory_in_bytes == TRTLLM_MIN_MEMORY_REQUEST_GI * 1024**3
