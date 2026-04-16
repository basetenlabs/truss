from unittest import mock
from unittest.mock import MagicMock

import pytest

from truss.base.truss_config import (
    AdditionalAutoscalingConfig,
    AutoscalingMetric,
    AutoscalingSettings,
)
from truss.remote.baseten.core import (
    ModelVersionHandle,
    create_llm_service,
)
from truss.remote.baseten.service import BasetenService
from truss.truss_handle.truss_handle import TrussHandle


# ============== API Tests ==============


def test_create_llm_model(baseten_api):
    """Test create_llm_model REST API call with all fields."""
    mock_rest_client = MagicMock()
    mock_rest_client.post.return_value = {
        "id": "llm-model-123",
        "name": "my-llm",
        "created_at": "2025-01-01T00:00:00Z",
    }
    baseten_api._rest_api_client = mock_rest_client

    result = baseten_api.create_llm_model(
        name="my-llm",
        resources={"accelerator": "H100"},
        llm_config={
            "checkpoint_name": "unsloth/Llama-3.2-3B-Instruct",
            "tensor_parallel_size": 1,
            "backend": "pytorch",
        },
        llm_version="1.0",
        environment_variables={"HF_TOKEN": "secret"},
        autoscaling_settings={"min_replica": 1, "max_replica": 3},
        additional_autoscaling_config={
            "metrics": [{"name": "in_flight_tokens", "target": 50000}]
        },
        metadata={"git_sha": "abc123"},
    )

    mock_rest_client.post.assert_called_once_with("v1/llm_models", body=mock.ANY)
    body = mock_rest_client.post.call_args[1]["body"]
    assert body["name"] == "my-llm"
    assert body["resources"] == {"accelerator": "H100"}
    assert body["llm_config"]["checkpoint_name"] == "unsloth/Llama-3.2-3B-Instruct"
    assert body["llm_version"] == "1.0"
    assert body["environment_variables"] == {"HF_TOKEN": "secret"}
    assert body["autoscaling_settings"] == {"min_replica": 1, "max_replica": 3}
    assert body["additional_autoscaling_config"]["metrics"][0]["target"] == 50000
    assert body["metadata"] == {"git_sha": "abc123"}
    assert result["id"] == "llm-model-123"


def test_create_llm_model_minimal(baseten_api):
    """Test create_llm_model with only required parameters."""
    mock_rest_client = MagicMock()
    mock_rest_client.post.return_value = {"id": "llm-model-123"}
    baseten_api._rest_api_client = mock_rest_client

    baseten_api.create_llm_model(
        name="my-llm",
        resources={"accelerator": "H100"},
        llm_config={"backend": "pytorch"},
    )

    body = mock_rest_client.post.call_args[1]["body"]
    assert body["name"] == "my-llm"
    assert body["resources"] == {"accelerator": "H100"}
    assert body["llm_config"] == {"backend": "pytorch"}
    assert body["llm_version"] == "1.0"
    assert "environment_variables" not in body
    assert "autoscaling_settings" not in body
    assert "additional_autoscaling_config" not in body
    assert "metadata" not in body


def test_create_llm_model_deployment(baseten_api):
    """Test create_llm_model_deployment REST API call."""
    mock_rest_client = MagicMock()
    mock_rest_client.post.return_value = {
        "deployment_id": "dep-789",
    }
    baseten_api._rest_api_client = mock_rest_client

    result = baseten_api.create_llm_model_deployment(
        model_id="llm-model-123",
        resources={"accelerator": "H100"},
        llm_config={"backend": "pytorch"},
        autoscaling_settings={"min_replica": 2},
        metadata={"run_id": "42"},
    )

    mock_rest_client.post.assert_called_once_with(
        "v1/llm_models/llm-model-123/deployments", body=mock.ANY
    )
    body = mock_rest_client.post.call_args[1]["body"]
    assert body["resources"] == {"accelerator": "H100"}
    assert body["llm_config"] == {"backend": "pytorch"}
    assert body["autoscaling_settings"] == {"min_replica": 2}
    assert body["metadata"] == {"run_id": "42"}
    assert "name" not in body
    assert result["deployment_id"] == "dep-789"


# ============== Core Tests ==============


def test_create_llm_service_new_model():
    """Test create_llm_service for a new LLM model (model_id=None)."""
    api = MagicMock()
    api.create_llm_model.return_value = {
        "id": "new-model-id",
        "name": "my-llm",
        "created_at": "2025-01-01T00:00:00Z",
    }

    handle = create_llm_service(
        api=api,
        model_name="my-llm",
        resources={"accelerator": "H100"},
        llm_config={"backend": "pytorch"},
        model_id=None,
        autoscaling_settings={"min_replica": 1},
        metadata={"key": "val"},
    )

    assert handle.model_id == "new-model-id"
    api.create_llm_model.assert_called_once()
    _, kwargs = api.create_llm_model.call_args
    assert kwargs["name"] == "my-llm"
    assert kwargs["resources"] == {"accelerator": "H100"}
    assert kwargs["llm_config"] == {"backend": "pytorch"}
    assert kwargs["autoscaling_settings"] == {"min_replica": 1}
    assert kwargs["metadata"] == {"key": "val"}


def test_create_llm_service_existing_model():
    """Test create_llm_service for an existing LLM model (model_id set)."""
    api = MagicMock()
    api.create_llm_model_deployment.return_value = {
        "deployment_id": "dep-456",
        "hostname": "https://host.test.co",
    }

    handle = create_llm_service(
        api=api,
        model_name="my-llm",
        resources={"accelerator": "H100"},
        llm_config={"backend": "pytorch"},
        model_id="existing-model-id",
        metadata={"count": 42},
    )

    assert handle.model_id == "existing-model-id"
    assert handle.version_id == "dep-456"
    api.create_llm_model_deployment.assert_called_once()
    _, kwargs = api.create_llm_model_deployment.call_args
    assert kwargs["model_id"] == "existing-model-id"
    assert kwargs["resources"] == {"accelerator": "H100"}
    assert kwargs["metadata"] == {"count": 42}


# ============== Remote push_llm Tests ==============


def test_push_rejects_draft_for_llm_config(
    custom_model_truss_dir_with_pre_and_post, remote
):
    """Test that push() raises when publish=False and llm_config is set."""
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    th._spec._config.llm_config = {"backend": "pytorch"}

    with pytest.raises(ValueError, match="Development deployment is not supported"):
        remote.push(
            th,
            "my-llm-model",
            working_dir=custom_model_truss_dir_with_pre_and_post,
            publish=False,
        )


def test_push_llm_raises_value_error_for_empty_model_name(
    custom_model_truss_dir_with_pre_and_post, remote
):
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    with pytest.raises(ValueError, match="Model name cannot be empty"):
        remote.push_llm(th, "   ")


def test_push_llm_full_flow(custom_model_truss_dir_with_pre_and_post, remote):
    """Test the full push_llm flow: reads config, calls REST API directly."""
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    # Set llm_config on the truss config.
    th._spec._config.llm_config = {
        "checkpoint_name": "unsloth/Llama-3.2-3B-Instruct",
        "backend": "pytorch",
    }

    mock_llm_handle = ModelVersionHandle(
        model_id="llm-model-id",
        version_id="llm-dep-id",
        hostname="https://llm.test.co",
    )

    with (
        mock.patch(
            "truss.remote.baseten.remote.exists_model", return_value=None
        ) as mock_exists,
        mock.patch(
            "truss.remote.baseten.remote.create_llm_service",
            return_value=mock_llm_handle,
        ) as mock_create,
    ):
        result = remote.push_llm(
            th,
            "my-llm-model",
        )

        assert isinstance(result, BasetenService)
        assert result.model_id == "llm-model-id"
        assert result.is_draft is False

        mock_exists.assert_called_once_with(remote._api, "my-llm-model", team_id=None)
        mock_create.assert_called_once()
        _, kwargs = mock_create.call_args
        assert kwargs["model_name"] == "my-llm-model"
        assert kwargs["llm_config"]["checkpoint_name"] == "unsloth/Llama-3.2-3B-Instruct"
        assert kwargs["model_id"] is None


def test_push_llm_existing_model(custom_model_truss_dir_with_pre_and_post, remote):
    """Test push_llm when model already exists creates a deployment."""
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    th._spec._config.llm_config = {"backend": "pytorch"}

    mock_llm_handle = ModelVersionHandle(
        model_id="existing-model-id",
        version_id="new-dep-id",
        hostname="https://llm.test.co",
    )

    with (
        mock.patch(
            "truss.remote.baseten.remote.exists_model",
            return_value="existing-model-id",
        ),
        mock.patch(
            "truss.remote.baseten.remote.create_llm_service",
            return_value=mock_llm_handle,
        ) as mock_create,
    ):
        result = remote.push_llm(th, "my-llm-model")

        assert result.model_id == "existing-model-id"
        _, kwargs = mock_create.call_args
        assert kwargs["model_id"] == "existing-model-id"


def test_push_llm_with_autoscaling(custom_model_truss_dir_with_pre_and_post, remote):
    """Test push_llm passes autoscaling settings through."""
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    th._spec._config.llm_config = {"backend": "pytorch"}

    mock_llm_handle = ModelVersionHandle(
        model_id="m1",
        version_id="d1",
        hostname="https://llm.test.co",
    )

    th._spec._config.autoscaling_settings = {"min_replica": 1, "max_replica": 5}
    th._spec._config.additional_autoscaling_config = {
        "metrics": [{"name": "in_flight_tokens", "target": 50000}]
    }

    with (
        mock.patch(
            "truss.remote.baseten.remote.exists_model", return_value=None
        ),
        mock.patch(
            "truss.remote.baseten.remote.create_llm_service",
            return_value=mock_llm_handle,
        ) as mock_create,
    ):
        remote.push_llm(
            th,
            "my-llm-model",
        )

        _, kwargs = mock_create.call_args
        assert kwargs["autoscaling_settings"] == AutoscalingSettings(
            min_replica=1, max_replica=5
        )
        assert kwargs["additional_autoscaling_config"] == AdditionalAutoscalingConfig(
            metrics=[AutoscalingMetric(name="in_flight_tokens", target=50000)]
        )


def test_push_llm_extracts_resources_from_config(
    custom_model_truss_dir_with_pre_and_post, remote
):
    """Test that push_llm extracts accelerator from truss config resources."""
    th = TrussHandle(custom_model_truss_dir_with_pre_and_post)
    th._spec._config.llm_config = {"backend": "pytorch"}

    mock_llm_handle = ModelVersionHandle(
        model_id="m1",
        version_id="d1",
        hostname="https://llm.test.co",
    )

    with (
        mock.patch(
            "truss.remote.baseten.remote.exists_model", return_value=None
        ),
        mock.patch(
            "truss.remote.baseten.remote.create_llm_service",
            return_value=mock_llm_handle,
        ) as mock_create,
    ):
        remote.push_llm(th, "my-llm-model")

        _, kwargs = mock_create.call_args
        # Default truss config has no accelerator, so resources should be empty dict.
        assert kwargs["resources"] == {}
