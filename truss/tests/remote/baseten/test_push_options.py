import pytest

from truss.base.constants import PRODUCTION_ENVIRONMENT_NAME
from truss.base.truss_config import ModelServer
from truss.remote.baseten.custom_types import ModelOrigin, PushOptions


def test_push_options_default_publish_is_false():
    # Matches the public SDK (truss.api.push) default. Internal callers
    # must opt-in to publishing.
    options = PushOptions()
    assert options.publish is False


def test_deploy_timeout_minutes_validator_rejects_below_range():
    with pytest.raises(
        ValueError,
        match=r"deploy-timeout-minutes must be between 10 minutes and 1440 minutes \(24 hours\)",
    ):
        PushOptions(deploy_timeout_minutes=9)


def test_deploy_timeout_minutes_validator_rejects_above_range():
    with pytest.raises(
        ValueError,
        match=r"deploy-timeout-minutes must be between 10 minutes and 1440 minutes \(24 hours\)",
    ):
        PushOptions(deploy_timeout_minutes=1441)


@pytest.mark.parametrize("value", [10, 500, 1440, None])
def test_deploy_timeout_minutes_validator_accepts_valid(value):
    options = PushOptions(deploy_timeout_minutes=value)
    assert options.deploy_timeout_minutes == value


def test_deployment_name_validator_rejects_invalid_chars():
    with pytest.raises(
        ValueError,
        match="Deployment name must only contain alphanumeric, -, _ and . characters",
    ):
        PushOptions(deployment_name="has space")


@pytest.mark.parametrize("value", ["abc_123", "dep.name-v1", "", None])
def test_deployment_name_validator_accepts_valid(value):
    options = PushOptions(deployment_name=value)
    assert options.deployment_name == value


def test_preserve_requires_promote():
    with pytest.raises(
        ValueError,
        match="preserve-previous-production-deployment can only be used with the '--promote' option",
    ):
        PushOptions(preserve_previous_prod_deployment=True, promote=False)


def test_preserve_with_promote_is_valid():
    options = PushOptions(preserve_previous_prod_deployment=True, promote=True)
    assert options.preserve_previous_prod_deployment is True


def test_normalize_non_truss_server_forces_publish():
    options = PushOptions(publish=False)
    normalized = options.normalize(ModelServer.TRT_LLM)
    assert normalized.publish is True
    # Original is unchanged (frozen).
    assert options.publish is False


def test_normalize_promote_sets_environment_to_production():
    options = PushOptions(publish=True, promote=True)
    normalized = options.normalize(ModelServer.TrussServer)
    assert normalized.environment == PRODUCTION_ENVIRONMENT_NAME


def test_normalize_environment_forces_publish():
    options = PushOptions(publish=False, environment="staging")
    normalized = options.normalize(ModelServer.TrussServer)
    assert normalized.publish is True
    assert normalized.environment == "staging"


def test_normalize_deployment_name_without_publish_raises():
    options = PushOptions(publish=False, deployment_name="dep_name")
    with pytest.raises(
        ValueError, match="Deployment name cannot be used for development deployment"
    ):
        options.normalize(ModelServer.TrussServer)


def test_normalize_noop_when_already_published():
    options = PushOptions(
        publish=True, environment="staging", origin=ModelOrigin.BASETEN
    )
    normalized = options.normalize(ModelServer.TrussServer)
    assert normalized.publish is True
    assert normalized.environment == "staging"
    assert normalized.origin == ModelOrigin.BASETEN


def test_push_options_is_frozen():
    options = PushOptions()
    with pytest.raises(ValueError):
        options.publish = True  # type: ignore[misc]
