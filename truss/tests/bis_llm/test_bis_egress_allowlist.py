import pytest

from truss.base.truss_config import (
    BISLLM,
    EgressRestrictions,
    Runtime,
    TrussConfig,
    Weights,
    WeightsSource,
)
from truss.bis_llm.config_checks import HF_EGRESS_FQDNS, bis_egress_allowlist_warnings


def _bis_config(fqdn_allow_list=None, weights=None, llm_config=None, egress=True):
    return TrussConfig(
        bis_llm=BISLLM(config=llm_config or {"model": "test-llm"}),
        weights=Weights(weights or []),
        runtime=Runtime(
            egress_restrictions=EgressRestrictions(fqdn_allow_list=fqdn_allow_list)
            if egress
            else None
        ),
    )


_ENCODER_LLM_CONFIG = {
    "model": "test-llm",
    "b10_vision_config": {"encoder_url": "https://model-abc.api.baseten.co/v1"},
}


@pytest.mark.parametrize(
    ("config", "expected_hosts"),
    [
        # No BDN mount: weights come from Hugging Face at runtime.
        (_bis_config(), ["huggingface.co", "us.aws.cdn.hf.co"]),
        # A wildcard covers the CDN hosts; only the apex is still missing.
        (_bis_config(fqdn_allow_list=["*.hf.co"]), ["huggingface.co"]),
        # The exact encoder host, not a wildcard over every model on the Baseten API.
        (_bis_config(llm_config=_ENCODER_LLM_CONFIG), ["model-abc.api.baseten.co"]),
        # Any Baseten-API URL counts, e.g. an openai_http tokenizer endpoint.
        (
            _bis_config(
                llm_config={
                    "engine_backend": "openai_http",
                    "tokenizer_endpoint": "https://model-wxpev45q.api.baseten.co/environments/production/sync",
                }
            ),
            ["model-wxpev45q.api.baseten.co"],
        ),
    ],
)
def test_bis_egress_allowlist_warnings_name_missing_hosts(config, expected_hosts):
    text = "\n".join(bis_egress_allowlist_warnings(config))

    for host in expected_hosts:
        assert f'      - "{host}"' in text


@pytest.mark.parametrize(
    "config",
    [
        _bis_config(
            weights=[
                WeightsSource(source="hf://model-1", mount_location="/models/base")
            ]
        ),
        _bis_config(
            fqdn_allow_list=[*HF_EGRESS_FQDNS, "*.api.baseten.co"],
            llm_config=_ENCODER_LLM_CONFIG,
        ),
        _bis_config(
            fqdn_allow_list=[*HF_EGRESS_FQDNS, "model-abc.api.baseten.co"],
            llm_config=_ENCODER_LLM_CONFIG,
        ),
        # Malformed URLs, and hosts that only start like the Baseten API.
        _bis_config(
            weights=[
                WeightsSource(source="hf://model-1", mount_location="/models/base")
            ],
            llm_config={
                "model": "test-llm",
                "endpoint": "https://[HOST]/v1",
                "other": "https://model-abc.api.baseten.co.example.com/v1",
            },
        ),
        # Unrestricted, or not BIS: nothing to warn about.
        _bis_config(egress=False),
        TrussConfig(runtime=Runtime(egress_restrictions=EgressRestrictions())),
    ],
)
def test_bis_egress_allowlist_warnings_are_empty_when_covered(config):
    assert bis_egress_allowlist_warnings(config) == []
