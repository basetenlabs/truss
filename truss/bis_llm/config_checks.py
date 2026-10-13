import fnmatch
import json
import re
from typing import Any

# Hosts a Hugging Face weight pull touches when the model has no BDN `weights:`
# mount. Hugging Face lists them for firewalled downloads and warns they change
# as its CDN evolves:
# https://huggingface.co/docs/hub/main/en/models-downloading#downloading-behind-a-proxy-or-firewall
HF_EGRESS_FQDNS = (
    "huggingface.co",
    "cas-server.xethub.hf.co",
    "cas-server.xethub-eu.hf.co",
    "transfer.xethub.hf.co",
    "transfer.xethub-eu.hf.co",
    "us.aws.cdn.hf.co",
    "us.gcp.cdn.hf.co",
    "cdn-lfs-us-1.hf.co",
    "cdn-lfs-eu-1.hf.co",
)


# The host must end the URL, so model-x.api.baseten.co.example.com is not matched;
# `"` ends it inside the JSON-encoded config.
_BASETEN_API_HOST = re.compile(
    r'https?://([a-z0-9.-]+\.api\.baseten\.co)(?=[:/?#"])', re.IGNORECASE
)


def _baseten_api_hosts(llm_config: dict) -> set[str]:
    """Hosts of every Baseten-API URL in bis_llm.config, e.g. encoder_url or tokenizer_endpoint."""
    return {
        h.lower()
        for h in _BASETEN_API_HOST.findall(json.dumps(llm_config, default=str))
    }


def bis_egress_allowlist_warnings(config: Any) -> list[str]:
    """One warning per kind of host a restricted BIS model needs but its allowlist lacks."""
    egress_restrictions = config.runtime.egress_restrictions
    if config.bis_llm is None or egress_restrictions is None:
        return []
    allowed = egress_restrictions.fqdn_allow_list or []

    def is_allowed(host: str) -> bool:
        return any(fnmatch.fnmatchcase(host, entry) for entry in allowed)

    gaps: list[tuple[str, list[str]]] = []
    if not config.weights.sources:
        hf_hosts = [h for h in HF_EGRESS_FQDNS if not is_allowed(h)]
        reason = "it has no BDN `weights:` mount, so it pulls weights from Hugging Face"
        gaps.append((reason, hf_hosts))
    # Suggest exact hosts: *.api.baseten.co would reach every model there.
    api_hosts = _baseten_api_hosts(config.bis_llm.config or {})
    gaps.append(
        (
            "its bis_llm.config has URLs on the Baseten API",
            sorted(h for h in api_hosts if not is_allowed(h)),
        )
    )
    # Quoted: a leading `*` would start a YAML alias.
    return [
        f"runtime.egress_restrictions is set, but {reason}. These hosts are "
        "not in the allowlist, so the deployment cannot reach them. Add:\n"
        "runtime:\n  egress_restrictions:\n    fqdn_allow_list:\n"
        + "\n".join(f'      - "{h}"' for h in hosts)
        for reason, hosts in gaps
        if hosts
    ]
