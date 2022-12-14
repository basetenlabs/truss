import copy
import inspect
from typing import Dict

from truss.local.local_config_handler import LocalConfigHandler
from truss.truss_spec import TrussSpec


def prepare_secrets(spec: TrussSpec) -> Dict[str, str]:
    secrets = copy.deepcopy(spec.secrets)
    local_secerts = LocalConfigHandler.get_config().secrets
    for secret_name in secrets:
        if secret_name in local_secerts:
            secrets[secret_name] = local_secerts[secret_name]
    return secrets


def signature_accepts_keyword_arg(signature: inspect.Signature, kwarg: str) -> bool:
    return kwarg in signature.parameters or _signature_accepts_kwargs(signature)


def _signature_accepts_kwargs(signature: inspect.Signature) -> bool:
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False
