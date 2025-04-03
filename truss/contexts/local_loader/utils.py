import copy
import inspect
from typing import Mapping, Optional

from truss.base.truss_spec import TrussSpec
from truss.local.local_config_handler import LocalConfigHandler


def prepare_secrets(spec: TrussSpec) -> Mapping[str, Optional[str]]:
    secrets = copy.deepcopy(spec.secrets)
    local_secrets = LocalConfigHandler.get_config().secrets
    for secret_name in secrets:
        if secret_name in local_secrets:
            secrets[secret_name] = local_secrets[secret_name]
    return secrets


def signature_accepts_keyword_arg(signature: inspect.Signature, kwarg: str) -> bool:
    return kwarg in signature.parameters or _signature_accepts_kwargs(signature)


def _signature_accepts_kwargs(signature: inspect.Signature) -> bool:
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False
